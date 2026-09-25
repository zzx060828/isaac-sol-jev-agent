import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.control import blocked, candidates, path_step, pickup_target
from isaac_agent.demo import frame_message
from isaac_agent.state import World
from isaac_agent.runtime import Controller
from test_agent import MemoryLog
from test_safety_strategy import refresh


def snapshot():
    msg = frame_message(player=(300, 280))
    msg['payload']['PLAYER_INVENTORY'][0]['coins'] = 0
    msg['payload'].update(BOMBS=[], FIRE_HAZARDS=[], INTERACTABLES=[])
    msg['payload']['PICKUPS'] = [
        dict(id=20, variant=20, sub_type=1, price=0, pos=dict(x=320, y=280)),
        dict(id=21, variant=20, sub_type=1, price=0, pos=dict(x=240, y=280))]
    return msg


def prepared(kind='fire'):
    msg = snapshot()
    if kind == 'fire':
        msg['payload']['FIRE_HAZARDS'] = [dict(id=1, type='FIREPLACE', variant=0,
            pos=dict(x=340,y=280), collision_radius=20)]
    elif kind in ('spikes','rock','pit'):
        typ, collision = {'spikes':(8,0),'rock':(2,3),'pit':(7,1)}[kind]
        msg['payload']['ROOM_LAYOUT']['grid'] = {
            '1':dict(x=340,y=280,type=typ,collision=collision)}
    else:
        msg['payload']['INTERACTABLES'] = [dict(id=1, variant=2, pos=dict(x=340,y=280))]
    w = World()
    w.ingest(msg)
    return w


def overlapping(kind):
    msg=snapshot()
    msg['payload']['PLAYER_POSITION'][0]['pos']['x']=280
    if kind in ('paid_pickup','option','trinket'):
        variant,subtype,price,group={'paid_pickup':(100,1,15,0),'option':(20,2,0,1),
                                     'trinket':(350,1,0,0)}[kind]
        msg['payload']['PICKUPS'].append(dict(id=30,variant=variant,sub_type=subtype,price=price,
                                              options_index=group,pos=dict(x=320,y=280)))
    elif kind=='machine':
        msg['payload']['INTERACTABLES']=[dict(id=30,variant=2,pos=dict(x=320,y=280))]
    elif kind in ('trapdoor','crawlspace'):
        msg['payload']['ROOM_LAYOUT']['grid']={'1':dict(type=17 if kind=='trapdoor' else 18,
            variant=0,state=1,x=320,y=280,collision=0)}
    w=World();w.ingest(msg)
    return w


class PickupApproachTests(unittest.TestCase):
    def test_free_resource_cannot_exempt_a_coincident_unselected_contact(self):
        for kind in ('paid_pickup','option','trinket','machine','trapdoor','crawlspace'):
            with self.subTest(kind=kind):
                w=overlapping(kind)
                self.assertEqual(pickup_target(w,{},True),(240,280))
                target=pickup_target(w,{},True)
                action=candidates(w,dict(strategy_enabled=True,navigation_target=target,
                                         navigation_contact=False))[0][0]['action']
                self.assertLess(action.move[0],0)
                self.assertFalse(w.pickup_failures)

    def test_resource_becomes_available_when_overlap_disappears(self):
        w=overlapping('paid_pickup')
        w.payload['PICKUPS']=[w.pickups[0],w.pickups[-1]]
        self.assertIsNone(pickup_target(w,{},True))
        w.payload['PICKUPS']=w.pickups[:1]
        refresh(w,1)
        self.assertEqual(pickup_target(w,{},True),(320,280))

    def test_free_resources_may_overlap_each_other_and_chest_shell(self):
        w=overlapping('paid_pickup')
        for entity in (dict(id=30,variant=30,sub_type=1,price=0,pos=dict(x=320,y=280)),
                       dict(id=30,variant=50,sub_type=0,price=0,pos=dict(x=320,y=280))):
            w.payload['PICKUPS'][-1]=entity
            self.assertEqual(pickup_target(w,{},True),(320,280))
        # Explicit contact handling outside the ordinary selector is unchanged.
        w=overlapping('machine')
        self.assertFalse(blocked(w,(320,280),exit_point=(320,280)))

    def test_nearby_hazard_or_payment_selects_another_resource(self):
        for kind in ('fire','spikes','rock','pit','machine'):
            with self.subTest(kind=kind):
                w = prepared(kind)
                self.assertFalse(blocked(w,(300,280)))
                self.assertTrue(blocked(w,(320,280)))
                # The generic near-goal shortcut remains; pickup selection
                # must not mistake this return value for a checked approach.
                self.assertEqual(path_step(w,(320,280)),(320,280))
                target = pickup_target(w,{},basic_only=True)
                self.assertEqual(target,(240,280))
                action = candidates(w,dict(strategy_enabled=True,navigation_target=target))[0][0]['action']
                self.assertLess(action.move[0],0)
                self.assertEqual(action.shoot,(0,0))

    def test_only_hazardous_resource_is_deferred_without_marking_failed(self):
        w = prepared()
        w.payload['PICKUPS'] = w.pickups[:1]
        self.assertIsNone(pickup_target(w,{},True))
        self.assertFalse(w.pickup_failures)
        w.payload['FIRE_HAZARDS'][0]['is_extinguished'] = True
        refresh(w,1)
        self.assertEqual(pickup_target(w,{},True),(320,280))

    def test_flight_reopens_ground_hazards_but_not_fire_or_rock(self):
        for kind in ('spikes','pit','fire','rock'):
            w = prepared(kind)
            w.stats['can_fly'] = True
            target = pickup_target(w,{},True)
            self.assertEqual(target,(320,280) if kind in ('spikes','pit') else (240,280))

    def test_player_radius_and_options_still_apply(self):
        w = prepared()
        w.payload['FIRE_HAZARDS'][0]['pos']['x'] = 353
        self.assertEqual(pickup_target(w,{},True),(320,280))
        w.stats['size'] = 16
        self.assertEqual(pickup_target(w,{},True),(240,280))
        w.payload['FIRE_HAZARDS'] = []
        w.pickups[0]['options_index'] = 1
        self.assertEqual(pickup_target(w,{},True),(240,280))


class PickupApproachControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_floor_exit_overlap_collects_safe_coin_without_model(self):
        w=overlapping('trapdoor')
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),decide=AsyncMock())
        c=Controller(models,MemoryLog())
        try:
            command=c.step(w)
            self.assertEqual(c.route,'collect_resources')
            self.assertEqual(c.execution_plan['navigation_target'],(240,280))
            self.assertFalse(c.execution_plan['navigation_contact'])
            self.assertLess(command['move']['x'],0)
            await asyncio.sleep(0)
            models.plan.assert_not_called();models.choose_goal.assert_not_called();models.decide.assert_not_called()
        finally:await c.close()

    async def test_collect_safe_coin_then_newly_accessible_coin_without_model(self):
        w = prepared('spikes')
        models = SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),decide=AsyncMock())
        c = Controller(models,MemoryLog())
        try:
            command = c.step(w)
            self.assertEqual(c.route,'collect_resources')
            self.assertEqual(c.execution_plan['navigation_target'],(240,280))
            self.assertLess(command['move']['x'],0)
            # Inject observed pickup/terrain changes; this is not a physics run.
            w.payload['PICKUPS'] = w.pickups[:1]
            w.inventory['coins'] = 1
            w.layout['grid']['1'].update(type=0,collision=0)
            refresh(w,1)
            command = c.step(w)
            await asyncio.sleep(0)
            self.assertEqual(c.execution_plan['navigation_target'],(320,280))
            self.assertGreater(command['move']['x'],0)
            models.plan.assert_not_called()
            models.choose_goal.assert_not_called()
            models.decide.assert_not_called()
        finally:
            await c.close()
