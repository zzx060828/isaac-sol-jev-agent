import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.chests import ChestObserver
from isaac_agent.control import blocked, pickup_target
from isaac_agent.fast_policy import lane
from isaac_agent.journey import options
from isaac_agent.runtime import Controller
from isaac_agent.strategy import offers
from isaac_agent.state import World
from test_fast_policy import world, Log
from test_safety_strategy import refresh


def setup(**change):
    w = world()
    w.info['room_type'] = 7
    w.payload.update(BOMBS=[], FIRE_HAZARDS=[], INTERACTABLES=[])
    w.payload['PICKUPS'] = [dict(id=20, variant=50, sub_type=1, price=0,
                              options_index=0, pos=dict(x=400, y=280), **change)]
    refresh(w)
    return w


class ChestTests(unittest.IsolatedAsyncioTestCase):
    async def test_sol_sees_chest_resources_but_cannot_request_local_opening(self):
        w=setup()
        w.payload['PICKUPS'].append(dict(id=7,variant=100,sub_type=378,price=0,
                                       pos=dict(x=480,y=280)))
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(),decide=AsyncMock())
        c=Controller(m,Log())
        try:
            c.step(w)
            request=next(r['observation'] for r in c.log.rows if r['event']=='plan_request')
            self.assertTrue(any(p['variant']==50 for p in request['pickups']))
            self.assertFalse(any(o['entity'].get('variant')==50 for o in request['strategy_offers']))
            self.assertTrue(c.execution_plan['parallel_resource_collection'])
            signature=w.strategy_signature()
            w.pickups[0]['sub_type']=0
            self.assertEqual(w.strategy_signature(),signature)
            # Newly generated consequential loot still invalidates the request.
            w.payload['PICKUPS'].append(dict(id=9,variant=100,sub_type=531,price=0,pos=dict(x=410,y=300)))
            self.assertNotEqual(w.strategy_signature(),signature)
        finally:
            await c.close()

    async def test_open_collect_and_exit_wait_for_fresh_hazards_without_model(self):
        w = setup()
        m = SimpleNamespace(mode='hybrid', plan=AsyncMock(), decide=AsyncMock())
        c = Controller(m, Log())
        try:
            self.assertEqual(lane(w, w.frame)[0], 'collect_resources')
            self.assertEqual(pickup_target(w, {}, True), (400, 280))
            c.step(w)
            self.assertGreater(c.execution_plan['navigation_target'][0], 320)
            w.pickups[0]['sub_type'] = 0
            w.payload['PICKUPS'].append(dict(id=21, variant=20, sub_type=1, price=0, pos=dict(x=420,y=280)))
            refresh(w, 1); c.step(w)
            self.assertEqual(c.route, 'settle_chest')
            opened = w.frame
            refresh(w, 6); w.sampled['BOMBS'] = opened; c.step(w)
            self.assertEqual(c.route, 'settle_chest')
            refresh(w, 1); c.step(w)
            self.assertEqual(c.route, 'collect_resources')
            self.assertEqual(c.execution_plan['navigation_target'], (420, 280))
            self.assertFalse(blocked(w, (400,280))) # opened shell must not trap player
            m.plan.assert_not_called(); m.decide.assert_not_called()
        finally:
            await c.close()

    def test_special_chests_and_challenge_room_do_not_use_free_touch(self):
        for field, value in (('variant',51),('variant',52),('variant',53),('variant',54),
                             ('variant',55),('variant',360),('price',5),('options_index',1),
                             ('sub_type',0)):
            w = setup(); w.pickups[0][field] = value
            self.assertIsNone(pickup_target(w, {}, True))
            self.assertFalse(any(o['kind']=='pickup' for o in offers(w)))
            if field != 'sub_type':
                self.assertTrue(blocked(w, (400,280)))
        w = setup(); w.info['room_type'] = 11
        self.assertIsNone(pickup_target(w, {}, True))
        self.assertTrue(blocked(w, (400,280)))
        self.assertFalse(any(o['kind']=='pickup' for o in offers(w)))

    def test_troll_drop_and_stale_sensors_stop_collection_before_bomb_channel(self):
        w = setup(); w.sampled['BOMBS'] = w.frame-4
        self.assertIsNone(pickup_target(w, {}, True))
        refresh(w)
        w.payload['PICKUPS'].append(dict(id=21, variant=40, sub_type=3, pos=dict(x=420,y=280)))
        self.assertIsNone(pickup_target(w, {}, True))
        self.assertTrue(blocked(w, (420,280)))

    def test_loot_spawning_under_player_can_be_left_but_not_approached(self):
        from isaac_agent.control import candidates
        w=setup(); w.player['pos']=dict(x=400,y=280)
        w.payload['PICKUPS']=[dict(id=21,variant=40,sub_type=3,pos=dict(x=405,y=280))]
        self.assertTrue(blocked(w,(402,280)))
        self.assertFalse(blocked(w,(392,280)))
        action=candidates(w,{'strategy_enabled':True,'navigation_target':(400,280)})[0][0]['action']
        self.assertLess(action.move[0],0)

    def test_only_closed_supported_chest_makes_remote_resource_destination(self):
        w = setup(); w.payload['PICKUPS'] = []
        target = w.layout['doors']['2']['target_room']
        w.rooms[str(target)] = dict(type=7, clear=True, resources=[dict(variant=50,sub_type=1,price=0)])
        self.assertTrue(any(o['destination_room']==target for o in options(w)))
        w.rooms[str(target)]['resources'][0]['sub_type'] = 0
        self.assertFalse(any(o['destination_room']==target for o in options(w)))

    def test_observer_does_not_repeat_opening_or_confuse_room_reset(self):
        w = setup(); observer = ChestObserver(); log = Log()
        self.assertFalse(observer.update(w,log))
        w.payload['PICKUPS'] = []; refresh(w,1)
        self.assertTrue(observer.update(w,log))
        refresh(w,6); self.assertFalse(observer.update(w,log))
        refresh(w,1); self.assertFalse(observer.update(w,log))
        self.assertEqual([r['event'] for r in log.rows], ['chest_changed','chest_settled'])
        self.assertFalse(ChestObserver().update(w,log))

    def test_recorded_secret_chest_has_a_reachable_approach_without_shooting_tnt(self):
        from isaac_agent.control import candidates
        data=json.loads((Path(__file__).parent/'fixtures/secret-ordinary-chest.json').read_text())
        w=World(); w.ingest(data['snapshot'])
        self.assertEqual(pickup_target(w,{},True),(320,320))
        self.assertEqual(lane(w,w.frame)[0],'collect_resources')
        action=candidates(w,{'strategy_enabled':True,'navigation_target':(320,320)})[0][0]['action']
        self.assertEqual(action.shoot,(0,0))
