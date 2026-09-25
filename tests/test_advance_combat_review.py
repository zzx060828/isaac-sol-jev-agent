import asyncio
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from test_agent import MemoryLog
from test_advance import setup, preparation
from isaac_agent.advance import AdvancePlanner
from isaac_agent.combat import incoming_jump, select_enemy
from isaac_agent.control import candidates, blocked, physical_collision, combat_position, jump_evasion_waypoint
from isaac_agent.physics import trajectory
from isaac_agent.state import World, xy


def recorded(frame):
    data=json.loads((Path(__file__).parent/'fixtures/advance-combat-20260924.json').read_text())
    state=next(s for s in data['states'] if s['frame']==frame)
    w=World(frame=frame,room=state['room'],payload=deepcopy(state['payload']),
            sampled=state['sampled'],capabilities=state['capabilities'],updated=time.monotonic())
    w.combat_breach=deepcopy(state['breach'])
    return w,state['plan']


class AdvanceOpportunityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.w=setup();self.w.layout['doors']['2']['target_room_type']=1
        self.w.rooms={str(self.w.room):dict(type=1,shape=1,doors=deepcopy(self.w.layout['doors'])),
                      '10':dict(type=1,shape=1),'11':dict(type=4,shape=1)}
        self.models=SimpleNamespace(mode='hybrid',prepare=AsyncMock(return_value=(preparation(),{})))
        self.p=AdvancePlanner(self.models,MemoryLog())

    async def asyncTearDown(self): await self.p.close()

    async def test_unlocated_boss_preparation_is_once_after_exploration_during_combat(self):
        now=time.monotonic()
        self.p.maybe_start(self.w,'tactical',True,now)
        self.assertIsNone(self.p.task)
        self.p.maybe_start(self.w,'tactical',False,now)
        await self.p.task
        obs=self.models.prepare.call_args.args[0]
        self.assertEqual(obs['trigger'],'explored_build_with_unlocated_boss')
        self.assertIsNone(obs['preparation_window']['boss_room'])
        self.assertIsNone(obs['preparation_window']['observed_hops'])
        for delta in (50,190,250):self.p.maybe_start(self.w,'tactical',False,now+delta)
        self.assertEqual(self.models.prepare.await_count,1)
        # Discovery after expiry is real new information; None hops is not a number.
        self.w.rooms[str(self.w.room)]['doors']['2']['target_room_type']=5
        self.p.maybe_start(self.w,'journey',False,now+300)
        await self.p.task
        self.assertEqual(self.models.prepare.await_count,2)
        self.assertEqual(self.models.prepare.call_args.args[0]['trigger'],'observed_boss_approach')

    async def test_no_speculative_request_in_start_room_clear_room_or_stale_build(self):
        for change in ('early','clear','stale'):
            w=deepcopy(self.w)
            if change=='early':w.rooms.pop('10');w.rooms.pop('11')
            if change=='clear':w.payload['ENEMIES']=[];w.info['is_clear']=True
            if change=='stale':w.sampled['PLAYER_STATS']=w.frame-7
            self.p.next_check_at=0
            self.p.maybe_start(w,'routine',False)
            self.assertIsNone(self.p.task,change)

    async def test_committed_traversal_does_not_disable_known_boss_preparation(self):
        w=setup()
        self.p.maybe_start(w,'journey',False)
        await self.p.task
        self.assertEqual(self.models.prepare.await_count,1)
        self.assertEqual(self.models.prepare.call_args.args[0]['preparation_window']['observed_hops'],1)


class CombatReviewTests(unittest.TestCase):
    def test_close_pursuer_suspends_stored_tnt_breach_without_erasing_it(self):
        w,plan=recorded(1230);intent=deepcopy(w.combat_breach)
        option=candidates(w,plan)[0][0]
        self.assertNotIn('shoot TNT',option['description'])
        self.assertEqual(w.combat_breach,intent)
        for point in trajectory(w,option['action'].move,3):
            self.assertFalse(blocked(w,point,w.stats['size']))
        self.assertFalse(any('shoot TNT' in o['description'] for o in candidates(w,plan)[0]))

    def test_next_path_step_does_not_replace_final_firing_position(self):
        w,plan=recorded(1260);target=xy(select_enemy(w,plan)['pos'])
        # Force a legal path's intermediate step to lie much closer to target.
        with patch('isaac_agent.control.path_step',return_value=(target[0]+60,target[1])):
            route=combat_position(w,target,desired=210,flexible=True)
        self.assertIsNotNone(route)
        self.assertNotEqual(route[0],route[1])
        self.assertAlmostEqual(math.dist(route[0],target),60)

    def test_airborne_boss_is_evaded_while_a_different_add_remains_shooting_target(self):
        w,plan=recorded(5294)
        jumper=incoming_jump(w)
        self.assertIsNotNone(jumper)
        self.assertNotEqual(select_enemy(w,plan)['id'],jumper['id'])
        option=candidates(w,plan)[0][0]
        self.assertEqual(option['action'].move,(-1,0))
        self.assertEqual(option['action'].shoot,(0,-1))
        self.assertIn('evade observed jump',option['description'])
        for point in trajectory(w,option['action'].move,3):
            self.assertFalse(physical_collision(w,point,w.stats['size']))
            self.assertFalse(blocked(w,point,w.stats['size']))
        self.assertEqual(w.health['damage_cooldown'],0)

    def test_jump_detector_requires_observed_approach_and_airborne_class(self):
        for change in ('grounded','away','stopped','other_enemy','off_axis'):
            w,_=recorded(5294);e=incoming_jump(w)
            if change=='grounded':e['collision_class']=2
            if change=='away':e['vel']={k:-v for k,v in e['vel'].items()}
            if change=='stopped':e['vel']={'x':0,'y':0}
            if change=='other_enemy':e['type']=10
            if change=='off_axis':e['pos']['x']-=250
            self.assertIsNone(incoming_jump(w),change)

    def test_jump_flank_does_not_create_permission_to_cross_hazards_or_payments(self):
        w,_=recorded(5294);enemy=incoming_jump(w)
        w.payload['FIRE_HAZARDS']=[dict(pos=dict(x=x,y=402),collision_radius=60)
                                   for x in (417,457,617,657)]
        self.assertIsNone(jump_evasion_waypoint(w,enemy))
        w,_=recorded(5294);enemy=incoming_jump(w)
        w.payload['PICKUPS']=[dict(id=i,variant=100,sub_type=1,price=15,pos=dict(x=x,y=402))
                             for i,x in enumerate((417,457,617,657))]
        self.assertIsNone(jump_evasion_waypoint(w,enemy))
