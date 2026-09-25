import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from isaac_agent.control import Action, blocked, pickup_target
from isaac_agent.fast_policy import fast_offers
from isaac_agent.hazards import explosives
from isaac_agent.pickups import routine, useful
from isaac_agent.runtime import Controller
from isaac_agent.strategy import offers
from test_agent import MemoryLog
from test_fast_policy import snapshot, world
from test_safety_strategy import refresh


def resource(variant=20, subtype=1, **extra):
    return dict(id=20,variant=variant,sub_type=subtype,price=0,options_index=0,
                pos={'x':240,'y':280},**extra)


def setup():
    msg=snapshot();msg['payload']['ROOM_INFO']['room_type']=4
    msg['payload']['PICKUPS'][0].update(sub_type=531,pos={'x':440,'y':280})
    msg['payload']['PICKUPS'].append(resource())
    w=world(msg);w.payload.update(BOMBS=[],FIRE_HAZARDS=[],INTERACTABLES=[])
    refresh(w)
    return w


class ParallelResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_free_coin_moves_while_sol_pending_without_discarding_item_plan(self):
        w=setup();gate=asyncio.Event()
        async def plan(obs):
            await gate.wait()
            return {'objective':'Take item','mode':'collect','strategy_choice':'pickup:7:100:531:0'},{}
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(side_effect=plan),decide=AsyncMock())
        c=Controller(m,MemoryLog())
        try:
            command=c.step(w);await asyncio.sleep(0)
            self.assertEqual(c.route,'sol')
            self.assertTrue(c.execution_plan['parallel_resource_collection'])
            self.assertLess(command['move']['x'],0)
            self.assertEqual(m.plan.await_count,1)
            # Collecting one independent coin must not invalidate the item choice.
            w.payload['PICKUPS'].pop();w.inventory['coins']=1;refresh(w,1)
            gate.set();await c.plan_task
            self.assertEqual(c.plan['strategy_choice'],'pickup:7:100:531:0')
            self.assertFalse(any(r['event']=='plan_discarded' for r in c.log.rows))
            c.step(w)
            self.assertEqual(c.execution_plan['navigation_target'],(440,280))
            m.decide.assert_not_called()
        finally:await c.close()

    async def test_parallel_collection_does_not_touch_options_or_enter_live_bomb(self):
        for change in ('options','bomb','stale'):
            w=setup()
            if change=='options':w.pickups[-1]['options_index']=2
            if change=='bomb':w.payload['BOMBS']=[{'pos':{'x':240,'y':280}}]
            if change=='stale':w.sampled['BOMBS']=w.frame-4
            m=SimpleNamespace(mode='hybrid',plan=AsyncMock(),decide=AsyncMock())
            c=Controller(m,MemoryLog())
            try:
                c.step(w)
                self.assertFalse(c.execution_plan.get('parallel_resource_collection'))
                if change=='options':self.assertTrue(blocked(w,(240,280)))
            finally:await c.close()

    def test_subtypes_are_consistent_across_routes_targets_and_hazards(self):
        for variant,subtype in ((40,3),(40,5),(40,6),(20,6),(90,4),(10,12),(30,4)):
            w=setup();w.payload['PICKUPS']=[resource(variant,subtype)]
            self.assertFalse(routine(w.pickups[0]))
            self.assertIsNone(pickup_target(w,{},basic_only=True))
            self.assertTrue(blocked(w,(240,280)))
            if variant==40:
                self.assertFalse(offers(w))
                self.assertTrue(any(x['armed'] for x in explosives(w)))
        w=setup();w.payload['PICKUPS']=[resource(40,1)]
        self.assertEqual(pickup_target(w,{},basic_only=True),(240,280))
        self.assertFalse(explosives(w))

    def test_stock_cap_and_actual_heart_capacity_prevent_pointless_collection(self):
        w=setup();w.inventory.update(coins=99,keys=99,bombs=99)
        for variant in (20,30,40):self.assertFalse(useful(w,resource(variant)))
        w.inventory['collectibles']={'416':1}
        self.assertTrue(useful(w,resource(20)))
        w.health.update(max_hearts=20,red_hearts=2,soul_hearts=4)
        self.assertFalse(useful(w,resource(10,3)))
        self.assertTrue(useful(w,resource(10,1)))

    def test_paid_or_mutually_exclusive_changes_still_invalidate_signature(self):
        w=setup();before=w.strategy_signature()
        w.pickups[-1]['price']=5
        self.assertNotEqual(w.strategy_signature(),before)
        w.pickups[-1]['price']=0;w.pickups[-1]['options_index']=1
        self.assertNotEqual(w.strategy_signature(),before)

    def test_reviewed_resource_passives_use_jev_but_options_still_use_sol(self):
        for item in (183,198,199,456):
            w=world();w.pickups[0]['sub_type']=item
            self.assertEqual(fast_offers(w)[0]['entity']['sub_type'],item)
            w.pickups[0]['options_index']=1
            self.assertFalse(fast_offers(w))

    def test_failed_route_reused_across_poop_damage_but_rechecked_after_opening(self):
        from isaac_agent.state import World,xy
        from isaac_agent.control import path_step
        w=World();w.ingest(json.loads((Path(__file__).parent/'fixtures/enclosed-key.json').read_text()))
        target=xy(w.pickups[0]['pos'])
        self.assertEqual(path_step(w,target),xy(w.player['pos']))
        self.assertTrue(w.unreachable_paths)
        component=next(iter(w.unreachable_paths.values()))
        node=next(p for p in component if p[0]*20<200)
        w.player['pos']=dict(x=node[0]*20,y=node[1]*20)
        for cell in w.layout['grid'].values():
            if cell.get('type')==14:cell['state']=1
        w.navigation_revision+=1;w.path_cache.clear()
        with patch('isaac_agent.control.heapq.heappop',side_effect=AssertionError('repeated exhausted search')):
            self.assertEqual(path_step(w,target),xy(w.player['pos']))
        w.layout['grid']={};w.navigation_revision+=1;w.path_cache.clear()
        self.assertEqual(path_step(w,target),target)

    def test_recorded_glass_eye_boss_reward_can_use_fast_lane(self):
        from isaac_agent.state import World
        from isaac_agent.fast_policy import lane
        w=World();w.ingest(json.loads((Path(__file__).parent/'fixtures/boss-loot-glass-eye.json').read_text())['snapshot'])
        route,choices=lane(w,w.frame)
        self.assertEqual(route,'fast')
        self.assertEqual(choices[0]['entity']['sub_type'],730)
