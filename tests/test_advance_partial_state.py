import asyncio
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.advance import AdvancePlanner, build_key
from isaac_agent.runtime import Controller
from isaac_agent.state import World
from test_advance import setup, preparation
from test_agent import MemoryLog
from test_safety_strategy import refresh


def fixture():
    return json.loads((Path(__file__).parent/'fixtures/advance-partial-room-entry.json').read_text())


def cached(data):
    return dict(key=data['source_build_key'],purpose='boss_preparation',room=data['source_room'],
                frame=data['source_frame'],resources=data['source_resources'],plan=data['preparation'],
                completed_at=time.monotonic()-data['cache_age_at_entry_seconds'])


class AdvancePartialStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_boss_entry_keeps_reply_and_applies_after_three_updates(self):
        data=fixture();w=World();w.ingest(data['partial_snapshot'])
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),prepare=AsyncMock(),decide=AsyncMock())
        controller=Controller(models,MemoryLog());controller.advance.cache=cached(data)
        controller.plan_retry_at=controller.last_decision=float('inf')
        original=controller.advance.cache
        try:
            controller.step(w)
            self.assertIs(controller.advance.cache,original)
            self.assertNotEqual(controller.execution_plan.get('goal_source'),'advance')
            w.ingest(data['settled_snapshot']);controller.step(w)
            self.assertEqual(w.frame-data['partial_snapshot']['frame'],3)
            self.assertEqual(controller.execution_plan['goal_source'],'advance')
            self.assertEqual(controller.execution_plan['boss_focus'],data['preparation']['boss_focus'])
            self.assertNotIn('resource_action',controller.execution_plan)
            self.assertFalse(any(r['event']=='advance_invalidated' for r in controller.log.rows))
            self.assertTrue(any(r['event']=='advance_revalidated' for r in controller.log.rows))
            models.plan.assert_not_called();models.prepare.assert_not_called();models.decide.assert_not_called()
        finally:await controller.close()

    async def test_reply_arriving_during_missing_channels_waits_without_repeating_request(self):
        w=setup();gate=asyncio.Event()
        async def prepare(obs):await gate.wait();return preparation(),{'usage':{'total_tokens':123}}
        models=SimpleNamespace(mode='hybrid',prepare=AsyncMock(side_effect=prepare))
        log=MemoryLog();planner=AdvancePlanner(models,log)
        try:
            planner.maybe_start(w,'tactical',False);await asyncio.sleep(0)
            inventory=w.payload.pop('PLAYER_INVENTORY');stats=w.payload.pop('PLAYER_STATS')
            gate.set();await planner.task
            self.assertIsNone(planner.current(w));self.assertIsNotNone(planner.cache)
            self.assertEqual(log.rows[-1]['event'],'advance_waiting_for_state')
            self.assertEqual(log.rows[-1]['usage']['total_tokens'],123)
            planner.maybe_start(w,'tactical',False,time.monotonic()+50)
            self.assertEqual(models.prepare.await_count,1)
            w.payload.update(PLAYER_INVENTORY=inventory,PLAYER_STATS=stats);refresh(w,1)
            self.assertIsNotNone(planner.current(w))
            self.assertEqual(models.prepare.await_count,1)
        finally:await planner.close()

    def test_real_build_change_is_rejected_after_fresh_channels_return(self):
        data=fixture();w=World();w.ingest(data['partial_snapshot'])
        planner=AdvancePlanner(SimpleNamespace(),MemoryLog());planner.cache=cached(data)
        self.assertIsNone(planner.current(w));self.assertIsNotNone(planner.cache)
        w.ingest(data['settled_snapshot']);w.inventory['collectibles']['999']=1
        self.assertIsNone(planner.current(w));self.assertIsNone(planner.cache)
        self.assertEqual(planner.log.rows[-1]['reason'],'build changed')

    def test_floor_death_and_expiry_do_not_wait_for_missing_inventory(self):
        for change in ('floor','death','expired'):
            data=fixture();w=World();w.ingest(data['partial_snapshot'])
            p=AdvancePlanner(SimpleNamespace(),MemoryLog());p.cache=cached(data)
            if change=='floor':w.level+=':changed'
            elif change=='death':w.dead=True
            else:p.cache['completed_at']-=181
            self.assertIsNone(p.current(w));self.assertIsNone(p.cache)

    def test_resources_and_missing_health_do_not_invalidate_unchanged_build(self):
        data=fixture();w=World();w.ingest(data['settled_snapshot'])
        p=AdvancePlanner(SimpleNamespace(),MemoryLog());p.cache=cached(data)
        self.assertEqual(build_key(w),data['source_build_key'])
        health=w.payload.pop('PLAYER_HEALTH')
        self.assertIsNone(p.current(w));self.assertIsNotNone(p.cache)
        w.payload['PLAYER_HEALTH']=health;refresh(w)
        w.inventory['coins']+=1;w.inventory['keys']+=1;w.health['red_hearts']-=1
        self.assertTrue(p.current(w)['resources_changed']);self.assertIsNotNone(p.cache)
