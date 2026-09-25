import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from isaac_agent.runtime import Controller
from isaac_agent.strategy import StrategyExecutor
from test_scavenging import prepared
from test_safety_strategy import refresh
from test_agent import MemoryLog


def setup():
    w=prepared();w.payload.update(BOMBS=[],INTERACTABLES=[])
    w.inventory.update(pill_0=0,card_0=0,coins=0,active_items={})
    refresh(w)
    return w


def coin(**change):
    return dict(id=99,variant=20,sub_type=1,price=0,options_index=0,
                pos=dict(x=250,y=320))|change


def plan():return dict(strategy_enabled=True,strategy_choice='scavenge',mode='collect')


class ScavengeCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_drop_is_collected_then_same_sweep_resumes_without_models(self):
        w=setup();m=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),decide=AsyncMock())
        c=Controller(m,MemoryLog());c.reset_actions(w);c.plan=plan()
        try:
            c.step(w)
            self.assertEqual(c.execution_plan['shoot_target']['key'],'fire:11')
            started=c.strategy.started
            w.payload['PICKUPS']=[coin()];refresh(w,1)
            command=c.step(w)
            self.assertTrue(c.execution_plan['scavenge_resource_collection'])
            self.assertEqual(c.execution_plan['navigation_target'],(250,320))
            self.assertNotIn('shoot_target',c.execution_plan)
            self.assertEqual(command['shoot'],{'x':0,'y':0})
            self.assertIsNotNone(c.pickup_progress.previous)
            w.payload['PICKUPS']=[];w.inventory['coins']+=1;refresh(w,1);c.step(w)
            self.assertEqual(c.execution_plan['shoot_target']['key'],'fire:11')
            self.assertEqual(c.strategy.started,started)
            self.assertNotIn('scavenge',c.strategy.finished)
            for method in (m.plan,m.choose_goal,m.decide):method.assert_not_called()
        finally:await c.close()

    def test_special_paid_options_bombs_and_stale_channels_do_not_authorize_touch(self):
        for case in ('paid','options','troll','armed','stale','full'):
            w=setup();w.payload['PICKUPS']=[coin()]
            if case=='paid':w.pickups[0]['price']=1
            elif case=='options':w.pickups[0]['options_index']=2
            elif case=='troll':w.pickups[0].update(variant=40,sub_type=3)
            elif case=='armed':w.payload['BOMBS']=[dict(pos=dict(x=250,y=320))]
            elif case=='stale':w.sampled['BOMBS']=w.frame-4
            else:w.inventory['coins']=99
            execution=StrategyExecutor().update(w,plan(),MemoryLog())
            self.assertFalse(execution.get('scavenge_resource_collection'),case)
            self.assertFalse(execution.get('navigation_contact'),case)

    async def test_failed_pickup_yields_back_to_sweep(self):
        w=setup();m=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock())
        c=Controller(m,MemoryLog());c.reset_actions(w);c.plan=plan()
        c.pickup_progress.STAGNANT_FRAMES=3 # Existing tests cover the production 120-update budget.
        try:
            w.payload['PICKUPS']=[coin()]
            for _ in range(5):
                refresh(w,1);c.step(w) # Fault injection: the player never moves.
            self.assertTrue(w.pickup_failures)
            self.assertFalse(c.execution_plan.get('scavenge_resource_collection'))
            self.assertEqual(c.execution_plan['shoot_target']['key'],'fire:11')
            self.assertEqual(sum(r['event']=='local_pickup_failed' for r in c.log.rows),1)
            m.plan.assert_not_called();m.choose_goal.assert_not_called()
        finally:await c.close()

    def test_collecting_does_not_reset_sweep_budget(self):
        w=setup();e=StrategyExecutor();log=MemoryLog();e.update(w,plan(),log)
        w.payload['PICKUPS']=[coin()];refresh(w,601)
        result=e.update(w,plan(),log)
        self.assertIn('scavenge',e.finished)
        self.assertFalse(result.get('scavenge_resource_collection'))
        self.assertEqual(log.rows[-1]['result'],'timeout')

    async def test_chest_settling_clears_sweep_shooting_until_loot_is_fresh(self):
        w=setup();w.payload['PICKUPS']=[coin(variant=50)]
        c=Controller(SimpleNamespace(mode='hybrid',plan=AsyncMock()),MemoryLog())
        c.reset_actions(w);c.plan=plan()
        try:
            c.step(w)
            self.assertTrue(c.execution_plan['scavenge_resource_collection'])
            w.pickups[0]['sub_type']=0;refresh(w,1)
            command=c.step(w)
            self.assertEqual(c.route,'settle_chest')
            self.assertNotIn('shoot_target',c.execution_plan)
            self.assertFalse(c.execution_plan.get('scavenge_resource_collection'))
            self.assertIsNone(c.pickup_progress.previous)
            self.assertEqual(command['shoot'],{'x':0,'y':0})
            refresh(w,7);c.step(w)
            self.assertEqual(c.execution_plan['shoot_target']['key'],'fire:11')
        finally:await c.close()
