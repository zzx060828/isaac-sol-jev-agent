import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.state import World
from isaac_agent.strategy import offers, StrategyExecutor
from isaac_agent.control import traversable_door
from isaac_agent.fast_policy import lane
from isaac_agent.runtime import Controller
from test_agent import MemoryLog


def snapshot():
    w = World()
    w.ingest(json.loads((Path(__file__).parent/'fixtures/uncleared-switch-room.json').read_text()))
    return w


class PressurePlateTests(unittest.IsolatedAsyncioTestCase):
    def test_trap_room_requires_plate_before_unlocking_treasure(self):
        w = snapshot()
        self.assertFalse(traversable_door(w, w.layout['doors']['0']))
        self.assertEqual(lane(w,w.frame)[0], 'fast')
        self.assertEqual(offers(w)[0]['choice'], 'pressure_plate:22')
        executor = StrategyExecutor(); log = MemoryLog()
        plan = dict(strategy_enabled=True, strategy_choice='pressure_plate:22')
        execution = executor.update(w, plan, log)
        self.assertEqual(execution['navigation_target'], (320,160))
        w.layout['grid']['22']['state'] = 3
        w.info['is_clear'] = True
        execution = executor.update(w,plan,log)
        self.assertNotIn('navigation_target',execution)
        self.assertIn('pressure_plate:22',executor.finished)
        self.assertTrue(traversable_door(w, w.layout['doors']['0']))

    def test_variants_combat_and_stale_geometry_do_not_authorize_plate(self):
        for variant in (1,2,3,9,10):
            w=snapshot();w.layout['grid']['22']['variant']=variant
            self.assertFalse(offers(w))
        w=snapshot();w.frame+=7
        self.assertFalse(offers(w))
        w=snapshot();w.payload['ENEMIES']=[{'id':1,'pos':{'x':320,'y':280}}]
        self.assertFalse(offers(w))
        w=snapshot();w.capabilities['repentance_plus']=True
        self.assertFalse(offers(w))

    def test_press_completion_does_not_create_spurious_slow_access_decision(self):
        from isaac_agent.planning import planning_context
        w=snapshot()
        self.assertEqual(planning_context(w)['access'], ())
        w.layout['grid']['22']['state']=3
        self.assertEqual(planning_context(w)['access'], ())
        w.info['is_clear']=True
        self.assertEqual(planning_context(w)['access'], ())  # Two keys: routine treasure.
        w.inventory['keys']=1
        self.assertEqual(planning_context(w)['access'], ('0',))
        w.inventory['keys']=2
        w.layout['doors']['0']['target_room_type']=2
        self.assertEqual(planning_context(w)['access'], ('0',))  # Shop opportunity cost.

    async def test_jev_plate_choice_executes_without_sol(self):
        w=snapshot();log=MemoryLog()
        models=SimpleNamespace(mode='hybrid',choose_goal=AsyncMock(return_value=('pressure_plate:22',{'confidence':.9})),plan=AsyncMock(),decide=AsyncMock())
        c=Controller(models,log)
        # Invoke the production fast goal path with a current inventory snapshot.
        from isaac_agent.planning import planning_context
        c.reset_actions(w)
        obs=w.observation();obs['fast_items']=planning_context(w)['items']
        await c._fast_goal(w,obs,w.token,offers(w))
        self.assertEqual(c.plan['strategy_choice'],'pressure_plate:22')
        models.plan.assert_not_called()
        c.strategy.update(w,c.plan,log)
        self.assertEqual(c.strategy.choice,'pressure_plate:22')
