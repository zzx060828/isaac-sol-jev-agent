from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.state import World
from isaac_agent.demo import frame_message
from isaac_agent.scavenging import reachable,targets
from isaac_agent.strategy import StrategyExecutor
from isaac_agent.runtime import Controller
from test_agent import MemoryLog
from test_safety_strategy import refresh


def setup():
    w=World();w.ingest(frame_message(30,player=(220,240)))
    w.payload.update(BOMBS=[],INTERACTABLES=[],PICKUPS=[],FIRE_HAZARDS=[
        dict(id=11,type='FIREPLACE',variant=0,pos=dict(x=360,y=240),collision_radius=20)])
    w.inventory.update(pill_0=0,card_0=0,active_items={})
    w.layout['grid']={str(i):dict(x=320,y=y,type=14 if y==400 else 2,variant=0,collision=3)
                      for i,y in enumerate(range(160,421,40))}
    refresh(w)
    return w


def plan():return dict(strategy_enabled=True,strategy_choice='scavenge',mode='collect')


def open_path(w):
    w.layout['grid']['6']['collision']=0
    w.navigation_revision+=1;w.path_cache.clear();w.unreachable_paths.clear();refresh(w,1)


class ScavengeReachabilityTests(unittest.IsolatedAsyncioTestCase):
    def test_opening_observed_path_reconsiders_ignored_source(self):
        w=setup();log=MemoryLog();e=StrategyExecutor()
        sources={t['key']:t for t in targets(w)}
        self.assertFalse(reachable(w,sources['fire:11']))
        self.assertTrue(reachable(w,sources['poop:6']))
        first=e.update(w,plan(),log)
        self.assertEqual(first['shoot_target']['key'],'poop:6')
        self.assertEqual(e.scavenge_ignored,{'fire:11'})
        started=e.started;open_path(w)
        self.assertTrue(reachable(w,sources['fire:11']))
        second=e.update(w,plan(),log)
        self.assertEqual(second['shoot_target']['key'],'fire:11')
        self.assertEqual(e.started,started)
        self.assertNotIn('scavenge',e.finished)
        self.assertEqual([r['targets'] for r in log.rows if r['event']=='scavenge_reconsidered'],[['fire:11']])

    def test_damage_to_intact_obstacle_does_not_invalidate_failed_approach(self):
        w=setup();e=StrategyExecutor();log=MemoryLog();e.update(w,plan(),log)
        old_basis=e.scavenge_reachability_basis
        w.layout['grid']['6']['state']=1;w.navigation_revision+=1;w.path_cache.clear();refresh(w,1)
        e.scavenge_target=None # A target reselection under unchanged walkability.
        result=e.update(w,plan(),log)
        self.assertEqual(result['shoot_target']['key'],'poop:6')
        self.assertEqual(e.scavenge_reachability_basis,old_basis)
        self.assertFalse(any(r['event']=='scavenge_reconsidered' for r in log.rows))

    def test_relocation_rechecks_sources_and_budget_still_ends_sweep(self):
        w=setup();e=StrategyExecutor();log=MemoryLog();e.update(w,plan(),log)
        w.player['pos']=dict(x=400,y=240);refresh(w,1);e.scavenge_target=None
        result=e.update(w,plan(),log)
        self.assertEqual(result['shoot_target']['key'],'fire:11')
        refresh(w,601);result=e.update(w,plan(),log)
        self.assertIn('scavenge',e.finished)
        self.assertNotIn('shoot_target',result)
        self.assertEqual(log.rows[-1]['result'],'timeout')

    async def test_controller_continues_same_authorization_without_jev_or_sol(self):
        w=setup();m=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),decide=AsyncMock())
        c=Controller(m,MemoryLog());c.reset_actions(w);c.plan=plan()
        try:
            c.step(w);started=c.strategy.started
            open_path(w);c.step(w)
            self.assertEqual(c.execution_plan['shoot_target']['key'],'fire:11')
            self.assertEqual(c.strategy.started,started)
            self.assertEqual(c.plan['strategy_choice'],'scavenge')
            for method in (m.plan,m.choose_goal,m.decide):method.assert_not_called()
        finally:await c.close()
