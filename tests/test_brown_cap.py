from copy import deepcopy
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import unittest

from isaac_agent.control import candidates,shield,Action
from isaac_agent.hazards import explosives,poop_explosions,safe_shot,threatened_explosives
from isaac_agent.scavenging import targets,reachable,explosive_firing_spot
from isaac_agent.strategy import offers,StrategyExecutor
from isaac_agent.runtime import Controller
from test_agent import world,MemoryLog
from test_safety_strategy import refresh


def prepared():
    w=world(player=(250,280));w.stats['range']=260
    w.inventory['trinket_0']=90
    w.payload.update(BOMBS=[],FIRE_HAZARDS=[],INTERACTABLES=[])
    w.layout['grid']={'1':dict(type=14,variant=0,collision=2,x=320,y=280)}
    refresh(w)
    return w


class BrownCapTests(unittest.TestCase):
    def test_normal_golden_and_effective_cap_classify_all_intact_poop(self):
        w=prepared()
        for slot in ('trinket_0','trinket_1'):
            for value in (90,90|0x8000):
                w.inventory.update(trinket_0=0,trinket_1=0);w.inventory[slot]=value
                self.assertTrue(poop_explosions(w));self.assertEqual(len(explosives(w)),1)
        w.inventory.update(trinket_0=0,trinket_1=0,poop_explosions=True)
        for variant in (0,1,2,3,4):
            w.layout['grid']['1']['variant']=variant
            self.assertEqual(len(explosives(w)),1)
        w.layout['grid']['1']['collision']=0
        self.assertEqual(explosives(w),[])
        w.inventory['poop_explosions']=False
        self.assertFalse(poop_explosions(w))

    def test_near_shots_and_chain_reactions_are_blocked_but_safe_distance_fires(self):
        w=prepared()
        self.assertFalse(safe_shot(w,(0,0),(1,0)))
        w.player['pos']['x']=140
        self.assertTrue(safe_shot(w,(0,0),(1,0)))
        w.layout['grid']['2']=dict(type=12,collision=2,x=220,y=280)
        self.assertFalse(safe_shot(w,(0,0),(1,0)))
        w.payload['PROJECTILES']['player_tears']=[dict(pos=dict(x=290,y=280),vel=dict(x=5,y=0))]
        sources=explosives(w)
        self.assertEqual(len(threatened_explosives(w,sources)),2)

    def test_sweep_repositions_then_fires_without_reauthorizing(self):
        w=prepared();executor=StrategyExecutor();log=MemoryLog()
        plan=dict(strategy_enabled=True,strategy_choice='scavenge',mode='collect')
        execution=executor.update(w,plan,log)
        self.assertTrue(execution['shoot_target']['explosive'])
        self.assertEqual(next(o for o in offers(w) if o['kind']=='scavenge')['explosive_sources'],1)
        action=candidates(w,execution)[0][0]['action']
        self.assertLess(action.move[0],0);self.assertEqual(action.shoot,(0,0))
        w.player['pos']['x']=140;refresh(w,1)
        execution=executor.update(w,plan,log)
        action=candidates(w,execution)[0][0]['action']
        self.assertEqual(action.shoot,(1,0));self.assertTrue(safe_shot(w,action.move,action.shoot))
        self.assertEqual(sum(r['event']=='strategy_start' for r in log.rows),1)

    def test_no_safe_in_range_position_means_no_sweep_offer(self):
        w=prepared();w.stats['range']=100
        self.assertFalse(reachable(w,targets(w)[0]))
        self.assertFalse(any(o['kind']=='scavenge' for o in offers(w)))
        execution=dict(strategy_enabled=True,shoot_target=targets(w)[0])
        self.assertTrue(all(o['action'].shoot==(0,0) for o in candidates(w,execution)[0]))

    def test_lost_explosive_approach_switches_to_other_source_without_resetting_budget(self):
        w=prepared();executor=StrategyExecutor();log=MemoryLog()
        plan=dict(strategy_enabled=True,strategy_choice='scavenge')
        executor.update(w,plan,log);started=executor.started
        w.stats['range']=100
        w.payload['FIRE_HAZARDS']=[dict(id=11,type='FIREPLACE',variant=0,pos=dict(x=250,y=340),collision_radius=20)]
        refresh(w,1)
        execution=executor.update(w,plan,log)
        self.assertEqual(execution['shoot_target']['key'],'fire:11')
        self.assertEqual(executor.started,started)
        self.assertEqual(sum(r['event']=='strategy_start' for r in log.rows),1)

    def test_spot_search_does_not_mutate_observed_world_and_avoids_chained_barrel(self):
        w=prepared();w.player['pos']['x']=260;w.layout['grid']['1']['x']=340
        w.layout['grid']['2']=dict(type=12,collision=2,x=220,y=280)
        before=deepcopy(w.payload)
        point=explosive_firing_spot(w,targets(w)[0])
        self.assertIsNotNone(point)
        self.assertGreater(point[0],340)
        self.assertEqual(w.payload,before)

    def test_reflex_filters_unsafe_model_shooting_and_cap_loss_restores_normal_poop(self):
        w=prepared();desired=Action((0,0),(1,0))
        result,overridden=shield(w,desired,Action(),None)
        self.assertTrue(overridden);self.assertEqual(result.shoot,(0,0))
        w.inventory['trinket_0']=0
        self.assertTrue(safe_shot(w,(0,0),(1,0)))
        self.assertNotIn('explosive',targets(w)[0])


class BrownCapControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_sweep_retreats_then_shoots_without_more_model_calls(self):
        w=prepared()
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),decide=AsyncMock())
        c=Controller(models,MemoryLog());c.reset_actions(w)
        c.plan.update(strategy_choice='scavenge',mode='collect',awaiting_strategy=False)
        try:
            command=c.step(w)
            self.assertLess(command['move']['x'],0)
            self.assertEqual(command['shoot'],dict(x=0,y=0));self.assertFalse(command['use_bomb'])
            w.player['pos']['x']=140;refresh(w,1)
            command=c.step(w)
            self.assertEqual(command['shoot'],dict(x=1,y=0))
            await asyncio.sleep(0)
            models.plan.assert_not_called();models.choose_goal.assert_not_called();models.decide.assert_not_called()
        finally:await c.close()
