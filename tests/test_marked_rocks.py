import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.control import Action
from isaac_agent.rocks import observed, offers, RockBomb
from isaac_agent.memory import RunMemory
from isaac_agent.runtime import Controller
from isaac_agent.planning import enrich_planning
from test_agent import MemoryLog
from test_secret_search import setup as base, settled_update
from test_safety_strategy import refresh


def setup(typ=4):
    w=base();w.capabilities['rock_bomb_protocol']=1
    w.layout['grid']={'67':{'grid_index':67,'type':typ,'x':320,'y':280,'collision':3,'state':1}}
    w.player['pos']={'x':400,'y':280};w.player['vel']={'x':0,'y':0}
    w.inventory['bombs']=4;refresh(w)
    return w


class MarkedRockTests(unittest.TestCase):
    def test_distinguishes_costs_and_non_marked_objects(self):
        for typ,cost in ((4,1),(22,2)):
            w=setup(typ);o=offers(w)[0]
            self.assertEqual(o['bomb_cost'],cost)
            self.assertEqual(o['bombs_reserved_after'],4-cost)
            self.assertEqual(observed(w)[0]['type'],typ)
        for typ in (2,5,6,12,26):
            self.assertFalse(observed(setup(typ)))
            self.assertFalse(offers(setup(typ)))

    def test_protocol_resources_terrain_and_freshness_gate_execution(self):
        for change in ('old_bridge','one_bomb','super_two_bombs','fast_bomb','low_hp','slow',
                       'combat','alt_stage','old_bombs','near_machine','surrounded'):
            with self.subTest(change=change):
                w=setup(22 if change=='super_two_bombs' else 4)
                if change=='old_bridge':w.capabilities.pop('rock_bomb_protocol')
                if change=='one_bomb':w.inventory['bombs']=1
                if change=='super_two_bombs':w.inventory['bombs']=2
                if change=='fast_bomb':w.inventory['trinket_0']=133
                if change=='low_hp':w.health['red_hearts']=2
                if change=='slow':w.stats['speed']=.5
                if change=='combat':w.payload['ENEMIES']=[{'id':1,'pos':dict(x=200,y=200)}]
                if change=='alt_stage':w.info['stage_type']=4
                if change=='old_bombs':w.sampled['BOMBS']=w.frame-7
                if change=='near_machine':w.payload['INTERACTABLES']=[{'variant':2,'pos':dict(x=320,y=280)}]
                if change=='surrounded':
                    for i,(x,y) in enumerate(((360,280),(280,280),(320,240),(320,320))):
                        w.layout['grid'][str(100+i)]=dict(type=7,collision=1,x=x,y=y)
                self.assertFalse(offers(w))

    def test_normal_one_pulse_retreat_then_observed_destruction(self):
        w=setup();o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=RockBomb(w,o);log=MemoryLog()
        self.assertIn('rock_bomb_request',settled_update(job,w,{},log))
        w.inventory['bombs']-=1;refresh(w,1)
        execution=job.update(w,{},log)
        self.assertNotIn('rock_bomb_request',execution)
        self.assertEqual(execution['navigation_target'],tuple(o['escape']))
        w.layout['grid']['67']['collision']=0;refresh(w,120)
        job.update(w,{},log)
        self.assertEqual(job.done,'rock_destroyed_observed')
        self.assertEqual(job.shots,1)
        self.assertFalse(offers(w))

    def test_super_rechecks_after_first_explosion_then_spends_only_second(self):
        w=setup(22);o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=RockBomb(w,o);log=MemoryLog()
        first=settled_update(job,w,{},log)['rock_bomb_request']
        w.inventory['bombs']-=1;refresh(w,121)
        second=settled_update(job,w,{},log)['rock_bomb_request']
        self.assertNotEqual(first['id'],second['id'])
        self.assertEqual(second['bombs_before'],3)
        w.inventory['bombs']-=1;refresh(w,121)
        result=job.update(w,{},log)
        self.assertNotIn('rock_bomb_request',result)
        self.assertEqual(job.done,'rock_survived_budget_exhausted')
        self.assertFalse(offers(w))

    def test_super_stops_early_if_one_explosion_destroyed_it(self):
        w=setup(22);o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=RockBomb(w,o);settled_update(job,w,{},MemoryLog())
        w.inventory['bombs']-=1;w.layout['grid']['67']['collision']=0;refresh(w,121)
        self.assertNotIn('rock_bomb_request',job.update(w,{},MemoryLog()))
        self.assertEqual(job.shots,1)
        self.assertEqual(job.done,'rock_destroyed_observed')

    def test_unconfirmed_bomb_never_repeats_even_for_super(self):
        w=setup(22);o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=RockBomb(w,o);settled_update(job,w,{},MemoryLog());refresh(w,121)
        self.assertNotIn('rock_bomb_request',job.update(w,{},MemoryLog()))
        self.assertEqual(job.done,'bomb_unconfirmed_no_retry')
        self.assertFalse(offers(w))

    def test_stale_layout_or_live_bomb_cannot_trigger_second_pulse(self):
        for channel in ('ROOM_LAYOUT','BOMBS','PLAYER_INVENTORY'):
            w=setup(22);o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
            job=RockBomb(w,o);settled_update(job,w,{},MemoryLog())
            w.inventory['bombs']-=1;refresh(w,121);w.sampled[channel]=job.fired
            self.assertNotIn('rock_bomb_request',job.update(w,{},MemoryLog()))
            self.assertEqual(job.shots,1)
        refresh(w);w.payload['BOMBS']=[{'id':2,'pos':dict(x=365,y=280)}]
        self.assertNotIn('rock_bomb_request',job.update(w,{},MemoryLog()))

    def test_super_revalidates_health_before_second_bomb(self):
        w=setup(22);o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=RockBomb(w,o);settled_update(job,w,{},MemoryLog())
        w.inventory['bombs']-=1;w.health['red_hearts']=2;refresh(w,121)
        self.assertNotIn('rock_bomb_request',job.update(w,{},MemoryLog()))
        self.assertEqual(job.done,'preflight_invalid')

    def test_attempt_memory_persists_and_is_not_secret_search_budget(self):
        with tempfile.TemporaryDirectory() as d:
            w=setup();w.run_memory.path=Path(d)/'memory.json'
            o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
            settled_update(RockBomb(w,o),w,{},MemoryLog())
            restored=RunMemory(w.run_memory.path);w.capabilities['continued_run']=True
            restored.observe(w,{})
            self.assertIn(o['attempt_key'],restored.rock_attempts)
            self.assertFalse(restored.secret_attempts)

    def test_planner_receives_rocks_even_when_old_bridge_cannot_execute(self):
        w=setup();w.capabilities.pop('rock_bomb_protocol')
        data=enrich_planning(w,w.observation())
        self.assertEqual(data['marked_rocks'][0]['name'],'Tinted Rock')
        self.assertTrue(any(b['topic']=='marked_rocks' for b in data['expert_knowledge']['briefs']))
        self.assertFalse(any(o['kind']=='rock_bomb' for o in data['strategy_offers']))


class RockRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_rock_sends_typed_pulse_and_does_not_use_other_items(self):
        w=setup();o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),
                               decide=AsyncMock(return_value=(Action(),{'confidence':0})))
        c=Controller(models,MemoryLog());c.reset_actions(w)
        c.plan.update(strategy_choice=o['choice'],mode='collect',resource_action='item')
        try:
            command=c.step(w);await asyncio.sleep(0)
            self.assertFalse(command['use_bomb']);self.assertFalse(command['use_item'])
            self.assertEqual(command['move'],{'x':0,'y':0})
            refresh(w,4);command=c.step(w)
            self.assertTrue(command['use_bomb']);self.assertFalse(command['use_item'])
            self.assertEqual(command['agent_bomb_kind'],'rock')
            self.assertEqual(command['agent_bomb_grid'],67)
            self.assertNotIn('agent_bomb_slot',command)
            refresh(w,1);self.assertFalse(c.step(w)['use_bomb'])
            models.plan.assert_not_called();models.choose_goal.assert_not_called()
        finally:await c.close()
