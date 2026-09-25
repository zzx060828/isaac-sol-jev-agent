import asyncio
from copy import deepcopy
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from test_agent import MemoryLog
from test_safety_strategy import prepared, refresh
from test_secret_search import setup as secret_world
from isaac_agent.advance import AdvancePlanner, validate_preparation
from isaac_agent.floor_map import floor_map
from isaac_agent.control import Action
from isaac_agent.runtime import Controller
from isaac_agent.planning import enrich_planning


def preparation():
    return {'objective':'Prepare spacing and recovery for the next boss',
            'boss_range_fraction':.86,'boss_focus':'spawners_first',
            'next_checks':['heal_before_boss','compare_deal'],
            'resource_reasoning':'Recheck health before spending',
            'uncertainties':'Boss identity is not known'}


def setup():
    w=prepared();w.info.update(room_type=1,is_clear=False)
    w.payload['ENEMIES']=[dict(id=19,type=10,variant=0,is_vulnerable=True,
                               pos=dict(x=500,y=280),vel=dict(x=0,y=0))]
    w.layout['doors']={'2':dict(x=600,y=280,target_room=2,target_room_type=5,is_open=False)}
    w.rooms={str(w.room):dict(type=1,shape=1,doors=deepcopy(w.layout['doors']))}
    refresh(w)
    return w


class MapTests(unittest.TestCase):
    def test_graph_frontier_distance_and_missing_walls_are_explicit(self):
        w=setup(); graph=floor_map(w)
        frontier=graph['unvisited_observed_rooms'][0]
        self.assertEqual(frontier['id'],2)
        self.assertEqual(frontier['observed_hops'],1)
        self.assertEqual(frontier['first_door_slot'],'2')
        self.assertFalse(graph['coverage']['no_known_frontier'])
        self.assertFalse(graph['coverage']['engine_complete_map'])
        self.assertIn(w.room,graph['coverage']['wall_geometry_missing'])

    def test_global_hypotheses_are_advisory_and_failed_cells_removed(self):
        w=secret_world(); graph=floor_map(w)
        candidate=next(c for c in graph['secret_hypotheses'] if c['cell']==85)
        self.assertEqual(candidate['hypothesis'],'regular')
        self.assertFalse(candidate['executable_now'])
        self.assertEqual(set(candidate['observed_neighbors']),{84,72})
        w.run_memory.secret_attempts[w.level+':85']={'status':'no_entrance_observed'}
        self.assertNotIn(85,[c['cell'] for c in floor_map(w)['secret_hypotheses']])

    def test_super_secret_contract_requires_an_ordinary_neighbor(self):
        from isaac_agent.secrets import offers
        w=secret_world();w.rooms.pop('72');w.rooms['99']={'type':5,'shape':1}
        self.assertTrue(any(o['evidence']['hypothesis']=='super_secret_guess' for o in offers(w)))
        w.rooms['84']['type']=2;w.info['room_type']=2
        self.assertFalse(offers(w))
        for candidate in floor_map(w)['secret_hypotheses']:
            self.assertNotIn('super',candidate['possible_types'])

    def test_sol_gets_all_secret_types_without_expanding_jev_observation(self):
        w=secret_world();original=w.observation();enriched=enrich_planning(w,original)
        self.assertNotIn('floor_map',original)
        self.assertIn('floor_map',enriched)
        topics={b['topic'] for b in enriched['expert_knowledge']['briefs']}
        self.assertTrue({'secret_regular','secret_super','secret_ultra'}.issubset(topics))
        self.assertFalse(enriched['expert_knowledge']['omitted_topics'])

    def test_preparation_cannot_create_action_permissions(self):
        p=validate_preparation({**preparation(),'resource_action':'bomb',
            'strategy_choice':'buy_anything','target_enemy':999,'boss_range_fraction':100})
        self.assertEqual(p['boss_range_fraction'],.88)
        self.assertNotIn('strategy_choice',p);self.assertNotIn('resource_action',p)
        for value in (float('nan'),True,'far'):
            with self.assertRaises(ValueError):
                validate_preparation({**preparation(),'boss_range_fraction':value})
        with self.assertRaises(ValueError):
            validate_preparation({**preparation(),'next_checks':['spend_all_bombs']})

    def test_golden_battery_is_neither_routine_nor_an_unbudgeted_trade(self):
        from isaac_agent.control import pickup_target, blocked
        from isaac_agent.strategy import offers
        w=prepared();w.payload['ENEMIES']=[];w.info['is_clear']=True
        w.inventory['active_items']['0']['charge']=0
        w.payload['PICKUPS']=[dict(id=77,variant=90,sub_type=4,price=0,pos=dict(x=400,y=280))]
        refresh(w)
        self.assertIsNone(pickup_target(w,{}))
        self.assertFalse(any(o['entity'].get('id')==77 for o in offers(w)))
        self.assertTrue(blocked(w,(400,280),12))
        w.payload['PICKUPS'][0]['sub_type']=2
        self.assertEqual(pickup_target(w,{}),(400,280))

    def test_large_room_and_resource_memory_survive_continue_without_wall_inference(self):
        import tempfile
        from pathlib import Path
        from isaac_agent.memory import RunMemory
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'map.json';w=setup();m=RunMemory(path);w.run_memory=m
            w.rooms[str(w.room)].update(visited=True,shape=9,walls={},
                resources=[dict(variant=10,sub_type=1,price=0)],collectibles=[],floor_exit=False)
            m.observe(w,{})
            restored=setup();restored.rooms={};restored.room=3;restored.frame=0
            restored.capabilities['continued_run']=True
            RunMemory(path).observe(restored,{})
            node=next(n for n in floor_map(restored)['nodes'] if n['id']==w.room)
            self.assertEqual(node['shape'],9)
            self.assertFalse(node['walls_observed'])
            self.assertEqual(node['resources_last_seen'][0]['variant'],10)


class AdvanceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.w=setup();self.log=MemoryLog()
        self.models=SimpleNamespace(mode='hybrid',prepare=AsyncMock(return_value=(preparation(),{})),
            plan=AsyncMock(return_value=({'objective':'Live boss assessment','mode':'combat'},{})),
            decide=AsyncMock(return_value=(Action(),{'confidence':0})))
        self.p=AdvancePlanner(self.models,self.log)

    async def asyncTearDown(self):
        await self.p.close()

    async def test_cached_plan_survives_room_change_but_not_build_or_floor(self):
        self.p.maybe_start(self.w,'tactical',False)
        await self.p.task
        self.w.room=2
        self.assertIsNotNone(self.p.current(self.w))
        self.w.health['red_hearts']-=1
        self.assertTrue(self.p.current(self.w)['resources_changed'])
        self.w.inventory['collectibles']['999']=1
        self.assertIsNone(self.p.current(self.w))
        self.w.inventory['collectibles'].pop('999');self.w.level+='other'
        self.assertIsNone(self.p.current(self.w))

    async def test_no_heartbeat_or_spawning_during_foreground_decision(self):
        self.p.maybe_start(self.w,'sol',False)
        self.p.maybe_start(self.w,'tactical',True)
        self.assertIsNone(self.p.task)
        now=time.monotonic();self.p.maybe_start(self.w,'tactical',False,now)
        await self.p.task
        for t in (now+50,now+100,now+200):
            self.w.health['red_hearts']=6
            self.p.maybe_start(self.w,'tactical',False,t)
        self.assertEqual(self.models.prepare.await_count,1)
        self.w.inventory['collectibles']['999']=1
        self.p.maybe_start(self.w,'tactical',False,now+250);await self.p.task
        self.w.inventory['collectibles']['998']=1
        self.p.maybe_start(self.w,'tactical',False,now+350)
        self.assertEqual(self.models.prepare.await_count,2)

    async def test_late_old_floor_response_is_discarded(self):
        gate=asyncio.Event()
        async def delayed(obs):
            await gate.wait();return preparation(),{'usage':{'total_tokens':123}}
        self.models.prepare.side_effect=delayed
        self.p.maybe_start(self.w,'tactical',False);await asyncio.sleep(0)
        self.w.level+='new';gate.set();await self.p.task
        self.assertIsNone(self.p.current(self.w))
        self.assertEqual(self.log.rows[-1]['event'],'advance_discarded')
        self.assertEqual(self.log.rows[-1]['usage']['total_tokens'],123)

    async def test_pending_preparation_does_not_block_actions_or_live_boss_request(self):
        gate=asyncio.Event()
        async def delayed(obs):
            await gate.wait();return preparation(),{}
        self.models.prepare.side_effect=delayed
        c=Controller(self.models,self.log)
        try:
            wire=c.step(self.w);await asyncio.sleep(0)
            self.assertIn('move',wire)
            self.assertEqual(self.models.prepare.await_count,1)
            self.assertFalse(c.advance.task.done())
            self.w.room=2;self.w.info['room_type']=5;refresh(self.w,1)
            c.step(self.w);await asyncio.sleep(0)
            self.assertEqual(self.models.plan.await_count,1)
            self.assertFalse(c.advance.task.done())
            gate.set();await c.advance.task
        finally:await c.close()

    async def test_ready_preparation_applies_only_combat_preferences_at_boss_entry(self):
        c=Controller(self.models,self.log)
        try:
            c.step(self.w);await c.advance.task
            self.w.room=2;self.w.info['room_type']=5;refresh(self.w,1)
            command=c.step(self.w)
            self.assertEqual(c.execution_plan['goal_source'],'advance')
            self.assertEqual(c.execution_plan['boss_focus'],'spawners_first')
            self.assertFalse(command['use_bomb'])
            self.assertIsNone(c.execution_plan.get('strategy_choice'))
            self.assertTrue(any(r['event']=='advance_applied' for r in self.log.rows))
        finally:await c.close()

    async def test_prepare_client_uses_sol_and_bounds_timeout(self):
        from isaac_agent.models import Models
        import json
        model=Models({'astra':{'model':'gpt-5.6-sol','timeout':25}},mode='local')
        result={'choices':[{'message':{'content':json.dumps(preparation())}}],'usage':{'total_tokens':12}}
        with patch('isaac_agent.models.post',return_value=result) as post:
            _,meta=await model.prepare({'purpose':'boss_preparation'})
        self.assertEqual(post.call_args.args[0]['timeout'],60)
        self.assertEqual(post.call_args.args[1]['model'],'gpt-5.6-sol')
        self.assertEqual(meta['usage']['total_tokens'],12)
