import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from test_safety_strategy import prepared, refresh
from test_agent import MemoryLog
from isaac_agent.deals import health_contract
from isaac_agent.strategy import offers, StrategyExecutor
from isaac_agent.control import traversable_door, target_point, blocked
from isaac_agent.memory import RunMemory
from isaac_agent.models import Models


def deal_world():
    w=prepared();w.info.update(is_clear=True,enemy_count=0,room_type=14)
    w.payload['ENEMIES']=[];w.stats['player_type']=1
    w.capabilities['repentance_plus']=False
    w.health.update(max_hearts=8,red_hearts=7,soul_hearts=2)
    w.payload['PICKUPS']=[{'id':9,'variant':100,'sub_type':51,'price':-2,'pos':{'x':450,'y':280}}]
    w.run_memory=RunMemory();w.run_memory.observe(w,{})
    refresh(w)
    return w


class DealTests(unittest.TestCase):
    def test_container_cost_clamps_remaining_red_and_counts_black_only_once(self):
        w=deal_world();w.health['black_hearts']=3
        c=health_contract(w,w.pickups[0])
        self.assertEqual(c['remaining_red_half_hearts'],4)
        self.assertEqual(c['remaining_soul_half_hearts'],2)
        w.health.update(max_hearts=4,red_hearts=4,soul_hearts=2)
        self.assertIsNone(health_contract(w,w.pickups[0]))

    def test_mixed_soul_and_unknown_price_contracts(self):
        w=deal_world();item=w.pickups[0];item['price']=-4
        self.assertIsNone(health_contract(w,item))
        w.health['soul_hearts']=6
        self.assertEqual(health_contract(w,item)['remaining_soul_half_hearts'],2)
        for price in (-5,-6,-10,-999):
            item['price']=price
            self.assertFalse(offers(w))
        item['price']=-1000
        self.assertEqual(offers(w)[0]['coin_cost'],0)

    def test_unsupported_character_version_and_stale_health_do_not_buy(self):
        w=deal_world()
        w.stats['player_type']=10;self.assertFalse(offers(w))
        w.stats['player_type']=1;w.capabilities['repentance_plus']=True;self.assertFalse(offers(w))
        w.capabilities['repentance_plus']=False;w.sampled['PLAYER_HEALTH']=w.frame-10
        self.assertFalse(offers(w))

    def test_unselected_paid_item_blocked_and_health_drop_revokes_contact(self):
        w=deal_world();point=(450,280)
        self.assertTrue(blocked(w,point,10))
        ex=StrategyExecutor();log=MemoryLog();plan={'strategy_enabled':True,'strategy_choice':offers(w)[0]['choice']}
        execution=ex.update(w,plan,log)
        self.assertEqual(target_point(w,execution),(point,True))
        w.health.update(max_hearts=4,red_hearts=2,soul_hearts=0);refresh(w,1)
        execution=ex.update(w,plan,log)
        self.assertNotIn('navigation_target',execution)
        self.assertTrue(blocked(w,point,10))

    def test_price_change_revokes_old_choice_before_second_purchase(self):
        w=deal_world();ex=StrategyExecutor();log=MemoryLog()
        plan={'strategy_enabled':True,'strategy_choice':offers(w)[0]['choice']}
        ex.update(w,plan,log);w.pickups[0]['price']=-3
        self.assertNotIn('navigation_target',ex.update(w,plan,log))

    def test_deal_room_negative_index_is_live(self):
        w=deal_world();w.room=-1
        self.assertTrue(w.ready(time.monotonic()))
        w.info['room_type']=1
        self.assertFalse(w.ready(time.monotonic()))

    def test_skip_door_is_persisted_and_generic_path_cannot_enter(self):
        w=deal_world();w.info['room_type']=5;w.payload['PICKUPS']=[]
        door={'target_room_type':14,'target_room':-1,'is_open':True,'x':640,'y':280}
        w.layout['doors']={'2':door}
        self.assertFalse(traversable_door(w,door))
        choice=next(o for o in offers(w) if o['kind']=='deal_skip')
        ex=StrategyExecutor();ex.update(w,{'strategy_enabled':True,'strategy_choice':choice['choice']},MemoryLog())
        self.assertFalse(any(o['kind'].startswith('deal_') for o in offers(w)))
        self.assertTrue(w.run_memory.skipped)

    def test_deal_enter_normalizes_real_bridge_door_coordinates(self):
        w=deal_world();w.info['room_type']=5;w.payload['PICKUPS']=[]
        w.layout['doors']={'0':{'target_room_type':14,'target_room':-1,'is_open':True,'x':320,'y':80}}
        offer=next(o for o in offers(w) if o['kind']=='deal_enter')
        execution=StrategyExecutor().update(w,{'strategy_enabled':True,'strategy_choice':offer['choice']},MemoryLog())
        self.assertEqual(execution['navigation_target'],(320,80))
        self.assertTrue(execution['navigation_contact'])

    def test_appearing_deal_door_invalidates_pending_strategy_snapshot(self):
        w=deal_world();w.info['room_type']=5;w.layout['doors']={}
        before=w.strategy_signature()
        w.layout['doors']['0']={'target_room_type':14,'is_open':True,'x':320,'y':80}
        self.assertNotEqual(w.strategy_signature(),before)
        after=w.strategy_signature()
        w.layout['doors']['0']['is_open']=False
        self.assertNotEqual(w.strategy_signature(),after)

    def test_paid_item_memory_requires_actual_inventory_change(self):
        w=deal_world();ex=StrategyExecutor();log=MemoryLog();key=offers(w)[0]['choice']
        ex.update(w,{'strategy_enabled':True,'strategy_choice':key},log)
        w.inventory['collectibles']['51']=1;w.payload['PICKUPS']=[]
        ex.update(w,{'strategy_enabled':True,'strategy_choice':key},log)
        self.assertEqual(w.run_memory.nodes[-1]['kind'],'paid_item_acquired')

    def test_reconnect_damage_is_not_attributed_to_current_health_or_retrieved(self):
        w=deal_world();m=RunMemory();m.observe(w,{})
        m.observe(w,{'event':'PLAYER_DAMAGE','data':{'hp_before':9,'source_type':100}})
        node=m.nodes[-1]
        self.assertTrue(node['evidence']['possibly_queued_before_connection'])
        self.assertNotIn('health_snapshot',node['evidence'])
        self.assertNotIn(node,m.candidates(w))

    def test_generic_visits_do_not_spend_memory_model_calls(self):
        w=deal_world();w.info['room_type']=1;m=RunMemory();m.observe(w,{})
        self.assertEqual(m.candidates(w),[])
        refresh(w,10)
        m.observe(w,{'event':'PLAYER_DAMAGE','data':{'source_type':12,'hp_before':7}})
        self.assertEqual(m.candidates(w),[])
        w.payload['ENEMIES']=[{'id':2,'type':12,'pos':{'x':400,'y':280}}]
        w.info.update(is_clear=False,enemy_count=1)
        self.assertEqual(m.candidates(w)[0]['kind'],'damage')

    def test_memory_restores_same_run_and_resets_on_new_seed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'memory.json';w=deal_world();w.level='seed1:2:1'
            m=RunMemory(path);m.observe(w,{});m.skipped.append('door');m.save()
            n=RunMemory(path);n.observe(w,{})
            self.assertEqual(n.skipped,['door']);self.assertFalse(n.complete)
            w.level='seed2:1:1';n.observe(w,{})
            self.assertEqual(n.skipped,[])
            self.assertTrue(all(x['seed']=='seed2' for x in n.nodes))


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_low_confidence_backs_off_without_accepting_action(self):
        from isaac_agent.runtime import Controller
        from isaac_agent.control import Action, candidates
        class Fake:
            mode='jev'
            async def decide(self,*args): return Action(), {'confidence':.2}
        w=deal_world();w.payload['ENEMIES']=[{'id':2,'type':12,'pos':{'x':500,'y':280}}]
        w.info.update(is_clear=False,enemy_count=1)
        c=Controller(Fake(),MemoryLog());c.reset_actions(w);c.execution_plan=c.plan
        for expected in (1,2,3):
            refresh(w)
            await c._decide(w,w.observation(),w.token,w.frame,candidates(w,{})[0])
            self.assertGreater(c.tactical_retry_at-time.monotonic(),expected-.2)
            self.assertEqual(c.action_until,-1)
        c.reset_actions(w)
        self.assertEqual(c.tactical_retry_at,0)

    async def test_batched_retrieval_has_budget_and_does_not_invent_evidence(self):
        model=Models.__new__(Models);model.config={'jev':{'model':'test'}}
        nodes=[{'id':str(i),'evidence':'observed'} for i in range(5)]
        answers={'view':{'choice':'temporal'},'budget':{'choice':'2'},'sufficient':{'noul':.2}}
        answers.update({'m'+str(i):{'noul':.9-i*.1} for i in range(5)})
        with patch('isaac_agent.models.post',return_value={'answers':answers}):
            result,meta=await model.retrieve_memories({'run_history':{'history_complete_from_run_start':False}},nodes)
        self.assertEqual([x['id'] for x in result],['0','1'])
        self.assertEqual(meta['evidence_sufficient'],.2)
        self.assertEqual(meta['stop_reason'],'single_round_budget')
