import asyncio
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.models import validate_plan
from isaac_agent.post_pickup import PostPickup
from isaac_agent.runtime import Controller
from test_consumable_decisions import prepared
from test_safety_strategy import refresh
from test_planning import MemoryLog


def ground(card=4):
    w=prepared();w.payload['PICKUPS']=[dict(id=81,variant=300,sub_type=card,
        price=0,wait=0,options_index=0,pos=dict(x=400,y=280))]
    return w


def plan_for(w,action):
    obs=w.observation();choice=next(o['choice'] for o in obs['strategy_offers'] if o['kind']=='pickup')
    return validate_plan(dict(objective='Collect supplies',mode='collect',strategy_choice=choice,
                              after_pickup=action),obs)


class PostPickupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from synthetic_eid import enable
        enable(self)

    def test_schema_only_accepts_selected_consumable_and_no_current_use(self):
        w=ground()
        for value in ('keep','use'):self.assertEqual(plan_for(w,value)['after_pickup'],value)
        obs=w.observation();p=plan_for(w,'keep')
        for changes in ({'strategy_choice':None},{'after_pickup':'maybe'},
                        {'after_pickup':{'action':'keep'}},{'resource_action':'card'}):
            with self.assertRaises(ValueError):validate_plan(p|changes,obs)
        w.payload['PICKUPS'][0].update(variant=100,sub_type=30)
        with self.assertRaises(ValueError):plan_for(w,'keep')

    async def run_chain(self,action,card):
        w=ground(card);reply=plan_for(w,action)
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=(reply,{})))
        c=Controller(models,MemoryLog())
        try:
            c.step(w);await c.plan_task;refresh(w,1);c.step(w)
            # Disappearance arrives before the inventory acquisition.
            w.payload['PICKUPS']=[];refresh(w,1);command=c.step(w)
            self.assertEqual(c.route,'settle_consumable');self.assertFalse(command['use_card'])
            self.assertEqual(models.plan.await_count,1)
            w.inventory['card_0']=card;refresh(w,1);command=c.step(w)
            self.assertEqual(command['use_card'],action=='use')
            if action=='use':self.assertEqual(command['agent_resource_id'],card)
            self.assertEqual(models.plan.await_count,1)
            self.assertTrue(any(r['event']=='after_pickup_confirmed' for r in c.log.rows))
            refresh(w,31);c.last_plan=-100;c.step(w);await asyncio.sleep(0)
            self.assertEqual(models.plan.await_count,1)
            self.assertEqual(sum(r['event']=='resource_use' for r in c.log.rows),int(action=='use'))
        finally:await c.close()

    async def test_keep_avoids_the_second_sol_request(self):await self.run_chain('keep',4)
    async def test_keep_is_not_overridden_by_nonurgent_auto_card(self):await self.run_chain('keep',6)
    async def test_use_reaches_one_pulse_without_second_sol(self):await self.run_chain('use',9)

    async def test_pickup_animation_preserves_keep_and_delays_use(self):
        for action in ('keep','use'):
            w=ground(9);reply=plan_for(w,action)
            models=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=(reply,{})))
            c=Controller(models,MemoryLog())
            try:
                c.step(w);await c.plan_task;refresh(w,1);c.step(w)
                w.payload['PICKUPS']=[];w.inventory.update(card_0=9,can_use=False);refresh(w,1)
                self.assertFalse(c.step(w)['use_card'])
                w.inventory['can_use']=True;refresh(w,1);c.last_plan=-100
                command=c.step(w);await asyncio.sleep(0)
                self.assertEqual(command['use_card'],action=='use')
                self.assertEqual(models.plan.await_count,1)
            finally:await c.close()

    def test_no_assumed_acquisition_and_changed_context_cancels(self):
        for change in ('room','health','build','target','resources'):
            w=ground();p=PostPickup();p.arm(w,w.observation(),plan_for(w,'use'));log=MemoryLog()
            refresh(w,1)
            if change=='room':w.room+=1
            if change=='health':w.health['red_hearts']-=1
            if change=='build':w.inventory['collectibles']['1']=1
            if change=='resources':w.inventory['bombs']+=1
            if change=='target':w.payload['PICKUPS'][0]['sub_type']=9
            self.assertIsNone(p.update(w,log));self.assertIsNone(p.pending)
        w=ground();p=PostPickup();p.arm(w,w.observation(),plan_for(w,'use'))
        w.inventory['card_0']=4;refresh(w,1)
        self.assertIsNone(p.update(w,MemoryLog())) # Matching slot alone is insufficient.
        w=ground();w.inventory['card_0']=4;p=PostPickup()
        p.arm(w,w.observation(),plan_for(w,'keep'));self.assertIsNone(p.pending)

    def test_missing_receipt_times_out_and_unknown_pill_retains_policy(self):
        w=ground();p=PostPickup();p.arm(w,w.observation(),plan_for(w,'use'))
        w.payload['PICKUPS']=[];refresh(w,1);self.assertIsNone(p.update(w,MemoryLog()))
        refresh(w,30);self.assertIsNone(p.update(w,MemoryLog()));self.assertIsNone(p.pending)
        w=ground();w.health['red_hearts']=2
        w.payload['PICKUPS'][0].update(variant=70,sub_type=1)
        p=PostPickup();p.arm(w,w.observation(),plan_for(w,'use'))
        w.payload['PICKUPS']=[];w.inventory.update(pill_0=1,pill_identified=False);refresh(w,1)
        log=MemoryLog();self.assertIsNone(p.update(w,log))
        self.assertEqual(log.rows[-1]['reason'],'use_conditions_changed')

    def test_paid_card_checks_observed_price_after_acquisition(self):
        w=ground();w.inventory['coins']=10;w.payload['PICKUPS'][0]['price']=5
        p=PostPickup();p.arm(w,w.observation(),plan_for(w,'keep'))
        refresh(w,1);self.assertIsNone(p.update(w,MemoryLog()));self.assertIsNotNone(p.pending)
        w.inventory['coins']=5;refresh(w,1)
        self.assertIsNone(p.update(w,MemoryLog()));self.assertIsNotNone(p.pending)
        w.payload['PICKUPS']=[];w.inventory['card_0']=4;refresh(w,1)
        self.assertEqual(p.update(w,MemoryLog())['action'],'keep')
