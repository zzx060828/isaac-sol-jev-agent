import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

from isaac_agent.resources import ResourcePolicy,bind_resource_intent,resource_preference,resource_signature
from isaac_agent.runtime import Controller
from test_agent import MemoryLog
from test_safety_strategy import prepared,refresh


def setup():
    w=prepared();w.health.update(red_hearts=8,max_hearts=8,soul_hearts=0)
    w.inventory.update(pill_0=0,card_0=0,collectibles={'105':1,'44':1,'534':1})
    w.inventory['active_items']={'0':dict(item=105,charge=6,max_charge=6,battery_charge=0,charge_type=0),
                                 '1':dict(item=44,charge=6,max_charge=6,battery_charge=0,charge_type=0)}
    w.payload['PICKUPS']=[];refresh(w)
    return w


def intent(w,kind='item'):
    plan={'objective':'Use the observed resource','mode':'collect','resource_action':kind}
    bind_resource_intent(plan,w.observation())
    return plan


class ResourceIntentTests(unittest.IsolatedAsyncioTestCase):
    async def test_slot_swap_during_sol_keeps_route_but_drops_wrong_item_use(self):
        w=setup();obs=w.observation();signature=w.strategy_signature()
        async def answer(observation):
            slots=w.inventory['active_items'];slots['0'],slots['1']=slots['1'],slots['0'];refresh(w,1)
            return {'objective':'Move on','mode':'explore','door_slot':'2','resource_action':'item'},{}
        c=Controller(SimpleNamespace(mode='hybrid',plan=answer),MemoryLog())
        try:
            await c._plan(w,obs,w.token,w.revision)
            self.assertEqual(w.strategy_signature(),signature) # Existing broad signature misses this swap.
            self.assertIsNone(c.plan['resource_action'])
            self.assertEqual(c.plan['door_slot'],'2')
            self.assertEqual(c.log.rows[-1]['event'],'plan')
            self.assertTrue(any(r.get('reason')=='resource_slot_changed' for r in c.log.rows))
            self.assertIsNone(c.resources.choose(w)) # Replacement Teleport needs its own decision.
        finally:await c.close()

    async def test_same_slot_authorization_reaches_one_runtime_pulse(self):
        w=setup();m=SimpleNamespace(mode='hybrid',plan=AsyncMock())
        c=Controller(m,MemoryLog());c.reset_actions(w)
        c.plan=intent(w);c.plan.update(strategy_enabled=True)
        c.last_plan=10**20
        try:
            command=c.step(w)
            self.assertTrue(command['use_item'])
            self.assertEqual(command['agent_resource_id'],105)
            self.assertIsNone(c.plan['resource_action'])
            self.assertNotIn('resource_authorization',c.plan)
            self.assertFalse(c.step(w)['use_item'])
            m.plan.assert_not_called()
        finally:await c.close()

    def test_delayed_execution_revalidates_charge_and_keeps_parallel_coin_change(self):
        w=setup();p=ResourcePolicy();log=MemoryLog();plan=intent(w)
        w.inventory['coins']+=1
        self.assertEqual(resource_preference(w,plan,p,log),'item')
        w.inventory['active_items']['0']['charge']=0
        self.assertIsNone(resource_preference(w,plan,p,log))
        self.assertIsNone(plan['resource_action'])

    def test_pills_cards_and_identification_bind_to_observed_slot(self):
        for kind,field in (('pill','pill_0'),('card','card_0')):
            w=setup();w.inventory[field]=10;w.inventory.update(pill_identified=True,pill_effect=5)
            plan=intent(w,kind);log=MemoryLog();p=ResourcePolicy()
            w.inventory[field]=11
            self.assertIsNone(resource_preference(w,plan,p,log))
            self.assertEqual(log.rows[-1]['reason'],'resource_slot_changed')
        w=setup();w.inventory.update(pill_0=10,pill_identified=False,pill_effect=None)
        plan=intent(w,'pill');w.inventory.update(pill_identified=True,pill_effect=4)
        self.assertIsNone(resource_preference(w,plan,ResourcePolicy(),MemoryLog()))

    def test_stale_inventory_waits_then_changed_slot_cannot_consume(self):
        w=setup();plan=intent(w);p=ResourcePolicy();log=MemoryLog()
        w.sampled['PLAYER_INVENTORY']=w.frame-7
        self.assertIsNone(resource_preference(w,plan,p,log))
        self.assertEqual(plan['resource_action'],'item')
        self.assertFalse(log.rows)
        w.inventory['active_items']['0']['item']=44;refresh(w,1)
        self.assertIsNone(resource_preference(w,plan,p,log))
        self.assertIsNone(p.choose(w))

    def test_already_used_then_recharged_same_item_needs_new_authorization(self):
        w=setup();p=ResourcePolicy();log=MemoryLog();plan=intent(w)
        signature=deepcopy(resource_signature(w.inventory,'item'))
        self.assertEqual(p.choose(w,'item')['id'],105)
        refresh(w,50) # Identical observed charge can recur after recharge.
        self.assertEqual(resource_signature(w.inventory,'item'),signature)
        self.assertIsNone(resource_preference(w,plan,p,log))
        self.assertEqual(log.rows[-1]['reason'],'resource_already_pulsed_since_request')
        new=intent(w)
        self.assertEqual(resource_preference(w,new,p,log),'item')
        # A different floor/run does not borrow an old frame-based rejection.
        w.level='different:1:0';new=intent(w);new['resource_authorization']['source_frame']=0
        self.assertEqual(resource_preference(w,new,p,log),'item')

    def test_missing_binding_cannot_grant_model_only_action_or_suppress_emergency(self):
        w=setup();p=ResourcePolicy();log=MemoryLog()
        plan={'resource_action':'item'}
        self.assertIsNone(resource_preference(w,plan,p,log))
        self.assertIsNone(p.choose(w))
        w.inventory['active_items']['0'].update(item=45,charge=4,max_charge=4)
        w.health['red_hearts']=2
        self.assertEqual(p.choose(w)['id'],45)
