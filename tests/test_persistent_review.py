from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent import review
from isaac_agent.memory import RunMemory
from isaac_agent.models import validate_plan
from isaac_agent.journey import options
from isaac_agent.runtime import Controller
from isaac_agent.strategy import offers
from test_review_continuity import setup
from test_planning import MemoryLog
from test_safety_strategy import refresh


def prepared(variant=100, subtype=378, price=0, options_index=0):
    w=setup();w.level='test-seed:3:0';w.info.update(curses=0,room_type=1)
    w.payload['PICKUPS']=[dict(id=7,variant=variant,sub_type=subtype,price=price,
                             options_index=options_index,item_type=1,pos=dict(x=400,y=280))]
    w.inventory.update(coins=10,bombs=4,keys=1,collectibles={'45':1},pill_0=10,pill_identified=True,
                       active_items={'0':dict(item=45,charge=2,max_charge=4)})
    return w


def remember(w,deps):
    choice=review.choice_id(w.pickups[0])
    obs={'pickup_review_basis':review.basis(w)}
    plan={'declined_pickups':[choice],'decline_reconsider_on':{choice:deps},'rationale':'Keep current build'}
    review.remember(w,obs,plan,MemoryLog())
    return choice


class PersistentReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_scavenge_end_waits_for_delayed_coin_before_paid_review(self):
        from test_review_continuity import trinket_room
        w=trinket_room()
        w.inventory['coins']=10
        w.payload['PICKUPS']=[dict(id=7,variant=30,sub_type=1,price=3,pos=dict(x=450,y=280))]
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=(
            {'objective':'Sweep first','mode':'collect','strategy_choice':'scavenge'},{})))
        c=Controller(m,MemoryLog())
        try:
            c.step(w);await c.plan_task;c.step(w)
            w.layout['grid']['10']['collision']=0;refresh(w,10);c.step(w)
            self.assertEqual(c.route,'settle_transaction')
            self.assertEqual(m.plan.await_count,1)
            w.inventory['coins']=11;refresh(w,6);c.last_plan=-100;c.step(w)
            await c.plan_task
            self.assertEqual(m.plan.call_args.args[0]['inventory']['coins'],11)
        finally:await c.close()

    def test_declared_dependencies_survive_unrelated_collection_and_charge(self):
        w=prepared(70,3);choice=remember(w,['health','consumables'])
        w.inventory.update(coins=20,bombs=5,keys=3)
        w.inventory['active_items']['0']['charge']=4
        refresh(w,10);review.refresh(w,MemoryLog())
        self.assertEqual(review.active(w),{choice})
        w.inventory['pill_0']=11
        self.assertFalse(review.active(w))
        review.refresh(w,MemoryLog());self.assertFalse(review.pool(w))

    def test_cost_options_and_identity_force_reconsideration(self):
        for mode in ('coin','health','options','identity','build'):
            w=prepared(price=5 if mode=='coin' else -1 if mode=='health' else 0,
                       options_index=1 if mode=='options' else 0)
            remember(w,[])
            if mode=='coin':w.inventory['coins']=20
            elif mode=='health':w.health['red_hearts']=2
            elif mode=='options':w.payload['PICKUPS'].append(dict(id=8,variant=100,sub_type=531))
            elif mode=='identity':w.pickups[0]['is_hidden']=True
            else:w.inventory['collectibles']['531']=1
            self.assertFalse(review.active(w),mode)

    def test_incomplete_entry_sensors_do_not_destroy_persistent_review(self):
        w=prepared();remember(w,['health'])
        original=deepcopy(w.payload['PICKUPS'])
        w.payload['PICKUPS']=[];w.sampled['PICKUPS']=w.frame-20
        review.refresh(w,MemoryLog());self.assertTrue(review.pool(w))
        w.payload['PICKUPS']=original;refresh(w)
        self.assertTrue(review.active(w))

    async def test_restart_and_return_route_skip_reviewed_reward_until_health_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'memory.json'
            w=prepared();w.run_memory=RunMemory(path)
            w.run_memory.observe(w,{'type':'DATA'})
            choice=remember(w,['health','bombs'])
            # Simulate actual disk reload and rejoin of this same running seed.
            w.run_memory=RunMemory(path);refresh(w,10)
            w.run_memory.observe(w,{'type':'DATA'})
            self.assertEqual(review.active(w),{choice})
            c=Controller(SimpleNamespace(mode='hybrid',plan=AsyncMock()),MemoryLog())
            try:
                c.step(w)
                self.assertIn(choice,c.execution_plan['excluded_choices'])
                self.assertNotEqual(c.route,'sol')
                c.models.plan.assert_not_called()
            finally:await c.close()
            # Preserve the observed reward while evaluating a route from another room.
            target=w.room
            w.rooms[str(target)]=dict(type=1,clear=True,curses=0,collectibles=deepcopy(w.pickups),resources=[],doors={})
            w.room=99;w.payload['PICKUPS']=[]
            w.layout['doors']={'0':dict(x=40,y=280,target_room=target,target_room_type=1,is_open=True,is_locked=False)}
            w.inventory['coins']=25
            self.assertFalse(any(o['destination_room']==target for o in options(w)))
            w.health['red_hearts']=4
            self.assertTrue(any(o['destination_room']==target for o in options(w)))
            # A fresh seed must never inherit these suppressions.
            w.level='another-seed:1:0';refresh(w,1);w.run_memory.observe(w,{'type':'DATA'})
            self.assertFalse(w.run_memory.pickup_reviews)

    async def test_reply_retains_decline_after_unrelated_parallel_pickup(self):
        for changed in ('coins','health'):
            w=prepared();obs=w.observation();obs['pickup_review_basis']=review.basis(w)
            choice=review.choice_id(w.pickups[0])
            async def plan(_):
                if changed=='coins':w.inventory['coins']=15
                else:w.health['red_hearts']=4
                return {'objective':'Leave item','mode':'explore','declined_pickups':[choice],
                        'decline_reconsider_on':{choice:['health','bombs']}},{}
            c=Controller(SimpleNamespace(mode='hybrid',plan=plan),MemoryLog())
            try:
                await c._plan(w,obs,w.token,w.revision)
                self.assertEqual(bool(review.active(w)),changed=='coins')
            finally:await c.close()

    def test_schema_accepts_only_declined_observed_choices_and_known_conditions(self):
        w=prepared();obs=w.observation();obs['strategy_offers']=offers(w)
        choice=review.choice_id(w.pickups[0])
        base={'objective':'Leave it','declined_pickups':[choice]}
        p=validate_plan({**base,'decline_reconsider_on':{choice:['health','bombs']}},obs)
        self.assertEqual(p['decline_reconsider_on'][choice],['bombs','health'])
        for conditions in ({'invented':['health']},{choice:['magic']},{choice:'health'},[]):
            with self.assertRaises(ValueError):
                validate_plan({**base,'decline_reconsider_on':conditions},obs)
