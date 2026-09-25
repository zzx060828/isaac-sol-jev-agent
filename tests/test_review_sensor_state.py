import asyncio
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent import review
from isaac_agent.diagnostics import pickup_contact_facts
from isaac_agent.runtime import Controller
from test_review_continuity import trinket_room
from test_planning import MemoryLog
from test_safety_strategy import refresh


def prepared():
    w=trinket_room();w.info['room_type']=5
    w.layout['grid']={'37':dict(type=17,variant=0,state=1,grid_index=37,
                              collision=0,x=320,y=200)}
    w.inventory['can_pickup_item']=True
    return w


class ReviewSensorTests(unittest.IsolatedAsyncioTestCase):
    def test_pickup_eligibility_does_not_change_value_review_but_remains_observed(self):
        w=prepared();key=review.fingerprint(w)
        for value in (False,True,None):
            if value is None:w.inventory.pop('can_pickup_item')
            else:w.inventory['can_pickup_item']=value
            self.assertEqual(review.fingerprint(w),key)
            self.assertIs(pickup_contact_facts(w,w.pickups[0])['can_pickup_item_reported'],value)
            self.assertIs(w.observation()['inventory'].get('can_pickup_item'),value)
        for changes in (dict(coins=10),dict(golden_key=True),dict(golden_bomb=True),
                        dict(poop_explosions=True),dict(trinket_0=90),
                        dict(active_items={'0':dict(item=45,charge=4,max_charge=4)})):
            v=deepcopy(w);v.inventory.update(changes)
            self.assertNotEqual(review.fingerprint(v),key)

    async def test_pickup_eligibility_toggle_does_not_repeat_floor_planning(self):
        w=prepared();choice=review.choice_id(w.pickups[0])
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=(
            dict(objective='Leave trinket and explore before descending',mode='explore',
                 door_slot='2',declined_pickups=[choice]),{})))
        c=Controller(models,MemoryLog())
        try:
            c.step(w);await c.plan_task
            # Applying the model's own decline is not a new external decision.
            c.last_plan=-100;c.step(w)
            if c.plan_task and not c.plan_task.done():await c.plan_task
            requests=models.plan.await_count
            self.assertEqual(requests,1)
            for value in (False,True,False):
                w.inventory['can_pickup_item']=value;refresh(w,1);c.last_plan=-100
                c.step(w);await asyncio.sleep(0)
                self.assertIsNotNone(c.pickup_review)
                self.assertIn(choice,c.execution_plan['excluded_choices'])
                self.assertEqual(models.plan.await_count,requests)
            self.assertFalse(any(r['event']=='pickup_review_invalidated' for r in c.log.rows))
            w.inventory['coins']=10;refresh(w,1);c.last_plan=-100
            c.step(w);await c.plan_task
            self.assertGreater(models.plan.await_count,requests)
        finally:await c.close()

    async def test_consuming_declines_preserves_new_affordability_during_reply(self):
        w=prepared();choice=review.choice_id(w.pickups[0]);w.inventory['coins']=0
        w.payload['PICKUPS'].append(dict(id=8,variant=30,sub_type=1,price=5,pos=dict(x=480,y=320)))
        async def reply(_):
            w.inventory['coins']=5
            return dict(objective='Keep current trinket and explore',mode='explore',door_slot='2',
                        declined_pickups=[choice],decline_reconsider_on={choice:['health']}),{}
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(side_effect=reply))
        c=Controller(models,MemoryLog())
        try:
            c.step(w);await c.plan_task
            self.assertEqual(review.active(w),{choice})
            self.assertNotIn(choice,c.plan_context['choices'])
            self.assertNotIn('pickup:8:30:1:5',c.plan_context['choices'])
            refresh(w,1);c.last_plan=-100;c.step(w);await c.plan_task
            self.assertEqual(models.plan.await_count,2)
            latest=[r for r in c.log.rows if r['event']=='plan_request'][-1]
            self.assertIn('resource_choices_changed',latest['triggers'])
            self.assertIn('pickup:8:30:1:5',{o['choice'] for o in latest['observation']['strategy_offers']})
        finally:await c.close()

    async def test_response_keeps_review_when_only_pickup_eligibility_changes(self):
        w=prepared();choice=review.choice_id(w.pickups[0]);obs=w.observation()
        obs.update(pickup_review_context=review.fingerprint(w),pickup_review_basis=review.basis(w))
        async def reply(_):
            w.inventory['can_pickup_item']=False
            return dict(objective='Keep current trinket',mode='explore',declined_pickups=[choice]),{}
        c=Controller(SimpleNamespace(mode='hybrid',plan=reply),MemoryLog());c.reset_actions(w)
        try:
            await c._plan(w,obs,w.token,w.revision)
            self.assertIsNotNone(c.pickup_review)
            self.assertEqual(c.pickup_review['choices'],{choice})
            self.assertEqual(review.active(w),{choice})
        finally:await c.close()
