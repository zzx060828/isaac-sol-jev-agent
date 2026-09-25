"""Known descriptions enable model choices, without enabling blind reflex use."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.knowledge import item_descriptions
from isaac_agent.resources import resource_options, ResourcePolicy, bind_resource_intent, resource_preference
from isaac_agent.runtime import Controller
from test_review_continuity import setup
from test_safety_strategy import refresh
from test_planning import MemoryLog


def prepared():
    w=setup();w.inventory.update(can_use=True,coins=0,card_0=0,pill_0=0,
        pill_identified=False,pill_effect=None,collectibles={})
    return w


class ConsumableDecisionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from synthetic_eid import enable
        enable(self)

    def test_all_described_cards_and_identified_pill_forms_have_a_use_option(self):
        for card in item_descriptions('card'):
            w=prepared();w.inventory['card_0']=card
            self.assertIn('card',{o['kind'] for o in resource_options(w)},card)
        for kind,color in (('pill',1),('horse_pill',0x801)):
            for effect in item_descriptions(kind):
                w=prepared();w.inventory.update(pill_0=color,pill_identified=True,pill_effect=effect)
                opts=resource_options(w)
                self.assertIn('pill',{o['kind'] for o in opts},(kind,effect))
                selected=next(o for o in opts if o['kind']=='pill')
                if selected.get('mechanics'):
                    self.assertEqual(selected['mechanics']['effect_id'],effect)
                    self.assertEqual(selected['mechanics']['form'],'horse' if kind=='horse_pill' else 'normal')

    def test_generic_use_requires_model_and_current_slot_identity(self):
        for kind,values in (('card',dict(card_0=9)),
                            ('pill',dict(pill_0=1,pill_identified=True,pill_effect=0))):
            w=prepared();w.inventory.update(values)
            option=next(o for o in resource_options(w) if o['kind']==kind)
            self.assertTrue(option['model_only']);self.assertTrue(option['mechanics']['description'])
            policy=ResourcePolicy();self.assertIsNone(policy.choose(w))
            plan={'resource_action':kind};bind_resource_intent(plan,w.observation())
            pref=resource_preference(w,plan,policy,MemoryLog())
            self.assertEqual(policy.choose(w,pref)['kind'],kind)
            self.assertIsNone(resource_preference(w,plan,policy,MemoryLog()))
            self.assertIsNone(policy.choose(w))

    def test_unknown_random_and_stale_inputs_do_not_gain_generic_permission(self):
        for values in (dict(pill_0=1,pill_identified=False,pill_effect=0),
                       dict(pill_0=14,pill_identified=True,pill_effect=0)):
            w=prepared();w.inventory.update(values);w.health['red_hearts']=2
            self.assertNotIn('pill',{o['kind'] for o in resource_options(w)})
        w=prepared();w.inventory['card_0']=99999
        self.assertNotIn('card',{o['kind'] for o in resource_options(w)})
        w=prepared();w.inventory['card_0']=9;w.sampled['BOMBS']=w.frame-7
        self.assertNotIn('card',{o['kind'] for o in resource_options(w)})
        w=prepared();w.inventory['card_0']=9;w.inventory['can_use']=False
        self.assertFalse(resource_options(w))

    async def test_keep_is_reviewed_once_across_rooms_and_changed_slot_reopens_review(self):
        w=prepared();w.inventory['card_0']=9
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=(
            dict(objective='Keep card for later',mode='explore',resource_action=None),{})))
        c=Controller(models,MemoryLog())
        try:
            c.step(w);await c.plan_task
            self.assertEqual(models.plan.await_count,1)
            for room in (2,3):
                w.room=room;refresh(w,1);c.last_plan=-100;c.step(w);await asyncio.sleep(0)
                self.assertEqual(models.plan.await_count,1)
            w.inventory['card_0']=10;refresh(w,1);c.last_plan=-100
            c.step(w);await c.plan_task
            self.assertEqual(models.plan.await_count,2)
        finally:await c.close()

    async def test_new_consumable_review_is_not_overridden_by_existing_journey(self):
        w=prepared();w.inventory['card_0']=9
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=(
            dict(objective='Evaluate supplies before travel',mode='explore'),{})))
        c=Controller(models,MemoryLog());c.reset_actions(w)
        c.journey.start(w,dict(destination_room=2,path=[dict(
            **{'from':1,'to':2},slot='2',type=1,locked=False,key_cost=0)]))
        try:
            c.step(w);await c.plan_task
            self.assertEqual(c.route,'sol')
            self.assertEqual(models.plan.await_count,1)
        finally:await c.close()

    async def test_model_decision_reaches_exactly_one_card_pulse(self):
        w=prepared();w.inventory['card_0']=9
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=(
            dict(objective='Use Justice to generate supplies',mode='collect',resource_action='card'),{})))
        c=Controller(models,MemoryLog())
        try:
            c.step(w);await c.plan_task;refresh(w,1)
            command=c.step(w)
            self.assertTrue(command['use_card']);self.assertEqual(command['agent_resource_id'],9)
            self.assertFalse(c.step(w)['use_card'])
            self.assertIsNone(c.plan['resource_action'])
        finally:await c.close()
