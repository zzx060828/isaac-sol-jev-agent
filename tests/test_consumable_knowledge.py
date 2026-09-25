import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from isaac_agent.advance import AdvancePlanner
from isaac_agent.fast_policy import compact_goal_observation
from isaac_agent.knowledge import held_consumable_mechanics,item_descriptions
from isaac_agent.models import validate_plan
from isaac_agent.resources import resource_options
from isaac_agent.strategy import describe_pickup
from test_advance import setup,preparation
from test_agent import MemoryLog
from test_safety_strategy import prepared,refresh


class ConsumableKnowledgeTests(unittest.TestCase):
    def test_golden_pill_cannot_claim_a_beneficial_effect_at_low_health(self):
        for color in (14,14|0x800):
            for effect in (2,5,7):
                w=prepared();w.inventory.update(pill_0=color,pill_identified=True,pill_effect=effect,active_items={})
                w.health.update(red_hearts=2,max_hearts=8,soul_hearts=0)
                self.assertFalse(any(o['kind']=='pill' for o in resource_options(w)))
                w.health['red_hearts']=6
                option=next(o for o in resource_options(w) if o['kind']=='pill')
                self.assertEqual(option['priority'],30)
                self.assertIn('random golden',option['reason'])
                w.payload['BOMBS']=[{'pos':{'x':480,'y':280}}]
                refresh(w)
                self.assertFalse(any(o['kind']=='pill' for o in resource_options(w)))

    def test_identified_ordinary_healing_still_works_at_low_health(self):
        w=prepared();w.inventory.update(pill_0=3,pill_identified=True,pill_effect=5,active_items={})
        w.health.update(red_hearts=2,max_hearts=8,soul_hearts=0)
        self.assertEqual(next(o for o in resource_options(w) if o['kind']=='pill')['priority'],99)

    def test_eid_tables_are_separate_and_rep_overrides_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for edition in ('ab+','rep'):(root/'descriptions'/edition).mkdir(parents=True)
            (root/'descriptions/ab+/en_us.lua').write_text('''
EID.descriptions[languageCode].cards={
{"20", "Sun", "Full health#Room damage"},
}
EID.descriptions[languageCode].pills={
{"0", "Bad Gas", "Poison nearby"},
{"5", "Full Health", "Heal red health"},
}
''')
            (root/'descriptions/rep/en_us.lua').write_text('''
local repCards={
[20] = {"20", "Sun rep", "{{HealingRed}} Heal"},
}
local repPills={
[1] = {"0", "Bad Gas rep", "Poison nearby rep"},
}
EID.descriptions[languageCode].horsepills={
{"5", "Full Health", "Heal red health#Gain 3 Soul Hearts"},
}
''')
            item_descriptions.cache_clear()
            try:
                with patch.dict('os.environ',{'ISAAC_EID_DIR':tmp}):
                    self.assertEqual(item_descriptions('card')[20]['name'],'Sun rep')
                    self.assertEqual(item_descriptions('pill')[0]['name'],'Bad Gas rep')
                    self.assertNotIn('Soul Hearts',item_descriptions('pill')[5]['description'])
                    self.assertIn('Soul Hearts',item_descriptions('horse_pill')[5]['description'])
                    self.assertNotIn(20,item_descriptions('pill'))
            finally:item_descriptions.cache_clear()

    def test_unidentified_pill_does_not_look_up_effect_even_if_field_present(self):
        for effect in (5,20,41):
            with patch('isaac_agent.knowledge.item_descriptions',side_effect=AssertionError('effect leak')):
                rows=held_consumable_mechanics(dict(pill_0=3,pill_identified=False,pill_effect=effect))
            self.assertNotIn('effect_id',rows[0])
            self.assertNotIn('description',rows[0])
            self.assertFalse(rows[0]['effect_known'])

    def test_horse_and_golden_forms_do_not_borrow_normal_effect(self):
        def lookup(kind):
            return {5:dict(name='Full Health',description=kind,description_source='test')}
        with patch('isaac_agent.knowledge.item_descriptions',side_effect=lookup):
            normal=held_consumable_mechanics(dict(pill_0=3,pill_identified=True,pill_effect=5))[0]
            horse=held_consumable_mechanics(dict(pill_0=3|0x800,pill_identified=True,pill_effect=5))[0]
            golden=held_consumable_mechanics(dict(pill_0=14|0x800,pill_identified=True,pill_effect=5))[0]
        self.assertEqual(normal['description'],'pill')
        self.assertEqual(horse['description'],'horse_pill')
        self.assertEqual(golden['name'],'Golden Pill')
        self.assertNotIn('effect_id',golden)
        self.assertFalse(golden['effect_known'])

    def test_missing_descriptions_and_hidden_ground_cards_remain_unknown(self):
        with patch('isaac_agent.knowledge.item_descriptions',return_value={}):
            rows=held_consumable_mechanics(dict(card_0=999,pill_0=3,pill_identified=True,pill_effect=999))
        self.assertEqual(rows[0]['name'],'Unknown card or rune')
        self.assertEqual(rows[1]['name'],'Unknown identified pill effect')
        with patch('isaac_agent.strategy.item_descriptions',side_effect=AssertionError('hidden lookup')):
            card=describe_pickup(dict(variant=300,sub_type=20,is_hidden=True))
            pill=describe_pickup(dict(variant=70,sub_type=5))
        self.assertEqual(card['name'],'Hidden card or rune')
        self.assertNotIn('description',pill)

    def test_observation_and_jev_get_held_mechanics_with_model_use_option(self):
        w=setup();w.inventory.update(card_0=5,pill_0=3,pill_identified=True,pill_effect=5)
        detail=dict(name='Emperor',description='Teleport to boss')
        with patch('isaac_agent.knowledge.item_descriptions',return_value={5:detail}):
            obs=w.observation()
        self.assertEqual({r['slot'] for r in obs['held_consumable_mechanics']},{'card_0','pill_0'})
        self.assertEqual(compact_goal_observation(obs,[])['held_consumable_mechanics'],obs['held_consumable_mechanics'])
        option=next(o for o in obs['resource_options'] if o['kind']=='card')
        self.assertTrue(option['model_only'])
        self.assertEqual(validate_plan(dict(objective='Teleport',resource_action='card'),obs)['resource_action'],'card')
        with patch('isaac_agent.strategy.item_descriptions',return_value={5:detail}):
            ground=describe_pickup(dict(variant=300,sub_type=5))
        self.assertEqual(ground['name'],'Emperor')


class AdvanceConsumableTests(unittest.IsolatedAsyncioTestCase):
    async def test_boss_preparation_receives_held_pill_effect(self):
        w=setup();w.inventory.update(card_0=0,pill_0=3,pill_identified=True,pill_effect=41)
        models=SimpleNamespace(mode='hybrid',prepare=AsyncMock(return_value=(preparation(),{})))
        planner=AdvancePlanner(models,MemoryLog())
        try:
            with patch('isaac_agent.knowledge.item_descriptions',return_value={41:dict(name='Drowsy',description='Slow for room')}):
                planner.maybe_start(w,'tactical',False)
                await planner.task
            obs=models.prepare.call_args.args[0]
            self.assertEqual(obs['held_consumable_mechanics'][0]['effect_id'],41)
        finally:await planner.close()
