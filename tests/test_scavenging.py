import unittest

from test_agent import world, MemoryLog
from isaac_agent.control import candidates, blocked
from isaac_agent.knowledge import room_knowledge
from isaac_agent.planning import planning_context, planning_reasons
from isaac_agent.scavenging import targets, economy
from isaac_agent.strategy import offers, StrategyExecutor


def prepared():
    w = world(player=(220, 280))
    w.payload['FIRE_HAZARDS'] = [{'id': 11, 'type': 'FIREPLACE', 'variant': 0,
                                'pos': {'x': 370, 'y': 280}, 'collision_radius': 20}]
    w.sampled['FIRE_HAZARDS'] = w.frame
    w.layout['grid'] = {'12': {'type': 14, 'variant': 0, 'collision': 2, 'x': 420, 'y': 350}}
    return w


class ScavengingTests(unittest.TestCase):
    def test_one_decision_continues_across_targets_without_a_call_per_pile(self):
        w = prepared(); log = MemoryLog(); executor = StrategyExecutor()
        plan = {'strategy_enabled': True, 'strategy_choice': 'scavenge', 'mode': 'collect'}
        before = planning_context(w)
        self.assertEqual(sum(o['kind']=='scavenge' for o in offers(w)), 1)
        execution = executor.update(w, plan, log)
        self.assertEqual(execution['shoot_target']['key'], 'fire:11')
        actions, contact = candidates(w, execution)
        self.assertIsNone(contact)
        self.assertTrue(any(a['action'].shoot == (1, 0) for a in actions))
        self.assertTrue(blocked(w, (370, 280)))
        w.payload['FIRE_HAZARDS'][0]['is_extinguished'] = True
        execution = executor.update(w, plan, log)
        self.assertEqual(execution['shoot_target']['key'], 'poop:12')
        self.assertEqual(planning_reasons(before, planning_context(w)), [])
        self.assertNotIn('scavenge', executor.finished)
        w.layout['grid']['12']['collision'] = 0
        execution = executor.update(w, plan, log)
        self.assertIn('scavenge', executor.finished)
        self.assertNotIn('shoot_target', execution)

    def test_shot_requires_authorization_and_rechecks_current_source(self):
        w = prepared()
        actions, _ = candidates(w, {'strategy_enabled': True, 'mode': 'explore'})
        self.assertTrue(all(a['action'].shoot == (0, 0) for a in actions))
        source = targets(w)[0]
        w.payload['FIRE_HAZARDS'][0]['is_extinguished'] = True
        actions, _ = candidates(w, {'strategy_enabled': True, 'shoot_target': source})
        self.assertTrue(all(a['action'].shoot == (0, 0) for a in actions))

    def test_special_fires_and_poop_are_not_misclassified(self):
        w = prepared()
        for variant in (1, 2, 3, 4, 10):
            w.payload['FIRE_HAZARDS'][0]['variant'] = variant
            w.layout['grid']['12']['variant'] = variant
            self.assertEqual(targets(w), [])
        w.payload['FIRE_HAZARDS'][0]['variant'] = 0
        w.frame += 7
        self.assertEqual(targets(w), [])

    def test_sweep_stops_on_damage_or_budget_and_not_during_combat(self):
        for damage in (True, False):
            w = prepared(); executor = StrategyExecutor(); log = MemoryLog()
            plan = {'strategy_enabled': True, 'strategy_choice': 'scavenge'}
            executor.update(w, plan, log)
            if damage:
                w.last_damage_frame = w.frame
            else:
                w.frame += 601
                w.sampled.update({k:w.frame for k in w.sampled})
            execution = executor.update(w, plan, log)
            self.assertNotIn('shoot_target', execution)
            self.assertIn('scavenge', executor.finished)
        w = prepared(); w.info['is_clear'] = False
        w.payload['ENEMIES'] = [{'id': 9, 'pos': {'x': 500, 'y': 250}}]
        self.assertEqual(targets(w), [])

    def test_observed_shop_shortfall_and_bounded_context(self):
        w = prepared()
        w.rooms['4'] = {'type': 2, 'collectibles': [{'sub_type': 307, 'price': 15}]}
        w.inventory['coins'] = 13
        self.assertEqual(economy(w)['observed_shop_items'][0]['coins_short'], 2)
        w.inventory['coins'] = 16
        self.assertEqual(economy(w)['observed_shop_items'][0]['coins_short'], 0)
        self.assertEqual(room_knowledge(w, budget=10), [])
        knowledge = room_knowledge(w)
        self.assertIn('fire', {r['topic'] for r in knowledge})
        self.assertIn('poop', {r['topic'] for r in knowledge})
        self.assertTrue(all(r['source_url'].startswith('https://bindingofisaacrebirth.wiki.gg/') for r in knowledge))
