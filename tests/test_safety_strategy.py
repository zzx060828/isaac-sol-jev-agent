import time
import unittest

from test_agent import world, MemoryLog
from isaac_agent.control import Action, blocked, candidates, next_door, risk, shield, target_point
from isaac_agent.hazards import safe_shot
from isaac_agent.models import validate_plan
from isaac_agent.resources import ResourcePolicy, resource_options, safe_to_experiment
from isaac_agent.strategy import StrategyExecutor, offers


def prepared():
    w = world()
    w.payload.update(BOMBS=[], FIRE_HAZARDS=[], INTERACTABLES=[])
    w.inventory.update(pill_0=3, pill_identified=False, can_use=True, coins=5,
                       active_items={"0": {"item": 45, "charge": 4, "max_charge": 4, "battery_charge": 0}})
    refresh(w)
    return w


def refresh(w, frames=0):
    w.frame += frames
    w.updated = time.monotonic()
    w.sampled.update({name: w.frame for name in w.payload})


def tnt(x, y):
    return {"x": x, "y": y, "type": 12, "collision": 2, "state": 0}


class HazardTests(unittest.TestCase):
    def test_legacy_ashes_leave_scavenge_and_hazard_sets(self):
        from isaac_agent.scavenging import targets
        from isaac_agent.state import World
        from isaac_agent.demo import frame_message
        msg = frame_message()
        msg.setdefault('agent', {})['repentance_plus'] = False
        msg['payload']['ENEMIES'] = []
        msg['payload']['ROOM_INFO']['is_clear'] = True
        fire = dict(id=1, type='FIREPLACE', variant=0, hp=1, max_hp=5,
                    is_extinguished=False, pos=dict(x=320,y=280))
        msg['payload']['FIRE_HAZARDS'] = [fire]
        w = World(); w.ingest(msg)
        self.assertTrue(w.payload['FIRE_HAZARDS'][0]['is_extinguished'])
        self.assertFalse(targets(w))
        self.assertFalse(fire['is_extinguished'])  # Preserve the raw evidence.
        for changes in ({'hp': 2}, {'state': 4}, {'variant': 4}, {'max_hp': 10}):
            msg['payload']['FIRE_HAZARDS'] = [dict(fire, **changes)]
            w = World(); w.ingest(msg)
            self.assertFalse(w.payload['FIRE_HAZARDS'][0]['is_extinguished'])
        msg['payload']['FIRE_HAZARDS'] = [fire]
        msg['agent']['repentance_plus'] = True
        w = World(); w.ingest(msg)
        self.assertFalse(w.payload['FIRE_HAZARDS'][0]['is_extinguished'])

    def test_trinket_offer_and_held_effects_are_explicit_under_blind(self):
        from unittest.mock import patch
        from isaac_agent.knowledge import held_trinket_mechanics
        w = prepared()
        w.payload['ENEMIES'] = []
        w.info.update(is_clear=True, curses=64)
        w.inventory['trinket_0'] = 95
        w.payload['PICKUPS'] = [dict(id=8, variant=350, sub_type=63, price=0, pos=dict(x=320,y=280))]
        desc = {63: {'name': 'Safety Scissors', 'description': 'Converts troll bombs'},
                95: {'name': 'Black Tooth', 'description': 'Poison teeth'}}
        with patch('isaac_agent.strategy.item_descriptions', return_value=desc), patch('isaac_agent.knowledge.item_descriptions', return_value=desc):
            offer = next(o for o in offers(w) if o['kind'] == 'pickup')
            self.assertEqual(offer['entity']['pickup_type'], 'trinket')
            self.assertEqual(offer['entity']['name'], 'Safety Scissors')
            self.assertIn('replace a held trinket', offer['effect'])
            self.assertEqual(held_trinket_mechanics(w.inventory)[0]['name'], 'Black Tooth')
        before = w.strategy_signature()
        w.inventory['trinket_0'] = 63
        self.assertNotEqual(before, w.strategy_signature())

    def test_sol_combat_distance_is_bounded_and_changes_positioning(self):
        w = world(player=(320, 280))
        w.stats['range'] = 260
        w.payload['ENEMIES'] = [{'id': 2, 'pos': {'x': 500, 'y': 280}}]
        near = validate_plan({'objective': 'fight', 'target_enemy': 2, 'combat_distance': 100}, w.observation())
        far = validate_plan({'objective': 'keep away', 'target_enemy': 2, 'combat_distance': 1000}, w.observation())
        self.assertLessEqual(far['combat_distance'], w.stats['range'] * .88)
        self.assertLessEqual(candidates(w, near)[0][0]['action'].move[0], 0)
        self.assertEqual(candidates(w, far)[0][0]['action'].move[0], -1)
        with self.assertRaises(ValueError):
            validate_plan({'objective': 'invalid', 'combat_distance': float('nan')}, w.observation())

    def test_leave_turning_space_before_passing_a_nearby_enemy(self):
        w = world(player=(320, 280))
        w.payload['ENEMIES'] = [{'id': 2, 'pos': {'x': 400, 'y': 280},
                                'vel': {'x': 0, 'y': 3}, 'collision_radius': 13}]
        # A lateral velocity is not evidence that an enemy cannot turn toward
        # us. Do not close the gap merely because linear paths do not intersect.
        options = candidates(w, {})[0]
        self.assertTrue(options)
        self.assertFalse(any(o['action'].move == (1, 0) for o in options))
        self.assertTrue(any(o['action'].move == (-1, 0) for o in options))

    def test_selected_item_surrounded_by_fire_gets_a_clearing_shot(self):
        w = world(player=(80, 280))
        w.stats.update(size=10, speed=.85)
        w.payload['PICKUPS'] = [{'id': 6, 'variant': 100, 'sub_type': 506,
                                'price': 0, 'pos': {'x': 320, 'y': 280}}]
        w.payload['FIRE_HAZARDS'] = [
            {'type': 'FIREPLACE', 'variant': 0, 'collision_radius': 13,
             'pos': {'x': x, 'y': y}} for x, y in
            [(280, 280), (320, 240), (360, 280), (320, 320)]]
        plan = {'strategy_enabled': True, 'navigation_target': (320, 280),
                'navigation_contact': True}
        best = candidates(w, plan)[0][0]['action']
        self.assertEqual(best.shoot, (1, 0))
        self.assertLess(risk(w, best.move, (320, 280)), 5)
        w.payload['FIRE_HAZARDS'] = w.payload['FIRE_HAZARDS'][1:]
        w.path_cache.clear()
        best = candidates(w, plan)[0][0]['action']
        self.assertEqual(best.shoot, (0, 0))
        self.assertEqual(best.move, (1, 0))

    def test_invulnerable_enemy_is_avoided_but_vulnerable_enemy_is_targeted(self):
        w = world(player=(320, 280))
        w.payload['ENEMIES'] = [
            {'id': 1, 'is_vulnerable': False, 'pos': {'x': 350, 'y': 280}, 'collision_radius': 35},
            {'id': 2, 'is_vulnerable': True, 'pos': {'x': 200, 'y': 280}}]
        self.assertEqual(target_point(w, {'target_enemy': 1})[0], (200, 280))
        self.assertGreater(risk(w, (1, 0)), risk(w, (-1, 0)))

    def test_enemy_behind_rock_requires_flanking(self):
        w = world(player=(320, 160))
        w.payload['ENEMIES'] = [{'id': 2, 'type': 226, 'hp': 14,
                                'collision_radius': 13, 'pos': {'x': 320, 'y': 280}}]
        w.layout['grid'] = {
            '1': {'type': 2, 'collision': 3, 'x': 320, 'y': 240},
            '2': {'type': 2, 'collision': 3, 'x': 320, 'y': 320},
            '3': {'type': 14, 'collision': 3, 'x': 280, 'y': 280},
            '4': {'type': 14, 'collision': 3, 'x': 360, 'y': 280}}
        best = candidates(w, {'target_enemy': 2})[0][0]['action']
        self.assertNotEqual(best.move[0], 0)

    def test_releasing_input_does_not_instantly_stop_before_spikes(self):
        w = world(player=(270, 280))
        w.player['vel'] = {'x': 4, 'y': 0}
        w.layout['grid'] = {'1': {'type': 8, 'collision': 0, 'x': 320, 'y': 280}}
        # Existing momentum can carry us into the hazard even with no new input.
        self.assertGreater(risk(w, (0, 0)), 5)
        self.assertLess(risk(w, (-1, 0)), risk(w, (0, 0)))

    def test_clear_room_can_collect_double_bomb_among_unlit_tnt(self):
        # Recorded room 72, frame 5624: the old blast penalty trapped the
        # controller along y=394 despite Sol selecting the pickup at (160,280).
        w = world(player=(299.616, 394.544))
        w.stats.update(speed=.85, size=10)
        w.info.update(top_left={'x': 60, 'y': 140}, bottom_right={'x': 580, 'y': 420})
        w.layout['grid'] = {str(i): tnt(x, y) for i, (x, y) in enumerate(
            [(80, 280), (120, 240), (120, 200), (120, 320),
             (120, 360), (440, 280), (80, 160), (280, 280)])}
        w.payload['PICKUPS'] = [{'id': 2, 'variant': 40, 'sub_type': 2,
                                'price': 0, 'pos': {'x': 160, 'y': 280}}]
        plan = {'strategy_enabled': True, 'navigation_target': (160, 280),
                'navigation_contact': True}
        best = candidates(w, plan)[0][0]['action']
        self.assertEqual(best.move, (-1, -1))
        # Released fire must still leave a danger zone for tears already in flight.
        w.payload['PROJECTILES']['player_tears'] = [
            {'pos': {'x': 280, 'y': 380}, 'vel': {'x': 0, 'y': -10}}]
        self.assertGreater(risk(w, (-1, -1)), 5)

    def test_nearby_tnt_blocks_firing_but_not_safe_direction(self):
        w = prepared()
        w.layout['grid'] = {'1': tnt(400, 280)}
        self.assertFalse(safe_shot(w, (0, 0), (1, 0)))
        self.assertTrue(safe_shot(w, (0, 0), (-1, 0)))
        safe, overridden = shield(w, Action((0, 0), (1, 0)), Action(), None)
        self.assertTrue(overridden)
        self.assertEqual(safe.shoot, (0, 0))

    def test_distant_barrel_can_be_shot_but_chain_to_player_cannot(self):
        w = world(player=(200, 280))
        w.layout['grid'] = {'1': tnt(400, 280)}
        self.assertTrue(safe_shot(w, (0, 0), (1, 0)))
        w.layout['grid']['2'] = tnt(300, 380)
        self.assertFalse(safe_shot(w, (0, 0), (1, 0)))

    def test_live_bomb_repels_using_blast_radius(self):
        w = prepared()
        w.payload['BOMBS'] = [{'pos': {'x': 400, 'y': 280}, 'explosion_radius': 150}]
        self.assertGreater(risk(w, (1, 0)), risk(w, (-1, 0)))

    def test_cycling_spikes_and_creep_avoidance_respect_flight(self):
        w = prepared()
        w.layout['grid'] = {'1': {'x': 320, 'y': 280, 'type': 9, 'collision': 0, 'state': 0}}
        self.assertTrue(blocked(w, (320, 280)))
        w.stats['can_fly'] = True
        self.assertFalse(blocked(w, (320, 280)))
        w.payload['FIRE_HAZARDS'] = [{'pos': {'x': 320, 'y': 280}, 'ground_only': True, 'collision_radius': 30}]
        self.assertFalse(blocked(w, (320, 280)))
        w.stats['can_fly'] = False
        self.assertTrue(blocked(w, (320, 280)))

    def test_curse_door_not_used_as_normal_exploration(self):
        w = prepared()
        w.layout['doors']['2']['target_room_type'] = 10
        w.rooms[str(w.room)]['doors']['2']['target_room_type'] = 10
        self.assertIsNone(next_door(w, '2'))


class ResourceTests(unittest.TestCase):
    def setUp(self):
        from synthetic_eid import enable
        enable(self)

    def test_yum_heart_cannot_spend_charge_at_full_health_even_if_model_requests_it(self):
        w = prepared()
        w.inventory['pill_0'] = 0
        w.health.update(red_hearts=8, max_hearts=8)
        self.assertFalse(any(o['kind'] == 'item' for o in resource_options(w)))
        self.assertIsNone(ResourcePolicy().choose(w, 'item'))
        w.health['red_hearts'] = 6
        self.assertEqual(ResourcePolicy().choose(w, 'item')['id'], 45)

    def test_heals_before_critical_health_and_bounds_retries(self):
        w = prepared()
        w.health['red_hearts'] = 4
        policy = ResourcePolicy()
        self.assertEqual(policy.choose(w)['kind'], 'item')
        refresh(w, 1)
        self.assertIsNone(policy.choose(w))
        refresh(w, 45)
        self.assertEqual(policy.choose(w)['kind'], 'item')
        refresh(w, 60)
        self.assertIsNone(policy.choose(w))
        w.inventory['active_items']['0']['charge'] = 0
        self.assertIsNone(policy.choose(w))
        w.inventory['active_items']['0']['charge'] = 4
        self.assertEqual(policy.choose(w)['kind'], 'item')

    def test_unknown_pill_requires_health_and_clear_escape_space(self):
        w = prepared()
        self.assertTrue(safe_to_experiment(w))
        self.assertTrue(any(c['kind'] == 'pill' for c in resource_options(w)))
        w.health['red_hearts'] = 4
        self.assertFalse(safe_to_experiment(w))
        w.health['red_hearts'] = 6
        w.payload['BOMBS'] = [{'pos': {'x': 500, 'y': 300}}]
        self.assertFalse(safe_to_experiment(w))
        w.payload['BOMBS'] = []
        w.payload['ENEMIES'] = [{'pos': {'x': 500, 'y': 300}}]
        self.assertFalse(safe_to_experiment(w))

    def test_known_bad_pill_not_mistaken_for_unknown(self):
        w = prepared()
        w.inventory.update(pill_identified=True, pill_effect=4)
        option=next(c for c in resource_options(w) if c['kind']=='pill')
        self.assertTrue(option['model_only'])
        self.assertEqual(option['mechanics']['effect_id'],4)
        self.assertIsNone(ResourcePolicy().choose(w))

    def test_damage_requires_updated_health_and_inventory(self):
        w = prepared()
        w.health['red_hearts'] = 4
        w.last_damage_frame = w.frame
        self.assertEqual(resource_options(w), [])
        refresh(w, 1)
        self.assertEqual(resource_options(w)[0]['kind'], 'item')

    def test_generic_active_requires_explicit_model_choice(self):
        w = prepared()
        w.inventory.update(pill_0=0)
        w.inventory['active_items']['0']['item'] = 105  # D6 is not an emergency reflex.
        p = ResourcePolicy()
        self.assertIsNone(p.choose(w))
        self.assertEqual(p.choose(w, 'item')['id'], 105)


class StrategyTests(unittest.TestCase):
    def test_model_choice_controls_which_item_is_approached(self):
        w = prepared()
        w.payload['PICKUPS'] = [
            {'id': 10, 'variant': 100, 'sub_type': 1, 'price': 0, 'pos': {'x': 340, 'y': 280}},
            {'id': 11, 'variant': 100, 'sub_type': 2, 'price': 0, 'pos': {'x': 440, 'y': 280}}]
        choice = offers(w)[1]['choice']
        plan = validate_plan({'objective': 'Choose second item', 'mode': 'collect', 'strategy_choice': choice}, w.observation())
        plan['strategy_enabled'] = True
        execution = StrategyExecutor().update(w, plan, MemoryLog())
        self.assertEqual(target_point(w, execution)[0], (440, 280))
        # A deliberate skip must not revert to nearest-item collection.
        self.assertTrue(target_point(w, {'strategy_enabled': True, 'mode': 'explore', 'door_slot': '2'})[1])

    def test_shop_choice_invalidates_when_cost_is_unaffordable(self):
        w = prepared()
        w.payload['PICKUPS'] = [{'id': 1, 'variant': 100, 'sub_type': 1, 'price': 5, 'pos': {'x': 440, 'y': 280}}]
        choice = offers(w)[0]['choice']
        w.inventory['coins'] = 4
        with self.assertRaises(ValueError):
            validate_plan({'objective': 'buy', 'strategy_choice': choice}, w.observation())

    def test_donation_stops_after_first_damage_and_retracts_contact_permission(self):
        w = prepared()
        w.health.update(red_hearts=8, max_hearts=8)
        w.payload['INTERACTABLES'] = [{'id': 8, 'variant': 2, 'pos': {'x': 350, 'y': 280}}]
        choice = offers(w)[0]['choice']
        plan = {'strategy_enabled': True, 'strategy_choice': choice, 'mode': 'collect'}
        e, log = StrategyExecutor(), MemoryLog()
        first = e.update(w, plan, log)
        self.assertTrue(first['navigation_contact'])
        w.last_damage_frame = w.frame + 1
        refresh(w, 1)
        after = e.update(w, plan, log)
        self.assertFalse(after['navigation_contact'])
        self.assertLess(after['navigation_target'][0], 320)
        self.assertIn(choice, e.finished)

    def test_machines_require_selected_contact_and_low_hp_trade_not_offered(self):
        w = prepared()
        w.payload['INTERACTABLES'] = [{'id': 8, 'variant': 2, 'pos': {'x': 350, 'y': 280}}]
        self.assertTrue(blocked(w, (350, 280)))
        self.assertFalse(blocked(w, (350, 280), exit_point=(350, 280)))
        w.health['red_hearts'] = 4
        self.assertFalse(any(o['kind'] == 'donate' for o in offers(w)))


if __name__ == '__main__':
    unittest.main()
