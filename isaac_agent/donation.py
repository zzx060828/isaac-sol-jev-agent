"""Bounded, model-authorized donation and observed red-heart recovery."""
import math
import hashlib
import json

from .state import records, xy
from .pickups import RED_HEARTS, red_healing


def recovery_hearts(world):
    return [p for p in world.pickups if p.get('variant') == 10
            and p.get('sub_type') in RED_HEARTS and not p.get('price', 0)
            and not p.get('options_index', 0) and p.get('wait', 0) <= 0]


def accessible_recovery_hearts(world):
    """Observed free red hearts with a current walking approach, not owned HP."""
    from .control import blocked, path_step
    from .pickup_progress import blocked_choices
    from .review import choice_id
    if not all(world.fresh(k, 6) for k in ('PICKUPS', 'ROOM_LAYOUT', 'ROOM_INFO',
                                          'FIRE_HAZARDS', 'PLAYER_STATS')):
        return []
    excluded = blocked_choices(world)
    point = xy(world.player.get('pos'))
    return [h for h in recovery_hearts(world) if choice_id(h) not in excluded
            and not blocked(world, xy(h['pos']))
            and (math.dist(point, xy(h['pos'])) < 20
                 or math.dist(path_step(world, xy(h['pos'])), point) > 1)]


def contract_key(contract):
    # Position jitter does not create a new decision; the authorized identities,
    # healing values and payment bounds do. Single contact has a separate key.
    terms = {k: contract[k] for k in ('heart_healing', 'max_contacts',
             'min_red_half_hearts_after', 'max_damage_half_hearts_per_contact')}
    return hashlib.sha256(json.dumps(terms,sort_keys=True,separators=(',', ':')).encode()).hexdigest()


def cycle_contract(world):
    # Character-specific health and healing conversions need separate review.
    if (world.capabilities.get('repentance_plus') is not False
            or world.stats.get('player_type') not in (0, 1)
            or any(world.health.get(k, 0) for k in ('bone_hearts', 'rotten_hearts', 'broken_hearts'))):
        return None
    hearts = accessible_recovery_hearts(world)
    if not hearts:
        return None
    return {'heart_ids': [p['id'] for p in hearts],
            'heart_healing': sorted([{'id': h['id'], 'sub_type': h['sub_type'],
                                      'half_hearts': red_healing(world,h)} for h in hearts],key=lambda h:str(h['id'])), 'max_contacts': min(3, sum(red_healing(world,p) for p in hearts)),
            'min_red_half_hearts_after': 4, 'max_damage_half_hearts_per_contact': 2,
            'recovery_half_hearts_observed': sum(red_healing(world,p) for p in hearts),
            'effect': 'Donate once, withdraw, recover toward starting red health using only these free red hearts with a currently checked walking approach. '
                      'May defer recovery to avoid wasting an oversized heart only while actual red health covers '
                      'another conservative payment. Repeat within max_contacts. Floor hearts are not owned health. '
                      'Stop on missing healing/machine, new danger, health reserve, or timeout. No guaranteed payout.'}


class DonationCycle:
    def __init__(self, world, offer):
        self.offer = offer
        self.token = world.token
        self.started = self.phase_started = world.frame
        self.phase = 'contact'
        self.damage_frame = world.last_damage_frame
        self.contacts = 0
        self.baseline = world.health.get('red_hearts', 0)
        self.before_contact = self.baseline
        self.heal_target = None
        self.heal_before = None
        self.heal_missing_since = None
        self.done = None

    def update(self, world, execution, log):
        from .control import blocked, path_step
        p = xy(world.player.get('pos'))
        machine = self.offer['entity']
        center = xy(machine.get('pos'))
        contract = self.offer['recovery_contract']
        execution = dict(execution, navigation_contact=False)
        def stop(reason):
            self.done = reason
            execution.update(navigation_target=p, navigation_contact=False)
            return execution

        if world.token != self.token:
            return stop('room_changed')
        if world.combat_enemies or not world.info.get('is_clear'):
            return stop('new_danger')
        if not all(world.fresh(k, 6) for k in ('PLAYER_HEALTH', 'PLAYER_INVENTORY', 'PICKUPS', 'INTERACTABLES')):
            return stop('stale_donation_state')
        if world.frame-self.started > 900:
            return stop('cycle_timeout')
        red = world.health.get('red_hearts', 0)
        damaged = world.last_damage_frame > self.damage_frame or red < self.before_contact
        if self.phase == 'contact' and damaged:
            self.contacts += 1
            self.phase, self.phase_started = 'withdraw', world.frame
            self.damage_frame = world.last_damage_frame
            log.write('donation_phase', phase=self.phase, contacts=self.contacts, frame=world.frame)
        if self.phase == 'withdraw':
            if math.dist(p, center) < 70:
                # Pick a reachable retreat rather than blindly stepping into a wall.
                options = [(center[0]+90*math.cos(i*math.pi/4), center[1]+90*math.sin(i*math.pi/4)) for i in range(8)]
                options = [q for q in options if not blocked(world, q)
                           and math.dist(path_step(world, q), p) > 1]
                if not options or world.frame-self.phase_started > 90:
                    return stop('withdrawal_blocked')
                execution['navigation_target'] = min(options, key=lambda q: math.dist(p, q))
                return execution
            if world.sampled.get('PLAYER_HEALTH', -1) <= self.damage_frame:
                execution['navigation_target'] = p
                return execution
            self.phase, self.phase_started = 'recover', world.frame
            self.heal_before = red
        if (records(world.payload.get('BOMBS')) or world.projectiles
                or any(not f.get('is_extinguished') for f in records(world.payload.get('FIRE_HAZARDS')))):
            return stop('new_hazard')
        if red < 4:
            return stop('health_reserve_breached')
        live_machine = next((m for m in records(world.payload.get('INTERACTABLES'))
                             if m.get('id') == machine.get('id') and m.get('variant') == 2
                             and not m.get('broken')), None)
        # New/morphed hearts do not silently expand the accepted recovery terms.
        terms = {h['id']: h for h in contract['heart_healing']}
        hearts = [h for h in recovery_hearts(world) if h['id'] in terms
                  and h['sub_type'] == terms[h['id']]['sub_type']
                  and red_healing(world,h) == terms[h['id']]['half_hearts']]
        valid_ids = {h['id'] for h in hearts}
        accessible = {h['id']: h for h in accessible_recovery_hearts(world) if h['id'] in valid_ids}
        if self.phase == 'contact' and not accessible:
            return stop('recovery_unreachable_before_payment')
        if self.phase == 'recover':
            save_whole_heart = (accessible and live_machine and red >= 6
                                and self.contacts < contract['max_contacts']
                                and 0 < self.baseline-red < min(red_healing(world,h) for h in accessible.values()))
            if red >= self.baseline or save_whole_heart:
                if self.contacts >= contract['max_contacts'] or not live_machine or not accessible:
                    return stop('recovered_and_finished')
                self.phase, self.phase_started = 'contact', world.frame
                self.before_contact = red
                self.damage_frame = world.last_damage_frame
                self.heal_target = None
            else:
                if self.heal_target is not None and not any(h['id'] == self.heal_target for h in hearts):
                    if red <= self.heal_before:
                        # Allow one fresh-health update to follow pickup disappearance.
                        if self.heal_missing_since is None:
                            self.heal_missing_since = world.frame
                        if world.frame-self.heal_missing_since <= 6:
                            execution['navigation_target'] = p
                            return execution
                        return stop('healing_not_observed')
                    self.heal_target = None
                    self.heal_missing_since = None
                candidates = list(accessible.values())
                if not candidates or world.frame-self.phase_started > 180:
                    return stop('healing_unavailable')
                target = next((h for h in candidates if h['id'] == self.heal_target),
                              min(candidates, key=lambda h: (max(0,red_healing(world,h)-(self.baseline-red)),
                                                             math.dist(p, xy(h['pos'])))))
                if target['id'] != self.heal_target:
                    self.heal_target, self.heal_before = target['id'], red
                    self.phase_started = world.frame
                execution['navigation_target'] = xy(target['pos'])
                return execution
        if not live_machine or red < 6 or not hearts or world.inventory.get('coins', 0) >= 99:
            return stop('donation_budget_finished')
        if world.frame-self.phase_started > 180:
            return stop('contact_timeout')
        execution.update(navigation_target=center, navigation_contact=True)
        return execution
