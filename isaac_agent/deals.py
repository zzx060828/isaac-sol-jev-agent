"""Observed Repentance deal contracts. Prices are engine enums, health is half-hearts."""
# https://wofsauge.github.io/IsaacDocs/rep/enums/PickupPrice.html
HEART_COSTS = {-1: (2, 0), -2: (4, 0), -3: (0, 6), -4: (2, 4),
               -7: (0, 2), -8: (0, 4), -9: (2, 2)}

DEAL_EVENTS = {'deal_door_observed', 'deal_room_entered', 'deal_door_skipped', 'paid_item_acquired'}


def history(memory):
    """Typed observed evidence; no inferred deal percentage or future door."""
    records = memory.deal_records
    doors = [r for r in records if r['kind'] == 'deal_door_observed']
    entries = [r for r in records if r['kind'] == 'deal_room_entered']
    purchases = [r for r in records if r['kind'] == 'paid_item_acquired']
    skips = [r for r in records if r['kind'] == 'deal_door_skipped']
    def compact(r):
        evidence = r['evidence']
        result = {k:r[k] for k in ('level','room','frame','kind')}
        result.update({k:evidence[k] for k in ('key','type','room_type','item_id','observed_price') if k in evidence})
        if r['kind'] == 'paid_item_acquired':
            result['classification'] = ('devil_room_health_price_acquisition'
                if evidence.get('room_type') == 14 and evidence.get('observed_price') in HEART_COSTS else
                'devil_room_coin_price_acquisition' if evidence.get('room_type') == 14
                    and evidence.get('observed_price',0) > 0 else
                'boss_room_health_price_acquisition' if evidence.get('room_type') == 5
                    and evidence.get('observed_price') in HEART_COSTS else
                'shop_acquisition' if evidence.get('room_type') == 2 else 'other_paid_acquisition')
        return result
    return {
        'history_complete_from_run_start': bool(memory.complete and memory.deal_records_complete),
        'typed_records_complete_since_observer_start': memory.deal_records_complete,
        'observed_counts': {kind:sum(r['kind']==kind for r in records) for kind in sorted(DEAL_EVENTS)},
        'first_retained_devil_boss_door': next((compact(r) for r in doors if r['evidence'].get('type')==14),None),
        'recent_doors': [compact(r) for r in doors[-6:]],
        'recent_entries': [compact(r) for r in entries[-6:]],
        'recent_acquisitions': [compact(r) for r in purchases[-6:]],
        'recent_skip_decisions': [compact(r) for r in skips[-6:]],
        'caution': 'A skip decision is intent, not proof of never entering later on that floor. '
                   'Acquisition classification uses observed room and listed price, not an engine eligibility flag. '
                   'Unknown or incomplete history is not zero history; no exact deal chance or next-floor guarantee is inferred.'}


def health_contract(world, item):
    price = item.get('price', 0)
    if price not in HEART_COSTS or item.get('variant') != 100:
        return None
    # Ordinary red-heart characters only until alternate health systems are modeled.
    h = world.health
    if not world.fresh('PLAYER_HEALTH', 6) or not world.fresh('PLAYER_INVENTORY', 6):
        return None
    if (world.capabilities.get('repentance_plus') is not False
            or world.stats.get('player_type') not in (0, 1)
            or any(h.get(k, 0) for k in ('bone_hearts', 'rotten_hearts', 'broken_hearts'))):
        return None
    red_cost, soul_cost = HEART_COSTS[price]
    maximum, red, soul = (h.get(k, 0) for k in ('max_hearts', 'red_hearts', 'soul_hearts'))
    if maximum < red_cost or soul < soul_cost:
        return None
    remaining_max = maximum-red_cost
    remaining_red = min(red, remaining_max)
    remaining_soul = soul-soul_cost  # Black hearts are already included in soul_hearts.
    if remaining_red+remaining_soul < 4:
        return None
    return {'red_container_cost_half_hearts': red_cost, 'soul_cost_half_hearts': soul_cost,
            'remaining_max_red_half_hearts': remaining_max,
            'remaining_red_half_hearts': remaining_red,
            'remaining_soul_half_hearts': remaining_soul,
            'remaining_total_half_hearts': remaining_red+remaining_soul,
            'minimum_survival_half_hearts': 4,
            'assumptions': 'No credit for promised item healing, revives or later pickups; revalidate before contact.'}


def door_offers(world):
    if world.info.get('room_type') != 5:
        return []
    skipped = getattr(world, 'run_memory', None)
    skipped = skipped.skipped if skipped else []
    result = []
    for slot, door in world.layout.get('doors', {}).items():
        if door.get('target_room_type') not in (14, 15) or not door.get('is_open') or door.get('is_locked'):
            continue
        key = f'{world.level}:{world.room}:{slot}'
        if key in skipped:
            continue
        # An entered room can be revisited if requested, but not offered forever.
        if str(door.get('target_room')) in world.rooms:
            continue
        for action in ('enter', 'skip'):
            devil = door['target_room_type'] == 14
            effect = ('Enter the observed Devil Room. If this is the first offered Devil Room, entering '
                      'without buying still forfeits the next-successful-deal Angel replacement from skipping entry.'
                      if devil and action == 'enter' else
                      'Decline entry to this Devil door for this floor. The first-offer Angel replacement applies '
                      'only if the relevant first Devil Room remains entirely unentered; not a guaranteed next-floor door.'
                      if devil else
                      'Enter the observed Angel Room and inspect its rewards. This is not entering a Devil Room; '
                      'free item alternatives may be mutually exclusive.' if action == 'enter' else
                      'Decline this Angel Room entrance and its currently unseen rewards. Skipping an Angel Room '
                      'does not grant the first-Devil-room skip-entry reward.')
            result.append({'choice': f'deal_{action}:{slot}', 'kind': f'deal_{action}',
                'entity': {**door, 'pos': door.get('pos', {'x':door.get('x',0),'y':door.get('y',0)})}, 'door_key': key,
                'deal_type': 'devil' if devil else 'angel', 'effect': effect})
    return result
