"""Separate routine interactions from slow, consequential planning."""
from .strategy import offers, pressure_plate_offers
from .knowledge import item_descriptions
from .pickups import routine

# Reviewed from installed EID stat/health descriptions. This is an execution
# policy, not a claim that these are optimal for every character/build.
SIMPLE_PASSIVES = {1, 4, 12, 14, 16, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                   50, 70, 92, 101, 119, 138, 143, 165, 183, 189, 193, 196, 197,
                   198, 199, 218, 219, 253, 254, 343, 345, 354, 355, 370, 438,
                   456, 564, 600, 707, 730}
BASIC = {10, 20, 30, 40, 90}


def ordinary_pickup(offer):
    e = offer.get('entity', {})
    return offer['kind'] == 'pickup' and routine(e)


def fast_offers(world, finished=()):
    if not world.strategy_ready() or world.combat_enemies:
        return []
    plates = [o for o in pressure_plate_offers(world) if o['choice'] not in finished]
    if plates:
        return plates
    if not world.info.get('is_clear'):
        return []
    available = [o for o in offers(world) if o['choice'] not in finished]
    important = [o for o in available if not ordinary_pickup(o)]
    pedestals = [e for e in world.pickups if e.get('variant') == 100]
    items = world.inventory.get('collectibles', {})
    # Reroll/absorption/duplication tools create an opportunity cost even for
    # an otherwise routine pedestal. Unknown character item rules are deferred.
    special_tool = False
    for active in world.inventory.get('active_items', {}).values():
        text = item_descriptions().get(active.get('item'), {}).get('description', '').lower()
        if any(word in text for word in ('reroll', 'absorb', 'duplicate', 'destroys all', 'removes all')):
            special_tool = True
    ordinary_character = world.stats.get('player_type') in (0, 1)
    if len(pedestals) == 1 and ordinary_character and not special_tool and world.info.get('room_type') in (1, 4, 5, 7, 8):
        item = pedestals[0]
        if (item.get('sub_type') in SIMPLE_PASSIVES and not item.get('price', 0)
                and not item.get('options_index', 0) and not item.get('is_hidden')
                and world.info.get('curses') is not None
                and not (world.info['curses'] & 64)
                and world.info.get('stage_type', 0) < 4
                and item.get('item_type', 1) == 1
                and not any(str(i) in items for i in (258, 689, 619))):
            target = next((o for o in important if o['kind'] == 'pickup' and o['entity'].get('id') == item.get('id')), None)
            if target and all(o is target or o['kind'] in ('descend', 'scavenge','secret_bomb','rock_bomb') for o in important):
                return [target]
    # A single bounded sweep is cheap to decide and needs no per-fire API calls.
    if important and all(o['kind'] in ('scavenge', 'descend') for o in important):
        return [o for o in important if o['kind'] == 'scavenge']
    return []


def lane(world, room_started, finished=(), deferred=(), combat_stalled=False):
    from .control import pickup_target
    available = [o for o in offers(world) if o['choice'] not in finished]
    important = [o for o in available if not ordinary_pickup(o)]
    # Entering a room briefly gives existing pickups a positive native Wait.
    # Keep a consequential pickup visible to scheduling until it can be reviewed;
    # executable offers still exclude Wait > 0. Bound this entry grace period so
    # a broken/stale timer cannot keep the player in the room indefinitely.
    if (all(world.fresh(k, 8) for k in ('PICKUPS', 'ROOM_LAYOUT', 'ROOM_INFO'))
            and world.info.get('is_clear') and not world.combat_enemies
            and 0 <= world.frame - room_started < 60
            and any(e.get('wait', 0) > 0 for e in world.pickups)
            and any(o['kind'] == 'pickup' and o['choice'] not in finished
                    and o['entity'].get('wait', 0) > 0 and not ordinary_pickup(o)
                    for o in offers(world, include_waiting=True))):
        return 'settle_pickups', []
    if (world.strategy_ready() and world.info.get('is_clear') and not world.combat_enemies
            and all(o['kind'] in ('descend','scavenge') for o in important)
            and pickup_target(world, {'excluded_choices':finished}, basic_only=True) is not None):
        return 'collect_resources', []
    fast = fast_offers(world, finished)
    if fast and fast[0]['choice'] not in deferred:
        return 'fast', fast
    if world.combat_enemies:
        if world.info.get('room_type') in (5, 6) or combat_stalled:
            return 'sol', []  # Boss or observed stalled/prolonged fighting.
        return 'tactical', []
    if important:
        return 'sol', []
    return 'routine', []


def compact_goal_observation(observation, choices):
    return {k: observation.get(k) for k in ('frame', 'room', 'level', 'health', 'stats',
            'inventory', 'held_consumable_mechanics', 'ordinary_battery_target', 'economy', 'resource_policy', 'decision_knowledge', 'room_knowledge')} | {
        'fast_offers': choices,
        'contract': 'Only supplied choices are authorized. No paid purchase, active swap, option-group pickup or wall bombing.'}
