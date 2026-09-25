"""Shared pickup semantics for routing, navigation and planner invalidation.

Subtype IDs: IsaacDocs rep/enums/{Heart,Coin,Key,Bomb,Battery}SubType.html.
An unpriced entity is not necessarily a safe, collectible resource.
"""

ROUTINE_SUBTYPES = {
    10: {1, 2, 3, 5, 6, 8, 9},  # Red, soul, black and scared red hearts.
    20: {1, 2, 3, 4, 5, 7},     # Sticky nickel (6) needs destruction first.
    30: {1, 2, 3},               # Normal/golden/double. Charged key (4) stays strategic.
    40: {1, 2, 4},               # Normal, double and golden; never troll bombs.
    50: {1},                     # Ordinary CLOSED chest only; OPENED is 0.
    90: {1, 2},                  # Mega/golden batteries need separate assessment.
}

CHEST_ROOMS = {1, 4, 5, 7, 8}  # Challenge-room rewards can trigger another fight.
RED_HEARTS = {1:2, 2:1, 5:4, 9:2}


def has_golden_key(world):
    return world.fresh('PLAYER_INVENTORY', 6) and world.inventory.get('golden_key') is True


def ordinary_red_health(world):
    return (world.stats.get('player_type') in (0,1)
            and not any(world.health.get(k,0) for k in ('bone_hearts','rotten_hearts','broken_hearts')))


def red_healing(world, item):
    if item.get('variant') != 10:
        return 0
    return RED_HEARTS.get(item.get('sub_type'),0) * (2 if world.inventory.get('collectibles',{}).get('312',0) else 1)


def red_waste(world, item):
    # The Jar can store overflow; do not call that health wasted.
    if any(a.get('item') == 290 for a in world.inventory.get('active_items',{}).values()):
        return 0
    missing=max(0,world.health.get('max_hearts',0)-world.health.get('red_hearts',0))
    return max(0,red_healing(world,item)-missing)


def automatic_touch(world, item):
    if item.get('variant') == 50:
        return (item.get('price', 0) == 0 and not item.get('options_index', 0)
                and (item.get('sub_type') == 0 or
                     item.get('sub_type') == 1 and world.info.get('room_type') in CHEST_ROOMS))
    return routine(item)


def routine(item):
    return (item.get('sub_type') in ROUTINE_SUBTYPES.get(item.get('variant'), ())
            and item.get('price', 0) == 0 and not item.get('options_index', 0))


def touch_collectible(item):
    """Reject objects whose current form cannot be collected by touching."""
    return not (item.get('variant') == 40 and item.get('sub_type') in (3, 5, 6)
                or item.get('variant') == 20 and item.get('sub_type') == 6)


def useful(world, item):
    if not routine(item) or item.get('wait', 0) > 0:
        return False
    variant, subtype = item['variant'], item['sub_type']
    inv, hp = world.inventory, world.health
    if variant == 50:
        return automatic_touch(world, item)
    if variant == 20:
        cap = 999 if inv.get('collectibles', {}).get('416', 0) else 99
        return inv.get('coins', 0) < cap
    if variant == 30:
        return not has_golden_key(world) if subtype == 2 else inv.get('keys', 0) < 99
    if variant == 40:
        return not (world.fresh('PLAYER_INVENTORY',6) and inv.get('golden_bomb') is True) if subtype == 4 else inv.get('bombs', 0) < 99
    if variant == 90:
        from .charge import battery_target
        return battery_target(world) is not None
    if variant == 10:
        if subtype in (1, 2, 5, 9):
            return hp.get('red_hearts', 0) < hp.get('max_hearts', 0)
        capacity_used = hp.get('max_hearts', 0) + hp.get('soul_hearts', 0) + 2*hp.get('bone_hearts', 0)
        return capacity_used < 24 - 2*hp.get('broken_hearts', 0)
    return False
