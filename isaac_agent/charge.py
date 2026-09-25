"""Recipient eligibility for ordinary battery pickups, not permission to use items."""


def battery_target(world):
    if not world.fresh('PLAYER_INVENTORY',6):
        return None
    inv=world.inventory
    overcharge=bool(inv.get('collectibles',{}).get('63',0))
    # Schoolbag's inactive slot 1 is not charged by ordinary batteries.
    # The current primary item has priority over the pocket active (slot 2).
    for slot in ('0','2'):
        active=inv.get('active_items',{}).get(slot,{})
        if not active.get('item'):
            continue
        if 'needs_charge' in active:
            if active['needs_charge'] is not True:
                continue
            source='game NeedsCharge'
        else:
            # Older recordings lack the native check. Only infer ordinary bar
            # charging; timed/special items have different pickup behavior.
            maximum=active.get('max_charge',0)
            capacity=maximum*(2 if overcharge else 1)
            if (active.get('charge_type',0)!=0 or maximum<=0
                    or active.get('charge',0)+active.get('battery_charge',0)>=capacity):
                continue
            source='observed normal-charge capacity estimate'
        return dict(slot=slot,item=active['item'],eligibility_source=source)
    return None
