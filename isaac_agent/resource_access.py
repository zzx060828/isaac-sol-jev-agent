"""Carry measured pickup approach limits into strategic room memory."""
from copy import deepcopy
import math
from .state import xy
from .review import choice_id


def movement_context(world):
    return {'can_fly':world.stats.get('can_fly'), 'size':world.stats.get('size'),
            'character':world.stats.get('player_type'),
            'collectibles':world.inventory.get('collectibles',{})}


def record(world,item,status):
    # No extra path search: the ordinary picker reports its existing result.
    # Sensor gaps and unknown entry geometry are not evidence of disconnection.
    if not all(world.fresh(k,6) for k in ('ROOM_INFO','ROOM_LAYOUT','PICKUPS',
            'PLAYER_STATS','PLAYER_INVENTORY','FIRE_HAZARDS')):
        return
    room=world.rooms.get(str(world.room))
    if room is None:return
    key=choice_id(item);point=xy(item.get('pos'))
    entries=room.setdefault('resource_access',{})
    previous=entries.get(key,{})
    context=movement_context(world)
    if (previous.get('status')==status and previous.get('pos')==list(point)
            and previous.get('movement_context')==context
            and previous.get('_navigation_revision')==world.navigation_revision):
        return
    evidence={'choice':key,'pos':list(point),'status':status,
              'movement_context':deepcopy(context),
              '_navigation_revision':world.navigation_revision,
              'nearby_solid_types':sorted({int(g['type']) for g in world.layout.get('grid',{}).values()
                    if g.get('collision') in (1,2,3,4)
                    and math.dist(point,(g['x'],g['y']))<85}),
              'scope':'Observed local navigation result, not permanent unreachability or permission to bomb.'}
    # Preserve the observation frame until its meaning changes; avoid a disk
    # write per walking tick through RunMemory's geometry persistence.
    if {k:v for k,v in previous.items() if k!='frame'} != evidence:
        entries[key]={**evidence,'frame':world.frame}
    present={choice_id(p) for p in world.pickups}
    for old in list(entries):
        if old not in present:del entries[old]


def evidence(world,room):
    entries=room.get('resource_access') or {}
    present={choice_id(p) for p in room.get('resources',[])}
    current=movement_context(world)
    return [{k:deepcopy(v) for k,v in entry.items() if k not in ('movement_context','_navigation_revision')}
            | {'movement_context_changed':entry.get('movement_context')!=current}
            for key,entry in entries.items() if key in present][:24]


def needs_access(world,room,item):
    entry=(room.get('resource_access') or {}).get(choice_id(item),{})
    return (entry.get('status') in ('endpoint_blocked','no_local_route')
            and entry.get('movement_context')==movement_context(world))
