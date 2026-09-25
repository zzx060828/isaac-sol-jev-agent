"""One ordinary-rock opening for already observed, useful free resources."""
from copy import copy, deepcopy
import math
from .state import xy, records
from .pickups import routine, useful
from .bomb_budget import budget


def identity(item):
    return {k:deepcopy(item.get(k)) for k in ('id','variant','sub_type','price','options_index','pos')}


def eligible(world):
    return [p for p in world.pickups if routine(p) and useful(world,p)
            and p.get('variant') in (10,20,30,40,90) and p.get('wait',0)==0]


def route_and_loot(world,rock,expected=None):
    from .control import blocked,path_step
    from .rocks import placement
    from .hazards import poop_explosions
    route=placement(world,rock)
    if route is None:return None
    spot,escape=route
    if any(not h.get('is_extinguished') and math.dist(spot,xy(h.get('pos')))<175
           for h in records(world.payload.get('FIRE_HAZARDS'))):return None
    if poop_explosions(world) and any(g.get('type')==14 and g.get('collision',0)
            and math.dist(spot,(g['x'],g['y']))<175 for g in world.layout['grid'].values()):return None
    p=xy(world.player.get('pos'));radius=float(world.stats.get('size',12))
    loot=[e for e in eligible(world) if expected is None or identity(e) in expected]
    loot=[e for e in loot if blocked(world,xy(e['pos']),radius)
          or math.dist(path_step(world,xy(e['pos'])),p)<1]
    if not loot:return None  # Never spend just to walk to already accessible loot.
    # Change only the selected ordinary rock. Do not assume collateral blasts,
    # diagonal corner cutting, pushing pickups, flight, or new random drops.
    probe=copy(world);probe.payload=dict(world.payload)
    probe.payload['ROOM_LAYOUT']=deepcopy(world.layout)
    probe.layout['grid'][str(rock['grid_index'])]['collision']=0
    probe.payload['PLAYER_POSITION']=[{**world.player,'pos':dict(zip(('x','y'),escape)),
                                      'vel':dict(x=0,y=0)}]
    probe.path_cache={};probe.unreachable_paths={};probe.terrain_index=();probe.navigation_revision+=1
    reachable=[identity(e) for e in loot if not blocked(probe,xy(e['pos']),radius)
               and math.dist(path_step(probe,xy(e['pos'])),escape)>1]
    return (spot,escape,reachable) if reachable else None


def offers(world):
    from .rocks import ready
    from .control import walkability_signature,blocked
    if (world.capabilities.get('resource_rock_protocol')!=1 or not ready(world)
            or not world.fresh('PLAYER_STATS',6) or not world.fresh('PLAYER_POSITION',3)):return []
    funding=budget(world,1)
    if funding is None:return []
    items=eligible(world)
    if not items:return []
    p=xy(world.player.get('pos'))
    if blocked(world,p,float(world.stats.get('size',12))):return []
    key=(walkability_signature(world),tuple((e['id'],e['variant'],e.get('sub_type'),xy(e['pos'])) for e in items))
    cached=getattr(world,'resource_rock_cache',None)
    if cached is None or cached['key']!=key or (not cached['offers']
            and world.frame-cached['frame']>=30 and math.dist(p,cached['origin'])>=80):
        from .control import path_step
        radius=float(world.stats.get('size',12))
        blocked_loot=[e for e in items if blocked(world,xy(e['pos']),radius)
                      or math.dist(path_step(world,xy(e['pos'])),p)<1]
        result=[]
        nearby=[(index,g) for index,g in world.layout.get('grid',{}).items()
                if g.get('type')==2 and g.get('collision',0) in (2,3,4)
                and any(math.dist(xy(e['pos']),(g['x'],g['y']))<100 for e in blocked_loot)]
        nearby.sort(key=lambda entry:math.dist(p,(entry[1]['x'],entry[1]['y'])))
        for index,g in nearby[:16]:  # Bounded search, not proof no other opening exists.
            rock={'grid_index':int(index),'type':2,'pos':dict(x=g['x'],y=g['y']),
                  'name':'Ordinary rock blocking observed resources','placement_count_upper_bound':1}
            route=route_and_loot(world,rock,[identity(e) for e in blocked_loot])
            if route is None:continue
            spot,escape,loot=route
            if any(o['expected_resources']==loot for o in result):continue
            marker=f'{world.level}:{world.room}:resource:{index}'
            result.append({'choice':f'resource_rock:{index}:2','kind':'rock_bomb','bomb_kind':'resource_rock',
                'entity':{'pos':dict(zip(('x','y'),spot))},'rock':rock,'attempt_key':marker,
                'escape':escape,'expected_resources':loot,
                'effect':'Spend at most one normal-fuse bomb, retreat, and recheck a walking route to these already visible free resources. '
                         'Only selected-rock removal was simulated. Counts are potential loot, not guaranteed acquisition or profit. '
                         'Normal payment reserves one bomb; compare stock consumed, missing health and next-floor needs. '
                         'Other drops are unknown; do not count random rock drops as income.'})
        cached={'key':key,'origin':p,'frame':world.frame,'offers':result}
        world.resource_rock_cache=cached
    result=[]
    for source in cached['offers']:
        if source['attempt_key'] in world.run_memory.rock_attempts:continue
        offer=deepcopy(source);offer.update(funding)
        if funding['bomb_payment']=='golden':offer['choice']+=':golden'
        result.append(offer)
    return result[:3]
