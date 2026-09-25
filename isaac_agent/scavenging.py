"""Observed shootable resource sources; a whole-room choice, not one API call per pile."""
from .state import records


def explosive_firing_spot(world,target):
    """Find a reachable firing position checked against the whole blast chain."""
    from copy import copy
    import math
    from .state import xy
    from .control import blocked,firing_lane,path_step
    from .hazards import safe_shot
    p,q=xy(world.player.get('pos')),xy(target['pos'])
    reach=float(world.stats.get('range',260))
    distance=125+float(world.stats.get('size',12))+40
    points=[p]+sorted([(q[0]+dx*distance,q[1]+dy*distance) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))],
                      key=lambda point:math.dist(p,point))
    for point in points:
        dx,dy=q[0]-point[0],q[1]-point[1]
        if (min(abs(dx),abs(dy))>=16 or not distance-1<=math.dist(point,q)<reach
                or blocked(world,point,float(world.stats.get('size',12))) or not firing_lane(world,point,q)):
            continue
        shoot=(1 if dx>0 else -1,0) if abs(dx)>abs(dy) else (0,1 if dy>0 else -1)
        probe=copy(world);probe.payload=dict(world.payload)
        probe.payload['PLAYER_POSITION']=[{**world.player,'pos':dict(x=point[0],y=point[1]),'vel':dict(x=0,y=0)}]
        if not safe_shot(probe,(0,0),shoot):
            continue
        if point==p or math.dist(path_step(world,point),p)>1:
            return point
    return None


def reachable(world, target):
    if target.get('explosive'):
        return explosive_firing_spot(world,target) is not None
    from .state import xy
    from .control import combat_waypoint, firing_lane
    point,q=xy(world.player.get('pos')),xy(target['pos'])
    desired=min(150,max(60,float(world.stats.get('range',260))*.7))
    return ((min(abs(point[0]-q[0]),abs(point[1]-q[1]))<16 and firing_lane(world,point,q))
            or combat_waypoint(world,q,desired=desired) is not None)


def targets(world):
    if (world.combat_enemies or not world.info.get('is_clear')
            or not world.fresh('ROOM_LAYOUT', 6) or not world.fresh('FIRE_HAZARDS', 6)):
        return []
    result = []
    from .hazards import poop_explosions
    cap=poop_explosions(world)
    for fire in records(world.payload.get('FIRE_HAZARDS')):
        if fire.get('type') == 'FIREPLACE' and fire.get('variant') == 0 and not fire.get('is_extinguished'):
            result.append({'key': f"fire:{fire.get('id')}", 'pos': dict(fire['pos']), 'kind': 'ordinary_fire'})
    for index, cell in world.layout.get('grid', {}).items():
        if cell.get('type') == 14 and cell.get('variant', 0) == 0 and cell.get('collision', 0) != 0:
            target={'key': f"poop:{index}", 'pos': {'x': cell['x'], 'y': cell['y']}, 'kind': 'brown_poop'}
            if cap:target.update(explosive=True,blast_radius_estimate=125)
            result.append(target)
    return result


def economy(world):
    """Visible stock and observed shortages, without guessing unseen shop contents."""
    stock = []
    for room, memory in world.rooms.items():
        if memory.get('type') != 2:
            continue
        for item in memory.get('collectibles', []):
            price = item.get('price', 0)
            if price > 0:
                stock.append({'room': room, 'item_id': item.get('sub_type'), 'price': price,
                              'coins_short': max(0, price-world.inventory.get('coins', 0))})
    from .secrets import ready as secret_ready
    from .rocks import offers as rock_offers
    return {'observed_shop_items': stock[:16],
            'ordinary_rock_access_protocol':world.capabilities.get('resource_rock_protocol')==1,
            'ordinary_rock_access_scope':'Only supplied resource_rock choices authorize one ordinary-rock bomb. '
                'Compare known blocked resources and stock cost; one-rock removal must open a checked route. '
                'Remote access-review rooms may be inspected for such a contract, but it is not guaranteed. '
                'Opening terrain is not proof of collection or profit.',
            'missing_red_half_hearts': max(0, world.health.get('max_hearts', 0)-world.health.get('red_hearts', 0)),
            'shootable_resource_sources': len(targets(world)),
            'bomb_wall_execution_available': secret_ready(world),
            'marked_rock_execution_available': bool(rock_offers(world)),
            'marked_rock_scope': 'Only current rock_bomb contracts. bomb_limit permits at most one/two normal-fuse placements. Normal payment reserves one; golden payment consumes zero stock. Retreat and observe each explosion; uncertain drops.',
            'bomb_wall_scope': 'Only supplied single-placement contracts; checked geometry, observed map, normal fuse, max two attempts/floor. Normal payment reserves one; golden payment consumes zero stock. No guaranteed hidden room.',
            'heart_price_purchase_execution_available': (world.capabilities.get('repentance_plus') is False and world.stats.get('player_type') in (0,1)),
            'heart_price_scope': 'Only supplied contracts; ordinary Isaac/Magdalene, Repentance, supported health prices, >=2 hearts remaining'}
