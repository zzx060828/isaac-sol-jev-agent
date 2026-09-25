"""Conservative explosion geometry; radii are estimates, not engine physics."""
import math

from .state import records, xy
from .physics import trajectory, PHYSICS_TICKS_PER_UPDATE


def distance_to_segment(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    t = max(0, min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx*dx + dy*dy or 1)))
    return math.dist(point, (start[0] + t*dx, start[1] + t*dy))


def advancing_creep(world):
    """Infer short attack fronts from causally observed new patches.

    Static creep and a first snapshot cannot establish propagation. Each front
    expires four updates after its latest patch; projections are capped at80
    units from two patches, or120 after a third confirms the same velocity.
    This estimates an imminent hazard, not an engine damage guarantee.
    """
    if world.stats.get('can_fly'):
        return []
    result = []
    history = world.ground_history
    for tip in history:
        if world.frame-tip['frame'] > 4:
            continue
        p = xy(tip.get('pos'))
        previous = [row for row in history if row.get('variant') == tip.get('variant')
                    and 1 <= tip['frame']-row['frame'] <= 4
                    and 20 <= math.dist(p, xy(row.get('pos'))) <= 65]
        for middle in previous:
            q = xy(middle.get('pos'))
            dt = tip['frame']-middle['frame']
            v = (p[0]-q[0])/dt, (p[1]-q[1])/dt
            speed = math.hypot(*v)
            if not 6 <= speed <= 30:
                continue
            consistent = False
            for older in history:
                gap = middle['frame']-older['frame']
                if older.get('variant') != tip.get('variant') or not 1 <= gap <= 4:
                    continue
                r = xy(older.get('pos'))
                projected = r[0]+v[0]*gap, r[1]+v[1]*gap
                if math.dist(projected, q) <= 12:
                    consistent = True
                    break
            result.append({'pos': p, 'velocity': v, 'age': world.frame-tip['frame'],
                           'radius': float(tip.get('collision_radius', 24)),
                           'limit': 120 if consistent else 80,
                           'confidence': 1 if consistent else .5})
            break
    return result


def creep_front_risk(point, t, fronts, player_radius):
    score = 0.0
    for front in fronts:
        p, v = front['pos'], front['velocity']
        distance = min(front['limit'], math.hypot(*v)*(t+front['age']))
        norm = math.hypot(*v)
        end = p[0]+v[0]*distance/norm, p[1]+v[1]*distance/norm
        clearance = distance_to_segment(point, p, end)-front['radius']-player_radius
        if clearance < 10:
            score += front['confidence']*(20 + max(0, -clearance)*2)/t
    return score


def explosives(world):
    result = []
    # A spawned troll pickup can precede its entity-bomb sensor sample.
    for item in world.pickups:
        if item.get('variant') == 40 and item.get('sub_type') in (3,5,6):
            result.append({'pos':xy(item.get('pos')),'radius':125.0,
                           'armed':True,'vel':xy(item.get('vel'))})
    cap=poop_explosions(world)
    for cell in world.layout.get("grid", {}).values():
        if (cell.get("type") in (5, 12) or cap and cell.get('type')==14) and cell.get("collision", 0) != 0:
            result.append({"pos": (cell["x"], cell["y"]), "radius": 125.0,
                           # TNT states 0..3 encode accumulated damage, not a
                           # running fuse. State 4 is exploded. Incoming tears,
                           # bombs, and other ignition sources are handled below.
                           # https://isaacscript.github.io/isaac-typescript-definitions/enums/tntstate/
                           "armed": False, "vel": (0, 0)})
    for bomb in records(world.payload.get("BOMBS")):
        radius = float(bomb.get("explosion_radius", 0))
        result.append({"pos": xy(bomb.get("pos")), "radius": max(125.0, radius),
                       "armed": True, "vel": xy(bomb.get("vel"))})
    return result


def poop_explosions(world):
    # Native HasTrinket also observes swallowed effects; old recordings only
    # expose held normal/golden IDs. Preserve observed danger until replaced.
    inv=world.inventory
    return inv.get('poop_explosions') is True or any(
        type(inv.get(slot)) is int and inv[slot]&0x7fff==90 for slot in ('trinket_0','trinket_1'))


def chain_indices(sources, seeds):
    reached, todo = set(seeds), list(seeds)
    while todo:
        i = todo.pop()
        for j, other in enumerate(sources):
            if j not in reached and math.dist(sources[i]["pos"], other["pos"]) <= sources[i]["radius"] + 20:
                reached.add(j)
                todo.append(j)
    return reached


def threatened_explosives(world, sources):
    seeds = {i for i, source in enumerate(sources) if source["armed"]}
    # Already fired tears can trigger TNT even after shooting is released.
    for tear in world.payload.get("PROJECTILES", {}).get("player_tears", []):
        p, v = xy(tear.get("pos")), xy(tear.get("vel"))
        end = p[0] + v[0]*12, p[1] + v[1]*12
        for i, source in enumerate(sources):
            if distance_to_segment(source["pos"], p, end) < 24:
                seeds.add(i)
    return chain_indices(sources, seeds)


def explosion_risk(world, point, t, sources, threatened):
    radius = float(world.stats.get("size", 12)) + 18
    # An intact barrel does not explode merely because we walk near it. Keep
    # clearance when there are possible ignition sources; in a quiet room its
    # physical collision is already handled by blocked(). Even a small blast
    # penalty here can overwhelm pickup progress across several chained barrels.
    # Ordinary enemy presence is not an ignition source. Counting every spider
    # as one pinned the player to a corner of a TNT room despite released fire.
    possible_ignition = (bool(world.projectiles)
                         or any(laser.get("is_enemy", True) for laser in
                                world.payload.get("PROJECTILES", {}).get("lasers", []))
                         or any(not fire.get("is_extinguished") and not fire.get("ground_only")
                                for fire in records(world.payload.get("FIRE_HAZARDS"))))
    result = 0.0
    for i, source in enumerate(sources):
        center = (source["pos"][0] + source["vel"][0]*t*PHYSICS_TICKS_PER_UPDATE,
                  source["pos"][1] + source["vel"][1]*t*PHYSICS_TICKS_PER_UPDATE)
        clearance = math.dist(point, center) - source["radius"] - radius
        if clearance < 0:
            # Do not hug an unlit barrel during combat: other entities can ignite it.
            weight = 8 if i in threatened else 2 if possible_ignition else 0
            result += (20 - clearance) * weight / t
    return result


def safe_shot(world, move, shoot):
    if shoot == (0, 0):
        return True
    from .knowledge import enemy_briefs
    sources = explosives(world)
    death_explosives = [e for e in world.combat_enemies if 'death_explosion' in
                       enemy_briefs().get(f"{e.get('type')}:{e.get('variant', 0)}", {}).get('tags', [])]
    if not sources and not death_explosives:
        return True
    p = xy(world.player.get('pos'))
    path = trajectory(world, move, 8)
    # Reviewed death explosions can be triggered by our shot. Require room
    # before firing into them even when no planted bomb/TNT exists.
    for enemy in death_explosives:
        q = xy(enemy.get('pos'))
        travel = max(30, float(world.stats.get('range', 260)))
        for start in (p, path[3], path[7]):
            end = start[0]+shoot[0]*travel, start[1]+shoot[1]*travel
            if distance_to_segment(q, start, end) < float(enemy.get('collision_radius', 12))+16:
                if math.dist(start, q) < 125+float(world.stats.get('size', 12))+24:
                    return False
    if not sources:
        return True
    p = xy(world.player.get("pos"))
    travel = max(80, float(world.stats.get("range", 300)))
    margin = float(world.stats.get("size", 12)) + 24
    for start in (p, path[3], path[7]):
        end = start[0] + shoot[0]*travel, start[1] + shoot[1]*travel
        hit = {i for i, source in enumerate(sources)
               if distance_to_segment(source["pos"], start, end) < 28}
        for i in chain_indices(sources, hit):
            if min(math.dist(p, sources[i]["pos"]), math.dist(start, sources[i]["pos"])) < sources[i]["radius"] + margin:
                return False
    return True
