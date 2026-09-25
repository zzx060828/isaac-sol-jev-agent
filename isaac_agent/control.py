"""Bounded inputs, conservative geometric reflex, and room navigation.

Player and hostile velocities use different time units; see physics.py.
This is an approximate controller, not a complete Isaac physics simulator.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math

from .state import records, xy
from .hazards import explosives, threatened_explosives, explosion_risk, safe_shot, advancing_creep, creep_front_risk
from .physics import trajectory, hostile_position, swept_separation
from .combat import spacing, select_enemy, band_cost, crowding_override, nearest_distance, traits
from .pickups import automatic_touch, useful as useful_pickup, has_golden_key

MOVES = ((0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
SHOTS = ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True)
class Action:
    move: tuple = (0, 0)
    shoot: tuple = (0, 0)

    @property
    def id(self):
        return f"m{MOVES.index(self.move)}s{SHOTS.index(self.shoot)}"

    def wire(self, world, seq):
        return {"move": dict(zip(("x", "y"), self.move)),
                "shoot": dict(zip(("x", "y"), self.shoot)),
                "use_item": False, "use_bomb": False, "use_card": False,
                "use_pill": False, "drop": False, "agent_seq": seq,
                "agent_room": world.room, "agent_level": world.level,
                "agent_deadline": world.frame + 6}


def segment_distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0, min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy or 1)))
    return math.dist(point, (a[0] + t * dx, a[1] + t * dy))


def nearby_grid(world, p, radius):
    """Spatial broad phase; exact collision geometry remains in blocked()."""
    grid = world.layout.get('grid', {})
    cached = world.terrain_index
    if not cached or cached[0] is not grid or cached[1] != len(grid):
        buckets = {}
        for key, cell in grid.items():
            bucket = math.floor(cell['x'] / 40), math.floor(cell['y'] / 40)
            buckets.setdefault(bucket, []).append(key)
        world.terrain_index = (grid, len(grid), buckets)
    buckets = world.terrain_index[2]
    extent = radius + 20  # Covers 19-unit square cells and 20-unit trapdoors.
    for x in range(math.floor((p[0]-extent)/40), math.floor((p[0]+extent)/40)+1):
        for y in range(math.floor((p[1]-extent)/40), math.floor((p[1]+extent)/40)+1):
            for key in buckets.get((x, y), ()):
                yield grid[key]


def entering_open_door(world, point):
    """Allow motion inward from a just-entered open doorway, never outward."""
    pos=world.player.get('pos',{})
    start=pos.get('x',0),pos.get('y',0)
    for door in world.layout.get('doors',{}).values():
        q=(door['x'],door['y'])
        if (not door.get('is_open') or abs(start[0]-q[0])>=32 or abs(start[1]-q[1])>=32
                or math.dist(start,q)>=32 or math.dist(point,q)>=32):
            continue
        lo,hi=xy(world.info.get('top_left')),xy(world.info.get('bottom_right'))
        center=((lo[0]+hi[0])/2,(lo[1]+hi[1])/2)
        axis=0 if abs(q[0]-center[0])>abs(q[1]-center[1]) else 1
        sign=1 if center[axis]>q[axis] else -1
        if (point[axis]-start[axis])*sign>.01:return True
    return False


def blocked(world, p, radius=12, exit_point=None, hazards_only=False, ignored_hazards=()):
    # Allow the final approach to a selected open door at the room boundary.
    door_approach = (exit_point is not None and math.dist(p, exit_point) < 32) or entering_open_door(world,p)
    lo, hi = xy(world.info.get("top_left")), xy(world.info.get("bottom_right", {"x": 640, "y": 560}))
    if not hazards_only and not door_approach and not (lo[0] + radius <= p[0] <= hi[0] - radius and lo[1] + radius <= p[1] <= hi[1] - radius):
        return True
    # Empty pedestals still collide with the player but aren't grid entities.
    for pedestal in world.empty_pedestals:
        if not hazards_only and math.dist(p, xy(pedestal.get("pos"))) < radius + 12:
            return True
    # Touching an unselected pedestal can irreversibly consume an item choice.
    for item in world.pickups:
        if automatic_touch(world, item):
            continue
        center = xy(item.get("pos"))
        selected = exit_point is not None and math.dist(center, exit_point) < 1
        if not selected and math.dist(p, center) < radius + 22:
            current_gap = math.dist(xy(world.player.get('pos')), center)
            if not (current_gap < radius+22 and math.dist(p, center) > current_gap+.1):
                return True  # New chest loot can appear around the player; allow withdrawal.
    # Avoid accidental payments while walking past machines/beggars. Only an
    # explicitly selected contact goal may approach its own machine.
    for machine in records(world.payload.get("INTERACTABLES")):
        center = xy(machine.get("pos"))
        selected = exit_point is not None and math.dist(center, exit_point) < 1
        if not selected and math.dist(p, center) < radius + 22:
            current_gap = math.dist(xy(world.player.get('pos')), center)
            if not (current_gap < radius+22 and math.dist(p, center) > current_gap+.1):
                return True  # A previous authorized contact must be able to withdraw.
    for hazard in records(world.payload.get("FIRE_HAZARDS")):
        if id(hazard) in ignored_hazards:
            continue
        if hazard.get("is_extinguished") or (hazard.get("ground_only") and world.stats.get("can_fly")):
            continue
        if math.dist(p, xy(hazard.get("pos"))) < radius + float(hazard.get("collision_radius", 20)):
            return True
    can_fly = bool(world.stats.get('can_fly'))
    for cell in nearby_grid(world, p, radius+8):
        typ, collision = int(cell.get("type", 0)), int(cell.get("collision", 0))
        if typ in (17, 18):
            center = (cell['x'], cell['y'])
            if not (exit_point is not None and math.dist(center, exit_point) < 1):
                if math.dist(p, center) < radius + 20:
                    return True  # Do not descend accidentally while collecting rewards.
        # GridCollisionClass: 1 pit, 2 object, 3 solid, 4 wall, 5 excludes player.
        solid = collision in (2, 3, 4) or (collision == 1 and not can_fly)
        if hazards_only:
            solid = False
        # Cycling spikes remain excluded while retracted; timing isn't reliable yet.
        hazard = (typ in (8, 9, 25) or (typ == 14 and cell.get("variant") == 1
                  and collision != 0)) and not can_fly
        if not solid and not hazard:
            continue
        if door_approach and typ in (15, 16):  # wall or door at selected door
            continue
        # Observed spiked-rock contact occurred outside the physical-rock
        # footprint after delayed braking. Do not rely on collision clamping
        # to stop at the damaging surface, especially after size reduction.
        padding=8 if typ==25 and hazard else 0
        dx = max(abs(p[0] - cell["x"]) - 19-padding, 0)
        dy = max(abs(p[1] - cell["y"]) - 19-padding, 0)
        if math.hypot(dx, dy) < radius:
            return True
    return False


def motion(world, move):
    speed = 4.0 * float(world.stats.get("speed", 1))
    norm = math.hypot(*move) or 1
    return move[0] * speed / norm, move[1] * speed / norm


def physical_collision(world, point, radius):
    lo, hi = xy(world.info.get('top_left')), xy(world.info.get('bottom_right', {'x':640,'y':560}))
    entering=entering_open_door(world,point)
    if not entering and not (lo[0]+radius <= point[0] <= hi[0]-radius and lo[1]+radius <= point[1] <= hi[1]-radius):
        return True
    for cell in nearby_grid(world, point, radius):
        if entering and cell.get('type') in (15,16):continue
        if cell.get('collision',0) not in (2,3,4) and not (cell.get('collision')==1 and not world.stats.get('can_fly')):
            continue
        if math.hypot(max(abs(point[0]-cell['x'])-19,0),max(abs(point[1]-cell['y'])-19,0)) < radius:
            return True
    return False


def risk(world, move, exit_point=None, horizon=8, maneuver=False):
    radius = float(world.stats.get("size", 12))
    score = 0.0
    hazards = world.projectiles + world.enemies
    enemy_objects = {id(entity) for entity in world.enemies}
    projectile_objects = {id(entity) for entity in world.projectiles}
    hazards += [e for e in records(world.payload.get("FIRE_HAZARDS")) if not e.get("is_extinguished")
                and not (e.get("ground_only") and world.stats.get("can_fly"))]
    # Entity coordinates belong to their channel's collect_frame. Player and
    # hostile sensors can update on different frames; do not restart the
    # hostile forecast at an older position. Stale channels remain gated by
    # World.ready; extrapolation is bounded to that six-update validity window.
    hazard_motion = {}
    for entity in hazards:
        channel = ('ENEMIES' if id(entity) in enemy_objects else
                   'PROJECTILES' if id(entity) in projectile_objects else None)
        age = min(6, max(0, world.frame-world.sampled.get(channel,world.frame))) if channel else 0
        hazard_motion[id(entity)] = (hostile_position(entity,age), xy(entity.get('vel')))
    sources = explosives(world)
    threatened = threatened_explosives(world, sources)
    creep_fronts = advancing_creep(world)
    # Use an observed damage cooldown only to escape existing body contact.
    # Conservatively count two cooldown ticks per sensor update. Never infer
    # invulnerability just from an old damage event or a flashing sprite.
    escaping_contact = (world.capabilities.get('repentance_plus') is False
                        and world.fresh('PLAYER_HEALTH', 1)
                        and 0 <= world.frame-world.last_damage_frame < 30
                        and nearest_distance(world, xy(world.player.get('pos'))) < 60)
    if escaping_contact and world.health.get('damage_cooldown', 0) >= 24:
        horizon = max(horizon, 16)
    # An observed hit can leave us on creep beside a wall. Treating the
    # existing overlap as an impassable wall makes every exit look illegal.
    # Start only on an observed overlapping patch, use the measured remaining
    # cooldown, and require hazard clearance at the prediction endpoint.
    cooldown = world.health.get('damage_cooldown',0)
    starting_ground = set()
    if (world.capabilities.get('repentance_plus') is False and not world.stats.get('can_fly')
            and world.fresh('PLAYER_HEALTH',1) and world.fresh('FIRE_HAZARDS',1)
            and 0 <= world.frame-world.last_damage_frame < 30 and cooldown >= 24):
        start=xy(world.player.get('pos'))
        starting_ground={id(h) for h in records(world.payload.get('FIRE_HAZARDS'))
                         if h.get('ground_only') and not h.get('is_extinguished')
                         and math.dist(start,xy(h.get('pos'))) < radius+float(h.get('collision_radius',20))}
        if starting_ground:
            horizon=max(horizon,min(20,(cooldown-5)//2))
    ground_ids={id(h) for h in hazards if h.get('ground_only')} if starting_ground else set()
    raw_path = trajectory(world, move, horizon)
    path = trajectory(world, move, horizon, collides=lambda p: physical_collision(world,p,radius))
    for t, point in enumerate(path, 1):
        previous_point = path[t-2] if t > 1 else xy(world.player.get('pos'))
        ignored=ground_ids if t < horizon and cooldown > 2*t+4 else ()
        if blocked(world, raw_path[t-1], radius, exit_point, ignored_hazards=ignored):
            # Replan physical turns every sensor update. Holding this direction
            # for eight updates can hit a wall beyond the next waypoint, even
            # when a short step followed by a turn is legal. Real damage and
            # unselected item/trade contacts retain the full prediction horizon.
            immediate = t <= 3 or blocked(world, point, radius, exit_point, hazards_only=True,
                                          ignored_hazards=ignored)
            score += (1000 if immediate else .2) / t
        score += explosion_risk(world, point, t, sources, threatened)
        score += creep_front_risk(point, t, creep_fronts, radius)
        for entity in hazards:
            if id(entity) in ignored:
                continue
            if (id(entity) in enemy_objects and 'no_contact_damage' in traits(entity)
                    and not entity.get('is_champion') and not entity.get('subtype',0)
                    and world.capabilities.get('repentance_plus') is False):
                continue  # Its observed bullets remain hazards; body proximity is not a damage source.
            if (id(entity) in enemy_objects and (escaping_contact or starting_ground) and t < horizon
                    and world.health.get('damage_cooldown', 0) > 2*t+4):
                continue  # Terminal body clearance, bullets, blasts and terrain still count.
            ep, velocity = hazard_motion[id(entity)]
            future = (ep[0]+velocity[0]*t,ep[1]+velocity[1]*t)
            clearance = math.dist(point, future) - radius - float(entity.get("collision_radius", 12))
            if id(entity) in enemy_objects or id(entity) in projectile_objects:
                previous_hostile = (future[0]-velocity[0],future[1]-velocity[1])
                # Minimum *simultaneous* separation during this interval.
                # World-space paths can cross at different times safely;
                # checking their intersection alone would falsely reject that.
                clearance = min(clearance, swept_separation(previous_point,point,previous_hostile,future)
                                -radius-float(entity.get('collision_radius',12)))
            if id(entity) in enemy_objects:
                # Enemies can turn toward the player during this horizon.
                # Constant-velocity prediction alone repeatedly rated a close
                # pass beside chasing Globins as safe. Reserve maneuvering
                # room before contact becomes unavoidable with player inertia.
                clearance = min(clearance, math.dist(point, ep)-radius-float(entity.get('collision_radius', 12)))
                clearance -= min(18, .15 * t * t)
                reserve = min(120, max(55, float(world.stats.get('range', 260))*.42))
                if maneuver:
                    reserve = 55  # Allow a checked flank through narrow terrain.
                if set(traits(entity)).intersection(('jump', 'charge', 'cardinal_charge')):
                    reserve += 20
                if 'death_explosion' in traits(entity):
                    reserve = max(reserve, 140)
                if clearance < reserve:
                    score += (max(0, reserve - max(0, clearance)) / 18) ** 2 / t
            if id(entity) in projectile_objects and clearance < 36:
                score += (max(0,36-clearance)/10)**2/t
            if id(entity) in projectile_objects and entity.get('is_explosive') is True:
                # EXPLODE bullets can hit terrain before the forecast endpoint.
                # Reserve an estimated blast radius along the swept path, even
                # while airborne. Height alone does not establish impact time.
                blast_clearance = segment_distance(point, ep, future)-radius-125
                if blast_clearance < 18:
                    score += (18-blast_clearance)*8/t
            if clearance < 6:
                score += (60 + max(0, -clearance) * 4) / t
            else:
                score += 0.2 / (clearance + 1)
        for laser in world.payload.get("PROJECTILES", {}).get("lasers", []):
            if not laser.get("is_enemy", True):
                continue
            a = xy(laser.get("pos"))
            angle = math.radians(float(laser.get("angle", 0)))
            length = float(laser.get("max_distance", 500))
            b = a[0] + math.cos(angle) * length, a[1] + math.sin(angle) * length
            if segment_distance(point, a, b) < radius + 10:
                score += 100 / t
    return score


def walkability_signature(world):
    """Only barriers used by blocked(); damage to an intact pile is not a new path."""
    return (world.token, bool(world.stats.get('can_fly')), float(world.stats.get('size',12)),
            xy(world.info.get('top_left')),xy(world.info.get('bottom_right')),
            tuple(sorted((g['x'],g['y'],g.get('type',0),g.get('variant',0),g.get('collision',0))
                         for g in world.layout.get('grid',{}).values())),
            tuple(sorted((d['x'],d['y'],bool(d.get('is_open'))) for d in world.layout.get('doors',{}).values())),
            tuple(sorted(xy(e.get('pos')) for e in world.empty_pedestals)),
            tuple(sorted(xy(e.get('pos')) for e in world.pickups if not automatic_touch(world, e))),
            tuple(sorted(xy(e.get('pos')) for e in records(world.payload.get('INTERACTABLES')))),
            tuple(sorted((xy(e.get('pos')),float(e.get('collision_radius',20)),bool(e.get('ground_only')))
                         for e in records(world.payload.get('FIRE_HAZARDS')) if not e.get('is_extinguished'))))


def path_step(world, target, avoid=None):
    """A* on a 20-unit local lattice; no diagonal corner cutting."""
    start = xy(world.player.get("pos"))
    if math.dist(start, target) < 28:
        return target
    spacing = 20
    origin = tuple(round(x / spacing) for x in start)
    goal = tuple(round(x / spacing) for x in target)
    radius = float(world.stats.get("size", 12))
    # A stats-only size change can close a shortcut or open a previously
    # disconnected passage without changing any terrain/pickup channel.
    cache_key = (world.token, world.navigation_revision, bool(world.stats.get("can_fly")), radius, origin, tuple(target), avoid)
    if cache_key in world.path_cache:
        return world.path_cache[cache_key] or start
    # No walking route can end inside an observed solid object or pit.
    # Shooting/terrain-destruction callers handle those targets separately.
    if blocked(world,target,radius,exit_point=target):
        world.path_cache[cache_key]=None
        return start
    distance = math.dist(start, target)
    steps = max(1, math.ceil(distance / 10))
    def excluded(point):
        return avoid is not None and math.dist(point,avoid[:2]) < avoid[2]
    if all(not excluded(point) and not blocked(world, point, radius, exit_point=target)
           for i in range(1, steps+1)
           for point in [(start[0]+(target[0]-start[0])*i/steps, start[1]+(target[1]-start[1])*i/steps)]):
        world.path_cache[cache_key] = target
        return target
    failure_key=(walkability_signature(world),tuple(target),avoid)
    exhausted=world.unreachable_paths.get(failure_key,())
    if origin in exhausted:
        world.path_cache[cache_key]=None
        return start
    walkable = {}
    def obstruction(point):
        if point not in walkable:
            walkable[point] = excluded(point) or blocked(world, point, radius, exit_point=target)
        return walkable[point]
    queue, cost, parent = [(0, origin)], {origin: 0}, {}
    end = None
    for _ in range(2500):
        if not queue:
            break
        _, cell = heapq.heappop(queue)
        if cell == goal:
            end = cell
            break
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nxt = cell[0] + dx, cell[1] + dy
            p = nxt[0] * spacing, nxt[1] * spacing
            # Rounding the current position can put the origin inside a barrel.
            # Connect the actual position to the first node, not the snapped
            # point; subsequent lattice edges remain unchanged.
            segment_start = start if cell == origin else (cell[0]*spacing, cell[1]*spacing)
            edge_steps = max(2, math.ceil(math.dist(segment_start, p) / 5)) if cell == origin else 2
            if obstruction(p) or any(obstruction((segment_start[0]+(p[0]-segment_start[0])*i/edge_steps,
                                                  segment_start[1]+(p[1]-segment_start[1])*i/edge_steps))
                                      for i in range(1, edge_steps)):
                continue
            g = cost[cell] + 1
            if g < cost.get(nxt, math.inf):
                cost[nxt], parent[nxt] = g, cell
                heapq.heappush(queue, (g + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]), nxt))
    if end is None:
        world.path_cache[cache_key] = None
        # Only exhausted searches prove disconnection; a search budget timeout
        # does not. Reuse failure only from nodes reached in that component.
        # The first expansion used the exact player position. Check that this
        # did not omit an edge available from its rounded lattice node.
        snapped=(origin[0]*spacing,origin[1]*spacing)
        origin_closed=obstruction(snapped) or all(
            (origin[0]+dx,origin[1]+dy) in cost
            or obstruction((snapped[0]+dx*spacing,snapped[1]+dy*spacing))
            or obstruction((snapped[0]+dx*spacing/2,snapped[1]+dy*spacing/2))
            for dx,dy in ((0,-1),(1,0),(0,1),(-1,0)))
        if not queue and origin_closed:
            if len(world.unreachable_paths)>=16:world.unreachable_paths.clear()
            world.unreachable_paths[failure_key]=frozenset(set(cost)-{origin})
        return start
    route = []
    while end != origin:
        route.append((end[0]*spacing,end[1]*spacing))
        end = parent[end]
    route.reverse()
    waypoint = route[0]
    direction = (route[0][0]-origin[0]*spacing,route[0][1]-origin[1]*spacing)
    # A single 20-unit lattice step can be nearer than the predicted movement
    # at high speed, making standing still score better than advancing. Look
    # several nodes ahead, but only across a fully checked straight segment.
    for index,point in enumerate(route[1:4],1):
        if (point[0]-route[index-1][0],point[1]-route[index-1][1]) != direction:
            break  # Preserve the next corner before looking along its new heading.
        checks = max(1,math.ceil(math.dist(start,point)/5))
        if any(obstruction((start[0]+(point[0]-start[0])*i/checks,
                            start[1]+(point[1]-start[1])*i/checks)) for i in range(1,checks+1)):
            break
        waypoint = point
    if len(world.path_cache) > 128:
        world.path_cache.clear()
    world.path_cache[cache_key] = waypoint
    return waypoint


def traversable_door(world, door):
    if door.get('target_room_type') in (10, 14, 15):
        return False
    if door.get('is_locked'):
        return (world.info.get('is_clear') is True and (has_golden_key(world) or world.inventory.get('keys', 0) > 0)
                and door.get('target_room_type') in (2, 4, 12, 24))
    return bool(door.get('is_open'))


def next_door(world, desired=None):
    doors = world.layout.get("doors", {})
    available = {str(k): v for k, v in doors.items() if traversable_door(world, v)}
    if str(desired) in available:
        return available[str(desired)]
    # BFS to a frontier through previously observed connections.
    todo = deque([(str(world.room), None)])
    seen = {str(world.room)}
    exit_route = None
    while todo:
        room, first_slot = todo.popleft()
        node = world.rooms.get(room, {})
        if node.get('floor_exit') and first_slot in available and exit_route is None:
            exit_route = available[first_slot]
        edges = node.get("doors", {})
        ordered = sorted(edges.items(), key=lambda kv: kv[1].get("target_room_type") != 4)
        for slot, door in ordered:
            if not traversable_door(world, door):
                continue
            target = str(door.get("target_room", -1))
            if int(target) < 0:
                continue
            chosen = str(slot) if first_slot is None else first_slot
            if target not in world.rooms and chosen in available:
                return available[chosen]
            if target not in seen:
                seen.add(target)
                todo.append((target, chosen))
    return exit_route


def exploration_target(world, desired=None, allow_descent=True):
    door = next_door(world, desired)
    if door:
        return (float(door['x']), float(door['y'])), True
    if allow_descent and world.info.get('is_clear') and world.info.get('room_type') == 5:
        exits = world.boss_exits
        if exits:
            return (exits[0]['x'], exits[0]['y']), True
    return xy(world.player.get('pos')), False


def firing_lane(world, start, target):
    """Ordinary tears cannot reach a target through solid rocks or walls.

    Poop can be cleared by shooting; explosive objects retain safe_shot checks.
    Spectral/other special weapons will need their actual tear flags here.
    """
    for cell in world.layout.get('grid', {}).values():
        if cell.get('collision') not in (2, 3, 4) or cell.get('type') in (12, 14):
            continue
        center = (cell['x'], cell['y'])
        if segment_distance(center, start, target) < 21:
            return False
    return True


def combat_waypoint(world, target, desired=None, flexible=False, flank_axis=None):
    route = combat_position(world, target, desired, flexible, flank_axis)
    return route[0] if route else None


def combat_position(world, target, desired=None, flexible=False, flank_axis=None):
    """Return the next walking step and its final firing position separately."""
    p = xy(world.player.get('pos'))
    desired = desired if desired is not None else spacing(world.stats)['preferred_distance']
    body=max((float(e.get('collision_radius',12)) for e in world.combat_enemies
              if math.dist(xy(e.get('pos')),target)<1),default=12)
    lateral_clearance=max(60,body+float(world.stats.get('size',12))+35)
    if flank_axis is None and min(abs(p[0]-target[0]), abs(p[1]-target[1])) < 16 and firing_lane(world, p, target):
        return None
    radii = (desired, desired*.9, desired*.8) if flexible else (desired,)
    points = [(target[0]+dx*radius, target[1]+dy*radius)
              for radius in radii for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    if flexible:
        # A long tear range does not make the room taller. Include a firing
        # position that fits between the target and each actual room boundary.
        lo, hi = xy(world.info.get('top_left')), xy(world.info.get('bottom_right'))
        margin = float(world.stats.get('size',12))+10
        limits = (hi[0]-target[0]-margin, target[0]-lo[0]-margin,
                  hi[1]-target[1]-margin, target[1]-lo[1]-margin)
        for (dx,dy), limit in zip(((1,0),(-1,0),(0,1),(0,-1)),limits):
            reach = min(desired,max(0,limit))
            if reach >= (lateral_clearance if flank_axis is not None else max(85,float(world.stats.get('range',260))*.4)):
                points.append((target[0]+dx*reach,target[1]+dy*reach))
    if flank_axis is not None:
        points=[q for q in points if abs(q[flank_axis]-target[flank_axis])>=lateral_clearance]
    points.sort(key=lambda point: math.dist(p, point))
    for point in points:
        if blocked(world, point, float(world.stats.get('size', 12))) or not firing_lane(world, point, target):
            continue
        clearance = min(desired*.7, math.dist(p,target)-8, math.dist(point,target)-8)
        avoid = (target[0],target[1],max(0,clearance)) if flexible else None
        waypoint = path_step(world, point, avoid=avoid)
        if flexible and math.dist(p, waypoint) <= 1:
            # Preferred standoff can seal a narrow passage behind a rock.
            # Retry with a range-scaled clearance; hazards still gate actions.
            clearance = min(clearance,max(85,float(world.stats.get('range',260))*.4))
            waypoint = path_step(world, point, avoid=(target[0],target[1],clearance))
        if flexible and math.dist(p, waypoint) <= 1:
            # A narrow passage may require briefly leaving the preferred range
            # band. Keep body clearance in route search; the action risk filter
            # still checks every enemy, projectile and terrain hazard.
            body = max((float(e.get('collision_radius',12)) for e in world.combat_enemies
                        if math.dist(xy(e.get('pos')),target) < 1), default=12)
            clearance = min(clearance,float(world.stats.get('size',12))+body+25)
            waypoint = path_step(world, point, avoid=(target[0],target[1],clearance))
        if math.dist(p, waypoint) > 1:
            return waypoint, point
    return None


def target_point(world, plan):
    p = xy(world.player.get("pos"))
    if world.combat_enemies:
        target = select_enemy(world, plan)
        return xy(target.get("pos")), False
    if plan.get("strategy_enabled"):
        if plan.get('shoot_target'):
            return xy(plan['shoot_target']['pos']), False
        if plan.get("navigation_target") is not None:
            return tuple(plan["navigation_target"]), bool(plan.get("navigation_contact"))
        if plan.get("strategy_choice"):
            return p, False
        if world.info.get('room_type') == 5 and world.info.get('is_clear'):
            closed = [(g['x'], g['y']) for g in world.layout.get('grid', {}).values()
                      if g.get('type') == 17 and g.get('variant', 0) == 0 and g.get('state') == 0]
            # Leave room for a closed exit's opening animation while waiting.
            # This authorizes no descent: closed doors remain absent from offers.
            if any(math.dist(p, q) < 85 for q in closed):
                for dx, dy in ((-1,0),(1,0),(0,1),(0,-1),(-1,1),(1,1)):
                    point = (p[0]+dx*95, p[1]+dy*95)
                    if (all(math.dist(point, q) >= 90 for q in closed)
                            and not blocked(world, point, float(world.stats.get('size', 12)))
                            and math.dist(path_step(world, point), p) > 1):
                        return point, False
        if plan.get("awaiting_strategy"):
            # Let Sol decide irreversible item choices, but don't halt ordinary
            # empty-room traversal while an asynchronous plan is in flight.
            if not world.strategy_ready() or any(item.get("variant") == 100 or item.get("price", 0) != 0
                                                 or item.get("options_index", 0) for item in world.pickups):
                return p, False
            if plan.get('hold_for_strategy'):
                return p, False
        elif plan.get("mode") == "explore":
            return exploration_target(world, plan.get('door_slot'), allow_descent=False)
        # Combat plan may still be pending after room clear: resume local travel.
    pickup = pickup_target(world, plan)
    if pickup is not None:
        return pickup, False
    return exploration_target(world, plan.get('door_slot'), allow_descent=not plan.get('strategy_enabled'))


def pickup_target(world, plan, basic_only=False):
    """Nearest useful free pickup with a walking approach; costs/choices stay out."""
    from .pickup_progress import blocked_choices
    excluded = set(plan.get('excluded_choices', ())) | blocked_choices(world)
    p = xy(world.player.get('pos'))
    if records(world.payload.get('BOMBS')) or world.projectiles or any(e['armed'] for e in explosives(world)):
        return None  # Let the reflex move away before resuming collection.
    pickups = []
    for item in world.pickups:
        # Basic, free resources only; choices involving costs need richer policies.
        variant = item.get("variant")
        if (basic_only or variant != 100) and not useful_pickup(world,item):
            continue
        choice = f"pickup:{item.get('id')}:{variant}:{item.get('sub_type', 0)}:{item.get('price', 0)}"
        if choice in excluded:
            continue
        if (item.get("price", 0) != 0 or item.get("wait", 0) > 0 or item.get('options_index', 0)
                or variant not in (10, 20, 30, 40, 50, 90, 100)):
            continue
        if variant == 50 and not all(world.fresh(k, 3) for k in
                ('BOMBS', 'PROJECTILES', 'FIRE_HAZARDS', 'ENEMIES', 'ROOM_INFO')):
            continue
        if variant == 10:
            subtype = item.get("sub_type", 1)
            # Full red hearts must not hide soul/black-heart pickups.
            if subtype in (1, 2, 5, 9) and world.health.get("red_hearts", 0) >= world.health.get("max_hearts", 0):
                continue
            if subtype in (3, 6, 8) and world.health.get("red_hearts", 0) + world.health.get("soul_hearts", 0) >= 24:
                continue
        if math.dist(p, xy(item.get("pos"))) < 5:
            continue
        # path_step deliberately short-circuits the last <28 units. That is
        # not a reachability proof for loot beside fire/spikes or inside an
        # obstacle; repeatedly selecting it can pin collection at the hazard.
        # Routine resources need no contact exemption: automatic_touch already
        # allows them. A coordinate exemption would also authorize an overlapping
        # unselected purchase, option, machine or floor exit at this endpoint.
        contact = xy(item.get('pos')) if not automatic_touch(world,item) else None
        from .resource_access import record as record_access
        if blocked(world, xy(item.get("pos")), float(world.stats.get('size', 12)),
                   exit_point=contact):
            record_access(world,item,'endpoint_blocked')
            continue
        if math.dist(path_step(world, xy(item.get("pos"))), p) < 1:
            record_access(world,item,'no_local_route')
            continue  # Requires terrain destruction or a different approach.
        record_access(world,item,'local_route_found')
        pickups.append(item)
    if pickups:
        closest=min(pickups, key=lambda e: math.dist(p, xy(e.get('pos'))))
        from .pickups import red_healing, red_waste, ordinary_red_health
        if (red_healing(world,closest) and ordinary_red_health(world)
                and world.health.get('red_hearts',0)+world.health.get('soul_hearts',0)>4):
            limit=math.dist(p,xy(closest['pos']))+80
            alternatives=[e for e in pickups if red_healing(world,e)
                          and math.dist(p,xy(e['pos']))<=limit
                          and (e is closest or
                               path_step(world,xy(e['pos']))==xy(e['pos']) and all(
                                   segment_distance(xy(other['pos']),p,xy(e['pos']))
                                   >float(world.stats.get('size',12))+24
                                   for other in pickups if other is not e
                                   and red_healing(world,other) and red_waste(world,other)>red_waste(world,e)))]
            closest=min(alternatives,key=lambda e:(red_waste(world,e),math.dist(p,xy(e['pos']))))
        return xy(closest['pos'])
    return None


def clearing_target(world, goal):
    """Shoot ordinary fire/poop obstructing a selected goal with no walking path.

    Explosive and bomb-only obstacles remain excluded; safe_shot checks
    incidental TNT in the tear path.
    """
    p = xy(world.player.get('pos'))
    obstacles = [xy(f.get('pos')) for f in records(world.payload.get('FIRE_HAZARDS'))
                 if f.get('type') == 'FIREPLACE' and f.get('variant') == 0
                 and not f.get('is_extinguished')]
    obstacles += [(g['x'], g['y']) for g in world.layout.get('grid', {}).values()
                  if g.get('type') == 14 and g.get('variant', 0) != 1 and g.get('collision', 0) != 0]
    obstacles = [q for q in obstacles if math.dist(q, goal) < 65 or segment_distance(q, p, goal) < 25]
    for q in sorted(obstacles, key=lambda q: math.dist(p, q)):
        waypoint = combat_waypoint(world, q)
        if waypoint is not None or (firing_lane(world, p, q)
                and min(abs(p[0]-q[0]), abs(p[1]-q[1])) < 16):
            return q
    return None


def jump_evasion_waypoint(world, enemy):
    p = xy(world.player.get('pos'))
    vx, vy = xy(enemy.get('vel'))
    speed = math.hypot(vx,vy)
    lo, hi = xy(world.info.get('top_left')), xy(world.info.get('bottom_right'))
    radius = float(world.stats.get('size',12))
    points = [(p[0]-vy/speed*distance, p[1]+vx/speed*distance)
              for distance in (120,-120,80,-80)]
    # Prefer a side with turning room. Path and full hazard checks still own
    # legality, including the unselected pickup/trade contacts on that side.
    points.sort(key=lambda q: -min(q[0]-lo[0],hi[0]-q[0],q[1]-lo[1],hi[1]-q[1]))
    q = xy(enemy.get('pos'))
    for point in points:
        if blocked(world,point,radius):
            continue
        waypoint = path_step(world,point,avoid=(*q,radius+float(enemy.get('collision_radius',12))+25))
        if math.dist(p,waypoint)>1:
            return waypoint
    return None


def candidates(world, plan):
    target, is_exit = target_point(world, plan)
    p = xy(world.player.get("pos"))
    breach = None
    # A stored terrain-clearing target must yield to an approaching enemy.
    # Keep the intent for a later safe opportunity, but do not let it disable
    # enemy spacing/flanking while a pursuer is already inside that band.
    breach_safe = nearest_distance(world, p) >= min(180, float(world.stats.get('range',260))*.72)
    intent = world.combat_breach
    if intent:
        still_present = any(g.get('type') == 12 and g.get('collision', 0) != 0
                            and (g['x'], g['y']) == tuple(intent['point'])
                            for g in world.layout.get('grid', {}).values())
        if (world.combat_enemies and still_present and world.frame-intent['frame'] < 180
                and any(e.get('id') == intent['enemy_id'] for e in world.combat_enemies)):
            if breach_safe:
                breach = tuple(intent['point'])
        else:
            world.combat_breach.clear()
    if world.combat_enemies and breach_safe and breach is None and not firing_lane(world, p, target):
        barrels = [(g['x'], g['y']) for g in world.layout.get('grid', {}).values()
                   if g.get('type') == 12 and g.get('collision', 0) != 0
                   and math.dist((g['x'], g['y']), target) < 100]
        for point in sorted(barrels, key=lambda q: math.dist(p, q)):
            if firing_lane(world, p, point) or combat_waypoint(world, point, desired=210) is not None:
                breach = point
                enemy = min(world.combat_enemies, key=lambda e: math.dist(xy(e.get('pos')), target))
                world.combat_breach.update(point=point, frame=world.frame, enemy_id=enemy.get('id'))
                break
    contact = target if is_exit else None
    from .scavenging import targets as scavenge_targets
    selected_source = next((t for t in scavenge_targets(world)
                            if t['key'] == plan.get('shoot_target', {}).get('key')), None)
    # A shootable pile's center is intentionally not walkable. Searching for a
    # walking path into it before computing a firing position exhausted A* in
    # large rooms on every moving frame, delaying even neutral input updates.
    waypoint = target if world.combat_enemies or selected_source else path_step(world,target)
    clearing = (xy(selected_source['pos']) if selected_source else
                clearing_target(world, target) if not world.combat_enemies and world.info.get('is_clear')
                and math.dist(target, p) > 28 and math.dist(waypoint, p) < 1 else None)
    shooting = bool(world.combat_enemies or clearing)
    target = breach or clearing or target
    band = spacing(world.stats, plan, select_enemy(world, plan))
    standoff = 210 if breach else band['preferred_distance'] if world.combat_enemies else 150
    if selected_source:
        standoff = min(150, max(60, float(world.stats.get('range', 260))*.7))
    from .combat import edge_flank_axis
    flank_axis=edge_flank_axis(world,target) if world.combat_enemies and not breach else None
    firing_route = combat_position(world, target, desired=standoff,
                            flexible=bool(world.combat_enemies and not breach),flank_axis=flank_axis) if shooting else None
    flank = firing_route[0] if firing_route else None
    explosive_source=selected_source if selected_source and selected_source.get('explosive') else None
    if clearing and explosive_source is None:
        from .hazards import poop_explosions
        if poop_explosions(world) and any(g.get('type')==14 and g.get('collision',0)!=0
                and (g['x'],g['y'])==tuple(clearing) for g in world.layout.get('grid',{}).values()):
            explosive_source={'pos':dict(zip(('x','y'),clearing))}
    if explosive_source:
        from .scavenging import explosive_firing_spot
        spot=explosive_firing_spot(world,explosive_source)
        if spot is None:
            shooting=False;flank=None;target=p;waypoint=p
        else:
            flank=path_step(world,spot) if math.dist(p,spot)>1 else p
    if flank and world.combat_enemies and not breach:
        # A wall can make the preferred ring unreachable. Permit the shorter
        # firing lane while retaining a range-scaled contact safety margin.
        body = float((select_enemy(world,plan) or {}).get('collision_radius',12))
        passage_distance = max(float(world.stats.get('size',12))+body+25,
                               math.dist(firing_route[1],target)-8)
        band['min_distance'] = min(band['min_distance'], passage_distance)
    if shooting:
        waypoint = flank or target
    from .combat import incoming_jump
    jumper = incoming_jump(world)
    evasion = jump_evasion_waypoint(world,jumper) if jumper else None
    if evasion:
        # Movement can evade a boss while shooting a different vulnerable add.
        # Airborne bosses are often not selectable firing targets.
        waypoint = evasion
    if not plan.get("strategy_enabled") and not world.combat_enemies and any(
            math.dist(target, xy(item.get("pos"))) < 1 for item in world.pickups):
        contact = target
    danger = {m: risk(world, m, contact, maneuver=bool(flank)) for m in MOVES}
    minimum = min(danger.values())
    options = []
    for move in MOVES:
        if danger[move] > max(5, minimum + 5):
            continue
        dest = trajectory(world, move, 4)[-1]
        if shooting and flank is None and evasion is None:
            # Keep a firing lane while maintaining distance.
            alignment = min(abs(dest[0] - target[0]), abs(dest[1] - target[1]))
            progress = (-band_cost(dest, target, band) if world.combat_enemies and not breach
                        else -abs(math.dist(dest, target) - standoff)) - alignment * 0.5
        else:
            progress = -math.dist(dest, waypoint)
            # A checked route can temporarily leave the firing band to pass
            # terrain. Follow that route while moving; apply the band when
            # holding a firing lane. Penalizing the intermediate step's range
            # either stalled these turns or required globally shrinking it.
        if world.combat_enemies:
            # Preserve turning room. This is a soft preference; the hazard
            # filter still permits wall-adjacent escape when it is safest.
            lo, hi = xy(world.info.get('top_left')), xy(world.info.get('bottom_right'))
            wall_gaps = (min(dest[0]-lo[0], hi[0]-dest[0]), min(dest[1]-lo[1], hi[1]-dest[1]))
            progress -= .3*sum(max(0, 48-gap) for gap in wall_gaps)
            from .tactics import bias
            progress += bias(world,dest,target,plan.get('tactical_posture','range'))
        vulnerable = not world.combat_enemies or (select_enemy(world, plan) or {}).get('is_vulnerable', True)
        for shoot in SHOTS if shooting and (vulnerable or breach or clearing) else ((0, 0),):
            if not safe_shot(world, move, shoot):
                continue
            aim = 0.0
            if shooting and shoot != (0, 0):
                delta = target[0] - dest[0], target[1] - dest[1]
                forward = delta[0] * shoot[0] + delta[1] * shoot[1]
                cross = abs(delta[0] * shoot[1] - delta[1] * shoot[0])
                # A tear fired 100 units beside an enemy is not useful merely
                # because its direction has a positive dot product. That reward
                # outweighed walking to the firing lane and caused static misses.
                aim = (.2 + 20 * math.exp(-.5 * (cross / 22) ** 2)
                       if 0 < forward < float(world.stats.get('range', 260)) else -1)
                if not firing_lane(world, dest, target):
                    aim = -5  # Relocate instead of shooting an indestructible rock forever.
            action = Action(move, shoot)
            options.append({"action": action, "risk": round(danger[move], 3),
                            "score": progress + aim - danger[move] * 5,
                            "description": (f"move={move}, shoot={shoot}, predicted_risk={danger[move]:.2f}, "
                                            f"goal={target}, waypoint={waypoint}, predicted_position={tuple(round(v, 1) for v in dest)}, "
                                            f"progress_score={progress + aim:.1f}"
                                            + (f", evade observed jump id={jumper.get('id')}" if evasion else '')
                                            + (f", range={band['observed_range']}, distance_band={band['min_distance']}..{band['max_distance']}, "
                                               f"target_distance={math.dist(dest,target):.1f}, nearest_enemy_distance={nearest_distance(world,dest):.1f}"
                                               if world.combat_enemies else '')
                                            + (", shoot TNT from safe range to reach protected enemy" if breach else ""))})
    return sorted(options, key=lambda o: o["score"], reverse=True), contact


def shield(world, action, fallback, exit_point):
    chosen = risk(world, action.move, exit_point)
    baseline = risk(world, fallback.move, exit_point)
    if chosen > max(5, baseline + 5):
        return fallback, True
    if baseline <= chosen+3 and crowding_override(world, action, fallback):
        return fallback, True
    if not safe_shot(world, action.move, action.shoot):
        return Action(action.move, (0, 0)), True
    return action, False
