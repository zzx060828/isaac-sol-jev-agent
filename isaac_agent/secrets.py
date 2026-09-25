"""Observed-map hypotheses and one-bomb secret-room search contracts."""
import math

from .state import records, xy
from .bomb_budget import budget, golden_spawn_observed

OFFSETS = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}


def neighbor(index, slot):
    if not 0 <= index < 169:
        return None
    dx, dy = OFFSETS[slot]
    x, y = index % 13+dx, index//13+dy
    return y*13+x if 0 <= x < 13 and 0 <= y < 13 else None


def wall_observations(world):
    """Describe geometry, never inspect the engine's undiscovered room table."""
    if world.info.get('room_shape') != 1:
        return {}
    from .control import physical_collision
    result = {}
    for slot, wall in world.layout.get('wall_slots', {}).items():
        slot = int(slot)
        if slot not in OFFSETS:
            continue
        dx, dy = OFFSETS[slot]
        q = xy(wall)
        approach = (q[0]-dx*40, q[1]-dy*40)
        # Treat pits as obstacles even for flying characters when assessing
        # generation evidence. Do not alter the player's actual flight state.
        clear = not physical_collision(world, approach, 15)
        for cell in world.layout.get('grid', {}).values():
            if cell.get('type') in (8,9,25) or cell.get('collision') == 1:
                if math.dist(approach, (cell['x'],cell['y'])) < 42:
                    clear = False
        result[str(slot)] = {'x': q[0], 'y': q[1], 'approach_clear': clear,
                             'has_door': str(slot) in world.layout.get('doors', {})}
    return result


def ordinary_bombs(world):
    # Unknown fuse/placement transformations cannot inherit a normal-bomb
    # escape contract (e.g. Dr. Fetus, Epic Fetus, fast/troll/rocket/scatter bombs).
    altered = {52,106,125,137,140,149,256,257,353,366,367,483,517,563,583,614,646}
    short_fuse = any(int(world.inventory.get(k,0)) & 32767 == 133 for k in ('trinket_0','trinket_1'))
    return not short_fuse and not altered.intersection(int(k) for k in world.inventory.get('collectibles', {}))


def ready(world):
    return (world.capabilities.get('secret_bomb_protocol') == 1
            and world.capabilities.get('repentance_plus') is False
            and world.run_memory is not None and world.info.get('room_type') in (1,2,4,6)
            and world.info.get('room_shape') == 1 and world.info.get('is_clear') is True
            and not world.combat_enemies and budget(world, 1) is not None
            and ordinary_bombs(world)
            and world.health.get('red_hearts',0)+world.health.get('soul_hearts',0) >= 4
            and all(world.fresh(k,6) for k in ('ROOM_INFO','ROOM_LAYOUT','PLAYER_INVENTORY',
                                              'PLAYER_HEALTH','BOMBS','FIRE_HAZARDS','PROJECTILES')))


def placement(world, slot, wall):
    from .control import blocked, path_step
    dx, dy = OFFSETS[slot]; q = xy(wall)
    spot = (q[0]-dx*40, q[1]-dy*40)
    escape = (spot[0]-dx*190, spot[1]-dy*190)
    if records(world.payload.get('BOMBS')) or world.projectiles:
        return None
    # Require a straight safe retreat before spending a bomb. The room can
    # still contain distant resources; no chain reactions along this route.
    for i in range(13):
        p = (spot[0]+(escape[0]-spot[0])*i/12, spot[1]+(escape[1]-spot[1])*i/12)
        if blocked(world, p, 15):
            return None
        if any(g.get('type') in (5,12) and g.get('collision',0)
               and math.dist(p,(g['x'],g['y'])) < 170 for g in world.layout.get('grid',{}).values()):
            return None
    player = xy(world.player.get('pos'))
    if math.dist(player,spot) > 15 and math.dist(path_step(world,spot),player) < 1:
        return None
    return spot, escape


def offers(world):
    if not ready(world):
        return []
    attempts = world.run_memory.secret_attempts
    if sum(k.startswith(world.level+':') for k in attempts) >= 2:
        return []  # At most two exploratory bombs per floor in this policy.
    rooms = {int(k): v for k,v in world.rooms.items() if str(k).isdigit()}
    occupied = set(rooms)
    for idx, data in rooms.items():
        for d in data.get('doors',{}).values():
            if isinstance(d.get('target_room'),int) and d['target_room'] >= 0:
                occupied.add(d['target_room'])
        if data.get('shape',1) != 1:
            # Conservative exclusion for uncertain multi-cell footprints.
            occupied.update(i for i in (idx+1,idx+13,idx+14) if i < 169)
    walls = wall_observations(world)
    result = []
    for s, wall in walls.items():
        slot = int(s); candidate = neighbor(world.room,slot)
        marker = f'{world.level}:{candidate}'
        if (candidate is None or candidate in occupied or marker in attempts
                or wall['has_door'] or not wall['approach_clear']):
            continue
        adjacent = [neighbor(candidate,d) for d in OFFSETS]
        known = [i for i in adjacent if i in occupied]
        verified = []
        for idx in known:
            data = rooms.get(idx,{})
            back = next((str(d) for d in OFFSETS if neighbor(idx,d) == candidate), None)
            w = data.get('walls',{}).get(back,{})
            if (data.get('shape') == 1 and data.get('type') not in (5,7,8,14,15)
                    and w.get('approach_clear') and not w.get('has_door')):
                verified.append(idx)
        # Don't spend based on an approach in another room we haven't checked.
        if len(known) != len(verified):
            continue
        kind = 'regular_secret_guess'
        if len(verified) < 2:
            near_boss = any(v.get('type') == 5 and
                            abs(idx%13-candidate%13)+abs(idx//13-candidate//13) <= 2
                            for idx,v in rooms.items())
            if (len(verified) != 1 or rooms[verified[0]].get('type') != 1
                    or not near_boss or budget(world, 1, reserve=2) is None):
                continue
            kind = 'super_secret_guess'
        route = placement(world,slot,wall)
        if not route:
            continue
        spot, escape = route
        funding = budget(world, 1)
        suffix = ':golden' if funding['bomb_payment'] == 'golden' else ''
        result.append({'choice': f'secret_bomb:{candidate}:{slot}'+suffix, 'kind':'secret_bomb',
                       'entity':{'pos':dict(zip(('x','y'),spot))}, 'slot':s,
                       'candidate_cell':candidate, 'attempt_key':marker,
                       'wall':{'x':wall['x'],'y':wall['y']}, 'escape':escape,
                       **funding,
                       'evidence':{'hypothesis':kind,'observed_adjacent_rooms':verified,
                                   'unknown_adjacent_cells':[i for i in adjacent if i is not None and i not in occupied],
                                   'certainty':'heuristic, not a guaranteed room; unseen map cells remain unknown'},
                       'effect':'Place one normal-fuse bomb, retreat, verify a newly open secret door, then enter. '
                                'bomb_cost is ordinary stock consumed; bomb_limit bounds placements even with golden payment. '
                                'No automatic retry; at most two exploratory placements per floor. Normal payment retains one bomb.'})
    return sorted(result,key=lambda o:0 if len(o['evidence']['observed_adjacent_rooms']) >= 3
                  else 1 if len(o['evidence']['observed_adjacent_rooms']) == 2 else 2)[:3]


def settled_for_bomb(job,world,spot,execution):
    """Brake for several sensor updates so queued movement cannot spoil a pulse."""
    if math.dist(xy(world.player.get('pos')),spot)>7:
        job.settle_started=None
        return False
    execution['bomb_brake']=True
    if getattr(job,'settle_started',None) is None:job.settle_started=world.frame
    return (world.frame-job.settle_started>=4 and world.fresh('PLAYER_POSITION',1)
            and math.hypot(*xy(world.player.get('vel')))<.2)


class SecretSearch:
    def __init__(self,world,offer):
        self.offer, self.token = offer,world.token
        self.started = world.frame
        self.fired = None
        self.bombs_before = world.inventory['bombs']
        self.spawned = False
        self.done = None
        self.phase = 'approach'
        self.request_id = offer['attempt_key']

    def record(self,world,status):
        entry = {'status':status,'room':self.token[2],'candidate':self.offer['candidate_cell'],
                 'slot':self.offer['slot'],'frame':world.frame}
        world.run_memory.secret_attempts[self.offer['attempt_key']] = entry
        world.run_memory.add(world,'secret_search',entry)

    def update(self,world,plan,log):
        p=xy(world.player.get('pos')); o=self.offer; spot=xy(o['entity']['pos'])
        execution=dict(plan,navigation_contact=False)
        def stop(reason):
            self.done=reason
            execution.update(navigation_target=p,navigation_contact=False)
            if self.fired is not None:self.record(world,reason)
            return execution
        if world.token != self.token:
            return stop('entered_secret' if world.info.get('room_type') in (7,8) else 'room_changed')
        if self.fired is None:
            current=next((v for v in offers(world) if v['choice']==o['choice']),None)
            if not current or world.frame-self.started > 240:
                return stop('preflight_invalid')
            execution['navigation_target']=spot
            if settled_for_bomb(self,world,spot,execution):
                self.record(world,'attempt_pending')  # Persist before any input can reach the game.
                self.fired=world.frame; self.phase='retreat'
                self.bombs_before=world.inventory['bombs']
                execution['secret_bomb_request']={'id':self.request_id,'slot':o['slot'],
                    'payment':o.get('bomb_payment','normal'),
                    'bombs_before':self.bombs_before,'spot':o['entity']['pos']}
                log.write('secret_bomb_request',frame=world.frame,choice=o['choice'],contract=o)
            return execution
        execution['navigation_target']=tuple(o['escape'])
        if (golden_spawn_observed(world,self.fired,self.request_id,o['entity']['pos'])
                if o.get('bomb_payment') == 'golden' else
                world.inventory.get('bombs',self.bombs_before) < self.bombs_before or records(world.payload.get('BOMBS'))):
            self.spawned=True
        elapsed=world.frame-self.fired
        if records(world.payload.get('BOMBS')) or elapsed < 120:
            if elapsed > 240:return stop('bomb_clearance_unconfirmed')
            return execution
        if not self.spawned:
            return stop('bomb_unconfirmed_no_retry')
        if world.combat_enemies or not world.info.get('is_clear'):
            return stop('combat_interrupt')
        if not all(world.fresh(k,2) and world.sampled[k]>self.fired
                   for k in ('ROOM_LAYOUT','BOMBS','PLAYER_INVENTORY')):
            return stop('state_confirmation_timeout') if elapsed>360 else execution
        door=world.layout.get('doors',{}).get(o['slot'])
        if door and door.get('is_open') and door.get('target_room_type') in (7,8):
            if self.phase != 'enter':
                self.record(world,'entrance_open'); self.phase='enter'
                log.write('secret_entrance_open',frame=world.frame,door=door)
            execution.update(navigation_target=(door['x'],door['y']),navigation_contact=True)
            if elapsed > 300:return stop('entry_timeout')
            return execution
        return stop('no_entrance_observed')
