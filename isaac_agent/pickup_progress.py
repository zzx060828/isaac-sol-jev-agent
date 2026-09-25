"""Bound local collection attempts without turning failure into acquisition."""
import math

from .pickups import routine
from .review import choice_id
from .state import xy


def abilities(world):
    return (tuple((k, world.stats.get(k)) for k in ('can_fly', 'size', 'speed', 'player_type')),
            tuple(sorted(world.inventory.get('collectibles', {}).items())))


def blocked_choices(world, room=None):
    room = world.room if room is None else room
    return {r['choice'] for r in world.pickup_failures.values()
            if r['scope'] == world.token[:2] and r['room'] == room and r['abilities'] == abilities(world)}


def ready(world):
    from .hazards import explosives
    from .resources import safe_room
    from .control import risk
    return (world.control_mode != 'MANUAL' and world.strategy_ready() and safe_room(world)
            and world.fresh('PLAYER_STATS', 6)
            and all(world.fresh(k, 3) for k in ('PLAYER_POSITION', 'PLAYER_HEALTH',
                                               'BOMBS', 'ENEMIES', 'PROJECTILES', 'FIRE_HAZARDS'))
            and not any(e['armed'] for e in explosives(world)) and risk(world, (0, 0)) < 1)


class PickupProgress:
    # SocketBridge updates run at 30 Hz. Only consecutive safe collection
    # intervals count; wall time, combat and missing samples never count.
    STAGNANT_FRAMES = 120
    MAX_FRAMES = 600

    def __init__(self):
        self.attempts = {}
        self.previous = None
        self.geometry = None
        self.context = None

    def observe(self, world, log):
        from .control import walkability_signature
        if not self.previous and not world.pickup_failures and not any(routine(p) for p in world.pickups):
            self.geometry = self.context = None
            self.attempts.clear()
            return
        if not ready(world):
            self.previous = None
            return
        self.geometry = walkability_signature(world)
        context = (self.geometry, abilities(world))
        if context != self.context:
            self.attempts.clear()
            self.previous = None
            self.context = context
        present = {choice_id(p): p for p in world.pickups if routine(p)}
        for key, record in list(world.pickup_failures.items()):
            reason = None
            if record['scope'] != world.token[:2]:
                reason = 'floor_changed'
            elif record['abilities'] != abilities(world):
                reason = 'movement_or_build_changed'
            elif record['room'] == world.room:
                p = present.get(record['choice'])
                if p is None:
                    reason = 'pickup_form_gone'
                elif (self.geometry != record['geometry'] or p.get('wait', 0) > 0
                      or math.dist(xy(p.get('pos')), record['target']) > 40):
                    reason = 'approach_conditions_changed'
            if reason:
                del world.pickup_failures[key]
                log.write('local_pickup_failure_expired', frame=world.frame, room=record['room'],
                          choice=record['choice'], reason=reason)
        self.attempts = {k: v for k, v in self.attempts.items() if k in present}
        previous, self.previous = self.previous, None
        if previous is None:
            return
        choice, frame = previous
        p = present.get(choice)
        delta = world.frame - frame
        if p is None or not 0 < delta <= 6 or p.get('wait', 0) > 0:
            return
        pos = xy(world.player.get('pos'))
        distance = math.dist(pos, xy(p.get('pos')))
        cell = tuple(math.floor(v / 20) for v in pos)
        attempt = self.attempts[choice]
        attempt['elapsed'] += delta
        attempt['stagnant'] += delta
        if distance < attempt['best'] - 8 or cell not in attempt['cells']:
            attempt['stagnant'] = 0
            attempt['best'] = min(distance, attempt['best'])
            attempt['cells'].add(cell)
        reason = ('no_approach_progress' if attempt['stagnant'] >= self.STAGNANT_FRAMES else
                  'collection_time_limit' if attempt['elapsed'] >= self.MAX_FRAMES else None)
        if reason:
            record = dict(scope=world.token[:2], room=world.room, choice=choice,
                          geometry=self.geometry, abilities=abilities(world), target=xy(p.get('pos')),
                          frame=world.frame, reason=reason, collection_frames=attempt['elapsed'])
            world.pickup_failures[(*world.token, choice)] = record
            log.write('local_pickup_failed', frame=world.frame, room=world.room, choice=choice,
                      reason=reason, collection_frames=attempt['elapsed'], position=pos,
                      target=record['target'], result='not_collected; defer_unchanged_approach')
            del self.attempts[choice]

    def follow(self, world, target):
        self.previous = None
        if target is None or not ready(world) or self.geometry is None:
            return
        p = next((p for p in world.pickups if routine(p) and xy(p.get('pos')) == tuple(target)), None)
        if p is None:
            return
        choice = choice_id(p)
        if choice in blocked_choices(world):
            return
        pos = xy(world.player.get('pos'))
        self.attempts.setdefault(choice, dict(elapsed=0, stagnant=0,
            best=math.dist(pos, tuple(target)), cells={tuple(math.floor(v / 20) for v in pos)}))
        self.previous = choice, world.frame
