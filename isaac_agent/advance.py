"""Bounded advance planning while the independent tactical loop keeps moving."""
import asyncio
from copy import deepcopy
import json
import math
import time

CHECKS = {'heal_before_boss','recharge_active','preserve_key','preserve_bomb',
          'compare_deal','review_secret_candidates','collect_known_value','avoid_unnecessary_backtracking'}
# Recorded four-room cleared corridor took only 12.9 seconds, shorter than
# recent 30-43 second preparation replies. Start farther out; bounded refresh
# still requires real approach/build change and at most two calls per floor.
BOSS_PREP_HOPS = 12


def build_key(world):
    inv = world.inventory
    return json.dumps({'level': world.level, 'epoch': world.epoch,
        'items': inv.get('collectibles', {}),
        'active_ids': {k: v.get('item') for k,v in inv.get('active_items', {}).items()},
        'trinkets': [inv.get('trinket_0'),inv.get('trinket_1')],
        'character': world.stats.get('player_type')}, sort_keys=True)


def build_ready(world):
    return all(world.fresh(k,6) for k in ('PLAYER_INVENTORY','PLAYER_STATS'))


def validate_preparation(value):
    if not isinstance(value, dict):
        raise ValueError('Preparation must be an object')
    objective = value.get('objective')
    distance = value.get('boss_range_fraction')
    focus = value.get('boss_focus')
    checks = value.get('next_checks', [])
    if not isinstance(objective,str) or not objective.strip():
        raise ValueError('Missing preparation objective')
    if isinstance(distance,bool) or not isinstance(distance,(float,int)) or not math.isfinite(distance):
        raise ValueError('Invalid preparation range')
    if focus not in ('balanced','dangerous_adds','spawners_first'):
        raise ValueError('Invalid preparation focus')
    if not isinstance(checks,list) or any(not isinstance(c,str) or c not in CHECKS for c in checks):
        raise ValueError('Unknown preparation check')
    # Allowlist output fields: never copy model-invented actions, enemy IDs,
    # spending contracts or coordinates into the executor.
    return {'objective': objective[:300], 'boss_range_fraction': max(.72,min(.88,distance)),
            'boss_focus': focus, 'next_checks': list(dict.fromkeys(checks))[:6],
            'resource_reasoning': str(value.get('resource_reasoning',''))[:600],
            'uncertainties': str(value.get('uncertainties',''))[:400]}


class AdvancePlanner:
    def __init__(self, models, log):
        self.models, self.log = models, log
        self.task = None
        self.floor = None
        self.attempts = {}
        self.attempt_count = 0
        self.deferred = None
        self.cache = None
        self.last_started = -float('inf')
        self.retry_at = 0
        self.quota_exhausted = False
        self.next_check_at = 0

    def current(self, world, now=None):
        now = time.monotonic() if now is None else now
        c = self.cache
        if not c:
            return None
        identity = json.loads(c['key'])
        reason = ('run ended' if world.dead else 'run or floor changed'
                  if (identity['epoch'],identity['level']) != (world.epoch,world.level)
                  else 'expired' if now-c['completed_at'] > 180 else
                  'build changed' if build_ready(world) and c['key'] != build_key(world) else None)
        if reason:
            self.log.write('advance_invalidated', reason=reason, frame=world.frame, room=world.room)
            self.cache = None
            return None
        if not build_ready(world) or not world.fresh('PLAYER_HEALTH',6):
            if not c.get('awaiting_state'):
                self.log.write('advance_waiting_for_state',frame=world.frame,room=world.room,
                               source_frame=c['frame'],reason='fresh build and health required before use')
            c['awaiting_state'] = True
            return None  # Keep the reply; missing room-entry channels are not changed equipment.
        if c.pop('awaiting_state',False):
            self.log.write('advance_revalidated',frame=world.frame,room=world.room,source_frame=c['frame'])
        return {'purpose': c['purpose'], 'source_room': c['room'], 'source_frame': c['frame'],
                'age_seconds': round(max(0,now-c['completed_at']),3),
                'plan': deepcopy(c['plan']),
                'resources_changed': c['resources'] != self.resources(world),
                'scope': 'Advisory. Current health, inventory, enemies and legal contracts override this preparation.'}

    @staticmethod
    def resources(world):
        return {'health': {k:world.health.get(k) for k in ('red_hearts','soul_hearts','max_hearts')},
                'inventory': {k:world.inventory.get(k) for k in ('coins','keys','golden_key','bombs','golden_bomb','pill_0','card_0')},
                'active': deepcopy(world.inventory.get('active_items',{}))}

    def maybe_start(self, world, route, foreground_pending, now=None):
        now = time.monotonic() if now is None else now
        if (not hasattr(self.models,'prepare') or self.models.mode != 'hybrid'
                or self.task is not None and not self.task.done()):
            return
        if self.floor != (world.epoch,world.level):
            self.floor = (world.epoch,world.level)
            self.attempts.clear(); self.cache = None; self.attempt_count = 0
            self.deferred = None
        if (foreground_pending or route not in ('routine','tactical','collect_resources','journey')
                or world.dead or now < self.retry_at or now-self.last_started < 45 or self.attempt_count >= 2
                or not build_ready(world)
                or not all(world.fresh(k,6) for k in ('ROOM_INFO','ROOM_LAYOUT','PLAYER_HEALTH'))):
            return
        if now < self.next_check_at:
            return
        self.next_check_at = now+1  # Do not rebuild the floor graph on every movement tick.
        from .floor_map import floor_map
        graph = floor_map(world)
        bosses = [n for n in graph['nodes']+graph['unvisited_observed_rooms']
                  if n.get('type') == 5 and n.get('clear_last_seen') is not True]
        reachable = [n for n in bosses if isinstance(n.get('observed_hops'),int)]
        nearest = min(reachable,key=lambda n:n['observed_hops']) if reachable else None
        if bosses and world.info.get('room_type') != 5 and (
                nearest is None or nearest['observed_hops'] > BOSS_PREP_HOPS):
            reason = 'boss too distant' if nearest else 'boss route unknown'
            marker = (reason, nearest['id'] if nearest else None)
            if self.deferred != marker:
                self.log.write('advance_deferred', reason=reason, frame=world.frame, room=world.room,
                               nearest_boss=nearest, max_observed_hops=BOSS_PREP_HOPS)
                self.deferred = marker
            return
        purpose = 'boss_preparation' if bosses and world.info.get('room_type') != 5 else (
            'floor_review' if graph['coverage']['no_known_frontier'] and len(graph['nodes']) >= 3 else None)
        # Some observed runs enter the boss without ever observing its door.
        # Once exploration reaches an ordinary fight, prepare from the build
        # even with an incomplete map. This is one event-driven attempt, not
        # a request on every room/kill or a claim to know the missing boss.
        unlocated = (not bosses and world.info.get('room_type') == 1
                     and bool(world.combat_enemies) and len(graph['nodes']) >= 3)
        if not purpose and unlocated:
            purpose = 'boss_preparation'
        if not purpose:
            marker = ('no preparation opportunity', world.room)
            if self.deferred != marker:
                self.log.write('advance_deferred', reason=marker[0], frame=world.frame,
                               room=world.room, known_rooms=len(graph['nodes']),
                               boss_observed=bool(bosses))
                self.deferred = marker
            return
        key = build_key(world)
        signature = (key,purpose)
        hops = nearest['observed_hops'] if purpose == 'boss_preparation' and nearest else None
        # A timer alone never renews a preparation. A second attempt needs a
        # changed build, or an expired/failed plan plus actual progress to boss.
        if self.current(world,now) is not None:
            return
        if signature in self.attempts and (hops is None or self.attempts[signature] is not None
                                          and hops >= self.attempts[signature]):
            return
        from .knowledge import expert_knowledge, held_item_mechanics, held_trinket_mechanics, held_consumable_mechanics
        from .charge import battery_target
        observation = {'purpose': purpose, 'frame':world.frame,'room':world.room,'level':world.level,
            'preparation_window': {'boss_room': nearest['id'] if nearest else None,
                                   'observed_hops': hops, 'max_observed_hops': BOSS_PREP_HOPS},
            'trigger': ('explored_build_with_unlocated_boss' if unlocated and purpose == 'boss_preparation'
                        else 'observed_boss_approach' if purpose == 'boss_preparation' else 'known_frontier_exhausted'),
            'stats':deepcopy(world.stats), 'resources':self.resources(world), 'floor_map':graph,
            'held_item_mechanics':held_item_mechanics(world.inventory),
            'held_trinket_mechanics':held_trinket_mechanics(world.inventory),
            'held_consumable_mechanics':held_consumable_mechanics(world.inventory),
            'ordinary_battery_target':battery_target(world),
            'expert_knowledge':expert_knowledge(world),
            'run_history':world.run_memory.ledger() if world.run_memory else {},
            'boss_identity':'Unknown unless already observed; do not predict the next boss as fact.',
            'contract':'No spending, route execution, item use or enemy selection is authorized by this request.'}
        self.attempts[signature] = hops; self.attempt_count += 1; self.last_started = now
        self.deferred = None
        self.log.write('advance_request',observation=observation)
        self.task = asyncio.create_task(self._run(world,observation,key,purpose))

    async def _run(self,world,observation,key,purpose):
        try:
            plan, meta = await self.models.prepare(observation)
            plan = validate_preparation(plan)
            identity = json.loads(key)
            if (world.dead or (identity['epoch'],identity['level']) != (world.epoch,world.level)
                    or build_ready(world) and build_key(world) != key):
                self.log.write('advance_discarded',reason='run, floor or build changed',**meta)
                return
            self.cache = {'key':key,'purpose':purpose,'room':observation['room'],'frame':observation['frame'],
                          'resources':observation['resources'],'plan':plan,'completed_at':time.monotonic()}
            waiting = not build_ready(world) or not world.fresh('PLAYER_HEALTH',6)
            if waiting:
                self.cache['awaiting_state'] = True
            self.log.write('advance_waiting_for_state' if waiting else 'advance_ready',
                           source_frame=observation['frame'],ready_frame=world.frame,
                           source_room=observation['room'],ready_room=world.room,plan=plan,**meta)
        except Exception as exc:
            if getattr(exc,'code',None) in ('insufficient_quota','billing_hard_limit_reached'):
                self.quota_exhausted = True
            self.retry_at = time.monotonic()+max(60,getattr(exc,'retry_after',0))
            self.log.write('advance_error',error=type(exc).__name__+': '+str(exc)[:200],
                           **getattr(exc,'metadata',{}))

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task,return_exceptions=True)
