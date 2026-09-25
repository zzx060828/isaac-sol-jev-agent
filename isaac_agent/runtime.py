"""Asynchronous planner, controller, and direct SocketBridge TCP server."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import hashlib
from pathlib import Path
import time

from .control import Action, candidates, shield
from .state import World
from .models import jev_enabled
from .post_pickup import PostPickup
from .resources import (ResourcePolicy, bind_resource_intent, resource_preference,
                        resource_options, consumable_review_key)
from .strategy import StrategyExecutor, offers
from .planning import planning_context, planning_reasons, planning_observation, enrich_planning
from .advance import AdvancePlanner
from .fast_policy import lane, fast_offers, ordinary_pickup
from .combat_progress import CombatProgress
from .review import fingerprint as review_fingerprint
from .journey import Journey
from .chests import ChestObserver
from .pickup_progress import PickupProgress, blocked_choices
from . import review
from .diagnostics import gate_facts, decision_trace


class Log:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.file = (self.directory / "events.jsonl").open("w", encoding="utf-8")
        self.counts = {}
        package = Path(__file__).parent
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(package.iterdir()) if p.suffix in ('.py', '.json')}
        (self.directory / 'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def record_trial(self, config, execution):
        from .trial_metadata import build_manifest
        sources = json.loads((self.directory / 'source_manifest.json').read_text())
        manifest = build_manifest(sources, config, execution)
        (self.directory / 'trial_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        self.write('trial_manifest', fingerprint=manifest['fingerprint'], schema_version=1)

    def write(self, event, **data):
        self.counts[event] = self.counts.get(event, 0) + 1
        self.file.write(json.dumps({"time": time.time(), "event": event, **data}, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
        (self.directory / "summary.json").write_text(json.dumps(self.counts, indent=2) + "\n")


class Controller:
    def __init__(self, models, log):
        self.models, self.log = models, log
        self.plan = {"objective": "Clear enemies, collect free resources, explore the floor", "door_slot": None}
        self.plan_task = self.decision_task = None
        self.last_plan = self.last_decision = -100.0
        self.plan_revision = -1
        self.plan_context = None
        self.plan_pending = None
        self.room_started = 0
        self.reviewed_consumables = None
        self.post_pickup = PostPickup()
        self.combat_progress = CombatProgress()
        self.pickup_review = None
        self.pickup_reviews = {}
        self.journey = Journey()
        self.fast_deferred = set()
        self.resource_settle_until = -1
        self.transaction_finished_frame = None
        self.chests = ChestObserver()
        self.pickup_progress = PickupProgress()
        self.route = None
        self.tactical_retry_at = 0.0
        self.low_confidence_streak = 0
        self.token = None
        self.action = Action()
        self.action_until = -1
        self.tactic = None
        self.seq = 0
        self.resources = ResourcePolicy()
        self.strategy = StrategyExecutor()
        self.advance = AdvancePlanner(models, log)
        self.preparation_applied_token = None
        self.execution_plan = self.plan
        self.plan_failures = 0
        self.quota_exhausted = False
        self.plan_retry_at = 0.0
        directory = getattr(log, 'directory', None)
        memory_dir = Path(directory) if getattr(log, 'isolated_memory', False) else Path(directory).parent if directory else None
        self.backoff_path = memory_dir / 'planner-backoff.json' if memory_dir else None
        if self.backoff_path and self.backoff_path.exists():
            try:
                saved = json.loads(self.backoff_path.read_text())
                remaining = min(300, max(0, float(saved['retry_at_unix'])-time.time()))
                self.plan_retry_at = time.monotonic()+remaining
                self.plan_failures = min(32, max(0, int(saved.get('failures', 0)))) if remaining else 0
            except (ValueError, KeyError, OSError, TypeError):
                pass

    def save_plan_backoff(self, delay):
        if self.backoff_path:
            temp = self.backoff_path.with_suffix('.tmp')
            temp.write_text(json.dumps({'retry_at_unix': time.time()+delay,
                                        'failures': self.plan_failures}))
            temp.replace(self.backoff_path)

    def reset_actions(self, world):
        self.token = world.token
        self.action_until = -1
        self.tactic = None
        self.plan = {"objective": "Survive and clear the current room", "door_slot": None}
        self.plan.update(strategy_enabled=self.models.mode == "hybrid", awaiting_strategy=True)
        self.plan_revision = -1
        self.plan_context = None
        self.plan_pending = None
        self.room_started = world.frame
        self.combat_progress = CombatProgress()
        self.pickup_review = self.pickup_reviews.get(world.token)
        self.pickup_reviews={k:v for k,v in self.pickup_reviews.items() if k[:2]==world.token[:2]}
        self.fast_deferred = set()
        self.resource_settle_until = -1
        self.transaction_finished_frame = None
        self.chests = ChestObserver()
        self.pickup_progress = PickupProgress()
        self.route = None
        self.tactical_retry_at = 0.0
        self.low_confidence_streak = 0
        self.expired_decision_streak = 0
        self.preparation_applied_token = None

    async def _plan(self, world, observation, token, revision):
        try:
            memory_candidates = observation.pop('memory_candidates', [])
            observation['retrieved_memories'] = memory_candidates[:4]
            if memory_candidates and jev_enabled(self.models, 'memory') and hasattr(self.models, 'retrieve_memories'):
                try:
                    selected, routing = await self.models.retrieve_memories(observation, memory_candidates)
                    observation['retrieved_memories'] = selected
                    self.log.write('memory_retrieval', source_frame=observation['frame'], **routing)
                except Exception as exc:
                    self.log.write('memory_retrieval_fallback', error=type(exc).__name__)
            if world.token != token or world.dead:
                self.log.write('plan_skipped', reason='state_changed_during_memory_retrieval')
                return
            plan, meta = await self.models.plan(observation)
            self.plan_failures = 0
            self.plan_retry_at = 0.0
            self.save_plan_backoff(0)
            damaged = world.revision != revision and world.last_damage_frame >= observation["frame"]
            changed = ("strategy_signature" in observation
                       and observation["strategy_signature"] != world.strategy_signature())
            # Damage does not invalidate a still-present combat target. Reject
            # stale trades/healing as before, but do not starve tactical updates
            # throughout a difficult boss fight by discarding every response.
            combat_plan = (plan.get('mode') in ('combat', 'survive') and bool(observation.get('enemies'))
                           and not plan.get('strategy_choice') and not plan.get('resource_action'))
            combat_target_valid = combat_plan and bool(world.combat_enemies)
            reason = ('room_changed' if world.token != token else
                      'combat_finished' if combat_plan and not world.combat_enemies else
                      'damage_changed_trade' if damaged and not combat_target_valid else
                      'strategy_state_changed' if changed else
                      'goal_already_finished' if plan.get('strategy_choice') in self.strategy.finished else None)
            if not reason and plan.get('strategy_choice'):
                # A plan is never authorization for an offer that disappeared
                # or became unaffordable while independent resources were taken.
                if plan['strategy_choice'] not in {o['choice'] for o in offers(world)}:
                    reason='selected_offer_unavailable'
            if reason:
                self.log.write("plan_discarded", reason=reason, plan=plan,
                               source_frame=observation['frame'], applied_frame=world.frame,
                               source_room=observation.get('room'), current_room=world.room, **meta)
                self.plan_revision = -1
                if world.token == token and reason != 'combat_finished':
                    self.plan_pending = 'invalidated_plan'
                return
            if combat_target_valid and not any(str(e.get('id')) == str(plan.get('target_enemy'))
                                               for e in world.combat_enemies):
                plan['target_enemy'] = None
            bind_resource_intent(plan, observation)
            resource_preference(world, plan, self.resources, self.log)
            self.plan = plan
            self.post_pickup.arm(world, observation, plan)
            self.reviewed_consumables = consumable_review_key(
                observation.get('level'), observation.get('inventory', {}),
                observation.get('resource_options', []))
            self.plan.update(strategy_enabled=True, awaiting_strategy=False, goal_source='sol')
            declined=plan.get('declined_pickups',[])
            review.remember(world,observation,plan,self.log)
            if declined and observation.get('pickup_review_context') == review_fingerprint(world):
                present={o['choice'] for o in offers(world) if o['kind']=='pickup'}
                choices=set(declined)&present
                choices.discard(plan.get('strategy_choice'))
                self.pickup_review={'key':observation['pickup_review_context'],'choices':choices}
                self.pickup_reviews[world.token]=self.pickup_review
                self.log.write('pickup_review',frame=world.frame,declined=sorted(choices))
            if self.plan_context and self.plan_context['room'] == world.token:
                # Filtering a just-accepted decline is our own plan result, not
                # a fresh external choice. Consume only those removals in the
                # request baseline; preserve concurrent gains/new offers so
                # their next comparison can still trigger a needed review.
                accepted=set(declined) & (review.active(world) |
                    (self.pickup_review['choices'] if self.pickup_review else set()))
                consumed=accepted & set(self.plan_context['choices'])
                if consumed:
                    self.plan_context={**self.plan_context,'choices':tuple(
                        choice for choice in self.plan_context['choices'] if choice not in consumed)}
                    self.log.write('planning_declines_consumed',frame=world.frame,choices=sorted(consumed))
            if plan.get('route_contract'):
                self.journey.start(world,plan['route_contract'])
                self.log.write('journey_start',frame=world.frame,contract=plan['route_contract'])
            elif plan.get('mode')=='explore':
                self.journey.contract=None
            if world.run_memory:
                world.run_memory.record_decision(world,plan)
            if combat_target_valid:
                # New strategic guidance can make a previously ambiguous
                # local question useful again. Do not cancel an in-flight one.
                self.low_confidence_streak = 0
                self.tactical_retry_at = min(self.tactical_retry_at,time.monotonic()+1)
            self.log.write("plan", plan=plan, **meta)
        except Exception as exc:
            self.plan_revision = -1
            self.plan_pending = 'api_retry'
            self.plan_failures += 1
            status = getattr(exc, 'status', 0)
            base = 30 if status == 429 else 10
            delay = min(300, base * 2 ** min(self.plan_failures-1, 5))
            delay = max(delay, getattr(exc, 'retry_after', 0))
            if getattr(exc, 'code', None) in ('insufficient_quota', 'billing_hard_limit_reached'):
                delay = 300
                self.quota_exhausted = True
            self.plan_retry_at = time.monotonic() + delay
            self.save_plan_backoff(delay)
            self.log.write("plan_error", error=type(exc).__name__ + ": " + str(exc)[:200],
                           retry_after_seconds=delay, **getattr(exc,'metadata',{}))

    async def _fast_goal(self, world, observation, token, choices):
        started = time.monotonic()
        requested = choices[0]['choice']
        try:
            choice, meta = await self.models.choose_goal(observation, choices)
            current = {o['choice'] for o in fast_offers(world, self.strategy.finished)}
            supplied = {o['choice'] for o in choices}
            selected = choice if choice in supplied else requested
            reason = ('room_changed' if world.token != token else
                      'player_dead' if world.dead else
                      'manual_control' if world.control_mode == 'MANUAL' else
                      'state_not_ready' if not world.ready(time.monotonic()) else
                      'reply_expired' if time.monotonic()-started > 3 else
                      'choice_not_supplied' if choice not in supplied | {'skip','defer'} else
                      'selected_offer_unavailable' if selected not in current else
                      'inventory_changed' if planning_context(world)['items'] != observation['fast_items'] else None)
            if reason:
                if world.token == token:
                    self.fast_deferred.add(requested)
                self.log.write('fast_goal_discarded', choice=choice, reason=reason,
                               source_frame=observation.get('frame'), applied_frame=world.frame, **meta)
                return
            if choice == 'defer' or meta.get('confidence', 0) < .6:
                self.fast_deferred.add(requested)
                self.log.write('fast_goal_deferred', choice=choice,
                               reason='model_deferred' if choice == 'defer' else 'confidence_below_threshold',
                               confidence_threshold=.6, **meta)
                return
            if choice == 'skip':
                self.strategy.finished.add(requested)
                self.plan.update(strategy_choice=None, awaiting_strategy=False)
            else:
                if choice not in current:
                    raise ValueError('JEV selected an unavailable goal')
                self.plan = {'objective': 'Execute reviewed routine pickup or room sweep', 'mode': 'collect',
                             'strategy_choice': choice, 'strategy_enabled': True, 'awaiting_strategy': False,
                             'goal_source': 'jev'}
            self.log.write('fast_goal', choice=choice, source_frame=observation['frame'],
                           applied_frame=world.frame, **meta)
        except Exception as exc:
            if world.token == token:
                self.fast_deferred.add(requested)
            self.log.write('fast_goal_error', error=type(exc).__name__ + ': ' + str(exc)[:200])

    async def _tactic(self,world,observation,token,targets):
        started=time.monotonic()
        try:
            intent,meta=await self.models.choose_tactic(observation,dict(self.execution_plan),targets)
            elapsed=time.monotonic()-started
            source=observation['frame']
            legal={t['choice'] for t in targets}
            from .tactics import POSTURES
            live={str(e.get('id')) for e in world.combat_enemies}
            reason=('room_changed' if world.token!=token else
                    'state_unavailable' if world.dead or world.control_mode=='MANUAL' or not world.ready(time.monotonic()) else
                    'combat_finished' if not live else
                    'intent_expired' if elapsed>4 or world.frame-source>120 else
                    'invalid_choice' if str(intent.get('target_enemy')) not in legal or intent.get('posture') not in POSTURES else
                    'target_gone' if str(intent['target_enemy']) not in live else None)
            confidence=meta.get('confidence_by_part',{})
            parts={k for k in ('target','posture') if confidence.get(k,0)>=.45}
            if reason or not parts:
                if world.token==token:self.tactical_retry_at=time.monotonic()+2
                self.log.write('tactic_discarded',reason=reason or 'low_confidence',
                               source_frame=source,applied_frame=world.frame,**meta)
                return
            self.tactic={'token':token,'target_enemy':intent['target_enemy'],
                         'posture':intent['posture'] if 'posture' in parts else 'range',
                         'parts':parts,'until':time.monotonic()+3,'frame_until':world.frame+90}
            self.tactical_retry_at=0
            self.log.write('tactic',intent=intent,applied_parts=sorted(parts),source_frame=source,
                           applied_frame=world.frame,lease_seconds=3,**meta)
        except Exception as exc:
            if world.token==token:self.tactical_retry_at=time.monotonic()+2
            self.log.write('tactic_error',error=type(exc).__name__+': '+str(exc)[:200])

    async def _decide(self, world, observation, token, frame, options):
        started = time.monotonic()
        try:
            guidance = dict(self.execution_plan)
            if observation.get('combat_positioning'):
                guidance['target_enemy'] = observation['combat_positioning']['selected_enemy']
            action, meta = await self.models.decide(observation, guidance, options)
            # Revalidate delayed responses against live state. The network
            # budget is separate from the short input lease sent to the game.
            elapsed=time.monotonic()-started
            reason=('room_changed' if world.token != token else
                    'dead' if world.dead else
                    'manual_control' if world.control_mode == 'MANUAL' else
                    'state_stale' if not world.ready(time.monotonic()) else
                    'combat_changed' if bool(world.combat_enemies) != bool(observation.get('enemies')) else
                    'observation_expired' if world.frame-frame > 45 or elapsed > 1.5 else None)
            if reason:
                delay=0
                if reason=='observation_expired':
                    self.expired_decision_streak=getattr(self,'expired_decision_streak',0)+1
                    delay=min(12,2**min(self.expired_decision_streak,4))
                    self.tactical_retry_at=time.monotonic()+delay
                self.log.write('decision_discarded',reason=reason,source_frame=frame,
                               applied_frame=world.frame,elapsed_ms=round(elapsed*1000),
                               retry_after_seconds=delay,**meta)
            elif meta.get("confidence", 0) < 0.45:
                self.low_confidence_streak = getattr(self, 'low_confidence_streak', 0)+1
                delay = min(12, 2**min(self.low_confidence_streak, 4))
                self.tactical_retry_at = time.monotonic()+delay
                self.log.write("decision_discarded", reason="low confidence", retry_after_seconds=delay, **meta)
            else:
                current_options, _ = candidates(world, self.execution_plan)
                if action.id not in {o["action"].id for o in current_options}:
                    self.tactical_retry_at = time.monotonic()+1
                    self.log.write("decision_discarded", reason="action no longer legal", **meta)
                    return
                if not world.combat_enemies and current_options:
                    score = next(o["score"] for o in current_options if o["action"].id == action.id)
                    if score < current_options[0]["score"] - 6:
                        self.log.write("decision_discarded", reason="delayed move no longer advances current goal", **meta)
                        return
                self.low_confidence_streak = 0
                self.expired_decision_streak = 0
                self.tactical_retry_at = 0.0
                self.action, self.action_until = action, world.frame + 6
                self.log.write("decision", action=action.id, source_frame=frame,
                               applied_frame=world.frame, revalidated=True, **meta)
        except Exception as exc:
            self.tactical_retry_at = time.monotonic()+max(2,getattr(exc,'retry_after',0))
            self.log.write("decision_error", error=type(exc).__name__ + ": " + str(exc)[:200])

    def step(self, world, now=None):
        step_started=time.perf_counter()
        now = time.monotonic() if now is None else now
        self.seq += 1
        if self.token != world.token:
            self.reset_actions(world)
        if not world.ready(now) or world.control_mode == "MANUAL":
            self.pickup_progress.previous = None
            facts = gate_facts(world, now)
            # Missing/paused samples can repeat at the same frame. Log changes
            # and at most once per second while a gate remains closed.
            signature = (world.token, world.control_mode, facts['world_ready'], world.dead, int(now))
            if signature != getattr(self, '_logged_gate', None):
                self.log.write('control_gate', frame=world.frame, room=world.room, facts=facts,
                               command_action='m0s0')
                self._logged_gate = signature
            return Action().wire(world, self.seq)
        if getattr(self, '_logged_gate', None) is not None:
            self.log.write('control_gate_opened', frame=world.frame, room=world.room)
            self._logged_gate = None
        self.pickup_progress.observe(world, self.log)
        escalation=self.combat_progress.update(world)
        review.refresh(world,self.log)
        chest_settling = self.chests.update(world, self.log)
        if escalation and self.route != 'sol':
            self.log.write('combat_review_needed',frame=world.frame,reason=escalation)
        if self.pickup_review and self.pickup_review['key'] != review_fingerprint(world):
            self.log.write('pickup_review_invalidated',frame=world.frame,reason='context_changed')
            self.pickup_review=None
            self.pickup_reviews.pop(world.token,None)
            self.plan_pending='pickup_review_context_changed'
        preparation = self.advance.current(world,now)
        if (preparation and preparation['purpose'] == 'boss_preparation'
                and world.info.get('room_type') == 5 and world.combat_enemies
                and all(world.fresh(k,6) for k in ('PLAYER_STATS','PLAYER_INVENTORY','PLAYER_HEALTH'))
                and self.preparation_applied_token != world.token
                and self.plan.get('goal_source') != 'sol' and not self.plan.get('strategy_choice')):
            prepared = preparation['plan']
            self.plan.update(mode='combat', objective=prepared['objective'],
                combat_distance=float(world.stats.get('range',260))*prepared['boss_range_fraction'],
                boss_focus=prepared['boss_focus'], goal_source='advance')
            self.preparation_applied_token = world.token
            self.log.write('advance_applied',frame=world.frame,room=world.room,
                           frames_since_room_entry=world.frame-self.room_started,
                           scope='combat spacing and focus only',preparation=preparation)
        completed = len(self.strategy.finished)
        prior_offer = self.strategy.started_offer or {}
        self.execution_plan = self.strategy.update(world, self.plan, self.log)
        excluded=(self.strategy.finished | review.active(world) | blocked_choices(world)
                  | (self.pickup_review['choices'] if self.pickup_review else set()))
        self.execution_plan['excluded_choices']=sorted(excluded)
        if len(self.strategy.finished) > completed:
            if (prior_offer.get('kind') == 'scavenge' or prior_offer.get('kind') == 'pickup'
                    and (prior_offer.get('entity', {}).get('price', 0) not in (0, -1000)
                         or prior_offer.get('entity', {}).get('variant') in (100, 350))):
                # Scavenge loot, payment counters and mutually exclusive pedestals can settle
                # after acquisition. Wait for fresh inventory and pickup samples
                # before planning from the remaining offers.
                self.transaction_finished_frame = world.frame
            self.plan.update(strategy_choice=None, awaiting_strategy=True)
            # Replan when another meaningful choice remains. Finishing a free
            # coin/bomb pickup alone does not need a new high-level request.
            if planning_context(world, excluded)['choices']:
                self.plan_pending = 'goal_resolved'
            else:
                self.plan['awaiting_strategy'] = False
        followup = self.post_pickup.update(world, self.log)
        if followup:
            self.reviewed_consumables = consumable_review_key(
                world.level, world.inventory, (), require_option=False)
            if followup['action'] == 'use':
                self.plan['resource_action'] = followup['kind']
                bind_resource_intent(self.plan, world.observation())
        context = planning_context(world, excluded)
        route, goals = lane(world, self.room_started, excluded, self.fast_deferred, bool(escalation))
        if route == 'collect_resources':
            self.resource_settle_until = world.frame + 18
        elif (world.frame < self.resource_settle_until and not world.combat_enemies
              and world.info.get('is_clear') and not self.strategy.choice and not context['access']
              and all(o['kind'] in ('descend','scavenge') or ordinary_pickup(o)
                      for o in offers(world) if o['choice'] not in excluded)):
            # Resource counters update before the pickup's disappearance animation.
            # Let the next sensor samples settle before asking Sol about the floor.
            route, goals = 'settle_resources', []
        if route == 'routine' and (context['charged_item'] or context['access']):
            route = 'sol'
        if route == 'routine' and world.strategy_ready() and not world.combat_enemies:
            key = consumable_review_key(world.level, world.inventory, resource_options(world))
            if key is not None and key != self.reviewed_consumables:
                route = 'sol'
                self.plan_pending = self.plan_pending or 'held_consumable_review'
        settled = self.transaction_finished_frame
        if settled is not None and not world.combat_enemies:
            if (world.frame >= settled+6 and all(world.sampled.get(k,-1) > settled
                    for k in ('PLAYER_INVENTORY','PLAYER_HEALTH','PICKUPS'))):
                self.transaction_finished_frame = None
            else:
                route, goals = 'settle_transaction', []
        had_journey=self.journey.contract is not None
        if chest_settling and not world.combat_enemies:
            route, goals = 'settle_chest', []
        if (self.post_pickup.pending and self.post_pickup.pending['disappeared'] is not None
                and not world.combat_enemies):
            route, goals = 'settle_consumable', []
        journey_door=self.journey.step(world,self.log)
        if had_journey and self.journey.contract is None and self.journey.end_reason!='arrived':
            self.plan.update(door_slot=None,awaiting_strategy=True)
            self.execution_plan.update(door_slot=None,awaiting_strategy=True)
            self.plan_pending='route_changed'
            route='sol'
        if (journey_door is not None and route in ('routine','sol') and not context['choices']
                and not context['charged_item'] and not world.combat_enemies
                and self.plan_pending != 'held_consumable_review'
                and world.info.get('is_clear') and not self.plan.get('strategy_choice')):
            route='journey'
            self.plan.update(mode='explore',door_slot=journey_door,awaiting_strategy=False,goal_source='journey')
            self.execution_plan.update(mode='explore',door_slot=journey_door,awaiting_strategy=False,goal_source='journey')
        if route == 'fast' and not jev_enabled(self.models, 'goal'):
            # Identical fallback in tactics/off arms; no new pickup whitelist.
            route, goals = 'sol', []
        if route != self.route:
            self.log.write('decision_route', frame=world.frame, room=world.room, lane=route)
            if route == 'sol':
                self.plan_pending = self.plan_pending or 'strategic_decision'
            if route in ('sol', 'fast') and not self.plan.get('strategy_choice'):
                self.plan['awaiting_strategy'] = True
                self.execution_plan['awaiting_strategy'] = True
            self.route = route
        local_pickup_target = (self.execution_plan.get('navigation_target')
                               if self.execution_plan.get('scavenge_resource_collection') else None)
        if self.models.mode == 'hybrid':
            self.plan['hold_for_strategy'] = route in ('sol', 'fast')
            self.execution_plan['hold_for_strategy'] = self.plan['hold_for_strategy']
            if route not in ('sol', 'fast'):
                self.plan_context, self.plan_pending = context, None
                self.plan['awaiting_strategy'] = False
                self.execution_plan['awaiting_strategy'] = False
            if route == 'collect_resources' and not self.strategy.choice:
                from .control import pickup_target
                target = pickup_target(world, self.execution_plan, basic_only=True)
                local_pickup_target = target
                self.execution_plan.update(navigation_target=target, navigation_contact=False)
            elif route in ('settle_resources', 'settle_transaction', 'settle_chest', 'settle_pickups', 'settle_consumable'):
                from .state import xy
                self.execution_plan.pop('shoot_target', None)
                self.execution_plan.pop('scavenge_resource_collection', None)
                local_pickup_target = None
                self.execution_plan.update(navigation_target=xy(world.player.get('pos')), navigation_contact=False)
            elif (route in ('sol','fast') and not self.strategy.choice
                  and not self.plan.get('strategy_choice') and world.info.get('is_clear')
                  and not world.combat_enemies and world.strategy_ready()
                  and all(world.fresh(k,3) for k in ('BOMBS','PROJECTILES','FIRE_HAZARDS','PLAYER_HEALTH'))):
                # Slow reasoning and independent free collection can overlap.
                # Keep the Sol/JEV lane and its pending request; don't replace
                # active trades, bombing, donation or a selected pedestal.
                from .control import pickup_target
                target=pickup_target(world,self.execution_plan,basic_only=True)
                if target is not None:
                    local_pickup_target = target
                    self.execution_plan.update(navigation_target=target,navigation_contact=False,
                                               parallel_resource_collection=True)
        self.pickup_progress.follow(world, local_pickup_target)
        if self.tactic:
            live={str(e.get('id')) for e in world.combat_enemies}
            if (self.tactic['token']!=world.token or now>self.tactic['until']
                    or world.frame>self.tactic['frame_until'] or str(self.tactic['target_enemy']) not in live):
                self.log.write('tactic_end',frame=world.frame,reason='lease_or_target_changed')
                self.tactic=None
            else:
                self.execution_plan['tactical_posture']=self.tactic['posture']
                if 'target' in self.tactic['parts']:
                    self.execution_plan['strategic_enemy']=self.execution_plan.get('target_enemy')
                    self.execution_plan['target_enemy']=self.tactic['target_enemy']
        options, exit_point = candidates(world, self.execution_plan)
        fallback = options[0]["action"] if options else Action()
        slow_idle = self.plan_task is None or self.plan_task.done()
        goal_active = bool(self.plan.get('strategy_choice'))
        if self.models.mode == "hybrid" and route == 'sol' and not goal_active and world.strategy_ready() and slow_idle:
            reasons = planning_reasons(self.plan_context, context)
            if self.plan_pending:
                reasons.append(self.plan_pending)
            if reasons and now >= self.plan_retry_at and now - self.last_plan >= 3:
                self.last_plan, self.plan_revision = now, world.revision
                self.plan_context, self.plan_pending = context, None
                obs = enrich_planning(world,world.observation(),preparation)
                obs['pickup_review_context']=review_fingerprint(world)
                obs['pickup_review_basis']=review.basis(world)
                obs['combat_review_reason']=escalation
                obs['navigation_commitment']=self.journey.contract
                # Local collection owns these contacts. Sol still sees raw
                # resources/inventory, but must not spend a reply reopening a
                # chest or collecting a coin that local movement already took.
                obs["strategy_offers"] = [o for o in obs["strategy_offers"]
                                          if o["choice"] not in excluded and not ordinary_pickup(o)]
                self.log.write("plan_request", observation=obs, triggers=reasons)
                self.plan_task = asyncio.create_task(self._plan(world, obs, world.token, world.revision))
        if now >= self.plan_retry_at:
            self.advance.maybe_start(world,route,
                self.plan_task is not None and not self.plan_task.done(),now)
        if (self.models.mode == 'hybrid' and route == 'fast' and not goal_active and slow_idle
                and (self.decision_task is None or self.decision_task.done()) and now-self.last_decision >= .25):
            self.last_decision = now
            obs = world.observation()
            obs['fast_items'] = context['items']
            self.log.write('fast_goal_request', observation=obs, choices=goals)
            self.decision_task = asyncio.create_task(self._fast_goal(world, obs, world.token, goals))
        if (self.models.mode in ("jev", "hybrid") and jev_enabled(self.models, 'tactic')
                and (self.decision_task is None or self.decision_task.done())):
            # In hybrid mode a reviewed route/pickup already determines clear
            # room motion. Remote choice remains for combat and typed goals;
            # local navigation and hazard checks continue at sensor frequency.
            needs_tactical_choice = bool(world.combat_enemies) or (
                self.models.mode == 'jev' and fallback != Action())
            use_intent=(self.models.mode=='hybrid' and bool(world.combat_enemies)
                        and hasattr(self.models,'choose_tactic'))
            interval = 2.0 if use_intent else 1.0 if world.combat_enemies else .5
            if (now - self.last_decision >= interval and now >= getattr(self, 'tactical_retry_at', 0)
                    and options and needs_tactical_choice):
                self.last_decision = now
                obs = world.observation()
                # One best shot per safe move: at most nine choices. The local
                # controller recalculates shots and safety between replies.
                tactical_options, moves = [], set()
                for option in options:
                    if option['action'].move not in moves:
                        tactical_options.append(option)
                        moves.add(option['action'].move)
                from .combat import context as combat_context
                obs['combat_positioning'] = combat_context(world, self.execution_plan)
                if use_intent:
                    from .tactics import targets
                    obs['tactical_sensor_age_updates']={k:world.frame-world.sampled[k] if k in world.sampled else None
                        for k in ('PLAYER_POSITION','ENEMIES','PROJECTILES','BOMBS','FIRE_HAZARDS','ROOM_LAYOUT')}
                    choices=targets(world,self.execution_plan)
                    self.log.write('tactic_request',observation=obs,targets=choices,
                                   min_interval_seconds=interval,plan=self.execution_plan)
                    self.decision_task=asyncio.create_task(self._tactic(world,obs,world.token,choices))
                else:
                    self.log.write("decision_request", observation=obs, plan=self.execution_plan,
                                   min_interval_seconds=interval,
                                   options=[o["action"].id for o in tactical_options])
                    self.decision_task = asyncio.create_task(self._decide(world, obs, world.token, world.frame, tactical_options))
        intended = self.action if world.frame <= self.action_until else fallback
        if not world.combat_enemies and options:
            score = next((o["score"] for o in options if o["action"] == intended), float("-inf"))
            if score < options[0]["score"] - 6:
                intended = fallback
        actual, override = shield(world, intended, fallback, exit_point)
        rock_request = self.execution_plan.get('rock_bomb_request')
        bomb_request = rock_request or self.execution_plan.get('secret_bomb_request')
        if bomb_request or self.execution_plan.get('bomb_brake'):
            actual = Action()
        command = actual.wire(world, self.seq)
        if bomb_request:
            command.update(move={'x':0,'y':0},shoot={'x':0,'y':0},use_bomb=True,
                           agent_bomb_id=bomb_request['id'],
                           agent_bomb_payment=bomb_request.get('payment','normal'),
                           agent_bombs_before=bomb_request['bombs_before'])
            if rock_request:
                command.update(agent_bomb_kind=rock_request.get('kind','rock'),agent_bomb_grid=rock_request['grid_index'],
                               agent_bomb_grid_type=rock_request['grid_type'],agent_bomb_spot=rock_request['spot'])
            else:
                command['agent_bomb_slot']=int(bomb_request['slot'])
        recovering_donation = (self.strategy.donation is not None and not world.combat_enemies
                               and world.health.get('red_hearts', 0) >= 4)
        preferred_resource = resource_preference(world, self.plan, self.resources, self.log)
        resource = None if (recovering_donation or self.strategy.secret is not None
                            or self.strategy.rock is not None) else self.resources.choose(
                                world, preferred_resource, keep=self.post_pickup.keep_kinds(world))
        if resource:
            command["use_" + resource["kind"]] = True
            # Let the inventory/health sensors catch up before a clear-room
            # comparison. Otherwise an immediate heal can make its own Sol
            # observation obsolete a few updates after the request starts.
            self.transaction_finished_frame=world.frame
            # Bind the pulse to the observed inventory so it cannot consume a
            # newly picked-up replacement while a command was in transit.
            command["agent_resource_kind"] = resource["kind"]
            command["agent_resource_id"] = resource["id"]
            self.log.write("resource_use", frame=world.frame, room=world.room, **resource)
            if self.plan.get("resource_action") == resource["kind"]:
                self.plan["resource_action"] = None  # A model use request is one action, not a held intent.
                self.plan.pop("resource_authorization", None)
        if world.frame % 30 == 0:
            self.log.write('control_state', frame=world.frame, room=world.room,
                           execution_plan=self.execution_plan, breach=dict(world.combat_breach),
                           decision_trace=decision_trace(self,world,now,local_pickup_target,
                                                         fallback,actual,override,command),
                           candidates=[{'action': o['action'].id, 'risk': o['risk'],
                                        'score': o['score'], 'description': o['description']}
                                       for o in options[:3]])
        self.log.write("input", frame=world.frame, room=world.room, action=actual.id,
                       control_step_ms=round((time.perf_counter()-step_started)*1000,3),
                       source="rock_contract" if rock_request else "secret_contract" if bomb_request else "jev" if intended == self.action and world.frame <= self.action_until else "local_with_jev_tactic" if self.tactic else "local",
                       intended=intended.id, shield_override=override,
                       bomb_request_id=bomb_request['id'] if bomb_request else None,
                       resource=resource["kind"] if resource else None)
        return command

    async def close(self):
        await self.advance.close()
        tasks = [t for t in (self.plan_task, self.decision_task) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def floor_finished(world, stage):
    parts = str(world.level).split(':')
    return (len(parts) == 3 and parts[1] == str(stage)
            and world.fresh('ROOM_INFO', 8) and world.fresh('ROOM_LAYOUT', 8)
            and world.info.get('room_type') == 5 and world.info.get('is_clear')
            and not world.combat_enemies
            and any(g.get('type') == 17 and g.get('variant', 0) == 0 and g.get('state') == 1
                    for g in world.layout.get('grid', {}).values()))


class BridgeServer:
    def __init__(self, models, log, observe_only=False, memory_path=None, stop_after_floor=None):
        self.models, self.log, self.observe_only = models, log, observe_only
        self.world = World()
        from .memory import RunMemory
        memory_dir = log.directory if getattr(log, 'isolated_memory', False) else log.directory.parent if isinstance(getattr(log, 'directory', None), Path) else None
        memory_file = memory_dir / 'run-memory.json' if memory_dir else None
        self.world.run_memory = RunMemory(memory_file)
        self.controller = Controller(models, log)
        self.writer = None
        self.client_tasks = set()
        self.closing = False
        self.last_frame = -1
        self.skipped_frames = 0
        self.run_ended = asyncio.Event()
        self.stop_after_floor = stop_after_floor
        self.memory_path = memory_path
        self.map_memory = {}
        if memory_path and memory_path.is_file():
            try:
                self.map_memory = json.loads(memory_path.read_text())
            except (ValueError, OSError):
                pass
        self.map_signature = None
        self.environment_signature = None
        from .evaluation import StartGuard
        registration = getattr(log, 'evaluation', None)
        self.evaluation_guard = StartGuard(registration) if registration else None

    def remember_map(self):
        if not self.memory_path or not self.world.level:
            return
        # Only merge rooms previously observed in this exact seed/stage/type.
        current = self.world.rooms
        saved = self.map_memory.get(self.world.level, {})
        self.world.rooms = {**saved, **current}
        signature = json.dumps(self.world.rooms, sort_keys=True)
        if signature == self.map_signature:
            return
        self.map_signature = signature
        self.map_memory[self.world.level] = json.loads(signature)
        self.map_memory = dict(list(self.map_memory.items())[-24:])
        temporary = self.memory_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.map_memory))
        temporary.replace(self.memory_path)

    async def send(self, payload):
        if self.writer is not None:
            self.writer.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
            await asyncio.wait_for(self.writer.drain(), 0.2)

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.client_tasks.add(task)
        if self.writer is not None:
            writer.close()
            self.client_tasks.discard(task)
            return
        self.writer = writer
        self.world.reset()
        self.last_frame = -1
        self.log.write("connected")
        print("SocketBridge connected; waiting for agent capability handshake", flush=True)
        configured = False
        try:
            while not self.closing:
                try:
                    line = await asyncio.wait_for(reader.readline(), 0.5)
                except asyncio.TimeoutError:
                    # Lua watchdog releases inputs if the game stops sending frames.
                    continue
                if not line:
                    break
                try:
                    msg = json.loads(line)
                    self.log.write("bridge", message=msg)
                    if msg.get("type") == "EVENT" and msg.get("event") in ("PLAYER_DEATH", "GAME_END"):
                        self.run_ended.set()
                    updated = self.world.ingest(msg)
                    self.world.run_memory.observe(self.world, msg)
                    if not updated:
                        continue
                    from .trial_metadata import observed_environment, fingerprint
                    environment = observed_environment(self.world, msg.get('agent', {}))
                    environment_signature = (self.world.epoch, fingerprint(environment))
                    if environment_signature != self.environment_signature:
                        self.environment_signature = environment_signature
                        self.log.write('environment_observed', frame=self.world.frame, **environment)
                    if self.evaluation_guard:
                        already_accepted = self.evaluation_guard.accepted
                        try:
                            if not self.evaluation_guard.check(self.world):
                                continue
                        except RuntimeError as exc:
                            self.log.write('evaluation_rejected', reason=str(exc), frame=self.world.frame)
                            self.run_ended.set()
                            raise
                        if not already_accepted:
                            self.log.write('evaluation_start_verified', frame=self.world.frame,
                                           **self.evaluation_guard.registration)
                    if self.stop_after_floor is not None and floor_finished(self.world, self.stop_after_floor):
                        if not self.run_ended.is_set():
                            self.log.write('floor_complete', level=self.world.level, room=self.world.room,
                                           frame=self.world.frame, health=self.world.health)
                        self.run_ended.set()
                        await self.send(Action().wire(self.world, self.controller.seq + 1))
                        # Keep state and the connection alive until serve() sends
                        # the targeted pause. Disconnecting here resets World
                        # and races the pause-before-release cleanup.
                        continue
                    # Catch up on already buffered snapshots before calculating
                    # inputs. Calculating every old frame after a slow A* search
                    # can keep every command past its six-frame lease forever.
                    # StreamReader has no public complete-line queue interface.
                    if b'\n' in reader._buffer:
                        self.skipped_frames += 1
                        if self.skipped_frames % 60 == 1:
                            self.log.write('control_backlog', skipped_frames=self.skipped_frames)
                        continue
                    self.remember_map()
                    if not configured:
                        if self.world.capabilities.get("protocol") != 1:
                            raise RuntimeError("Install the generated SocketBridge_AstraJev mod (watchdog handshake missing)")
                        mode = "MANUAL" if self.observe_only else "FORCE_AI"
                        await self.send({"command": "SET_CONTROL_MODE", "params": {"mode": mode}})
                        configured = True
                        print(f"Bridge ready: {mode}, room {self.world.room}, level {self.world.level}", flush=True)
                    if self.world.frame != self.last_frame and not self.observe_only:
                        self.last_frame = self.world.frame
                        await self.send(self.controller.step(self.world))
                except (ValueError, KeyError, TypeError) as exc:
                    self.log.write("invalid_frame", error=type(exc).__name__)
                    await self.send(Action().wire(self.world, self.controller.seq + 1))
        except (ConnectionError, asyncio.TimeoutError, RuntimeError) as exc:
            self.log.write("bridge_error", error=str(exc)[:200])
            print(f"Bridge: {exc}", flush=True)
        finally:
            await self.controller.close()
            self.controller = Controller(self.models, self.log)
            with suppress(ConnectionError, asyncio.TimeoutError):
                await self.send({"command": "SET_CONTROL_MODE", "params": {"mode": "MANUAL"}})
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self.writer = None
            self.world.reset()
            self.client_tasks.discard(task)
            self.log.write("disconnected")

    async def close(self):
        # A completed read can race with wait_for cancellation on Python 3.11.
        # The explicit condition prevents the client loop from starting another
        # read after shutdown; its finally block still sends MANUAL and closes.
        self.closing = True
        tasks = list(self.client_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.controller.close()


async def pause_game_before_release(bridge, log):
    """Keep live control running until a window-targeted pause has been sent."""
    if not bridge.world.ready(time.monotonic()):
        return
    from .platform_tools import windows_helper
    script = Path(__file__).resolve().parents[1] / 'scripts/background_window_windows.py'
    started = time.monotonic()
    try:
        command = windows_helper(script, 'key', '--key', 'Escape', '--hold-ms', '50')
        process = await asyncio.create_subprocess_exec(*command,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    except (OSError, ValueError) as exc:
        log.write('game_pause', exit_code=-1, error=type(exc).__name__,
                  reason='Windows pause helper unavailable; input release continues')
        return
    try:
        code = await asyncio.wait_for(process.wait(), 5)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        code = -1
    log.write('game_pause', exit_code=code, latency_ms=round((time.monotonic()-started)*1000),
              reason='before_control_release')


async def serve(models, log, host, port, seconds=None, observe_only=False, until_run_end=False, stop_after_floor=None, pause_on_stop=False, stop_file=None):
    memory_dir = log.directory if getattr(log, 'isolated_memory', False) else log.directory.parent if isinstance(log.directory, Path) else None
    memory = memory_dir / 'navigation-memory.json' if memory_dir else None
    bridge = BridgeServer(models, log, observe_only, memory, stop_after_floor)
    server = await asyncio.start_server(bridge.handle, host, port, limit=2 * 1024 * 1024)
    print(f"Listening on {host}:{port}; mode={models.mode}; logs={log.directory}", flush=True)
    try:
        if seconds or until_run_end or stop_after_floor is not None:
            # Let the user return to the game before starting the trial.
            waiting_since = time.monotonic()
            while not bridge.world.ready(time.monotonic()) and not bridge.run_ended.is_set():
                if time.monotonic() - waiting_since > 300:
                    print("No fresh game state within 5 minutes; trial not started.", flush=True)
                    return
                await asyncio.sleep(0.1)
            duration = f"{seconds:g} seconds" if seconds else "the end of this run"
            print(f"Game state received; running until {duration}.", flush=True)
            end = time.monotonic() + seconds if seconds else float("inf")
            while (time.monotonic() < end and not bridge.run_ended.is_set()
                   and not (stop_file and Path(stop_file).exists())
                   and not planner_quota_exhausted(bridge.controller)):
                await asyncio.sleep(0.1)
            if bridge.run_ended.is_set():
                print("Run ended; releasing control.", flush=True)
        else:
            # start_server already accepts connections. serve_forever() itself
            # waits for live clients during cancellation on Python 3.12.
            while not planner_quota_exhausted(bridge.controller):
                await asyncio.sleep(.1)
    finally:
        if planner_quota_exhausted(bridge.controller):
            log.write('planner_unavailable_stop',reason='insufficient_quota',frame=bridge.world.frame)
            print('Planner balance is insufficient; ending this control session.',flush=True)
        if (pause_on_stop and not observe_only
                and (bridge.evaluation_guard is None or bridge.evaluation_guard.accepted)):
            try:
                await pause_game_before_release(bridge, log)
            except OSError as exc:
                log.write('game_pause_error', error=type(exc).__name__)
        server.close()
        # Python 3.12+ waits for accepted connections too. Release the game
        # connection before waiting, or the old controller survives a restart.
        await bridge.close()
        await server.wait_closed()


def planner_quota_exhausted(controller):
    return (getattr(controller,'quota_exhausted',False)
            or getattr(getattr(controller,'advance',None),'quota_exhausted',False))
