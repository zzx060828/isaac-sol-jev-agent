"""Observed controller facts for auditing pauses, without authorizing actions."""


def pickup_contact_facts(world, expected):
    """Record contact evidence, never infer acquisition or a failure cause."""
    import math
    from .state import xy
    current=next((p for p in world.pickups if expected.get('id') is not None
                  and p.get('id')==expected['id']),None)
    age=lambda channel: world.frame-world.sampled[channel] if channel in world.sampled else None
    return dict(expected={k:expected.get(k) for k in ('id','variant','sub_type','price')},
                observed=None if current is None else {k:current.get(k) for k in
                    ('id','variant','sub_type','price','wait','pos','entity_collision_class','collision_radius')},
                distance=None if current is None else round(math.dist(xy(world.player.get('pos')),xy(current.get('pos'))),2),
                can_pickup_item_reported=world.inventory.get('can_pickup_item'),
                held_trinkets=[world.inventory.get('trinket_0'),world.inventory.get('trinket_1')],
                sample_age_updates={k:age(k) for k in ('PICKUPS','PLAYER_INVENTORY','PLAYER_POSITION')},
                scope='Observed contact facts only. Missing fields are unknown; proximity does not prove acquisition.')


def gate_facts(world, now):
    return {'world_ready': world.ready(now), 'control_mode': world.control_mode,
            'dead': world.dead, 'state_age_seconds': round(max(0, now-world.updated), 3),
            'sensor_age_updates': {k: world.frame-world.sampled[k] if k in world.sampled else None
                for k in ('PLAYER_POSITION', 'ENEMIES', 'PROJECTILES', 'PLAYER_INVENTORY',
                          'PLAYER_HEALTH', 'BOMBS', 'FIRE_HAZARDS', 'ROOM_LAYOUT', 'ROOM_INFO')}}


def decision_trace(controller, world, now, local_target, fallback, actual, override, command):
    """Use already computed decisions; no second path search or model call."""
    c=controller
    pending=lambda task: task is not None and not task.done()
    return {'lane': c.route, 'sol_pending': pending(c.plan_task),
            'jev_pending': pending(c.decision_task),
            'sol_retry_in_seconds': round(max(0, c.plan_retry_at-now), 3),
            'plan_pending_reason': c.plan_pending,
            'strategy_ready': world.strategy_ready(),
            'strategy_choice': c.plan.get('strategy_choice'),
            'awaiting_strategy': bool(c.execution_plan.get('awaiting_strategy')),
            'hold_for_strategy': bool(c.execution_plan.get('hold_for_strategy')),
            'local_pickup_target': local_target,
            'parallel_resource_collection': bool(c.execution_plan.get('parallel_resource_collection')),
            'scavenge_resource_collection': bool(c.execution_plan.get('scavenge_resource_collection')),
            'fallback_action': fallback.id, 'actual_action': actual.id, 'shield_override': override,
            'resource_pulses': [k for k in ('use_item','use_bomb','use_card','use_pill','drop') if command.get(k)],
            'gate': gate_facts(world,now),
            'scope': 'Sampled controller facts. Missing local target does not prove no reachable pickup; '
                     'neutral input does not prove physical rest or avoidable waiting.'}
