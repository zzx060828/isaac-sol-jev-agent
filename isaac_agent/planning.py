"""Stable high-level decision triggers, independent of combat event frequency."""
from .strategy import offers
from .pickups import routine, has_golden_key


def planning_context(world, finished=()):
    inv = world.inventory
    active = inv.get('active_items', {}).get('0', {})
    # These effects already have local use policies. Their health/risk thresholds
    # must not turn ordinary combat damage into a high-level model trigger.
    automatic_items = {45, 78, 292, 58, 34, 35, 192, 97, 145}
    item = active.get('item', 0)
    charge = active.get('charge', 0) + active.get('battery_charge', 0)
    maximum = active.get('max_charge', 0)
    ready = (maximum > 0 and charge >= maximum
             or maximum == 0 and active.get('charge_type') == 0)
    choices = []
    for offer in offers(world):
        if offer['choice'] in finished:
            continue
        entity = offer.get('entity', {})
        # Ordinary free hearts/coins/keys/bombs/batteries are local pickups.
        if offer['kind'] == 'pickup' and routine(entity):
            continue
        choices.append(offer['choice'])
    return {
        'room': world.token,
        'items': (tuple(sorted(inv.get('collectibles', {}).items())),
                  tuple(sorted((str(slot), data.get('item', 0))
                               for slot, data in inv.get('active_items', {}).items())),
                  inv.get('trinket_0', 0), inv.get('trinket_1', 0),
                  inv.get('pill_0', 0), inv.get('card_0', 0)),
        'choices': tuple(sorted(choices)),
        'charged_item': item if item and ready and item not in automatic_items else None,
        'access': tuple(sorted(str(slot) for slot, door in world.layout.get('doors', {}).items()
                               if world.info.get('is_clear') is True and door.get('is_locked')
                               and not has_golden_key(world)
                               and door.get('target_room_type') in (2, 4, 12, 24)
                               and inv.get('keys', 0) > 0
                               # Local exploration already opens ordinary treasure
                               # doors. With a spare key this needs no extra Sol
                               # pause; spending the last key remains strategic.
                               and (door.get('target_room_type') != 4 or inv.get('keys') == 1))),
    }


def planning_reasons(previous, current):
    if previous is None or previous['room'] != current['room']:
        return ['new_room']
    labels = {'items': 'items_changed', 'choices': 'resource_choices_changed',
              'charged_item': 'active_item_ready_changed', 'access': 'locked_room_access_changed'}
    return [reason for field, reason in labels.items() if previous[field] != current[field]]


def planning_observation(observation):
    """Keep strategic evidence; low-level geometry stays in the local controller."""
    result = dict(observation)
    for key in ('enemies', 'nonblocking_entities', 'projectiles', 'bombs', 'fires'):
        result[key] = [{k: v for k, v in e.items() if k in (
            'id', 'type', 'variant', 'name', 'pos', 'vel', 'hp', 'max_hp', 'is_boss',
            'is_vulnerable', 'collision_radius', 'is_extinguished', 'ground_only')}
            for e in observation.get(key, [])[:8]]
    from collections import Counter
    terrain = observation.get('terrain', [])
    result['terrain_summary'] = dict(Counter(str(c.get('type')) for c in terrain))
    result['terrain'] = [c for c in terrain if c.get('type') in (4, 5, 8, 9, 12, 14, 17, 18, 20, 22, 25)][:24]
    result['local_execution'] = ('Local navigation handles full collision geometry. Routine pickups and short fights '
                                'use JEV/local control; slow planning is reserved for consequential choices and bosses.')
    return result


def model_observation(observation):
    """Send one observed map, retaining remote item descriptions and wall evidence."""
    from copy import deepcopy
    result={k:v for k,v in observation.items()
            if k not in ('pickup_review_basis','pickup_review_context')}
    graph=observation.get('floor_map')
    rooms=observation.get('explored_rooms')
    if not isinstance(graph,dict) or not isinstance(rooms,dict):
        return result
    nodes=graph.get('nodes',[])
    # A partial map must not silently remove rooms only present in the source.
    if not set(rooms).issubset({str(n['id']) for n in nodes}):
        return result
    result['floor_map']=deepcopy(graph)
    for node in result['floor_map']['nodes']:
        room=rooms.get(str(node['id']),{})
        if 'collectibles' in room:node['remaining_items']=deepcopy(room['collectibles'])
        if 'walls' in room:node['walls_last_seen']=deepcopy(room['walls'])
        if 'curses' in room:node['curses_last_seen']=room['curses']
    result.pop('explored_rooms',None)
    return result


def enrich_planning(world, observation, preparation=None):
    from .floor_map import floor_map
    from .knowledge import expert_knowledge
    result = planning_observation(observation)
    result['floor_map'] = floor_map(world)
    result['expert_knowledge'] = expert_knowledge(world)
    from .rocks import observed
    result['marked_rocks'] = observed(world)
    result['advance_preparation'] = preparation
    from .journey import options as navigation_options
    result['navigation_options'] = navigation_options(world)
    from .review import pool, valid
    result['deferred_pickups'] = [{k:r[k] for k in ('room','choice','rationale')}
                                  | {'reconsider_on':list(r['basis'])}
                                  for r in pool(world).values() if valid(world,r)]
    from .pickup_progress import blocked_choices
    result['local_pickup_failures'] = [{k:r[k] for k in ('room','choice','reason','collection_frames')}
                                      | {'status':'not collected; previous approach timed out'}
                                      for r in world.pickup_failures.values()
                                      if r['choice'] in blocked_choices(world,r['room'])]
    return result
