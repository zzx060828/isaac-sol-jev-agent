"""Compact observed floor graph; unknown geometry remains explicitly unknown."""
from collections import deque
from .secrets import neighbor, OFFSETS
from .state import ROOM_NAMES


def floor_map(world):
    from .resource_access import evidence
    rooms = {int(k): v for k, v in world.rooms.items() if str(k).isdigit() and int(k) < 169}
    nodes, links, frontier = [], [], {}
    occupied = set(rooms)
    for index, room in sorted(rooms.items()):
        nodes.append({'id': index, 'anchor': [index % 13, index // 13],
                      'type': room.get('type'), 'name': ROOM_NAMES.get(room.get('type'), 'Unknown'),
                      'shape': room.get('shape'), 'clear_last_seen': room.get('clear'),
                      'walls_observed': bool(room.get('walls')),
                      'remaining_items': room.get('collectibles') or [],
                      'resources_last_seen':room.get('resources') or [],
                      'resource_access_last_seen':evidence(world,room),
                      'resources_omitted':room.get('resources_omitted') or 0,
                      'marked_rocks_last_seen':room.get('marked_rocks') or [],
                      'machines_last_seen':room.get('machines') or [],
                      'floor_exit_observed':room.get('floor_exit') is True})
        for slot, door in room.get('doors', {}).items():
            target = door.get('target_room')
            if not isinstance(target, int) or not 0 <= target < 169:
                continue
            occupied.add(target)
            links.append({'from': index, 'slot': str(slot), 'to': target,
                          'target_type': door.get('target_room_type'),
                          'locked_last_seen': door.get('is_locked'),
                          'open_last_seen': door.get('is_open')})
            if target not in rooms:
                frontier[target] = {'id': target, 'type': door.get('target_room_type')}
    # Graph distance is useful for backtracking costs, not permission to cross
    # a locked, cursed or currently closed door.
    paths = {world.room: []}
    queue = deque([world.room])
    while queue:
        source = queue.popleft()
        for edge in links:
            if edge['from'] != source or edge['to'] in paths or edge['target_type'] in (10,14,15):
                continue
            paths[edge['to']] = paths[source]+[edge['slot']]
            queue.append(edge['to'])
    for node in nodes + list(frontier.values()):
        path = paths.get(node['id'])
        node['observed_hops'] = len(path) if path is not None else None
        node['first_door_slot'] = path[0] if path else None

    excluded = set(occupied)
    for index, room in rooms.items():
        if room.get('shape') not in (None, 1):
            excluded.update(i for i in (index+1,index+13,index+14) if i < 169)
    candidates = {}
    for index, room in rooms.items():
        if room.get('shape') != 1:
            continue
        for slot, wall in room.get('walls', {}).items():
            if int(slot) not in OFFSETS or not wall.get('approach_clear') or wall.get('has_door'):
                continue
            cell = neighbor(index, int(slot))
            if cell is None or cell in excluded:
                continue
            adjacent = [neighbor(cell, side) for side in OFFSETS]
            known = [i for i in adjacent if i in occupied]
            verified = []
            for other in known:
                data = rooms.get(other, {})
                back = next((str(s) for s in OFFSETS if neighbor(other,s) == cell), None)
                evidence = data.get('walls', {}).get(back, {})
                if (data.get('shape') == 1 and data.get('type') not in (5,7,8,14,15)
                        and evidence.get('approach_clear') and not evidence.get('has_door')):
                    verified.append(other)
            if len(verified) != len(known) or not verified:
                continue
            attempts = getattr(world.run_memory, 'secret_attempts', {})
            if f'{world.level}:{cell}' in attempts:
                continue
            entry = candidates.setdefault(cell, {'cell': cell, 'anchor': [cell%13,cell//13],
                'observed_neighbors': verified,
                'unknown_neighbors': [i for i in adjacent if i is not None and i not in occupied],
                'hypothesis': 'regular' if len(verified) >= 2 else 'possible_dead_end',
                'possible_types': ['regular']+(['super'] if len(verified)==1
                                   and rooms[verified[0]].get('type')==1 else []),
                'entrances': [], 'executable_now': False})
            path = paths.get(index)
            entry['entrances'].append({'room': index, 'wall_slot': slot,
                'observed_hops': len(path) if path is not None else None,
                'first_door_slot': path[0] if path else None})
    # Three and four neighbors share a tier; do not invent exact probabilities.
    hypotheses = sorted(candidates.values(), key=lambda c: (
        0 if len(c['observed_neighbors']) >= 3 else 1 if len(c['observed_neighbors']) == 2 else 2,
        min((e['observed_hops'] for e in c['entrances'] if e['observed_hops'] is not None), default=999)))
    return {'level': world.level, 'current_room': world.room, 'grid_width': 13,
            'visibility': 'observed rooms and observed door targets only',
            'nodes': nodes, 'connections': links, 'unvisited_observed_rooms': list(frontier.values()),
            'coverage': {'known_rooms': len(nodes), 'known_frontiers': len(frontier),
                'no_known_frontier': len(nodes) > 1 and not frontier,
                'wall_geometry_missing': [n['id'] for n in nodes if not n['walls_observed']],
                'engine_complete_map': False},
            'secret_hypotheses': hypotheses[:8], 'hypotheses_omitted': max(0,len(hypotheses)-8),
            'limitations': 'No known frontier does not prove the floor is fully explored. Anchors are not '
                'full footprints for large/L rooms. Remote doors and resources need fresh observation. '
                'Map hypotheses do not authorize bombing; only current strategy_offers do.'}
