"""Summarize observed gameplay, including evidence distinct from model intent."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaac_agent.state import World, xy


def report(path):
    world = World()
    counts, rooms = Counter(), {}
    damage, strategies, uses, plans, boss_kills = [], [], [], [], []
    model_usage = Counter()
    levels = []
    last_input = None
    last_state_time, last_status = None, None
    changes = []
    previous_inventory = None
    for line in path.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A currently appended final row may not be complete yet.
        event = row['event']
        counts[event] += 1
        usage = row.get('usage', {})
        model_usage['input_tokens'] += usage.get('input_tokens', usage.get('prompt_tokens', 0))
        model_usage['output_tokens'] += usage.get('output_tokens', usage.get('completion_tokens', 0))
        if event == 'input':
            last_input = row
        elif event == 'plan':
            plans.append(row['plan'])
        elif event == 'strategy_result':
            strategies.append(row)
        elif event == 'bridge':
            msg = row['message']
            if msg.get('type') == 'STATUS':
                last_status = {'time': row['time'], **msg}
            if msg.get('event') in ('PLAYER_DAMAGE', 'PLAYER_DEATH', 'GAME_END'):
                damage.append(msg)
            if msg.get('event') == 'RESOURCE_USED':
                uses.append(msg)
            if msg.get('event') == 'NPC_DEATH' and msg.get('data', {}).get('is_boss'):
                boss_kills.append(msg)
            if not world.ingest(msg):
                continue
            last_state_time = row['time']
            key = f'{world.epoch}:{world.level}:{world.room}'
            if world.level not in levels:
                levels.append(world.level)
            entry = rooms.setdefault(key, {'room': world.room, 'frames': 0, 'first': world.frame,
                                         'last': world.frame, 'cleared': False, 'positions': []})
            entry['last'] = world.frame
            entry['frames'] += 1
            entry['cleared'] |= bool(world.info.get('is_clear'))
            entry['positions'].append((world.frame, xy(world.player.get('pos'))))
            inv = world.inventory
            now_inventory = {k: inv.get(k) for k in ('bombs', 'keys', 'coins', 'collectibles')}
            if 'PLAYER_INVENTORY' in world.payload and now_inventory != previous_inventory:
                changes.append({'frame': world.frame, 'room': world.room, **now_inventory})
                previous_inventory = now_inventory
    for entry in rooms.values():
        positions = entry.pop('positions')
        window = [p for f, p in positions if f >= entry['last'] - 180]
        entry['recent_span'] = [round(max(p[i] for p in window)-min(p[i] for p in window), 1)
                                for i in (0, 1)] if window else []
    return {'log': str(path), 'age_seconds': round(time.time()-path.stat().st_mtime, 1),
            'data_age_seconds': round(time.time()-last_state_time, 1) if last_state_time else None,
            'last_status': last_status,
            'events': dict(counts), 'state': {'frame': world.frame, 'room': world.room,
            'level': world.level,
            'input_feedback': world.capabilities,
            'mode': world.control_mode, 'position': xy(world.player.get('pos')),
            'enemies': len(world.enemies), 'combat_enemies': len(world.combat_enemies), 'health': world.health, 'inventory': world.inventory},
            'rooms': list(rooms.values()), 'damage_and_end': damage, 'resource_confirmations': uses,
            'boss_kills': boss_kills, 'levels_observed': levels, 'reported_model_usage': dict(model_usage),
            'strategy_results': strategies, 'last_plan': plans[-1] if plans else None,
            'inventory_changes': changes[-20:], 'last_input': last_input}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()
    path = Path(args.path) if args.path else max(Path('runs').glob('live-20*/events.jsonl'),
                                                key=lambda p: p.stat().st_mtime)
    if path.is_dir():
        path = path / 'events.jsonl'
    print(json.dumps(report(path), ensure_ascii=False, indent=2))
