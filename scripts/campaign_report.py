"""Combine observations of one game seed across controller restarts.

This reports development runs, not an uninterrupted benchmark or a win rate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segment_metrics import SegmentMetrics


def summarize(paths, seed):
    levels = {}
    files = []
    bosses, endings = [], []
    sources = Counter()
    inventories = set()
    segments=[]
    for path in paths:
        metrics=SegmentMetrics(seed)
        level = ''
        used = False
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    metrics.corrupt_lines+=1
                    continue
                metrics.consume(row)
                if row.get('event') == 'bridge':
                    msg = row['message']
                    if msg.get('type') == 'EVENT' and msg.get('event') == 'GAME_START':
                        level = ''
                    if msg.get('type') in ('DATA', 'FULL'):
                        level = str(msg.get('agent', {}).get('level_id', level))
                    if level.split(':')[0] != str(seed):
                        continue
                    used = True
                    stats = levels.setdefault(level, {'rooms': set(), 'damage_events': 0,
                                                       'reported_damage_amount': 0,
                                                       'damage_sources': Counter(), 'resource_uses': Counter()})
                    if msg.get('type') in ('DATA', 'FULL'):
                        if msg.get('room_index') is not None:
                            stats['rooms'].add(msg['room_index'])
                        inv = msg.get('payload', {}).get('PLAYER_INVENTORY', [])
                        inv = inv[0] if isinstance(inv, list) and inv else inv
                        if isinstance(inv, dict):
                            items = inv.get('collectibles', {})
                            if isinstance(items, dict):
                                inventories.update(k for k, count in items.items() if count and str(k) != '733')
                    elif msg.get('type') == 'EVENT':
                        event, data = msg.get('event'), msg.get('data', {})
                        evidence = {'log': str(path.parent), 'frame': msg.get('frame'), 'level': level, **data}
                        if event == 'PLAYER_DAMAGE':
                            stats['damage_events'] += 1
                            stats['reported_damage_amount'] += data.get('amount', 0)
                            stats['damage_sources'][str(data.get('source_type'))] += 1
                        elif event == 'RESOURCE_USED':
                            stats['resource_uses'][str(data.get('kind'))] += 1
                        elif event == 'NPC_DEATH' and data.get('is_boss'):
                            bosses.append(evidence)
                        elif event in ('PLAYER_DEATH', 'GAME_END'):
                            endings.append({'event': event, **evidence})
                elif level.split(':')[0] == str(seed) and row.get('event') == 'input':
                    sources[row.get('source', 'unknown')] += 1
        if used or metrics.requests or metrics.usage:
            files.append(str(path.parent))
            segments.append(metrics.result(path))
    for stats in levels.values():
        stats['rooms'] = sorted(stats['rooms'])
        stats['damage_sources'] = dict(stats['damage_sources'])
        stats['resource_uses'] = dict(stats['resource_uses'])
    usage_totals={};request_totals=Counter()
    for segment in segments:
        request_totals.update(segment['logged_requests'])
        for model,usage in segment['reported_model_usage'].items():
            total=usage_totals.setdefault(model,dict(usage_rows=0,input_tokens=0,output_tokens=0,reported_cost=None))
            for k in ('usage_rows','input_tokens','output_tokens'):total[k]+=usage[k]
            if usage['reported_cost'] is not None:total['reported_cost']=(total['reported_cost'] or 0)+usage['reported_cost']
    return {'seed': str(seed), 'controller_segments': files, 'levels': levels,
            'segment_metrics':segments,
            'logged_model_requests':dict(request_totals),'reported_model_usage':usage_totals,
            'recorded_python_manifest_variants':len({s['recorded_python_manifest_fingerprint'] for s in segments
                                                     if s['recorded_python_manifest_fingerprint']}),
            'segments_without_source_manifest':sum(s['recorded_python_manifest_fingerprint'] is None for s in segments),
            'recorded_trial_manifest_variants':len({s['trial_manifest']['fingerprint'] for s in segments if s['trial_manifest']}),
            'segments_without_verified_trial_manifest':sum(s['trial_manifest_status']!='verified' for s in segments),
            'boss_deaths_observed': bosses, 'end_events': endings,
            'items_observed': sorted(inventories, key=int), 'input_sources': dict(sources),
            'victory_verified': False,
            'scope': 'Observed development segments with controller/game restarts. '
                     'Damage amounts are callback values, not guaranteed health loss. '
                     'No victory is inferred from boss kills or exit events; verify the ending separately.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True)
    parser.add_argument('--runs-dir', type=Path, default=Path('runs'))
    args = parser.parse_args()
    result = summarize(sorted(args.runs_dir.glob('*/events.jsonl')), args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
