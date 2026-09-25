"""Freeze held-out cases or check a frozen snapshot. Does not start the game."""
import argparse
import json
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaac_agent.evaluation import development_seeds, freeze, load_batch, verify_sources
from isaac_agent.models import configuration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    create = commands.add_parser('freeze')
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--config', type=Path, default=ROOT/'config.local.toml')
    create.add_argument('--runs-dir', type=Path, default=ROOT/'runs')
    create.add_argument('--count', type=int, default=6)
    create.add_argument('--seeds', nargs='+', type=int)
    create.add_argument('--seconds', type=int, default=900)
    check = commands.add_parser('check')
    check.add_argument('batch', type=Path)
    report = commands.add_parser('report')
    report.add_argument('batch', type=Path)
    a = p.parse_args()
    try:
        if a.command == 'report':
            from scripts.campaign_report import summarize
            batch = load_batch(a.batch)
            cases = []
            for case in batch['identity']['cases']:
                attempt = a.batch.parent/'attempts'/(case['id']+'.json')
                row = dict(case=case, status='not_attempted', victory_verified=False)
                if attempt.exists():
                    attempt_data = json.loads(attempt.read_text())
                    registration = attempt_data['registration']
                    if registration['batch_fingerprint'] != batch['fingerprint'] or registration['case'] != case:
                        raise ValueError('Attempt registration does not match batch')
                    events = Path(attempt_data['run_dir'])/'events.jsonl'
                    row.update(status='attempt_without_log', run_dir=attempt_data['run_dir'])
                    if events.exists():
                        last_check, rejected, corrupt = None, [], 0
                        with events.open() as stream:
                            for line in stream:
                                try: event = json.loads(line)
                                except json.JSONDecodeError:
                                    corrupt += 1
                                    continue
                                if event.get('event') == 'evaluation_end_checked': last_check = event
                                if event.get('event') == 'evaluation_rejected': rejected.append(event)
                        row.update(status='attempt_recorded', end_check=last_check,
                                   rejected=rejected, corrupt_lines=corrupt,
                                   metrics=summarize([events], case['seed']))
                cases.append(row)
            print(json.dumps(dict(batch=batch['fingerprint'], cases=cases,
                scope='All registered cases, including failures/unstarted cases. End checks alone do not verify '
                      'matching game binaries, unlocks, other mods, recordings or victory.'), indent=2))
            return
        if a.command == 'check':
            batch = load_batch(a.batch)
            verify_sources(ROOT, batch)
            verify_sources(a.batch.parent/'snapshot', batch)
            print(json.dumps({'source_verified': True, 'batch': batch['fingerprint'],
                              'registered_cases': len(batch['identity']['cases']), 'gameplay_verified': False}))
            return
        if not 1 <= a.count <= 50:
            raise ValueError('count must be 1..50')
        paths = sorted(a.runs_dir.rglob('events.jsonl')) + sorted((ROOT/'tests/fixtures').glob('*.json'))
        seen = development_seeds(paths)
        # Reserve prior batches too: a viewed/tested holdout must not be recycled.
        for old in (ROOT/'evaluations').glob('*/batch.json'):
            seen.update(load_batch(old)['identity']['held_out_seeds'])
        seeds = a.seeds or []
        if not seeds:
            while len(seeds) < a.count:
                seed = secrets.randbelow(2**32-1)+1
                if seed not in seen and seed not in seeds:
                    seeds.append(seed)
        batch = freeze(ROOT, a.output, configuration(a.config), seeds, seen, seconds=a.seconds)
        print(json.dumps({'batch': str(a.output/'batch.json'), 'fingerprint': batch['fingerprint'],
                          'excluded_seeds': len(seen), 'registered_cases': len(batch['identity']['cases']),
                          'gameplay_verified': False}))
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
