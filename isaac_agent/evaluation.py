"""Preregistered evaluation identities; no route knowledge or model calls."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from .trial_metadata import fingerprint, public_models


def sources(root):
    root = Path(root)
    paths = []
    for directory in ('isaac_agent', 'scripts', 'tests', 'vendor/socketbridge', 'build/SocketBridge_AstraJev'):
        paths.extend(p for p in (root / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json', '.lua', '.xml')
                     and '__pycache__' not in p.parts)
    paths.extend(root / name for name in ('pyproject.toml', 'config.example.toml') if (root / name).is_file())
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def development_seeds(paths):
    """Scan only explicit public logs/fixtures, never private config or keys."""
    found = set()
    pattern = re.compile(rb'"(?:level_id|level)"\s*:\s*"(\d+):|"seed"\s*:\s*"?(\d+)')
    for path in paths:
        with Path(path).open('rb') as stream:
            for line in stream:
                for match in pattern.finditer(line):
                    found.add(int(match.group(1) or match.group(2)))
    return found


def freeze(root, output, config, seeds, seen, *, seconds=900, player_type=1, difficulty=0):
    root, output = Path(root), Path(output)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Provide distinct held-out seeds')
    if any(type(s) is not int or not 0 < s < 2**32 for s in seeds):
        raise ValueError('Seeds must be nonzero unsigned 32-bit integers')
    if set(seeds) & set(seen):
        raise ValueError('Held-out seed was already observed in development')
    if not 10 <= seconds <= 3600:
        raise ValueError('Evaluation duration must be 10..3600 seconds')
    manifest = sources(root)
    cases = []
    for index, seed in enumerate(seeds):
        # Counterbalance the order, keeping both arms on the same seed.
        for arm in (('off', 'tactics') if index % 2 == 0 else ('tactics', 'off')):
            cases.append(dict(id=f's{index+1:02d}-{arm}', seed=seed, jev_scope=arm))
    identity = dict(schema_version=1, created_at=time.time(), sources=manifest,
        models=public_models(config), development_seeds=sorted(seen), held_out_seeds=seeds,
        cases=cases, execution=dict(seconds=seconds, player_type=player_type, difficulty=difficulty,
            repentance_plus=False, fresh_frame_limit=120, stop_after_floor=None,
            match_initial_state=True),
        comparison='tactics versus off; both use Sol for fast goals and local top-four memory retrieval',
        limitations='Preregistration is not a result. Numeric seeds must match the actual new game. '
            'Game binary, unlocks, other installed mods and recorder still require matching external evidence. '
            'Same seed does not guarantee the same later RNG trajectory. No victory is inferred from stopping.')
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        destination = output / 'snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    value = dict(identity=identity, fingerprint=fingerprint(identity))
    (output / 'batch.json').write_text(json.dumps(value, indent=2)+'\n')
    return value


def load_batch(path):
    value = json.loads(Path(path).read_text())
    identity = value['identity']
    if identity.get('schema_version') != 1 or fingerprint(identity) != value.get('fingerprint'):
        raise ValueError('Evaluation batch fingerprint mismatch')
    return value


def verify_sources(root, batch):
    if sources(root) != batch['identity']['sources']:
        raise ValueError('Frozen source changed; create a new evaluation batch')


def bind_case(root, batch_path, case_id, config, args):
    batch = load_batch(batch_path)
    verify_sources(root, batch)
    if public_models(config) != batch['identity']['models']:
        raise ValueError('Model settings differ from the frozen batch')
    lua_hash = batch['identity']['sources'].get('build/SocketBridge_AstraJev/main.lua')
    if (not args.installed_lua or not lua_hash
            or hashlib.sha256(Path(args.installed_lua).read_bytes()).hexdigest() != lua_hash):
        raise ValueError('Installed Lua must match the frozen build')
    case = next((c for c in batch['identity']['cases'] if c['id'] == case_id), None)
    if case is None:
        raise ValueError('Unknown evaluation case')
    expected = batch['identity']['execution']
    if (args.mode != 'hybrid' or args.jev_scope != case['jev_scope'] or not args.isolated_memory
            or args.observe_only or args.seconds != expected['seconds']
            or args.stop_after_floor is not None or not args.until_run_end):
        raise ValueError('Evaluation execution differs from the registered case')
    registration = dict(batch_fingerprint=batch['fingerprint'], case=case, expected=expected)
    if expected.get('match_initial_state'):
        first = next(c for c in batch['identity']['cases'] if c['seed'] == case['seed'])
        registration['initial_state'] = dict(
            directory=str((Path(batch_path).parent/'starts').resolve()), reference_case=first['id'])
    return registration


def claim_case(batch_path, registration, run_dir):
    """Keep every attempt, including connection failures; never overwrite it."""
    directory = Path(batch_path).parent / 'attempts'
    directory.mkdir(exist_ok=True)
    with (directory / (registration['case']['id']+'.json')).open('x') as stream:
        json.dump(dict(registration=registration, run_dir=str(Path(run_dir).resolve()),
                       claimed_at=time.time()), stream, indent=2)


START_CHANNELS = ('PLAYER_STATS', 'PLAYER_HEALTH', 'PLAYER_INVENTORY',
                  'PLAYER_POSITION', 'ROOM_INFO', 'ROOM_LAYOUT', 'PICKUPS')


def initial_state(world):
    """Observed start only; no hidden map or future reward information.

    Ignore pickup instance IDs and spawn-animation timers, but preserve effects,
    prices, choice groups and positions. This does not certify later RNG or mods.
    """
    from .state import first, records
    payload = world.payload
    pickups = [{k: v for k, v in item.items() if k not in ('id', 'wait')}
               for item in records(payload['PICKUPS'])]
    return json.loads(json.dumps(dict(
        level=world.level, room=world.room,
        stats=first(payload['PLAYER_STATS']),
        health={k: v for k, v in first(payload['PLAYER_HEALTH']).items() if k != 'damage_cooldown'},
        inventory=first(payload['PLAYER_INVENTORY']),
        player_position=first(payload['PLAYER_POSITION']).get('pos'),
        room_info=first(payload['ROOM_INFO']), layout=payload['ROOM_LAYOUT'],
        pickups=sorted(pickups, key=lambda item: json.dumps(item, sort_keys=True)))))


def record_initial_state(registration, world):
    """Retain both starts, rejecting a mismatched second arm before control."""
    settings = registration['initial_state']
    directory = Path(settings['directory'])
    case = registration['case']['id']
    reference_case = settings['reference_case']
    state = initial_state(world)
    receipt = dict(batch_fingerprint=registration['batch_fingerprint'], case=case,
                   seed=registration['case']['seed'], state=state,
                   state_fingerprint=fingerprint(state), reference_case=reference_case,
                   matched=True, differences=[])
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if reference_case != case:
            reference = json.loads((directory/(reference_case+'.json')).read_text())
            if (reference['batch_fingerprint'] != receipt['batch_fingerprint']
                    or reference['case'] != reference_case or reference['seed'] != receipt['seed']
                    or reference.get('matched') is not True
                    or fingerprint(reference['state']) != reference['state_fingerprint']):
                raise RuntimeError('Evaluation initial-state reference is invalid')
            receipt['differences'] = sorted(k for k in set(state) | set(reference['state'])
                                            if state.get(k) != reference['state'].get(k))
            receipt['matched'] = not receipt['differences']
        with (directory/(case+'.json')).open('x') as stream:
            json.dump(receipt, stream, indent=2)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError('Evaluation initial-state evidence unavailable or already recorded') from exc
    if not receipt['matched']:
        raise RuntimeError('Evaluation initial state mismatch: '+', '.join(receipt['differences']))


class StartGuard:
    """Check observed start before FORCE_AI, then reject reset/seed changes."""
    def __init__(self, registration):
        self.registration = registration
        self.accepted = False
        self.epoch = None

    def check(self, world):
        case, expected = self.registration['case'], self.registration['expected']
        level = world.level.split(':')
        if not world.level:
            return False
        if level[0] != str(case['seed']):
            raise RuntimeError('Evaluation seed mismatch')
        if self.accepted:
            if world.epoch != self.epoch:
                raise RuntimeError('Evaluation run reset or reconnected')
            return True
        if len(level) != 3 or level[1] != '1' or world.frame > expected['fresh_frame_limit']:
            raise RuntimeError('Evaluation requires a fresh first-floor start')
        actual = dict(player_type=world.stats.get('player_type'), difficulty=world.info.get('difficulty'),
                      repentance_plus=world.capabilities.get('repentance_plus'))
        continued = world.capabilities.get('continued_run')
        if any(v is None for v in actual.values()) or continued is None:
            return False
        if continued is not False or any(actual[k] != expected[k] for k in actual):
            raise RuntimeError('Evaluation character, difficulty, edition or continue state mismatch')
        if not world.fresh('PLAYER_STATS', 6) or not world.fresh('ROOM_INFO', 6):
            return False
        if expected.get('match_initial_state'):
            if any(channel not in world.payload or not world.fresh(channel, 6)
                   for channel in START_CHANNELS):
                return False
            record_initial_state(self.registration, world)
        self.accepted, self.epoch = True, world.epoch
        return True
