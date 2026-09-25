"""Receive secrets through stdin; output only configuration status."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaac_agent.credentials import KEY_NAMES, read_keys, save_keys

try:
    if '--providers' in sys.argv:
        from isaac_agent.models import configuration
        config_path = Path(__file__).resolve().parents[1] / 'config.local.toml'
        config = configuration(config_path if config_path.exists() else None)
        result = {lane: data['key_env'] for lane, data in config.items()}
    elif '--status' in sys.argv:
        saved = read_keys()
        result = {name: bool(saved.get(name)) for name in KEY_NAMES}
    else:
        raw = sys.stdin.read(16385)
        if len(raw) > 16384:
            raise ValueError('Configuration input too large')
        result = save_keys(json.loads(raw))
    print(json.dumps(result))
except Exception:
    print('Could not save/read configuration; no secret values were logged.', file=sys.stderr)
    sys.exit(1)
