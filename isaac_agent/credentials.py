"""Project-local credentials; never printed or included in run logs."""
import json
import os
from pathlib import Path
import tempfile

KEY_NAMES = ("OPENAI_API_KEY", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY")
SECRET_FILE = Path(__file__).resolve().parents[1] / ".secrets" / "api-keys.json"


def read_keys(path=SECRET_FILE):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError()
        if any(k not in KEY_NAMES or not isinstance(v, str) for k, v in data.items()):
            raise ValueError()
    except (ValueError, UnicodeError):
        raise ValueError("Invalid local credentials file; open key setup again") from None
    return data


def save_keys(values, path=SECRET_FILE):
    if not isinstance(values, dict) or any(k not in KEY_NAMES for k in values):
        raise ValueError("Unsupported credential fields")
    data = read_keys(path)
    for name, value in values.items():
        if not isinstance(value, str):
            raise ValueError("Keys must be text")
        value = value.strip()
        if value:
            if any(c.isspace() for c in value):
                raise ValueError("Keys must not contain spaces or newlines")
            data[name] = value
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="keys-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {name: bool(data.get(name)) for name in KEY_NAMES}


def load_keys():
    for name, value in read_keys().items():
        if value and not os.environ.get(name):
            os.environ[name] = value
