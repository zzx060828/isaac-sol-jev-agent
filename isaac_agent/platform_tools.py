"""Resolve optional Windows helpers without workstation-specific paths."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def windows_helper(script, *arguments):
    script = Path(script).resolve()
    if sys.platform == 'win32':
        return [sys.executable, str(script), *arguments]
    if not os.environ.get('WSL_DISTRO_NAME'):
        raise OSError('Windows helper requires Windows or WSL')
    python = os.environ.get('ISAAC_WINDOWS_PYTHON') or shutil.which('python.exe')
    if not python:
        raise OSError('Set ISAAC_WINDOWS_PYTHON to a Windows Python executable accessible from WSL')
    converted = subprocess.run(['wslpath', '-w', str(script)], check=True,
                               capture_output=True, text=True).stdout.strip()
    if not converted:
        raise OSError('wslpath returned no Windows path')
    return [python, converted, *arguments]
