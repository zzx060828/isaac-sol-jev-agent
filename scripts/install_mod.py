"""Install into an explicitly supplied mods directory without overwriting files."""
import argparse
from pathlib import Path
import shutil

from build_mod import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mods-dir', required=True, type=Path)
    args = parser.parse_args()
    if not args.mods_dir.is_dir():
        parser.error('Supply the existing mods directory reported in the game savedatapath.txt')
    source = build()
    dest = args.mods_dir / source.name
    if dest.exists():
        parser.error(f'Already exists: {dest}; no files overwritten')
    shutil.copytree(source, dest)
    print(f'Installed {dest}')
    print('Steam launch option: --luadebug; enable SocketBridge Astra JEV in the game Mods menu.')


if __name__ == '__main__':
    main()
