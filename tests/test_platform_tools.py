import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from isaac_agent.platform_tools import windows_helper


class PlatformToolsTests(unittest.TestCase):
    def test_native_windows_uses_current_interpreter(self):
        with patch('isaac_agent.platform_tools.sys.platform', 'win32'), \
             patch('isaac_agent.platform_tools.sys.executable', 'C:/Python/python.exe'):
            command = windows_helper('scripts/background_window_windows.py', 'status')
        self.assertEqual(command[0], 'C:/Python/python.exe')
        self.assertTrue(Path(command[1]).is_absolute())
        self.assertEqual(command[-1], 'status')

    def test_wsl_respects_explicit_python_and_converts_path(self):
        with patch('isaac_agent.platform_tools.sys.platform', 'linux'), \
             patch.dict(os.environ, {'WSL_DISTRO_NAME':'CustomDistro',
                 'ISAAC_WINDOWS_PYTHON':'/mnt/e/Python/python.exe'}, clear=True), \
             patch('isaac_agent.platform_tools.subprocess.run', return_value=SimpleNamespace(stdout='E:\\agent\\helper.py\n')) as run:
            command = windows_helper('helper.py', 'key')
        self.assertEqual(command, ['/mnt/e/Python/python.exe', 'E:\\agent\\helper.py', 'key'])
        self.assertEqual(run.call_args.args[0][:2], ['wslpath', '-w'])

    def test_non_windows_host_fails_with_clear_error(self):
        with patch('isaac_agent.platform_tools.sys.platform', 'linux'), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(OSError, 'Windows or WSL'):
                windows_helper('helper.py')

    def test_missing_windows_python_fails_before_starting_helper(self):
        with patch('isaac_agent.platform_tools.sys.platform', 'linux'), \
             patch.dict(os.environ, {'WSL_DISTRO_NAME':'CustomDistro'}, clear=True), \
             patch('isaac_agent.platform_tools.shutil.which', return_value=None):
            with self.assertRaisesRegex(OSError, 'ISAAC_WINDOWS_PYTHON'):
                windows_helper('helper.py')
