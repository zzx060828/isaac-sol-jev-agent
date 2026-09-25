import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io
import urllib.error

from isaac_agent.credentials import read_keys, save_keys


class CredentialsTests(unittest.TestCase):
    def test_model_error_diagnostics_never_echo_provider_text(self):
        from isaac_agent.models import post, ModelHTTPError
        body = {'error': {'message': 'Rate limit reached for sk-test-secret',
                          'code': 'arbitrary-secret-value'}}
        error = urllib.error.HTTPError('https://example.test', 429, 'untrusted',
                                       {'Retry-After': '40'}, io.BytesIO(json.dumps(body).encode()))
        config = {'url': 'https://example.test', 'key_env': 'MODEL_TEST_KEY', 'timeout': 1}
        with patch.dict(os.environ, {'MODEL_TEST_KEY': 'test-auth'}), \
             patch('urllib.request.urlopen', side_effect=error):
            with self.assertRaises(ModelHTTPError) as caught:
                post(config, {})
        self.assertEqual(caught.exception.indicators, ('rate_limit',))
        self.assertEqual(caught.exception.retry_after, 40)
        self.assertNotIn('secret', str(caught.exception))
        self.assertNotIn('test-auth', str(caught.exception))

    def test_save_preserves_blanks_and_restricts_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'private' / 'keys.json'
            status = save_keys({'OPENAI_API_KEY': 'test-key'}, path)
            self.assertTrue(status['OPENAI_API_KEY'])
            self.assertNotIn('test-key', json.dumps(status))
            save_keys({'OPENAI_API_KEY': '', 'OPENROUTER_API_KEY': 'other-test-key'}, path)
            self.assertEqual(read_keys(path)['OPENAI_API_KEY'], 'test-key')
            if os.name == 'posix':
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)

    def test_rejects_unknown_fields_without_changing_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'keys.json'
            save_keys({'OPENAI_API_KEY': 'test-key'}, path)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                save_keys({'OTHER': 'bad'}, path)
            self.assertEqual(path.read_bytes(), before)
