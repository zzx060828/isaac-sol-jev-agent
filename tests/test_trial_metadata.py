from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock,patch

from isaac_agent.trial_metadata import build_manifest,observed_environment
from isaac_agent.runtime import Log
from isaac_agent.state import World
from isaac_agent.demo import frame_message
from scripts.campaign_report import summarize
from test_segment_metrics import data,write


def config():
    return {'astra':dict(model='gpt-5.6-sol',url='https://user:URL_SECRET@relay.example/v1/PATH_SECRET?key=QUERY_SECRET#FRAGMENT_SECRET',
                         key_env='KEY_ENV_SECRET',api_key='API_SECRET',headers={'Authorization':'HEADER_SECRET'},
                         max_completion_tokens=1200,reasoning_effort='medium',timeout=25),
            'jev':dict(model='typesafe/jev-1.13',url='https://openrouter.ai/api/alpha/decisions',timeout=2)}


class TrialMetadataTests(unittest.TestCase):
    def test_credentials_and_unlisted_fields_are_absent_and_not_hashed(self):
        first=build_manifest({'runtime.py':'hash'},config(),{'mode':'hybrid','config':'FILE_SECRET'})
        serialized=json.dumps(first)
        for forbidden in ('URL_SECRET','PATH_SECRET','QUERY_SECRET','FRAGMENT_SECRET','KEY_ENV_SECRET','API_SECRET','HEADER_SECRET','FILE_SECRET'):
            self.assertNotIn(forbidden,serialized)
        modified=config();modified['astra'].update(url='https://else:DIFFERENT@relay.example/other?anything=CHANGED',api_key='CHANGED')
        self.assertEqual(first['fingerprint'],build_manifest({'runtime.py':'hash'},modified,{'mode':'hybrid'})['fingerprint'])
        self.assertEqual(first['identity']['models']['astra']['endpoint_origin'],'https://relay.example')
        self.assertEqual(first['identity']['models']['jev']['standard_api_path'],'/api/alpha/decisions')
        modified['astra']['max_completion_tokens']=1500
        self.assertNotEqual(first['fingerprint'],build_manifest({'runtime.py':'hash'},modified,{'mode':'hybrid'})['fingerprint'])

    def test_observed_environment_uses_bridge_evidence_and_preserves_unknowns(self):
        w=World();w.ingest(frame_message(30));w.capabilities.update(repentance_plus=False,protocol=1,unknown_secret='HIDDEN')
        env=observed_environment(w)
        self.assertFalse(env['capabilities']['repentance_plus'])
        self.assertNotIn('unknown_secret',env['capabilities'])
        self.assertNotIn('game_version',env)
        self.assertEqual(observed_environment(w,{})['capabilities'],{})
        w.payload.pop('PLAYER_STATS',None)
        self.assertIsNone(observed_environment(w)['player_type'])

    def test_campaign_checks_trial_hash_source_binding_and_legacy_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for name in ('valid','tampered','changed-source','legacy','bad-shape'):
                p=write(root,name,[data(0)]);paths.append(p)
                source={'agent.py':'original'};(p.parent/'source_manifest.json').write_text(json.dumps(source))
                if name=='legacy':continue
                value=build_manifest(source,config(),{'mode':'hybrid'})
                if name=='tampered':value['identity']['execution']['mode']='local'
                if name=='changed-source':(p.parent/'source_manifest.json').write_text(json.dumps({'agent.py':'changed'}))
                if name=='bad-shape':value=[]
                (p.parent/'trial_manifest.json').write_text(json.dumps(value))
            report=summarize(paths,'seed')
            self.assertEqual([s['trial_manifest_status'] for s in report['segment_metrics']],
                             ['verified','invalid_fingerprint_or_schema','source_manifest_mismatch','missing','invalid_json_or_shape'])
            self.assertEqual(report['recorded_trial_manifest_variants'],1)
            self.assertEqual(report['segments_without_verified_trial_manifest'],4)

    def test_cli_records_effective_observe_only_mode_without_api_or_game(self):
        from isaac_agent.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=root/'empty.toml';cfg.write_text('')
            output=root/'trial'
            argv=['isaac_agent','live','--observe-only','--mode','hybrid','--seconds','0.1',
                  '--config',str(cfg),'--run-dir',str(output)]
            with patch('sys.argv',argv),patch('isaac_agent.__main__.serve',new_callable=AsyncMock) as serve,redirect_stdout(io.StringIO()):
                main()
            manifest=json.loads((output/'trial_manifest.json').read_text())
            self.assertEqual(manifest['identity']['execution']['mode'],'local')
            self.assertTrue(manifest['identity']['execution']['observe_only'])
            self.assertEqual(serve.await_count,1) # Mock: no listener or model request.
            rows=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows],['start','trial_manifest'])
            self.assertEqual(rows[-1]['fingerprint'],manifest['fingerprint'])

    def test_environment_events_are_attributed_by_observed_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=write(Path(tmp),'mixed',[data(0),{'event':'environment_observed','level':'seed:1:0',
                    'frame':1,'capabilities':{'repentance_plus':False}},
                    {'event':'environment_observed','level':'other:1:0','frame':2,'capabilities':{'repentance_plus':True}}])
            s=summarize([p],'seed')['segment_metrics'][0]
            self.assertEqual(len(s['observed_environments']),1)
            self.assertFalse(s['observed_environments'][0]['capabilities']['repentance_plus'])
