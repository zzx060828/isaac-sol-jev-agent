import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from isaac_agent.evaluation import (StartGuard, bind_case, claim_case, development_seeds,
                                    freeze, load_batch, verify_sources)
from isaac_agent.models import Models, configuration
from isaac_agent.runtime import BridgeServer, Controller
from isaac_agent.demo import frame_message
from test_agent import MemoryLog
from test_advance import setup
from test_fast_policy import world as pickup_world


def models(scope):
    return SimpleNamespace(mode='hybrid', jev_scope=scope,
        plan=AsyncMock(return_value=({'objective':'Continue safely', 'mode':'collect'}, {})),
        choose_goal=AsyncMock(return_value=('defer', {'confidence':1})),
        choose_tactic=AsyncMock(return_value=({'target_enemy':'19','posture':'range'},
                                            {'confidence_by_part':{'target':.9,'posture':.9}})),
        retrieve_memories=AsyncMock(return_value=([], {})))


def registration():
    return dict(case=dict(id='s01-off', seed=222, jev_scope='off'), expected=dict(
        player_type=1, difficulty=0, repentance_plus=False, fresh_frame_limit=120))


class AblationTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_public_methods_cannot_send_http(self):
        m = Models.__new__(Models); m.jev_scope = 'off'
        with patch('isaac_agent.models.post') as post:
            for method, args in ((m.decide, ({},{},[])), (m.choose_tactic, ({},{},[])),
                                 (m.choose_goal, ({},[])), (m.retrieve_memories, ({},[]))):
                with self.assertRaises(ValueError): await method(*args)
            post.assert_not_called()

    def test_sol_only_needs_no_jev_key_but_still_requires_sol(self):
        config = configuration()
        with patch('isaac_agent.models.load_keys'), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError): Models(config, 'hybrid', 'off')
            os.environ[config['astra']['key_env']] = 'test-only'
            m = Models(config, 'hybrid', 'off')
            self.assertEqual(m.mode, 'hybrid')
            with self.assertRaises(ValueError): Models(config, 'hybrid', 'tactics')
            with self.assertRaises(ValueError): Models(config, 'jev', 'off')

    async def test_combat_continues_with_no_jev_and_tactics_arm_calls_it(self):
        for scope in ('off', 'tactics'):
            m, w = models(scope), setup()
            c = Controller(m, MemoryLog())
            try:
                command = c.step(w)
                await asyncio.sleep(0)
                self.assertEqual(command['agent_deadline'], w.frame+6)
                self.assertEqual(m.choose_tactic.await_count, int(scope == 'tactics'))
                self.assertTrue(any(r['event']=='input' for r in c.log.rows))
                m.choose_goal.assert_not_awaited()
            finally: await c.close()

    async def test_both_arms_send_fast_goal_to_sol_and_leave_original_all_mode(self):
        for scope in ('off', 'tactics', 'all'):
            m = models(scope); c = Controller(m, MemoryLog())
            try:
                c.step(pickup_world())
                await asyncio.sleep(0)
                self.assertEqual(m.choose_goal.await_count, int(scope == 'all'))
                self.assertEqual(m.plan.await_count, int(scope != 'all'))
            finally: await c.close()

    async def test_sol_memory_input_is_identical_in_both_arms(self):
        for scope in ('off', 'tactics'):
            m, w = models(scope), setup(); c = Controller(m, MemoryLog())
            observation = w.observation()
            observation['memory_candidates'] = [{'id':str(n)} for n in range(6)]
            try:
                await c._plan(w, observation, w.token, w.revision)
                m.retrieve_memories.assert_not_awaited()
                self.assertEqual(m.plan.await_args.args[0]['retrieved_memories'],
                                 [{'id':str(n)} for n in range(4)])
            finally: await c.close()

    async def test_isolated_trial_does_not_load_previous_case_memory_or_backoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); run = parent/'new'; run.mkdir()
            (parent/'run-memory.json').write_text('{"seed":"222","nodes":[]}')
            (parent/'planner-backoff.json').write_text('{"retry_at_unix":9999999999,"failures":9}')
            log = MemoryLog(); log.directory = run; log.isolated_memory = True
            bridge = BridgeServer(models('off'), log)
            try:
                self.assertIsNone(bridge.world.run_memory.loaded)
                self.assertEqual(bridge.world.run_memory.path, run/'run-memory.json')
                self.assertEqual(bridge.controller.backoff_path, run/'planner-backoff.json')
                self.assertEqual(bridge.controller.plan_failures, 0)
            finally: await bridge.controller.close()


class EvaluationTests(unittest.TestCase):
    def test_initial_pair_rejects_changed_pickup_before_accepting_and_keeps_evidence(self):
        from isaac_agent.evaluation import START_CHANNELS
        from isaac_agent.state import World
        with tempfile.TemporaryDirectory() as tmp:
            w = World(level='222:1:0', room=84, frame=5)
            w.payload = dict(PLAYER_STATS=[{'player_type':1}], ROOM_INFO={'difficulty':0},
                PLAYER_HEALTH=[{'red_hearts':8}], PLAYER_INVENTORY=[{'bombs':0}],
                PLAYER_POSITION=[{'pos':{'x':320,'y':380}}], ROOM_LAYOUT={'grid':{}},
                PICKUPS=[{'id':6,'variant':100,'sub_type':629,'price':0,'wait':18}])
            w.sampled = dict.fromkeys(START_CHANNELS, 5)
            w.capabilities.update(repentance_plus=False, continued_run=False)
            reg = registration(); reg['batch_fingerprint'] = 'test-batch'
            reg['expected']['match_initial_state'] = True
            reg['initial_state'] = dict(directory=tmp, reference_case='s01-off')
            guard = StartGuard(reg)
            w.sampled.pop('PICKUPS')
            self.assertFalse(guard.check(w))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            w.sampled['PICKUPS'] = 5
            self.assertTrue(guard.check(w))
            # Animation progress and per-run entity IDs are not different opportunities.
            reg2=deepcopy(reg); reg2['case'].update(id='s01-tactics', jev_scope='tactics')
            w.payload['PICKUPS'][0].update(id=99, wait=14)
            self.assertTrue(StartGuard(reg2).check(w))
            # A separate observed mismatch must remain recorded and cannot enable control.
            reg3=deepcopy(reg2); reg3['case']['id']='mismatch'
            w.payload['PICKUPS'][0]['sub_type']=153
            rejected=StartGuard(reg3)
            with self.assertRaisesRegex(RuntimeError, 'initial state mismatch: pickups'):
                rejected.check(w)
            self.assertFalse(rejected.accepted)
            receipt=json.loads((Path(tmp)/'mismatch.json').read_text())
            self.assertFalse(receipt['matched'])
            self.assertEqual(receipt['state']['pickups'][0]['sub_type'],153)
            with self.assertRaises(RuntimeError):StartGuard(reg2).check(w)

    def test_initial_state_rejects_missing_or_tampered_reference(self):
        from isaac_agent.evaluation import record_initial_state
        from isaac_agent.state import World
        with tempfile.TemporaryDirectory() as tmp:
            w=World(level='222:1:0', room=84, frame=5)
            w.payload=dict(PLAYER_STATS=[], PLAYER_HEALTH=[], PLAYER_INVENTORY=[],
                           PLAYER_POSITION=[], ROOM_INFO={}, ROOM_LAYOUT={}, PICKUPS=[])
            reg=registration(); reg['batch_fingerprint']='test-batch'
            reg['initial_state']=dict(directory=tmp, reference_case='s01-off')
            second=deepcopy(reg); second['case']['id']='s01-tactics'
            with self.assertRaises(RuntimeError):record_initial_state(second,w)
            record_initial_state(reg,w)
            path=Path(tmp)/'s01-off.json'; receipt=json.loads(path.read_text())
            receipt['state']['inventory']={'coins':99}; path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError,'reference is invalid'):
                record_initial_state(second,w)

    def test_holdout_overlap_source_drift_and_secret_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'isaac_agent').mkdir()
            code = root/'isaac_agent/code.py'; code.write_text('original')
            (root/'config.local.toml').write_text('PRIVATE-MARKER')
            (root/'.secrets').mkdir(); (root/'.secrets/key.json').write_text('PRIVATE-MARKER')
            log = root/'events.jsonl'
            log.write_text('{"agent":{"level_id":"222:1:0"}}\n{"seed":333}\n')
            seen = development_seeds([log]); self.assertEqual(seen, {222,333})
            with self.assertRaises(ValueError):freeze(root, root/'bad', {}, [222], seen)
            batch = freeze(root, root/'batch', {}, [444,555], seen)
            verify_sources(root, batch); verify_sources(root/'batch/snapshot', batch)
            self.assertNotIn('PRIVATE-MARKER', json.dumps(batch))
            self.assertFalse((root/'batch/snapshot/.secrets').exists())
            self.assertEqual([c['jev_scope'] for c in batch['identity']['cases']],
                             ['off','tactics','tactics','off'])
            self.assertEqual(load_batch(root/'batch/batch.json'), batch)
            code.write_text('changed')
            with self.assertRaises(ValueError):verify_sources(root, batch)

    def test_case_binds_model_settings_execution_installed_lua_and_single_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); lua = root/'build/SocketBridge_AstraJev/main.lua'
            lua.parent.mkdir(parents=True); lua.write_text('frozen-lua')
            config = configuration(); freeze(root, root/'batch', config, [444], {222})
            path = root/'batch/batch.json'
            args = SimpleNamespace(installed_lua=lua, mode='hybrid', jev_scope='off', isolated_memory=True,
                                   observe_only=False, seconds=900, stop_after_floor=None, until_run_end=True)
            registered = bind_case(root, path, 's01-off', config, args)
            claim_case(path, registered, root/'run')
            with self.assertRaises(FileExistsError):claim_case(path, registered, root/'retry')
            args.jev_scope='all'
            with self.assertRaises(ValueError):bind_case(root, path, 's01-off', config, args)
            args.jev_scope='off'; altered=deepcopy(config); altered['astra']['timeout']=26
            with self.assertRaises(ValueError):bind_case(root, path, 's01-off', altered, args)
            installed=root/'old.lua'; installed.write_text('old-lua'); args.installed_lua=installed
            with self.assertRaises(ValueError):bind_case(root, path, 's01-off', config, args)

    def test_observed_start_rejects_wrong_seed_continue_late_start_and_reset(self):
        def ready():
            w=setup(); w.level='222:1:0'; w.frame=5
            w.sampled['PLAYER_STATS']=w.sampled['ROOM_INFO']=5
            w.stats['player_type']=1; w.info['difficulty']=0
            w.capabilities.update(repentance_plus=False, continued_run=False)
            return w
        for change in (lambda w:setattr(w,'level','223:1:0'),
                       lambda w:setattr(w,'frame',121),
                       lambda w:w.capabilities.update(continued_run=True),
                       lambda w:w.info.update(difficulty=1)):
            w=ready(); change(w)
            with self.assertRaises(RuntimeError):StartGuard(registration()).check(w)
        w=ready(); w.capabilities.pop('continued_run')
        self.assertFalse(StartGuard(registration()).check(w))
        w=ready(); guard=StartGuard(registration()); self.assertTrue(guard.check(w))
        w.level='222:2:0'; self.assertTrue(guard.check(w))
        w.epoch+=1
        with self.assertRaises(RuntimeError):guard.check(w)


class BridgeEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrong_seed_cannot_enable_control_or_call_models(self):
        log=MemoryLog(); log.evaluation=registration(); m=models('off')
        bridge=BridgeServer(m,log)
        reader=asyncio.StreamReader(); reader.feed_data((json.dumps(frame_message())+'\n').encode()); reader.feed_eof()
        class Writer:
            def __init__(self):self.commands=[]
            def write(self,data):self.commands.append(json.loads(data))
            async def drain(self):pass
            def close(self):pass
            async def wait_closed(self):pass
        writer=Writer()
        await bridge.handle(reader,writer)
        self.assertTrue(bridge.run_ended.is_set())
        self.assertTrue(any(r['event']=='evaluation_rejected' for r in log.rows))
        self.assertFalse(any(c.get('params',{}).get('mode')=='FORCE_AI' for c in writer.commands))
        self.assertFalse(any('move' in c for c in writer.commands))
        m.plan.assert_not_awaited(); m.choose_tactic.assert_not_awaited()
