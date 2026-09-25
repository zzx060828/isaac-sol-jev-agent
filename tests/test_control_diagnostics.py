import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from isaac_agent.runtime import Controller
from isaac_agent.control import Action
from isaac_agent.demo import frame_message
from isaac_agent.state import World
from test_fast_policy import Log
from test_parallel_resources import setup
from test_safety_strategy import refresh


class ControlDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_pickup_trace_records_actual_command_and_pending_sol(self):
        w=setup();w.frame=30;refresh(w)
        gate=asyncio.Event()
        async def plan(obs):await gate.wait()
        c=Controller(SimpleNamespace(mode='hybrid',plan=AsyncMock(side_effect=plan)),Log())
        try:
            command=c.step(w)
            trace=next(r['decision_trace'] for r in c.log.rows if r['event']=='control_state')
            self.assertTrue(trace['sol_pending'])
            self.assertTrue(trace['parallel_resource_collection'])
            self.assertEqual(tuple(trace['local_pickup_target']),(240,280))
            self.assertEqual(trace['actual_action'],next(r['action'] for r in c.log.rows if r['event']=='input'))
            self.assertLess(command['move']['x'],0)
            self.assertEqual(trace['resource_pulses'],[])
        finally:await c.close()

    async def test_closed_gate_is_logged_without_repeated_same_second_or_model_calls(self):
        w=World();w.ingest(frame_message(30));w.control_mode='MANUAL'
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(),decide=AsyncMock())
        c=Controller(m,Log());now=int(w.updated)+.8
        try:
            for _ in range(5):
                command=c.step(w,now)
                self.assertEqual(command['move'],{'x':0,'y':0})
            rows=[r for r in c.log.rows if r['event']=='control_gate']
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['facts']['control_mode'],'MANUAL')
            c.step(w,now+1)
            self.assertEqual(sum(r['event']=='control_gate' for r in c.log.rows),2)
            m.plan.assert_not_called();m.decide.assert_not_called()
            w.control_mode='AUTO';refresh(w);w.updated=now+1
            c.step(w,now+1)
            self.assertTrue(any(r['event']=='control_gate_opened' for r in c.log.rows))
        finally:await c.close()

    async def test_stale_position_is_visible_and_blocks_commands(self):
        w=World();w.ingest(frame_message(30));w.control_mode='AUTO';w.sampled['PLAYER_POSITION']=26
        c=Controller(SimpleNamespace(mode='hybrid'),Log())
        try:
            command=c.step(w,w.updated)
            row=next(r for r in c.log.rows if r['event']=='control_gate')
            self.assertFalse(row['facts']['world_ready'])
            self.assertEqual(row['facts']['sensor_age_updates']['PLAYER_POSITION'],4)
            self.assertEqual(command,Action().wire(w,1))
        finally:await c.close()

    def test_wait_audit_distinguishes_legacy_candidate_from_sent_action(self):
        from scripts.audit_waits import audit
        msg=frame_message(30)
        rows=[{'event':'bridge','message':msg},
              {'event':'plan_request','observation':{'frame':30,'strategy_offers':[]}},
              {'event':'control_state','frame':30,'execution_plan':{},'candidates':[{'action':'m0s0'}]},
              {'event':'control_state','frame':31,'execution_plan':{},'candidates':[{'action':'m0s0'}],
               'decision_trace':{'actual_action':'m3s0'}},
              {'event':'plan','plan':{}},
              {'event':'control_state','frame':32,'execution_plan':{},'candidates':[{'action':'m0s0'}]}]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'events.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=audit(p)
        self.assertEqual(result['sampled_neutral_pending'],1)
        self.assertEqual(result['samples'][0]['neutral_evidence'],'legacy_top_candidate_only')
