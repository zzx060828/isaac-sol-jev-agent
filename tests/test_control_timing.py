from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from isaac_agent.runtime import Controller
from scripts.campaign_report import summarize
from test_agent import world,MemoryLog
from test_segment_metrics import data,input_row,write


class ControlTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_step_measures_elapsed_time_and_manual_gate_is_not_zero_work_sample(self):
        w=world();c=Controller(SimpleNamespace(mode='local'),MemoryLog())
        try:
            with patch('isaac_agent.runtime.time.perf_counter',side_effect=(10,10.04)):
                c.step(w)
            rows=[r for r in c.log.rows if r['event']=='input']
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['control_step_ms'],40)
            w.control_mode='MANUAL';c.step(w)
            self.assertEqual(len([r for r in c.log.rows if r['event']=='input']),1)
            self.assertTrue(any(r['event']=='control_gate' for r in c.log.rows))
        finally:await c.close()

    def test_report_preserves_missing_invalid_samples_and_uses_measured_duration(self):
        rows=[data(0)]
        for i,value in enumerate((0,10,20,40,100,None,True,-1,'20',float('inf'),float('nan'))):
            row=input_row(i*.04,100+i)
            if value is not None:row['control_step_ms']=value
            rows.append(row)
        with tempfile.TemporaryDirectory() as d:
            result=summarize([write(Path(d),'timing',rows)],'seed')['segment_metrics'][0]
        t=result['control_step_timing']
        self.assertEqual((t['samples'],t['missing_samples'],t['invalid_samples']),(5,1,5))
        self.assertEqual((t['median_ms'],t['p95_ms'],t['max_ms']),(20,100,100))
        self.assertEqual(t['over_30hz_interval_samples'],2)
        self.assertEqual(result['inputs'],11)

    def test_old_and_other_seed_inputs_do_not_become_false_fast_measurements(self):
        rows=[data(0,seed='old'),input_row(0,100),
              data(1,1,seed='new'),input_row(1,1)|{'control_step_ms':80},
              input_row(5,2)|{'control_step_ms':10}]
        with tempfile.TemporaryDirectory() as d:
            path=write(Path(d),'mixed',rows)
            old=summarize([path],'old')['segment_metrics'][0]['control_step_timing']
            new=summarize([path],'new')['segment_metrics'][0]
        self.assertEqual(old['samples'],0);self.assertEqual(old['missing_samples'],1)
        self.assertIsNone(old['median_ms']);self.assertIsNone(old['p95_ms']);self.assertIsNone(old['max_ms'])
        self.assertEqual(new['control_step_timing']['samples'],2)
        self.assertEqual(new['control_step_timing']['median_ms'],45)
        self.assertEqual(new['excluded_input_intervals'],1) # A wall-time gap isn't step latency.
