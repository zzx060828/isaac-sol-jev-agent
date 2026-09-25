import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.campaign_report import summarize


def data(stamp,frame=100,seed='seed',coins=10,keys=1,bombs=3):
    return dict(time=stamp,event='bridge',message=dict(type='DATA',frame=frame,room_index=1,
        agent={'level_id':seed+':1:0'},payload={'PLAYER_POSITION':[{'pos':{'x':320,'y':280}}],
        'PLAYER_INVENTORY':[dict(coins=coins,keys=keys,bombs=bombs,collectibles={})]}))


def input_row(stamp,frame,action='m0s0'):
    return dict(time=stamp,event='input',frame=frame,action=action,source='local')


def write(root,name,rows):
    path=root/name/'events.jsonl';path.parent.mkdir()
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows));return path


class SegmentMetricTests(unittest.TestCase):
    def test_neutral_pending_overlap_gaps_and_resource_deltas_have_distinct_meanings(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[data(0),input_row(0,100),
                dict(time=.01,event='plan_request',observation={'level':'seed:1:0'}),
                input_row(.04,101),input_row(.07,102,'m1s0'),
                dict(time=.08,event='plan',plan={},model='sol',usage={'prompt_tokens':100,'completion_tokens':20}),
                data(.1,103,coins=17,keys=0,bombs=4),input_row(.1,103),input_row(5,104),
                dict(time=5,event='bridge',message=dict(type='EVENT',event='PLAYER_DAMAGE',frame=102,data={})),
                dict(time=5.1,event='bridge',message=dict(type='EVENT',event='PLAYER_DAMAGE',frame=104,data={})),
                dict(time=6,event='bridge',message=dict(type='EVENT',event='GAME_END',frame=104,data={'reason':'exit_save'}))]
            r=summarize([write(root,'background-test',rows)],'seed');s=r['segment_metrics'][0]
            self.assertEqual(s['input_wall_span_seconds'],5)
            self.assertAlmostEqual(s['contiguous_input_seconds'],.1)
            self.assertAlmostEqual(s['neutral_input_during_sol_pending_seconds'],.06)
            self.assertEqual(s['excluded_input_intervals'],1)
            self.assertEqual(s['net_counter_changes'],{'coins':7,'keys':-1,'bombs':1})
            self.assertEqual(s['damage_events'],2);self.assertEqual(s['early_delivery_damage_uncertain'],1)
            self.assertFalse(s['victory_verified']);self.assertEqual(s['exit_events'],{'exit_save':1})
            self.assertEqual(r['reported_model_usage']['sol']['input_tokens'],100)

    def test_late_reply_usage_is_attributed_to_request_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=write(root,'mixed',[data(0,seed='old'),
                dict(time=.1,event='plan_request',observation={'level':'old:1:0'}),
                dict(time=.2,event='bridge',message=dict(type='EVENT',event='GAME_START',data={'continued':False})),
                data(.3,1,seed='new'),
                dict(time=.4,event='plan_discarded',model='sol',usage={'input_tokens':30,'output_tokens':5})])
            old=summarize([path],'old');new=summarize([path],'new')
            self.assertEqual(old['reported_model_usage']['sol']['output_tokens'],5)
            self.assertFalse(new['reported_model_usage'])
            self.assertFalse(old['segment_metrics'][0]['unfinished_logged_requests'])

    def test_cache_wait_without_reply_does_not_close_model_request(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=write(root,'probe',[
                dict(time=0,event='advance_request',observation={'level':'seed:1:0'}),
                dict(time=.1,event='advance_waiting_for_state',source_frame=1),
                dict(time=.2,event='advance_waiting_for_state',plan={},model='sol',usage={'input_tokens':50})])
            r=summarize([path],'seed');s=r['segment_metrics'][0]
            self.assertEqual(s['logged_responses'],{'advance_waiting_for_state':1})
            self.assertEqual(s['inputs'],0);self.assertEqual(r['reported_model_usage']['sol']['input_tokens'],50)
            self.assertFalse(s['unfinished_logged_requests'])

    def test_cli_includes_non_live_names_and_excludes_other_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=write(root,'background-test',[data(0),input_row(.1,100)])
            b=write(root,'recorded-floor',[data(1),input_row(1.1,101)])
            write(root,'unrelated',[data(0,seed='other')])
            for p in (a,b):(p.parent/'source_manifest.json').write_text(json.dumps({'a.py':'same_hash'}))
            command=[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/campaign_report.py'),
                     '--seed','seed','--runs-dir',str(root)]
            r=json.loads(subprocess.check_output(command,text=True))
            self.assertEqual(len(r['controller_segments']),2)
            self.assertEqual(r['recorded_python_manifest_variants'],1)
            self.assertFalse(r['victory_verified'])
