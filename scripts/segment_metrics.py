"""Evidence metrics for one controller segment, without inferring a win rate."""
from collections import Counter
import hashlib
import json
import math
import statistics

from isaac_agent.state import first, xy

REQUESTS={'plan_request':'sol','advance_request':'sol_advance','tactic_request':'jev_tactic',
          'decision_request':'jev_action','fast_goal_request':'jev_goal'}
RESPONSES={
    **{k:'sol' for k in ('plan','plan_error','plan_discarded','plan_skipped')},
    **{k:'sol_advance' for k in ('advance_ready','advance_error','advance_discarded','advance_waiting_for_state')},
    **{k:'jev_tactic' for k in ('tactic','tactic_error','tactic_discarded')},
    **{k:'jev_action' for k in ('decision','decision_error','decision_discarded')},
    **{k:'jev_goal' for k in ('fast_goal','fast_goal_deferred','fast_goal_discarded','fast_goal_error')}}


class SegmentMetrics:
    def __init__(self,seed):
        self.seed=str(seed);self.level='';self.first_data_frame=None;self.pending={}
        self.requests=Counter();self.responses=Counter();self.usage={};self.errors=0
        self.first_input=self.last_input=None;self.previous_input=None;self.input_count=0
        self.intervals=[];self.sol_windows=[];self.skipped_intervals=0;self.position=None
        self.first_inventory=self.last_inventory=None
        self.damage_count=0;self.early_damage=0;self.deaths=0;self.exits=Counter()
        self.pickup_failures=0;self.corrupt_lines=0;self.environments=[]
        self.control_step_times=[];self.control_step_missing=0;self.control_step_invalid=0

    def matches(self,level):return str(level).split(':')[0]==self.seed

    def consume(self,row):
        event=row.get('event');stamp=row.get('time')
        if event=='environment_observed' and self.matches(row.get('level','')):
            self.environments.append({k:v for k,v in row.items() if k not in ('event','time')})
        if event=='connected':
            self.first_data_frame=None;self.previous_input=None
        if event=='bridge':
            msg=row['message'];kind=msg.get('type');name=msg.get('event')
            if name=='GAME_START':
                self.level='';self.first_data_frame=None;self.previous_input=None
            if kind in ('DATA','FULL'):
                level=str(msg.get('agent',{}).get('level_id',self.level))
                if level.split(':')[0] != self.level.split(':')[0]:
                    self.first_data_frame=None;self.previous_input=None
                self.level=level
                if self.first_data_frame is None:self.first_data_frame=msg.get('frame')
                payload=msg.get('payload',{})
                if 'PLAYER_POSITION' in payload:self.position=xy(first(payload['PLAYER_POSITION']).get('pos'))
                if self.matches(self.level) and 'PLAYER_INVENTORY' in payload:
                    inv=first(payload['PLAYER_INVENTORY'])
                    counters={k:inv.get(k) for k in ('coins','keys','bombs')}
                    if all(isinstance(v,(int,float)) for v in counters.values()):
                        self.first_inventory=self.first_inventory or counters
                        self.last_inventory=counters
            if self.matches(self.level):
                if name=='PLAYER_DAMAGE':
                    self.damage_count+=1
                    if self.first_data_frame is None or msg.get('frame',self.first_data_frame)<=self.first_data_frame+3:
                        self.early_damage+=1
                elif name=='PLAYER_DEATH':self.deaths+=1
                elif name=='GAME_END':self.exits[str(msg.get('data',{}).get('reason','unknown'))]+=1
        if event in REQUESTS:
            lane=REQUESTS[event];source=row.get('observation',{}).get('level',self.level)
            self.pending[lane]={'level':source,'time':stamp}
            if self.matches(source):self.requests[lane]+=1
        lane=RESPONSES.get(event)
        # current() can log a cache wait without a new model response. Only
        # _run's wait row carries the returned plan and closes an API request.
        if event=='advance_waiting_for_state' and 'plan' not in row:lane=None
        ticket=self.pending.get(lane) if lane else None
        source=ticket['level'] if ticket else self.level
        if lane and ticket:
            if self.matches(source):
                self.responses[event]+=1
                if event.endswith('error'):self.errors+=1
                if lane=='sol' and ticket['time'] is not None and stamp is not None:
                    self.sol_windows.append((ticket['time'],stamp))
            del self.pending[lane]
        # Memory retrieval has no separate request event in existing logs; use
        # its parent Sol request for attribution instead of inventing a count.
        if event.startswith('memory_retrieval') and 'sol' in self.pending:source=self.pending['sol']['level']
        usage=row.get('usage')
        if self.matches(source) and isinstance(usage,dict) and usage:
            model=row.get('model','unknown_model')
            entry=self.usage.setdefault(model,dict(usage_rows=0,input_tokens=0,output_tokens=0,reported_cost=None))
            entry['usage_rows']+=1
            entry['input_tokens']+=usage.get('input_tokens',usage.get('prompt_tokens',0)) or 0
            entry['output_tokens']+=usage.get('output_tokens',usage.get('completion_tokens',0)) or 0
            if isinstance(usage.get('cost'),(int,float)):
                entry['reported_cost']=(entry['reported_cost'] or 0)+usage['cost']
        if not self.matches(self.level):return
        if event=='local_pickup_failed':self.pickup_failures+=1
        if event=='input' and isinstance(stamp,(int,float)):
            duration=row.get('control_step_ms')
            if duration is None:self.control_step_missing+=1
            elif type(duration) in (int,float) and math.isfinite(duration) and duration>=0:
                self.control_step_times.append(duration)
            else:self.control_step_invalid+=1
            current=dict(time=stamp,frame=row.get('frame'),level=self.level,
                         action=row.get('action'),position=self.position)
            self.first_input=self.first_input or current;self.last_input=current;self.input_count+=1
            prev=self.previous_input
            if prev:
                dt=stamp-prev['time']
                df=current['frame']-prev['frame'] if isinstance(current['frame'],(int,float)) and isinstance(prev['frame'],(int,float)) else 0
                if prev['level']==self.level and 0<df<=6 and 0<dt<=.5:
                    self.intervals.append((prev['time'],stamp,prev['action']=='m0s0'))
                else:self.skipped_intervals+=1
            self.previous_input=current

    def result(self,path):
        windows=list(self.sol_windows)
        if self.last_input and 'sol' in self.pending and self.matches(self.pending['sol']['level']):
            windows.append((self.pending['sol']['time'],self.last_input['time']))
        active=sum(b-a for a,b,_ in self.intervals)
        waiting=sum(max(0,min(b,end)-max(a,start)) for a,b,neutral in self.intervals if neutral
                    for start,end in windows if start is not None)
        manifest=path.parent/'source_manifest.json';fingerprint=None
        if manifest.is_file():
            try:fingerprint=hashlib.sha256(json.dumps(json.loads(manifest.read_text()),sort_keys=True).encode()).hexdigest()
            except ValueError:pass
        trial_path=path.parent/'trial_manifest.json';trial=None;trial_status='missing'
        if trial_path.is_file():
            try:
                from isaac_agent.trial_metadata import fingerprint as metadata_fingerprint
                value=json.loads(trial_path.read_text())
                if (value.get('identity',{}).get('schema_version') == 1
                        and value.get('fingerprint') == metadata_fingerprint(value['identity'])):
                    if (manifest.is_file() and value['identity'].get('package_source_fingerprint')
                            == metadata_fingerprint(json.loads(manifest.read_text()))):
                        trial=value;trial_status='verified'
                    else:trial_status='source_manifest_mismatch'
                else:trial_status='invalid_fingerprint_or_schema'
            except (ValueError,KeyError,TypeError,AttributeError):trial_status='invalid_json_or_shape'
        directory=path.resolve().parents[2]/'recordings'/path.parent.name
        durations=sorted(self.control_step_times)
        timing=dict(samples=len(durations),missing_samples=self.control_step_missing,
            invalid_samples=self.control_step_invalid,
            median_ms=round(statistics.median(durations),3) if durations else None,
            p95_ms=durations[math.ceil(len(durations)*.95)-1] if durations else None,
            max_ms=durations[-1] if durations else None,
            over_30hz_interval_samples=sum(v>1000/30 for v in durations),
            scope='Elapsed synchronous Controller.step through command/diagnostic preparation, before input log write. '
                  'Includes synchronous work/logging inside step; excludes bridge ingest, input log write, socket send and awaited model latency. '
                  'p95 uses nearest rank. 30Hz comparison is a reference interval, not measured missed frames. Missing old samples are unknown.')
        return dict(log=str(path),inputs=self.input_count,
            control_step_timing=timing,
            first_input_frame=self.first_input['frame'] if self.first_input else None,
            last_input_frame=self.last_input['frame'] if self.last_input else None,
            input_wall_span_seconds=round(self.last_input['time']-self.first_input['time'],3) if self.first_input else None,
            contiguous_input_seconds=round(active,3),neutral_input_during_sol_pending_seconds=round(waiting,3),
            included_input_intervals=len(self.intervals),excluded_input_intervals=self.skipped_intervals,
            logged_requests=dict(self.requests),logged_responses=dict(self.responses),model_error_responses=self.errors,
            unfinished_logged_requests=[lane for lane,t in self.pending.items() if self.matches(t['level'])],
            reported_model_usage=self.usage,damage_events=self.damage_count,
            early_delivery_damage_uncertain=self.early_damage,player_death_callbacks=self.deaths,
            exit_events=dict(self.exits),victory_verified=False,local_pickup_failures=self.pickup_failures,
            initial_resource_counters=self.first_inventory,final_resource_counters=self.last_inventory,
            net_counter_changes={k:self.last_inventory[k]-v for k,v in self.first_inventory.items()}
                if self.first_inventory and self.last_inventory else None,
            recorded_python_manifest_fingerprint=fingerprint,
            trial_manifest_status=trial_status,trial_manifest=trial,observed_environments=self.environments,
            recording_files=[str(p) for p in sorted(directory.glob('*')) if p.suffix.lower() in ('.mp4','.mkv')]
                if directory.is_dir() else [],malformed_rows=self.corrupt_lines,
            limitations='Adjacent input intervals require 1..6 frame progress and <=0.5s wall time; gaps are excluded, not assumed idle. '
                'Neutral input can retain motion and is not proof of avoidable waiting. Counter deltas are net changes, not gross loot or profit. '
                'Early damage may be queued; callbacks are not deduplicated or equated to HP lost. '
                'Only returned usage is counted; missing or internal request logs prevent a complete bill. '
                'Exit callbacks do not prove victory. Source fingerprint describes the recorded Python manifest, not full game/config identity.')
