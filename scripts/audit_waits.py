"""Audit sampled neutral control states while Sol was pending; no API or inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaac_agent.control import pickup_target
from isaac_agent.pickups import routine, useful
from isaac_agent.state import World


def audit(path):
    world=World();request=None;rows=[];counts=Counter()
    with path.open() as stream:
        for line in stream:
            r=json.loads(line);event=r['event']
            if event=='bridge':world.ingest(r['message'])
            elif event=='plan_request':request=r
            elif event in ('plan','plan_error','plan_discarded','plan_skipped'):request=None
            elif event=='control_state' and request:
                trace=r.get('decision_trace',{})
                action=trace.get('actual_action')
                if action is None:
                    action=next(iter(r.get('candidates',[])),{}).get('action')
                if action!='m0s0':continue
                plan=r['execution_plan'];excluded=set(plan.get('excluded_choices',[]))
                eligible=[p for p in world.pickups if useful(world,p)]
                target=pickup_target(world,plan,basic_only=True)
                reason=('current_policy_has_local_target' if target is not None else
                        'useful_resources_but_no_current_target' if eligible else
                        'routine_resources_not_currently_useful' if any(routine(p) for p in world.pickups) else
                        'no_routine_resources_observed')
                counts[reason]+=1
                rows.append(dict(frame=r['frame'],room=world.room,
                    request_frame=request.get('observation',{}).get('frame'),
                    neutral_evidence='actual_action' if trace else 'legacy_top_candidate_only',
                    current_policy_class=reason,current_policy_target=target,
                    recorded_navigation_target=plan.get('navigation_target'),
                    recorded_strategy_choice=plan.get('strategy_choice'),excluded_choices=sorted(excluded),
                    pickups=[{k:p.get(k) for k in ('id','variant','sub_type','price','options_index','wait')}
                             for p in world.pickups],
                    request_choices=[o['choice'] for o in request.get('observation',{}).get('strategy_offers',[])],
                    shield_override=trace.get('shield_override')))
    return dict(run=path.parent.name,sampled_neutral_pending=len(rows),classes=dict(counts),samples=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=dict(scope='Historical sampled control states with current pickup policy; no input or model calls.',
                limitations='Legacy top candidate is not necessarily the sent action. No interpolation between '
                    'samples, no simulated movement/drops, and no inference of avoidable waiting. '
                    'Current target reconstruction does not replay the full controller or its persistent memory.',
                runs=[audit(p) for p in args.logs])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in result['runs']],indent=2))
