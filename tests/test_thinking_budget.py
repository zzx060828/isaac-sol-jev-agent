import asyncio
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from test_agent import MemoryLog
from test_advance import setup, preparation
from test_safety_strategy import refresh
from isaac_agent.advance import AdvancePlanner
from isaac_agent.control import Action, candidates
from isaac_agent.runtime import Controller
from isaac_agent.memory import RunMemory


def boss_distance(w,hops):
    w.rooms = {str(w.room+i): {'type':1,'shape':1,'doors': {
        '2': {'target_room':w.room+i+1,'target_room_type':5 if i==hops-1 else 1}}}
        for i in range(hops)}


class ThinkingBudgetTests(IsolatedAsyncioTestCase):
    def test_recent_item_reasoning_persists_with_observed_outcome_but_not_across_runs(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'memory.json';w=setup();m=RunMemory(path)
            m.observe(w,{})
            p={'objective':'Prefer soul health','rationale':'Keep renewable protection',
               'strategy_choice':'pickup:2:100:78:0'}
            m.record_decision(w,p)
            self.assertEqual(m.ledger()['recent_decisions'][-1]['status'],'model_intent_not_confirmed')
            w.inventory['active_items']['0']['item']=78
            m.record_outcome(w,p['strategy_choice'],True)
            restored=RunMemory(path);restored.observe(w,{})
            record=restored.ledger()['recent_decisions'][-1]
            self.assertEqual(record['status'],'observed_item_acquisition')
            self.assertEqual(record['active_items_after']['0']['item'],78)
            self.assertEqual(record['rationale'],p['rationale'])
            w.level='other-seed:1:0';restored.observe(w,{})
            self.assertEqual(restored.ledger()['recent_decisions'],[])

    def test_failed_acquisition_is_not_promoted_to_success_and_history_is_bounded(self):
        w=setup();m=RunMemory();m.observe(w,{})
        for i in range(10):m.record_decision(w,{'strategy_choice':str(i),'rationale':'tradeoff'})
        self.assertEqual(len(m.ledger()['recent_decisions']),6)
        m.record_outcome(w,'9',False)
        self.assertEqual(m.ledger()['recent_decisions'][-1]['status'],'ended_without_confirmed_acquisition')

    async def test_far_boss_defers_and_expired_preparation_renews_only_after_approach(self):
        w=setup();log=MemoryLog()
        model=SimpleNamespace(mode='hybrid',prepare=AsyncMock(return_value=(preparation(),{})))
        p=AdvancePlanner(model,log);now=time.monotonic()
        try:
            boss_distance(w,14);p.maybe_start(w,'tactical',False,now)
            self.assertIsNone(p.task)
            self.assertEqual(log.rows[-1]['reason'],'boss too distant')
            boss_distance(w,3);p.maybe_start(w,'tactical',False,now+2);await p.task
            self.assertEqual(model.prepare.await_count,1)
            p.maybe_start(w,'tactical',False,now+190)
            self.assertEqual(model.prepare.await_count,1)  # Same position, no heartbeat refresh.
            boss_distance(w,2);p.maybe_start(w,'tactical',False,now+192);await p.task
            self.assertEqual(model.prepare.await_count,2)
            w.inventory['collectibles']['999']=1
            boss_distance(w,1);p.maybe_start(w,'tactical',False,now+400)
            self.assertEqual(model.prepare.await_count,2)  # Two actual calls per floor.
        finally:await p.close()

    async def test_expired_floor_review_or_changed_build_never_applies_as_boss_plan(self):
        for change in ('expired','review','build'):
            w=setup();m=SimpleNamespace(mode='hybrid',prepare=AsyncMock(return_value=(preparation(),{})))
            c=Controller(m,MemoryLog())
            try:
                c.advance.maybe_start(w,'tactical',False);await c.advance.task
                if change=='expired':c.advance.cache['completed_at']-=181
                if change=='review':c.advance.cache['purpose']='floor_review'
                if change=='build':w.inventory['collectibles']['999']=1
                w.room+=1;w.info['room_type']=5;refresh(w,1)
                c.plan_retry_at=float('inf');c.last_decision=time.monotonic()
                c.step(w)
                self.assertNotEqual(c.plan.get('goal_source'),'advance')
                self.assertFalse(any(r['event']=='advance_applied' for r in c.log.rows))
            finally:await c.close()

    async def test_boss_uses_preparation_once_while_current_assessment_is_pending(self):
        gate=asyncio.Event()
        async def plan(obs):
            await gate.wait()
            return {'objective':'New live assessment','mode':'combat','boss_focus':'balanced'},{}
        w=setup();m=SimpleNamespace(mode='hybrid',prepare=AsyncMock(return_value=(preparation(),{})),
            plan=AsyncMock(side_effect=plan),decide=AsyncMock(return_value=(Action(),{'confidence':0})))
        c=Controller(m,MemoryLog())
        try:
            c.step(w);await c.advance.task
            w.room+=1;w.info['room_type']=5;refresh(w,1)
            c.step(w);await asyncio.sleep(0)
            self.assertEqual(c.execution_plan['boss_focus'],'spawners_first')
            self.assertFalse(c.plan_task.done())
            refresh(w,1);command=c.step(w)
            self.assertIn('move',command)
            applied=[r for r in c.log.rows if r['event']=='advance_applied']
            self.assertEqual(len(applied),1)
            self.assertEqual(applied[0]['frames_since_room_entry'],0)
            gate.set();await c.plan_task
            refresh(w,1);c.step(w)
            self.assertEqual(c.execution_plan['goal_source'],'sol')
            self.assertEqual(c.execution_plan['boss_focus'],'balanced')
        finally:await c.close()

    async def test_low_confidence_backoff_does_not_stop_local_inputs(self):
        w=setup();m=SimpleNamespace(mode='jev',decide=AsyncMock(return_value=(Action(),{'confidence':.1})))
        c=Controller(m,MemoryLog());c.reset_actions(w)
        try:
            options,_=candidates(w,c.execution_plan)
            for delay in (2,4,8,12):
                await c._decide(w,w.observation(),w.token,w.frame,options)
                remaining=c.tactical_retry_at-time.monotonic()
                self.assertAlmostEqual(remaining,delay,delta=.1)
                before=m.decide.await_count
                refresh(w,1);command=c.step(w);await asyncio.sleep(0)
                self.assertIn('move',command)
                self.assertEqual(m.decide.await_count,before)
            self.assertEqual(c.log.rows[-1]['event'],'input')
        finally:await c.close()

    async def test_combat_request_keeps_each_move_once_and_does_not_poll_subsecond(self):
        w=setup();m=SimpleNamespace(mode='jev',decide=AsyncMock(return_value=(Action(),{'confidence':.1})))
        c=Controller(m,MemoryLog())
        try:
            c.step(w);await c.decision_task
            options=m.decide.call_args.args[2]
            moves=[o['action'].move for o in options]
            self.assertEqual(len(moves),len(set(moves)))
            self.assertLessEqual(len(moves),9)
            c.tactical_retry_at=0
            refresh(w,1);c.step(w);await asyncio.sleep(0)
            self.assertEqual(m.decide.await_count,1)
        finally:await c.close()

    async def test_expired_decisions_back_off_without_extending_action_lease(self):
        w=setup()
        async def late(obs,plan,options):
            refresh(w,46)
            return options[0]['action'],{'confidence':.99}
        m=SimpleNamespace(mode='jev',decide=AsyncMock(side_effect=late))
        c=Controller(m,MemoryLog());c.reset_actions(w)
        try:
            for delay in (2,4,8,12):
                options,_=candidates(w,c.execution_plan)
                await c._decide(w,w.observation(),w.token,w.frame,options)
                self.assertEqual(c.action_until,-1)
                self.assertEqual(c.log.rows[-1]['reason'],'observation_expired')
                self.assertEqual(c.log.rows[-1]['retry_after_seconds'],delay)
                before=m.decide.await_count
                self.assertIn('move',c.step(w));await asyncio.sleep(0)
                self.assertEqual(m.decide.await_count,before)
            options,_=candidates(w,c.execution_plan)
            m.decide.side_effect=None
            m.decide.return_value=(options[0]['action'],{'confidence':.99})
            await c._decide(w,w.observation(),w.token,w.frame,options)
            self.assertEqual(c.expired_decision_streak,0)
            self.assertEqual(c.tactical_retry_at,0)
            self.assertEqual(c.action_until,w.frame+6)
            w.room+=1;c.reset_actions(w)
            self.assertEqual(c.expired_decision_streak,0)
        finally:await c.close()
