import asyncio
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from isaac_agent.combat_progress import CombatProgress
from isaac_agent.demo import frame_message
from isaac_agent.fast_policy import lane
from isaac_agent.models import Models, validate_plan, PlanValidationError
from isaac_agent.review import fingerprint
from isaac_agent.runtime import Controller
from isaac_agent.state import World
from isaac_agent.strategy import offers
from test_planning import MemoryLog
from test_safety_strategy import refresh


def setup(combat=False):
    w=World();w.ingest(frame_message(enemies=[dict(id=1,type=18,hp=100,pos={'x':470,'y':280})] if combat else []))
    w.payload.update(BOMBS=[],FIRE_HAZARDS=[],INTERACTABLES=[])
    refresh(w)
    return w


def trinket_room():
    w=setup()
    w.payload['PICKUPS']=[dict(id=7,variant=350,sub_type=126,price=0,pos={'x':400,'y':280})]
    w.inventory['trinket_0']=102
    w.layout['grid']={'10':dict(type=14,variant=0,collision=1,x=200,y=280)}
    return w


class ProgressTests(unittest.TestCase):
    def test_damage_kills_stalls_and_prolonged_spawns(self):
        w=setup(True);p=CombatProgress();self.assertIsNone(p.update(w))
        for i in range(6):
            refresh(w,100);w.payload['ENEMIES'][0]['hp']-=1
            self.assertIsNone(p.update(w))
            self.assertEqual(lane(w,1,combat_stalled=bool(p.escalated))[0],'tactical')
        refresh(w,359);self.assertIsNone(p.update(w))
        w.events.append({'event':'NPC_DEATH','frame':w.frame})
        self.assertIsNone(p.update(w))
        refresh(w,360);self.assertEqual(p.update(w),'no_damage_or_kill_progress')
        w.payload['ENEMIES'][0]['hp']-=1;refresh(w,1)
        self.assertEqual(p.update(w),'no_damage_or_kill_progress') # one review, no lane oscillation
        w.payload['ENEMIES']=[];self.assertIsNone(p.update(w))
        w=setup(True);p=CombatProgress();p.update(w)
        for i in range(18):
            refresh(w,100);w.payload['ENEMIES'][0]['hp']-=1;p.update(w)
        self.assertEqual(p.escalated,'prolonged_combat')

    def test_spawns_and_healing_do_not_fake_progress_and_boss_is_immediate(self):
        w=setup(True);p=CombatProgress();p.update(w)
        refresh(w,200);w.payload['ENEMIES'][0]['hp']=110;p.update(w)
        refresh(w,160);w.payload['ENEMIES'][0]['hp']=105
        w.payload['ENEMIES'].append(dict(id=2,hp=100))
        self.assertEqual(p.update(w),'no_damage_or_kill_progress')
        w.room=2;p.update(w);self.assertIsNone(p.escalated)
        w.info['room_type']=5;self.assertEqual(lane(w,w.frame)[0],'sol')


class ReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_noncombat_null_range_has_default_and_invalid_reply_keeps_usage(self):
        w=setup();obs=w.observation()
        p=validate_plan({'objective':'Travel','mode':'explore','combat_distance':None},obs)
        self.assertGreater(p['combat_distance'],150)
        with self.assertRaises(ValueError):
            validate_plan({'objective':'Fight','mode':'combat','combat_distance':None},obs)
        model=Models.__new__(Models);model.mode='hybrid';model.config={'astra':{'model':'test'}}
        result={'model':'test','usage':{'prompt_tokens':25,'completion_tokens':10},
                'choices':[{'message':{'content':'{"objective":"Bad","door_slot":"invented"}'}}]}
        with patch('isaac_agent.models.post',return_value=result):
            with self.assertRaises(PlanValidationError) as caught:await model.plan(obs)
        self.assertEqual(caught.exception.metadata['usage']['prompt_tokens'],25)
        c=Controller(SimpleNamespace(mode='hybrid',plan=AsyncMock(side_effect=caught.exception)),MemoryLog())
        try:
            await c._plan(w,obs,w.token,w.revision)
            error=next(r for r in c.log.rows if r['event']=='plan_error')
            self.assertEqual(error['usage']['completion_tokens'],10)
            self.assertNotIn('content',error)
        finally:await c.close()

    async def test_heal_pulse_settles_before_new_pickup_comparison(self):
        w=setup();w.info['is_clear']=False
        w.health.update(red_hearts=4,max_hearts=6)
        w.inventory.update(can_use=True,active_items={'0':{'item':45,'charge':4,'max_charge':4}})
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=({'objective':'Review','mode':'explore'},{})))
        c=Controller(m,MemoryLog());c.last_decision=float('inf')
        try:
            command=c.step(w);self.assertTrue(command.get('use_item'))
            w.info['is_clear']=True
            w.payload['PICKUPS']=[dict(id=7,variant=350,sub_type=126,price=0,pos={'x':400,'y':280})]
            refresh(w,1);c.step(w)
            self.assertEqual(c.route,'settle_transaction');self.assertIsNone(c.plan_task)
            w.health['red_hearts']=6;w.inventory['active_items']['0']['charge']=0
            refresh(w,6);c.step(w);await c.plan_task
            self.assertEqual(m.plan.call_args.args[0]['health']['red_hearts'],6)
            self.assertEqual(m.plan.await_count,1)
        finally:await c.close()

    async def test_scavenge_completion_reuses_explicit_decline_without_second_sol_call(self):
        w=trinket_room();choice='pickup:7:350:126:0'
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=(
            {'objective':'Sweep and keep current trinket','mode':'collect','strategy_choice':'scavenge',
             'declined_pickups':[choice]},{})))
        c=Controller(m,MemoryLog());c.last_decision=float('inf')
        try:
            c.step(w);await c.plan_task
            self.assertEqual(c.pickup_review['choices'],{choice})
            c.step(w);self.assertEqual(c.strategy.choice,'scavenge')
            w.layout['grid']['10']['collision']=0;refresh(w,10)
            c.step(w);await asyncio.sleep(0)
            self.assertIsNone(c.strategy.choice)
            self.assertEqual(c.route,'settle_transaction')
            refresh(w,6);c.step(w)
            self.assertEqual(c.route,'routine')
            self.assertIn(choice,c.execution_plan['excluded_choices'])
            self.assertFalse(c.execution_plan['awaiting_strategy'])
            self.assertEqual(m.plan.await_count,1)
            # A new resource changes opportunity costs and re-opens this review.
            w.inventory['coins']=1;refresh(w,1);c.last_plan=-100
            c.step(w)
            self.assertIsNone(c.pickup_review)
            self.assertEqual(c.route,'sol')
            await c.plan_task;self.assertEqual(m.plan.await_count,2)
        finally:await c.close()

    def test_review_scope_changes_for_resources_build_ground_health_and_room(self):
        w=trinket_room();key=fingerprint(w)
        refresh(w,30);w.player['pos']['x']+=5;w.health['damage_cooldown']=40
        self.assertEqual(fingerprint(w),key)
        for mutation in (
            lambda v:v.health.update(red_hearts=4),
            lambda v:v.inventory.update(coins=5),
            lambda v:v.inventory.update(collectibles={'1':1}),
            lambda v:v.payload['PICKUPS'].append(dict(id=8,variant=10,sub_type=1)),
            lambda v:setattr(v,'room',2),
            lambda v:v.layout['doors']['2'].update(is_locked=True)):
            v=deepcopy(w);mutation(v);self.assertNotEqual(fingerprint(v),key)

    async def test_reply_cannot_install_old_review_after_parallel_resource_collection(self):
        w=trinket_room();obs=w.observation();obs['pickup_review_context']=fingerprint(w)
        async def reply(observation):
            w.inventory['coins']=1
            return {'objective':'Leave it','mode':'explore','declined_pickups':['pickup:7:350:126:0']},{}
        c=Controller(SimpleNamespace(mode='hybrid',plan=reply),MemoryLog())
        try:
            await c._plan(w,obs,w.token,w.revision)
            self.assertIsNone(c.pickup_review)
        finally:await c.close()

    async def test_return_visit_reuses_review_only_with_same_context(self):
        w=trinket_room();choice='pickup:7:350:126:0'
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=(
            {'objective':'Leave it','mode':'collect','declined_pickups':[choice]},{})))
        c=Controller(m,MemoryLog());c.reset_actions(w)
        obs=w.observation();obs['pickup_review_context']=fingerprint(w)
        try:
            await c._plan(w,obs,w.token,w.revision)
            original=w.room;w.room=99;c.reset_actions(w);self.assertIsNone(c.pickup_review)
            w.room=original;c.reset_actions(w);self.assertEqual(c.pickup_review['choices'],{choice})
            w.inventory['coins']=8;c.last_decision=float('inf');c.step(w)
            self.assertIsNone(c.pickup_review);self.assertNotIn(w.token,c.pickup_reviews)
        finally:await c.close()

    def test_declines_must_be_observed_pickups_and_never_selected_contract(self):
        w=trinket_room();obs=w.observation();obs['strategy_offers']=offers(w)
        choice='pickup:7:350:126:0'
        valid=validate_plan({'objective':'Leave trinket','declined_pickups':[choice,choice]},obs)
        self.assertEqual(valid['declined_pickups'],[choice])
        for change in ({'declined_pickups':['scavenge']},{'declined_pickups':['invented']},
                       {'declined_pickups':choice},{'declined_pickups':[choice],'strategy_choice':choice}):
            with self.assertRaises(ValueError):validate_plan({'objective':'Invalid',**change},obs)
