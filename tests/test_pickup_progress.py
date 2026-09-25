import asyncio
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.control import pickup_target
from isaac_agent.journey import options
from isaac_agent.pickup_progress import PickupProgress, blocked_choices
from isaac_agent.planning import enrich_planning
from isaac_agent.resources import ResourcePolicy
from isaac_agent.runtime import Controller
from isaac_agent.state import World, xy
from test_chests import setup
from test_fast_policy import Log
from test_healing_economy import prepared
from test_safety_strategy import refresh


def tick(w, tracker, log, target=(400, 280), frames=1):
    tracker.follow(w, target)
    refresh(w, frames)
    tracker.observe(w, log)


def stationary(w):
    tracker, log = PickupProgress(), Log()
    tracker.observe(w, log)
    for _ in range(120):
        tick(w, tracker, log)
    return tracker, log


class PickupProgressTests(unittest.IsolatedAsyncioTestCase):
    def test_recorded_secret_chest_approach_is_not_discarded_by_guard(self):
        from isaac_agent.control import candidates, physical_collision
        from test_ranged_combat import advance
        data=json.loads((Path(__file__).parent/'fixtures/secret-ordinary-chest.json').read_text())
        w=World();w.ingest(data['snapshot']);t=PickupProgress();log=Log()
        # Static geometry only: contact distance is checked, no engine opening
        # or reward is simulated. Two TNTs stay intact throughout the approach.
        for _ in range(100):
            t.observe(w,log);target=pickup_target(w,{},True)
            self.assertIsNotNone(target)
            t.follow(w,target)
            action=candidates(w,{'strategy_enabled':True,'navigation_target':target})[0][0]['action']
            self.assertEqual(action.shoot,(0,0))
            advance(w,action.move)
            self.assertFalse(physical_collision(w,xy(w.player['pos']),float(w.stats.get('size',12))))
            self.assertFalse(w.pickup_failures)
            if math.dist(xy(w.player['pos']),(320,320))<20:break
        else:self.fail('Did not reach recorded chest contact distance')

    async def test_stationary_chest_yields_to_other_resource_once_without_model(self):
        w=setup();w.payload['PICKUPS'].append(dict(id=21,variant=20,sub_type=1,price=0,pos=dict(x=440,y=280)))
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(),decide=AsyncMock())
        c=Controller(m,Log());c.plan_retry_at=float('inf')
        try:
            c.step(w)
            for _ in range(120):
                refresh(w,1);c.step(w)
            self.assertEqual(c.execution_plan['navigation_target'],(440,280))
            failed=[r for r in c.log.rows if r['event']=='local_pickup_failed']
            self.assertEqual(len(failed),1)
            self.assertEqual(failed[0]['choice'],'pickup:20:50:1:0')
            self.assertFalse(c.strategy.finished) # Timeout is not acquisition.
            m.plan.assert_not_called();m.decide.assert_not_called()
        finally:await c.close()

    def test_detour_progress_then_orbit_cannot_extend_collection_forever(self):
        w=setup();t=PickupProgress();log=Log();t.observe(w,log)
        # A detour can temporarily increase target distance: new walk cells
        # still count as approach progress, without requiring monotonic distance.
        for i in range(600):
            t.follow(w,(400,280))
            w.player['pos']=dict(x=320-(i//100+1)*20,y=280)
            refresh(w,1);t.observe(w,log)
            if i<599:self.assertFalse(w.pickup_failures)
        self.assertEqual(next(iter(w.pickup_failures.values()))['reason'],'collection_time_limit')
        w=setup();t=PickupProgress();log=Log();t.observe(w,log)
        for i in range(150):
            t.follow(w,(400,280));w.player['pos']=dict(x=320+(i%2)*20,y=280)
            refresh(w,1);t.observe(w,log)
        self.assertEqual(len([r for r in log.rows if r['event']=='local_pickup_failed']),1)

    def test_combat_stale_samples_and_pauses_do_not_consume_budget(self):
        w=setup();t=PickupProgress();log=Log();t.observe(w,log)
        for _ in range(100):tick(w,t,log)
        for mode in ('combat','stale','manual','gap'):
            t.follow(w,(400,280));refresh(w,1000)
            if mode=='combat':w.payload['ENEMIES']=[dict(id=9,hp=10,pos=dict(x=500,y=280))]
            elif mode=='stale':w.sampled['PLAYER_POSITION']=w.frame-10
            elif mode=='manual':w.control_mode='MANUAL'
            t.observe(w,log)
            w.payload['ENEMIES']=[];w.control_mode='AI';refresh(w);t.observe(w,log)
        self.assertFalse(w.pickup_failures)
        for _ in range(20):tick(w,t,log)
        self.assertEqual(len(w.pickup_failures),1)

    def test_disappearance_or_opened_form_never_becomes_failed_acquisition(self):
        for opened in (False,True):
            w=setup();t=PickupProgress();log=Log();t.observe(w,log)
            for _ in range(119):tick(w,t,log)
            t.follow(w,(400,280))
            if opened:w.pickups[0]['sub_type']=0
            else:w.payload['PICKUPS']=[]
            refresh(w,1);t.observe(w,log)
            self.assertFalse(w.pickup_failures)
            self.assertFalse(t.attempts)

    def test_failure_survives_revisit_but_physical_changes_allow_retry(self):
        w=setup();t,log=stationary(w);room=w.room
        self.assertIsNone(pickup_target(w,{},True))
        self.assertTrue(blocked_choices(w))
        record=deepcopy(next(iter(w.pickup_failures.values())))
        for change in ('position','geometry','flight','identity'):
            w=setup();stationary(w);t=PickupProgress()
            if change=='position':w.pickups[0]['pos']['y']+=50
            elif change=='geometry':w.layout['grid']['new']=dict(type=2,collision=1,x=200,y=200)
            elif change=='flight':w.stats['can_fly']=True
            else:w.pickups[0]['id']=99
            t.observe(w,log)
            self.assertFalse(blocked_choices(w),change)
        w=setup();t,log=stationary(w)
        w.room+=1;t.observe(w,log)
        self.assertTrue(blocked_choices(w,room))
        w.room=room;t.observe(w,log)
        self.assertTrue(blocked_choices(w))
        w.level+=':next';t.observe(w,log)
        self.assertFalse(w.pickup_failures)
        w.pickup_failures['old']=record;w.reset();self.assertFalse(w.pickup_failures)

    def test_failed_chest_does_not_create_a_remote_return_goal(self):
        w=setup();stationary(w);source=w.room
        chest=deepcopy(w.pickups[0]);w.rooms[str(source)]=dict(type=7,clear=True,resources=[chest])
        w.room+=1;w.payload['PICKUPS']=[]
        w.layout['doors']={'0':dict(x=40,y=280,is_open=True,is_locked=False,target_room=source,target_room_type=7)}
        self.assertFalse(any(o['destination_room']==source for o in options(w)))
        w.rooms[str(source)]['resources'].append(dict(id=21,variant=20,sub_type=1,price=0))
        self.assertTrue(any(o['destination_room']==source for o in options(w)))
        obs=enrich_planning(w,w.observation())
        self.assertEqual(len(obs['local_pickup_failures']),1)
        self.assertIn('not collected',obs['local_pickup_failures'][0]['status'])

    def test_failed_ground_healing_does_not_indefinitely_hold_emergency_pill(self):
        w=prepared();w.payload['PICKUPS']=w.pickups[:1];w.pickups[0]['sub_type']=5
        self.assertIsNone(ResourcePolicy().choose(w))
        stationary(w)
        self.assertEqual(ResourcePolicy().choose(w)['kind'],'pill')

    async def test_parallel_collection_can_expire_while_same_sol_request_continues(self):
        w=setup();w.payload['PICKUPS'].append(dict(id=7,variant=100,sub_type=378,price=0,pos=dict(x=480,y=280)))
        gate=asyncio.Event()
        async def plan(obs):await gate.wait()
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(side_effect=plan),decide=AsyncMock())
        c=Controller(m,Log())
        try:
            c.step(w);await asyncio.sleep(0)
            self.assertTrue(c.execution_plan['parallel_resource_collection'])
            task=c.plan_task
            for _ in range(120):refresh(w,1);c.step(w)
            self.assertEqual(len(w.pickup_failures),1)
            self.assertNotIn('parallel_resource_collection',c.execution_plan)
            self.assertIs(c.plan_task,task);self.assertFalse(task.done())
            m.plan.assert_awaited_once();m.decide.assert_not_called()
        finally:await c.close()
