import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from test_agent import MemoryLog
from test_safety_strategy import prepared, refresh
from isaac_agent.control import Action
from isaac_agent.memory import RunMemory
from isaac_agent.runtime import Controller
from isaac_agent.secrets import neighbor, offers, wall_observations, SecretSearch


def setup():
    w=prepared(); w.room=84; w.info.update(room_shape=1,room_type=1)
    w.capabilities.update(repentance_plus=False,secret_bomb_protocol=1)
    w.inventory['bombs']=3; w.health.update(red_hearts=8,max_hearts=8)
    w.layout['doors']={}
    w.layout['wall_slots']={'0':{'x':40,'y':280},'1':{'x':320,'y':120},
                            '2':{'x':600,'y':280},'3':{'x':320,'y':440}}
    w.run_memory=RunMemory();w.run_memory.seed=w.level.split(':')[0]
    w.rooms={'84':{'visited':True,'shape':1,'type':1,'doors':{},'walls':wall_observations(w)},
             '72':{'shape':1,'type':1,'doors':{},'walls':{'3':{'approach_clear':True,'has_door':False}}}}
    refresh(w)
    return w


def settled_update(job,w,plan,log):
    braking=job.update(w,plan,log)
    if not braking.get('bomb_brake'):
        raise AssertionError('Expected brake before an authorized bomb')
    refresh(w,4)
    return job.update(w,plan,log)


class SecretSearchTests(unittest.TestCase):
    def test_observed_adjacent_walls_form_uncertain_candidate(self):
        w=setup(); offer=offers(w)[0]
        self.assertEqual(offer['candidate_cell'],85)
        self.assertEqual(set(offer['evidence']['observed_adjacent_rooms']),{84,72})
        self.assertEqual(offer['bomb_cost'],1)
        self.assertEqual(offer['bombs_reserved_after'],2)
        self.assertIn('unknown_adjacent_cells',offer['evidence'])

    def test_no_map_row_wrap(self):
        self.assertIsNone(neighbor(12,2));self.assertIsNone(neighbor(13,0))
        self.assertIsNone(neighbor(0,1));self.assertIsNone(neighbor(168,3))

    def test_blocked_other_approach_and_unobserved_geometry_reject_guess(self):
        w=setup();w.rooms['72']['walls']['3']['approach_clear']=False
        self.assertFalse(offers(w))
        w=setup();w.rooms['72'].pop('walls')
        self.assertFalse(offers(w))

    def test_resource_protocol_and_escape_gates(self):
        for change in ('one_bomb','low_health','old_bridge','altered_bomb','pit','tnt','stale'):
            with self.subTest(change=change):
                w=setup()
                if change=='one_bomb':w.inventory['bombs']=1
                if change=='low_health':w.health['red_hearts']=3
                if change=='old_bridge':w.capabilities.pop('secret_bomb_protocol')
                if change=='altered_bomb':w.inventory['collectibles']={'106':1}
                if change=='stale':w.sampled['BOMBS']=w.frame-7
                if change in ('pit','tnt'):
                    w.layout['grid']={'1':{'x':480,'y':280,'type':7 if change=='pit' else 12,'collision':1 if change=='pit' else 2}}
                self.assertFalse(offers(w))

    def test_one_pulse_retreat_confirm_and_enter(self):
        w=setup();o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        search=SecretSearch(w,o);log=MemoryLog();plan={'strategy_enabled':True}
        command=settled_update(search,w,plan,log)
        self.assertIn('secret_bomb_request',command)
        self.assertIn(o['attempt_key'],w.run_memory.secret_attempts)
        refresh(w,1);w.inventory['bombs']=2
        command=search.update(w,plan,log)
        self.assertNotIn('secret_bomb_request',command)
        self.assertEqual(command['navigation_target'],tuple(o['escape']))
        w.layout['doors']['2']={'x':600,'y':280,'is_open':True,'target_room':85,'target_room_type':7}
        refresh(w,120);command=search.update(w,plan,log)
        self.assertTrue(command['navigation_contact'])
        self.assertEqual(w.run_memory.secret_attempts[o['attempt_key']]['status'],'entrance_open')
        w.room=85;w.info['room_type']=7;refresh(w,1)
        search.update(w,plan,log)
        self.assertEqual(search.done,'entered_secret')

    def test_failed_or_unconfirmed_attempt_never_retries(self):
        for consumed in (True,False):
            w=setup();o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
            search=SecretSearch(w,o);log=MemoryLog()
            settled_update(search,w,{},log)
            if consumed:w.inventory['bombs']=2
            refresh(w,121);search.update(w,{},log)
            self.assertEqual(search.done,'no_entrance_observed' if consumed else 'bomb_unconfirmed_no_retry')
            self.assertFalse(offers(w))

    def test_failure_memory_survives_confirmed_continue_frame_reset(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'memory.json';w=setup();w.run_memory.path=path
            w.run_memory.secret_attempts={'demo:1:0:85':{'status':'no_entrance_observed'}}
            w.run_memory.last_frame=9999;w.run_memory.save()
            restored=RunMemory(path);w.capabilities['continued_run']=True
            restored.observe(w,{})
            self.assertIn('demo:1:0:85',restored.secret_attempts)
            fresh=RunMemory(path);w.capabilities['continued_run']=False;w.frame=0
            fresh.observe(w,{})
            self.assertFalse(fresh.secret_attempts)

    def test_two_attempt_budget_is_scoped_to_current_floor(self):
        w=setup()
        w.run_memory.secret_attempts={w.level+':1':{},w.level+':2':{}}
        self.assertFalse(offers(w))
        w.run_memory.secret_attempts={'demo:2:0:1':{},'demo:2:0:2':{}}
        self.assertTrue(offers(w))

    def test_observed_wall_geometry_survives_continue(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'memory.json';w=setup();w.run_memory.path=path
            w.run_memory.observe(w,{})
            original=w.rooms['84']['walls']
            restored=setup();restored.room=85;restored.rooms={}
            restored.capabilities['continued_run']=True
            restored.frame=0
            memory=RunMemory(path);memory.observe(restored,{})
            self.assertEqual(restored.rooms['84']['walls'],original)
            restored.level='demo:2:0';restored.rooms={}
            memory.observe(restored,{})
            self.assertNotIn('84',restored.rooms)


class SecretRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_authorized_goal_emits_one_bomb_and_suppresses_other_resources(self):
        w=setup();o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),
                               decide=AsyncMock(return_value=(Action(),{'confidence':0})))
        c=Controller(models,MemoryLog());c.reset_actions(w)
        c.plan.update(strategy_choice=o['choice'],mode='collect')
        try:
            command=c.step(w);await asyncio.sleep(0)
            self.assertFalse(command['use_bomb']);self.assertEqual(command['move'],{'x':0,'y':0})
            refresh(w,4);command=c.step(w)
            self.assertTrue(command['use_bomb']);self.assertFalse(command['use_item'])
            pulse=next(row for row in c.log.rows if row['event']=='input' and row.get('bomb_request_id'))
            self.assertEqual(pulse['action'],Action().id)
            self.assertEqual(pulse['source'],'secret_contract')
            self.assertEqual(pulse['bomb_request_id'],o['attempt_key'])
            refresh(w,1);command=c.step(w)
            self.assertFalse(command['use_bomb'])
            models.plan.assert_not_called()
        finally:await c.close()
