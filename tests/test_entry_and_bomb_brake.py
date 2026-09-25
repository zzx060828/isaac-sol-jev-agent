import json
import math
from pathlib import Path
import unittest

from isaac_agent.control import blocked, candidates, physical_collision
from isaac_agent.secrets import SecretSearch, offers
from isaac_agent.state import World, xy
from isaac_agent.strategy import StrategyExecutor
from test_agent import MemoryLog
from test_ranged_combat import advance
from test_secret_search import setup
from test_safety_strategy import refresh


class EntryAndBrakeTests(unittest.TestCase):
    def recorded_entry(self):
        f=json.loads((Path(__file__).parent/'fixtures/boss-entry-stall.json').read_text())
        w=World();w.ingest(f['snapshot'])
        return w,f['plan']

    def test_recorded_doorway_can_reach_boss_reward(self):
        w,plan=self.recorded_entry();executor=StrategyExecutor();log=MemoryLog()
        # Replay observed geometry and approximate motion, not a game simulation.
        for _ in range(100):
            execution=executor.update(w,plan,log)
            move=candidates(w,execution)[0][0]['action'].move
            advance(w,move)
            if math.dist(xy(w.player['pos']),(320,360))<15:break
        self.assertLess(math.dist(xy(w.player['pos']),(320,360)),15)
        self.assertFalse(any(r.get('result')=='timeout' for r in log.rows))

    def test_entry_exception_does_not_allow_outward_closed_or_obstructed_movement(self):
        for change in ('inward','outward','closed','rock','spike'):
            with self.subTest(change=change):
                w,_=self.recorded_entry();point=(62,280)
                if change=='outward':point=(53,280)
                if change=='closed':
                    for d in w.layout['doors'].values():d['is_open']=False
                if change in ('rock','spike'):
                    w.layout['grid']['extra']={'x':62,'y':280,'type':2 if change=='rock' else 8,
                                                'collision':3 if change=='rock' else 0}
                self.assertEqual(blocked(w,point),change!='inward')
                if change!='spike':self.assertEqual(physical_collision(w,point,10),change!='inward')

    def test_velocity_reversal_must_brake_before_pulse(self):
        w=setup();o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=SecretSearch(w,o);log=MemoryLog()
        for velocity in ((.33,-.12),(-.46,-.73),(-.15,-.2),(0,0)):
            w.player['vel']=dict(zip(('x','y'),velocity))
            result=job.update(w,{},log)
            self.assertTrue(result['bomb_brake'])
            self.assertNotIn('secret_bomb_request',result)
            self.assertFalse(w.run_memory.secret_attempts)
            refresh(w,1)
        self.assertIn('secret_bomb_request',job.update(w,{},log))

    def test_enclosed_key_stays_observed_without_impossible_touch_contract(self):
        from isaac_agent.strategy import offers as strategy_offers
        w=World();w.ingest(json.loads((Path(__file__).parent/'fixtures/enclosed-key.json').read_text()))
        self.assertTrue(w.pickups)
        self.assertFalse(any(o['kind']=='pickup' for o in strategy_offers(w)))
        # The same key becomes collectible after its blocking terrain changes.
        w.layout['grid']={};w.navigation_revision+=1
        self.assertTrue(any(o['choice']=='pickup:1:30:1:0' for o in strategy_offers(w)))

    def test_drift_resets_settle_and_stale_position_cannot_fire(self):
        w=setup();o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=SecretSearch(w,o);log=MemoryLog();job.update(w,{},log)
        refresh(w,4);w.player['pos']['y']+=8
        self.assertNotIn('secret_bomb_request',job.update(w,{},log))
        w.player['pos']=dict(o['entity']['pos'])
        self.assertNotIn('secret_bomb_request',job.update(w,{},log))
        refresh(w,4);w.sampled['PLAYER_POSITION']=w.frame-2
        self.assertNotIn('secret_bomb_request',job.update(w,{},log))
        refresh(w,1)
        self.assertIn('secret_bomb_request',job.update(w,{},log))
