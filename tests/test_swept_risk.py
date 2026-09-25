from copy import deepcopy
import math
import time
import unittest

from isaac_agent.control import MOVES, risk
from isaac_agent.physics import swept_separation
from test_review_continuity import setup
from test_safety_strategy import refresh


def world():
    w=setup();w.player.update(pos=dict(x=320,y=280),vel=dict(x=0,y=0))
    w.payload['PICKUPS']=[];w.layout['grid']={};w.layout['doors']={}
    return w


class SweptRiskTests(unittest.TestCase):
    def test_bullet_crosses_between_safe_endpoints(self):
        w=world();w.payload['PROJECTILES']={'enemy_projectiles':[
            dict(id=9,pos=dict(x=280,y=280),vel=dict(x=80,y=0),collision_radius=5)]}
        refresh(w)
        # At both sampled endpoints the bullet is 40 units away; halfway
        # through it overlaps the player. Endpoint-only checks miss contact.
        radius=float(w.stats.get('size',12))+5
        self.assertGreater(40-radius,6)
        self.assertGreater(risk(w,(0,0),horizon=1),100)

    def test_old_projectile_sample_matches_fresh_extrapolated_state(self):
        for explosive in (False,True):
            for move in MOVES:
                stale=world();stale.payload['PROJECTILES']={'enemy_projectiles':[
                    dict(id=9,pos=dict(x=380,y=280),vel=dict(x=-4,y=1),
                         collision_radius=5,is_explosive=explosive)]}
                refresh(stale);stale.sampled['PROJECTILES']=stale.frame-3
                fresh=deepcopy(stale);fresh.projectiles[0]['pos']=dict(x=368,y=283)
                fresh.sampled['PROJECTILES']=fresh.frame
                self.assertAlmostEqual(risk(stale,move),risk(fresh,move),places=8)

    def test_old_enemy_sample_matches_fresh_extrapolated_state(self):
        stale=world();stale.payload['ENEMIES']=[dict(id=9,type=10,variant=0,
            pos=dict(x=420,y=280),vel=dict(x=-4,y=1),collision_radius=12)]
        refresh(stale);stale.sampled['ENEMIES']=stale.frame-2
        fresh=deepcopy(stale);fresh.enemies[0]['pos']=dict(x=412,y=282)
        fresh.sampled['ENEMIES']=fresh.frame
        for move in MOVES:self.assertAlmostEqual(risk(stale,move),risk(fresh,move),places=8)

    def test_path_crossing_at_different_times_is_not_collision(self):
        # Both world-space paths cross (2,0), at times .2 and .8. Closest
        # simultaneous positions are (5,0) and (2,-3), separated by sqrt(18).
        self.assertAlmostEqual(swept_separation((0,0),(10,0),(2,-8),(2,2)),math.sqrt(18))
        self.assertEqual(swept_separation((0,0),(10,0),(5,-5),(5,5)),0)
        self.assertEqual(swept_separation((0,0),(10,0),(0,10),(10,10)),10)
        self.assertEqual(swept_separation((0,0),(0,0),(3,4),(3,4)),5)

    def test_stale_channel_is_still_rejected_before_live_control(self):
        w=world();refresh(w);self.assertTrue(w.ready(time.monotonic()))
        for name in ('PROJECTILES','ENEMIES'):
            refresh(w);w.sampled[name]=w.frame-7
            self.assertFalse(w.ready(time.monotonic()))
