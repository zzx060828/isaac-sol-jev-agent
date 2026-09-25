import json
from pathlib import Path
import unittest
import math

from isaac_agent.state import World, xy, records
from isaac_agent.control import candidates, risk, physical_collision
from isaac_agent.physics import trajectory


class CornerEscapeTests(unittest.TestCase):
    def test_recorded_creep_trap_exits_before_observed_cooldown_ends(self):
        from test_ranged_combat import advance
        f=json.loads((Path(__file__).parent/'fixtures/pestilence-creep-trap.json').read_text())
        w=World();w.ingest(f['snapshot']);w.last_damage_frame=f['last_damage_frame']
        # Replay terrain/creep geometry, not frozen moving bullets or boss AI.
        w.payload['PROJECTILES']={'enemy_projectiles':[]}
        escaped=False
        for _ in range(21):
            action=candidates(w,f['plan'])[0][0]['action'];advance(w,action.move)
            w.health['damage_cooldown']=max(0,w.health['damage_cooldown']-2)
            point=xy(w.player['pos'])
            self.assertFalse(physical_collision(w,point,10))
            gap=min(math.dist(point,xy(h['pos']))-h['collision_radius']-10
                    for h in records(w.payload['FIRE_HAZARDS']) if h.get('ground_only'))
            if gap>0 and not escaped:
                escaped=True
                self.assertGreater(w.health['damage_cooldown'],0)
            if w.health['damage_cooldown']==0:self.assertGreater(gap,0)
        self.assertTrue(escaped)

    def test_creep_escape_requires_fresh_cooldown_and_real_overlap(self):
        f=json.loads((Path(__file__).parent/'fixtures/pestilence-creep-trap.json').read_text())
        w=World();w.ingest(f['snapshot']);w.last_damage_frame=f['last_damage_frame']
        active=risk(w,(0,-1))
        self.assertLess(active,risk(w,(0,1)))
        w.sampled['PLAYER_HEALTH']=w.frame-2
        self.assertGreater(risk(w,(0,-1)),active+1000)
        w.sampled['PLAYER_HEALTH']=w.frame
        w.health['damage_cooldown']=0
        self.assertGreater(risk(w,(0,-1)),active+1000)

    def test_observed_cooldown_allows_leaving_recorded_contact_trap(self):
        w = World(); w.ingest(json.loads((Path(__file__).parent/'fixtures/gurgle-corner-damage.json').read_text()))
        original = candidates(w,{})[0][0]['action']
        self.assertEqual(original.move, (0,0))
        # Fixture predates this sensor. Explicitly simulate a freshly reported
        # cooldown; a damage event by itself must not authorize contact escape.
        w.last_damage_frame = w.frame-1
        self.assertEqual(candidates(w,{})[0][0]['action'].move, (0,0))
        w.health['damage_cooldown'] = 30
        action = candidates(w,{})[0][0]['action']
        self.assertEqual(action.move[0], -1)
        path = trajectory(w, action.move, 16, collides=lambda p:physical_collision(w,p,10))
        self.assertGreater(w.player['pos']['x']-path[-1][0], 65)
        self.assertTrue(all(not physical_collision(w,p,10) for p in path))
        self.assertLess(risk(w,action.move),risk(w,(0,0)))
        w.sampled['PLAYER_HEALTH'] = w.frame-2
        self.assertEqual(candidates(w,{})[0][0]['action'].move, (0,0))
