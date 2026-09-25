import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from isaac_agent.state import World,xy
from isaac_agent.combat import edge_flank_axis,select_enemy
from isaac_agent.control import candidates,combat_waypoint,physical_collision,blocked
from isaac_agent.physics import trajectory


def recorded():
    f=json.loads((Path(__file__).parent/'fixtures/rotty-wall-approach.json').read_text())
    w=World();w.ingest(f['snapshot']);return w


class EdgeFlankTests(unittest.TestCase):
    def test_recorded_wall_approach_turns_before_contact_instead_of_waiting(self):
        w=recorded();target=xy(select_enemy(w,{}).get('pos'))
        self.assertEqual(edge_flank_axis(w,target),0)
        with patch('isaac_agent.combat.edge_flank_axis',return_value=None):
            self.assertEqual(candidates(w,{})[0][0]['action'].move,(0,0))
        option=candidates(w,{})[0][0]
        self.assertNotEqual(option['action'].move[0],0)
        self.assertLess(option['risk'],5)
        for point in trajectory(w,option['action'].move,3):
            self.assertFalse(physical_collision(w,point,w.stats['size']))
            self.assertFalse(blocked(w,point,w.stats['size']))
            self.assertGreater(math.dist(point,target),100)
        self.assertEqual(w.health['damage_cooldown'],0) # no invulnerability permission

    def test_stationary_or_distant_target_does_not_force_a_wall_flank(self):
        w=recorded();e=select_enemy(w,{});target=xy(e['pos'])
        e['vel']={'x':0,'y':0};self.assertIsNone(edge_flank_axis(w,target))
        e['vel']={'x':0,'y':-1};e['pos']['y']=500
        self.assertIsNone(edge_flank_axis(w,xy(e['pos'])))

    def test_segmented_motion_does_not_replace_the_checked_firing_lane(self):
        w=recorded();e=select_enemy(w,{})
        e.update(type=19,variant=0)
        self.assertIsNone(edge_flank_axis(w,xy(e['pos'])))
        w.player['pos']['y']=320;e['pos']['y']=400
        self.assertIsNone(edge_flank_axis(w,xy(e['pos'])))

    def test_too_narrow_passage_or_live_creep_does_not_create_a_flank(self):
        w=recorded();target=xy(select_enemy(w,{}).get('pos'))
        w.info['top_left']['x']=280;w.info['bottom_right']['x']=360
        self.assertIsNone(combat_waypoint(w,target,desired=213,flexible=True,flank_axis=0))
        w=recorded()
        # Cover both possible firing positions with observed damaging creep.
        w.payload['FIRE_HAZARDS'] += [dict(type='GROUND',ground_only=True,id=900+i,
            pos={'x':x,'y':target[1]},collision_radius=65) for i,x in enumerate((240,400))]
        self.assertIsNone(combat_waypoint(w,target,desired=213,flexible=True,flank_axis=0))
