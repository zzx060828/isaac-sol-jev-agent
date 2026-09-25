import json
import math
from pathlib import Path
import unittest

from isaac_agent.state import World, xy
from isaac_agent.control import candidates, path_step, blocked


class WaypointStallTests(unittest.TestCase):
    def test_recorded_fast_character_advances_instead_of_stopping_at_lattice_node(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/anticipation-waypoint-stall.json').read_text())
        w=World();w.ingest(fixture['snapshot']);w.rooms=fixture['rooms']
        options,_=candidates(w,fixture['plan'])
        self.assertEqual(options[0]['action'].move,(0,-1))
        self.assertLess(options[0]['risk'],1)
        start=xy(w.player['pos']);waypoint=path_step(w,(320,120))
        self.assertGreater(math.dist(start,waypoint),40)
        for i in range(1,21):
            point=tuple(start[a]+(waypoint[a]-start[a])*i/20 for a in (0,1))
            self.assertFalse(blocked(w,point,float(w.stats['size']),exit_point=(320,120)))
