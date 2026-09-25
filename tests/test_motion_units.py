import json
import math
from pathlib import Path
import unittest

from isaac_agent.physics import hostile_position
from isaac_agent.state import World, xy
from isaac_agent.control import risk


class RecordedMotionTests(unittest.TestCase):
    def test_hostile_projection_matches_recorded_next_update(self):
        # Consecutive fresh updates from live recordings. Only near-constant
        # velocities are selected; acceleration is not a velocity-unit error.
        samples = json.loads((Path(__file__).parent / 'fixtures' /
                              'motion-unit-observations.json').read_text())
        self.assertEqual({row['group'] for row in samples}, {'npc', 'projectile'})
        for row in samples:
            with self.subTest(run=row['run'], frame=row['frame'], group=row['group']):
                expected = xy(row['current']['pos'])
                predicted = hostile_position(row['previous'], 1)
                self.assertLess(math.dist(predicted, expected), .06)
                # The former player-tick conversion produces a material miss.
                old = hostile_position(row['previous'], 2)
                self.assertGreater(math.dist(old, expected), 1.9)

    def test_explosive_flag_protects_recorded_near_miss(self):
        world = World()
        world.ingest(json.loads((Path(__file__).parent / 'fixtures' /
                                 'walking-boil-before-hit.json').read_text()))
        # This recording predates the flag export. Add the reviewed sensor
        # boolean explicitly; the old recording alone cannot identify flags.
        bullet = world.projectiles[0]
        baseline = risk(world, (1, -1))
        self.assertLess(baseline, 1)
        bullet['is_explosive'] = True
        self.assertGreater(risk(world, (1, -1)), baseline + 100)
        self.assertLess(risk(world, (1, 0)), risk(world, (-1, 0)))
        bullet['is_explosive'] = False
        self.assertAlmostEqual(risk(world, (1, -1)), baseline)
