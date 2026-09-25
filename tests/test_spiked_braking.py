import json
import math
from pathlib import Path
import unittest

from isaac_agent.state import World,xy
from isaac_agent.control import blocked,candidates
from test_ranged_combat import advance


class SpikedBrakingTests(unittest.TestCase):
    def test_recorded_surface_and_delayed_braking_keep_spike_clearance(self):
        f=json.loads((Path(__file__).parent/'fixtures/spiked-rock-braking.json').read_text())
        w=World();w.ingest(f['snapshot'])
        # Last observed position before the actual hit must be unsafe even
        # with Caffeine Pill's smaller PLAYER_STATS size.
        self.assertTrue(blocked(w,(82.7627,190.4519),w.stats['size'],hazards_only=True))
        # Static geometry/movement regression with two updates of queued up
        # input; this is not a replay of the full engine or a live success.
        queued=[(0,-1)]*2
        plan={'strategy_enabled':True,'mode':'explore','door_slot':'1'}
        minimum=999
        for _ in range(45):
            queued.append(candidates(w,plan)[0][0]['action'].move)
            advance(w,queued.pop(0));p=xy(w.player['pos'])
            minimum=min(minimum,math.dist(p,(80,160)))
        self.assertGreater(minimum,38)
        self.assertGreater(w.player['pos']['x'],170) # gets around, not permanent stop
        w.stats['can_fly']=True
        self.assertFalse(blocked(w,(82.7627,190.4519),w.stats['size'],hazards_only=True))
