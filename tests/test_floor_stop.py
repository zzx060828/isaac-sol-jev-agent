import unittest
from test_agent import world
from isaac_agent.runtime import floor_finished


class FloorStopTests(unittest.TestCase):
    def test_requires_current_cleared_boss_room_and_open_exit_on_requested_floor(self):
        w = world()
        w.info.update(room_type=5, is_clear=True)
        w.layout['grid'] = {'1': {'type': 17, 'variant': 0, 'state': 1}}
        self.assertTrue(floor_finished(w, 1))
        self.assertFalse(floor_finished(w, 2))
        w.info['room_type'] = 1
        self.assertFalse(floor_finished(w, 1))
        w.info.update(room_type=5, is_clear=False)
        self.assertFalse(floor_finished(w, 1))
        w.info['is_clear'] = True
        w.layout['grid']['1']['state'] = 0
        self.assertFalse(floor_finished(w, 1))
        w.layout['grid']['1']['state'] = 1
        w.frame += 9
        self.assertFalse(floor_finished(w, 1))

    def test_closed_revisited_exit_remains_unavailable_and_gets_space(self):
        import json, math
        from pathlib import Path
        from isaac_agent.state import World, xy
        from isaac_agent.strategy import offers
        from isaac_agent.control import target_point
        msg = json.loads((Path(__file__).parent/'fixtures/revisited-boss-trapdoor.json').read_text())
        w = World(); w.ingest(msg)
        self.assertFalse(w.boss_exits)
        self.assertFalse(any(o['kind'] == 'descend' for o in offers(w)))
        goal, contact = target_point(w, {'strategy_enabled':True, 'awaiting_strategy':True, 'hold_for_strategy':True})
        self.assertFalse(contact)
        self.assertGreaterEqual(math.dist(goal, (320,200)), 90)
        w.layout['grid']['37']['state'] = 1
        self.assertTrue(any(o['choice'] == 'descend:37' for o in offers(w)))
