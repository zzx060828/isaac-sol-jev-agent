from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.control import path_step,candidates,blocked
from isaac_agent.runtime import Controller
from test_agent import world,MemoryLog
from test_safety_strategy import refresh


TARGET=(440,280)


def prepared(sealed=False, size=12):
    w=world(player=(280,280));w.stats['size']=size
    ys=(120,160,200,240,320,360,400,440) if sealed else (240,320)
    w.layout['grid']={str(i):dict(type=2,collision=3,x=320,y=y) for i,y in enumerate(ys)}
    w.payload.update(BOMBS=[],FIRE_HAZARDS=[],INTERACTABLES=[],PICKUPS=[
        dict(id=8,variant=20,sub_type=1,price=0,pos=dict(x=440,y=280))])
    refresh(w)
    return w


def resize(w,size):
    stats={**w.stats,'size':size}
    w.ingest(dict(type='DATA',frame=w.frame+1,room_index=w.room,agent=deepcopy(w.capabilities),
                  payload={'PLAYER_STATS':[stats]}))


class PathBodySizeTests(unittest.IsolatedAsyncioTestCase):
    def test_growth_recomputes_cached_shortcut_and_allows_retreat_to_detour(self):
        w=prepared();self.assertEqual(path_step(w,TARGET),TARGET)
        revision=w.navigation_revision;resize(w,25)
        self.assertEqual(w.navigation_revision,revision) # No terrain/pickup packet changed.
        self.assertTrue(blocked(w,(320,280),25))
        cached=path_step(w,TARGET)
        w.path_cache.clear();fresh=path_step(w,TARGET)
        self.assertEqual(cached,fresh)
        self.assertLess(cached[0],w.player['pos']['x'])
        action=candidates(w,dict(strategy_enabled=True,navigation_target=TARGET))[0][0]['action']
        self.assertLess(action.move[0],0)

    def test_shrinking_reopens_previously_cached_disconnected_resource(self):
        w=prepared(sealed=True,size=25)
        self.assertEqual(path_step(w,TARGET),(280,280))
        self.assertTrue(any(value is None for value in w.path_cache.values()))
        resize(w,12)
        self.assertFalse(blocked(w,(320,280),12))
        self.assertEqual(path_step(w,TARGET),TARGET)

    async def test_controller_replans_local_approach_on_stats_only_growth_without_models(self):
        w=prepared();m=SimpleNamespace(mode='hybrid',plan=AsyncMock(),decide=AsyncMock(),choose_goal=AsyncMock())
        c=Controller(m,MemoryLog())
        try:
            c.step(w);self.assertEqual(c.execution_plan['navigation_target'],TARGET)
            resize(w,25);command=c.step(w)
            self.assertEqual(c.route,'collect_resources')
            self.assertEqual(c.execution_plan['navigation_target'],TARGET)
            self.assertLess(command['move']['x'],0)
            self.assertEqual(command['shoot'],dict(x=0,y=0))
            self.assertFalse(command['use_bomb'])
            m.plan.assert_not_called();m.decide.assert_not_called();m.choose_goal.assert_not_called()
        finally:await c.close()
