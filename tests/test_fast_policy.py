import asyncio
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.demo import frame_message
from isaac_agent.state import World
from isaac_agent.fast_policy import fast_offers, lane
from isaac_agent.runtime import Controller
from isaac_agent.control import Action, target_point


class Log:
    def __init__(self):
        self.rows = []
    def write(self, event, **data):
        self.rows.append(dict(event=event, **data))


def snapshot():
    msg = frame_message()
    msg['payload']['ROOM_INFO'].update(room_type=5, curses=0, stage_type=0)
    msg['payload']['PLAYER_STATS'][0]['player_type'] = 1
    msg['payload']['PICKUPS'] = [dict(id=7, variant=100, sub_type=30, item_type=1,
                                    price=0, options_index=0, pos=dict(x=400, y=280))]
    return msg


def world(msg=None):
    w = World()
    w.ingest(msg or snapshot())
    return w


class FastPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_only_reviewed_free_unambiguous_items_can_use_fast_lane(self):
        self.assertEqual(fast_offers(world())[0]['entity']['sub_type'], 30)
        for overrides in ({'price': 1}, {'options_index': 1}, {'item_type': 3},
                          {'sub_type': 531}, {'is_hidden': True}):
            msg = snapshot()
            msg['payload']['PLAYER_INVENTORY'][0]['coins'] = 10
            msg['payload']['PICKUPS'][0].update(overrides)
            self.assertFalse(fast_offers(world(msg)))
        for overrides in ({'curses': 64}, {'curses': None}, {'stage_type': 4}):
            msg = snapshot()
            msg['payload']['ROOM_INFO'].update(overrides)
            self.assertFalse(fast_offers(world(msg)))
        msg = snapshot()
        msg['payload']['PICKUPS'].append(dict(msg['payload']['PICKUPS'][0], id=8))
        self.assertFalse(fast_offers(world(msg)))

    async def test_jev_goal_is_single_flight_and_then_executed_without_sol(self):
        gate = asyncio.Event()
        async def choose(obs, choices):
            await gate.wait()
            return choices[0]['choice'], {'confidence': .9}
        models = SimpleNamespace(mode='hybrid', choose_goal=AsyncMock(side_effect=choose),
                                 plan=AsyncMock(), decide=AsyncMock(return_value=(Action(), {'confidence': 0})))
        w, log = world(), Log()
        c = Controller(models, log)
        try:
            c.step(w)
            await asyncio.sleep(0)
            c.step(w)
            self.assertEqual(models.choose_goal.await_count, 1)
            models.plan.assert_not_called()
            models.decide.assert_not_called()
            gate.set()
            await c.decision_task
            c.step(w)
            self.assertEqual(target_point(w, c.execution_plan), ((400, 280), True))
            self.assertEqual(c.plan['goal_source'], 'jev')
            models.plan.assert_not_called()
        finally:
            await c.close()

    async def test_low_confidence_goal_defers_once_to_sol(self):
        models = SimpleNamespace(mode='hybrid',
            choose_goal=AsyncMock(return_value=('pickup:7:100:30:0', {'confidence': .2})),
            plan=AsyncMock(return_value=({'objective': 'Skip', 'mode': 'explore'}, {})))
        w, c = world(), Controller(models, Log())
        try:
            c.step(w)
            await c.decision_task
            c.step(w)
            await c.plan_task
            self.assertEqual(models.choose_goal.await_count, 1)
            self.assertEqual(models.plan.await_count, 1)
        finally:
            await c.close()

    async def test_fast_goal_cannot_cross_rooms(self):
        models = SimpleNamespace(mode='hybrid')
        w, c = world(), Controller(models, Log())
        async def choose(obs, choices):
            msg = snapshot()
            msg['room_index'] = 9
            w.ingest(msg)
            return choices[0]['choice'], {'confidence': 1}
        models.choose_goal = choose
        try:
            c.step(w)
            await c.decision_task
            self.assertIsNone(c.plan.get('strategy_choice'))
            self.assertTrue(any(r['event'] == 'fast_goal_discarded' for r in c.log.rows))
        finally:
            await c.close()

    async def test_boss_plan_survives_damage_and_dead_target_retargets_locally(self):
        w = world(frame_message(enemies=[{'id': 1, 'pos': {'x': 400, 'y': 280}}]))
        obs = deepcopy(w.observation())
        revision = w.revision
        async def plan(observation):
            w.ingest(frame_message(3, enemies=[{'id': 2, 'pos': {'x': 420, 'y': 280}}]))
            w.revision += 1
            w.last_damage_frame = 3
            return {'objective': 'Keep distance', 'mode': 'combat', 'target_enemy': 1}, {}
        c = Controller(SimpleNamespace(mode='hybrid', plan=plan), Log())
        try:
            await c._plan(w, obs, w.token, revision)
            self.assertEqual(c.log.rows[-1]['event'], 'plan')
            self.assertIsNone(c.plan['target_enemy'])
            self.assertEqual(target_point(w, c.plan)[0], (420, 280))
        finally:
            await c.close()

    def test_ordinary_short_fight_and_empty_room_need_no_sol(self):
        w = world(frame_message(enemies=[{'id': 1}]))
        self.assertEqual(lane(w, w.frame)[0], 'tactical')
        self.assertEqual(lane(w, w.frame-360)[0], 'tactical')
        self.assertEqual(lane(w, w.frame-360, combat_stalled=True)[0], 'sol')
        w = world(frame_message())
        self.assertEqual(lane(w, w.frame)[0], 'routine')

    def test_boss_exit_waits_while_slow_choice_is_pending(self):
        w = world(frame_message())
        w.info['room_type'] = 5
        w.layout['grid'] = {'22': dict(type=17, variant=0, state=1, grid_index=22, x=400, y=300)}
        p, contact = target_point(w, dict(strategy_enabled=True, awaiting_strategy=True, hold_for_strategy=True))
        self.assertEqual(p, (320, 280))
        self.assertFalse(contact)
