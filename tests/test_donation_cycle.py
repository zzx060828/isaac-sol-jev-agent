import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_agent import MemoryLog
from test_safety_strategy import prepared, refresh
from isaac_agent.control import Action, blocked
from isaac_agent.strategy import StrategyExecutor, offers
from isaac_agent.runtime import Controller


def setup():
    w = prepared()
    w.capabilities['repentance_plus'] = False
    w.stats['player_type'] = 1
    w.health.update(red_hearts=8, max_hearts=8)
    w.payload['INTERACTABLES'] = [{'id': 8, 'variant': 2, 'pos': {'x': 350, 'y': 280}}]
    w.payload['PICKUPS'] = [dict(id=i, variant=10, sub_type=1, price=0, pos=dict(x=200,y=240-i)) for i in (1,2)]
    refresh(w)
    offer = next(o for o in offers(w) if o['kind'] == 'donate_cycle')
    return w, {'strategy_enabled': True, 'strategy_choice': offer['choice'], 'mode': 'collect'}


class DonationCycleTests(unittest.TestCase):
    def test_whole_heart_recovers_two_half_heart_payments_without_waste(self):
        w, plan = setup(); e = StrategyExecutor(); log = MemoryLog()
        w.payload['PICKUPS'] = w.pickups[:1]
        plan['strategy_choice'] = next(o['choice'] for o in offers(w) if o['kind'] == 'donate_cycle')
        self.assertTrue(e.update(w, plan, log)['navigation_contact'])
        for index in (1,2):
            w.player['pos'] = dict(x=340,y=280)
            w.health['red_hearts'] = 8-index
            refresh(w, 1); w.last_damage_frame = w.frame
            action = e.update(w, plan, log)
            self.assertFalse(action['navigation_contact'])
            w.player['pos'] = dict(x=250,y=280); refresh(w, 1)
            action = e.update(w, plan, log)
            if index == 1:
                self.assertTrue(action['navigation_contact'])
                continue  # Missing half a heart: save the full heart until useful.
            self.assertEqual(action['navigation_target'], (200,239))
            self.assertFalse(action['navigation_contact'])
            w.health['red_hearts'] = 8
            w.payload['PICKUPS'] = []
            refresh(w, 1); action = e.update(w, plan, log)
            self.assertFalse(action['navigation_contact'])
        self.assertIsNone(e.choice)
        self.assertIn(plan['strategy_choice'], e.finished)
        self.assertEqual(log.rows[-1]['contacts'], 2)

    def test_disappearing_heart_is_not_assumed_to_heal(self):
        w, plan = setup(); e = StrategyExecutor(); log = MemoryLog()
        e.update(w, plan, log)
        w.health['red_hearts'] = 6; w.player['pos'] = dict(x=250,y=280)
        refresh(w, 1); w.last_damage_frame = w.frame
        e.update(w, plan, log)
        refresh(w, 1); e.update(w, plan, log)
        w.payload['PICKUPS'] = []
        refresh(w, 1); e.update(w, plan, log)
        refresh(w, 8); action = e.update(w, plan, log)
        self.assertIsNone(e.choice)
        self.assertFalse(action['navigation_contact'])

    def test_low_health_and_stale_machine_never_start(self):
        w, _ = setup(); w.health['red_hearts'] = 5
        self.assertFalse(any(o['kind'].startswith('donate') for o in offers(w)))
        w.health['red_hearts'] = 8; w.sampled['INTERACTABLES'] = w.frame-7
        self.assertFalse(any(o['kind'].startswith('donate') for o in offers(w)))

    def test_new_combat_cancels_cycle(self):
        w, plan = setup(); e = StrategyExecutor(); log = MemoryLog()
        e.update(w, plan, log)
        w.payload['ENEMIES'] = [dict(id=1,pos=dict(x=400,y=280))]
        w.info.update(is_clear=False)
        action = e.update(w, plan, log)
        self.assertIsNone(e.choice)
        self.assertFalse(action['navigation_contact'])

    def test_special_or_paid_hearts_are_not_cycle_funding(self):
        w, _ = setup()
        for change in ({'price': 1}, {'options_index': 2}, {'sub_type': 3}):
            w.payload['PICKUPS'] = [dict(id=1,variant=10,sub_type=1,price=0,pos=dict(x=200,y=280)) | change]
            self.assertFalse(any(o['kind'] == 'donate_cycle' for o in offers(w)))

    def test_machine_contact_guard_allows_only_withdrawal_from_overlap(self):
        w, _ = setup(); w.player['pos'] = dict(x=340,y=280)
        self.assertFalse(blocked(w, (335,280)))
        self.assertTrue(blocked(w, (345,280)))
        w.player['pos'] = dict(x=250,y=280)
        self.assertTrue(blocked(w, (335,280)))


class DonationRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_cycle_needs_no_extra_sol_call_or_consumable(self):
        w, plan = setup()
        models = SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),
                                 decide=AsyncMock(return_value=(Action(),{'confidence':0})))
        c = Controller(models, MemoryLog()); c.reset_actions(w); c.plan = plan
        try:
            c.step(w); await asyncio.sleep(0)
            w.health['red_hearts'] = 6; refresh(w,1); w.last_damage_frame = w.frame
            command = c.step(w); await asyncio.sleep(0)
            models.plan.assert_not_called()
            self.assertFalse(command.get('use_item'))
            self.assertFalse(command.get('use_pill'))
        finally:
            await c.close()
