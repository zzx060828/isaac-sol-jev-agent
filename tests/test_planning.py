import asyncio
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.control import Action, target_point
from isaac_agent.demo import frame_message
from isaac_agent.planning import planning_context, planning_reasons
from isaac_agent.runtime import Controller
from isaac_agent.state import World


class MemoryLog:
    def __init__(self):
        self.rows = []

    def write(self, event, **data):
        self.rows.append({'event': event, **data})


def pickup(variant=100, price=0):
    return {'id': 7, 'variant': variant, 'sub_type': 1, 'price': price,
            'wait': 0, 'pos': {'x': 400, 'y': 280}}


class PlanningTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.world = World()
        self.message = frame_message()
        self.models = SimpleNamespace(mode='hybrid', plan=AsyncMock(
            return_value=({'objective': 'Continue', 'mode': 'combat'}, {})))
        self.log = MemoryLog()
        self.controller = Controller(self.models, self.log)
        self.controller.last_decision = float('inf')

    async def asyncTearDown(self):
        await self.controller.close()

    def step(self, now):
        self.message['frame'] += 1
        self.world.ingest(deepcopy(self.message))
        self.world.updated = now
        self.controller.step(self.world, now)

    async def settle(self, now):
        self.step(now)
        if self.controller.plan_task:
            await self.controller.plan_task

    def requests(self):
        return [r for r in self.log.rows if r['event'] == 'plan_request']

    async def test_no_heartbeat_or_ordinary_combat_event_calls(self):
        self.message['payload']['ENEMIES'] = [
            {'id': 2, 'pos': {'x': 470, 'y': 220}, 'collision_radius': 15}]
        self.message['payload']['ROOM_INFO']['is_clear'] = False
        await self.settle(100)
        self.world.revision += 1
        self.world.last_damage_frame = self.world.frame
        self.message['payload']['PLAYER_HEALTH'][0]['red_hearts'] = 4
        await self.settle(104)
        self.message['payload']['ENEMIES'] = []
        self.message['payload']['ROOM_INFO']['is_clear'] = True
        await self.settle(108)
        await self.settle(140)
        await self.settle(200)
        self.assertEqual(len(self.requests()), 0)

    async def test_jev_continues_and_local_target_switches_after_kill_and_damage(self):
        self.models.plan.return_value = ({'mode': 'combat', 'target_enemy': 2}, {})
        self.models.decide = AsyncMock(return_value=(Action(), {'confidence': 0}))
        self.controller.last_decision = -100
        enemies = [
            {'id': 2, 'pos': {'x': 370, 'y': 220}, 'collision_radius': 15},
            {'id': 3, 'pos': {'x': 250, 'y': 380}, 'collision_radius': 15}]
        self.message['payload']['ENEMIES'] = enemies
        self.message['payload']['ROOM_INFO']['is_clear'] = False
        await self.settle(100)
        await self.controller.decision_task
        self.assertEqual(target_point(self.world, self.controller.plan)[0], (370, 220))
        self.message['payload']['ENEMIES'] = enemies[1:]
        await self.settle(104)
        await self.controller.decision_task
        self.assertEqual(target_point(self.world, self.controller.plan)[0], (250, 380))
        self.assertEqual([e['id'] for e in self.models.decide.call_args.args[0]['enemies']], [3])
        self.world.revision += 1
        self.world.last_damage_frame = self.world.frame
        self.message['payload']['PLAYER_HEALTH'][0]['red_hearts'] = 4
        await self.settle(108)
        await self.controller.decision_task
        self.assertEqual(self.models.decide.await_count, 3)
        self.assertEqual(len(self.requests()), 0)

    async def test_room_item_and_inventory_changes_trigger(self):
        await self.settle(100)
        self.message['room_index'] = 2
        await self.settle(104)
        self.message['payload']['PICKUPS'] = [pickup()]
        await self.settle(108)
        self.message['payload']['PICKUPS'] = []
        self.message['payload']['PLAYER_INVENTORY'][0]['collectibles'] = {'1': 1}
        await self.settle(112)
        self.assertEqual(len(self.requests()), 1)
        self.assertIn('resource_choices_changed', self.requests()[0]['triggers'])

    async def test_free_bomb_is_local_but_shop_affordability_triggers(self):
        await self.settle(100)
        self.message['payload']['PICKUPS'] = [pickup(40)]
        await self.settle(104)
        self.message['payload']['PICKUPS'] = [pickup(40, 5)]
        self.message['payload']['PLAYER_INVENTORY'][0]['coins'] = 4
        await self.settle(108)
        self.assertEqual(len(self.requests()), 0)
        self.message['payload']['PLAYER_INVENTORY'][0]['coins'] = 5
        await self.settle(112)
        self.assertEqual(len(self.requests()), 1)
        self.message['payload']['PLAYER_INVENTORY'][0]['coins'] = 6
        await self.settle(116)
        self.assertEqual(len(self.requests()), 1)

    async def test_blood_trade_and_boss_exit_are_resource_decisions(self):
        await self.settle(100)
        self.message['payload']['INTERACTABLES'] = [
            {'id': 9, 'variant': 2, 'pos': {'x': 400, 'y': 280}}]
        await self.settle(104)
        self.assertIn('resource_choices_changed', self.requests()[-1]['triggers'])
        self.message['payload']['ROOM_INFO']['room_type'] = 5
        self.message['payload']['ROOM_LAYOUT']['grid'] = {
            '22': {'grid_index': 22, 'type': 17, 'variant': 0,
                   'state': 1, 'x': 400, 'y': 300}}
        await self.settle(108)
        self.assertEqual(len(self.requests()), 2)

    async def test_single_pending_request_and_cooldown_coalesce_changes(self):
        gate = asyncio.Event()

        async def delayed(observation):
            await gate.wait()
            return {'objective': 'Continue', 'mode': 'combat'}, {'usage': {'input_tokens': 10}}

        self.models.plan.side_effect = delayed
        self.message['payload']['INTERACTABLES'] = [{'id': 9, 'variant': 2, 'pos': {'x': 400, 'y': 280}}]
        self.step(100)
        await asyncio.sleep(0)
        self.message['payload']['PICKUPS'] = [pickup()]
        self.step(101)
        self.step(102)
        self.assertEqual(len(self.requests()), 1)
        gate.set()
        await self.controller.plan_task
        discarded = [r for r in self.log.rows if r['event'] == 'plan_discarded']
        self.assertEqual(discarded[0]['usage']['input_tokens'], 10)
        await self.settle(102.5)
        self.assertEqual(len(self.requests()), 1)
        await self.settle(103)
        await self.settle(130)
        self.assertEqual(len(self.requests()), 2)

    async def test_pending_retry_obeys_backoff_without_periodic_trigger(self):
        self.message['payload']['PICKUPS'] = [pickup()]
        await self.settle(100)
        self.controller.plan_pending = 'api_retry'
        self.controller.plan_retry_at = 160
        await self.settle(130)
        self.assertEqual(len(self.requests()), 1)
        await self.settle(160)
        self.assertEqual(self.requests()[-1]['triggers'], ['api_retry'])
        await self.settle(200)
        self.assertEqual(len(self.requests()), 2)

    async def test_finished_free_pickup_does_not_request_new_plan(self):
        self.message['payload']['PICKUPS'] = [pickup(40)]
        self.models.plan.return_value = ({'objective': 'Collect bomb',
                                         'strategy_choice': 'pickup:7:40:1:0'}, {})
        await self.settle(100)
        await self.settle(101)
        self.message['payload']['PICKUPS'] = []
        await self.settle(104)
        await self.settle(140)
        self.assertEqual(len(self.requests()), 0)
        self.assertIsNone(self.controller.plan.get('strategy_choice'))
        self.assertFalse(self.controller.plan['awaiting_strategy'])

    async def test_local_healing_charge_does_not_trigger_but_new_active_readiness_does(self):
        inv = self.message['payload']['PLAYER_INVENTORY'][0]
        inv['active_items'] = {'0': {'item': 45, 'charge': 0, 'max_charge': 6}}
        await self.settle(100)
        inv['active_items']['0']['charge'] = 6
        await self.settle(104)
        self.assertEqual(len(self.requests()), 0)
        inv['active_items']['0'].update(item=999, charge=0)
        await self.settle(108)
        inv['active_items']['0']['charge'] = 6
        await self.settle(112)
        self.assertIn('active_item_ready_changed', self.requests()[-1]['triggers'])

    async def test_locked_shop_affordability_not_every_key_change(self):
        self.world.ingest(self.message)
        self.world.layout['doors']['2'].update(is_locked=True, target_room_type=2)
        self.world.inventory['keys'] = 0
        before = planning_context(self.world)
        self.world.inventory['keys'] = 1
        after = planning_context(self.world)
        self.assertEqual(planning_reasons(before, after), ['locked_room_access_changed'])
        self.world.inventory['keys'] = 2
        self.assertEqual(planning_reasons(after, planning_context(self.world)), [])
