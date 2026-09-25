import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from isaac_agent.control import Action, blocked, candidates, next_door, path_step, risk, shield, target_point
from isaac_agent.demo import frame_message
from isaac_agent.models import Models, configuration, validate_plan
from isaac_agent.runtime import BridgeServer, Controller, Log
from isaac_agent.state import World, records


class MemoryLog:
    def __init__(self):
        self.rows = []

    def write(self, event, **data):
        self.rows.append({"event": event, **data})


def world(**kwargs):
    w = World()
    w.ingest(frame_message(**kwargs))
    return w


class StateTests(unittest.TestCase):
    def test_room_change_drops_old_projectiles_and_player(self):
        w = world(projectiles=[{"id": 1}])
        w.ingest({"type": "DATA", "frame": 2, "room_index": 2, "payload": {"ENEMIES": []}})
        self.assertEqual(w.projectiles, [])
        self.assertFalse(w.ready(time.monotonic()))

    def test_empty_lua_table_is_empty_entity_list(self):
        self.assertEqual(records({}), [])
        w = world()
        w.payload["ENEMIES"] = {}
        w.payload["PROJECTILES"]["enemy_projectiles"] = {}
        self.assertTrue(candidates(w, {})[0])

    def test_sensor_age_stops_control(self):
        w = world()
        w.ingest({"type": "DATA", "frame": 20, "room_index": 1, "payload": {}})
        self.assertFalse(w.ready(time.monotonic()))

    def test_new_level_clears_map_even_with_same_room(self):
        w = world()
        original = w.token
        w.rooms['999'] = {"visited": True}
        msg = frame_message(2)
        msg['agent']['level_id'] = 'demo:2:0'
        w.ingest(msg)
        self.assertNotEqual(original, w.token)
        self.assertNotIn('999', w.rooms)

    def test_restart_invalidates_epoch(self):
        w = world(frame=50)
        token = w.token
        w.ingest(frame_message(1))
        self.assertNotEqual(token, w.token)


class ControlTests(unittest.TestCase):
    def test_empty_pedestal_is_not_a_pickup_or_model_collection_goal(self):
        w = world(player=(250, 280))
        pedestal = {"id": 1, "variant": 100, "sub_type": 0, "price": 0, "wait": 0,
                    "pos": {"x": 250, "y": 300}}
        w.payload['PICKUPS'] = [pedestal]
        self.assertEqual(target_point(w, {}), ((600.0, 280.0), True))
        self.assertEqual(w.observation()['pickups'], [])
        self.assertEqual(w.observation()['empty_pedestals'], [pedestal])
        self.assertEqual(w.payload['PICKUPS'], [pedestal])  # Keep the raw sensor record.

    def test_real_and_unknown_collectibles_remain_targets_until_collected(self):
        w = world(player=(250, 280))
        item = {"id": 1, "variant": 100, "sub_type": 48, "price": 0, "wait": 0,
                "pos": {"x": 250, "y": 300}}
        w.payload['PICKUPS'] = [item]
        for subtype in (48, -1):  # A hidden/unknown item must not be assumed empty.
            item['sub_type'] = subtype
            self.assertEqual(target_point(w, {}), ((250.0, 300.0), False))
            self.assertEqual(w.observation()['pickups'][0]['sub_type'], subtype)
            self.assertIn('name', w.observation()['pickups'][0])
        item['sub_type'] = 0  # Same entity ID remains after the item is taken.
        self.assertTrue(target_point(w, {})[1])

    def test_navigation_routes_around_empty_pedestal(self):
        w = world(player=(280, 280))
        w.payload['PICKUPS'] = [{"variant": 100, "sub_type": 0, "pos": {"x": 320, "y": 280}}]
        self.assertTrue(blocked(w, (320, 280)))
        waypoint = path_step(w, (400, 280))
        self.assertNotEqual(waypoint[1], 280)
        self.assertFalse(blocked(w, waypoint))

    def test_incoming_projectile_triggers_override(self):
        w = world(projectiles=[{"pos": {"x": 370, "y": 280}, "vel": {"x": -6, "y": 0}, "collision_radius": 8}])
        options, exit_point = candidates(w, {})
        chosen, overridden = shield(w, Action((1, 0)), options[0]["action"], exit_point)
        self.assertTrue(overridden)
        self.assertLess(risk(w, chosen.move), risk(w, (1, 0)))

    def test_terrain_collision_enums_and_flight(self):
        w = world()
        w.layout['grid'] = {'1': {'x': 320, 'y': 280, 'collision': 1, 'type': 7}}
        self.assertTrue(blocked(w, (320, 280)))
        w.stats['can_fly'] = True
        self.assertFalse(blocked(w, (320, 280)))
        w.layout['grid']['1']['collision'] = 3
        self.assertTrue(blocked(w, (320, 280)))

    def test_navigation_detours_around_rock(self):
        w = world(player=(280, 280))
        w.layout['grid'] = {'1': {'x': 320, 'y': 280, 'collision': 3, 'type': 2}}
        step = path_step(w, (400, 280))
        self.assertNotEqual(step[1], 280)
        self.assertFalse(blocked(w, step))

    def test_bfs_backtracks_toward_frontier(self):
        w = world(room=2)
        w.layout['doors']['2']['target_room'] = 1
        w.rooms = {'2': {'doors': deepcopy(w.layout['doors'])}, '1': {'doors': {'0':
                  {'target_room': 3, 'is_open': True, 'is_locked': False}}}}
        self.assertEqual(next_door(w)['target_room'], 1)

    def test_input_has_expiry_and_explicit_release(self):
        command = Action((1, 0), (0, 1)).wire(world(frame=10), 7)
        self.assertEqual(command['agent_deadline'], 16)
        self.assertIs(command['use_bomb'], False)
        self.assertEqual(command['agent_seq'], 7)

    def test_planner_rejects_invented_entities(self):
        with self.assertRaises(ValueError):
            validate_plan({'objective': 'go', 'door_slot': 'fake'}, world().observation())


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_decision_revalidated_with_short_new_lease(self):
        w, log = world(), MemoryLog()
        class Fake:
            mode = 'jev'
            async def decide(self, obs, plan, options):
                w.ingest(frame_message(25))
                return Action((1, 0)), {'confidence': .99, 'latency_ms': 800}
        c = Controller(Fake(), log)
        await c._decide(w, w.observation(), w.token, w.frame, candidates(w, {})[0])
        self.assertEqual(c.action_until, 31)
        self.assertTrue(any(row['event'] == 'decision' and row['revalidated'] for row in log.rows))
        await c.close()

    async def test_slow_decision_does_not_block_inputs_and_is_discarded_after_room_change(self):
        gate = asyncio.Event()
        class Fake:
            mode = 'jev'
            async def decide(self, obs, plan, options):
                await gate.wait()
                return Action((1, 0)), {'confidence': .99, 'latency_ms': 900}
        w, log = world(), MemoryLog()
        c = Controller(Fake(), log)
        command = c.step(w)
        self.assertIn('move', command)
        await asyncio.sleep(0)
        w.ingest(frame_message(2, room=3))
        gate.set()
        await c.decision_task
        self.assertTrue(any(row['event'] == 'decision_discarded' for row in log.rows))
        await c.close()

    async def test_old_plan_discarded_when_damage_arrives(self):
        gate = asyncio.Event()
        class Fake:
            mode = 'hybrid'
            async def plan(self, obs):
                await gate.wait()
                return {'objective': 'old'}, {}
            async def decide(self, *args):
                await gate.wait()
                return Action(), {'confidence': .9}
        w, log = world(), MemoryLog()
        c = Controller(Fake(), log)
        c.plan_task = asyncio.create_task(c._plan(w, w.observation(), w.token, w.revision))
        await asyncio.sleep(0)
        w.ingest({'type': 'EVENT', 'event': 'PLAYER_DAMAGE'})
        gate.set()
        await c.plan_task
        self.assertTrue(any(r['event'] == 'plan_discarded' for r in log.rows))
        await c.close()

    async def test_timeout_falls_back_without_killing_controller(self):
        class Fake:
            mode = 'jev'
            async def decide(self, *args):
                raise TimeoutError('test timeout')
        w, log = world(), MemoryLog()
        c = Controller(Fake(), log)
        c.step(w)
        await c.decision_task
        command = c.step(w)
        self.assertIn('move', command)
        self.assertTrue(any(r['event'] == 'decision_error' for r in log.rows))
        await c.close()

    async def test_api_contract_and_candidate_validation(self):
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': 'test'}), patch('isaac_agent.models.load_keys'):
            models = Models(configuration(), 'jev')
            w = world()
            opts = candidates(w, {})[0]
            choice = opts[0]['action'].id
            with patch('isaac_agent.models.post', return_value={'answers': {'action': {'choice': choice, 'confidence': .8}}}) as post:
                result, _ = await models.decide(w.observation(), {}, opts)
                body = post.call_args.args[1]
                self.assertEqual(body['questions']['action']['type'], 'choice')
                self.assertEqual(result.id, choice)
            with patch('isaac_agent.models.post', return_value={'answers': {'action': {'choice': 'invalid', 'confidence': .8}}}):
                with self.assertRaises(ValueError):
                    await models.decide(w.observation(), {}, opts)

    async def test_openrouter_usage_and_probability_response(self):
        config = configuration()
        config['jev'].update(url='https://openrouter.ai/api/alpha/decisions',
                             model='typesafe/jev-1.13', key_env='OPENROUTER_API_KEY')
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test'}), patch('isaac_agent.models.load_keys'):
            models = Models(config, 'jev')
            w = world()
            opts = candidates(w, {})[0]
            choice = opts[0]['action'].id
            answer = {'choice': choice, 'probabilities': {choice: .9}}
            response = {'answers': {'action': answer}, 'usage': {
                'inputTokens': 20, 'outputTokens': 3, 'cost': .000001}}
            with patch('isaac_agent.models.post', return_value=response):
                action, meta = await models.decide(w.observation(), {}, opts)
                self.assertEqual(action.id, choice)
                self.assertEqual(meta['confidence'], 0)
                self.assertEqual(meta['confidence_source'], 'missing')
                self.assertEqual(meta['probabilities'], {choice: .9})
                self.assertEqual(meta['usage']['input_tokens'], 20)
                self.assertEqual(meta['usage']['output_tokens'], 3)
                self.assertEqual(meta['usage']['cost'], .000001)
                answer['confidence'] = 0
                _, meta = await models.decide(w.observation(), {}, opts)
                self.assertEqual(meta['confidence'], 0)
                answer['confidence'] = float('nan')
                with self.assertRaises(ValueError):
                    await models.decide(w.observation(), {}, opts)

    async def test_tcp_fragmented_frame_handshake_input_and_disconnect(self):
        models = Models(configuration(), 'local')
        log = MemoryLog()
        bridge = BridgeServer(models, log)
        try:
            server = await asyncio.start_server(bridge.handle, '127.0.0.1', 0)
        except PermissionError:
            self.skipTest('Sandbox disallows loopback sockets')
        try:
            port = server.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
            encoded = (json.dumps(frame_message()) + '\n').encode()
            writer.write(encoded[:20]); await writer.drain()
            await asyncio.sleep(.01)
            writer.write(encoded[20:]); await writer.drain()
            mode = json.loads(await asyncio.wait_for(reader.readline(), 2))
            command = json.loads(await asyncio.wait_for(reader.readline(), 2))
            self.assertEqual(mode['params']['mode'], 'FORCE_AI')
            self.assertIn('agent_deadline', command)
            writer.close(); await writer.wait_closed()
            await asyncio.sleep(.03)
        finally:
            server.close(); await server.wait_closed()
            await bridge.close()


if __name__ == '__main__':
    unittest.main()
