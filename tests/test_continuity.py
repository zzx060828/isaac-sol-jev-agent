import asyncio
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_agent import world, MemoryLog
from isaac_agent.control import Action, blocked, candidates, next_door, path_step, target_point
from isaac_agent.demo import frame_message
from isaac_agent.runtime import Controller, serve
from isaac_agent.strategy import StrategyExecutor, offers
from isaac_agent.state import World


class ContinuityTests(unittest.IsolatedAsyncioTestCase):
    def test_planner_backoff_survives_controller_replacement(self):
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            log = MemoryLog()
            log.directory = Path(directory) / 'run'
            with patch('isaac_agent.runtime.time.time', return_value=1000), \
                 patch('isaac_agent.runtime.time.monotonic', return_value=50):
                controller = Controller(SimpleNamespace(mode='hybrid'), log)
                controller.plan_failures = 4
                controller.save_plan_backoff(240)
            with patch('isaac_agent.runtime.time.time', return_value=1060), \
                 patch('isaac_agent.runtime.time.monotonic', return_value=10):
                replacement = Controller(SimpleNamespace(mode='hybrid'), log)
                self.assertEqual(replacement.plan_retry_at, 190)
                self.assertEqual(replacement.plan_failures, 4)

    def test_observed_creep_propagation_prompts_escape_before_contact(self):
        from isaac_agent.control import risk
        from isaac_agent.hazards import advancing_creep
        w = World()
        for msg in json.loads((Path(__file__).parent / 'fixtures/advancing_creep.json').read_text()):
            w.ingest(msg)
            if w.frame == 40232:
                break
        fronts = advancing_creep(w)
        self.assertTrue(any(front['confidence'] == 1 for front in fronts))
        self.assertLess(risk(w, (0, 1)), risk(w, (0, 0)))
        escape = candidates(w, {})[0][0]['action'].move
        self.assertEqual(escape[1], 1)
        self.assertLess(risk(w, escape), risk(w, (0, 0)))
        w.stats['can_fly'] = True
        self.assertEqual(advancing_creep(w), [])
        w.stats['can_fly'] = False
        w.frame += 5
        self.assertEqual(advancing_creep(w), [])
        w.ingest({'type': 'DATA', 'frame': w.frame+1, 'room_index': w.room+1, 'payload': {}})
        self.assertEqual(w.ground_history, [])

    def test_static_first_snapshot_does_not_invent_creep_motion(self):
        from isaac_agent.hazards import advancing_creep
        msg = frame_message()
        msg['payload']['FIRE_HAZARDS'] = [
            {'id': i, 'variant': 22, 'ground_only': True,
             'pos': {'x': 100+40*i, 'y': 300}, 'collision_radius': 24} for i in range(4)]
        w = World(); w.ingest(msg)
        self.assertEqual(advancing_creep(w), [])
        msg['frame'] = 3
        w.ingest(msg)
        self.assertEqual(advancing_creep(w), [])

    async def test_waiting_for_item_plan_does_not_spend_jev_calls_on_idle(self):
        from unittest.mock import AsyncMock
        from types import SimpleNamespace
        w = World()
        w.ingest(json.loads((Path(__file__).parent / 'fixtures/waiting_for_boss_reward_plan.json').read_text()))
        model = SimpleNamespace(mode='hybrid', plan=AsyncMock(), decide=AsyncMock())
        controller = Controller(model, MemoryLog())
        controller.plan_retry_at = float('inf')
        command = controller.step(w)
        self.assertEqual(command['move'], {'x': 0, 'y': 0})
        self.assertIsNone(controller.decision_task)
        # A reviewed clear-room route is executed locally without another JEV request.
        controller.plan.update(awaiting_strategy=False, mode='explore', door_slot='0')
        command = controller.step(w)
        self.assertNotEqual(command['move'], {'x':0,'y':0})
        self.assertIsNone(controller.decision_task)
        await controller.close()

    async def test_planner_rate_limit_backs_off_across_room_changes(self):
        from isaac_agent.models import ModelHTTPError
        from unittest.mock import AsyncMock
        from types import SimpleNamespace
        model = SimpleNamespace(mode='hybrid', plan=AsyncMock(side_effect=ModelHTTPError(429)),
                                decide=AsyncMock())
        w = world()
        controller = Controller(model, MemoryLog())
        with patch('isaac_agent.runtime.time.monotonic', return_value=100):
            await controller._plan(w, w.observation(), w.token, w.revision)
            self.assertEqual(controller.plan_retry_at, 130)
            controller.reset_actions(w)
            w.updated = 100
            controller.last_decision = 100
            controller.step(w, now=100)
            self.assertIsNone(controller.plan_task)
            self.assertEqual(model.plan.await_count, 1)
            await controller._plan(w, w.observation(), w.token, w.revision)
            self.assertEqual(controller.plan_retry_at, 160)
            model.plan.side_effect = ModelHTTPError(429, 'insufficient_quota')
            await controller._plan(w, w.observation(), w.token, w.revision)
            self.assertEqual(controller.plan_retry_at, 400)

    def test_damaged_tnt_has_no_fuse_but_still_blocks_unsafe_shots(self):
        from isaac_agent.hazards import explosives, threatened_explosives, safe_shot
        w = World()
        w.ingest(json.loads((Path(__file__).parent / 'fixtures/door_path_oscillation.json').read_text()))
        self.assertEqual(threatened_explosives(w, explosives(w)), set())
        best = candidates(w, {'strategy_enabled': True, 'mode': 'explore', 'door_slot': '2'})[0][0]
        self.assertGreater(best['action'].move[0], 0)
        w.player['pos'] = {'x': 400, 'y': 360}
        self.assertFalse(safe_shot(w, (0, 0), (1, 0)))
        w.payload['PROJECTILES']['player_tears'] = [{'pos': {'x': 450, 'y': 360}, 'vel': {'x': 8, 'y': 0}}]
        self.assertTrue(threatened_explosives(w, explosives(w)))

    def test_spatial_query_matches_full_grid_collision_scan(self):
        w = World()
        w.ingest(json.loads((Path(__file__).parent / 'fixtures/clear_room_with_npcs.json').read_text()))
        for radius in (6, 12, 30):
            for hazards_only in (False, True):
                for i in range(100):
                    p = (50 + (i*137 % 1060), 130 + (i*73 % 580))
                    actual = blocked(w, p, radius, exit_point=(40, 560), hazards_only=hazards_only)
                    with patch('isaac_agent.control.nearby_grid',
                               lambda world, point, radius: world.layout['grid'].values()):
                        expected = blocked(w, p, radius, exit_point=(40, 560), hazards_only=hazards_only)
                    self.assertEqual(actual, expected, (p, radius, hazards_only))

    def test_engine_clear_room_does_not_keep_fighting_remaining_npcs(self):
        from isaac_agent.control import risk
        w = World()
        w.ingest(json.loads((Path(__file__).parent / 'fixtures/clear_room_with_npcs.json').read_text()))
        self.assertEqual(len(w.enemies), 3)
        self.assertEqual(w.combat_enemies, [])
        self.assertEqual(w.observation()['enemies'], [])
        self.assertEqual(len(w.observation()['nonblocking_entities']), 3)
        best = candidates(w, {'strategy_enabled': True, 'mode': 'explore', 'door_slot': '4'})[0][0]['action']
        self.assertNotEqual(best.move, (0, 0))
        self.assertEqual(best.shoot, (0, 0))
        # Nonblocking entities still contribute to the collision shield.
        w.player['pos'] = dict(w.enemies[0]['pos'])
        with_entities = risk(w, (0, 0))
        w.payload['ENEMIES'] = []
        self.assertGreater(with_entities, risk(w, (0, 0)))

    def test_unknown_or_live_room_count_preserves_combat_targets(self):
        w = world(enemies=[{'id': 9, 'is_vulnerable': False, 'pos': {'x': 300, 'y': 300}}])
        for info in ({'is_clear': False, 'enemy_count': 1}, {'is_clear': True},
                     {'is_clear': True, 'enemy_count': 1}):
            w.payload['ROOM_INFO'].update(info)
            if 'enemy_count' not in info:
                w.payload['ROOM_INFO'].pop('enemy_count', None)
            self.assertEqual(w.combat_enemies, w.enemies)
        w.payload['ROOM_INFO'].update(is_clear=True, enemy_count=0)
        w.frame += 9
        self.assertEqual(w.combat_enemies, w.enemies)

    def test_tnt_approach_survives_small_changes_in_diagonal_visibility(self):
        w = World()
        w.ingest(json.loads((Path(__file__).parent / 'fixtures/off_axis_shooting_stall.json').read_text()))
        w.player['pos'] = {'x': 296, 'y': 264}
        first = candidates(w, {'target_enemy': 7})[0][0]
        self.assertIn('shoot TNT', first['description'])
        self.assertEqual(tuple(w.combat_breach['point']), (160, 360))
        w.player['pos'] = {'x': 280, 'y': 250}
        w.frame += 2
        second = candidates(w, {'target_enemy': 7})[0][0]
        self.assertIn('shoot TNT', second['description'])
        for g in w.layout['grid'].values():
            if g['type'] == 12 and (g['x'], g['y']) == (160, 360):
                g['collision'] = 0
        candidates(w, {'target_enemy': 7})
        self.assertFalse(w.combat_breach)

    def test_off_axis_misses_do_not_outweigh_reaching_a_firing_lane(self):
        w = World()
        w.ingest(json.loads((Path(__file__).parent / 'fixtures/off_axis_shooting_stall.json').read_text()))
        best = candidates(w, {'target_enemy': 7, 'combat_distance': 220})[0][0]['action']
        # Make lateral progress toward the firing lane; a longer-range policy
        # may first increase separation rather than cut through the inner arc.
        self.assertEqual(best.move[0], 1)

    def test_protected_spider_can_be_reached_by_safe_tnt_shot(self):
        w = World()
        w.ingest(json.loads((Path(__file__).parent / 'fixtures/tnt_protected_spiders.json').read_text()))
        best = candidates(w, {'target_enemy': 2})[0][0]
        self.assertEqual(best['action'].move, (1, 1))
        self.assertIn('shoot TNT from safe range', best['description'])
        self.assertEqual(best['action'].shoot, (-1, 0))

    def test_encounter_knowledge_is_bounded_and_deduplicated(self):
        from isaac_agent.knowledge import encounter_knowledge
        enemies = [{'type': t, 'variant': 0} for t in (94, 94, 24, 99, 27, 255, 85, 123456)]
        notes = encounter_knowledge(enemies, budget=1200)
        self.assertLessEqual(len(json.dumps(notes)), 1200)
        self.assertEqual(sum(n['entity'] == '94:0' for n in notes), 1)
        self.assertTrue(all('wiki.gg' in n['source_url'] for n in notes))
        self.assertFalse(any(n['entity'].startswith('123456:') for n in notes))

    def test_unlit_tnt_does_not_pin_player_away_from_nonshooting_spiders(self):
        w = World()
        w.ingest(json.loads((Path(__file__).parent / 'fixtures/unlit_tnt_combat.json').read_text()))
        best = candidates(w, {'mode': 'survive', 'target_enemy': 7})[0][0]
        self.assertNotEqual(best['action'].move, (0, 0))
        self.assertLess(best['risk'], 5)

    def test_short_step_can_reach_turn_in_narrow_pit_corridor(self):
        w = World()
        w.ingest(json.loads((Path(__file__).parent / 'fixtures/pit_corner.json').read_text()))
        best = candidates(w, {'strategy_enabled': True, 'mode': 'explore', 'door_slot': '2'})[0][0]['action']
        self.assertEqual(best.move[1], 1)
        self.assertNotEqual(best.move, (0, 0))

    async def test_damage_does_not_starve_valid_combat_target_updates(self):
        w, log = world(), MemoryLog()
        w.payload['ENEMIES'] = [{'id': 2, 'pos': {'x': 200, 'y': 280}}]
        class Fake:
            mode = 'hybrid'
            async def plan(self, observation):
                w.ingest({'type': 'EVENT', 'event': 'PLAYER_DAMAGE', 'frame': w.frame})
                return {'objective': 'Focus the boss', 'mode': 'combat', 'target_enemy': 2}, {}
        c = Controller(Fake(), log)
        await c._plan(w, w.observation(), w.token, w.revision)
        self.assertEqual(c.plan['target_enemy'], 2)
        self.assertTrue(any(r['event'] == 'plan' for r in log.rows))

    async def test_cancel_server_closes_live_connection_before_waiting(self):
        original_start = asyncio.start_server
        connected = asyncio.Event()
        endpoint = []
        async def start(*args, **kwargs):
            server = await original_start(*args, **kwargs)
            endpoint.append(server.sockets[0].getsockname()[1])
            connected.set()
            return server
        class Fake:
            mode = 'local'
        log = MemoryLog()
        log.directory = 'test'
        with patch('isaac_agent.runtime.asyncio.start_server', side_effect=start):
            task = asyncio.create_task(serve(Fake(), log, '127.0.0.1', 0))
            await asyncio.wait_for(connected.wait(), 2)
            reader, writer = await asyncio.open_connection('127.0.0.1', endpoint[0])
            writer.write((json.dumps(frame_message())+'\n').encode())
            await writer.drain()
            await asyncio.wait_for(reader.readline(), 2)
            task.cancel()
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
                tail = await asyncio.wait_for(reader.read(), 2)
                self.assertIn(b'MANUAL', tail)
            finally:
                writer.close()
                await writer.wait_closed()

    async def test_trial_pauses_while_control_connection_is_still_live(self):
        original_start = asyncio.start_server
        connected = asyncio.Event()
        endpoint = []
        async def start(*args, **kwargs):
            server = await original_start(*args, **kwargs)
            endpoint.append(server.sockets[0].getsockname()[1])
            connected.set()
            return server
        class Fake:
            mode = 'local'
        log = MemoryLog()
        log.directory = 'test'
        pause_calls = []
        async def pause(bridge, current_log):
            self.assertIsNotNone(bridge.writer)
            self.assertFalse(bridge.writer.is_closing())
            self.assertFalse(any(r['event'] == 'disconnected' for r in log.rows))
            pause_calls.append(True)
        with patch('isaac_agent.runtime.asyncio.start_server', side_effect=start), \
             patch('isaac_agent.runtime.pause_game_before_release', side_effect=pause):
            task = asyncio.create_task(serve(Fake(), log, '127.0.0.1', 0, pause_on_stop=True))
            await asyncio.wait_for(connected.wait(), 2)
            reader, writer = await asyncio.open_connection('127.0.0.1', endpoint[0])
            writer.write((json.dumps(frame_message())+'\n').encode())
            await writer.drain()
            await asyncio.wait_for(reader.readline(), 2)
            task.cancel()
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
                tail = await asyncio.wait_for(reader.read(), 2)
                self.assertIn(b'MANUAL', tail)
                self.assertEqual(pause_calls, [True])
            finally:
                writer.close()
                await writer.wait_closed()

    async def test_floor_completion_keeps_state_until_pause_and_then_releases(self):
        original_start=asyncio.start_server
        connected=asyncio.Event();endpoint=[];pause_calls=[]
        async def start(*args,**kwargs):
            server=await original_start(*args,**kwargs)
            endpoint.append(server.sockets[0].getsockname()[1]);connected.set()
            return server
        class Fake:
            mode='local'
        log=MemoryLog();log.directory='test'
        async def pause(bridge,current_log):
            self.assertTrue(bridge.world.ready(__import__('time').monotonic()))
            self.assertIsNotNone(bridge.writer)
            self.assertFalse(bridge.writer.is_closing())
            self.assertEqual(sum(r['event']=='floor_complete' for r in log.rows),1)
            self.assertFalse(any(r['event']=='disconnected' for r in log.rows))
            pause_calls.append(True)
        with patch('isaac_agent.runtime.asyncio.start_server',side_effect=start), \
             patch('isaac_agent.runtime.pause_game_before_release',side_effect=pause), \
             patch('isaac_agent.runtime.floor_finished',return_value=True):
            task=asyncio.create_task(serve(Fake(),log,'127.0.0.1',0,
                                         stop_after_floor=1,pause_on_stop=True))
            await asyncio.wait_for(connected.wait(),2)
            reader,writer=await asyncio.open_connection('127.0.0.1',endpoint[0])
            try:
                # Two successive completion samples must not duplicate the event.
                writer.write(((json.dumps(frame_message())+'\n')*2).encode());await writer.drain()
                await asyncio.wait_for(task,2)
                tail=await asyncio.wait_for(reader.read(),2)
                self.assertIn(b'MANUAL',tail)
                self.assertEqual(pause_calls,[True])
            finally:
                writer.close();await writer.wait_closed()

    def test_timed_out_pickup_does_not_hold_a_finished_choice(self):
        w = world()
        w.payload['PICKUPS'] = [{'id': 1, 'variant': 20, 'sub_type': 1,
                               'pos': {'x': 400, 'y': 280}}]
        key = offers(w)[0]['choice']
        executor = StrategyExecutor()
        plan = {'strategy_enabled': True, 'strategy_choice': key, 'mode': 'collect'}
        executor.update(w, plan, MemoryLog())
        w.frame += 181
        w.sampled.update({name: w.frame for name in w.payload})
        execution = executor.update(w, plan, MemoryLog())
        self.assertIsNone(execution['strategy_choice'])
        self.assertTrue(target_point(w, execution)[1])
        execution = executor.update(w, plan, MemoryLog())
        self.assertTrue(target_point(w, execution)[1])

    def test_actual_position_can_escape_when_snapped_origin_hits_barrel(self):
        w = world(player=(90.19, 383.2))
        w.stats['size'] = 10
        w.layout['grid'] = {'1': {'type': 12, 'collision': 2, 'x': 120, 'y': 360}}
        self.assertNotEqual(path_step(w, (320, 400)), (90.19, 383.2))

    def test_locked_treasure_room_requires_a_key(self):
        w = world()
        door = w.layout['doors']['2']
        door.update(is_open=False, is_locked=True, target_room_type=4)
        w.rooms[str(w.room)]['doors']['2'] = deepcopy(door)
        w.inventory['keys'] = 0
        self.assertIsNone(next_door(w))
        w.inventory['keys'] = 1
        self.assertEqual(next_door(w), door)

    def test_floor_exit_is_deliberate_and_can_be_reached_after_exploration(self):
        w = world()
        w.info.update(room_type=5, is_clear=True)
        w.layout['grid'] = {'37': {'type': 17, 'variant': 0, 'state': 1,
                                  'grid_index': 37, 'collision': 0, 'x': 320, 'y': 200}}
        door = w.layout['doors']['2']
        w.rooms[str(door['target_room'])] = {'visited': True, 'doors': {}}
        self.assertTrue(blocked(w, (320, 200)))
        self.assertFalse(blocked(w, (320, 200), exit_point=(320, 200)))
        self.assertEqual(target_point(w, {}), ((320, 200), True))
        self.assertIn('descend:37', {offer['choice'] for offer in offers(w)})
        waiting = {'strategy_enabled': True, 'awaiting_strategy': True}
        self.assertEqual(target_point(w, waiting), ((320.0, 280.0), False))
        self.assertEqual(target_point(w, {**waiting, 'mode': 'explore', 'awaiting_strategy': False}),
                         ((320.0, 280.0), False))
        executor = StrategyExecutor()
        execution = executor.update(w, {**waiting, 'strategy_choice': 'descend:37'}, MemoryLog())
        self.assertEqual(target_point(w, execution), ((320, 200), True))

    def test_opening_animation_does_not_erase_observed_floor_exit(self):
        msg = frame_message()
        msg['payload']['ROOM_INFO']['room_type'] = 5
        msg['payload']['ROOM_LAYOUT']['grid'] = {
            '37': {'type': 17, 'variant': 0, 'state': 0, 'grid_index': 37,
                   'collision': 0, 'x': 320, 'y': 200}}
        w = World()
        w.ingest(msg)
        self.assertTrue(w.rooms[str(w.room)]['floor_exit'])
        self.assertNotIn('descend:37', {offer['choice'] for offer in offers(w)})

    def test_explored_floor_routes_back_to_known_exit(self):
        w = world()
        destination = str(w.layout['doors']['2']['target_room'])
        w.rooms[destination] = {'floor_exit': True, 'doors': {}}
        self.assertEqual(next_door(w), w.layout['doors']['2'])

    async def test_strategy_waits_for_inventory_and_pickup_snapshot(self):
        class Fake:
            mode = 'hybrid'
            async def plan(self, obs):
                return {'objective': 'go', 'mode': 'explore'}, {}
            async def decide(self, *args):
                return Action(), {'confidence': .1}
        message = frame_message()
        message['payload'].pop('PLAYER_INVENTORY')
        message['payload'].pop('PICKUPS')
        w, log = World(), MemoryLog()
        w.ingest(message)
        c = Controller(Fake(), log)
        c.step(w)
        self.assertFalse(any(r['event'] == 'plan_request' for r in log.rows))
        complete = frame_message(5)
        complete['payload']['PICKUPS'] = [{'id': 9, 'variant': 100, 'sub_type': 531,
                                         'pos': {'x': 400, 'y': 280}}]
        w.ingest(complete)
        c.step(w)
        self.assertTrue(any(r['event'] == 'plan_request' for r in log.rows))
        await c.close()

    async def test_empty_lua_inventory_maps_do_not_disconnect_controller(self):
        class Fake:
            mode = 'local'
        w, log = world(), MemoryLog()
        w.inventory.update(active_items=[], collectibles=[], can_use=False)
        c = Controller(Fake(), log)
        result = c.step(w)
        self.assertIn('move', result)
        self.assertEqual(w.inventory['active_items'], {})
        self.assertEqual(w.observation()['inventory']['item_names'], {})
        await c.close()

    async def test_changed_item_choices_invalidate_old_exploration_plan(self):
        w, log = world(), MemoryLog()
        class Fake:
            mode = 'hybrid'
            async def plan(self, observation):
                w.payload['PICKUPS'] = [{'id': 9, 'variant': 100, 'sub_type': 459, 'pos': {'x': 400, 'y': 280}}]
                return {'objective': 'leave', 'mode': 'explore'}, {}
        c = Controller(Fake(), log)
        await c._plan(w, w.observation(), w.token, w.revision)
        self.assertTrue(any(r['event'] == 'plan_discarded' for r in log.rows))
        await c.close()

    def test_waiting_plan_does_not_stop_ordinary_room_traversal(self):
        w = world()
        plan = {'strategy_enabled': True, 'awaiting_strategy': True}
        opts, _ = candidates(w, plan)
        self.assertNotEqual(opts[0]['action'].move, (0, 0))
        w.payload['PICKUPS'] = [{'id': 2, 'variant': 100, 'sub_type': 482,
                               'options_index': 1, 'pos': {'x': 400, 'y': 280}}]
        self.assertEqual(candidates(w, plan)[0][0]['action'].move, (0, 0))

    def test_exit_route_cannot_touch_unselected_item(self):
        w = world(player=(280, 280))
        w.payload['PICKUPS'] = [{'id': 2, 'variant': 100, 'sub_type': 482,
                               'options_index': 1, 'pos': {'x': 320, 'y': 280}}]
        self.assertTrue(blocked(w, (320, 280), exit_point=(600, 280)))
        step = path_step(w, (600, 280))
        self.assertNotEqual(step[1], 280)
        self.assertFalse(blocked(w, (320, 280), exit_point=(320, 280)))

    def test_navigation_cache_invalidates_on_changed_terrain(self):
        w = world(player=(280, 280))
        self.assertEqual(path_step(w, (400, 280)), (400, 280))
        changed = frame_message(2, player=(280, 280))
        changed['payload']['ROOM_LAYOUT']['grid'] = {'1': {'x': 320, 'y': 280, 'type': 2, 'collision': 3}}
        w.ingest(changed)
        self.assertNotEqual(path_step(w, (400, 280))[1], 280)

    def test_initial_pedestal_spawn_wait_finishes_before_choice_request(self):
        w = world()
        w.payload['PICKUPS'] = [{'id': 1, 'variant': 100, 'sub_type': 459, 'wait': 15,
                               'pos': {'x': 400, 'y': 280}}]
        signature = w.strategy_signature()
        self.assertFalse(w.strategy_ready())
        w.payload['PICKUPS'][0]['wait'] = 0
        self.assertTrue(w.strategy_ready())
        self.assertNotEqual(signature, w.strategy_signature())


if __name__ == '__main__':
    unittest.main()
