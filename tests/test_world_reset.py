import asyncio
import json
from types import SimpleNamespace
import unittest

from isaac_agent.demo import frame_message
from isaac_agent.memory import RunMemory
from isaac_agent.runtime import BridgeServer
from isaac_agent.state import World


class ResetTests(unittest.TestCase):
    def old_world(self):
        w = World()
        w.ingest(frame_message(90))
        w.ingest({'type': 'EVENT', 'event': 'PLAYER_DAMAGE', 'frame': 90,
                  'data': {'source_type': 10}})
        w.ingest({'type': 'EVENT', 'event': 'PLAYER_DEATH', 'frame': 90})
        return w

    def test_game_start_keeps_only_new_run_events(self):
        w = self.old_world()
        old_token = w.token
        w.ingest({'type': 'EVENT', 'event': 'GAME_START', 'frame': 0,
                  'data': {'continued': False}})
        self.assertNotEqual(w.token, old_token)
        self.assertFalse(w.dead)
        self.assertEqual(w.control_mode, 'MANUAL')
        self.assertEqual(w.capabilities, {})
        w.ingest(frame_message(1))
        self.assertEqual([e['event'] for e in w.observation()['recent_events']], ['GAME_START'])
        self.assertEqual(w.control_mode, 'FORCE_AI')

    def test_frame_rollback_without_game_start_drops_old_authority(self):
        w = self.old_world()
        msg = frame_message(1)
        del msg['agent']['control_mode']
        w.ingest(msg)
        self.assertFalse(w.dead)
        self.assertEqual(w.events, [])
        self.assertEqual(w.control_mode, 'MANUAL')
        self.assertEqual(w.last_damage_frame, -1000)

    def test_floor_change_retains_same_run_events_and_epoch(self):
        w = World()
        w.ingest(frame_message(90))
        w.ingest({'type': 'EVENT', 'event': 'PLAYER_DAMAGE', 'frame': 90})
        epoch = w.epoch
        msg = frame_message(93)
        msg['agent']['level_id'] = 'demo:2:0'
        w.ingest(msg)
        self.assertEqual(w.epoch, epoch)
        self.assertEqual([e['event'] for e in w.events], ['PLAYER_DAMAGE'])
        self.assertEqual(w.control_mode, 'FORCE_AI')

    def test_reset_preserves_memory_object_and_received_packet(self):
        w = World()
        memory = w.run_memory = RunMemory()
        msg = frame_message(90)
        w.ingest(msg)
        memory.observe(w, msg)
        memory.add(w, 'test_evidence', {'observed': True})
        nodes = list(memory.nodes)
        w.reset()
        self.assertIs(w.run_memory, memory)
        self.assertEqual(memory.nodes, nodes)
        self.assertEqual(msg['agent']['protocol'], 1)
        self.assertEqual(w.capabilities, {})
        resumed = frame_message(93)
        resumed['agent']['continued_run'] = True
        w.ingest(resumed)
        memory.observe(w, resumed)
        self.assertTrue(any(n['kind'] == 'test_evidence' for n in memory.nodes))


class ReconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconnect_waits_for_current_mode_and_rejects_missing_protocol(self):
        class Log:
            def __init__(self):
                self.rows = []
            def write(self, event, **data):
                self.rows.append({'event': event, **data})

        log = Log()
        # No key loading, model implementation or paid endpoint in this test.
        bridge = BridgeServer(SimpleNamespace(mode='local'), log)
        server = await asyncio.start_server(bridge.handle, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        writers = []

        async def connect(msg):
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
            writers.append(writer)
            writer.write((json.dumps(msg) + '\n').encode())
            await writer.drain()
            return reader, writer

        async def read(reader):
            return json.loads(await asyncio.wait_for(reader.readline(), 2))

        async def disconnect(writer, count):
            writer.close()
            await writer.wait_closed()
            async def finished():
                while sum(r['event'] == 'disconnected' for r in log.rows) < count:
                    await asyncio.sleep(.005)
            await asyncio.wait_for(finished(), 2)

        try:
            reader, writer = await connect(frame_message(90))
            self.assertEqual((await read(reader))['params']['mode'], 'FORCE_AI')
            await read(reader)
            bridge.world.ingest({'type': 'EVENT', 'event': 'PLAYER_DAMAGE', 'frame': 90})
            await disconnect(writer, 1)
            self.assertEqual(bridge.world.events, [])
            self.assertEqual(bridge.world.capabilities, {})

            partial = frame_message(93)
            del partial['agent']['control_mode']
            reader, writer = await connect(partial)
            self.assertEqual((await read(reader))['params']['mode'], 'FORCE_AI')
            command = await read(reader)
            self.assertEqual(command['move'], {'x': 0, 'y': 0})
            self.assertEqual(bridge.world.control_mode, 'MANUAL')
            writer.write((json.dumps(frame_message(96)) + '\n').encode())
            await writer.drain()
            command = await read(reader)
            self.assertEqual(bridge.world.control_mode, 'FORCE_AI')
            self.assertNotEqual(command['move'], {'x': 0, 'y': 0})
            await disconnect(writer, 2)

            no_protocol = frame_message(99)
            del no_protocol['agent']['protocol']
            reader, writer = await connect(no_protocol)
            self.assertEqual((await read(reader))['params']['mode'], 'MANUAL')
            self.assertEqual(await asyncio.wait_for(reader.readline(), 2), b'')
            self.assertTrue(any(r['event'] == 'bridge_error' for r in log.rows))
        finally:
            for writer in writers:
                writer.close()
            server.close()
            await server.wait_closed()
            await bridge.close()
