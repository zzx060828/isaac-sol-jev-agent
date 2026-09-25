"""Synthetic smoke test data; not an Isaac emulator or gameplay benchmark."""
import asyncio
import json

from .models import Models, configuration
from .runtime import Controller
from .state import World


def frame_message(frame=1, room=1, player=(320, 280), enemies=None, projectiles=None):
    return {"type": "DATA", "version": "3.0", "frame": frame, "room_index": room,
            "agent": {"protocol": 1, "level_id": "demo:1:0", "control_mode": "FORCE_AI"},
            "payload": {
                "PLAYER_POSITION": [{"pos": {"x": player[0], "y": player[1]}, "vel": {"x": 0, "y": 0}}],
                "PLAYER_STATS": [{"speed": 1, "size": 12, "can_fly": False}],
                "PLAYER_HEALTH": [{"red_hearts": 6, "max_hearts": 6}],
                "PLAYER_INVENTORY": [{"keys": 1, "bombs": 1}],
                "PICKUPS": [],
                "ENEMIES": enemies or [], "PROJECTILES": {"enemy_projectiles": projectiles or [], "lasers": []},
                "ROOM_INFO": {"room_type": 1, "room_idx": room, "is_clear": not enemies,
                              "top_left": {"x": 40, "y": 120}, "bottom_right": {"x": 600, "y": 440}},
                "ROOM_LAYOUT": {"grid": {}, "doors": {"2": {"x": 600, "y": 280,
                    "target_room": room + 1, "target_room_type": 1, "is_open": not enemies, "is_locked": False}}},
            }}


async def demo(log, frames=150):
    models = Models(configuration(), "local")
    controller, world = Controller(models, log), World()
    player = (320.0, 280.0)
    for frame in range(1, frames + 1):
        enemy = [{"id": 9, "pos": {"x": 470, "y": 220}, "vel": {"x": 0, "y": 0}, "collision_radius": 15}]
        projectiles = [{"id": 8, "pos": {"x": 500 - frame % 90 * 5, "y": 280},
                        "vel": {"x": -5, "y": 0}, "collision_radius": 8}]
        msg = frame_message(frame, player=player, enemies=enemy if frame < frames // 2 else [],
                            projectiles=projectiles if frame < frames // 2 else [])
        log.write("bridge", message=msg)
        world.ingest(msg)
        command = controller.step(world)
        m = command["move"]
        scale = 4 / ((m["x"] ** 2 + m["y"] ** 2) ** 0.5 or 1)
        player = player[0] + m["x"] * scale, player[1] + m["y"] * scale
        await asyncio.sleep(0)
    await controller.close()
    print(f"Synthetic demo: {frames} frames, final position={player}; no live model calls or game result")


async def replay(source, log):
    world = World()
    controller = Controller(Models(configuration(), "local"), log)
    count = 0
    with open(source, encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            msg = entry.get("message") if entry.get("event") == "bridge" else entry if "type" in entry else None
            if not msg:
                continue
            if world.ingest(msg):
                controller.step(world)
                count += 1
                await asyncio.sleep(0)
    await controller.close()
    print(f"Replayed {count} recorded frames with the local policy; no game connection")
