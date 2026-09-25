"""SocketBridge v3 snapshots, sensor freshness, and explored-room memory."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import time

ROOM_NAMES = {1: 'Normal', 2: 'Shop', 3: 'Error', 4: 'Treasure', 5: 'Boss',
              6: 'Miniboss', 7: 'Secret', 8: 'Super secret', 9: 'Arcade',
              10: 'Curse', 11: 'Challenge', 12: 'Library', 13: 'Sacrifice',
              14: 'Devil', 15: 'Angel', 16: 'Dungeon', 17: 'Boss rush',
              18: 'Isaacs', 19: 'Barren', 20: 'Chest', 21: 'Dice',
              22: 'Black market', 23: 'Greed exit', 24: 'Planetarium',
              25: 'Teleporter', 26: 'Teleporter exit', 27: 'Secret exit',
              28: 'Blue', 29: 'Ultra secret'}


def records(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values()) if value and all(str(k).isdigit() for k in value) else [value]
    return []


def first(value):
    return next(iter(records(value)), {})


def empty_pedestal(item):
    # PICKUP_COLLECTIBLE with COLLECTIBLE_NULL survives after collection.
    return item.get("variant") == 100 and item.get("sub_type") == 0


def xy(value):
    value = value or {}
    result = float(value.get("x", 0)), float(value.get("y", 0))
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Nonfinite coordinate")
    return result


@dataclass
class World:
    frame: int = -1
    room: int = -1
    epoch: int = 0
    level: str = ""
    payload: dict = field(default_factory=dict)
    sampled: dict = field(default_factory=dict)
    rooms: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    revision: int = 0
    updated: float = 0.0
    dead: bool = False
    control_mode: str = "MANUAL"
    capabilities: dict = field(default_factory=dict)
    last_damage_frame: int = -1000
    navigation_revision: int = 0
    path_cache: dict = field(default_factory=dict, repr=False)
    unreachable_paths: dict = field(default_factory=dict, repr=False)
    combat_breach: dict = field(default_factory=dict, repr=False)
    terrain_index: tuple = field(default_factory=tuple, repr=False)
    ground_seen: set = field(default_factory=set, repr=False)
    ground_history: list = field(default_factory=list, repr=False)

    run_memory: object = field(default=None, repr=False)
    review_cache: dict = field(default_factory=dict, repr=False)
    pickup_failures: dict = field(default_factory=dict, repr=False)

    @property
    def token(self):
        return self.epoch, self.level, self.room

    def reset(self):
        self.epoch += 1
        self.frame, self.room = -1, -1
        self.payload.clear()
        self.sampled.clear()
        self.rooms.clear()
        self.review_cache.clear()
        self.pickup_failures.clear()
        # Connection/run-local evidence must not survive a restart. Persisted
        # run memory has its own seed/continue checks and remains attached.
        self.events.clear()
        self.capabilities = {}
        self.control_mode = "MANUAL"
        self.level = ""
        self.dead = False
        self.updated = 0.0
        self.revision += 1
        self.last_damage_frame = -1000
        self.navigation_revision += 1
        self.path_cache.clear()
        self.unreachable_paths.clear()
        self.combat_breach.clear()
        self.terrain_index = ()
        self.ground_seen.clear()
        self.ground_history.clear()

    def ingest(self, msg, now=None):
        now = time.monotonic() if now is None else now
        kind = msg.get("type")
        if kind == "EVENT":
            event = msg.get("event", "")
            if event == "GAME_START":
                self.reset()
            if event in ("PLAYER_DEATH", "GAME_END"):
                self.dead = True
            if event == "PLAYER_DAMAGE":
                self.last_damage_frame = int(msg.get("frame", self.frame))
            # Room snapshots already have dedicated channels. Repeating their
            # entire grid in event history inflated each model request to ~8k tokens.
            data = msg.get("data", {})
            if isinstance(data, dict):
                data = {k: v for k, v in data.items() if k not in ("room_layout", "room_info")}
            self.events = (self.events + [{"event": event, "frame": msg.get("frame", self.frame), "data": data}])[-10:]
            self.revision += 1
            return False
        if kind not in ("DATA", "FULL"):
            return False
        frame = int(msg["frame"])
        # Game restart can reset frame before GAME_START reaches the socket.
        if frame < self.frame:
            self.reset()
        incoming = msg.get("payload", {})
        info = incoming.get("ROOM_INFO", {})
        room = int(msg.get("room_index", info.get("room_idx", self.room)))
        agent = msg.get("agent", {})
        level = str(agent.get("level_id", self.level))
        if level != self.level:
            self.rooms.clear()
        if room != self.room or level != self.level:
            self.payload.clear()
            self.sampled.clear()
            self.combat_breach.clear()
            self.ground_seen.clear()
            self.ground_history.clear()
            self.revision += 1
        self.frame, self.room, self.level = frame, room, level
        self.capabilities = agent
        self.control_mode = agent.get("control_mode", self.control_mode)
        for name, data in incoming.items():
            if name in ("ROOM_LAYOUT", "PICKUPS", "FIRE_HAZARDS", "INTERACTABLES") and data != self.payload.get(name):
                self.navigation_revision += 1
                self.path_cache.clear()
            self.payload[name] = deepcopy(data)
            self.sampled[name] = int(msg.get("sensors", {}).get(name, {}).get("collect_frame", frame))
            if name == 'FIRE_HAZARDS':
                # Legacy bridge reads Entity.State instead of ToNPC().State,
                # so the state is absent and ashes remain marked as burning.
                # Recorded Repentance normal/red fireplaces stop at 1/5 HP.
                # Restrict this compatibility fallback to that observed form;
                # explicit NPC state and other fireplace variants take priority.
                for fire in records(self.payload[name]):
                    if (agent.get('repentance_plus') is False and fire.get('type') == 'FIREPLACE'
                            and fire.get('variant') in (0, 1) and fire.get('state') is None
                            and fire.get('max_hp') == 5 and fire.get('hp') == 1):
                        fire['is_extinguished'] = True
                        fire['extinguished_source'] = 'legacy_rep_fireplace_hp_1_of_5'
                seen = set()
                for patch in records(data):
                    if not patch.get('ground_only'):
                        continue
                    key = (patch.get('id'), patch.get('variant'), xy(patch.get('pos')))
                    seen.add(key)
                    if key not in self.ground_seen:
                        self.ground_history.append({'frame': self.sampled[name], **patch})
                self.ground_seen = seen
                self.ground_history = [p for p in self.ground_history
                                       if frame-p['frame'] <= 12][-24:]
        self.updated = now
        if "ROOM_LAYOUT" in self.payload:
            from .secrets import wall_observations
            from .rocks import observed as marked_rocks
            access = self.rooms.get(str(room),{}).get('resource_access',{})
            self.rooms[str(room)] = {
                "visited": True, "clear": bool(self.info.get("is_clear")),
                "type": self.info.get("room_type"),
                "shape": self.info.get('room_shape'),
                "curses": self.info.get('curses'),
                "walls": wall_observations(self),
                "doors": deepcopy(self.layout.get("doors", {})),
                "collectibles": [{k: item.get(k) for k in ('id', 'variant', 'sub_type', 'price', 'options_index', 'is_hidden')}
                                 for item in self.pickups if item.get('variant') == 100],
                "resources": [{k:item.get(k) for k in ('id','variant','sub_type','price','options_index','is_hidden')}
                              for item in self.pickups if item.get('variant') != 100][:24],
                "resources_omitted": max(0,sum(item.get('variant') != 100 for item in self.pickups)-24),
                "resource_access": access,
                "marked_rocks": marked_rocks(self)[:12],
                "machines": [{'variant':m.get('variant')} for m in records(self.payload.get('INTERACTABLES'))][:12],
                "floor_exit": (self.info.get('room_type') == 5 and any(
                    g.get('type') == 17 and g.get('variant', 0) == 0
                    for g in self.layout.get('grid', {}).values())),
            }
        return True

    def fresh(self, channel, frames=6):
        return channel in self.payload and self.frame - self.sampled.get(channel, -1000) <= frames

    @property
    def player(self):
        return first(self.payload.get("PLAYER_POSITION"))

    @property
    def stats(self):
        return first(self.payload.get("PLAYER_STATS"))

    @property
    def health(self):
        return first(self.payload.get("PLAYER_HEALTH"))

    @property
    def inventory(self):
        data = first(self.payload.get("PLAYER_INVENTORY"))
        # Lua encodes empty maps as []; this occurs during active-item swaps.
        for name in ("active_items", "collectibles"):
            if not isinstance(data.get(name), dict):
                data[name] = {}
        if self.capabilities.get('repentance_plus') is False:
            # Old SocketBridge sampled the exclusive NUM_COLLECTIBLES sentinel
            # as an item. On this Repentance build it returns garbage memory.
            data['collectibles'].pop('733', None)
        return data

    def strategy_ready(self):
        return (all(self.fresh(name, 8) for name in
                    ("PLAYER_INVENTORY", "PICKUPS", "ROOM_LAYOUT", "ROOM_INFO"))
                and self.inventory.get("can_use", True) is not False
                and not any(i.get("variant") == 100 and i.get("wait", 0) > 0 for i in self.pickups))

    @property
    def boss_exits(self):
        if self.info.get('room_type') != 5 or not self.info.get('is_clear') or self.combat_enemies:
            return []
        return [cell for cell in self.layout.get('grid', {}).values()
                if cell.get('type') == 17 and cell.get('variant', 0) == 0
                and cell.get('state') == 1]

    def strategy_signature(self):
        from .pickups import routine
        return (tuple(sorted(self.inventory.get("collectibles", {}).items())),
                tuple(sorted((str(i.get("id")), i.get("variant"), i.get("sub_type"),
                              i.get("price", 0), i.get("options_index", 0), i.get("wait", 0) > 0) for i in self.pickups
                             if not routine(i) and not (i.get('variant') == 50 and i.get('sub_type') == 0
                                 and i.get('price',0) == 0 and not i.get('options_index',0)))),
                bool(self.combat_enemies),
                tuple(sorted((str(slot), door.get('target_room_type'), bool(door.get('is_open')),
                              bool(door.get('is_locked'))) for slot,door in self.layout.get('doors',{}).items()
                             if door.get('target_room_type') in (14,15))),
                (self.inventory.get('trinket_0', 0), self.inventory.get('trinket_1', 0)))

    @property
    def info(self):
        return self.payload.get("ROOM_INFO", {})

    @property
    def layout(self):
        return self.payload.get("ROOM_LAYOUT", {})

    @property
    def enemies(self):
        return records(self.payload.get("ENEMIES", []))

    @property
    def combat_enemies(self):
        # Room clear/count are engine authority. Some NPC entities remain in the
        # room after combat (including nonblocking or dormant entities). Preserve
        # them in enemies for collision safety, but do not fight them forever.
        if (self.fresh("ROOM_INFO", 8) and self.info.get("is_clear") is True
                and self.info.get("enemy_count") == 0):
            return []
        return self.enemies

    @property
    def projectiles(self):
        return records(self.payload.get("PROJECTILES", {}).get("enemy_projectiles", []))

    @property
    def pickups(self):
        return [item for item in records(self.payload.get("PICKUPS")) if not empty_pedestal(item)]

    @property
    def empty_pedestals(self):
        return [item for item in records(self.payload.get("PICKUPS")) if empty_pedestal(item)]

    def ready(self, now):
        return (not self.dead and (self.room >= 0 or self.info.get("room_type") in (14, 15)) and now - self.updated < 0.35
                and self.fresh("PLAYER_POSITION", 3)
                and self.fresh("ENEMIES", 6) and self.fresh("PROJECTILES", 6)
                and bool(self.layout) and bool(self.player))

    def observation(self):
        from .resources import resource_options
        from .strategy import offers, describe_pickup, ITEM_NAMES, ENTITY_NAMES
        from .knowledge import held_item_mechanics, held_trinket_mechanics, held_consumable_mechanics, encounter_knowledge, room_knowledge, decision_knowledge
        from .scavenging import economy
        from .control import traversable_door
        from .combat import context as combat_context
        from .knowledge import enemy_briefs, encounter_coverage
        from .charge import battery_target
        p = xy(self.player.get("pos"))
        def near(items, count):
            return sorted(items, key=lambda e: math.dist(xy(e.get("pos")), p))[:count]
        briefs = encounter_knowledge(near(self.combat_enemies, 16))
        return deepcopy({
            "frame": self.frame, "room": self.room, "level": self.level,
            "repentance_plus": self.capabilities.get('repentance_plus'),
            "strategy_signature": self.strategy_signature(),
            "player": self.player, "stats": self.stats, "health": self.health,
            "inventory": {**self.inventory, "item_names": {
                str(k): ITEM_NAMES.get(str(k), "Unknown collectible")
                for k in self.inventory.get("collectibles", {})}},
            "room_info": {**self.info, 'name': ROOM_NAMES.get(self.info.get('room_type'), 'Unknown')},
            "strategy_offers": offers(self),
            "run_history": self.run_memory.ledger() if self.run_memory else {"history_complete_from_run_start": False},
            "memory_candidates": self.run_memory.candidates(self) if self.run_memory else [],
            "held_item_mechanics": held_item_mechanics(self.inventory),
            "held_trinket_mechanics": held_trinket_mechanics(self.inventory),
            "held_consumable_mechanics": held_consumable_mechanics(self.inventory),
            "ordinary_battery_target": battery_target(self),
            "enemy_knowledge": briefs,
            "enemy_knowledge_coverage": encounter_coverage(self.combat_enemies, briefs),
            "combat_positioning": combat_context(self),
            "room_knowledge": room_knowledge(self),
            "decision_knowledge": decision_knowledge({'inventory': self.inventory, 'pickups': self.pickups,
                                                       'room_info': self.info}),
            "economy": economy(self),
            "interactables": near(records(self.payload.get("INTERACTABLES")), 16),
            "resource_options": resource_options(self),
            "resource_policy": "Unknown pills: only in clear safe rooms with >=6 half-heart units and escape space; never at low health.",
            "enemies": [{**enemy, 'name': enemy_briefs().get(f"{enemy.get('type')}:{enemy.get('variant', 0)}", {}).get(
                            'name', ENTITY_NAMES.get(str(enemy.get('type')), 'Unknown enemy'))}
                        for enemy in near(self.combat_enemies, 16)],
            "nonblocking_entities": near(self.enemies, 16) if not self.combat_enemies else [], "projectiles": near(self.projectiles, 32),
            "bombs": near(records(self.payload.get("BOMBS")), 16),
            "fires": near(records(self.payload.get("FIRE_HAZARDS")), 16),
            "lasers": self.payload.get("PROJECTILES", {}).get("lasers", []),
            "terrain": [cell for cell in self.layout.get("grid", {}).values()
                        if cell.get("type") not in (0, 1, 15, 16)],
            "last_damage_frame": self.last_damage_frame,
            "pickups": [describe_pickup(item) for item in near(self.pickups, 16)],
            "empty_pedestals": near(self.empty_pedestals, 16),
            "doors": {str(k): {**v, 'target_room_name': ROOM_NAMES.get(v.get('target_room_type'), 'Unknown'),
                               'traversable': traversable_door(self, v)}
                      for k, v in self.layout.get("doors", {}).items()},
            "explored_rooms": {k: {**v, 'name': ROOM_NAMES.get(v.get('type'), 'Unknown'),
                                   'collectibles': [describe_pickup(item) for item in (v.get('collectibles') or [])]}
                               for k, v in self.rooms.items()},
            "recent_events": self.events,
        })
