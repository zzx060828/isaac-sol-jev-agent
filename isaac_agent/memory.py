"""Bounded evidence memory inspired by Jev-Mem; raw bridge logs remain canonical."""
from copy import deepcopy
import json
from pathlib import Path


class RunMemory:
    def __init__(self, path=None):
        self.path = path
        self.seed = None
        self.nodes = []
        self.skipped = []
        self.doors = []
        self.entered = []
        self.paid = []
        self.deal_records = []
        self.deal_records_complete = True
        self.secret_attempts = {}
        self.rock_attempts = {}
        self.decisions = []
        self.pickup_reviews = {}
        self.geometry_level = None
        self.geometry = {}
        self.complete = False
        self.last_frame = -1
        self.last_token = None
        self.last_clear = None
        self.next_id = 0
        self.loaded = None
        self.reconnect_cutoff = None
        if path and path.is_file():
            try:
                self.loaded = json.loads(path.read_text())
                for node in self.loaded.get('nodes', []):
                    evidence = node.get('evidence', {})
                    if node.get('kind') == 'damage' and 'health_snapshot' in evidence:
                        evidence.pop('health_snapshot')
                        evidence['possibly_queued_before_connection'] = True
                        evidence['caution'] = 'Legacy imported event has unknown occurrence time; exclude from action attribution.' 
            except (OSError, ValueError):
                pass

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps({k: getattr(self, k) for k in
                ('seed','nodes','skipped','doors','entered','paid','deal_records','deal_records_complete','secret_attempts','rock_attempts','decisions','pickup_reviews','geometry_level','geometry','complete','last_frame','next_id')}))
            tmp.replace(self.path)

    def add(self, world, kind, evidence, entities=()):
        node = {'id': str(self.next_id), 'kind': kind, 'seed': self.seed, 'level': world.level,
                'room': world.room, 'frame': world.frame, 'entities': list(entities),
                'evidence': deepcopy(evidence), 'provenance': 'observed bridge/validated executor',
                'previous_id': self.nodes[-1]['id'] if self.nodes else None,
                'relations': 'previous_id is temporal order, not causation'}
        self.next_id += 1
        self.nodes = (self.nodes+[node])[-256:]
        from .deals import DEAL_EVENTS
        if kind in DEAL_EVENTS:
            self.deal_records.append(deepcopy(node))
            if len(self.deal_records) > 128:
                self.deal_records = self.deal_records[-128:]
                self.deal_records_complete = False
        self.last_frame = world.frame
        self.save()

    def observe(self, world, msg):
        if not world.level:
            return
        seed = world.level.split(':')[0]
        if self.seed != seed or (self.last_frame >= 0 and world.frame < self.last_frame):
            loaded, path = self.loaded, self.path
            self.__init__()  # A fresh run must never inherit another run's deal history.
            self.path, self.seed = path, seed
            if (loaded and loaded.get('seed') == seed and
                    (loaded.get('last_frame', 0) <= world.frame or world.capabilities.get('continued_run') is True)):
                for k in ('nodes','skipped','doors','entered','paid','deal_records','deal_records_complete','secret_attempts','rock_attempts','decisions','pickup_reviews','geometry_level','geometry','complete','last_frame','next_id'):
                    if k in loaded: setattr(self,k,loaded[k])
                if 'deal_records' not in loaded:
                    from .deals import DEAL_EVENTS
                    self.deal_records = [deepcopy(n) for n in self.nodes if n['kind'] in DEAL_EVENTS][-128:]
                    self.deal_records_complete = False  # Legacy rolling nodes may have lost earlier deals.
                elif 'deal_records_complete' not in loaded:
                    self.deal_records_complete = False
            else:
                self.complete = world.frame < 120 and world.level.split(':')[1:2] == ['1']
        self.last_frame = world.frame
        if self.geometry_level != world.level:
            self.geometry_level,self.geometry = world.level,{}
        for room, geometry in self.geometry.items():
            world.rooms.setdefault(room,deepcopy(geometry))
        current = world.rooms.get(str(world.room),{})
        if (current.get('visited') and world.fresh('ROOM_INFO',6) and world.fresh('ROOM_LAYOUT',6)):
            geometry = {k:deepcopy(current.get(k)) for k in ('visited','clear','type','shape','curses','doors','walls',
                'collectibles','resources','resources_omitted','resource_access','marked_rocks','machines','floor_exit')}
            if geometry != self.geometry.get(str(world.room)):
                self.geometry[str(world.room)] = geometry
                self.save()
        if self.reconnect_cutoff is None:
            self.reconnect_cutoff = world.frame+3
        token = world.token
        if token != self.last_token:
            self.last_token, self.last_clear = token, None
            self.add(world, 'room_entered', {'room_type': world.info.get('room_type')})
            if world.info.get('room_type') in (14,15):
                marker = f'{world.level}:{world.room}'
                if marker not in self.entered:
                    self.entered.append(marker)
                    self.add(world, 'deal_room_entered', {'room_type': world.info.get('room_type')})
        if self.last_clear is False and world.info.get('is_clear'):
            self.add(world, 'room_cleared', {'health': world.health})
        self.last_clear = world.info.get('is_clear')
        for slot, door in world.layout.get('doors', {}).items():
            if world.info.get('room_type') == 5 and door.get('target_room_type') in (14,15) and door.get('is_open'):
                marker = f'{world.level}:{world.room}:{slot}'
                if marker not in self.doors:
                    self.doors.append(marker)
                    self.add(world, 'deal_door_observed', {'key': marker, 'type': door['target_room_type']})
        if msg.get('event') == 'PLAYER_DAMAGE':
            self.add(world, 'damage', {'source': msg.get('data', {}),
                'possibly_queued_before_connection': world.frame <= self.reconnect_cutoff,
                'caution': 'Legacy bridge reports delivery frame, not necessarily occurrence frame. No current health/action is attributed to this event; hp_before belongs to the emitted event. Source alone is not causal explanation.'},
                [f"enemy:{msg.get('data',{}).get('source_type')}"])

    def ledger(self):
        from .deals import history
        return {'history_complete_from_run_start': self.complete, 'deal_doors_seen': len(self.doors),
                'deal_evidence': history(self),
                'recent_decisions': deepcopy(self.decisions[-6:]),
                'deal_rooms_entered': list(self.entered), 'paid_item_acquisitions': list(self.paid),
                'skipped_doors': list(self.skipped),
                'secret_search_attempts': deepcopy(self.secret_attempts),
                'caution': 'Unknown earlier history cannot imply first Devil offer. Paid acquisition is observed, Angel eligibility has item exceptions. No exact chance inferred.'}

    def record_decision(self, world, plan):
        if not plan.get('strategy_choice') and not plan.get('resource_action') and not plan.get('declined_pickups'):
            return
        self.decisions = (self.decisions+[{
            'level':world.level,'room':world.room,'frame':world.frame,
            'choice':plan.get('strategy_choice'),'resource_action':plan.get('resource_action'),
            'declined_pickups':plan.get('declined_pickups',[]),
            'objective':str(plan.get('objective',''))[:200],
            'rationale':str(plan.get('rationale',''))[:450],
            'active_items_before':deepcopy(world.inventory.get('active_items',{})),
            'status':'model_intent_not_confirmed',
            'provenance':'accepted Sol recommendation; rationale is not an observed game fact'}])[-6:]
        self.save()

    def record_outcome(self, world, choice, acquired):
        entry=next((d for d in reversed(self.decisions) if d['level']==world.level
                    and d['room']==world.room and d.get('choice')==choice
                    and d['status']=='model_intent_not_confirmed'),None)
        if entry is not None:
            entry.update(status='observed_item_acquisition' if acquired else 'ended_without_confirmed_acquisition',
                         outcome_frame=world.frame,
                         active_items_after=deepcopy(world.inventory.get('active_items',{})))
            self.save()

    def candidates(self, world):
        types = {f"enemy:{e.get('type')}" for e in world.combat_enemies}
        deal_context = world.info.get('room_type') in (5,14,15) or any(
            d.get('target_room_type') in (14,15) for d in world.layout.get('doors',{}).values())
        item_entities = {'item:'+str(i) for i in world.inventory.get('collectibles',{})}
        item_entities.update('item:'+str(p.get('sub_type')) for p in world.pickups if p.get('variant')==100)
        relevant = []
        for node in self.nodes:
            if node['evidence'].get('possibly_queued_before_connection'):
                continue
            if ((node['kind']=='damage' and types.intersection(node['entities']))
                    or (node['kind'].startswith('deal') and deal_context)
                    or (node['kind']=='paid_item_acquired' and (deal_context or item_entities.intersection(node['entities'])))):
                relevant.append(node)
        # Deal history remains mandatory in ledger even if JEV omits a node.
        return deepcopy(sorted(relevant, key=lambda n: (
            n['kind'] in ('damage','paid_item_acquired','deal_door_skipped'), int(n['id'])), reverse=True)[:8])
