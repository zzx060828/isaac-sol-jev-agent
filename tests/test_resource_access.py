from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest

from isaac_agent.control import pickup_target
from isaac_agent.floor_map import floor_map
from isaac_agent.journey import options
from isaac_agent.memory import RunMemory
from isaac_agent.resource_access import evidence,needs_access
from isaac_agent.state import World


def recorded():
    f=json.loads((Path(__file__).parent/'fixtures/resource-room-9480.json').read_text())
    return World(frame=f['frame'],room=f['room'],payload=f['payload'],sampled=f['sampled'],
                 capabilities=f['capabilities'],rooms=f['rooms'],level='3026184494:1:1',updated=time.monotonic())


def remote(w):
    w.room=71
    w.payload['ROOM_INFO']={**w.info,'room_type':1,'is_clear':True,'enemy_count':0}
    w.payload['ROOM_LAYOUT']={'doors':{'1':dict(target_room=58,target_room_type=1,is_open=True,is_locked=False)},'grid':{}}
    return w


class ResourceAccessTests(unittest.TestCase):
    def test_recorded_sealed_loot_is_not_described_as_directly_collectible_after_return(self):
        w=recorded()
        self.assertIsNone(pickup_target(w,{},True))
        facts=evidence(w,w.rooms['58'])
        self.assertEqual(len(facts),3)
        self.assertTrue(all(e['status']=='no_local_route' for e in facts))
        self.assertTrue(all(2 in e['nearby_solid_types'] for e in facts))
        remote(w)
        route=next(r for r in options(w) if r['destination_room']==58)
        self.assertEqual(route['purpose'],'resource_access_review')
        self.assertEqual(route['resource_access_summary']['needs_changed_approach'],3)
        node=next(n for n in floor_map(w)['nodes'] if n['id']==58)
        self.assertEqual(len(node['resource_access_last_seen']),3)
        self.assertEqual(len(node['resources_last_seen']),3)  # Preserve future operating opportunity.

    def test_missing_sensors_are_not_recorded_as_failed_paths(self):
        w=recorded();w.sampled['PLAYER_STATS']=w.frame-7
        pickup_target(w,{},True)
        self.assertFalse(evidence(w,w.rooms['58']))

    def test_opening_terrain_replaces_old_failure_with_observed_route(self):
        w=recorded();pickup_target(w,{},True)
        # Controlled geometry change, no claim that these rocks were bombed.
        w.layout['grid']={k:g for k,g in w.layout['grid'].items() if g['type'] not in (2,3,14)}
        w.navigation_revision+=1;w.path_cache.clear();w.terrain_index=()
        self.assertIsNotNone(pickup_target(w,{},True))
        self.assertTrue(all(e['status']=='local_route_found' for e in evidence(w,w.rooms['58'])))
        remote(w)
        self.assertEqual(next(r for r in options(w) if r['destination_room']==58)['purpose'],'observed_resources')

    def test_changed_movement_or_build_reopens_evaluation_but_does_not_invent_access(self):
        w=recorded();pickup_target(w,{},True);remote(w)
        w.stats['can_fly']=True
        facts=evidence(w,w.rooms['58'])
        self.assertTrue(all(e['movement_context_changed'] for e in facts))
        self.assertTrue(all(e['status']=='no_local_route' for e in facts))
        self.assertFalse(any(needs_access(w,w.rooms['58'],p) for p in w.rooms['58']['resources']))

    def test_partial_entry_packets_keep_facts_but_missing_pickup_identity_does_not_match(self):
        w=recorded();pickup_target(w,{},True)
        agent={**w.capabilities,'level_id':w.level}
        w.ingest({'type':'DATA','frame':w.frame+1,'room_index':58,'agent':agent,'payload':{'BOMBS':[]}})
        self.assertEqual(len(evidence(w,w.rooms['58'])),3)
        for p in w.rooms['58']['resources']:p['id']+=100
        self.assertFalse(evidence(w,w.rooms['58']))

    def test_memory_persists_access_and_does_not_write_new_timestamps_each_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=recorded();pickup_target(w,{},True)
            facts=deepcopy(w.rooms['58']['resource_access'])
            w.frame+=1;w.sampled={k:w.frame for k in w.payload};pickup_target(w,{},True)
            self.assertEqual(w.rooms['58']['resource_access'],facts)
            memory=RunMemory(Path(tmp)/'memory.json');memory.observe(w,{})
            data=json.loads((Path(tmp)/'memory.json').read_text())
            self.assertEqual(data['geometry']['58']['resource_access'],facts)
