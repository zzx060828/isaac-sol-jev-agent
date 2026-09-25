from copy import deepcopy
import unittest
from isaac_agent.memory import RunMemory
from isaac_agent.rocks import RockBomb
from isaac_agent.resource_rocks import offers
from test_resource_access import recorded
from test_agent import MemoryLog
from test_safety_strategy import refresh
from test_secret_search import settled_update


def setup():
    w=recorded();w.run_memory=RunMemory();w.capabilities['resource_rock_protocol']=1
    return w


class ResourceRockTests(unittest.TestCase):
    def test_recorded_opening_lists_only_loot_reachable_after_single_rock_removal(self):
        w=setup();o=offers(w)[0]
        self.assertEqual(o['bomb_kind'],'resource_rock')
        self.assertEqual(o['bomb_limit'],1);self.assertEqual(o['bomb_cost'],1)
        self.assertEqual(o['bombs_reserved_after'],1)
        self.assertEqual({p['id'] for p in o['expected_resources']},{4,5})
        # Double bomb remains enclosed; don't count it as predicted income.
        self.assertNotIn(6,[p['id'] for p in o['expected_resources']])
        self.assertEqual(w.layout['grid'][str(o['rock']['grid_index'])]['collision'],3)

    def test_old_protocol_budget_fuse_and_paid_or_exclusive_loot_do_not_authorize(self):
        for change in ('protocol','one_bomb','fast_fuse','paid','exclusive','stale_stats','near_machine'):
            w=setup()
            if change=='protocol':w.capabilities.pop('resource_rock_protocol')
            if change=='one_bomb':w.inventory['bombs']=1
            if change=='fast_fuse':w.inventory['trinket_0']=133
            if change=='paid':
                for p in w.pickups:p['price']=5
            if change=='exclusive':
                for p in w.pickups:p['options_index']=1
            if change=='stale_stats':w.sampled['PLAYER_STATS']=w.frame-7
            if change=='near_machine':w.payload['INTERACTABLES']=[dict(variant=2,pos=dict(x=160,y=200))]
            self.assertFalse(offers(w),change)

    def test_rechecks_target_before_spending_and_does_not_inherit_new_pickup(self):
        w=setup();o=offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=RockBomb(w,o)
        for p in w.pickups:p['id']+=100
        result=job.update(w,{},MemoryLog())
        self.assertNotIn('rock_bomb_request',result)
        self.assertEqual(job.shots,0)

    def test_one_typed_pulse_retreat_then_fresh_access_observation_not_profit(self):
        w=setup();o=offers(w)[0];w.player['pos']=dict(o['entity']['pos']);w.player['vel']=dict(x=0,y=0)
        log=MemoryLog();job=RockBomb(w,o)
        first=settled_update(job,w,{},log)['rock_bomb_request']
        self.assertEqual(first['kind'],'resource_rock')
        self.assertEqual(first['grid_type'],2)
        w.inventory['bombs']-=1;refresh(w,1)
        self.assertEqual(job.update(w,{},log)['navigation_target'],tuple(o['escape']))
        w.layout['grid'][str(o['rock']['grid_index'])]['collision']=0
        w.player['pos']=dict(zip(('x','y'),o['escape']));refresh(w,120)
        w.sampled['PICKUPS']=job.fired
        self.assertNotIn('rock_bomb_request',job.update(w,{},log));self.assertIsNone(job.done)
        refresh(w);job.update(w,{},log)
        self.assertEqual(job.done,'rock_destroyed_observed');self.assertEqual(job.shots,1)
        result=next(r for r in log.rows if r['event']=='resource_rock_access_observed')
        self.assertEqual({p['id'] for p in result['accessible_resources']},{4,5})
        self.assertNotIn('profit',result)
        self.assertFalse(offers(w))

    def test_surviving_rock_or_unconfirmed_bomb_never_gets_a_second_pulse(self):
        for confirmed in (False,True):
            w=setup();o=offers(w)[0];w.player['pos']=dict(o['entity']['pos']);w.player['vel']=dict(x=0,y=0)
            job=RockBomb(w,o);log=MemoryLog();settled_update(job,w,{},log)
            if confirmed:w.inventory['bombs']-=1
            refresh(w,121)
            self.assertNotIn('rock_bomb_request',job.update(w,{},log))
            self.assertEqual(job.shots,1)
            self.assertEqual(job.done,'rock_survived_budget_exhausted' if confirmed else 'bomb_unconfirmed_no_retry')

    def test_cache_revalidates_budget_and_geometry_without_mutating_source_offer(self):
        w=setup();o=offers(w)[0];original=deepcopy(o)
        o['expected_resources'].clear()
        self.assertEqual(offers(w)[0]['expected_resources'],original['expected_resources'])
        w.inventory['bombs']=1;self.assertFalse(offers(w))
        w.inventory['bombs']=2
        for g in w.layout['grid'].values():
            if g['type'] in (2,3,14):g['collision']=0
        w.navigation_revision+=1;w.path_cache.clear();w.terrain_index=()
        self.assertFalse(offers(w))
