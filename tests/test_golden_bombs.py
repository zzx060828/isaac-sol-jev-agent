from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent import rocks,secrets
from isaac_agent.bomb_budget import golden_spawn_observed
from isaac_agent.runtime import Controller
from test_marked_rocks import setup as rock_world
from test_secret_search import setup as secret_world,settled_update
from test_agent import MemoryLog
from test_safety_strategy import refresh


def enable(w):
    w.inventory.update(bombs=0,golden_bomb=True)
    w.capabilities['golden_bomb_protocol']=1
    refresh(w)
    return w


def observe_spawn(w,job):
    w.capabilities.update(last_bomb_request=job.request_id,last_bomb_frame=job.fired)
    w.payload['BOMBS']=[dict(id=900,variant=0,pos=dict(job.offer['entity']['pos']))]
    refresh(w,1)
    return job.update(w,{},MemoryLog())


class GoldenBombTests(unittest.TestCase):
    def test_free_cost_still_has_placement_limit_and_requires_protocol(self):
        for module,w,limit in ((secrets,secret_world(),1),(rocks,rock_world(),1),(rocks,rock_world(22),2)):
            enable(w);offer=module.offers(w)[0]
            self.assertEqual(offer['bomb_cost'],0)
            self.assertEqual(offer['bomb_limit'],limit)
            self.assertEqual(offer['bombs_reserved_after'],0)
            self.assertEqual(offer['bomb_payment'],'golden')
            self.assertTrue(offer['choice'].endswith(':golden'))
            if module is secrets:
                w.run_memory.secret_attempts={w.level+':1':{},w.level+':2':{}}
                self.assertFalse(module.offers(w))
                w.run_memory.secret_attempts={}
            w.capabilities.pop('golden_bomb_protocol')
            w.inventory['bombs']=5
            self.assertFalse(module.offers(w))

    def test_free_payment_still_checks_health_fuse_and_retreat(self):
        for change in ('health','fuse','tnt','stale'):
            w=enable(rock_world())
            if change=='health':w.health['red_hearts']=2
            elif change=='fuse':w.inventory['trinket_0']=133
            elif change=='tnt':w.layout['grid']['68']=dict(type=12,x=360,y=280,collision=2)
            else:w.sampled['PLAYER_INVENTORY']=w.frame-7
            self.assertFalse(rocks.offers(w))

    def test_loss_before_pulse_does_not_switch_to_paid_stock(self):
        for module,cls,w in ((rocks,rocks.RockBomb,rock_world()),(secrets,secrets.SecretSearch,secret_world())):
            enable(w);o=module.offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
            job=cls(w,o);job.update(w,{},MemoryLog())
            w.inventory.update(golden_bomb=False,bombs=5);refresh(w,4)
            result=job.update(w,{},MemoryLog())
            self.assertEqual(job.done,'preflight_invalid')
            self.assertNotIn('rock_bomb_request',result)
            self.assertNotIn('secret_bomb_request',result)

    def test_rock_accepts_ack_plus_spawn_without_stock_decrease(self):
        w=enable(rock_world());o=rocks.offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=rocks.RockBomb(w,o)
        request=settled_update(job,w,{},MemoryLog())['rock_bomb_request']
        self.assertEqual(request['payment'],'golden')
        retreat=observe_spawn(w,job)
        self.assertTrue(job.spawned)
        self.assertEqual(retreat['navigation_target'],tuple(o['escape']))
        self.assertEqual(w.inventory['bombs'],0)
        w.payload['BOMBS']=[];w.layout['grid']['67']['collision']=0;refresh(w,120)
        job.update(w,{},MemoryLog())
        self.assertEqual(job.done,'rock_destroyed_observed')
        self.assertEqual(job.shots,1)

    def test_super_rock_never_exceeds_two_free_pulses(self):
        w=enable(rock_world(22));o=rocks.offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=rocks.RockBomb(w,o);log=MemoryLog()
        first=settled_update(job,w,{},log)['rock_bomb_request']
        observe_spawn(w,job);w.payload['BOMBS']=[];refresh(w,120)
        second=settled_update(job,w,{},log)['rock_bomb_request']
        self.assertNotEqual(first['id'],second['id'])
        observe_spawn(w,job);w.payload['BOMBS']=[];refresh(w,120)
        result=job.update(w,{},log)
        self.assertNotIn('rock_bomb_request',result)
        self.assertEqual(job.shots,2)
        self.assertEqual(job.done,'rock_survived_budget_exhausted')
        self.assertEqual(w.inventory['bombs'],0)

    def test_no_spawn_evidence_never_retries_even_if_stock_falls(self):
        w=enable(rock_world());w.inventory['bombs']=5
        o=rocks.offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=rocks.RockBomb(w,o);settled_update(job,w,{},MemoryLog())
        w.inventory['bombs']=4;refresh(w,121)
        job.update(w,{},MemoryLog())
        self.assertEqual(job.done,'bomb_unconfirmed_no_retry')

    def test_matching_ack_time_location_and_normal_bomb_are_required(self):
        w=enable(rock_world());w.frame=110
        w.capabilities.update(last_bomb_request='request',last_bomb_frame=100)
        w.payload['BOMBS']=[dict(variant=0,pos=dict(x=320,y=280))]
        refresh(w)
        for change in ('id','time','sample','location','variant'):
            with self.subTest(change=change):
                w.capabilities.update(last_bomb_request='request',last_bomb_frame=100)
                w.sampled['BOMBS']=110
                w.payload['BOMBS'][0].update(variant=0,pos=dict(x=320,y=280))
                if change=='id':w.capabilities['last_bomb_request']='other'
                elif change=='time':w.capabilities['last_bomb_frame']=99
                elif change=='sample':w.sampled['BOMBS']=100
                elif change=='location':w.payload['BOMBS'][0]['pos']['x']=500
                else:w.payload['BOMBS'][0]['variant']=3
                self.assertFalse(golden_spawn_observed(w,100,'request',dict(x=320,y=280)))

    def test_secret_entry_needs_explosion_wait_and_fresh_clearance(self):
        w=enable(secret_world());o=secrets.offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        job=secrets.SecretSearch(w,o);settled_update(job,w,{},MemoryLog());observe_spawn(w,job)
        w.payload['BOMBS']=[]
        w.layout['doors'][o['slot']]=dict(x=600,y=280,is_open=True,target_room_type=7)
        refresh(w,120);w.sampled['BOMBS']=job.fired
        self.assertFalse(job.update(w,{},MemoryLog())['navigation_contact'])
        refresh(w,1)
        self.assertTrue(job.update(w,{},MemoryLog())['navigation_contact'])
        self.assertEqual(w.inventory['bombs'],0)


class GoldenBombControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_wire_command_binds_free_payment_and_releases_next_update(self):
        w=enable(rock_world());o=rocks.offers(w)[0];w.player['pos']=dict(o['entity']['pos'])
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock())
        c=Controller(m,MemoryLog());c.reset_actions(w)
        c.plan.update(mode='collect',strategy_choice=o['choice'])
        try:
            self.assertFalse(c.step(w)['use_bomb'])
            refresh(w,4);cmd=c.step(w)
            self.assertTrue(cmd['use_bomb'])
            self.assertEqual(cmd['agent_bomb_payment'],'golden')
            self.assertEqual(cmd['agent_bombs_before'],0)
            self.assertEqual(cmd['move'],dict(x=0,y=0))
            refresh(w,1);self.assertFalse(c.step(w)['use_bomb'])
            m.plan.assert_not_called();m.choose_goal.assert_not_called()
        finally:await c.close()
