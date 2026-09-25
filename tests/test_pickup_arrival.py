"""Entry animation must not make a planned reward room look empty."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.fast_policy import lane
from isaac_agent.review import choice_id
from isaac_agent.runtime import Controller
from isaac_agent.strategy import offers
from test_review_continuity import setup
from test_safety_strategy import refresh
from test_planning import MemoryLog


def arrival(subtype=378):
    w=setup();w.info.update(room_type=4,curses=0,stage_type=0)
    w.stats['player_type']=1
    w.payload['PICKUPS']=[dict(id=1,variant=100,sub_type=subtype,item_type=1,
        price=0,options_index=0,wait=18,pos=dict(x=320,y=280))]
    w.player['pos']=dict(x=100,y=280)
    return w


class PickupArrivalTests(unittest.IsolatedAsyncioTestCase):
    async def test_hold_then_review_without_touch_or_early_model_call(self):
        w=arrival();models=SimpleNamespace(mode='hybrid',plan=AsyncMock(
            return_value=(dict(objective='Review No. 2',mode='explore'),{})))
        c=Controller(models,MemoryLog())
        try:
            command=c.step(w);await asyncio.sleep(0)
            self.assertEqual(c.route,'settle_pickups')
            self.assertEqual(command['move'],{'x':0,'y':0})
            self.assertFalse(models.plan.called)
            self.assertFalse(offers(w))
            refresh(w,6);w.payload['PICKUPS'][0]['wait']=9
            c.step(w);self.assertEqual(c.route,'settle_pickups')
            refresh(w,6);w.payload['PICKUPS'][0]['wait']=0
            c.step(w);await c.plan_task
            self.assertEqual(c.route,'sol');self.assertEqual(models.plan.await_count,1)
            self.assertEqual(models.plan.call_args.args[0]['strategy_offers'][0]['entity']['sub_type'],378)
        finally:await c.close()

    def test_simple_item_uses_fast_lane_after_native_wait(self):
        w=arrival(30);start=w.frame
        self.assertEqual(lane(w,start)[0],'settle_pickups')
        w.payload['PICKUPS'][0]['wait']=0
        self.assertEqual(lane(w,start)[0],'fast')

    async def test_first_room_packet_can_precede_inventory_packet(self):
        w=arrival();w.payload.pop('PLAYER_INVENTORY');w.sampled.pop('PLAYER_INVENTORY')
        c=Controller(SimpleNamespace(mode='hybrid'),MemoryLog())
        try:
            command=c.step(w)
            self.assertEqual(c.route,'settle_pickups')
            self.assertEqual(command['move'],{'x':0,'y':0})
            self.assertIsNone(c.plan_task)
        finally:await c.close()

    def test_grace_is_bounded_and_does_not_override_combat_or_declines(self):
        w=arrival();start=w.frame;choice=choice_id(w.pickups[0])
        self.assertEqual(lane(w,start,finished={choice})[0],'routine')
        refresh(w,60);self.assertEqual(lane(w,start)[0],'routine')
        w=arrival();w.payload['ENEMIES']=[dict(id=9,type=18,hp=10,pos=dict(x=450,y=280))]
        self.assertEqual(lane(w,w.frame)[0],'tactical')
        w=arrival();w.payload['PICKUPS'][0]['price']=99
        self.assertEqual(lane(w,w.frame)[0],'routine')
        for variant,subtype in ((40,3),(20,6),(20,1)):
            w=arrival();w.payload['PICKUPS'][0].update(variant=variant,sub_type=subtype)
            self.assertNotEqual(lane(w,w.frame)[0],'settle_pickups')
