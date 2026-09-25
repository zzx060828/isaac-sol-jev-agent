import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.state import World
from isaac_agent.control import Action, pickup_target
from isaac_agent.fast_policy import lane
from isaac_agent.runtime import Controller
from test_agent import MemoryLog


def snapshot():
    w=World();w.ingest(json.loads((Path(__file__).parent/'fixtures/boss-loose-coins.json').read_text()))
    return w


class RoutineResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_free_collectible_waits_for_other_options_to_disappear(self):
        from isaac_agent.strategy import offers
        from test_safety_strategy import refresh
        w=snapshot();w.layout['doors']={};w.info['room_type']=1
        # Isolate disappearance settling from a separate held-consumable review.
        w.inventory.update(card_0=0,pill_0=0)
        w.payload['PICKUPS']=[dict(id=51,variant=100,sub_type=103,price=0,pos=dict(x=320,y=280)),
                              dict(id=52,variant=100,sub_type=308,price=0,pos=dict(x=400,y=280))]
        refresh(w)
        choice=next(o['choice'] for o in offers(w) if o['entity'].get('id')==51)
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),
            decide=AsyncMock(return_value=(Action(),{'confidence':0})))
        c=Controller(models,MemoryLog());c.reset_actions(w)
        c.plan.update(strategy_choice=choice,mode='collect')
        try:
            c.step(w)
            w.payload['PICKUPS']=w.payload['PICKUPS'][1:]
            w.inventory.setdefault('collectibles',{})['103']=1;refresh(w,1)
            c.step(w);await asyncio.sleep(0)
            self.assertEqual(c.route,'settle_transaction')
            models.plan.assert_not_called()
            refresh(w,5);w.payload['PICKUPS']=[]
            c.step(w);await asyncio.sleep(0)
            models.plan.assert_not_called()
            refresh(w,1);c.step(w);await asyncio.sleep(0)
            self.assertNotEqual(c.route,'settle_transaction')
            self.assertNotEqual(c.route,'sol')
            models.plan.assert_not_called()
        finally:await c.close()

    async def test_collects_multiple_coins_before_one_exit_decision(self):
        w=snapshot();w.layout['doors']={}
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),
            decide=AsyncMock(return_value=(Action(),{'confidence':0})))
        c=Controller(models,MemoryLog())
        try:
            for _ in range(2):
                c.step(w);await asyncio.sleep(0)
                self.assertEqual(c.route,'collect_resources')
                self.assertIsNotNone(c.execution_plan['navigation_target'])
                models.plan.assert_not_called();models.choose_goal.assert_not_called()
                coins=[p for p in w.pickups if p['variant']==20]
                chosen=min(coins,key=lambda p:abs(p['pos']['x']-c.execution_plan['navigation_target'][0])+abs(p['pos']['y']-c.execution_plan['navigation_target'][1]))
                w.payload['PICKUPS'].remove(chosen)
                w.inventory['coins']+=1
            self.assertIsNone(pickup_target(w,{},basic_only=True))  # Full red hearts ignored.
            self.assertEqual(lane(w,w.frame)[0],'sol')  # Now choose descent once.
        finally:
            await c.close()

    async def test_pickup_animation_settles_before_slow_plan(self):
        w=snapshot();w.layout['doors']={}
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),
            decide=AsyncMock(return_value=(Action(),{'confidence':0})))
        c=Controller(models,MemoryLog())
        try:
            c.step(w)
            # Coins have registered; their pickup entities disappear on a later update.
            w.payload['PICKUPS']=[p for p in w.pickups if p['variant']!=20]
            w.frame+=1
            w.sampled.update({k:w.frame for k in w.sampled})
            c.step(w);await asyncio.sleep(0)
            self.assertEqual(c.route,'settle_resources')
            models.plan.assert_not_called()
            w.frame+=18
            w.sampled.update({k:w.frame for k in w.sampled})
            c.step(w)
            self.assertEqual(c.route,'sol')
        finally:
            await c.close()

    def test_paid_options_and_unresolved_deal_do_not_become_routine(self):
        w=snapshot()
        self.assertEqual(lane(w,w.frame)[0],'sol')  # Deal door needs a choice.
        w.layout['doors']={}
        for coin in w.payload['PICKUPS']:
            if coin['variant']==20: coin['options_index']=1
        self.assertIsNone(pickup_target(w,{},basic_only=True))
        self.assertEqual(lane(w,w.frame)[0],'sol')
        for coin in w.payload['PICKUPS']:
            if coin['variant']==20: coin.update(options_index=0,price=1)
        self.assertIsNone(pickup_target(w,{},basic_only=True))
        self.assertEqual(lane(w,w.frame)[0],'sol')

    async def test_paid_heart_waits_for_post_purchase_inventory(self):
        from isaac_agent.strategy import offers
        from test_safety_strategy import refresh
        w=snapshot();w.layout['doors']={};w.info['room_type']=2
        w.health.update(red_hearts=6,max_hearts=10)
        w.inventory.update(coins=18,can_use=False)
        w.payload['PICKUPS']=[
            dict(id=51,variant=10,sub_type=1,price=3,pos=dict(x=320,y=280)),
            dict(id=52,variant=100,sub_type=439,price=15,pos=dict(x=400,y=280))]
        refresh(w)
        choice=next(o['choice'] for o in offers(w) if o['entity'].get('id')==51)
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),
            decide=AsyncMock(return_value=(Action(),{'confidence':0})))
        c=Controller(models,MemoryLog());c.reset_actions(w)
        c.plan.update(strategy_choice=choice,mode='collect')
        try:
            c.step(w)
            w.payload['PICKUPS']=w.payload['PICKUPS'][1:]
            w.health['red_hearts']=8;refresh(w,1)
            c.step(w);await asyncio.sleep(0)
            self.assertEqual(c.route,'settle_transaction')
            ended=w.frame
            refresh(w,7);w.sampled['PLAYER_INVENTORY']=ended
            c.step(w);await asyncio.sleep(0)
            self.assertEqual(c.route,'settle_transaction')
            models.plan.assert_not_called()
            w.inventory['coins']=15;refresh(w,1)
            c.step(w)
            self.assertEqual(c.route,'sol')
        finally:await c.close()
