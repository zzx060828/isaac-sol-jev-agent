import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.demo import frame_message
from isaac_agent.journey import Journey, options
from isaac_agent.models import validate_plan
from isaac_agent.runtime import Controller
from isaac_agent.state import World
from isaac_agent.planning import enrich_planning
from test_planning import MemoryLog
from test_safety_strategy import refresh


def recorded():
    o=json.loads((Path(__file__).parent/'fixtures/route-backtrack-room58.json').read_text())
    m=frame_message(o['frame'],o['room']);m['agent']['level_id']=o['level']
    for channel,key in (('PLAYER_POSITION','player'),('PLAYER_STATS','stats'),('PLAYER_HEALTH','health'),('PLAYER_INVENTORY','inventory')):
        m['payload'][channel]=[o[key]]
    m['payload']['ROOM_INFO']=o['room_info'];m['payload']['ROOM_LAYOUT']['doors']=o['doors']
    m['payload']['PICKUPS']=o['pickups']
    w=World();w.ingest(m);w.rooms=deepcopy(o['explored_rooms'])
    w.payload.update(BOMBS=[],FIRE_HAZARDS=[],INTERACTABLES=[]);refresh(w)
    return w


def enter(w,room):
    w.room=room;w.info.update(room_idx=room,is_clear=True,room_type=w.rooms.get(str(room),{}).get('type',1))
    w.layout['doors']=deepcopy(w.rooms.get(str(room),{}).get('doors',{}))
    w.payload['PICKUPS']=[];refresh(w,10)


class JourneyTests(unittest.IsolatedAsyncioTestCase):
    def test_recorded_route_goes_south_to_observed_frontier_not_north_to_shop(self):
        w=recorded();routes=options(w)
        goal=next(o for o in routes if o['destination_room']==85)
        self.assertEqual([e['to'] for e in goal['path']],[71,84,85])
        self.assertEqual(goal['path'][0]['slot'],'7')
        self.assertEqual(goal['keys_required'],0)
        self.assertFalse(any(o['type']==4 for o in routes)) # no observed treasure room
        obs=enrich_planning(w,w.observation())
        p=validate_plan({'objective':'Explore frontier','mode':'explore','destination_room':85,'door_slot':'5'},obs)
        self.assertEqual(p['door_slot'],'7') # derive next hop, do not trust direction text
        # Room45 now contains a supported ordinary chest; it is a valid resource destination.
        self.assertTrue(any(o['destination_room']==45 for o in routes))
        for goal in (999,True):
            with self.assertRaises(ValueError):
                validate_plan({'objective':'Go','mode':'explore','destination_room':goal},obs)
        with self.assertRaises(ValueError):
            validate_plan({'objective':'Go','mode':'explore','door_slot':'5'},obs)

    def test_cost_and_changed_door_cannot_authorize_unplanned_spending(self):
        w=recorded();goal=next(o for o in options(w) if o['destination_room']==32)
        self.assertEqual(goal['keys_required'],1)
        j=Journey();j.start(w,goal);log=MemoryLog()
        enter(w,45);self.assertEqual(j.step(w,log),'1')
        w.inventory['keys']=0;self.assertIsNone(j.step(w,log));self.assertIsNone(j.contract)
        w=recorded();j.start(w,next(o for o in options(w) if o['destination_room']==85))
        w.layout['doors']['7']['is_locked']=True
        self.assertIsNone(j.step(w,log));self.assertEqual(log.rows[-1]['reason'],'door_changed')
        w=recorded();w.inventory['keys']=0
        self.assertFalse(any(o['destination_room']==32 for o in options(w)))
        self.assertFalse(any(o['destination_room']==97 for o in options(w))) # curse entrance

    async def test_destination_survives_clear_transit_rooms_without_more_sol(self):
        w=recorded();w.payload['PICKUPS']=[]
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock())
        c=Controller(m,MemoryLog());c.last_decision=float('inf');c.reset_actions(w)
        contract=next(o for o in options(w) if o['destination_room']==85)
        c.journey.start(w,contract)
        # A received Sol route has already considered the same held supplies.
        from isaac_agent.resources import consumable_review_key, resource_options
        c.reviewed_consumables=consumable_review_key(w.level,w.inventory,resource_options(w))
        try:
            for room,slot in ((58,'7'),(71,'3'),(84,'2')):
                enter(w,room)
                # Avoid testing wall bombing here; an offer must interrupt below.
                w.capabilities['secret_bomb_protocol']=0
                c.step(w);await asyncio.sleep(0)
                self.assertEqual(c.route,'journey');self.assertEqual(c.execution_plan['door_slot'],slot)
                self.assertFalse(c.execution_plan['hold_for_strategy'])
            m.plan.assert_not_called()
            enter(w,85);c.step(w);self.assertIsNone(c.journey.contract)
        finally:await c.close()

    async def test_new_item_interrupts_route_and_floor_or_teleport_ends_it(self):
        w=recorded();w.payload['PICKUPS']=[]
        m=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=({'objective':'Review','mode':'collect'},{})))
        c=Controller(m,MemoryLog());c.reset_actions(w);c.last_decision=float('inf')
        c.journey.start(w,next(o for o in options(w) if o['destination_room']==85))
        try:
            w.payload['PICKUPS']=[dict(id=44,variant=100,sub_type=531,price=0,pos={'x':400,'y':280})]
            c.step(w);self.assertEqual(c.route,'sol');await c.plan_task
            self.assertTrue(c.execution_plan['hold_for_strategy'])
            w.level+='new';c.journey.step(w,c.log);self.assertIsNone(c.journey.contract)
            c.journey.start(w,{'destination_room':85,'path':[{'from':58,'to':71,'slot':'7','type':1,'locked':False}]})
            w.room=82;c.journey.step(w,c.log);self.assertIsNone(c.journey.contract)
        finally:await c.close()
