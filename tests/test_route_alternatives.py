from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.journey import Journey, options
from isaac_agent.models import validate_plan
from isaac_agent.runtime import Controller
from test_journey import recorded, enter
from test_planning import MemoryLog


def edge(target, locked=False, kind=1):
    return dict(target_room=target, target_room_type=kind, is_locked=locked, is_open=not locked,x=600,y=280)


def forked():
    """Synthetic directed sensor graph, not a claim about a generated floor."""
    w=recorded()
    w.inventory.update(keys=1,golden_key=False)
    w.rooms={
        '70':dict(type=1,clear=True,doors={'2':edge(71,True,2),'3':edge(83)}),
        '71':dict(type=2,clear=True,doors={'3':edge(84),'0':edge(70)}),
        '83':dict(type=1,clear=True,doors={'3':edge(96),'1':edge(70)}),
        '96':dict(type=1,clear=True,doors={'2':edge(84),'1':edge(83)}),
        '84':dict(type=1,clear=True,doors={'2':edge(85,True,4)},
                  resources=[dict(id=900,variant=20,sub_type=1,price=0)])}
    for node in w.rooms.values():
        for slot,door in node['doors'].items():
            door['x'],door['y']={'0':(40,280),'1':(320,40),'2':(600,280),'3':(320,520)}[slot]
    enter(w,70)
    return w


def observation(w):
    return {**w.observation(),'navigation_options':options(w)}


class RouteAlternativeTests(unittest.TestCase):
    def test_tradeoff_and_downstream_destination_are_both_retained(self):
        w=forked();routes=options(w)
        self.assertEqual({(r['hops'],r['keys_required']) for r in routes if r['destination_room']==84},
                         {(2,1),(3,0)})
        downstream=next(r for r in routes if r['destination_room']==85)
        self.assertEqual([e['to'] for e in downstream['path']],[83,96,84,85])
        self.assertEqual((downstream['hops'],downstream['keys_required']),(4,1))
        self.assertEqual(len(routes),3)
        self.assertEqual(len({r['route_id'] for r in routes}),3)
        self.assertEqual(routes,options(w))

    def test_equal_length_cheaper_route_replaces_first_discovered(self):
        w=forked();w.rooms['83']['doors']['3']=edge(84)
        routes=options(w)
        goal=[r for r in routes if r['destination_room']==84]
        self.assertEqual(len(goal),1)
        self.assertEqual((goal[0]['hops'],goal[0]['keys_required']),(2,0))
        self.assertEqual(goal[0]['path'][0]['slot'],'3')

    def test_gold_and_zero_inventory_do_not_expand_ordinary_key_permission(self):
        w=forked();w.inventory['keys']=0
        routes=options(w)
        self.assertEqual([(r['destination_room'],r['hops'],r['keys_required']) for r in routes],[(84,3,0)])
        w.inventory['golden_key']=True
        routes=options(w)
        self.assertEqual([(r['destination_room'],r['hops'],r['keys_required']) for r in routes],
                         [(84,2,0),(85,3,0)])
        j=Journey();j.start(w,routes[0]);w.inventory.update(golden_key=False,keys=5)
        self.assertIsNone(j.step(w,MemoryLog()))
        self.assertEqual(j.end_reason,'key_budget_changed')

    def test_unseen_uncleared_and_special_rooms_cannot_create_shortcuts(self):
        for change in ('unseen','uncleared','forbidden'):
            w=forked()
            if change=='unseen':del w.rooms['83']
            elif change=='uncleared':w.rooms['83']['clear']=False
            else:w.layout['doors']['3']['target_room_type']=15
            self.assertFalse(any(r['destination_room']==85 for r in options(w)))

    def test_validator_binds_exact_route_and_rejects_ambiguous_or_mismatched_ids(self):
        w=forked();obs=observation(w)
        base=dict(objective='Save keys',mode='explore',destination_room=84)
        routes=[r for r in obs['navigation_options'] if r['destination_room']==84]
        for route in routes:
            plan=validate_plan({**base,'route_id':route['route_id'],'door_slot':'bad'},obs)
            self.assertEqual(plan['route_contract'],route)
            self.assertEqual(plan['door_slot'],route['path'][0]['slot'])
        for extra in ({},{'route_id':'route:invented'},{'route_id':True},
                      {'route_id':routes[0]['route_id'],'destination_room':85},
                      {'route_id':routes[0]['route_id'],'destination_room':None},
                      {'route_id':routes[0]['route_id'],'resource_action':'pill'}):
            with self.assertRaises(ValueError):validate_plan({**base,**extra},obs)
        # Legacy destination-only replies still work when they name one route.
        self.assertEqual(validate_plan({**base,'destination_room':85},obs)['door_slot'],'3')
        w.inventory['golden_key']=True
        with self.assertRaises(ValueError):
            validate_plan({**base,'route_id':routes[0]['route_id']},observation(w))

    def test_selected_detour_executes_and_changed_last_lock_stops(self):
        w=forked();route=next(r for r in options(w) if r['destination_room']==85)
        j=Journey();j.start(w,route);log=MemoryLog()
        for room,slot in ((70,'3'),(83,'3'),(96,'2'),(84,'2')):
            enter(w,room)
            self.assertEqual(j.step(w,log),slot)
        w.inventory['keys']=0
        self.assertIsNone(j.step(w,log))
        self.assertEqual(j.end_reason,'key_budget_changed')
        w=forked();j.start(w,route)
        w.layout['doors']['3']=edge(71,True,2)
        self.assertIsNone(j.step(w,log))
        self.assertEqual(j.end_reason,'door_changed')


class RouteAlternativeControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_free_detour_persists_without_repeat_planning(self):
        w=forked();obs=observation(w)
        route=next(r for r in obs['navigation_options'] if r['destination_room']==84 and r['keys_required']==0)
        plan=validate_plan(dict(objective='Save key',mode='explore',destination_room=84,route_id=route['route_id']),obs)
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(return_value=(deepcopy(plan),{})))
        c=Controller(models,MemoryLog());c.reset_actions(w);c.last_decision=float('inf')
        try:
            await c._plan(w,obs,w.token,w.revision)
            self.assertEqual(c.journey.contract,route)
            for room,slot in ((70,'3'),(83,'3'),(96,'2')):
                enter(w,room);w.capabilities['secret_bomb_protocol']=0
                c.step(w)
                self.assertEqual(c.route,'journey')
                self.assertEqual(c.execution_plan['door_slot'],slot)
                self.assertFalse(c.execution_plan['hold_for_strategy'])
            self.assertEqual(models.plan.await_count,1)
            enter(w,84);c.step(w)
            self.assertIsNone(c.journey.contract)
        finally:await c.close()
