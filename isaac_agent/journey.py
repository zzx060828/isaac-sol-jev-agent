"""Observed destinations and persistent, revalidated room-to-room travel."""
from collections import deque
from copy import deepcopy
import hashlib
import json
from .pickups import routine, CHEST_ROOMS, has_golden_key
from .review import active as deferred, choice_id
from .pickup_progress import blocked_choices

FORBIDDEN = {10,14,15}  # Those entrances require their separate contracts.
KEY_DOORS = {2,4,12,24}
PICKUPS = {10,20,30,40,70,90,100,300,350}


def options(world):
    from .resource_access import needs_access, evidence
    if world.combat_enemies or not world.info.get('is_clear'):
        return []
    rooms={int(k):v for k,v in world.rooms.items() if str(k).isdigit()}
    rooms[world.room]={**rooms.get(world.room,{}),'doors':world.layout.get('doors',{}),'clear':True}
    todo=deque([(world.room,[],0)]);labels={world.room:[(0,0)]};result=[]
    golden = has_golden_key(world)
    while todo:
        source,path,cost=todo.popleft()
        if (len(path),cost) not in labels[source]:
            continue
        for slot,door in rooms.get(source,{}).get('doors',{}).items():
            target=door.get('target_room');kind=door.get('target_room_type')
            locked=bool(door.get('is_locked'))
            key_cost = int(locked and not golden)
            price=cost+key_cost
            if (type(target) is not int or not 0<=target<169 or kind in FORBIDDEN
                    or locked and kind not in KEY_DOORS
                    or not locked and not door.get('is_open') or price>world.inventory.get('keys',0)):
                continue
            hops=len(path)+1
            previous=labels.get(target,[])
            # A shorter route can spend more keys. Keep both tradeoffs, but
            # discard equal or worse labels (also prevents positive-hop loops).
            if any(h<=hops and k<=price for h,k in previous):
                continue
            labels[target]=[(h,k) for h,k in previous if not (hops<=h and price<=k)]+[(hops,price)]
            edge={'from':source,'slot':str(slot),'to':target,'type':kind,'locked':locked,
                  'key_cost':key_cost}
            route=path+[edge];node=rooms.get(target)
            ignored=(deferred(world,target) | blocked_choices(world,target)) if node else set()
            resources=[p for p in (node or {}).get('resources',[]) if choice_id(p) not in ignored
                       and (p.get('variant') in PICKUPS or p.get('variant')==50
                            and routine(p) and node.get('type') in CHEST_ROOMS)]
            purpose=('unvisited_frontier' if node is None else
                     'observed_reward' if any(choice_id(p) not in ignored for p in node.get('collectibles',[])) else
                     'observed_resources' if (any(not needs_access(world,node,p) for p in resources)
                         or any(m.get('variant')==2 for m in node.get('machines',[])) or node.get('marked_rocks')) else
                     'resource_access_review' if resources else
                     'observed_floor_exit' if node.get('floor_exit') else None)
            if purpose:
                result.append({'destination_room':target,'type':kind,'purpose':purpose,
                               'keys_required':price,'hops':len(route),'path':route})
                if resources:
                    facts=evidence(world,node)
                    if facts:
                        result[-1]['resource_access_summary']={
                            'checked_pickups':len(facts),
                            'needs_changed_approach':sum(needs_access(world,node,p) for p in resources),
                            'evidence_in_floor_map_room':target}
            # Unseen/uncleared intermediate rooms cannot establish a through route.
            if node and node.get('clear') is True:
                todo.append((target,route,price))
    result=[r for r in result if (r['hops'],r['keys_required']) in labels[r['destination_room']]]
    for route in result:
        # Bind the exact observed doors and their ordinary-key authorization.
        signature=json.dumps(route['path'],sort_keys=True,separators=(',',':')).encode()
        route['route_id']='route:'+hashlib.sha256(signature).hexdigest()[:20]
    return result


class Journey:
    def __init__(self):
        self.contract=None
        self.cursor=0
        self.scope=None
        self.end_reason=None

    def start(self,world,contract):
        self.contract=deepcopy(contract);self.cursor=0
        self.scope=(world.epoch,world.level)
        self.end_reason=None

    def step(self,world,log):
        if not self.contract:
            return None
        reason=None
        edges=self.contract['path']
        if self.scope != (world.epoch,world.level):reason='floor_changed'
        elif world.room==self.contract['destination_room']:reason='arrived'
        elif self.cursor>=len(edges):reason='route_ended'
        else:
            edge=edges[self.cursor]
            if world.room==edge['to']:
                self.cursor+=1
                edge=edges[self.cursor]
            if world.room!=edge['from']:reason='unexpected_room'
            elif world.combat_enemies or not world.strategy_ready():return None
            else:
                door=world.layout.get('doors',{}).get(edge['slot'])
                if (not door or door.get('target_room')!=edge['to'] or door.get('target_room_type')!=edge['type']
                        or bool(door.get('is_locked')) and not edge['locked']):reason='door_changed'
                elif self.remaining_cost(world,door,edges[self.cursor:]) > min(
                        world.inventory.get('keys',0),
                        sum(e.get('key_cost',int(e['locked'])) for e in edges[self.cursor:])):
                    reason='key_budget_changed'
                else:
                    from .control import traversable_door
                    if not traversable_door(world,door):reason='door_closed'
                    else:return edge['slot']
        log.write('journey_end',frame=world.frame,room=world.room,reason=reason,
                  destination=self.contract['destination_room'])
        self.contract=None
        self.end_reason=reason
        return None

    @staticmethod
    def remaining_cost(world, door, edges):
        if has_golden_key(world):
            return 0
        return int(bool(door.get('is_locked'))) + sum(e['locked'] for e in edges[1:])
