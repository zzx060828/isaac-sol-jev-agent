"""Observed marked rocks and bounded, model-authorized bomb contracts."""
import math

from .state import records, xy
from .secrets import ordinary_bombs, settled_for_bomb
from .bomb_budget import budget, payment_valid, golden_spawn_observed


def observed(world):
    return [{'grid_index':int(index), 'type':g['type'], 'pos':{'x':g['x'],'y':g['y']},
             'name':'Tinted Rock' if g['type']==4 else 'Super Special Rock',
             'placement_count_upper_bound':1 if g['type']==4 else 2,
             'reward':'Possible soul hearts, keys, bombs, chests or unlocked Small Rock; not guaranteed'}
            for index,g in world.layout.get('grid',{}).items()
            if g.get('type') in (4,22) and g.get('collision',0) in (2,3,4)]


def ready(world):
    return (world.capabilities.get('rock_bomb_protocol')==1
            and world.capabilities.get('repentance_plus') is False
            and world.run_memory is not None and world.info.get('is_clear') is True
            and world.info.get('room_type') in (1,2,4,5,6,7,8)
            and world.info.get('room_shape')==1 and not world.combat_enemies
            and world.info.get('stage_type',0)<4  # Mines/Ashpit rock-spider handling pending.
            and world.stats.get('speed',0)>=.75 and ordinary_bombs(world)
            and world.health.get('red_hearts',0)+world.health.get('soul_hearts',0)>=4
            and not records(world.payload.get('BOMBS')) and not world.projectiles
            and all(world.fresh(k,6) for k in ('ROOM_INFO','ROOM_LAYOUT','PLAYER_INVENTORY',
                'PLAYER_HEALTH','BOMBS','FIRE_HAZARDS','PROJECTILES','INTERACTABLES','PICKUPS')))


def placement(world, rock):
    from .control import blocked, path_step
    center=xy(rock['pos']);p=xy(world.player.get('pos'));routes=[]
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        spot=(center[0]+dx*45,center[1]+dy*45)
        escape=(spot[0]+dx*190,spot[1]+dy*190)
        # Do not bundle unpriced explosions, mushroom/skull spawns or machine
        # destruction into a rock contract. These need their own valuation.
        if any(g.get('type') in (5,6,12,21) and g.get('collision',0)
               and math.dist(spot,(g['x'],g['y']))<175 for g in world.layout.get('grid',{}).values()):
            continue
        if any(math.dist(spot,xy(e.get('pos')))<175
               for e in records(world.payload.get('INTERACTABLES'))):
            continue
        if any(blocked(world,(spot[0]+dx*190*i/20,spot[1]+dy*190*i/20),15)
               for i in range(21)):
            continue
        if math.dist(p,spot)>15 and math.dist(path_step(world,spot),p)<1:
            continue
        routes.append((spot,escape))
    return min(routes,key=lambda r:math.dist(p,r[0]),default=None)


def offers(world):
    if not ready(world):return []
    result=[]
    for rock in observed(world):
        marker=f'{world.level}:{world.room}:{rock["grid_index"]}'
        if marker in world.run_memory.rock_attempts:continue
        cost=rock['placement_count_upper_bound']
        funding=budget(world,cost)
        if funding is None:continue
        route=placement(world,rock)
        if not route:continue
        spot,escape=route
        suffix=':golden' if funding['bomb_payment']=='golden' else ''
        result.append({'choice':f'rock_bomb:{rock["grid_index"]}:{rock["type"]}'+suffix,
            'kind':'rock_bomb','entity':{'pos':dict(zip(('x','y'),spot))},
            'rock':rock,'attempt_key':marker,'escape':escape,**funding,
            'effect':'Place at most bomb_limit normal-fuse bombs, one at a time; retreat and verify destruction. '
                     'bomb_cost is ordinary stock consumed; golden payment costs zero stock but still has a placement limit. '
                     'Normal payment keeps one bomb. Compare soul-heart protection and alternative bomb uses. '
                     'Drops are uncertain and must be observed before collecting.'})
    from .resource_rocks import offers as resource_offers
    return result[:4]+resource_offers(world)


class RockBomb:
    def __init__(self,world,offer):
        self.offer,self.token=offer,world.token
        self.started=world.frame;self.fired=None;self.shots=0;self.done=None
        self.bombs_before=world.inventory['bombs'];self.spawned=False
        self.request_id=None
        self.pickups_before={e.get('id') for e in world.pickups}

    def record(self,world,status):
        entry={'status':status,'room':self.token[2],'grid_index':self.offer['rock']['grid_index'],
               'shots':self.shots,'frame':world.frame}
        world.run_memory.rock_attempts[self.offer['attempt_key']]=entry
        world.run_memory.add(world,'resource_rock_bomb' if self.offer.get('bomb_kind')=='resource_rock'
                             else 'marked_rock_bomb',entry)

    def update(self,world,plan,log):
        o=self.offer;p=xy(world.player.get('pos'));spot=xy(o['entity']['pos'])
        execution=dict(plan,navigation_contact=False)
        def stop(reason):
            self.done=reason
            execution.update(navigation_target=p,navigation_contact=False)
            if self.shots:self.record(world,reason)
            log.write('rock_bomb_result',frame=world.frame,choice=o['choice'],result=reason,
                      shots=self.shots,new_pickups_observed=[e for e in world.pickups
                          if e.get('id') not in self.pickups_before
                          and math.dist(xy(e.get('pos')),xy(o['rock']['pos']))<170],
                      drop_causality='nearby new pickups; not proof of source or acquisition')
            return execution
        if world.token!=self.token:return stop('room_changed')
        if world.frame-self.started>720:return stop('timeout')
        grid=world.layout.get('grid',{}).get(str(o['rock']['grid_index']),{})
        present=grid.get('type')==o['rock']['type'] and grid.get('collision',0) in (2,3,4)
        if self.fired is not None:
            execution['navigation_target']=tuple(o['escape'])
            self.spawned |= (golden_spawn_observed(world,self.fired,self.request_id,o['entity']['pos'])
                             if o.get('bomb_payment')=='golden' else
                             world.inventory.get('bombs',self.bombs_before)<self.bombs_before)
            elapsed=world.frame-self.fired
            # Keep retreating even if a drop or new enemy appears during the fuse.
            if records(world.payload.get('BOMBS')) or elapsed<120:
                return stop('bomb_clearance_unconfirmed') if elapsed>240 else execution
            channels=('ROOM_LAYOUT','BOMBS','PLAYER_INVENTORY')
            if o.get('bomb_kind')=='resource_rock':
                channels+=('PICKUPS','PLAYER_POSITION','PLAYER_STATS','FIRE_HAZARDS')
            if not all(world.fresh(k,2) and world.sampled[k]>self.fired for k in channels):
                return execution
            if not self.spawned:return stop('bomb_unconfirmed_no_retry')
            if not grid or grid.get('collision',0)==0:
                if o.get('bomb_kind')=='resource_rock':
                    # Destruction is not pickup. Ordinary collection resumes
                    # after this event; its inventory receipts remain separate.
                    from .control import blocked,path_step
                    from .resource_rocks import identity
                    def same_form(e,expected):
                        return all(e.get(k)==expected.get(k) for k in ('id','variant','sub_type','price','options_index'))
                    accessible=[identity(e) for e in world.pickups
                                if any(same_form(e,x) for x in o['expected_resources'])
                                and not blocked(world,xy(e['pos']),float(world.stats.get('size',12)))
                                and math.dist(path_step(world,xy(e['pos'])),p)>1]
                    log.write('resource_rock_access_observed',frame=world.frame,
                              choice=o['choice'],accessible_resources=accessible,
                              scope='Observed post-blast paths; no acquisition or net profit claimed.')
                return stop('rock_destroyed_observed')
            if not present:return stop('target_changed')
            if self.shots>=o.get('bomb_limit',o['bomb_cost']):return stop('rock_survived_budget_exhausted')
            # Only the explicit two-bomb contract may schedule a second pulse.
            self.fired=None;self.spawned=False;self.settle_started=None
        if not present:return stop('target_changed')
        if (not ready(world) or not payment_valid(world,o)
                or budget(world,o.get('bomb_limit',o['bomb_cost'])-self.shots) is None):
            return stop('preflight_invalid')
        if o.get('bomb_kind')=='resource_rock':
            if (world.capabilities.get('resource_rock_protocol')!=1 or not world.fresh('PLAYER_STATS',6)
                    or not world.fresh('PLAYER_POSITION',3)):return stop('preflight_invalid')
            from .resource_rocks import route_and_loot
            checked=route_and_loot(world,o['rock'],o['expected_resources'])
            route=checked[:2] if checked else None
        else:
            route=placement(world,o['rock'])
        if not route:return stop('escape_invalid')
        # Recheck a reachable safe side before each separately authorized pulse.
        spot,escape=route;o['entity']['pos']=dict(zip(('x','y'),spot));o['escape']=escape
        execution['navigation_target']=spot
        if settled_for_bomb(self,world,spot,execution):
            self.shots+=1;self.fired=world.frame;self.bombs_before=world.inventory['bombs']
            self.record(world,'attempt_pending')
            self.request_id=o['attempt_key']+f':{self.shots}'
            execution['rock_bomb_request']={'id':self.request_id,'payment':o.get('bomb_payment','normal'),
                'kind':o.get('bomb_kind','rock'),
                'grid_index':o['rock']['grid_index'],'grid_type':o['rock']['type'],
                'bombs_before':self.bombs_before,'spot':o['entity']['pos']}
            log.write('rock_bomb_request',frame=world.frame,choice=o['choice'],contract=o,shot=self.shots)
        return execution
