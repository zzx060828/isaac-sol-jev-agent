"""Short-lived combat intent, resolved against current geometry every update."""
import math
from .state import xy

POSTURES = {
    'range': 'Hold an effective firing lane at observed tear range; preserve turning room.',
    'clockwise': 'Kite clockwise relative to the current target, with room to turn at walls.',
    'counterclockwise': 'Kite counterclockwise relative to the current target, with room to turn at walls.',
    'disengage': 'Create separation from nearby pursuers before restoring a firing lane.',
}


def targets(world, plan):
    from .combat import select_enemy
    from .control import firing_lane
    live=[e for e in world.combat_enemies if e.get('is_vulnerable',True)] or world.combat_enemies
    selected=select_enemy(world,plan)
    p=xy(world.player.get('pos'))
    ordered=sorted(live,key=lambda e:(e is not selected,math.dist(p,xy(e.get('pos')))))
    return [{'choice':str(e['id']),'id':e['id'],'type':e.get('type'),'variant':e.get('variant',0),
             'hp':e.get('hp'),'is_boss':e.get('is_boss',False),'pos':e.get('pos'),
             **{k:e.get(k) for k in ('vel','is_vulnerable','is_champion','state','state_frame','collision_radius')},
             'sample_age_updates':world.frame-world.sampled['ENEMIES'] if 'ENEMIES' in world.sampled else None,
             'distance':round(math.dist(p,xy(e.get('pos'))),1),
             'alignment_error':round(min(abs(p[i]-xy(e.get('pos'))[i]) for i in (0,1)),1),
             'terrain_blocks_direct_tears':not firing_lane(world,p,xy(e.get('pos')))}
            for e in ordered[:5] if 'id' in e]


def compact_state(observation, plan, choices):
    """Bounded advisory motion/phase evidence, separate from action permission."""
    from .knowledge import tactical_knowledge
    from .hazards import distance_to_segment
    p=xy(observation.get('player',{}).get('pos'))
    def distance(e):return math.dist(p,xy(e.get('pos')))
    def approaching(e):
        q,v=xy(e.get('pos')),xy(e.get('vel'))
        end=(q[0]+12*v[0],q[1]+12*v[1])
        return distance_to_segment(p,q,end),distance(e)
    fields=('id','type','variant','sub_type','pos','vel','collision_radius','explosion_radius',
            'is_explosive','ground_only','angle','max_distance')
    def trim(e):return {k:e[k] for k in fields if k in e}
    groups={
        'projectiles':(sorted(observation.get('projectiles',[]),key=approaching),8),
        'bombs':(sorted(observation.get('bombs',[]),key=distance),4),
        'troll_pickups':(sorted((e for e in observation.get('pickups',[]) if e.get('variant')==40
                               and e.get('sub_type') in (3,5,6)),key=distance),4),
        'fires':(sorted((e for e in observation.get('fires',[]) if not e.get('is_extinguished')),key=distance),6),
        'lasers':(sorted((e for e in observation.get('lasers',[]) if e.get('is_enemy',True)),key=distance),4)}
    terrain=sorted(observation.get('terrain',[]),key=lambda e:math.dist(p,(e.get('x',0),e.get('y',0))))
    state={k:observation.get(k) for k in ('room','frame','player','stats','health','combat_positioning')}
    state.update(targets=choices,knowledge=tactical_knowledge(observation),
        current_plan={k:plan.get(k) for k in ('objective','boss_focus','target_enemy','combat_distance')},
        geometry={'bounds':{k:observation.get('room_info',{}).get(k) for k in ('top_left','bottom_right')},
                  'terrain':terrain[:32]},
        hazards={k:[trim(e) for e in rows[:limit]] for k,(rows,limit) in groups.items()},
        omitted_from_supplied_snapshot={k:max(0,len(rows)-limit) for k,(rows,limit) in groups.items()}
            | {'terrain':max(0,len(terrain)-32)},
        sensor_age_updates=observation.get('tactical_sensor_age_updates'),
        contract='Advisory intent only; local control rechecks hazards and may override. No resource actions. '
                 'Null means unknown. Lists are capped, including upstream limits. '
                 'Projectile ranking uses linear paths over 12 updates toward the current player position, '
                 'not collision certainty. NPC/projectile vel: per 30 Hz update; player vel: per 60 Hz tick.')
    return state


def bias(world, point, target, posture):
    """Soft preference only, applied after current motion danger filtering."""
    p=xy(world.player.get('pos'))
    radial=(p[0]-target[0],p[1]-target[1]);norm=math.hypot(*radial) or 1
    delta=(point[0]-p[0],point[1]-p[1])
    if posture in ('clockwise','counterclockwise'):
        tangent=(-radial[1]/norm,radial[0]/norm)
        sign=1 if posture=='clockwise' else -1
        return max(-8,min(8,sign*(delta[0]*tangent[0]+delta[1]*tangent[1])*.35))
    if posture=='disengage':
        from .combat import nearest_distance
        return max(-8,min(8,(nearest_distance(world,point)-nearest_distance(world,p))*.5))
    return 0
