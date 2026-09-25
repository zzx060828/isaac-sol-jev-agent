"""Range-aware positioning policy; distances are heuristic world units."""
import math
from .state import xy
from .knowledge import enemy_briefs
from .physics import trajectory


def traits(enemy):
    return enemy_briefs().get(f"{enemy.get('type')}:{enemy.get('variant', 0)}", {}).get('tags', [])


def spacing(stats, plan=None, enemy=None):
    reach = float(stats.get('range', 260))
    reach = max(30, min(600, reach if math.isfinite(reach) else 260))
    preferred = min(400, reach * (.85 if 'jump' in traits(enemy or {}) else .82))
    proposed = (plan or {}).get('combat_distance')
    if isinstance(proposed, (int, float)) and math.isfinite(proposed):
        preferred = max(reach*.72, min(reach*.88, proposed, 400))
    high = min(reach*.91, preferred+24)
    low = max(reach*.65, preferred-22)
    return {'observed_range': round(reach, 1), 'preferred_distance': round(preferred, 1),
            'min_distance': round(low, 1), 'max_distance': round(high, 1),
            'units': 'world units; approximate for ordinary tears'}


def select_enemy(world, plan):
    enemies = world.combat_enemies
    if not enemies:
        return None
    targets = [e for e in enemies if e.get('is_vulnerable', True)] or enemies
    p = xy(world.player.get('pos'))
    distance = lambda e: math.dist(p, xy(e.get('pos')))
    contact_targets = [e for e in targets if not (
        'no_contact_damage' in traits(e) and not e.get('is_champion')
        and not e.get('subtype',0) and world.capabilities.get('repentance_plus') is False)]
    closest = min(contact_targets, key=distance) if contact_targets else None
    preferred = next((e for e in targets if str(e.get('id')) == str(plan.get('target_enemy'))), None)
    # Kill a threatening close add before chasing the distant strategic target.
    if closest is not None and distance(closest) < min(150, float(world.stats.get('range', 260))*.6):
        if preferred is None or distance(closest) < distance(preferred)*.7:
            return closest
    if preferred:
        return preferred
    # Reachable nearby regenerators/spawners merit finishing; this small bonus
    # never outweighs the immediate close-enemy override above.
    focus = plan.get('boss_focus','balanced')
    return min(targets, key=lambda e: distance(e)-
               ((55 if focus == 'spawners_first' else 35)
                if set(traits(e)).intersection(('spawn', 'regenerate')) else 0)-
               (25 if focus == 'dangerous_adds' and not e.get('is_boss') else 0))


def band_cost(point, target, band):
    d = math.dist(point, target)
    return 1.8*max(0, band['min_distance']-d) + .8*max(0, d-band['max_distance'])


def nearest_distance(world, point):
    return min((math.dist(point, xy(e.get('pos'))) for e in world.combat_enemies), default=999)


def incoming_jump(world):
    """Observed airborne motion only; the NPC target is not a landing marker."""
    p = xy(world.player.get('pos'))
    threats = []
    for enemy in world.enemies:
        if 'jump' not in traits(enemy) or enemy.get('collision_class') != 0:
            continue
        q, v = xy(enemy.get('pos')), xy(enemy.get('vel'))
        speed = math.hypot(*v)
        delta = p[0]-q[0], p[1]-q[1]
        if speed <= 1 or sum(delta[i]*v[i] for i in (0,1)) <= 1:
            continue
        body = float(world.stats.get('size',12))+float(enemy.get('collision_radius',12))
        cross = abs(delta[0]*v[1]-delta[1]*v[0])/speed
        # A bounded early-turn window accommodates observed jump acceleration.
        # This guides movement; it grants no immunity or exact impact timing.
        if cross < body+45 and math.hypot(*delta) < min(300, max(180,speed*16)+body):
            threats.append((math.hypot(*delta)/speed, enemy))
    return min(threats, key=lambda pair: pair[0])[1] if threats else None


def edge_flank_axis(world, target):
    """A firing lane into a pursuer needs an early turn before a wall trap."""
    if not world.combat_enemies:
        return None
    p=xy(world.player.get('pos'))
    if math.dist(p,target)>min(180,float(world.stats.get('range',260))*.82):
        return None
    enemy=min(world.combat_enemies,key=lambda e:math.dist(xy(e.get('pos')),target))
    if 'chase' not in traits(enemy):
        return None  # Segmented/charging/jumping motion needs its own policy.
    v=xy(enemy.get('vel'))
    toward=(p[0]-target[0])*v[0]+(p[1]-target[1])*v[1]
    if toward<=1:
        return None  # Stationary shooters do not justify abandoning a good lane.
    lo,hi=xy(world.info.get('top_left')),xy(world.info.get('bottom_right'))
    if ((p[1]-lo[1]<48 and target[1]>p[1]+35)
            or (hi[1]-p[1]<48 and target[1]<p[1]-35)):
        return 0
    if ((p[0]-lo[0]<48 and target[0]>p[0]+35)
            or (hi[0]-p[0]<48 and target[0]<p[0]-35)):
        return 1
    return None


def crowding_override(world, action, fallback):
    if not world.combat_enemies or action.move == fallback.move:
        return False
    current = nearest_distance(world, xy(world.player.get('pos')))
    if current >= min(180, float(world.stats.get('range', 260))*.72):
        return False
    chosen = nearest_distance(world, trajectory(world, action.move, 4)[-1])
    alternative = nearest_distance(world, trajectory(world, fallback.move, 4)[-1])
    return chosen < current-4 and alternative > chosen+8


def context(world, plan=None):
    target = select_enemy(world, plan or {})
    if not target:
        return None
    band = spacing(world.stats, plan, target)
    p = xy(world.player.get('pos'))
    return {**band, 'selected_enemy': target.get('id'), 'strategic_enemy': (plan or {}).get('target_enemy'),
            'target_distance': round(math.dist(p, xy(target.get('pos'))), 1),
            'nearest_enemy_distance': round(nearest_distance(world, p), 1),
            'policy': 'Hold a clear firing lane inside the distance band; do not close merely to improve aim. '
                      'Make space from nearby adds, leave room to turn, and sidestep jumps/charges. '
                      'Immediate projectile/terrain safety takes priority over the band.'}
