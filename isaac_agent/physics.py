"""Approximate movement fitted to ordinary live Isaac input, not engine state edits.

SocketBridge publishes at 30 updates/s; player velocity is per 60 Hz tick.
Recorded NPC and hostile projectile velocities are per 30 Hz update instead.
Free travel in the 2026-09-23 recording gives 4.667 velocity units / MoveSpeed;
release decays velocity by about .775 per sensor update. Special movement items,
knockback and slippery floors can deviate from this simple model.
"""
import math

from .state import xy

PHYSICS_TICKS_PER_UPDATE = 2


def hostile_position(entity, updates):
    """NPC/projectile projection in sensor-update units, not player ticks."""
    p, v = xy(entity.get('pos')), xy(entity.get('vel'))
    return p[0]+v[0]*updates, p[1]+v[1]*updates


def swept_separation(player_start, player_end, hostile_start, hostile_end):
    """Closest simultaneous separation for two linear paths over one interval."""
    x, y = player_start[0]-hostile_start[0], player_start[1]-hostile_start[1]
    vx, vy = player_end[0]-hostile_end[0]-x, player_end[1]-hostile_end[1]-y
    square = vx*vx+vy*vy
    when = max(0, min(1, -(x*vx+y*vy)/square)) if square else 0
    return math.hypot(x+when*vx,y+when*vy)


def trajectory(world, move, frames=8, collides=None):
    p = list(xy(world.player.get('pos')))
    velocity = list(xy(world.player.get('vel')))
    scale = 4.667 * float(world.stats.get('speed', 1)) / (math.hypot(*move) or 1)
    desired = [axis*scale for axis in move]
    friction = math.sqrt(.775)
    points = []
    for _ in range(frames):
        for _ in range(PHYSICS_TICKS_PER_UPDATE):
            for axis in (0, 1):
                velocity[axis] = friction*velocity[axis] + (1-friction)*desired[axis]
            proposed = [p[axis]+velocity[axis] for axis in (0,1)]
            if collides and collides(tuple(proposed)):
                # Slide along one free axis instead of forecasting through a wall.
                for axis in (0,1):
                    step = list(p); step[axis] += velocity[axis]
                    if collides(tuple(step)):
                        velocity[axis] = 0
                    else:
                        p = step
            else:
                p = proposed
        points.append(tuple(p))
    return points
