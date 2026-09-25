"""Bomb stock cost, bounded pulses, and evidence for a free golden-bomb pulse."""
import math

from .state import records, xy


def golden(world):
    return (world.fresh('PLAYER_INVENTORY', 6)
            and world.inventory.get('golden_bomb') is True
            and world.capabilities.get('golden_bomb_protocol') == 1)


def budget(world, pulses, reserve=1):
    free = golden(world)
    if world.inventory.get('golden_bomb') is True and not free:
        return None  # Explicit golden state needs the matching game-side contract.
    cost = 0 if free else pulses
    stock = world.inventory.get('bombs', 0)
    if stock < cost + (0 if free else reserve):
        return None
    return {'bomb_payment': 'golden' if free else 'normal', 'bomb_limit': pulses,
            'bomb_cost': cost, 'bombs_reserved_after': stock-cost}


def payment_valid(world, offer):
    if offer.get('bomb_payment', 'normal') == 'golden':
        return golden(world)
    return world.inventory.get('golden_bomb') is not True


def golden_spawn_observed(world, fired, request_id, spot):
    """Accepted request plus a subsequent nearby normal bomb, not stock delta."""
    ack = world.capabilities
    accepted = ack.get('last_bomb_frame')
    sample = world.sampled.get('BOMBS', -1)
    return (ack.get('last_bomb_request') == request_id
            and type(accepted) is int and fired <= accepted < fired+6
            and world.fresh('BOMBS', 3) and accepted < sample <= fired+30
            and any(b.get('variant') == 0 and math.dist(xy(b.get('pos')), xy(spot)) <= 60
                    for b in records(world.payload.get('BOMBS'))))
