"""Observe ordinary chest opening before committing to the newly spawned loot."""
from .pickups import routine


class ChestObserver:
    def __init__(self):
        self.closed = set()
        self.sample = -1
        self.opened_frame = None

    def update(self, world, log):
        sample = world.sampled.get('PICKUPS', -1)
        if sample > self.sample:
            closed = {p.get('id') for p in world.pickups
                      if p.get('variant') == 50 and routine(p)}
            changed = self.closed - closed
            if changed:
                self.opened_frame = world.frame
                log.write('chest_changed', frame=world.frame, room=world.room,
                          ids=sorted(changed, key=str),
                          result='closed_form_gone; waiting_for_loot_and_hazards')
            self.closed, self.sample = closed, sample
        opened = self.opened_frame
        if opened is None:
            return False
        if (world.frame >= opened + 6 and all(world.sampled.get(k, -1) > opened
                for k in ('PICKUPS', 'BOMBS', 'ENEMIES', 'PROJECTILES',
                          'FIRE_HAZARDS', 'ROOM_INFO', 'PLAYER_INVENTORY', 'PLAYER_HEALTH'))):
            self.opened_frame = None
            log.write('chest_settled', frame=world.frame, room=world.room)
            return False
        return True
