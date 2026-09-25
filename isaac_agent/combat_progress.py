"""Ask for strategic help when fighting stalls, not merely on a short timer."""
import math


class CombatProgress:
    def __init__(self):
        self.token = None
        self.started = self.last_progress = -1
        self.lowest_hp = {}
        self.escalated = None

    def update(self, world):
        if self.token != world.token or not world.combat_enemies:
            self.__init__()
            self.token = world.token
        if not world.combat_enemies:
            return None
        if self.started < 0:
            self.started = self.last_progress = world.frame
        for event in world.events:
            frame=event.get('frame',-1)
            if (event.get('event')=='NPC_DEATH' and isinstance(frame,int)
                    and self.started <= frame <= world.frame):
                self.last_progress=max(self.last_progress,frame)
        if world.fresh('ENEMIES', 6):
            for e in world.combat_enemies:
                hp=e.get('hp')
                if isinstance(hp,bool) or not isinstance(hp,(int,float)) or not math.isfinite(hp):
                    continue
                key=(e.get('id'),e.get('type'),e.get('variant'))
                previous=self.lowest_hp.get(key)
                if previous is not None and hp < previous-.01:
                    self.last_progress=world.frame
                self.lowest_hp[key]=min(hp,previous) if previous is not None else hp
        # Latch until clear/room change: resuming damage must not repeatedly
        # toggle the Sol lane. Sustained spawns still get a bounded review.
        if not self.escalated:
            if world.frame-self.last_progress >= 360:
                self.escalated='no_damage_or_kill_progress'
            elif world.frame-self.started >= 1800:
                self.escalated='prolonged_combat'
        return self.escalated
