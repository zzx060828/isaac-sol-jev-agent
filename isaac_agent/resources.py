"""Observed inventory use, with single input pulses and bounded retries."""
import math

from .control import MOVES, blocked, motion, risk
from .hazards import explosives
from .state import records, xy
from .pickups import ordinary_red_health, red_healing, routine

# IDs from the installed Repentance enums. Unknown items are not guessed.
BENEFICIAL_PILLS = {2, 7, 10, 12, 14, 16, 18, 23, 33, 38, 48}
COMBAT_ITEMS = {34, 35, 192}  # Belial, Necronomicon, Telepathy for Dummies
RESOURCE_ITEMS = {97, 145}  # Book of Sin, Guppy's Head


def safe_room(world):
    return (bool(world.info.get("is_clear")) and not world.enemies and not world.projectiles
            and not records(world.payload.get("BOMBS"))
            and not any(e.get("is_enemy", True) for e in world.payload.get("PROJECTILES", {}).get("lasers", [])))


def safe_to_experiment(world):
    # Three full hearts minimum, with room to flee Explosive Diarrhea.
    health = world.health.get("red_hearts", 0) + world.health.get("soul_hearts", 0)
    if not all(world.fresh(name, 6) for name in ("BOMBS", "FIRE_HAZARDS", "ROOM_LAYOUT", "ROOM_INFO")):
        return False
    if health < 6 or not safe_room(world) or risk(world, (0, 0)) > 1:
        return False
    p = xy(world.player.get("pos"))
    if any(math.dist(p, source["pos"]) < source["radius"] + 80 for source in explosives(world)):
        return False
    if any(math.dist(p, (d["x"], d["y"])) < 70 for d in world.layout.get("doors", {}).values()):
        return False
    exits = 0
    for move in MOVES[1:]:
        v = motion(world, move)
        if all(not blocked(world, (p[0] + v[0]*t, p[1] + v[1]*t), radius=18)
               for t in range(1, 25)):
            exits += 1
    return exits >= 3


def nearby_red_healing(world):
    """Free observed red healing with a checked walking approach, not owned HP."""
    from .control import path_step
    from .pickup_progress import blocked_choices
    from .review import choice_id
    if (not ordinary_red_health(world) or not safe_room(world)
            or any(e['armed'] for e in explosives(world))
            or not all(world.fresh(k,6) for k in ('PICKUPS','BOMBS','ENEMIES','PROJECTILES',
                                                'FIRE_HAZARDS','ROOM_LAYOUT','ROOM_INFO'))):
        return 0
    p=xy(world.player.get('pos'));amount=0
    excluded=blocked_choices(world)
    for h in world.pickups:
        if (not red_healing(world,h) or not routine(h) or h.get('wait',0)>0
                or choice_id(h) in excluded):
            continue
        q=xy(h.get('pos'))
        if (math.dist(p,q)<=150 and not blocked(world,q)
                and (math.dist(p,q)<20 or math.dist(p,path_step(world,q))>1)):
            amount+=red_healing(world,h)
    return amount


def resource_options(world):
    if world.dead or world.control_mode == "MANUAL" or not world.fresh("PLAYER_INVENTORY", 6) or not world.fresh("PLAYER_HEALTH", 6):
        return []
    if world.sampled.get("PLAYER_HEALTH", -1) <= world.last_damage_frame:
        return []  # Damage callbacks precede the updated health snapshot.
    if world.inventory.get("can_use", True) is False:
        return []
    inv, hp = world.inventory, world.health
    red, maximum = hp.get("red_hearts", 0), hp.get("max_hearts", 0)
    total = red + hp.get("soul_hearts", 0)
    safe = safe_room(world) and risk(world, (0, 0)) < 1
    result = []

    def add(kind, item, reason, priority):
        result.append({"kind": kind, "id": item, "reason": reason, "priority": priority})

    def healing_consumable(kind, item, reason, priority):
        if total>4 and safe and nearby_red_healing(world)>=maximum-red:
            add(kind,item,'Nearby free red hearts cover missing health; reserve this one-use heal unless another strategic use is justified',20)
            result[-1]['model_only']=True
        else:
            add(kind,item,reason,priority)

    active = inv.get("active_items", {}).get("0", {})
    item = active.get("item", 0)
    charge = active.get("charge", 0) + active.get("battery_charge", 0)
    ready = (active.get("max_charge", 0) > 0 and charge >= active["max_charge"]
             or active.get("max_charge") == 0 and active.get("charge_type") == 0)
    if item and ready:
        if item == 45 and (maximum - red >= 2 or (maximum > red and total <= 2)):
            add("item", item, "Yum Heart: restore missing red health", 100)
        elif item in (78, 292) and total < 22 and (safe or total <= 4):
            add("item", item, "Generate a soul/black heart", 90)
        elif item == 58 and world.enemies and (total <= 4 or risk(world, (0, 0)) > 5):
            add("item", item, "Book of Shadows: defensive invulnerability", 95)
        elif item in COMBAT_ITEMS and world.enemies:
            add("item", item, "Use charged combat effect", 60)
        elif item in RESOURCE_ITEMS and safe:
            add("item", item, "Use rechargeable resource/summon item", 40)
        elif total >= 6 and not (item == 45 and maximum <= red):
            # Sol can choose any observed ready primary item, including items
            # not covered by emergency reflexes. One pulse per model decision.
            add("item", item, "Model decision required: evaluate this active item's effect and costs from inventory", 20)
            result[-1]["model_only"] = True

    pill = inv.get("pill_0", 0)
    random_pill = bool(pill and (pill & 0x7ff) == 14)  # PILL_GOLD, including horse form.
    if pill and inv.get("pill_identified") is True and not random_pill:
        effect = inv.get("pill_effect")
        if effect == 5 and maximum > red and (maximum - red >= 4 or total <= 2):
            healing_consumable("pill", pill, "Full Health: heal missing red health", 99)
        elif effect == 2 and total <= 4:
            add("pill", pill, "Balls of Steel: gain soul hearts", 98)
        elif effect in BENEFICIAL_PILLS and safe:
            add("pill", pill, "Identified beneficial pill", 50)
        elif effect == 20 and item and charge < active.get("max_charge", 0) and safe:
            add("pill", pill, "48 Hour Energy: recharge active item", 50)
    elif pill and (inv.get("pill_identified") is False or random_pill) and safe_to_experiment(world):
        add("pill", pill, "Try random golden pill in clear room with at least 3 hearts and escape space"
            if random_pill else "Try unidentified pill in clear room with at least 3 hearts and escape space", 30)

    card = inv.get("card_0", 0)
    if card == 20 and (maximum - red >= 4 or (maximum > red and total <= 2)):
        healing_consumable("card", card, "The Sun: heal red health", 99)
    elif card == 6 and safe and total <= 20:
        add("card", card, "The Hierophant: spawn soul hearts", 50)
    elif card == 7 and safe and maximum > red:
        add("card", card, "The Lovers: spawn needed red hearts", 50)
    elif card in (4, 12, 15) and world.enemies:  # Empress, Strength, Devil
        add("card", card, "Combat card", 60)
    # Reflexes above are shortcuts, not the complete set of legal model actions.
    # A described held consumable can be evaluated without a per-ID use rule.
    if all(world.fresh(k, 6) for k in ('ROOM_INFO', 'ROOM_LAYOUT', 'ENEMIES',
                                      'PROJECTILES', 'BOMBS', 'FIRE_HAZARDS')):
        from .knowledge import held_consumable_mechanics
        for mechanics in held_consumable_mechanics(inv):
            kind = 'card' if mechanics['slot'] == 'card_0' else 'pill'
            if any(o['kind'] == kind for o in result):
                continue
            if kind == 'pill' and (inv.get('pill_identified') is not True or random_pill):
                continue  # Unknown/random pills retain the experiment policy.
            if not mechanics.get('description'):
                continue  # Missing mechanics are not guessed from an ID.
            add(kind, card if kind == 'card' else pill,
                'Model decision: compare using now versus keeping, using the supplied effect, '
                'costs, held modifiers, health, floor progress and available resources', 20)
            result[-1].update(model_only=True, mechanics=mechanics)
    return sorted(result, key=lambda option: option["priority"], reverse=True)


def consumable_review_key(level, inventory, options, *, require_option=True):
    """Remember a model review across rooms, not across inventory/build changes."""
    if require_option and not any(o.get('model_only') and o['kind'] in ('card', 'pill') for o in options):
        return None
    return (level, inventory.get('card_0', 0), inventory.get('pill_0', 0),
            inventory.get('pill_identified'), inventory.get('pill_effect'),
            tuple(sorted(inventory.get('collectibles', {}).items())),
            inventory.get('trinket_0', 0), inventory.get('trinket_1', 0))


def resource_signature(inventory, kind):
    """Bind a strategic use to the observed primary slot, not just its category."""
    if kind == 'item':
        active = inventory.get('active_items', {}).get('0', {})
        if not active.get('item'):
            return None
        return {k: active.get(k) for k in ('item', 'charge', 'battery_charge', 'max_charge', 'charge_type')}
    if kind == 'pill' and inventory.get('pill_0'):
        return {k: inventory.get(k) for k in ('pill_0', 'pill_effect', 'pill_identified')}
    if kind == 'card' and inventory.get('card_0'):
        return {'card_0': inventory['card_0']}
    return None


def bind_resource_intent(plan, observation):
    # Generated by the controller; never accept an authorization from model JSON.
    plan.pop('resource_authorization', None)
    kind = plan.get('resource_action')
    if kind:
        plan['resource_authorization'] = {
            'kind': kind, 'signature': resource_signature(observation.get('inventory', {}), kind),
            'source_frame': observation.get('frame')}


def resource_preference(world, plan, policy, log):
    """Drop only an obsolete use request; retain navigation and local reflexes."""
    kind = plan.get('resource_action')
    if not kind:
        return None
    auth = plan.get('resource_authorization', {})
    reason = None
    if auth.get('kind') != kind or not auth.get('signature'):
        reason = 'missing_resource_binding'
    elif not world.fresh('PLAYER_INVENTORY', 6):
        return None  # Wait for the slot snapshot, without authorizing a pulse.
    elif auth['signature'] != resource_signature(world.inventory, kind):
        reason = 'resource_slot_changed'
    elif (policy.run == (world.epoch, world.level)
          and isinstance(auth.get('source_frame'), (int, float))
          and policy.pulse_frames.get(kind, -1) >= auth['source_frame']):
        reason = 'resource_already_pulsed_since_request'
    if reason:
        log.write('resource_intent_discarded', frame=world.frame, kind=kind, reason=reason,
                  source_frame=auth.get('source_frame'))
        plan['resource_action'] = None
        plan.pop('resource_authorization', None)
        return None
    return kind


class ResourcePolicy:
    def __init__(self):
        self.run = None
        self.pending = {}
        self.last_pulse = -1000
        self.pulse_frames = {}

    def choose(self, world, preferred=None, keep=()):
        if not world.fresh("PLAYER_INVENTORY", 6):
            return None
        run = world.epoch, world.level
        if run != self.run:
            self.run, self.pending, self.last_pulse = run, {}, -1000
            self.pulse_frames = {}
        # Changing room alone must not reset a pending inventory input.
        signatures = {
            "item": tuple(world.inventory.get("active_items", {}).get("0", {}).get(k, 0)
                          for k in ("item", "charge", "battery_charge")),
            "pill": (world.inventory.get("pill_0", 0),),
            "card": (world.inventory.get("card_0", 0),),
        }
        for kind, pending in list(self.pending.items()):
            if pending["signature"] != signatures[kind]:
                del self.pending[kind]
        if world.frame - self.last_pulse < 30:
            return None
        choices = resource_options(world)
        if preferred and choices and choices[0]["priority"] < 90:
            choices.sort(key=lambda c: c["kind"] != preferred)
        for choice in choices:
            kind = choice["kind"]
            if kind in keep and kind != preferred and choice['priority'] < 90:
                continue
            if choice.get("model_only") and kind != preferred:
                continue
            pending = self.pending.get(kind)
            if pending and (pending["attempts"] >= 2 or world.frame - pending["frame"] < 45):
                continue
            self.pending[kind] = {"signature": signatures[kind], "frame": world.frame,
                                  "attempts": pending["attempts"] + 1 if pending else 1}
            self.last_pulse = world.frame
            self.pulse_frames[kind] = world.frame
            return choice
        return None
