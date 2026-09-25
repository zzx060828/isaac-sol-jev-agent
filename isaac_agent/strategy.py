"""Model-selected goals bound to actual pickups and single-contact trades."""
import json
import math
from pathlib import Path

from .state import records, xy
from .knowledge import item_descriptions
from .deals import health_contract, door_offers
from .scavenging import targets as scavenge_targets
from .donation import DonationCycle, cycle_contract, contract_key
from .secrets import SecretSearch, offers as secret_offers
from .rocks import RockBomb, offers as rock_offers

ITEM_NAMES = json.loads(Path(__file__).with_name("item_names.json").read_text())
ENTITY_NAMES = json.loads(Path(__file__).with_name('entity_names.json').read_text())


def describe_pickup(item):
    result = dict(item)
    result['pickup_type'] = {10:'heart',20:'coin',30:'key',40:'bomb',50:'ordinary_chest',70:'pill',90:'battery',
                             100:'collectible',300:'card_or_rune',350:'trinket'}.get(item.get('variant'),'other')
    if item.get("variant") == 100:
        result["name"] = ITEM_NAMES.get(str(item.get("sub_type")), "Unknown collectible")
        result.update(item_descriptions().get(item.get("sub_type"), {}))
    elif item.get('variant') == 350:
        result.update(item_descriptions('trinket').get(item.get('sub_type'), {'name': 'Unknown trinket'}))
    elif item.get('variant') == 300:
        result.update(item_descriptions('card').get(item.get('sub_type'), {'name': 'Unknown card or rune'})
                      if not item.get('is_hidden') else {'name': 'Hidden card or rune'})
        if 'description' in result:
            result['description'] = result['description'][:500]
    elif item.get('variant') == 70:
        result['name'] = 'Pill; appearance does not identify effect'
    elif item.get('variant') == 30 and item.get('sub_type') == 4:
        result.update(name='Charged Key',description='Grants a key and ordinary battery charging. Compare current key needs and the charge opportunity before taking it.')
    elif item.get('variant') == 90:
        result['name']={1:'Lil Battery',2:'Micro Battery',3:'Mega Battery',4:'Golden Battery'}.get(item.get('sub_type'),'Unknown battery')
        if item.get('sub_type') in (1,2):
            result['description']='Adds up to '+str(6 if item['sub_type']==1 else 2)+' charges to the current eligible active item; does not charge the inactive Schoolbag slot. Check ordinary_battery_target and held modifiers.'
    return result


def offers(world, *, include_waiting=False):
    from .pickups import touch_collectible, useful
    if world.combat_enemies:
        return []
    plates = pressure_plate_offers(world)
    if not world.info.get('is_clear'):
        return plates
    result = door_offers(world)
    result += secret_offers(world)
    result += rock_offers(world)
    for item in world.pickups:
        variant, subtype = item.get("variant"), item.get("sub_type", 0)
        price = item.get("price", 0)
        if not touch_collectible(item):
            continue
        if variant not in (10, 20, 30, 40, 50, 70, 90, 100, 300, 350) or (item.get("wait", 0) > 0 and not include_waiting):
            continue
        if variant == 50 and not useful(world, item):
            continue
        if variant == 90 and subtype == 4:
            continue  # Golden battery hurts; no supported health-cost contract yet.
        if variant == 90 and subtype in (1,2):
            from .charge import battery_target
            if battery_target(world) is None:
                continue  # No eligible recipient, including paid ordinary batteries.
        contract = health_contract(world, item) if price < 0 and price != -1000 else None
        if price < 0 and price != -1000 and contract is None:
            continue
        if price > world.inventory.get("coins", 0):
            continue
        # A touch contract cannot collect a basic resource enclosed by rocks/pits.
        # Keep the pickup in raw observations/map memory for future planning;
        # terrain changes (or flight) can make it a valid offer later.
        from .control import path_step
        point = xy(world.player.get('pos'))
        if (variant in (10,20,30,40,50,70,90) and price == 0
                and math.dist(point, xy(item.get('pos'))) >= 28
                and math.dist(point, path_step(world, xy(item.get('pos')))) < 1):
            continue
        key = f"pickup:{item.get('id')}:{variant}:{subtype}:{price}"
        result.append({"choice": key, "kind": "pickup", "entity": describe_pickup(item),
                       "coin_cost": max(0, price), "health_contract": contract,
                       "effect": ("Trinket pickup: may replace a held trinket, dropping it on the floor. Compare held_trinket_mechanics; do not repeatedly exchange the same pair. Does not replace an active item."
                                  if variant == 350 else "Touch to open free ordinary chest; local controller waits for new loot/hazards before collecting"
                                  if variant == 50 else "Touch once to collect/buy; collectible choices and active swaps may remove alternatives")})
    # Trades are offered only while the observed health can cover a conservative
    # one-heart bound and retain at least two red hearts. Model decides value.
    red = world.health.get("red_hearts", 0)
    if world.info.get('room_type') == 5:
        for cell in world.boss_exits:
            if cell:
                result.append({'choice': f"descend:{cell.get('grid_index')}", 'kind': 'descend',
                               'entity': {'pos': {'x': cell['x'], 'y': cell['y']}},
                               'effect': 'Leave this floor through the open boss trapdoor. Collect useful boss rewards and consider unfinished treasure rooms first; returning is normally impossible.'})
    if (red >= 6 and world.inventory.get('coins', 0) < 99
            and all(world.fresh(k, 6) for k in ('PLAYER_HEALTH', 'INTERACTABLES'))
            and not records(world.payload.get("BOMBS")) and not world.projectiles):
        for machine in records(world.payload.get("INTERACTABLES")):
            if machine.get("variant") == 2 and not machine.get("broken", False):
                key = f"donate:{machine.get('id')}:{red}:{world.inventory.get('coins', 0)}"
                recovery = cycle_contract(world)
                if recovery:
                    result.append({'choice': key+':cycle:'+contract_key(recovery), 'kind': 'donate_cycle', 'entity': machine,
                                   'recovery_contract': recovery, 'effect': recovery['effect']})
                    continue
                result.append({"choice": key, "kind": "donate", "entity": machine,
                               "max_damage_half_hearts": 2, "min_red_half_hearts_after": 4,
                               "effect": "One contact with blood donation machine for coins, then withdraw; payout is uncertain"})
    from .scavenging import reachable
    sources = scavenge_targets(world)
    p=xy(world.player.get('pos'))
    first_reachable=next((t for t in sorted(sources,key=lambda t:math.dist(p,xy(t['pos'])))
                          if reachable(world,t)),None)
    if first_reachable:
        result.append({'choice': 'scavenge', 'kind': 'scavenge', 'source_count': len(sources),
                       'explosive_sources':sum(bool(t.get('explosive')) for t in sources),
                       'source_count_scope':'observed; only the first target has a checked firing approach',
                       'entity': first_reachable, 'coin_cost': 0, 'bomb_cost': 0,
                       'effect': 'Shoot reachable ordinary fires and brown poop for possible coins/hearts. '
                                 'Brown Cap-marked poop explodes: local execution must find a safe ranged firing position and check blast chains. '
                                 'One bounded room sweep; no guaranteed drop. Skip if detour/time/risk exceeds value.'})
    return result


def pressure_plate_offers(world):
    # Only ordinary trap-room plates. Reward/Greed/rail/kill plates have
    # different consequences and must not inherit this routine contract.
    if (world.capabilities.get('repentance_plus') is not False
            or world.info.get('room_type') != 1 or world.info.get('is_clear') is not False
            or world.combat_enemies or not world.fresh('ROOM_LAYOUT', 6)
            or not world.fresh('ROOM_INFO', 6)):
        return []
    return [{'choice': f'pressure_plate:{index}', 'kind': 'pressure_plate',
             'entity': {**cell, 'pos': {'x':cell['x'], 'y':cell['y']}},
             'effect': 'Press this ordinary pressure plate to disarm room hazards. All such plates must be pressed to clear the room. Local movement still avoids live bombs, spikes and projectiles.'}
            for index,cell in world.layout.get('grid', {}).items()
            if cell.get('type') == 20 and cell.get('variant') == 0 and cell.get('state') == 0]


class StrategyExecutor:
    def __init__(self):
        self.token = None
        self.choice = None
        self.started = -1
        self.damage_frame = -1
        self.contact = None
        self.retreat = None
        self.finished = set()
        self.scavenge_target = None
        self.scavenge_ignored = set()
        self.scavenge_reachability_basis = None
        self.started_offer = None
        self.started_item_count = 0
        self.donation = None
        self.secret = None
        self.rock = None

    def update_donation(self, world, execution, log):
        execution = self.donation.update(world, execution, log)
        if self.donation.done:
            log.write('strategy_result', choice=self.choice, frame=world.frame,
                      result=self.donation.done, contacts=self.donation.contacts)
            self.finished.add(self.choice)
            self.choice = self.donation = None
            execution.update(strategy_choice=None, awaiting_strategy=True,
                             excluded_choices=list(self.finished))
        return execution

    def update(self, world, plan, log):
        if self.rock is not None:
            execution=self.rock.update(world,plan,log)
            execution['excluded_choices']=list(self.finished)
            if self.rock.done:
                log.write('strategy_result',choice=self.choice,frame=world.frame,result=self.rock.done)
                self.finished.add(self.choice);self.choice=self.rock=None
                execution.update(strategy_choice=None,awaiting_strategy=True,
                                 excluded_choices=list(self.finished))
            return execution
        if self.secret is not None:
            execution = self.secret.update(world, plan, log)
            execution['excluded_choices'] = list(self.finished)
            if self.secret.done:
                log.write('strategy_result',choice=self.choice,frame=world.frame,result=self.secret.done)
                self.finished.add(self.choice)
                self.choice = self.secret = None
                execution.update(strategy_choice=None,awaiting_strategy=True,
                                 excluded_choices=list(self.finished))
            return execution
        if self.token != world.token:
            if self.choice and self.choice.startswith('descend:') and self.token and self.token[1] != world.level:
                log.write('strategy_result', choice=self.choice, frame=world.frame, result='floor_changed')
            self.__init__()
            self.token = world.token
        execution = dict(plan)
        execution['excluded_choices'] = list(self.finished)
        if self.donation is not None:
            return self.update_donation(world, execution, log)
        if not plan.get("strategy_enabled") or world.combat_enemies:
            return execution
        available = {o["choice"]: o for o in offers(world)}
        requested = plan.get("strategy_choice")
        if requested in self.finished:
            execution.update(strategy_choice=None, awaiting_strategy=True)
        if self.choice is not None:
            choice = available.get(self.choice)
            damaged = world.last_damage_frame > self.damage_frame
            expired = world.frame - self.started > (600 if self.choice == 'scavenge' else 180)
            if choice is None or damaged or expired:
                original = self.started_offer or {}
                entity = original.get('entity', {})
                item_id = str(entity.get('sub_type'))
                acquired = (original.get('kind') == 'pickup' and entity.get('variant') == 100
                            and world.inventory.get('collectibles', {}).get(item_id, 0) > self.started_item_count)
                paid = entity.get('price', 0) not in (0, -1000)
                if world.run_memory:
                    world.run_memory.record_outcome(world,self.choice,acquired)
                if acquired and paid and getattr(world, 'run_memory', None):
                    marker = f'{world.level}:{world.room}:{self.choice}'
                    if marker not in world.run_memory.paid:
                        world.run_memory.paid.append(marker)
                        world.run_memory.add(world, 'paid_item_acquired',
                            {'item_id': item_id, 'observed_price': entity.get('price'),
                             'room_type': world.info.get('room_type'), 'health': world.health}, ['item:'+item_id])
                details={}
                if original.get('kind')=='pickup':
                    from .diagnostics import pickup_contact_facts
                    details['pickup_contact']=pickup_contact_facts(world,entity)
                log.write("strategy_result", choice=self.choice, frame=world.frame,
                          result="state_changed" if choice is None else "damage_stop" if damaged else "timeout",**details)
                self.finished.add(self.choice)
                execution['excluded_choices'] = list(self.finished)
                execution.update(strategy_choice=None, awaiting_strategy=True)
                self.choice = None
                if self.contact is not None:
                    p = xy(world.player.get("pos"))
                    dx, dy = p[0] - self.contact[0], p[1] - self.contact[1]
                    norm = math.hypot(dx, dy) or 1
                    self.retreat = (self.contact[0] + (dx/norm if dx or dy else 1)*90,
                                    self.contact[1] + dy/norm*90)
        if self.retreat is not None:
            p = xy(world.player.get("pos"))
            if math.dist(p, self.contact) < 70:
                execution["navigation_target"] = self.retreat
                execution["navigation_contact"] = False
                return execution
            self.retreat = self.contact = None
        if self.choice is None and requested and requested not in available and requested not in self.finished:
            # An accepted plan can become stale before its first execution tick.
            # Resolve it so the controller can replan instead of holding forever.
            self.finished.add(requested)
            log.write('strategy_result', choice=requested, frame=world.frame,
                      result='offer_unavailable_before_start')
            execution.update(strategy_choice=None, awaiting_strategy=True,
                             navigation_contact=False, excluded_choices=list(self.finished))
            return execution
        if self.choice is None and requested in available and requested not in self.finished:
            self.choice, self.started = requested, world.frame
            self.damage_frame = world.last_damage_frame
            offer = available[requested]
            self.started_offer = offer
            self.started_item_count = world.inventory.get('collectibles', {}).get(str(offer['entity'].get('sub_type')), 0)
            self.contact = xy(offer["entity"]["pos"]) if offer["kind"] == "donate" else None
            log.write("strategy_start", choice=requested, frame=world.frame, kind=offer["kind"])
            if offer['kind'] == 'donate_cycle':
                self.donation = DonationCycle(world, offer)
                return self.update_donation(world, execution, log)
            if offer['kind'] == 'secret_bomb':
                self.secret = SecretSearch(world,offer)
                return self.update(world,plan,log)
            if offer['kind'] == 'rock_bomb':
                self.rock = RockBomb(world,offer)
                return self.update(world,plan,log)
        if self.choice in available:
            offer = available[self.choice]
            if offer['kind'] == 'deal_skip':
                memory = getattr(world, 'run_memory', None)
                if memory and offer['door_key'] not in memory.skipped:
                    memory.skipped.append(offer['door_key'])
                    memory.add(world, 'deal_door_skipped', {'key': offer['door_key'],
                        'type': offer['entity'].get('target_room_type'),
                        'status': 'entry_declined_not_proof_of_floor_unentered'})
                slot = self.choice.split(':', 1)[1]
                self.finished.update((self.choice, 'deal_enter:'+slot))
                log.write('strategy_result', choice=self.choice, frame=world.frame, result='door_skipped')
                self.choice = None
                execution.update(strategy_choice=None, awaiting_strategy=True)
                return execution
            if offer['kind'] == 'scavenge':
                from .control import pickup_target
                # Already-observed free loot is useful before another random
                # drop. Keep the same bounded sweep authorization while local
                # navigation collects it; do not pay for another model decision.
                if (world.strategy_ready() and all(world.fresh(k,3) for k in
                        ('PLAYER_HEALTH','BOMBS','PROJECTILES','FIRE_HAZARDS'))):
                    target = pickup_target(world, execution, basic_only=True)
                    if target is not None:
                        execution.update(navigation_target=target, navigation_contact=False,
                                         scavenge_resource_collection=True)
                        return execution
                from .scavenging import reachable
                targets = scavenge_targets(world)
                point = xy(world.player.get('pos'))
                live = {t['key']: t for t in targets}
                if (self.scavenge_target not in live or
                        live[self.scavenge_target].get('explosive') and not reachable(world,live[self.scavenge_target])):
                    from .control import walkability_signature
                    # A failed approach belongs to these barriers, reach and
                    # starting position. Shooting an opening (or relocating) can
                    # make an earlier source accessible within the same sweep.
                    basis = (walkability_signature(world), world.stats.get('range'), point)
                    if basis != self.scavenge_reachability_basis:
                        if self.scavenge_ignored:
                            log.write('scavenge_reconsidered', frame=world.frame,
                                      targets=sorted(self.scavenge_ignored), reason='approach_conditions_changed')
                        self.scavenge_ignored.clear()
                        self.scavenge_reachability_basis = basis
                    self.scavenge_target = None
                    for target in sorted(targets, key=lambda t: math.dist(point, xy(t['pos']))):
                        if target['key'] in self.scavenge_ignored:
                            continue
                        if reachable(world,target):
                            self.scavenge_target = target['key']
                            break
                        self.scavenge_ignored.add(target['key'])
                if self.scavenge_target is None:
                    log.write('strategy_result', choice=self.choice, frame=world.frame,
                              result='no_reachable_scavenge_target')
                    self.finished.add(self.choice)
                    self.choice = None
                    execution.update(strategy_choice=None, awaiting_strategy=True)
                else:
                    execution['shoot_target'] = live[self.scavenge_target]
                return execution
            execution["navigation_target"] = xy(offer["entity"]["pos"])
            execution["navigation_contact"] = True  # Authorize only this selected pickup/machine.
        return execution
