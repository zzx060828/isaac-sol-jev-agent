"""Room-local decisions to leave observed pickups until their context changes."""
import hashlib
import json
from copy import deepcopy

DEPENDENCIES = ('health','coins','keys','bombs','consumables','charge','ground','doors')


def choice_id(p):
    return f"pickup:{p.get('id')}:{p.get('variant')}:{p.get('sub_type',0)}:{p.get('price',0)}"


def basis(world, room=None):
    room = world.room if room is None else room
    node = world.rooms.get(str(room), {})
    local = room == world.room
    inv = world.inventory
    pickups = world.pickups if local else (node.get('collectibles',[]) + node.get('resources',[]))
    ground = sorted((str(p.get('id')),p.get('variant'),p.get('sub_type'),p.get('price',0),
                     p.get('options_index',0),bool(p.get('is_hidden'))) for p in pickups)
    return deepcopy({
        'build': {'collectibles':inv.get('collectibles',{}), 'stats':world.stats,
                  'active_items':{k:v.get('item') for k,v in inv.get('active_items',{}).items()},
                  'trinkets':[inv.get('trinket_0',0),inv.get('trinket_1',0)]},
        'health':{k:v for k,v in world.health.items() if k!='damage_cooldown'},
        'coins':inv.get('coins',0),'keys':{'count':inv.get('keys',0),'golden':inv.get('golden_key')},
        'bombs':{'count':inv.get('bombs',0),'golden':inv.get('golden_bomb')},
        'consumables':{k:v for k,v in inv.items() if k.startswith(('pill','card'))},
        'charge':inv.get('active_items',{}), 'ground':ground,
        'doors':world.layout.get('doors',{}) if local else node.get('doors',{}),
        'curses':world.info.get('curses') if local else node.get('curses')})


def pool(world):
    return world.run_memory.pickup_reviews if world.run_memory else world.review_cache


def save(world):
    if world.run_memory:
        world.run_memory.save()


def remember(world, observation, plan, log):
    source = observation.get('pickup_review_basis')
    if source is None:
        return
    current = basis(world)
    present = {choice_id(p):p for p in world.pickups}
    stored = []
    for choice in plan.get('declined_pickups',[]):
        p = present.get(choice)
        if not p or choice == plan.get('strategy_choice'):
            continue
        deps = set(plan.get('decline_reconsider_on',{}).get(choice, DEPENDENCIES))
        deps.update(('build','curses'))
        if p.get('price',0) > 0: deps.add('coins')
        if p.get('price',0) < 0: deps.add('health')
        if p.get('options_index',0): deps.add('ground')
        if p.get('variant') in (70,300): deps.add('consumables')
        if p.get('variant') == 100 and p.get('item_type') == 3: deps.add('charge')
        # Canonical JSON avoids list/tuple differences after disk round-trips.
        expected = json.loads(json.dumps({k:source[k] for k in sorted(deps)}))
        if expected != json.loads(json.dumps({k:current[k] for k in sorted(deps)})):
            changed=[k for k in expected if expected[k]!=json.loads(json.dumps(current[k]))]
            log.write('pickup_deferral_rejected',frame=world.frame,choice=choice,
                      reason='relevant_context_changed',changed_conditions=changed)
            continue
        identity = {k:p.get(k,0) for k in ('id','variant','sub_type','price','options_index')}
        identity['is_hidden'] = bool(p.get('is_hidden'))
        key = f'{world.level}:{world.room}:{choice}'
        pool(world)[key] = dict(version=1,level=world.level,room=world.room,choice=choice,
                               identity=identity,basis=expected,frame=world.frame,
                               rationale=str(plan.get('rationale',''))[:700])
        stored.append(choice)
    if stored:
        save(world)
        log.write('pickup_deferrals_saved',frame=world.frame,room=world.room,choices=stored)


def valid(world, record):
    if record.get('version') != 1 or record.get('level') != world.level:
        return False
    room = record['room']; node = world.rooms.get(str(room),{})
    pickups = world.pickups if room == world.room else node.get('collectibles',[])+node.get('resources',[])
    p = next((p for p in pickups if choice_id(p)==record['choice']),None)
    if not p or any((bool(p.get(k)) if k=='is_hidden' else p.get(k,0)) != v
                    for k,v in record['identity'].items()):
        return False
    current = basis(world,room)
    return record['basis'] == json.loads(json.dumps({k:current.get(k) for k in record['basis']}))


def active(world, room=None):
    room = world.room if room is None else room
    return {r['choice'] for r in pool(world).values() if r['room']==room and valid(world,r)}


def refresh(world, log):
    # Wait for actual room snapshots on entry; partial channels are not loss of an item.
    if not world.strategy_ready() or not world.fresh('PLAYER_HEALTH',6) or not world.fresh('PLAYER_STATS',6):
        return
    expired = [k for k,r in pool(world).items() if not valid(world,r)]
    for k in expired:
        r=pool(world).pop(k)
        current=basis(world,r['room'])
        changed=[k for k,v in r['basis'].items() if v!=json.loads(json.dumps(current.get(k)))]
        log.write('pickup_deferral_expired',frame=world.frame,room=r['room'],choice=r['choice'],
                  changed_conditions=changed or ['floor_or_pickup_identity'])
    if expired: save(world)


def fingerprint(world):
    # Current use/pickup eligibility is execution state, not item value. Keep it
    # in observations/diagnostics without reopening an unchanged value review.
    # Include ground resources: a new heart can change an earlier choice.
    inventory = {k:v for k,v in world.inventory.items() if k not in ('can_use','can_pickup_item')}
    health = {k:v for k,v in world.health.items() if k != 'damage_cooldown'}
    pickups = sorted((str(p.get('id')), p.get('variant'), p.get('sub_type'),
                      p.get('price',0), p.get('options_index',0), bool(p.get('is_hidden')))
                     for p in world.pickups)
    value = {'room':world.token, 'inventory':inventory, 'health':health,
             'stats':world.stats, 'pickups':pickups, 'curses':world.info.get('curses'),
             'doors':world.layout.get('doors',{})}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
