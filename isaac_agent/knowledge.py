"""Read installed EID descriptions as data, without executing mod Lua."""
from functools import lru_cache
import json
import os
from pathlib import Path
import re


@lru_cache(maxsize=1)
def enemy_briefs():
    return json.loads(Path(__file__).with_name('enemy_briefs.json').read_text())


def encounter_knowledge(enemies, budget=2500):
    """Small, curated Wiki summaries for observed type/variant pairs only."""
    result, seen = [], set()
    # Boss mechanics and recurring spawns must not disappear behind a list of
    # nearby copies of the same small enemy.
    ordered = sorted(enemies, key=lambda e: (not e.get('is_boss', False),
        'spawn' not in enemy_briefs().get(f"{e.get('type')}:{e.get('variant', 0)}", {}).get('tags', [])))
    for enemy in ordered:
        key = f"{enemy.get('type')}:{enemy.get('variant', 0)}"
        brief = enemy_briefs().get(key)
        if key in seen or brief is None:
            continue
        seen.add(key)
        entry = {'entity': key, **brief, 'edition': 'Repentance', 'reviewed': brief.get('reviewed','2026-09-23')}
        if enemy.get('is_champion') or enemy.get('subtype', 0):
            entry['variant_caution'] = 'Champion/subtype observed; base behavior may differ. Use live vulnerability and attacks.'
        size = len(json.dumps(entry)) + 2
        if size > budget:
            continue
        budget -= size
        result.append(entry)
    return result


def encounter_coverage(enemies, included):
    observed = {f"{e.get('type')}:{e.get('variant', 0)}" for e in enemies}
    known = observed.intersection(enemy_briefs())
    return {'observed_types': len(observed), 'known_types': len(known),
            'unknown_types': sorted(observed-known),
            'omitted_for_budget': sorted(known-{b['entity'] for b in included}),
            'caution': 'Unknown types use generic geometry only; a name is not a verified behavior model.'}


@lru_cache(maxsize=5)
def item_descriptions(kind='collectible'):
    eid_dir = os.environ.get("ISAAC_EID_DIR")
    if not eid_dir:
        return {}
    root = Path(eid_dir)
    descriptions = {}
    quoted = r'"(?:\\.|[^"\\])*"'
    row = re.compile(r'\{\s*("\d+")\s*,\s*(' + quoted + r')\s*,\s*(' + quoted + r')\s*\}')
    tables = {
        'collectible': (("ab+", "collectibles"), ("rep", "repCollectibles")),
        'trinket': (("ab+", "trinkets"), ("rep", "repTrinkets")),
        'card': (("ab+", "cards"), ("rep", "repCards")),
        'pill': (("ab+", "pills"), ("rep", "repPills")),
        'horse_pill': (("rep", "horsepills"),),
    }[kind]
    for edition, table in tables:
        path = root / "descriptions" / edition / "en_us.lua"
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8-sig")
        match = re.search(r'(?:local\s+|\.)' + table + r'\s*=\s*\{', source)
        if not match:
            continue
        end = source.find('\n}', match.end())
        section = source[match.end():end if end >= 0 else len(source)]
        for match in row.finditer(section):
            try:
                number, name, description = map(json.loads, match.groups())
            except ValueError:
                continue
            description = re.sub(r'\{\{([^}]+)\}\}', r'\1 ', description).replace('#', '; ')
            descriptions[int(number)] = {"name": name, "description": description,
                                         "description_source": "installed EID " + edition}
    return descriptions


def held_consumable_mechanics(inventory):
    """Primary slots only; pill color never supplies an unobserved effect ID."""
    result = []
    card = inventory.get('card_0', 0)
    if card:
        result.append({'slot': 'card_0', 'id': card, 'pickup_type': 'card_or_rune',
                       **item_descriptions('card').get(card, {'name': 'Unknown card or rune'})})
    color = inventory.get('pill_0', 0)
    if color:
        horse = bool(color & 0x800)  # Installed PillColor.PILL_GIANT_FLAG.
        entry = {'slot': 'pill_0', 'appearance_id': color, 'pickup_type': 'pill',
                 'form': 'horse' if horse else 'normal',
                 'identified': inventory.get('pill_identified') is True}
        if (color & 0x7ff) == 14:  # PILL_GOLD: color is known, next effect is random.
            entry.update(name='Golden Pill', description='Random pill effect; may disappear after use.',
                         description_source='installed EID rep golden pill', effect_known=False)
        elif entry['identified'] and type(inventory.get('pill_effect')) is int:
            effect = inventory['pill_effect']
            entry.update(effect_id=effect,
                         **item_descriptions('horse_pill' if horse else 'pill').get(
                             effect, {'name': 'Unknown identified pill effect'}))
        else:
            entry.update(name='Unidentified pill', effect_known=False)
        result.append(entry)
    for entry in result:
        if 'description' in entry:
            entry['description'] = entry['description'][:500]
        entry['scope'] = 'Base mechanics only; held modifiers may change effects. Use requires a current resource option.'
    return result


def held_trinket_mechanics(inventory):
    result = []
    for slot in ('trinket_0', 'trinket_1'):
        item_id = inventory.get(slot, 0)
        if item_id:
            result.append({'slot': slot, 'id': item_id, 'pickup_type': 'trinket',
                           **item_descriptions('trinket').get(item_id, {'name': 'Unknown trinket'})})
    return result


def held_item_mechanics(inventory, budget=3200):
    """Bounded mechanics context for Sol's build/synergy comparisons."""
    descriptions = item_descriptions()
    result = []
    active = [row.get('item') for row in inventory.get('active_items', {}).values()]
    ids = list(dict.fromkeys(active + [int(key) for key in inventory.get('collectibles', {})]))
    for item_id in ids:
        info = descriptions.get(item_id)
        if not info or budget <= 0:
            continue
        detail = info['description'][:min(500, budget)]
        budget -= len(detail)
        result.append({'id': item_id, 'name': info['name'], 'description': detail,
                       'description_source': info['description_source']})
    return result


@lru_cache(maxsize=1)
def room_briefs():
    return json.loads(Path(__file__).with_name('room_mechanics.json').read_text())


@lru_cache(maxsize=1)
def expert_briefs():
    return json.loads(Path(__file__).with_name('expert_briefs.json').read_text())


def expert_knowledge(world, budget=5200):
    keys = ['operating_policy', 'boss_preparation', 'deal_economy', 'charge_economy']
    if any(g.get('type') in (4,22) and g.get('collision',0)
           for g in world.layout.get('grid',{}).values()):
        keys.insert(1,'marked_rocks')
    if world.inventory.get('bombs',0) or len(world.rooms) > 1:
        keys += ['secret_regular','secret_super','secret_ultra']
    result, omitted = [], []
    for key in keys:
        entry = {'topic':key, **expert_briefs()[key], 'edition':'Repentance'}
        size = len(json.dumps(entry))
        if size <= budget:
            result.append(entry); budget -= size
        else:
            omitted.append(key)
    return {'briefs':result, 'omitted_topics':omitted,
            'distinction':'facts describe mechanics; policy describes this controller\'s decision principles.'}


def room_knowledge(world, budget=2400):
    from .scavenging import targets
    from .state import records
    kinds = {t['kind'] for t in targets(world)}
    keys = []
    if any(g.get('type') in (4,22) and g.get('collision',0)
           for g in world.layout.get('grid',{}).values()):
        keys.append('marked_rocks')
    if any(m.get('variant') == 2 for m in records(world.payload.get('INTERACTABLES'))):
        keys.append('donation')
    if any(g.get('type') == 20 for g in world.layout.get('grid', {}).values()):
        keys.append('pressure_plate')
    if 'brown_poop' in kinds:
        keys.append('poop')
    if 'ordinary_fire' in kinds:
        keys.append('fire')
    if world.inventory.get('bombs', 0) > 0 and len(world.rooms) > 1:
        keys.append('secret')
    if world.capabilities.get('repentance_plus') is False and (
            world.info.get('room_type') in (5, 14, 15)
            or any(d.get('target_room_type') in (14, 15) for d in world.layout.get('doors', {}).values())):
        keys.append('deal')
    result = []
    for key in keys:
        entry = {'topic': key, **room_briefs()[key], 'edition': 'Repentance', 'reviewed': '2026-09-23'}
        cost = len(json.dumps(entry)) + 2
        if cost <= budget:
            result.append(entry)
            budget -= cost
    return result


@lru_cache(maxsize=1)
def decision_briefs():
    return json.loads(Path(__file__).with_name('decision_knowledge.json').read_text())


def decision_knowledge(observation, audience='sol', budget=2200):
    inv = observation.get('inventory', {})
    ids = {int(i) for i in inv.get('collectibles', {})}
    pickups = observation.get('pickups', [])
    if audience == 'sol':
        ids.update(e.get('sub_type', 0) for e in pickups if e.get('variant') == 100)
    keys = []
    if audience == 'sol':
        if any(e.get('variant') == 100 or e.get('options_index') for e in pickups):
            keys.append('choices')
        if any(e.get('variant') == 10 for e in pickups):
            keys.append('hearts')
        if observation.get('room_info', {}).get('room_type') == 2:
            keys.append('shop')
        if observation.get('room_info', {}).get('curses', 0) & 64 or any(e.get('is_hidden') for e in pickups):
            keys.append('blind')
    keys += [k for k, v in decision_briefs().items() if ids.intersection(v.get('items', []))]
    result = []
    for key in keys:
        brief = decision_briefs()[key]
        entry = {'topic': key, 'facts': brief['facts'], 'source_urls': brief['source_urls'],
                 'edition': 'Repentance', 'reviewed': '2026-09-23'}
        size = len(json.dumps(entry))
        if size <= budget:
            result.append(entry)
            budget -= size
    return result


def tactical_knowledge(observation, budget=1500):
    result = decision_knowledge(observation, 'jev', budget)
    used = sum(len(json.dumps(x)) for x in result)
    for brief in observation.get('enemy_knowledge', []) + observation.get('room_knowledge', []):
        if brief.get('topic') not in (None, 'fire', 'poop'):
            continue
        # Provenance remains in the logged full observation and local library;
        # the fast action model only needs the applicable behavior and tactic.
        brief = {k: v for k, v in brief.items() if k in
                 ('entity', 'topic', 'name', 'facts', 'tags', 'tactics', 'variant_caution')}
        cost = len(json.dumps(brief))
        if used + cost <= budget:
            result.append(brief)
            used += cost
    return result
