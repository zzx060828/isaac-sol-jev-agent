"""Small HTTP clients. Model calls never run on the control event loop."""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
import tomllib
import urllib.error
import urllib.request
from urllib.parse import urlparse
from .credentials import load_keys

DEFAULTS = {
    "astra": {"url": "https://api.openai.com/v1/chat/completions", "model": "gpt-5.6-sol",
              "key_env": "OPENAI_API_KEY", "timeout": 25.0,
              "max_completion_tokens": 1200, "reasoning_effort": "medium"},
    "jev": {"url": "https://api.typesafe.ai/v1/systemone", "model": "jev-latest",
            "key_env": "TYPESAFE_API_KEY", "timeout": 2.0, "proxy": "direct"},
}


def configuration(path=None):
    override = {}
    if path:
        with open(path, "rb") as handle:
            override = tomllib.load(handle)
    result = {k: {**v, **override.get(k, {})} for k, v in DEFAULTS.items()}
    for lane, config in result.items():
        url = urlparse(config["url"])
        if url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")):
            raise ValueError(f"{lane}: use HTTPS or a localhost relay")
        if not 0 < float(config["timeout"]) <= 120:
            raise ValueError(f"{lane}: invalid timeout")
        if config.get("proxy", "environment") not in ("environment", "direct"):
            raise ValueError(f"{lane}: proxy must be environment or direct")
    completion_settings(result['astra'])
    return result


def completion_settings(config):
    limit=config.get('max_completion_tokens',1200)
    effort=config.get('reasoning_effort','medium')
    if isinstance(limit,bool) or not isinstance(limit,int) or not 256 <= limit <= 8192:
        raise ValueError('astra: max_completion_tokens must be an integer in 256..8192')
    if effort not in ('low','medium','high'):
        raise ValueError('astra: unsupported reasoning_effort')
    return {'max_completion_tokens':limit,'reasoning_effort':effort}


class PlanValidationError(ValueError):
    def __init__(self, reason, metadata):
        super().__init__(reason)
        self.metadata=metadata


class ModelHTTPError(RuntimeError):
    def __init__(self, status, code=None, retry_after=0, indicators=()):
        self.status, self.code, self.retry_after = status, code, retry_after
        self.indicators = tuple(indicators)
        super().__init__(f"Model HTTP {status}" + (f" ({code})" if code else "")
                         + (f" indicators={','.join(indicators)}" if indicators else ""))


def post(config, body):
    key = os.environ.get(config["key_env"], "")
    if not key:
        raise ValueError(f"Missing environment variable {config['key_env']}")
    request = urllib.request.Request(config["url"], data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key,
                 "User-Agent": "isaac-agent/0.1", "Accept": "application/json"}, method="POST")
    open_request = (urllib.request.build_opener(urllib.request.ProxyHandler({})).open
                    if config.get("proxy") == "direct" else urllib.request.urlopen)
    try:
        with open_request(request, timeout=config["timeout"]) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        # Never log provider error bodies, which may echo request/auth data.
        code, delay, indicators = None, 0, []
        try:
            raw = exc.read(16384).decode('utf-8', errors='replace')
            # Only fixed labels leave this scope, never arbitrary provider text.
            lower = raw.lower()
            for label, phrases in (
                ('quota', ('insufficient_quota', 'exceeded your current quota', 'usage_limit_reached',
                           '余额不足', '额度不足')),
                ('billing', ('billing_hard_limit', 'billing limit')),
                ('rate_limit', ('rate_limit_exceeded', 'rate limit', 'too many requests')),
                ('capacity', ('overloaded', 'capacity')),
                ('edge_challenge', ('cf-mitigated', 'cloudflare', 'just a moment'))):
                if any(phrase in lower for phrase in phrases):
                    indicators.append(label)
            if 'insufficient' in lower and ('pre-deduct' in lower or 'balance' in lower):
                if 'quota' not in indicators: indicators.append('quota')
            detail = json.loads(raw).get('error', {})
            candidate = detail.get('code') if isinstance(detail, dict) else None
            if candidate in ('rate_limit_exceeded', 'insufficient_quota', 'billing_hard_limit_reached'):
                code = candidate
            if code is None and 'quota' in indicators:
                code = 'insufficient_quota'
        except (ValueError, AttributeError, OSError):
            pass
        try:
            delay = float(exc.headers.get('Retry-After', 0))
            delay = min(300, max(0, delay)) if math.isfinite(delay) else 0
        except (TypeError, ValueError, AttributeError):
            pass
        raise ModelHTTPError(exc.code, code, delay, indicators) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError("Model connection failed or timed out") from None
    if "error" in data:
        raise RuntimeError("Provider returned an error")
    return data


def normalized_usage(result):
    """Keep provider cost/details while exposing the counters used by run reports."""
    usage = dict(result.get('usage') or {})
    for source, target in (('inputTokens', 'input_tokens'), ('outputTokens', 'output_tokens')):
        if source in usage:
            usage.setdefault(target, usage[source])
    return usage


def choice_confidence(answer):
    value = answer.get('confidence')
    if value is None:
        # The selected probability is not the provider's concentration score.
        # Missing confidence cannot satisfy a threshold calibrated for that score.
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Invalid JEV confidence')
    confidence = float(value)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid JEV confidence')
    return confidence


def confidence_source(answer):
    return 'provider' if answer.get('confidence') is not None else 'missing'


def validate_plan(plan, observation):
    if not isinstance(plan, dict) or not isinstance(plan.get("objective"), str):
        raise ValueError("Planner must return an objective")
    mode = plan.get("mode", "combat" if observation["enemies"] else "explore")
    if mode not in ("combat", "explore", "collect", "survive"):
        raise ValueError("Unknown planner mode")
    door = plan.get("door_slot")
    destination=plan.get('destination_room')
    route_id=plan.get('route_id')
    route=None
    if destination is not None:
        candidates=[o for o in observation.get('navigation_options',[])
                    if not isinstance(destination,bool) and str(o['destination_room'])==str(destination)]
        if route_id is not None:
            candidates=[o for o in candidates if isinstance(route_id,str) and o.get('route_id')==route_id]
        route=candidates[0] if len(candidates)==1 else None
        if route is None or mode!='explore' or plan.get('strategy_choice') or plan.get('resource_action'):
            raise ValueError('Planner selected an unavailable travel destination')
        door=route['path'][0]['slot']
    elif route_id is not None:
        raise ValueError('Travel route requires its observed destination')
    elif ('navigation_options' in observation and mode=='explore' and door is not None
          and not plan.get('strategy_choice')):
        raise ValueError('Explore requires an observed destination, not a guessed door')
    if door is not None and str(door) not in observation["doors"]:
        raise ValueError("Planner selected an unobserved door")
    if door is not None and observation['doors'][str(door)].get('traversable') is False:
        raise ValueError('Planner selected a currently nontraversable door')
    enemy = plan.get("target_enemy")
    if enemy is not None and str(enemy) not in {str(e.get("id")) for e in observation["enemies"]}:
        raise ValueError("Planner selected an unobserved enemy")
    resource = plan.get("resource_action")
    if resource is not None and resource not in {o["kind"] for o in observation.get("resource_options", [])}:
        raise ValueError("Planner selected an unavailable resource action")
    choice = plan.get("strategy_choice")
    if choice is not None and choice not in {o["choice"] for o in observation.get("strategy_offers", [])}:
        raise ValueError("Planner selected an unavailable strategy choice")
    from .post_pickup import validate as validate_after_pickup
    after_pickup = validate_after_pickup(plan.get('after_pickup'), choice,
                                        observation.get('strategy_offers', []), resource)
    declined = plan.get('declined_pickups', [])
    pickup_choices = {o['choice'] for o in observation.get('strategy_offers', []) if o['kind']=='pickup'}
    if (not isinstance(declined,list) or any(not isinstance(c,str) or c not in pickup_choices for c in declined)
            or choice is not None and choice in declined):
        raise ValueError('Planner declined an unavailable or selected pickup')
    from .review import DEPENDENCIES
    reconsider = plan.get('decline_reconsider_on',{})
    if (not isinstance(reconsider,dict) or any(c not in declined or not isinstance(deps,list)
            or any(not isinstance(d,str) or d not in DEPENDENCIES for d in deps)
            for c,deps in reconsider.items())):
        raise ValueError('Invalid pickup reconsideration conditions')
    distance = plan.get('combat_distance', 150)
    if distance is None and mode in ('explore','collect'):
        distance=float(observation.get('stats',{}).get('range',260))*.82
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
        raise ValueError('combat_distance must be finite')
    limit = max(80, min(480, float(observation.get('stats', {}).get('range', 260)) * .88))
    distance = max(80, min(limit, distance))
    return {"objective": plan["objective"][:400], "mode": mode,
            "door_slot": None if door is None else str(door), "target_enemy": enemy,
            "resource_action": resource, "strategy_choice": choice,
            "after_pickup": after_pickup,
            "declined_pickups": list(dict.fromkeys(declined)),
            "decline_reconsider_on": {c:sorted(set(deps)) for c,deps in reconsider.items()},
            "route_contract": route,
            "rationale": str(plan.get("rationale", ""))[:700], 'combat_distance': round(distance, 1)}


def jev_enabled(models, lane):
    """Explicit ablation without disabling Sol or the local executors."""
    scope = getattr(models, 'jev_scope', 'all')
    return scope == 'all' or scope == 'tactics' and lane == 'tactic'


class Models:
    def __init__(self, config, mode="hybrid", jev_scope="all"):
        if jev_scope not in ('all', 'tactics', 'off'):
            raise ValueError('Unknown JEV scope')
        if mode == 'jev' and jev_scope != 'all':
            raise ValueError('JEV ablation requires hybrid mode')
        if mode in ("jev", "hybrid"):
            load_keys()
        self.config, self.mode, self.jev_scope = config, mode, jev_scope
        lanes = ('astra', 'jev') if mode == 'hybrid' else ('jev',) if mode == 'jev' else ()
        for lane in lanes:
            if lane == 'jev' and jev_scope == 'off':
                continue
            if not os.environ.get(config[lane]["key_env"]):
                raise ValueError(f"Set {config[lane]['key_env']} before using {mode} mode")

    async def prepare(self, observation):
        from .advance import validate_preparation
        # Advance work does not hold the control loop or queue live decisions.
        # Give it a separate bounded deadline rather than the foreground one.
        config = {**self.config['astra'], 'timeout':min(60,self.config['astra'].get('prepare_timeout',60))}
        prompt = (
            'Prepare a conditional Binding of Isaac floor/boss strategy while another controller keeps playing. '
            'Use observed floor_map and expert_knowledge; acknowledge missing geometry and unknown boss identity. '
            'Return JSON: objective (string), boss_range_fraction (0.72..0.88 of actual range), '
            'boss_focus (balanced/dangerous_adds/spawners_first), next_checks (up to six values from '
            'heal_before_boss,recharge_active,preserve_key,preserve_bomb,compare_deal,review_secret_candidates,'
            'collect_known_value,avoid_unnecessary_backtracking), resource_reasoning (short string), '
            'uncertainties (short string). Weigh health, charge, build synergy and travel cost. '
            'red_hearts, max_hearts and soul_hearts use half-heart units: 8 means four hearts. '
            'Your output is preparation, not permission to buy, bomb, use an item, select an enemy or enter a door. '
            'Resource counts and map edges may change before this is used; write conditional recommendations. '
            'Prefer consistent survival; do not assume a speedrun objective.')
        body = {'model':config['model'], 'messages':[{'role':'system','content':prompt},
                {'role':'user','content':json.dumps(observation,separators=(',',':'))}],
                'response_format':{'type':'json_object'}}
        body.update(completion_settings(config))
        started = time.monotonic()
        result = await asyncio.to_thread(post,config,body)
        meta={'latency_ms':round((time.monotonic()-started)*1000),
              'model':result.get('model',config['model']), 'usage':result.get('usage',{})}
        try:
            if result['choices'][0].get('finish_reason') == 'length':
                raise ValueError('Preparation output reached configured token limit')
            plan = validate_preparation(json.loads(result['choices'][0]['message']['content']))
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            raise PlanValidationError(str(exc),meta) from None
        return plan, meta

    async def plan(self, observation):
        config = self.config["astra"]
        prompt = (
            "Plan a single-player Binding of Isaac run across floors. Use only observed state and explored rooms. "
            "Keep the player alive, clear enemies, collect useful free resources, explore toward treasure and boss rooms. "
            "Consult expert_knowledge and the observed floor_map for resource opportunity costs, frontier coverage, "
            "Bomb offers separate bomb_cost (ordinary stock consumed) from bomb_limit (maximum placements). "
            "Golden payment may cost zero stock but never authorizes extra placements or unsafe explosions. "
            "backtracking and uncertain secret hypotheses. Missing geometry is unknown. Advance_preparation is "
            "conditional advice from an earlier state: use current counters and contracts, particularly after healing "
            "or purchases. A remote map candidate authorizes no bomb; select only present strategy_offers. "
            "Return JSON with objective (short string), mode (combat/explore/collect/survive), "
            "door_slot (observed door key or null), target_enemy (observed enemy id or null), "
            "For travel return destination_room and the exact route_id from ONE navigation_options entry, "
            "with mode=explore and strategy_choice=null. Several routes may reach the same destination: compare "
            "hops and keys_required against current key reserves, known treasure/shop needs and backtracking. "
            "Hops count room transitions, not walking distance or danger; intermediate rooms were observed clear "
            "but their hazards and geometry still require local checks. "
            "The code derives doors from its observed path and keeps the destination across cleared rooms; "
            "set door_slot=null instead of calculating directions. The listed keys_required is the maximum authorized "
            "key budget for that route. New consequential offers pause travel for review. "
            "Never infer a treasure room from earlier commentary: only current navigation_options/floor_map establish "
            "a destination. If none is observed, select a listed frontier to look for one. Old rationales are not map facts. "
            "resource_access_last_seen reports actual local pickup approach checks. no_local_route and "
            "endpoint_blocked are NOT free collectible loot; nearby_solid_types describe nearby terrain, "
            "not a proven bomb solution. A resource_access_review destination needs a concrete changed approach "
            "(an available flight/teleport/terrain tool or a currently offered access contract), not repeated walking. "
            "Otherwise leave it pending and choose another useful destination. A found route is not acquisition. "
            "For non-travel decisions return destination_room=null and route_id=null. "
            "resource_action (kind from resource_options or null). Heal before critical health; use charged combat tools. "
            "Also return strategy_choice (exact choice string from strategy_offers, or null). This is executed, not just advice. "
            "Return declined_pickups (array, default []): exact pickup choice IDs you have evaluated and decided to leave "
            "in this room for the current build/resources. This prevents repeating the same comparison after a sweep; "
            "By default it expires when health, inventory, build or ground resources change. Optionally return "
            "decline_reconsider_on: an object mapping declined choice IDs to arrays chosen from health, coins, keys, "
            "bombs, consumables, charge, ground, doors. Select only factors that could reverse THIS rejection; "
            "build, held items, identity, price, options and floor always revalidate. Empty array means only these "
            "mandatory factors. A free item rejected for explosive self-damage might use [health,bombs]; an unknown "
            "pill kept on the ground because the held pill is more useful might use [health,consumables]. "
            "Never omit a resource or ground factor when its availability could reverse the decision. Code also "
            "forces currency/health checks for priced trades and ground checks for mutually exclusive options. "
            "Valid deferred_pickups persist across reconnects; don't return to re-review them without changed conditions. "
            "Include a rejected trinket replacement "
            "explicitly even when choosing scavenge. Never decline the selected choice, unknown unevaluated offers, "
            "doors, deals, donation, descent or bomb contracts. Empty is valid when undecided. "
            "Compare offered items using their names, descriptions, held collectibles and active item; reason about synergies, "
            "Read pickup_type: trinkets are not collectible pedestals or active items. Compare their effects against held_trinket_mechanics; "
            "a free replacement is not automatically an upgrade. Keep the current trinket if there is no clear benefit. "
            "Curse of the Blind concerns collectible pedestals; do not erase the supplied identity of ordinary trinkets. "
            "Return rationale (at most 70 words): for a mutually exclusive choice, compare the selected item against the other "
            "options using the provided mechanical descriptions and this build. Do not select from an empty or missing snapshot. "
            "active-item replacement, mutually exclusive choices, coin prices and opportunity cost. Skip bad trades by choosing "
            "mode=explore and strategy_choice=null. For blood donation evaluate spare red health, available healing, coin needs "
            "and observed shops: select donation only if worthwhile within the offered health budget. A donate_cycle offer "
            "authorizes its bounded donate-withdraw-recover loop using specified free red hearts; no new Sol call is needed "
            "inside that contract. Other donation offers authorize one contact only. Reassess after completion. "
            "Do not assume unseen shop stock, unknown item effects, guaranteed donation payout or hidden pill effects. "
            "Use held_consumable_mechanics to compare keeping a card or identified pill against replacing/using it. "
            "These are base descriptions, not extra action permissions; only current resource_options authorize use. "
            "A model_only consumable option is a real one-use action, not a recommendation to use it. "
            "Explicitly compare using it now with keeping it: consider resource generation, map information, "
            "teleportation, health costs, rerolls, duplication, combat timing and held modifiers. "
            "Choose resource_action=card or pill to use that held slot, or null to keep it. "
            "When selecting a ground card/rune or pill, also return after_pickup='keep' or 'use' "
            "to decide what to do after that exact pickup is confirmed in inventory. Return null otherwise. "
            "Do not combine after_pickup with resource_action: the latter uses the currently held item. "
            "A changed room, health, build, resources beyond the listed purchase cost, target, "
            "or unavailable use condition cancels the follow-up. "
            "Use 'keep' when collecting a combat card for a later fight; 'use' when its effect is useful now. "
            "If a resource-generating card solves a current shortage, evaluate using it in this room rather than "
            "merely saying to keep it for its benefit. Effects do not guarantee profit or safety. "
            "For pickup/donation choose mode=collect and the offer; for room travel choose mode=explore and destination_room. "
            "An offered descend choice enters the next floor through an observed open boss trapdoor. "
            "Take worthwhile boss rewards first and consider unexplored treasure rooms before leaving the floor. "
            "TNT (terrain type 12) and explosive rocks (type 5) can explode from shots and chain-react. "
            "With inventory.poop_explosions or Brown Cap, intact poop also explodes. Scavenge offers report "
            "explosive_sources; compare their risk and use only the checked ranged approach, never assume ordinary poop is harmless. "
            "Keep blast distance before firing; avoid spikes (8,9,25), pits without flight, red poop and curse doors. "
            "After damage reassess hazards and prioritize survival. Never invent pill effects; follow resource_policy. "
            "Door selection should avoid revisiting rooms unless needed to reach an unexplored branch. "
            "Use target_room_name and traversable on doors; choose only traversable doors. "
            "Secret rooms are not boss rooms. A discovered floor_exit stays known while its opening animation plays. "
            "Visited treasure rooms may still contain uncollected items: check floor_map.nodes.remaining_items. "
            "The local executor can shoot ordinary fires or poop blocking a selected pickup before approaching it. "
            "A scavenge strategy offer authorizes a bounded sweep of ordinary fires and brown poop for possible resources; "
            "choose mode=collect and strategy_choice=scavenge when its time/risk is worthwhile. "
            "Use economy.observed_shop_items to weigh coin shortages, healing and resource reserves. "
            "Consult room_knowledge for drop chances and room mechanics. A chance is not a guaranteed reward. "
            "Do not instruct wall bombing or heart-price purchases when economy reports those actions unavailable. "
            "A secret_bomb offer authorizes one bomb at a checked wall, retreat, verification and entry only if a secret "
            "door opens. Compare observed neighboring rooms, unknown cells, bomb reserve and remaining floor value. "
            "The hypothesis is uncertain; skip weak searches rather than treating them as guaranteed rewards. "
            "A rock_bomb offer authorizes the stated one/two-bomb budget for an observed marked rock, "
            "with retreat and destruction confirmation. Compare soul-heart protection before bosses, "
            "remaining bomb uses and travel cost; do not assume its drops. Full red health does not "
            "remove the value of soul hearts. Marked rocks remembered elsewhere require revisiting. "
            "A resource_rock choice is a separate one-bomb contract for ordinary rock: expected_resources are "
            "already visible loot predicted reachable after removing that one rock, not guaranteed earnings. "
            "Compare bomb_cost against useful loot (including bomb subtype quantity), missing health and reserves. "
            "It never authorizes a second blast or extra rock targets. If ordinary_rock_access_protocol is true, "
            "a remote resource_access_review room can be revisited to inspect a concrete contract, with travel cost; "
            "a previous failed inspection is not a reason to repeat without changed conditions. "
            "Consult run_history and retrieved_memories; history marked incomplete is unknown, not evidence of no past deal. "
            "Use run_history.deal_evidence for typed doors, entries and acquisitions. Its completeness can be weaker "
            "than the general run ledger after migration. A skip decision is not proof the room stayed unentered. "
            "For deal_enter/deal_skip offers explicitly select the desired contract. Skipping means never entering this offer; "
            "merely entering the FIRST DEVIL room without buying loses its skip-entry Angel replacement in Repentance. "
            "That rule does not apply to entering or skipping an Angel Room; the replacement concerns the next "
            "successful deal offer, not a guaranteed door on the next floor. Free Devil-room loot is not itself "
            "a health-paid Devil deal. Do not infer engine eligibility from an item-pool name or ordinary shop purchase. "
            "Coin-priced Devil purchases (Keeper/A Pound of Flesh) and Satanic Bible health-priced Boss rewards "
            "can still affect Angel eligibility; inspect observed context and held-item mechanics, not just price sign. "
            "Compare the permanent health-container price with build improvement and Angel opportunity cost. "
            "red_hearts, max_hearts and soul_hearts are half-heart units (8 means four hearts). "
            "Consult run_history.recent_decisions before replacing an active: distinguish model intent from "
            "observed acquisition. Preserve a recent preference unless new evidence or a stated correction "
            "justifies reversing it. Explain that change in rationale; avoid swapping the same pair back and forth. "
            "For heart-price pickups use health_contract remaining health; do not rely on unobserved healing or revives. "
            "Take one paid item then reassess changed prices and health. Angel free alternatives may be mutually exclusive. "
            "Paid normal shop purchases do not imply a Devil deal; distinguish the observed room and price. "
            "Memory scores express relevance, not truth or action permission; local live contracts remain authoritative. "
            "Combat observations include enemy names and current geometry. Consider eliminating enemies that continuously "
            "spawn others when reachable, instead of indefinitely fighting their spawned units. "
            "Consult enemy_knowledge when available; distinguish ongoing summons from enemies released only on death. "
            "Return combat_distance (80-480 world units, clamped to observed tear range): desired fighting separation. "
            "Choose more space for abrupt charges or dangerous contact while keeping targets within actual weapon range. "
            "Empty pedestals are obstacles with no item to collect; continue exploring when pickups are empty. "
            "No console commands, invented entities, or coordinates. Low-level movement is handled separately."
            " Consult decision_knowledge for pickup contracts, heart economy and weapon-specific caveats. "
            "Evaluate compatibility with the current executor: it has approximate straight-shot geometry, "
            "not complete charged, explosive or arcing-weapon physics. Favor survival and reliable execution "
            "when comparing a defensive item with a weapon replacement; do not rank only by listed damage."
            " For ordinary tears start near 80-85 percent of observed range; keep room to sidestep and turn. "
            "Use enemy_knowledge and its coverage report; unknown names are not verified mechanics. "
            "A close dangerous add can temporarily override your target. Do not chase airborne jumpers; "
            "clear nearby pursuers or a reachable recurring spawner before restoring boss focus."
        )
        from .planning import model_observation
        sent_observation = model_observation(observation)
        body = {"model": config["model"], "messages": [{"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(sent_observation, separators=(",", ":"))}],
                "response_format": {"type": "json_object"}}
        body.update(completion_settings(config))
        started = time.monotonic()
        result = await asyncio.to_thread(post, config, body)
        metadata={"latency_ms": round((time.monotonic() - started) * 1000),
                  "model": result.get("model", config["model"]), "usage": result.get("usage", {})}
        try:
            if result['choices'][0].get('finish_reason') == 'length':
                raise ValueError('Planner output reached configured token limit')
            plan = validate_plan(json.loads(result["choices"][0]["message"]["content"]), observation)
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            # Preserve metering for a billed response even when no action can
            # be accepted. Do not log raw model text or HTTP error bodies.
            raise PlanValidationError(str(exc)[:200],metadata) from None
        return plan, metadata

    async def decide(self, observation, plan, options):
        if not jev_enabled(self, 'action'):
            raise ValueError('JEV action lane is disabled')
        from .knowledge import tactical_knowledge
        config = self.config["jev"]
        criteria = {o["action"].id: o["description"] for o in options}
        # The tactical model needs local motion/hazards, not the floor map or
        # every offered item's description on each subsecond request.
        tactical = {k: observation.get(k) for k in ('frame', 'room', 'player', 'stats', 'health',
                    'enemies', 'nonblocking_entities', 'projectiles', 'bombs', 'fires', 'lasers')}
        position = observation.get('player', {}).get('pos', {})
        px, py = position.get('x', 0), position.get('y', 0)
        tactical['nearby_terrain'] = [cell for cell in observation.get('terrain', [])
                                     if math.hypot(cell.get('x', 0)-px, cell.get('y', 0)-py) < 220][:40]
        tactical['knowledge'] = tactical_knowledge(observation)
        tactical['combat_positioning'] = observation.get('combat_positioning')
        tactical['unknown_enemy_types'] = observation.get('enemy_knowledge_coverage', {}).get('unknown_types', [])
        for key in ('enemies', 'nonblocking_entities'):
            tactical[key] = [{k: v for k, v in e.items() if k in
                ('id', 'type', 'variant', 'subtype', 'name', 'pos', 'vel', 'hp', 'is_boss',
                 'is_champion', 'is_vulnerable', 'collision_radius', 'state', 'state_frame')}
                for e in observation.get(key, [])]
        guidance = {k: plan.get(k) for k in ('objective', 'mode', 'target_enemy', 'navigation_target', 'combat_distance')}
        body = {"model": config["model"],
                "state": json.dumps({"observation": tactical, "plan": guidance}, separators=(",", ":")),
                "questions": {"action": {"type": "choice", "instructions":
                    "Choose one listed movement/shooting input for the next short interval. Avoid incoming projectiles, "
                    "contact damage and terrain. Do not shoot nearby TNT or enter its blast radius. "
                    "Follow the route waypoint; a detour can initially lead away from the final goal. "
                    "Prefer high progress_score among comparably safe options. Standing still retains momentum. "
                    "Use combat_positioning and candidate distance_band: hold a clear shot from range, "
                    "not maximum proximity. Prefer spacing from the nearest enemy, including adds. "
                    "Do not chase jumpers while airborne; keep lateral escape space. Immediate hazards override spacing. "
                    "Risk is a geometric heuristic, not certainty.",
                    "criteria": criteria}}}
        started = time.monotonic()
        result = await asyncio.to_thread(post, config, body)
        answer = result.get("answers", {}).get("action", {})
        chosen = answer.get("choice")
        if chosen not in criteria:
            raise ValueError("JEV returned an action outside its candidate set")
        confidence = choice_confidence(answer)
        action = next(o["action"] for o in options if o["action"].id == chosen)
        return action, {"latency_ms": round((time.monotonic() - started) * 1000),
                        "confidence": confidence, "probabilities": answer.get("probabilities", {}),
                        "confidence_source": confidence_source(answer),
                        "usage": normalized_usage(result), "model": result.get("model", config["model"])}

    async def choose_tactic(self, observation, plan, targets):
        if not jev_enabled(self, 'tactic'):
            raise ValueError('JEV tactic lane is disabled')
        from .tactics import POSTURES, compact_state
        config=self.config['jev']
        # All current facts live once in state.targets; do not duplicate each
        # expanded entity JSON in both state and choice criteria.
        choices={t['choice']:f"Enemy {t['choice']} in state.targets" for t in targets}
        state=compact_state(observation,plan,targets)
        body={'model':config['model'],'state':json.dumps(state,separators=(',',':')),
              'questions':{'target':{'type':'choice','instructions':'Favor close threatening chasers, recurring spawners or finishable enemies with an accessible firing lane. Check vulnerability and alignment; do not chase airborne enemies. If all are invulnerable, choose a spacing reference.', 'criteria':choices},
                           'posture':{'type':'choice','instructions':'Choose a short positioning preference using target motion, terrain and hazards. Keep escape space at walls. Capped observations cannot establish a safe route; local checks decide each input.', 'criteria':POSTURES}}}
        started=time.monotonic();result=await asyncio.to_thread(post,config,body)
        answers=result.get('answers',{});a,b=answers.get('target',{}),answers.get('posture',{})
        if a.get('choice') not in choices or b.get('choice') not in POSTURES:
            raise ValueError('JEV tactic outside supplied choices')
        probabilities={k:choice_confidence(v) for k,v in (('target',a),('posture',b))}
        if any(not math.isfinite(v) or not 0<=v<=1 for v in probabilities.values()):
            raise ValueError('Invalid tactic confidence')
        return {'target_enemy':a['choice'],'posture':b['choice']},{
            'latency_ms':round((time.monotonic()-started)*1000),'confidence_by_part':probabilities,
            'confidence_source_by_part':{k:confidence_source(v) for k,v in (('target',a),('posture',b))},
            'usage':normalized_usage(result),'model':result.get('model',config['model'])}

    async def choose_goal(self, observation, choices):
        if not jev_enabled(self, 'goal'):
            raise ValueError('JEV goal lane is disabled')
        from .fast_policy import compact_goal_observation
        config = self.config['jev']
        criteria = {o['choice']: f"Execute the {o['kind']} offer with this choice ID in state.fast_offers."
                    for o in choices}
        criteria['defer'] = 'Ask Sol: unfamiliar mechanics, build tradeoff, reroll value or uncertainty.'
        if all(o['kind'] == 'scavenge' for o in choices):
            criteria['skip'] = 'Skip this room sweep if its time/risk exceeds the likely resource value.'
        body = {'model': config['model'],
                'state': json.dumps(compact_goal_observation(observation, choices), separators=(',', ':')),
                'questions': {'goal': {'type': 'choice', 'instructions':
                    'Choose a routine interaction. A single known free stat/health passive normally merits taking. '
                    'An offered ordinary pressure_plate is required to finish an enemy-free trap room; choose a reachable plate to disarm hazards and unlock progress. '
                    'Respect the actual build and provided mechanics; contact-damage items do not grant immunity. '
                    'For scavenge, compare missing health/coins and observed stock with time; drops are not guaranteed. '
                    'Choose defer for a real tradeoff. Output only a listed choice; local controls execute the goal.',
                    'criteria': criteria}}}
        started = time.monotonic()
        result = await asyncio.to_thread(post, config, body)
        answer = result.get('answers', {}).get('goal', {})
        choice, confidence = answer.get('choice'), choice_confidence(answer)
        if choice not in criteria or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid JEV goal choice')
        return choice, {'latency_ms': round((time.monotonic()-started)*1000), 'confidence': confidence,
                        'confidence_source': confidence_source(answer),
                        'probabilities': answer.get('probabilities', {}), 'usage': normalized_usage(result),
                        'model': result.get('model', config['model'])}

    async def retrieve_memories(self, observation, candidates):
        """One bounded typed retrieval round; never grants action authorization."""
        if not jev_enabled(self, 'memory'):
            raise ValueError('JEV memory lane is disabled')
        config = self.config['jev']
        candidates = candidates[:8]
        query = {k: observation.get(k) for k in ('room_info','health','inventory','run_history')}
        query['enemy_types'] = sorted({e.get('type', -1) for e in observation.get('enemies', [])})
        query['choices'] = [o['choice'] for o in observation.get('strategy_offers', [])]
        questions = {
            'view': {'type': 'choice', 'instructions': 'Which evidence relation most helps this immediate game decision?',
                'criteria': {'temporal': 'Order of previous deal entry/skipping or choices matters',
                             'entity': 'Same enemy/item experience matters',
                             'outcome': 'Previous observed action outcomes matter'}},
            'budget': {'type':'choice', 'instructions':'Choose a small evidence budget for the current decision.',
                'criteria': {'2':'Two direct facts suffice', '4':'Several related events are useful'}},
            'sufficient': {'type':'noul', 'instructions':'Does supplied observed evidence suffice to inform the decision without guessing prior deal history or causation?',
                'criteria': {'true':'Required history and facts are observed', 'false':'Important history or mechanism is unknown'}}}
        for node in candidates:
            questions['m'+node['id']] = {'type':'noul',
                'instructions':f"Is memory id={node['id']} relevant to the current choice? Temporal adjacency is not proof of cause. Prefer actual observed outcomes over generic room visits.",
                'criteria': {'true':'Direct useful evidence', 'false':'Irrelevant or redundant'}}
        body = {'model':config['model'], 'state':json.dumps({'query':query,'candidates':candidates},separators=(',',':')),
                'questions':questions}
        started = time.monotonic()
        result = await asyncio.wait_for(asyncio.to_thread(post, config, body), 2)
        answers = result.get('answers', {})
        def probability(key):
            value = answers.get(key, {}).get('noul')
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid memory probability')
            return value
        budget = answers.get('budget',{}).get('choice')
        view = answers.get('view',{}).get('choice')
        if budget not in ('2','4') or view not in ('temporal','entity','outcome'):
            raise ValueError('Invalid memory retrieval decision')
        scores = {n['id']:probability('m'+n['id']) for n in candidates}
        selected = []
        for node in sorted(candidates,key=lambda n:scores[n['id']],reverse=True):
            if scores[node['id']] < .55 or len(selected) >= int(budget):
                continue
            if len(json.dumps(selected+[node])) <= 2400:
                selected.append(node)
        return selected, {'selected_ids':[n['id'] for n in selected], 'scores':scores, 'view':view,
            'evidence_sufficient': probability('sufficient'), 'budget_nodes':int(budget),
            'stop_reason':'single_round_budget', 'latency_ms':round((time.monotonic()-started)*1000),
            'model':result.get('model',config['model']), 'usage':normalized_usage(result)}
