"""Public execution metadata; never serialize raw configuration or credentials."""
import hashlib
import json
import math
import platform
from urllib.parse import urlsplit

MODEL_FIELDS=('model','timeout','prepare_timeout','proxy','max_completion_tokens','reasoning_effort')
EXECUTION_FIELDS=('command','mode','seconds','observe_only','until_run_end','stop_after_floor',
                  'pause_on_stop','port','stop_file_enabled','jev_scope','isolated_memory',
                  'evaluation_batch','evaluation_case')
KNOWN_API_PATHS={'/v1/chat/completions','/v1/responses','/api/alpha/decisions','/v1/systemone'}
CAPABILITY_FIELDS=('protocol','safety_version','rock_bomb_protocol','secret_bomb_protocol','golden_bomb_protocol','repentance_plus')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def scalar(value):
    return isinstance(value,(str,bool,int)) or value is None or isinstance(value,float) and math.isfinite(value)


def public_models(config):
    result={}
    for lane in ('astra','jev'):
        if lane not in config:continue
        source=config[lane]
        result[lane]={k:source[k] for k in MODEL_FIELDS if k in source and scalar(source[k])}
        # Origin distinguishes relays/providers. Userinfo, custom paths, query and
        # fragment can contain secrets; none are stored or hashed.
        try:
            url=urlsplit(source.get('url',''))
            host=url.hostname
            if url.scheme in ('http','https') and host:
                host='['+host+']' if ':' in host else host
                result[lane]['endpoint_origin']=f'{url.scheme}://{host}'+(f':{url.port}' if url.port else '')
                if url.path in KNOWN_API_PATHS:result[lane]['standard_api_path']=url.path
        except ValueError:
            pass
    return result


def build_manifest(source_manifest,config,execution):
    identity=dict(schema_version=1,python_version=platform.python_version(),platform=platform.system(),
                  package_source_fingerprint=fingerprint(source_manifest),models=public_models(config),
                  execution={k:execution[k] for k in EXECUTION_FIELDS if k in execution and scalar(execution[k])})
    return dict(identity=identity,fingerprint=fingerprint(identity),
                limitations='Requested model/settings and recorded package hashes only. '
                    'Resolved model versions remain in response events. Custom API URL paths and credentials are excluded. '
                    'This does not identify the game binary, installed Lua/mods, graphics settings, '
                    'recorder settings or complete persisted game/controller memory.')


def observed_environment(world, wire_agent=None):
    agent=world.capabilities if wire_agent is None else wire_agent
    return dict(level=world.level,
        capabilities={k:agent[k] for k in CAPABILITY_FIELDS
                      if k in agent and scalar(agent[k])},
        difficulty=world.info.get('difficulty'),player_type=world.stats.get('player_type'),
        continued_run=agent.get('continued_run'),
        scope='Observed bridge fields; missing values are unknown, not inferred from configuration.')
