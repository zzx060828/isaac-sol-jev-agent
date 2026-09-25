"""Carry a typed consumable decision through one observed pickup transition."""
from copy import deepcopy
from .resources import resource_options, resource_signature


def validate(action, choice, offers, resource_action=None):
    if action is None:
        return None
    if action not in ('keep', 'use') or resource_action is not None:
        raise ValueError('Invalid after_pickup action')
    offer=next((o for o in offers if o['choice']==choice),None)
    if (not offer or offer['kind']!='pickup'
            or offer['entity'].get('variant') not in (70,300)):
        raise ValueError('after_pickup requires a selected pill/card pickup')
    return action


def context(inventory, health):
    return (deepcopy(inventory.get('collectibles',{})),
            inventory.get('trinket_0',0),inventory.get('trinket_1',0),
            tuple(inventory.get(k,0) for k in ('coins','keys','bombs')),
            tuple(health.get(k,0) for k in ('red_hearts','max_hearts','soul_hearts',
                  'bone_hearts','broken_hearts','rotten_hearts','eternal_hearts')))


class PostPickup:
    def __init__(self):
        self.pending=None
        self.kept=None

    def arm(self,world,observation,plan):
        self.pending=None
        self.kept=None
        action=plan.get('after_pickup')
        if action is None:
            return
        validate(action,plan.get('strategy_choice'),observation.get('strategy_offers',[]),
                 plan.get('resource_action'))
        offer=next(o for o in observation['strategy_offers'] if o['choice']==plan['strategy_choice'])
        entity=offer['entity'];kind='card' if entity['variant']==300 else 'pill'
        inv=observation['inventory'];expected=entity['sub_type']
        # Holding an identical item cannot prove that this target was acquired.
        if inv.get(kind+'_0',0)==expected:
            return
        expected_inventory=deepcopy(inv)
        expected_inventory['coins']=inv.get('coins',0)-max(0,entity.get('price',0))
        self.pending=dict(action=action,choice=offer['choice'],entity=deepcopy(entity),
            kind=kind,expected=expected,scope=world.token,start=world.frame,
            source_context=context(inv,observation.get('health',{})),
            acquired_context=context(expected_inventory,observation.get('health',{})),
            disappeared=None)

    def update(self,world,log):
        """Return one confirmed decision, or wait/cancel without authorizing use."""
        p=self.pending
        if p is None:return None
        if (world.token==p['scope'] and world.fresh('PICKUPS',6)
                and world.sampled.get('PICKUPS',-1)>p['start']
                and not any(e.get('id')==p['entity']['id'] for e in world.pickups)
                and p['disappeared'] is None):
            p['disappeared']=world.frame
        reason=None
        if world.token!=p['scope'] or world.dead:reason='room_or_run_changed'
        elif world.frame-p['start']>240:reason='acquisition_timeout'
        elif p['disappeared'] is not None and world.frame-p['disappeared']>=30:
            reason='slot_confirmation_timeout'
        elif not all(world.fresh(k,6) and world.sampled.get(k,-1)>p['start']
                     for k in ('PICKUPS','PLAYER_INVENTORY','PLAYER_HEALTH')):
            return None
        elif context(world.inventory,world.health) not in (p['source_context'],p['acquired_context']):
            reason='health_build_or_resources_changed'
        else:
            target=next((e for e in world.pickups if e.get('id')==p['entity']['id']),None)
            if target is not None:
                if any(target.get(k,0)!=p['entity'].get(k,0)
                       for k in ('variant','sub_type','price','options_index')):
                    reason='target_changed'
                else:return None
            else:
                if p['disappeared'] is None:p['disappeared']=world.frame
                if world.inventory.get(p['kind']+'_0',0)==p['expected']:
                    if p['action']=='use' and (world.inventory.get('can_use',True) is False
                            or not all(world.fresh(k,6) for k in ('ROOM_INFO','ROOM_LAYOUT',
                                'ENEMIES','PROJECTILES','BOMBS','FIRE_HAZARDS'))):
                        return None  # Pickup animation/sensor lag; retain the bounded intent.
                    if context(world.inventory,world.health)!=p['acquired_context']:
                        reason='purchase_cost_not_confirmed'
                    elif p['action']=='use' and not any(o['kind']==p['kind'] and o['id']==p['expected']
                                                     for o in resource_options(world)):
                        reason='use_conditions_changed'
                    else:
                        self.pending=None
                        if p['action']=='keep':
                            self.kept=dict(scope=world.token,kind=p['kind'],
                                signature=resource_signature(world.inventory,p['kind']),
                                context=context(world.inventory,world.health))
                        log.write('after_pickup_confirmed',frame=world.frame,choice=p['choice'],
                                  action=p['action'],kind=p['kind'],id=p['expected'])
                        return {'action':p['action'],'kind':p['kind'],'id':p['expected']}
                elif world.frame-p['disappeared']>=30:reason='slot_not_confirmed'
                else:return None
        if reason:
            log.write('after_pickup_discarded',frame=world.frame,choice=p['choice'],reason=reason)
            self.pending=None
        return None

    def keep_kinds(self,world):
        """Keep in this context; new room/health/build or explicit plan reopens use."""
        k=self.kept
        if k and (world.token!=k['scope'] or context(world.inventory,world.health)!=k['context']
                  or resource_signature(world.inventory,k['kind'])!=k['signature']):
            self.kept=None
        return (self.kept['kind'],) if self.kept else ()
