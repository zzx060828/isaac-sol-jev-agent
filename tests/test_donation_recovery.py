import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,patch

from isaac_agent.donation import cycle_contract
from isaac_agent.state import World
from isaac_agent.strategy import offers,StrategyExecutor
from isaac_agent.runtime import Controller
from test_agent import MemoryLog
from test_donation_cycle import setup
from test_safety_strategy import refresh


class DonationRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def test_enclosed_heart_is_not_recovery_credit(self):
        w=World();w.ingest(json.loads((Path(__file__).parent/'fixtures/enclosed-key.json').read_text()))
        w.capabilities['repentance_plus']=False;w.stats['player_type']=1
        w.health.update(red_hearts=8,max_hearts=8)
        w.pickups[0].update(variant=10,sub_type=1)
        refresh(w)
        self.assertIsNone(cycle_contract(w))

    def test_binding_is_stable_for_reordering_and_motion_but_changes_for_recovery_terms(self):
        w,plan=setup();key=plan['strategy_choice']
        w.payload['PICKUPS'].reverse();w.pickups[0]['pos']['y']+=1
        self.assertEqual(next(o['choice'] for o in offers(w) if o['kind']=='donate_cycle'),key)
        w.pickups[0]['sub_type']=2
        changed=next(o['choice'] for o in offers(w) if o['kind']=='donate_cycle')
        self.assertNotEqual(changed,key)
        w.payload['PICKUPS']=[]
        plain=next(o for o in offers(w) if o['kind']=='donate')
        self.assertNotEqual(plain['choice'],key)

    async def test_disappearing_hearts_during_sol_cannot_downgrade_to_single_payment(self):
        w,plan=setup();obs=w.observation();sig=w.strategy_signature()
        async def answer(observation):
            w.payload['PICKUPS']=[];refresh(w,1)
            return dict(plan,objective='Donate then recover'),{}
        c=Controller(SimpleNamespace(mode='hybrid',plan=answer),MemoryLog())
        try:
            await c._plan(w,obs,w.token,w.revision)
            self.assertEqual(w.strategy_signature(),sig) # Free hearts intentionally don't invalidate all plans.
            self.assertIsNone(c.plan.get('strategy_choice'))
            self.assertEqual(c.log.rows[-1]['reason'],'selected_offer_unavailable')
            self.assertTrue(any(o['kind']=='donate' for o in offers(w)))
        finally:await c.close()

    async def test_changed_contract_after_reply_resolves_and_allows_replanning(self):
        w,plan=setup();c=Controller(SimpleNamespace(mode='hybrid',plan=AsyncMock()),MemoryLog())
        c.reset_actions(w);c.plan=dict(plan);c.last_plan=10**20
        w.payload['PICKUPS']=[];refresh(w,1)
        try:
            command=c.step(w)
            self.assertIsNone(c.plan.get('strategy_choice'))
            self.assertIsNone(c.strategy.donation)
            self.assertFalse(c.execution_plan.get('navigation_contact'))
            self.assertIn(plan['strategy_choice'],c.strategy.finished)
            self.assertEqual(c.plan_pending,'goal_resolved')
            self.assertFalse(command['use_bomb'])
        finally:await c.close()

    def test_blocked_or_stale_hearts_do_not_fund_cycle(self):
        w,_=setup()
        with patch('isaac_agent.pickup_progress.blocked_choices',return_value={f"pickup:{h['id']}:10:1:0" for h in w.pickups}):
            self.assertIsNone(cycle_contract(w))
        for channel in ('PICKUPS','ROOM_LAYOUT','FIRE_HAZARDS','PLAYER_STATS'):
            v=deepcopy(w);v.sampled[channel]=v.frame-7
            self.assertIsNone(cycle_contract(v),channel)

    def test_path_loss_before_contact_stops_existing_cycle(self):
        w,plan=setup();e=StrategyExecutor();log=MemoryLog()
        self.assertTrue(e.update(w,plan,log)['navigation_contact'])
        # Same IDs remain, but a new solid block occupies each recovery target.
        for i,h in enumerate(w.pickups):
            w.layout['grid'][str(900+i)]=dict(type=2,collision=3,variant=0,x=h['pos']['x'],y=h['pos']['y'])
        w.navigation_revision+=1;w.path_cache.clear();refresh(w,1)
        result=e.update(w,plan,log)
        self.assertFalse(result['navigation_contact'])
        self.assertIsNone(e.donation)
        self.assertEqual(log.rows[-1]['result'],'recovery_unreachable_before_payment')
        self.assertEqual(w.health['red_hearts'],8)

    def test_newly_spawned_heart_does_not_expand_active_contract(self):
        w,plan=setup();e=StrategyExecutor();log=MemoryLog()
        e.update(w,plan,log)
        w.payload['PICKUPS']=[dict(w.pickups[0],id=999)];refresh(w,1)
        result=e.update(w,plan,log)
        self.assertFalse(result['navigation_contact'])
        self.assertIsNone(e.donation)
        self.assertEqual(log.rows[-1]['result'],'recovery_unreachable_before_payment')

    def test_same_id_with_different_healing_cannot_extend_active_terms(self):
        w,plan=setup();e=StrategyExecutor();log=MemoryLog();e.update(w,plan,log)
        for h in w.pickups:h['sub_type']=2
        refresh(w,1)
        result=e.update(w,plan,log)
        self.assertFalse(result['navigation_contact'])
        self.assertEqual(log.rows[-1]['result'],'recovery_unreachable_before_payment')
