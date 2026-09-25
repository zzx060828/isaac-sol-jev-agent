import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from test_advance import setup
from test_agent import MemoryLog
from test_safety_strategy import refresh
from isaac_agent.control import Action, candidates
from isaac_agent.models import Models, configuration
from isaac_agent.runtime import Controller
from isaac_agent.tactics import targets, bias


META={'confidence_by_part':{'target':.9,'posture':.8},'latency_ms':1900}
INTENT={'target_enemy':'19','posture':'clockwise'}


class TacticalIntentTests(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_intent_uses_live_geometry_and_keeps_short_wire_lease(self):
        w=setup()
        async def reply(*args):
            refresh(w,60)  # Two seconds of sensor progress invalidates old directions.
            w.payload['ENEMIES'][0]['pos']={'x':320,'y':410}
            return dict(INTENT),dict(META)
        models=SimpleNamespace(mode='hybrid',choose_tactic=AsyncMock(side_effect=reply))
        c=Controller(models,MemoryLog());c.reset_actions(w)
        try:
            await c._tactic(w,w.observation(),w.token,targets(w,{}))
            self.assertIsNotNone(c.tactic)
            self.assertEqual(c.action_until,-1)  # No old movement action was leased.
            command_frame=w.frame
            command=c.step(w);await asyncio.sleep(0)
            self.assertEqual(c.execution_plan['target_enemy'],'19')
            self.assertEqual(c.execution_plan['tactical_posture'],'clockwise')
            self.assertEqual(command['agent_deadline'],command_frame+6)
            self.assertFalse(command['use_bomb'])
            self.assertTrue(any(r['event']=='input' and r['source']=='local_with_jev_tactic' for r in c.log.rows))
        finally:await c.close()

    async def test_target_death_room_change_or_expiry_discards_reply(self):
        for change in ('death','room','expiry','manual'):
            w=setup()
            async def reply(*args):
                if change=='death':w.payload['ENEMIES']=[]
                if change=='room':w.room+=1
                if change=='expiry':refresh(w,121)
                if change=='manual':w.control_mode='MANUAL'
                return dict(INTENT),dict(META)
            c=Controller(SimpleNamespace(mode='hybrid',choose_tactic=reply),MemoryLog());c.reset_actions(w)
            try:
                await c._tactic(w,w.observation(),w.token,targets(w,{}))
                self.assertIsNone(c.tactic)
                self.assertEqual(c.log.rows[-1]['event'],'tactic_discarded')
            finally:await c.close()

    async def test_accepted_intent_expires_on_target_death_and_never_spends_resources(self):
        w=setup();models=SimpleNamespace(mode='hybrid',choose_tactic=AsyncMock(return_value=(dict(INTENT),dict(META))))
        c=Controller(models,MemoryLog());c.reset_actions(w)
        try:
            await c._tactic(w,w.observation(),w.token,targets(w,{}))
            w.payload['ENEMIES']=[dict(w.payload['ENEMIES'][0],id=20)]
            c.last_decision=10**20
            command=c.step(w)
            self.assertIsNone(c.tactic)
            self.assertNotIn('tactical_posture',c.execution_plan)
            self.assertFalse(command['use_bomb'])
        finally:await c.close()

    def test_posture_is_relative_and_cannot_expand_legal_actions(self):
        w=setup()
        right_target=(500,280)
        self.assertGreater(bias(w,(320,260),right_target,'clockwise'),0)
        self.assertLess(bias(w,(320,300),right_target,'clockwise'),0)
        left_target=(140,280)
        self.assertLess(bias(w,(320,260),left_target,'clockwise'),0)
        w.payload['PROJECTILES']={'enemy_projectiles':[{'pos':{'x':320,'y':230},'vel':{'x':0,'y':9}}]}
        plain,_=candidates(w,{})
        steered,_=candidates(w,{'tactical_posture':'clockwise'})
        self.assertEqual({o['action'].id for o in plain},{o['action'].id for o in steered})

    async def test_provider_uses_typed_intents_and_validates_returned_choices(self):
        w=setup();m=Models(configuration(),'local')
        answer={'answers':{'target':{'choice':'19','confidence':.9,'probabilities':{'19':.9}},
                           'posture':{'choice':'clockwise','confidence':.8,'probabilities':{'clockwise':.8}}}}
        with patch('isaac_agent.models.post',return_value=answer) as post:
            intent,meta=await m.choose_tactic(w.observation(),{},targets(w,{}))
            self.assertEqual(intent,INTENT)
            body=post.call_args.args[1]
            self.assertEqual(set(body['questions']),{'target','posture'})
            self.assertNotIn('use_bomb',json.loads(body['state']))
            self.assertEqual(meta['confidence_by_part'],META['confidence_by_part'])
        answer['answers']['target']['choice']='invented'
        with patch('isaac_agent.models.post',return_value=answer):
            with self.assertRaises(ValueError):await m.choose_tactic(w.observation(),{},targets(w,{}))
