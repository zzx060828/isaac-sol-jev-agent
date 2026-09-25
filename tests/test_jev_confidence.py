import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,patch

from isaac_agent.control import candidates
from isaac_agent.fast_policy import fast_offers
from isaac_agent.models import Models,configuration,choice_confidence
from isaac_agent.runtime import Controller
from isaac_agent.tactics import targets
from test_agent import MemoryLog
from test_fast_policy import world
from test_advance import setup as combat_world


class ConfidenceTests(unittest.TestCase):
    def test_missing_or_null_does_not_use_selected_probability(self):
        for extra in ({},{'confidence':None}):
            self.assertEqual(choice_confidence(dict(choice='take',probabilities={'take':1})|extra),0)

    def test_explicit_confidence_remains_authoritative_including_zero(self):
        for confidence in (0,.59,1):
            self.assertEqual(choice_confidence(dict(choice='take',confidence=confidence,
                                                    probabilities={'take':.99})),confidence)

    def test_non_numeric_and_nonfinite_confidence_rejected(self):
        for value in (True,False,'1',{},[],float('nan'),float('inf'),-1,1.01):
            with self.subTest(value=value),self.assertRaises(ValueError):
                choice_confidence(dict(choice='take',confidence=value,probabilities={'take':1}))


class ConfidencePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_confidence_in_each_action_lane_keeps_usage_and_provenance(self):
        models=Models(configuration(),'local')
        w=world();opts=candidates(w,{})[0];choices=fast_offers(w)
        usage={'input_tokens':20,'output_tokens':3,'cost':.000001}
        for lane in ('action','goal','tactic'):
            if lane=='action':
                response={'answers':{'action':dict(choice=opts[0]['action'].id,probabilities={opts[0]['action'].id:1})}}
                call=lambda:models.decide(w.observation(),{},opts)
            elif lane=='goal':
                response={'answers':{'goal':dict(choice=choices[0]['choice'],probabilities={choices[0]['choice']:1})}}
                call=lambda:models.choose_goal(w.observation(),choices)
            else:
                battle=combat_world();options=targets(battle,{})
                response={'answers':{'target':dict(choice=options[0]['choice'],probabilities={options[0]['choice']:1}),
                                     'posture':dict(choice='range',confidence=.8,probabilities={'range':1})}}
                call=lambda:models.choose_tactic(battle.observation(),{},options)
            response.update(usage=usage,model='test-jev')
            with patch('isaac_agent.models.post',return_value=response):
                _,meta=await call()
            self.assertEqual(meta['usage'],usage)
            if lane=='tactic':
                self.assertEqual(meta['confidence_by_part'],dict(target=0,posture=.8))
                self.assertEqual(meta['confidence_source_by_part'],dict(target='missing',posture='provider'))
            else:
                self.assertEqual(meta['confidence'],0)
                self.assertEqual(meta['confidence_source'],'missing')

    async def test_missing_goal_confidence_defers_once_then_sol_keeps_control(self):
        w=world();models=Models(configuration(),'local')
        # Use the real goal parser; all HTTP is replaced by a fixed response.
        calls=SimpleNamespace(mode='hybrid',choose_goal=models.choose_goal,
            plan=AsyncMock(return_value=({'objective':'Skip item','mode':'explore'},{})))
        c=Controller(calls,MemoryLog())
        choice=fast_offers(w)[0]['choice']
        response={'answers':{'goal':dict(choice=choice,probabilities={choice:1})}}
        try:
            with patch('isaac_agent.models.post',return_value=response) as post:
                c.step(w)
                await c.decision_task
                self.assertIsNone(c.plan.get('strategy_choice'))
                self.assertIn(choice,c.fast_deferred)
                row=next(r for r in c.log.rows if r['event']=='fast_goal_deferred')
                self.assertEqual(row['confidence_source'],'missing')
                c.step(w)
                await c.plan_task
                for _ in range(3):c.step(w)
                await asyncio.sleep(0)
                self.assertEqual(post.call_count,1)
                self.assertEqual(calls.plan.await_count,1)
        finally:await c.close()
