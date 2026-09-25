from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,patch

from isaac_agent.fast_policy import fast_offers
from isaac_agent.models import Models,configuration
from isaac_agent.planning import planning_context
from isaac_agent.runtime import Controller
from test_pressure_plates import snapshot
from test_agent import MemoryLog
from test_safety_strategy import refresh


def setup():
    w=snapshot()
    w.layout['grid']['23']=dict(w.layout['grid']['22'],grid_index=23,x=360)
    refresh(w)
    choices=fast_offers(w)
    assert len(choices)==2
    obs=w.observation();obs['fast_items']=planning_context(w)['items']
    return w,choices,obs


class FastGoalBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_second_plate_survives_first_plate_disappearing(self):
        w,choices,obs=setup();selected=choices[1]['choice']
        async def choose(observation, supplied):
            w.layout['grid']['22']['state']=3;refresh(w,1)
            return selected,{'confidence':.9}
        m=SimpleNamespace(mode='hybrid',choose_goal=choose,plan=AsyncMock())
        c=Controller(m,MemoryLog());c.reset_actions(w)
        try:
            await c._fast_goal(w,obs,w.token,choices)
            self.assertEqual(c.plan['strategy_choice'],selected)
            c.strategy.update(w,c.plan,c.log)
            self.assertEqual(c.strategy.choice,selected)
            m.plan.assert_not_called()
        finally:await c.close()

    async def test_selected_disappeared_or_new_unsupplied_plate_is_rejected(self):
        for case in ('gone','new'):
            w,choices,obs=setup()
            async def choose(observation,supplied):
                if case=='gone':
                    w.layout['grid']['23']['state']=3;selected='pressure_plate:23'
                else:
                    w.layout['grid']['24']=dict(w.layout['grid']['23'],grid_index=24,x=400)
                    selected='pressure_plate:24'
                refresh(w,1)
                return selected,{'confidence':1}
            c=Controller(SimpleNamespace(mode='hybrid',choose_goal=choose),MemoryLog());c.reset_actions(w)
            try:
                await c._fast_goal(w,obs,w.token,choices)
                self.assertIsNone(c.plan.get('strategy_choice'))
                self.assertEqual(c.log.rows[-1]['reason'],
                                 'selected_offer_unavailable' if case=='gone' else 'choice_not_supplied')
            finally:await c.close()

    async def test_probability_does_not_override_provider_confidence_threshold(self):
        w,choices,obs=setup()
        m=SimpleNamespace(mode='hybrid',choose_goal=AsyncMock(return_value=(choices[0]['choice'],
                          {'confidence':.59,'probabilities':{choices[0]['choice']:.72,'defer':.08}})))
        c=Controller(m,MemoryLog());c.reset_actions(w)
        try:
            await c._fast_goal(w,obs,w.token,choices)
            self.assertIsNone(c.plan.get('strategy_choice'))
            self.assertEqual(c.log.rows[-1]['reason'],'confidence_below_threshold')
            self.assertEqual(c.log.rows[-1]['confidence_threshold'],.6)
        finally:await c.close()

    async def test_fast_request_keeps_full_offers_once_and_same_choice_set(self):
        w,choices,obs=setup()
        response={'answers':{'goal':{'choice':choices[0]['choice'],'confidence':.9}},'usage':{}}
        with patch('isaac_agent.models.post',return_value=response) as post:
            choice,_=await Models(configuration(),'local').choose_goal(obs,choices)
        body=post.call_args.args[1];state=json.loads(body['state'])
        self.assertEqual(state['fast_offers'],choices)
        self.assertEqual(set(body['questions']['goal']['criteria']),{o['choice'] for o in choices}|{'defer'})
        self.assertNotIn('grid_index',json.dumps(body['questions']['goal']['criteria']))
        self.assertEqual(choice,choices[0]['choice'])
