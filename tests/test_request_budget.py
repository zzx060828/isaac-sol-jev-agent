import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import patch,AsyncMock

from isaac_agent.models import Models,configuration,completion_settings,post,ModelHTTPError,PlanValidationError
from isaac_agent.runtime import Controller,planner_quota_exhausted
from test_advance import setup,preparation
from test_planning import MemoryLog


class RequestBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_recorded_map_deduplication_keeps_items_walls_routes_and_costs(self):
        from copy import deepcopy
        from isaac_agent.planning import model_observation
        o=json.loads((Path(__file__).parent/'fixtures/planner-map-duplication.json').read_text())['observation']
        original=deepcopy(o);sent=model_observation(o)
        self.assertNotIn('explored_rooms',sent)
        self.assertEqual(sent['navigation_options'],o['navigation_options'])
        self.assertEqual(sent['strategy_offers'],o['strategy_offers'])
        self.assertEqual(sent['floor_map']['connections'],o['floor_map']['connections'])
        for node in sent['floor_map']['nodes']:
            room=o['explored_rooms'][str(node['id'])]
            self.assertEqual(node['remaining_items'],room['collectibles'])
            self.assertEqual(node['walls_last_seen'],room['walls'])
            self.assertEqual(node['resources_last_seen'],room['resources'])
            self.assertEqual(node['marked_rocks_last_seen'],room['marked_rocks'])
        self.assertEqual(o,original)
        self.assertLess(len(json.dumps(sent)),len(json.dumps(o)))
        o['floor_map']['nodes'].pop()
        self.assertIn('explored_rooms',model_observation(o))

    async def test_both_planners_send_one_output_budget_and_keep_selected_model(self):
        model=Models.__new__(Models);model.mode='hybrid';model.config=configuration()
        w=setup();obs=w.observation()
        for method,reply in ((model.plan,{'objective':'Wait','mode':'collect'}),(model.prepare,preparation())):
            response={'choices':[{'message':{'content':json.dumps(reply)},'finish_reason':'stop'}],'usage':{}}
            with patch('isaac_agent.models.post',return_value=response) as mocked:
                await method(obs)
            body=mocked.call_args.args[1]
            self.assertEqual(body['model'],'gpt-5.6-sol')
            self.assertEqual(body['max_completion_tokens'],1200)
            self.assertEqual(body['reasoning_effort'],'medium')
            self.assertNotIn('max_tokens',body)

    def test_invalid_budget_cannot_silently_turn_into_unlimited_request(self):
        for limit in (0,True,1200.5,9000):
            with self.assertRaises(ValueError):completion_settings({'max_completion_tokens':limit})
        self.assertEqual(completion_settings({'max_completion_tokens':2400})['max_completion_tokens'],2400)

    async def test_prepaid_balance_refusal_is_quota_without_provider_body_leak(self):
        secret='test-credential-never-log'
        raw=json.dumps({'error':{'message':'need pre-deduct $0.654, balance $0.548 is insufficient '+secret}}).encode()
        error=urllib.error.HTTPError('https://example.test',403,'Forbidden',{},io.BytesIO(raw))
        with patch.dict(os.environ,{'TEST_KEY':secret}),patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaises(ModelHTTPError) as caught:
                post({'url':'https://example.test','key_env':'TEST_KEY','timeout':1},{})
        self.assertEqual(caught.exception.code,'insufficient_quota')
        self.assertNotIn(secret,str(caught.exception));self.assertNotIn('0.548',str(caught.exception))
        c=Controller(SimpleNamespace(mode='hybrid',plan=AsyncMock(side_effect=caught.exception)),MemoryLog())
        w=setup()
        try:
            await c._plan(w,w.observation(),w.token,w.revision)
            self.assertEqual(next(r for r in c.log.rows if r['event']=='plan_error')['retry_after_seconds'],300)
            self.assertTrue(planner_quota_exhausted(c))
        finally:await c.close()

    async def test_truncated_json_is_not_adopted_even_if_prefix_parses(self):
        model=Models.__new__(Models);model.mode='hybrid';model.config=configuration()
        response={'choices':[{'message':{'content':'{"objective":"Partial","mode":"collect"}'},'finish_reason':'length'}],
                  'usage':{'completion_tokens':1200}}
        with patch('isaac_agent.models.post',return_value=response):
            with self.assertRaises(PlanValidationError) as caught:await model.plan(setup().observation())
        self.assertEqual(caught.exception.metadata['usage']['completion_tokens'],1200)
