import json
from pathlib import Path
import unittest
from unittest.mock import patch

from isaac_agent.models import Models, configuration
from isaac_agent.state import World
from isaac_agent.tactics import targets, compact_state
from test_advance import setup


class TacticalObservationTests(unittest.IsolatedAsyncioTestCase):
    def test_invulnerable_reference_and_motion_are_explicit(self):
        w=setup();enemy=w.enemies[0];enemy.update(is_vulnerable=False,state=3,state_frame=12,
                                                vel=dict(x=-4,y=1),collision_radius=15)
        rows=targets(w,{})
        self.assertFalse(rows[0]['is_vulnerable'])
        self.assertEqual(rows[0]['vel'],{'x':-4,'y':1})
        self.assertEqual(rows[0]['state_frame'],12)
        self.assertEqual(rows[0]['sample_age_updates'],0)
        w.payload['ENEMIES'].append(dict(enemy,id=20,is_vulnerable=True))
        self.assertEqual([r['id'] for r in targets(w,{})],[20])

    def test_bounded_threats_include_incoming_bullet_and_troll_pickup(self):
        w=setup();obs=w.observation()
        obs['projectiles']=[dict(id=i,pos=dict(x=360+i,y=280),vel=dict(x=5,y=0)) for i in range(20)]
        obs['projectiles'].append(dict(id=99,pos=dict(x=500,y=280),vel=dict(x=-20,y=0),is_explosive=True))
        obs['pickups']=[dict(id=100,variant=40,sub_type=3,pos=dict(x=340,y=280))]
        state=compact_state(obs,{},targets(w,{}))
        self.assertEqual(len(state['hazards']['projectiles']),8)
        self.assertEqual(state['hazards']['projectiles'][0]['id'],99)
        self.assertTrue(state['hazards']['projectiles'][0]['is_explosive'])
        self.assertEqual(state['omitted_from_supplied_snapshot']['projectiles'],13)
        self.assertEqual(state['hazards']['troll_pickups'][0]['id'],100)
        self.assertNotIn('pickups',state)

    def test_near_terrain_is_not_lost_behind_distant_array_entries(self):
        w=setup();obs=w.observation()
        obs['terrain']=[dict(x=1000+i*40,y=1000,type=2,collision=1) for i in range(50)]
        obs['terrain'].append(dict(x=321,y=281,type=25,collision=2))
        obs['fires']=[dict(id=1,pos=dict(x=320,y=280),is_extinguished=True),
                      dict(id=2,pos=dict(x=330,y=280),ground_only=True)]
        obs['lasers']=[dict(id=3,pos=dict(x=360,y=280),angle=180,max_distance=300,is_enemy=True),
                       dict(id=4,pos=dict(x=320,y=280),is_enemy=False)]
        state=compact_state(obs,{},targets(w,{}))
        self.assertEqual(state['geometry']['terrain'][0]['type'],25)
        self.assertEqual(len(state['geometry']['terrain']),32)
        self.assertEqual(state['omitted_from_supplied_snapshot']['terrain'],19)
        self.assertEqual([e['id'] for e in state['hazards']['fires']],[2])
        self.assertEqual([e['id'] for e in state['hazards']['lasers']],[3])

    def test_recorded_projectile_state_reaches_compact_tactical_input(self):
        data=json.loads((Path(__file__).parent/'fixtures/pooter-projectile-before-hit.json').read_text())
        w=World();w.ingest(data)
        state=compact_state(w.observation(),{},targets(w,{}))
        self.assertTrue(state['hazards']['projectiles'])
        self.assertTrue(all('vel' in p for p in state['hazards']['projectiles']))
        self.assertTrue(all('is_vulnerable' in t and 'alignment_error' in t for t in state['targets']))

    async def test_typed_provider_contains_facts_once_and_no_extra_action_questions(self):
        w=setup();obs=w.observation();obs['tactical_sensor_age_updates']={'PROJECTILES':2}
        obs['run_history']={'private_test_marker':'must_not_be_sent'}
        rows=targets(w,{})
        result={'answers':{'target':{'choice':'19','probabilities':{'19':1}},
                           'posture':{'choice':'range','probabilities':{'range':1}}}}
        with patch('isaac_agent.models.post',return_value=result) as post:
            await Models(configuration(),'local').choose_tactic(obs,{},rows)
        body=post.call_args.args[1];state=json.loads(body['state'])
        self.assertEqual(set(body['questions']),{'target','posture'})
        self.assertEqual(body['questions']['target']['criteria']['19'],'Enemy 19 in state.targets')
        self.assertEqual(state['sensor_age_updates']['PROJECTILES'],2)
        self.assertNotIn('run_history',state)
        self.assertNotIn('private_test_marker',body['state'])
