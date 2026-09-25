import json
from pathlib import Path
import unittest

from isaac_agent.state import World
from isaac_agent.strategy import StrategyExecutor
from test_agent import MemoryLog
from test_safety_strategy import refresh


def recorded():
    fixture=json.loads((Path(__file__).parent/'fixtures/trinket-contact-timeout.json').read_text())
    w=World();w.ingest(fixture['snapshot'])
    w.sampled.update(fixture['channel_sample_frames'])
    return w


class PickupContactFactsTests(unittest.TestCase):
    def begin(self,w):
        executor=StrategyExecutor();log=MemoryLog()
        plan=dict(strategy_enabled=True,strategy_choice='pickup:1217:350:73:0')
        executor.update(w,plan,log)
        self.assertIsNotNone(executor.choice)
        return executor,log,plan

    def test_recorded_timeout_does_not_infer_acquisition_from_proximity(self):
        w=recorded();executor,log,plan=self.begin(w)
        refresh(w,181);executor.update(w,plan,log)
        row=next(r for r in log.rows if r['event']=='strategy_result')
        self.assertEqual(row['result'],'timeout')
        facts=row['pickup_contact']
        self.assertLess(facts['distance'],2)
        self.assertEqual(facts['held_trinkets'],[63,0])
        self.assertEqual(facts['observed']['sub_type'],73)
        self.assertIsNone(facts['can_pickup_item_reported'])
        self.assertIsNone(facts['observed']['entity_collision_class'])
        self.assertIsNone(executor.choice)

    def test_native_false_collision_zero_and_sample_age_survive_result_logging(self):
        w=recorded();w.inventory['can_pickup_item']=False
        w.pickups[0].update(entity_collision_class=0,collision_radius=10)
        executor,log,plan=self.begin(w)
        refresh(w,181);w.sampled['PLAYER_INVENTORY']=w.frame-7
        executor.update(w,plan,log)
        facts=next(r for r in log.rows if r['event']=='strategy_result')['pickup_contact']
        self.assertIs(facts['can_pickup_item_reported'],False)
        self.assertEqual(facts['observed']['entity_collision_class'],0)
        self.assertEqual(facts['observed']['collision_radius'],10)
        self.assertEqual(facts['sample_age_updates']['PLAYER_INVENTORY'],7)

    def test_changed_form_and_disappearance_are_distinct_observations(self):
        for gone in (False,True):
            w=recorded();executor,log,plan=self.begin(w)
            if gone:w.payload['PICKUPS']=[]
            else:w.pickups[0]['sub_type']=95
            refresh(w,1);executor.update(w,plan,log)
            row=next(r for r in log.rows if r['event']=='strategy_result')
            self.assertEqual(row['result'],'state_changed')
            facts=row['pickup_contact']
            self.assertEqual(facts['expected']['sub_type'],73)
            if gone:
                self.assertIsNone(facts['observed']);self.assertIsNone(facts['distance'])
            else:self.assertEqual(facts['observed']['sub_type'],95)
