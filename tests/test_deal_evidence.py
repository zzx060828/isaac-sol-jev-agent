import json
from pathlib import Path
import tempfile
import unittest

from isaac_agent.deals import door_offers
from isaac_agent.memory import RunMemory
from isaac_agent.strategy import StrategyExecutor, offers
from test_memory_deals import deal_world
from test_agent import MemoryLog
from test_safety_strategy import refresh


def boss(kind):
    w=deal_world();w.info['room_type']=5;w.payload['PICKUPS']=[]
    w.layout['doors']={'0':dict(target_room_type=kind,target_room=-1,is_open=True,x=320,y=80)}
    return w


class DealEvidenceTests(unittest.TestCase):
    def test_angel_and_devil_entry_have_distinct_consequences(self):
        angel=door_offers(boss(15));devil=door_offers(boss(14))
        self.assertEqual({o['deal_type'] for o in angel},{'angel'})
        self.assertEqual({o['deal_type'] for o in devil},{'devil'})
        self.assertIn('not entering a Devil Room',angel[0]['effect'])
        self.assertIn('does not grant',angel[1]['effect'])
        self.assertIn('first offered Devil Room',devil[0]['effect'])
        self.assertIn('not a guaranteed next-floor door',devil[1]['effect'])

    def test_deal_evidence_survives_general_event_pruning_and_disk_reload(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'memory.json';w=boss(14);m=RunMemory(path);w.run_memory=m
            m.observe(w,{})
            for _ in range(270):
                refresh(w,1);m.add(w,'room_cleared',{})
            self.assertFalse(any(n['kind']=='deal_door_observed' for n in m.nodes))
            evidence=m.ledger()['deal_evidence']
            self.assertEqual(evidence['first_retained_devil_boss_door']['type'],14)
            n=RunMemory(path);n.observe(w,{})
            self.assertEqual(n.ledger()['deal_evidence'],evidence)
            w.level='new-seed:1:0';w.layout['doors']={};n.observe(w,{})
            self.assertIsNone(n.ledger()['deal_evidence']['first_retained_devil_boss_door'])

    def test_legacy_import_does_not_claim_complete_deal_history(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'memory.json';w=boss(14);m=RunMemory(path);m.observe(w,{})
            raw=json.loads(path.read_text());raw.pop('deal_records');raw.pop('deal_records_complete')
            raw['complete']=True;path.write_text(json.dumps(raw))
            n=RunMemory(path);n.observe(w,{})
            e=n.ledger()['deal_evidence']
            self.assertFalse(e['history_complete_from_run_start'])
            self.assertFalse(e['typed_records_complete_since_observer_start'])
            self.assertEqual(e['first_retained_devil_boss_door']['type'],14)

    def test_shop_health_price_and_free_devil_loot_are_not_paid_devil_equivalents(self):
        for room,price,classification in ((14,-2,'devil_room_health_price_acquisition'),
                (2,15,'shop_acquisition'),(2,-2,'shop_acquisition'),
                (5,-2,'boss_room_health_price_acquisition'),(14,15,'devil_room_coin_price_acquisition')):
            w=deal_world();w.info['room_type']=room;w.pickups[0]['price']=price
            w.inventory['coins']=30
            executor=StrategyExecutor();choice=offers(w)[0]['choice'];plan={'strategy_enabled':True,'strategy_choice':choice}
            executor.update(w,plan,MemoryLog())
            w.inventory['collectibles']['51']=1;w.payload['PICKUPS']=[];refresh(w,1)
            executor.update(w,plan,MemoryLog())
            entry=w.run_memory.ledger()['deal_evidence']['recent_acquisitions'][-1]
            self.assertEqual(entry['classification'],classification)
        w=deal_world();w.pickups[0]['price']=0
        executor=StrategyExecutor();plan={'strategy_enabled':True,'strategy_choice':offers(w)[0]['choice']}
        executor.update(w,plan,MemoryLog())
        w.inventory['collectibles']['51']=1;w.payload['PICKUPS']=[];refresh(w,1)
        executor.update(w,plan,MemoryLog())
        self.assertFalse(w.run_memory.ledger()['deal_evidence']['recent_acquisitions'])

    def test_declining_door_does_not_erase_later_observed_entry(self):
        w=boss(14);w.run_memory.observe(w,{})
        choice=next(o for o in offers(w) if o['kind']=='deal_skip')
        StrategyExecutor().update(w,{'strategy_enabled':True,'strategy_choice':choice['choice']},MemoryLog())
        w.room=-1;w.info['room_type']=14;w.layout['doors']={};refresh(w,1)
        w.run_memory.observe(w,{})
        e=w.run_memory.ledger()['deal_evidence']
        self.assertEqual(e['recent_skip_decisions'][-1]['type'],14)
        self.assertEqual(e['recent_entries'][-1]['room_type'],14)
        self.assertIn('not proof',e['caution'])

    def test_typed_history_bound_is_explicit_and_output_remains_compact(self):
        w=deal_world();m=RunMemory();m.observe(w,{})
        for i in range(130):m.add(w,'deal_door_observed',{'key':str(i),'type':14})
        self.assertEqual(len(m.deal_records),128)
        e=m.ledger()['deal_evidence']
        self.assertFalse(e['typed_records_complete_since_observer_start'])
        self.assertFalse(e['history_complete_from_run_start'])
        self.assertEqual(len(e['recent_doors']),6)
