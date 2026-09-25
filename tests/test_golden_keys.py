import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from isaac_agent.control import traversable_door
from isaac_agent.journey import Journey,options
from isaac_agent.pickups import has_golden_key,useful
from isaac_agent.planning import planning_context
from isaac_agent.review import basis
from isaac_agent.runtime import Controller
from test_agent import MemoryLog,world
from test_safety_strategy import refresh


def prepared():
    w=world()
    w.inventory.update(keys=0,golden_key=True)
    w.layout['doors']['2'].update(target_room_type=4,is_locked=True,is_open=False)
    refresh(w)
    return w


class GoldenKeyTests(unittest.TestCase):
    def test_zero_keys_can_unlock_only_supported_doors_with_fresh_explicit_flag(self):
        w=prepared();door=w.layout['doors']['2']
        self.assertTrue(traversable_door(w,door))
        self.assertEqual(planning_context(w)['access'],())
        self.assertTrue(options(w))
        for value in (False,None,1,'true'):
            w.inventory['golden_key']=value
            self.assertFalse(traversable_door(w,door))
        w.inventory['golden_key']=True
        w.sampled['PLAYER_INVENTORY']=w.frame-7
        self.assertFalse(traversable_door(w,door))
        refresh(w)
        for room_type in (10,14,15):
            door['target_room_type']=room_type
            self.assertFalse(traversable_door(w,door))

    def test_two_locks_cost_zero_and_loss_does_not_spend_owned_keys(self):
        w=prepared()
        w.rooms['2']={'clear':True,'type':4,'doors':{
            '2':dict(target_room=3,target_room_type=2,is_locked=True,is_open=False)}}
        goal=next(o for o in options(w) if o['destination_room']==3)
        self.assertEqual(goal['keys_required'],0)
        self.assertEqual([e['key_cost'] for e in goal['path']],[0,0])
        j=Journey();j.start(w,goal);log=MemoryLog()
        self.assertEqual(j.step(w,log),'2')
        w.inventory.update(golden_key=False,keys=2)
        self.assertIsNone(j.step(w,log))
        self.assertEqual(j.end_reason,'key_budget_changed')
        replacement=next(o for o in options(w) if o['destination_room']==3)
        self.assertEqual(replacement['keys_required'],2)

    def test_stale_flag_stops_free_contract_even_with_one_key(self):
        w=prepared();w.inventory['keys']=1
        j=Journey();j.start(w,options(w)[0])
        w.sampled['PLAYER_INVENTORY']=w.frame-7 # strategy-ready still permits <=8.
        self.assertTrue(w.strategy_ready())
        self.assertIsNone(j.step(w,MemoryLog()))
        self.assertEqual(j.end_reason,'key_budget_changed')

    def test_normal_contract_can_benefit_from_new_gold_and_still_cross_open_door(self):
        w=prepared();w.inventory.update(golden_key=False,keys=1)
        contract=options(w)[0]
        self.assertEqual(contract['keys_required'],1)
        j=Journey();j.start(w,contract)
        w.inventory.update(golden_key=True,keys=0)
        self.assertEqual(j.step(w,MemoryLog()),'2')
        w.inventory['golden_key']=False
        w.layout['doors']['2'].update(is_locked=False,is_open=True)
        self.assertEqual(j.step(w,MemoryLog()),'2')

    def test_gold_changes_review_and_duplicate_gold_is_not_a_pickup_goal(self):
        w=prepared();key=dict(variant=30,sub_type=2,price=0)
        self.assertFalse(useful(w,key))
        before=basis(w)
        w.inventory['golden_key']=False
        self.assertTrue(useful(w,key))
        self.assertNotEqual(basis(w)['keys'],before['keys'])
        w.inventory['keys']=1
        self.assertEqual(planning_context(w)['access'],('2',))

    def test_floor_change_drops_unobserved_golden_key(self):
        from isaac_agent.demo import frame_message
        w=prepared()
        msg=frame_message(w.frame+1)
        msg['agent']['level_id']='demo:2:0'
        msg['payload'].pop('PLAYER_INVENTORY')
        w.ingest(msg)
        self.assertFalse(has_golden_key(w))


class GoldenKeyControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_free_treasure_door_avoids_sol_but_loss_restores_last_key_review(self):
        w=prepared();models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock())
        c=Controller(models,MemoryLog())
        try:
            command=c.step(w)
            self.assertEqual(c.route,'routine')
            self.assertGreater(command['move']['x'],0)
            models.plan.assert_not_called();models.choose_goal.assert_not_called()
            w.inventory.update(keys=1,golden_key=False);refresh(w,1)
            command=c.step(w)
            self.assertEqual(c.route,'sol')
            self.assertEqual(command['move'],{'x':0,'y':0})
            self.assertTrue(any(row['event']=='plan_request' for row in c.log.rows))
        finally:await c.close()
