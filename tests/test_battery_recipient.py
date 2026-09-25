import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from isaac_agent.charge import battery_target
from isaac_agent.control import pickup_target
from isaac_agent.fast_policy import compact_goal_observation, lane
from isaac_agent.pickups import routine, useful
from isaac_agent.runtime import Controller
from isaac_agent.strategy import describe_pickup, offers
from test_agent import world, MemoryLog
from test_safety_strategy import refresh


def active(item=45,charge=4,maximum=4,**extra):
    return dict(item=item,charge=charge,max_charge=maximum,battery_charge=0,charge_type=0,**extra)


def prepared():
    w=world()
    w.inventory.update(coins=10,active_items={'0':active()},collectibles={})
    w.payload.update(BOMBS=[],FIRE_HAZARDS=[],INTERACTABLES=[])
    w.payload['PICKUPS']=[dict(id=1,variant=90,sub_type=1,price=0,pos=dict(x=400,y=280))]
    refresh(w)
    return w


class BatteryRecipientTests(unittest.TestCase):
    def test_inactive_schoolbag_cannot_make_normal_battery_a_target(self):
        w=prepared();w.inventory['active_items']['1']=active(item=105,charge=0,maximum=6)
        self.assertIsNone(battery_target(w))
        self.assertFalse(useful(w,w.pickups[0]))
        self.assertIsNone(pickup_target(w,{},True))
        self.assertFalse(any(o['kind']=='pickup' for o in offers(w)))
        w.inventory['active_items']['0']['charge']=3
        self.assertEqual(battery_target(w)['slot'],'0')
        self.assertEqual(pickup_target(w,{},True),(400,280))

    def test_overcharge_capacity_and_native_authority(self):
        w=prepared();w.inventory['collectibles']['63']=1
        self.assertEqual(battery_target(w)['item'],45)
        self.assertEqual(pickup_target(w,{},True),(400,280))
        w.inventory['active_items']['0']['battery_charge']=4
        self.assertIsNone(battery_target(w))
        w.inventory['active_items']['0'].update(charge=0,needs_charge=False)
        self.assertIsNone(battery_target(w))
        w.inventory['active_items']['0'].update(charge=4,needs_charge=True)
        self.assertEqual(battery_target(w)['eligibility_source'],'game NeedsCharge')

    def test_primary_then_pocket_but_no_inactive_or_temporary_slot_guess(self):
        w=prepared();w.inventory['active_items'].update({'0':active(charge=0),
            '1':active(item=105,charge=0),'2':active(item=78,charge=0),'3':active(item=97,charge=0)})
        self.assertEqual(battery_target(w)['slot'],'0')
        w.inventory['active_items']['0']['charge']=4
        self.assertEqual(battery_target(w)['slot'],'2')
        w.inventory['active_items']['2']['charge']=4
        self.assertIsNone(battery_target(w))

    def test_stale_invalid_and_special_charge_are_not_inferred_eligible(self):
        w=prepared();a=w.inventory['active_items']['0'];a['charge']=0
        for typ in (1,2):
            a['charge_type']=typ
            self.assertIsNone(battery_target(w))
        a['charge_type']=0
        for value in (False,None,1,'true'):
            a['needs_charge']=value
            self.assertIsNone(battery_target(w))
        a['needs_charge']=True
        self.assertIsNotNone(battery_target(w))
        w.sampled['PLAYER_INVENTORY']=w.frame-7
        self.assertIsNone(battery_target(w))

    def test_shop_offers_and_model_inputs_share_current_recipient(self):
        w=prepared();w.pickups[0]['price']=5
        self.assertFalse(any(o['kind']=='pickup' for o in offers(w)))
        w.inventory['active_items']['0']['charge']=0
        self.assertTrue(any(o['coin_cost']==5 for o in offers(w) if o['kind']=='pickup'))
        obs=w.observation()
        self.assertEqual(obs['ordinary_battery_target']['slot'],'0')
        self.assertEqual(compact_goal_observation(obs,[])['ordinary_battery_target'],obs['ordinary_battery_target'])
        self.assertEqual(lane(w,w.frame)[0],'sol')

    def test_charged_key_remains_strategic_and_dangerous_batteries_excluded(self):
        w=prepared();w.pickups[0].update(variant=30,sub_type=4)
        self.assertFalse(routine(w.pickups[0]))
        self.assertEqual(describe_pickup(w.pickups[0])['name'],'Charged Key')
        self.assertEqual(lane(w,w.frame)[0],'sol')
        w.pickups[0]['sub_type']=3
        self.assertTrue(routine(w.pickups[0]))
        self.assertEqual(lane(w,w.frame)[0],'collect_resources')
        for subtype in (3,4):
            w.pickups[0].update(variant=90,sub_type=subtype)
            self.assertIsNone(pickup_target(w,{},True))
        self.assertFalse(any(o['kind']=='pickup' for o in offers(w)))


class BatteryControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_overcharge_battery_collected_without_new_model_or_item_pulse(self):
        w=prepared();w.inventory['collectibles']['63']=1
        models=SimpleNamespace(mode='hybrid',plan=AsyncMock(),choose_goal=AsyncMock(),decide=AsyncMock())
        c=Controller(models,MemoryLog())
        try:
            command=c.step(w)
            self.assertEqual(c.route,'collect_resources')
            self.assertEqual(c.execution_plan['navigation_target'],(400,280))
            self.assertGreater(command['move']['x'],0)
            self.assertFalse(command['use_item'])
            # Inject actual capacity becoming full; no further battery chasing.
            w.inventory['active_items']['0']['battery_charge']=4
            refresh(w,1);c.step(w)
            self.assertEqual(c.route,'settle_resources')
            await asyncio.sleep(0)
            models.plan.assert_not_called();models.choose_goal.assert_not_called();models.decide.assert_not_called()
        finally:await c.close()
