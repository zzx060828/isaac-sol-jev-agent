import json
import math
from pathlib import Path
import unittest

from isaac_agent.control import pickup_target
from isaac_agent.resources import ResourcePolicy,nearby_red_healing,resource_options
from isaac_agent.pickups import red_healing
from isaac_agent.state import World
from test_review_continuity import setup
from test_safety_strategy import refresh


def heart(index,subtype=1,x=400,y=280,**extra):
    return dict(id=index,variant=10,sub_type=subtype,price=0,options_index=0,
                pos=dict(x=x,y=y))|extra


def prepared():
    w=setup();w.stats['player_type']=1
    w.health.update(red_hearts=6,max_hearts=10)
    w.inventory.update(pill_0=10,pill_identified=True,pill_effect=5,can_use=True,active_items={})
    w.payload['PICKUPS']=[heart(1,x=400),heart(2,x=420)]
    refresh(w)
    return w


class HealingEconomyTests(unittest.TestCase):
    def test_save_one_use_heal_when_free_hearts_cover_deficit_but_sol_can_override(self):
        for kind in ('pill','card'):
            w=prepared()
            if kind=='card':w.inventory.update(pill_0=0,card_0=20)
            p=ResourcePolicy()
            self.assertEqual(nearby_red_healing(w),4)
            self.assertIsNone(p.choose(w))
            option=next(o for o in resource_options(w) if o['kind']==kind)
            self.assertTrue(option['model_only'])
            self.assertEqual(p.choose(w,preferred=kind)['kind'],kind)

    def test_emergency_combat_stale_or_insufficient_ground_never_delays_heal(self):
        for case in ('emergency','combat','stale','too_little','troll'):
            w=prepared()
            if case=='emergency':w.health['red_hearts']=2
            elif case=='combat':w.payload['ENEMIES']=[dict(id=3,pos=dict(x=480,y=280),hp=10)]
            elif case=='stale':w.sampled['PICKUPS']=w.frame-7
            elif case=='too_little':w.payload['PICKUPS']=w.pickups[:1]
            else:w.payload['PICKUPS'].append(dict(id=9,variant=40,sub_type=3,pos=dict(x=500,y=280)))
            choice=ResourcePolicy().choose(w)
            self.assertEqual(choice['kind'],'pill',case)
            self.assertNotIn('model_only',choice)

    def test_paid_options_waiting_far_and_enclosed_hearts_are_not_owned_healing(self):
        for extra in ({'price':3},{'options_index':1},{'wait':20},{'pos':dict(x=580,y=280)}):
            w=prepared();w.payload['PICKUPS']=[heart(1,5,**extra)]
            self.assertEqual(nearby_red_healing(w),0)
            self.assertEqual(ResourcePolicy().choose(w)['kind'],'pill')
        data=json.loads((Path(__file__).parent/'fixtures/enclosed-key.json').read_text())
        w=World();w.ingest(data);w.stats['player_type']=1
        w.health.update(red_hearts=6,max_hearts=10)
        w.pickups[0].update(variant=10,sub_type=5)
        refresh(w)
        self.assertEqual(nearby_red_healing(w),0)

    def test_disappearing_ground_resumes_heal_without_phantom_health(self):
        w=prepared();p=ResourcePolicy()
        self.assertIsNone(p.choose(w))
        w.payload['PICKUPS']=[];refresh(w,1)
        self.assertEqual(w.health['red_hearts'],6)
        self.assertEqual(p.choose(w)['kind'],'pill')

    def test_half_heart_fit_has_bounded_detour_and_emergency_uses_nearest(self):
        w=prepared();w.health.update(red_hearts=9,max_hearts=10)
        w.payload['PICKUPS']=[heart(1,x=400),heart(2,2,x=400,y=360)]
        self.assertEqual(pickup_target(w,{},True),(400,360))
        w.pickups[1]['pos']['x']=580
        self.assertEqual(pickup_target(w,{},True),(400,280))
        w.pickups[1]['pos']['x']=400;w.health.update(red_hearts=3,max_hearts=4)
        self.assertEqual(pickup_target(w,{},True),(400,280))

    def test_maggys_bow_and_jar_change_waste_math(self):
        w=prepared();w.health.update(red_hearts=9,max_hearts=10)
        w.payload['PICKUPS']=[heart(1,x=400),heart(2,2,x=400,y=360)]
        w.inventory['collectibles']={'312':1}
        self.assertEqual(red_healing(w,w.pickups[1]),2)
        self.assertEqual(pickup_target(w,{},True),(400,360))
        w.inventory['collectibles']={'219':1}
        self.assertEqual(red_healing(w,w.pickups[1]),1) # Old Bandage is not Maggy's Bow.
        w.inventory['active_items']={'0':dict(item=290)}
        self.assertEqual(pickup_target(w,{},True),(400,280))

    def test_renewable_yum_heart_is_not_automatically_saved_like_single_use_pill(self):
        w=prepared();w.inventory['active_items']={'0':dict(item=45,charge=4,max_charge=4)}
        self.assertEqual(ResourcePolicy().choose(w)['id'],45)

    def test_movement_reaches_half_heart_without_crossing_larger_heart(self):
        from isaac_agent.control import candidates
        from test_ranged_combat import advance
        from isaac_agent.state import xy
        w=prepared();w.health.update(red_hearts=9,max_hearts=10)
        w.payload['PICKUPS']=[heart(1,x=400),heart(2,2,x=400,y=360)]
        for _ in range(40):
            target=pickup_target(w,{},True)
            action=candidates(w,{'strategy_enabled':True,'navigation_target':target})[0][0]['action']
            advance(w,action.move)
            self.assertGreater(math.dist(xy(w.player['pos']),(400,280)),32)
            if math.dist(xy(w.player['pos']),(400,360))<20:break
        else:self.fail('Did not reach the half heart')
        # Merely selecting a half heart behind a large heart would still pick
        # the large one on the way. Do not pretend that route saves health.
        w=prepared();w.health.update(red_hearts=9,max_hearts=10)
        w.payload['PICKUPS']=[heart(1,x=400),heart(2,2,x=440)]
        self.assertEqual(pickup_target(w,{},True),(400,280))
