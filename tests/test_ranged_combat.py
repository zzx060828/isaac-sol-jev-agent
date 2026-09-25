import json
import math
from pathlib import Path
import unittest
from test_agent import world
from isaac_agent.combat import spacing, select_enemy
from isaac_agent.control import Action, candidates, shield, target_point
from isaac_agent.hazards import safe_shot
from isaac_agent.knowledge import encounter_knowledge, encounter_coverage, tactical_knowledge
from isaac_agent.physics import trajectory
from isaac_agent.state import World,xy


def advance(w, move):
    velocity=xy(w.player.get('vel')); p=xy(w.player.get('pos'))
    desired=[axis*4.667*w.stats.get('speed',1)/(math.hypot(*move) or 1) for axis in move]
    for _ in range(2):
        velocity=tuple(math.sqrt(.775)*velocity[i]+(1-math.sqrt(.775))*desired[i] for i in (0,1))
        p=tuple(p[i]+velocity[i] for i in (0,1))
    w.player.update(pos=dict(zip(('x','y'),p)),vel=dict(zip(('x','y'),velocity)))
    w.frame+=1
    w.sampled.update({k:w.frame for k in w.payload})


class RangedCombatTests(unittest.TestCase):
    def test_recorded_larry_passage_reaches_distant_lane(self):
        from isaac_agent.control import physical_collision, firing_lane
        f=json.loads((Path(__file__).parent/'fixtures/larry-rock-stall.json').read_text())
        w=World();w.ingest(f['snapshot'])
        # Static geometry replay; this does not simulate a moving boss or hits.
        for _ in range(110):
            action=candidates(w,f['plan'])[0][0]['action'];advance(w,action.move)
            p=xy(w.player['pos'])
            self.assertFalse(physical_collision(w,p,float(w.stats['size'])))
            for enemy in w.combat_enemies:
                self.assertGreater(math.dist(p,xy(enemy['pos'])),
                                   w.stats['size']+enemy['collision_radius']+10)
        target=target_point(w,f['plan'])[0]
        self.assertLess(abs(p[0]-target[0]),16)
        self.assertTrue(firing_lane(w,p,target))
        self.assertGreater(math.dist(p,target),190)
        self.assertLess(math.dist(p,target),w.stats['range'])
        self.assertEqual(action.shoot,(0,1))

    def test_narrow_maneuver_keeps_contact_and_projectile_penalties(self):
        from isaac_agent.control import risk
        w=world();w.payload['ENEMIES']=[{'id':2,'pos':w.player['pos'],'collision_radius':20}]
        self.assertGreater(risk(w,(0,0),maneuver=True),100)
        w.payload['ENEMIES']=[]
        w.payload['PROJECTILES']={'enemy_projectiles':[{'id':9,'pos':{'x':380,'y':280},
            'vel':{'x':-10,'y':0},'collision_radius':5}]}
        self.assertEqual(risk(w,(0,0),maneuver=True),risk(w,(0,0)))
        self.assertGreater(risk(w,(0,0),maneuver=True),50)

    def test_long_range_holds_fire_without_chasing_into_contact(self):
        w=world(player=(280,280));w.stats['range']=360
        w.payload['ENEMIES']=[{'id':2,'pos':{'x':560,'y':280},'collision_radius':12}]
        for _ in range(40):
            action=candidates(w,{})[0][0]['action']
            self.assertLessEqual(action.move[0],0)
            self.assertEqual(action.shoot,(1,0))
            advance(w,action.move)
        self.assertLess(math.dist(xy(w.player['pos']),(560,280)),360)

    def test_short_range_still_approaches_to_hit(self):
        w=world(player=(160,280));w.stats['range']=100
        w.payload['ENEMIES']=[{'id':2,'pos':{'x':480,'y':280},'collision_radius':12}]
        self.assertEqual(candidates(w,{})[0][0]['action'].move[0],1)
        self.assertLess(spacing(w.stats)['max_distance'],100)

    def test_close_add_temporarily_overrides_distant_boss(self):
        w=world();w.stats['range']=260
        w.payload['ENEMIES']=[{'id':2,'type':100,'is_boss':True,'pos':{'x':540,'y':280}},
                              {'id':3,'type':85,'pos':{'x':390,'y':280}}]
        self.assertEqual(select_enemy(w,{'target_enemy':2})['id'],3)
        w.payload['ENEMIES'][1]['pos']={'x':580,'y':360}
        self.assertEqual(select_enemy(w,{'target_enemy':2})['id'],2)

    def test_crowding_guard_rejects_approach_when_retreat_is_equally_safe(self):
        w=world(player=(250,280));w.stats['range']=260
        w.payload['ENEMIES']=[{'id':2,'pos':{'x':400,'y':280},'collision_radius':12}]
        fallback=candidates(w,{})[0][0]['action']
        chosen,override=shield(w,Action((1,0),(1,0)),fallback,None)
        self.assertTrue(override)
        self.assertLessEqual(chosen.move[0],0)

    def test_off_axis_snapshot_eventually_reaches_a_shooting_lane(self):
        w=World();w.ingest(json.loads((Path(__file__).parent/'fixtures/off_axis_shooting_stall.json').read_text()))
        plan={'target_enemy':7,'combat_distance':220}
        for _ in range(120):
            action=candidates(w,plan)[0][0]['action'];advance(w,action.move)
        # This recorded layout includes TNT shielding the enemy; shooting the
        # selected barrel from blast clearance is also a useful firing lane.
        target=tuple(w.combat_breach.get('point', target_point(w,plan)[0]));p=xy(w.player['pos'])
        self.assertLess(min(abs(p[0]-target[0]),abs(p[1]-target[1])),25)
        self.assertLess(math.dist(p,target),w.stats['range'])

    def test_horf_corner_finds_shorter_firing_lane(self):
        w=World();w.ingest(json.loads((Path(__file__).parent/'fixtures/horf_corner_stall.json').read_text()))
        # Static geometry replay; old projectiles must not be frozen in place.
        w.payload['PROJECTILES']={'enemy_projectiles':[]}
        plan={'target_enemy':3,'combat_distance':238}
        for _ in range(100):
            action=candidates(w,plan)[0][0]['action'];advance(w,action.move)
        target=target_point(w,plan)[0];p=xy(w.player['pos'])
        self.assertLess(min(abs(p[i]-target[i]) for i in (0,1)),16)
        self.assertGreater(math.dist(p,target),w.stats['range']*.55)
        self.assertEqual(action.shoot,(1,0))

    def test_recorded_host_corridor_reaches_lane_without_switching_to_harmless_bodies(self):
        from isaac_agent.control import firing_lane
        f=json.loads((Path(__file__).parent/'fixtures/narrow-combat-stall.json').read_text())
        w=World();w.ingest(f['snapshot'])
        # Geometry replay, with no frozen projectiles or attack phase simulation.
        w.payload['PROJECTILES']={'enemy_projectiles':[]}
        for _ in range(90):
            self.assertEqual(select_enemy(w,f['plan'])['id'],1)
            action=candidates(w,f['plan'])[0][0]['action'];advance(w,action.move)
        p=xy(w.player['pos']);target=target_point(w,f['plan'])[0]
        self.assertLess(abs(p[0]-target[0]),16)
        self.assertTrue(firing_lane(w,p,target))
        self.assertGreater(math.dist(p,target),180)
        self.assertLess(math.dist(p,target),260)
        self.assertEqual(action.shoot,(0,-1))

    def test_recorded_rock_lane_relaxes_standoff_to_route_around_pit(self):
        from isaac_agent.control import firing_lane, blocked
        f=json.loads((Path(__file__).parent/'fixtures/host-rock-lane-stall.json').read_text())
        w=World();w.ingest(f['snapshot']);w.payload['PROJECTILES']={'enemy_projectiles':[]}
        for _ in range(90):
            action=candidates(w,f['plan'])[0][0]['action'];advance(w,action.move)
            self.assertFalse(blocked(w,xy(w.player['pos']),float(w.stats['size'])))
        p=xy(w.player['pos']);target=target_point(w,f['plan'])[0]
        self.assertLess(abs(p[1]-target[1]),16)
        self.assertTrue(firing_lane(w,p,target))
        self.assertGreater(math.dist(p,target),140)
        self.assertLess(math.dist(p,target),260)
        self.assertEqual(action.shoot,(1,0))

    def test_host_contact_exception_does_not_hide_real_close_threat_or_bullets(self):
        from isaac_agent.control import risk
        w=world();w.capabilities['repentance_plus']=False
        host={'id':3,'type':27,'variant':0,'pos':{'x':350,'y':280},'collision_radius':13}
        w.payload['ENEMIES']=[host]
        base=risk(w,(0,0));self.assertLess(base,1)
        host['variant']=3
        self.assertGreater(risk(w,(0,0)),base+20)
        host['variant']=0
        w.payload['PROJECTILES']={'enemy_projectiles':[{'id':9,'pos':{'x':380,'y':280},'vel':{'x':-10,'y':0},'collision_radius':5}]}
        self.assertGreater(risk(w,(0,0)),base+20)
        w.payload['ENEMIES'] += [{'id':2,'pos':{'x':540,'y':280},'is_boss':True},
                                {'id':4,'type':85,'pos':{'x':380,'y':280}}]
        self.assertEqual(select_enemy(w,{'target_enemy':2})['id'],4)

    def test_projectile_forecast_does_not_escape_through_room_wall(self):
        from isaac_agent.control import physical_collision
        w=World();w.ingest(json.loads((Path(__file__).parent/'fixtures/pooter-projectile-before-hit.json').read_text()))
        raw=trajectory(w,(0,-1),8)
        clipped=trajectory(w,(0,-1),8,collides=lambda p:physical_collision(w,p,10))
        self.assertLess(raw[-1][1],150)
        self.assertTrue(all(p[1]>=150 for p in clipped))
        self.assertTrue(all(not physical_collision(w,p,10) for p in clipped))

    def test_pot_room_uses_firing_position_that_fits_room_height(self):
        from isaac_agent.control import firing_lane
        w=World();w.ingest(json.loads((Path(__file__).parent/'fixtures/ghost-pepper-stall.json').read_text()))
        w.payload['PROJECTILES']={'enemy_projectiles':[]}
        plan={'target_enemy':3,'combat_distance':213}
        for _ in range(120):
            action=candidates(w,plan)[0][0]['action'];advance(w,action.move)
        p=xy(w.player['pos']);target=target_point(w,plan)[0]
        self.assertLess(min(abs(p[i]-target[i]) for i in (0,1)),16)
        self.assertGreater(math.dist(p,target),w.stats['range']*.4)
        self.assertTrue(firing_lane(w,p,target))
        self.assertEqual(action.shoot,(0,1))

    def test_flank_path_goes_around_enemy_clearance_circle(self):
        from isaac_agent.control import path_step, segment_distance
        w=world(player=(120,280))
        target=(520,280);circle=(320,280,130)
        waypoint=path_step(w,target,avoid=circle)
        self.assertNotEqual(waypoint,target)
        self.assertGreaterEqual(segment_distance(circle[:2],xy(w.player['pos']),waypoint),130)

    def test_standard_pooter_body_is_not_treated_as_contact_damage(self):
        from isaac_agent.control import risk
        w=world();w.capabilities['repentance_plus']=False
        e={'id':2,'type':14,'variant':0,'is_champion':False,'pos':{'x':350,'y':280},'collision_radius':17}
        w.payload['ENEMIES']=[e]
        base=risk(w,(0,0))
        self.assertLess(base,1)
        e['is_champion']=True
        self.assertGreater(risk(w,(0,0)),base+20)
        e['is_champion']=False
        # At ten units/update this bullet crosses the player within the
        # eight-update horizon. NPC/projectile velocity needs no tick doubling.
        bullet={'id':3,'pos':{'x':380,'y':280},'vel':{'x':-10,'y':0},'collision_radius':5}
        w.payload['PROJECTILES']={'enemy_projectiles':[bullet]}
        self.assertGreater(risk(w,(0,0)),base+20)
        bullet['vel']['x']=10
        self.assertLess(risk(w,(0,0)),base+1)

    def test_known_death_explosion_is_not_shot_at_point_blank(self):
        w=world();w.payload['ENEMIES']=[{'id':2,'type':25,'variant':0,'pos':{'x':400,'y':280}}]
        self.assertFalse(safe_shot(w,(0,0),(1,0)))
        w.payload['ENEMIES'][0]['pos']['x']=550
        self.assertTrue(safe_shot(w,(0,0),(1,0)))
        w.payload['ENEMIES'][0].update(variant=1,pos={'x':400,'y':280})
        self.assertTrue(safe_shot(w,(0,0),(1,0)))

    def test_boss_spawner_and_spiders_fit_in_tactical_knowledge(self):
        enemies=[{'type':85}]*12+[{'type':30,'variant':2},{'type':100,'is_boss':True}]
        briefs=encounter_knowledge(enemies)
        result=tactical_knowledge({'enemy_knowledge':briefs})
        self.assertLessEqual(len(json.dumps(result)),1500)
        self.assertTrue({'100:0','30:2','85:0'}.issubset({r.get('entity') for r in result}))
        report=encounter_coverage(enemies+[{'type':9999}],briefs)
        self.assertEqual(report['unknown_types'],['9999:0'])

    def test_invulnerable_single_enemy_does_not_receive_shots(self):
        w=world();w.payload['ENEMIES']=[{'id':2,'type':100,'is_vulnerable':False,'pos':{'x':500,'y':280}}]
        self.assertTrue(all(o['action'].shoot==(0,0) for o in candidates(w,{})[0]))
