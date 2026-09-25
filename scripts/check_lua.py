"""Behavioral checks for the game-side input lease. Requires optional lupa."""
from pathlib import Path
from lupa.lua53 import LuaRuntime

ROOT = Path(__file__).resolve().parents[1]
lua = LuaRuntime()
mod_source = (ROOT / 'build/SocketBridge_AstraJev/main.lua').read_text()
lua.compile(mod_source)
enemy_source = mod_source.split('-- 5. ENEMIES', 1)[1].split('-- 6. PROJECTILES', 1)[0]
enemy_source = enemy_source[enemy_source.index('SensorRegistry:register'):]
lua.execute('''
local SensorRegistry={sensors={}}
function SensorRegistry:register(name,sensor) self.sensors[name]=sensor end
local EntityPartition={ENEMY=1}
local Helpers={vectorToTable=function(v) return v end}
''' + enemy_source + '''
local e={Index=2,Type=100,Variant=0,SubType=0,Position={x=100,y=100},
         Velocity={x=0,y=0},HitPoints=20,MaxHitPoints=100,Size=35,EntityCollisionClass=0}
function e:IsActiveEnemy() return true end
function e:IsVulnerableEnemy() return false end
function e:IsBoss() return true end
function e:ToNPC() return nil end
local observed=SensorRegistry.sensors.ENEMIES.extract(e,nil)
assert(observed and observed.is_vulnerable==false and observed.collision_class==0,
       'invulnerable active enemy must remain observable')
''')
stub = '''
local State = { updateCount=10, currentRoom=1, connected=true, controlMode="FORCE_AI" }
local Protocol = { createDataMessage=function() return {} end, createFullStateMessage=function() return {} end }
local SensorRegistry = { sensors={} }
SensorRegistry.sensors.PICKUPS={cache={},extract=function(e) return {variant=e.Variant,sub_type=1} end}
local InputExecutor = { moveDirection={x=0,y=0} }
InputExecutor.reset=function()
    InputExecutor.moveDirection={x=0,y=0}; InputExecutor.useBomb=false
    InputExecutor.useItem=false; InputExecutor.usePill=false; InputExecutor.useCard=false
end
InputExecutor.applyCommand=function(c)
    InputExecutor.moveDirection=c.move; InputExecutor.useBomb=c.use_bomb
    InputExecutor.useItem=c.use_item; InputExecutor.usePill=c.use_pill; InputExecutor.useCard=c.use_card
end
local Network={ disconnect=function() State.connected=false end }
local Config={HOST="127.0.0.1",PORT=9527}
local json={}
local EventSystem={emit=function() end}
local currentRoom, wall = 1, 100
local pill, identified, effectReads = 3, false, 0
local goldenKey, goldenBomb = true, false
local chargeNeeded={[0]=true,[1]=false,[2]=false}
local canPickup=true
local brownCap=false
local player = {IsDead=function() return false end, IsExtraAnimationFinished=function() return true end,
                HasTrinket=function(_,id) assert(id==90);return brownCap end,
                CanPickupItem=function() return canPickup end,
                NeedsCharge=function(_,slot) return chargeNeeded[slot] end,
                HasGoldenKey=function() return goldenKey end,
                HasGoldenBomb=function() return goldenBomb end,
                GetActiveItem=function() return 45 end, GetPill=function() return pill end, GetCard=function() return 6 end}
local vector={}
vector.__index=vector
function Vector(x,y) return setmetatable({X=x,Y=y},vector) end
function vector:Distance(v) return math.sqrt((self.X-v.X)^2+(self.Y-v.Y)^2) end
function vector:Length() return self:Distance(Vector(0,0)) end
vector.__add=function(a,b) return Vector(a.X+b.X,a.Y+b.Y) end
vector.__mul=function(a,b) return Vector(a.X*b,a.Y*b) end
local bombs,rockType=4,4
local rock={CollisionClass=3,Position=Vector(320,280),GetType=function() return rockType end}
local room={GetRoomShape=function() return 1 end,IsClear=function() return true end,
    GetAliveEnemiesCount=function() return 0 end,GetGridSize=function() return 135 end,
    GetGridEntity=function(_,i) if i==67 then return rock end end,
    IsDoorSlotAllowed=function() return true end,GetDoorSlotPosition=function() return Vector(40,280) end}
RoomShape={ROOMSHAPE_1x1=1}
player.Position=Vector(365,280);player.Velocity=Vector(0,0)
player.GetNumBombs=function() return bombs end
Isaac={GetTime=function() return wall end, GetPlayer=function() return player end,
       GetItemConfig=function() return {GetCollectible=function() return {Name="Yum Heart",Description="Heal",ChargeType=0} end} end}
local Helpers={getPlayers=function() return {player} end}
local CustomCollectors={PLAYER_INVENTORY=function() return {{pill_0=pill,active_items={['0']={item=45},['1']={item=105},['2']={item=78}}}} end}
CustomCollectors.ROOM_INFO=function() return {room_type=1} end
local level={GetCurrentRoomIndex=function() return currentRoom end,GetStage=function() return 1 end,GetStageType=function() return 0 end}
level.GetCurses=function() return 64 end
Game=function() return {GetRoom=function() return room end,GetLevel=function() return level end,GetSeeds=function() return {GetStartSeed=function() return 42 end} end,
    GetItemPool=function() return {IsPillIdentified=function() return identified end,
                                  GetPillEffect=function() effectReads=effectReads+1; return 5 end} end} end
local AgentAddCallback=function() end
ModCallbacks={}
'''
assertions = '''
assert(CustomCollectors.ROOM_INFO().curses==64, 'curse state must be transmitted')
local groupedCoin={Variant=20,ToPickup=function() return {OptionsPickupIndex=7,EntityCollisionClass=0,Size=10} end}
assert(SensorRegistry.sensors.PICKUPS.extract(groupedCoin).options_index==7,
       'consumable option groups must be transmitted too')
local contactRow=SensorRegistry.sensors.PICKUPS.extract(groupedCoin)
assert(contactRow.entity_collision_class==0 and contactRow.collision_radius==10,
       'pickup collision facts must preserve zero and size')
local unknownContact={Variant=20,ToPickup=function() return {} end}
assert(SensorRegistry.sensors.PICKUPS.extract(unknownContact).entity_collision_class==nil,
       'missing collision facts must remain unknown')
local function command(seq,room,deadline)
 return {move={x=1,y=0},shoot={x=0,y=-1},use_bomb=true,agent_seq=seq,
         agent_room=room,agent_level="42:1:0",agent_deadline=deadline}
end
InputExecutor.applyCommand(command(1,1,16))
assert(InputExecutor.moveDirection.x==1 and not InputExecutor.useBomb,
       "unbound bomb input must be rejected")
State.updateCount=11; AgentWatchdog()
assert(InputExecutor.moveDirection.x==1 and not InputExecutor.useBomb, "bomb must release after one update")
State.updateCount=16; AgentWatchdog()
assert(InputExecutor.moveDirection.x==0, "expired input must release")
InputExecutor.applyCommand(command(2,2,20))
assert(InputExecutor.moveDirection.x==0, "old room input must be rejected")
InputExecutor.applyCommand(command(2,1,20))
assert(InputExecutor.moveDirection.x==1)
wall=500; AgentWallWatchdog()
assert(InputExecutor.moveDirection.x==0, "wall-clock lease must expire during pause")
InputExecutor.applyCommand(command(1,1,20))
assert(InputExecutor.moveDirection.x==0, "reordered input must be rejected")
InputExecutor.applyCommand(command(3,1,20))
Network.disconnect()
assert(InputExecutor.moveDirection.x==0, "disconnect must release")
State.connected=true; State.updateCount=17
local use=command(4,1,23); use.use_item=true; use.agent_resource_kind="item"; use.agent_resource_id=45
InputExecutor.applyCommand(use)
assert(InputExecutor.useItem, "matched active item must receive a use pulse")
State.updateCount=18; AgentWatchdog()
assert(not InputExecutor.useItem, "item pulse must release")
use=command(5,1,24); use.use_pill=true; use.agent_resource_kind="pill"; use.agent_resource_id=3
pill=4; InputExecutor.applyCommand(use)
assert(not InputExecutor.usePill, "replacement pill must not be consumed")
local inventory=CustomCollectors.PLAYER_INVENTORY()[1]
assert(inventory.can_pickup_item==true,'native pickup eligibility must be reported')
canPickup=false
assert(CustomCollectors.PLAYER_INVENTORY()[1].can_pickup_item==false,'false pickup eligibility must be explicit')
local canPickupMethod=player.CanPickupItem;player.CanPickupItem=nil
assert(CustomCollectors.PLAYER_INVENTORY()[1].can_pickup_item==nil,'missing pickup method is unknown')
player.CanPickupItem=canPickupMethod;canPickup=true
assert(CustomCollectors.PLAYER_INVENTORY()[1].poop_explosions==false,'Brown Cap false must be observed')
brownCap=true
assert(CustomCollectors.PLAYER_INVENTORY()[1].poop_explosions==true,'effective Brown Cap must be observed')
brownCap=false
assert(inventory.active_items['0'].needs_charge==true and inventory.active_items['1'].needs_charge==false,
       "charge eligibility must use each numeric slot")
chargeNeeded[0]=false;chargeNeeded[2]=true
local changedCharge=CustomCollectors.PLAYER_INVENTORY()[1]
assert(changedCharge.active_items['0'].needs_charge==false and changedCharge.active_items['2'].needs_charge==true,
       "charge eligibility changes must be explicit")
local needsCharge=player.NeedsCharge;player.NeedsCharge=nil
assert(CustomCollectors.PLAYER_INVENTORY()[1].active_items['0'].needs_charge==nil,
       "unavailable native charge check must not invent a result")
player.NeedsCharge=needsCharge
assert(inventory.golden_key==true, "golden key state must be transmitted")
goldenKey=false
assert(CustomCollectors.PLAYER_INVENTORY()[1].golden_key==false, "golden key loss must be explicit")
assert(inventory.pill_identified==false and inventory.pill_effect==nil and effectReads==0,
       "unknown pill effects must not be inspected")
identified=true; inventory=CustomCollectors.PLAYER_INVENTORY()[1]
assert(inventory.pill_effect==5 and effectReads==1, "identified effect should be reported")
local queued={"8", "9", "10"}
json.decode=function(line) return command(tonumber(line),1,24) end
State.socket={receive=function()
    if #queued>0 then return table.remove(queued,1) end
    return nil,"timeout",""
end}
local latest=Network.receive()
assert(latest.agent_seq==10 and #queued==0, "queued motion must coalesce to newest input")
InputExecutor.applyCommand(latest)
local ack=Protocol.createDataMessage().agent
assert(ack.last_input_sequence==10 and ack.input_applied_frame==18 and ack.safety_version==3,
       "state must acknowledge actual accepted input")
State.updateCount=25
InputExecutor.applyCommand(command(11,1,24))
assert(Protocol.createDataMessage().agent.rejected_expired==1, "expired input must be observable")
assert(Protocol.createDataMessage().agent.rock_bomb_protocol==1)
local seq=11
local function rockCommand(id)
 seq=seq+1
 local c=command(seq,1,State.updateCount+6)
 c.agent_bomb_id=id; c.agent_bomb_kind="rock";c.agent_bomb_grid=67
 c.agent_bomb_grid_type=4;c.agent_bombs_before=4;c.agent_bomb_spot={x=365,y=280}
 return c
end
InputExecutor.applyCommand(rockCommand("rock1"))
assert(InputExecutor.useBomb,"matched observed rock should receive one pulse")
State.updateCount=26;AgentWatchdog()
assert(not InputExecutor.useBomb,"rock bomb pulse must release")
InputExecutor.applyCommand(rockCommand("rock1"))
assert(not InputExecutor.useBomb,"replayed request ID must not spend another bomb")
local c=rockCommand("wrongtype");rockType=2;InputExecutor.applyCommand(c)
assert(not InputExecutor.useBomb,"ordinary rock cannot inherit marked-rock permission")
assert(Protocol.createDataMessage().agent.resource_rock_protocol==1)
c=rockCommand("resource-rock");c.agent_bomb_kind="resource_rock";c.agent_bomb_grid_type=2
InputExecutor.applyCommand(c)
assert(InputExecutor.useBomb,"ordinary rock requires its independent typed permission")
State.updateCount=State.updateCount+1;AgentWatchdog()
c=rockCommand("resource-rock");c.agent_bomb_kind="resource_rock";c.agent_bomb_grid_type=2
InputExecutor.applyCommand(c)
assert(not InputExecutor.useBomb,"resource rock permission is a single consumed request")
c=rockCommand("resource-wrong-type");c.agent_bomb_kind="resource_rock";c.agent_bomb_grid_type=2;rockType=4
InputExecutor.applyCommand(c)
assert(not InputExecutor.useBomb,"resource opening cannot silently target a marked rock")
rockType=4;c=rockCommand("destroyed");rock.CollisionClass=0;InputExecutor.applyCommand(c)
assert(not InputExecutor.useBomb,"destroyed rock must not receive another bomb")
rock.CollisionClass=3;c=rockCommand("staleinv");bombs=3;InputExecutor.applyCommand(c)
assert(not InputExecutor.useBomb,"inventory must still match")
bombs=4;c=rockCommand("badspot");c.agent_bomb_spot={x=500,y=280};InputExecutor.applyCommand(c)
assert(not InputExecutor.useBomb,"remote or arbitrary placement must be rejected")
c=rockCommand("moving");player.Velocity=Vector(1,0);InputExecutor.applyCommand(c)
assert(not InputExecutor.useBomb,"bomb must wait for the player to stop")
player.Velocity=Vector(0,0);rockType=22;c=rockCommand("super1");c.agent_bomb_grid_type=22
InputExecutor.applyCommand(c);assert(InputExecutor.useBomb,"super rock supports separately bound pulses")
c=rockCommand("wall1");c.agent_bomb_kind=nil;c.agent_bomb_slot=0;player.Position=Vector(80,280)
InputExecutor.applyCommand(c);assert(InputExecutor.useBomb,"existing secret wall contract still works")
goldenBomb=true;bombs=0;rockType=4;player.Position=Vector(365,280)
assert(CustomCollectors.PLAYER_INVENTORY()[1].golden_bomb==true)
c=rockCommand("gold-rock");c.agent_bomb_payment="golden";c.agent_bombs_before=0
InputExecutor.applyCommand(c);assert(InputExecutor.useBomb,"golden rock pulse works with zero stock")
local bombAck=Protocol.createDataMessage().agent
assert(bombAck.golden_bomb_protocol==1 and bombAck.last_bomb_request=="gold-rock"
       and bombAck.last_bomb_frame==State.updateCount,"accepted bomb request must be observable")
State.updateCount=State.updateCount+1;AgentWatchdog();assert(not InputExecutor.useBomb)
InputExecutor.applyCommand(c);assert(not InputExecutor.useBomb,"golden pulse request cannot replay")
c=rockCommand("gold-wall");c.agent_bomb_kind=nil;c.agent_bomb_slot=0
c.agent_bomb_payment="golden";c.agent_bombs_before=0;player.Position=Vector(80,280)
InputExecutor.applyCommand(c);assert(InputExecutor.useBomb,"golden wall pulse works with zero stock")
goldenBomb=false;bombs=4
assert(CustomCollectors.PLAYER_INVENTORY()[1].golden_bomb==false)
c=rockCommand("gold-lost");c.agent_bomb_payment="golden"
c.agent_bomb_kind=nil;c.agent_bomb_slot=0
InputExecutor.applyCommand(c);assert(not InputExecutor.useBomb,"free authorization cannot consume ordinary stock")
assert(Protocol.createDataMessage().agent.last_bomb_request=="gold-wall","rejection must not forge an acknowledgement")
c=rockCommand("bad-payment");c.agent_bomb_kind=nil;c.agent_bomb_slot=0;c.agent_bomb_payment="anything"
InputExecutor.applyCommand(c);assert(not InputExecutor.useBomb,"unknown payment must not execute")
local unpausedGame=Game
Game=function() local g=unpausedGame(); g.IsPaused=function() return true end; return g end
local status
Network.receive=function() return {command="SET_CONTROL_MODE",params={mode="MANUAL"}} end
Network.send=function(message) status=message end
CommandHandler={process=function(command) State.controlMode=command.params.mode end}
State.connected=true; State.controlMode="FORCE_AI"
AgentWallWatchdog()
assert(State.controlMode=="MANUAL" and status.paused==true,
       "paused game must process manual release and report pause status")
'''
lua.execute(stub + (ROOT / 'scripts/agent_bridge.lua').read_text() + assertions)
print('Lua 5.3: mod syntax, input leases, resource pulses/replacement guards, and pill visibility passed')
