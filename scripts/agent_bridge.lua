-- Local changes to MIT-licensed SocketBridge. Inserted by build_mod.py.
-- All gameplay continues through ordinary input hooks.
local Agent = { deadline = -1, pulseFrame = -1, lastSequence = -1, lastWall = -1,
                partial = "", wasDead = false, appliedFrame = -1,
                pendingCommands = {}, pendingInput = nil, rejectedExpired = 0,
                lastRetryWall = -10000, lastPauseStatus = -10000, bombRequests = {}, continued = false }

local function AgentLevelId()
    local game, level = Game(), Game():GetLevel()
    return tostring(game:GetSeeds():GetStartSeed()) .. ":" ..
           tostring(level:GetStage()) .. ":" .. tostring(level:GetStageType())
end

local oldDataMessage = Protocol.createDataMessage
Protocol.createDataMessage = function(...)
    local msg = oldDataMessage(...)
    msg.agent = { protocol = 1, level_id = AgentLevelId(),
                  control_mode = State.controlMode, repentance_plus = REPENTANCE_PLUS == true,
                  safety_version = 3, last_input_sequence = Agent.lastSequence,
                  input_applied_frame = Agent.appliedFrame, input_deadline = Agent.deadline,
                  rejected_expired = Agent.rejectedExpired, secret_bomb_protocol = 1, rock_bomb_protocol = 1,
                  resource_rock_protocol = 1, golden_bomb_protocol = 1, last_bomb_request = Agent.lastBombRequest,
                  last_bomb_frame = Agent.lastBombFrame, continued_run = Agent.continued }
    return msg
end
local oldFullMessage = Protocol.createFullStateMessage
Protocol.createFullStateMessage = function(...)
    local msg = oldFullMessage(...)
    msg.room_index = Game():GetLevel():GetCurrentRoomIndex()
    msg.agent = { protocol = 1, level_id = AgentLevelId(), control_mode = State.controlMode,
                  safety_version = 3, last_input_sequence = Agent.lastSequence,
                  input_applied_frame = Agent.appliedFrame, input_deadline = Agent.deadline,
                  rejected_expired = Agent.rejectedExpired, secret_bomb_protocol = 1, rock_bomb_protocol = 1,
                  resource_rock_protocol = 1, golden_bomb_protocol = 1, last_bomb_request = Agent.lastBombRequest,
                  last_bomb_frame = Agent.lastBombFrame, continued_run = Agent.continued,
                  repentance_plus = REPENTANCE_PLUS == true }
    return msg
end

-- Always publish complete fast sensor snapshots, including empty lists.
for name, sensor in pairs(SensorRegistry.sensors) do
    sensor.cache.strategy = "none"
    local fast = name == "PLAYER_POSITION" or name == "ENEMIES" or name == "PROJECTILES"
                 or name == "BOMBS" or name == "PLAYER_HEALTH"
    local terrain = name == "ROOM_LAYOUT" or name == "FIRE_HAZARDS"
    sensor.throttle = { base_interval = fast and 1 or (terrain and 2 or 5), dynamic = false }
end

-- Only report a pill's effect once the game says it has been identified.
if CustomCollectors then
    local oldRoomLayout = CustomCollectors.ROOM_LAYOUT
    CustomCollectors.ROOM_LAYOUT = function(...)
        local data = oldRoomLayout(...)
        local room = Game():GetRoom()
        if not data or not room then return data end
        -- Hidden doors can exist internally before discovery. Do not expose
        -- their target room/type as evidence for a wall-bomb decision.
        for slot, door in pairs(data.doors or {}) do
            if (door.target_room_type == RoomType.ROOM_SECRET or door.target_room_type == RoomType.ROOM_SUPERSECRET)
                and not door.is_open then data.doors[slot] = nil end
        end
        data.wall_slots = {}
        for slot = 0, 3 do
            if room:IsDoorSlotAllowed(slot) then
                data.wall_slots[tostring(slot)] = Helpers.vectorToTable(room:GetDoorSlotPosition(slot))
            end
        end
        return data
    end
    local oldRoomInfo = CustomCollectors.ROOM_INFO
    CustomCollectors.ROOM_INFO = function(...)
        local data = oldRoomInfo(...)
        if data then data.curses = Game():GetLevel():GetCurses() end
        return data
    end
    local oldInventory = CustomCollectors.PLAYER_INVENTORY
    CustomCollectors.PLAYER_INVENTORY = function(...)
        local data = oldInventory(...)
        for i, player in ipairs(Helpers.getPlayers()) do
            local row = data[i]
            row.can_use = player:IsExtraAnimationFinished() and not player:IsDead()
            if player.CanPickupItem then row.can_pickup_item = player:CanPickupItem() end
            if player.HasTrinket then row.poop_explosions = player:HasTrinket(90) end -- Brown Cap, including swallowed effects
            if player.HasGoldenKey then row.golden_key = player:HasGoldenKey() end
            if player.HasGoldenBomb then row.golden_bomb = player:HasGoldenBomb() end
            local color = row.pill_0 or 0
            if color > 0 then
                local pool = Game():GetItemPool()
                row.pill_identified = pool:IsPillIdentified(color)
                if row.pill_identified then row.pill_effect = pool:GetPillEffect(color, player) end
            end
            for slot, active in pairs(row.active_items or {}) do
                if player.NeedsCharge then active.needs_charge = player:NeedsCharge(tonumber(slot)) end
                local config = Isaac.GetItemConfig():GetCollectible(active.item)
                if config then
                    active.name = config.Name; active.description = config.Description
                    active.charge_type = config.ChargeType
                end
            end
        end
        return data
    end
end

local pickupSensor = SensorRegistry.sensors.PICKUPS
if pickupSensor then
    local oldPickupExtract = pickupSensor.extract
    pickupSensor.extract = function(entity, player)
        local row = oldPickupExtract(entity, player)
        if row then
            local pickup = entity:ToPickup()
            if pickup then
                row.options_index = pickup.OptionsPickupIndex
                row.entity_collision_class = pickup.EntityCollisionClass
                row.collision_radius = pickup.Size
            end
        end
        if row and row.variant == 100 and row.sub_type > 0 then
            local config = Isaac.GetItemConfig():GetCollectible(row.sub_type)
            if config then
                row.item_type = config.Type; row.description = config.Description
            end
        end
        return row
    end
end

local fireSensor = SensorRegistry.sensors.FIRE_HAZARDS
if fireSensor then
    local oldFireExtract = fireSensor.extract
    fireSensor.extract = function(entity, player)
        -- Hostile ground creep; player-creep variants are deliberately excluded.
        local ground = { [22]=true, [23]=true, [24]=true, [25]=true, [26]=true,
                         [55]=true, [56]=true, [94]=true, [155]=true, [169]=true, [174]=true }
        if entity.Type == EntityType.ENTITY_EFFECT and ground[entity.Variant] then
            local scale = math.max(math.abs(entity.SpriteScale.X), math.abs(entity.SpriteScale.Y))
            return { id=entity.Index, type="GROUND", variant=entity.Variant,
                     pos=Helpers.vectorToTable(entity.Position), ground_only=true,
                     collision_radius=math.max(entity.Size, 24 * scale) }
        end
        return oldFireExtract(entity, player)
    end
end

local oldApply = InputExecutor.applyCommand
InputExecutor.applyCommand = function(command)
    if not command or not command.move then return end
    if type(command.agent_deadline) ~= "number" or type(command.agent_seq) ~= "number" then return end
    if command.agent_seq <= Agent.lastSequence then return end
    if command.agent_room ~= Game():GetLevel():GetCurrentRoomIndex() then return end
    if command.agent_level ~= AgentLevelId() then return end
    if command.agent_deadline <= State.updateCount or command.agent_deadline > State.updateCount + 12 then
        Agent.rejectedExpired = Agent.rejectedExpired + 1
        return
    end
    for _, key in ipairs({"move", "shoot"}) do
        local v = command[key]
        if type(v) ~= "table" then return end
        for _, axis in ipairs({"x", "y"}) do
            if v[axis] ~= -1 and v[axis] ~= 0 and v[axis] ~= 1 then return end
        end
    end
    Agent.lastSequence = command.agent_seq
    Agent.appliedFrame = State.updateCount
    Agent.deadline, Agent.lastWall = command.agent_deadline, Isaac.GetTime()
    Agent.pulseFrame = State.updateCount
    local player = Isaac.GetPlayer(0)
    local kind, id = command.agent_resource_kind, command.agent_resource_id
    -- Match the selected pocket/active slot. Never consume a replacement.
    command.use_item = command.use_item == true and kind == "item" and player and player:GetActiveItem(0) == id
    command.use_pill = command.use_pill == true and kind == "pill" and player and player:GetPill(0) == id
    command.use_card = command.use_card == true and kind == "card" and player and player:GetCard(0) == id
    local bombAllowed = false
    if command.use_bomb == true and player and type(command.agent_bomb_id) == "string"
        and #command.agent_bomb_id > 0 and #command.agent_bomb_id <= 128
        and not Agent.bombRequests[command.agent_bomb_id] then
        local room, slot = Game():GetRoom(), command.agent_bomb_slot
        local hasGolden = player.HasGoldenBomb and player:HasGoldenBomb() == true
        local payment = command.agent_bomb_payment
        local paymentAllowed = (payment == "golden" and hasGolden)
            or ((payment == nil or payment == "normal") and not hasGolden and player:GetNumBombs() >= 2)
        paymentAllowed = paymentAllowed and player:GetNumBombs() == command.agent_bombs_before
        if (command.agent_bomb_kind == "rock" or command.agent_bomb_kind == "resource_rock")
            and room:GetRoomShape() == RoomShape.ROOMSHAPE_1x1 and room:IsClear()
            and room:GetAliveEnemiesCount() == 0 and paymentAllowed then
            local index, pos = command.agent_bomb_grid, command.agent_bomb_spot
            if type(index) == "number" and index == math.floor(index) and index >= 0
                and index < room:GetGridSize() and type(pos) == "table"
                and type(pos.x) == "number" and type(pos.y) == "number" then
                local grid = room:GetGridEntity(index)
                if grid and ((command.agent_bomb_kind == "rock" and (grid:GetType() == 4 or grid:GetType() == 22))
                    or (command.agent_bomb_kind == "resource_rock" and grid:GetType() == 2))
                    and grid:GetType() == command.agent_bomb_grid_type and grid.CollisionClass ~= 0 then
                    local center, spot = grid.Position, Vector(pos.x,pos.y)
                    local dx, dy = math.abs(spot.X-center.X), math.abs(spot.Y-center.Y)
                    local cardinal = (math.abs(dx-45) < .1 and dy < .1)
                                     or (math.abs(dy-45) < .1 and dx < .1)
                    bombAllowed = cardinal and player.Position:Distance(spot) <= 8
                                  and player.Velocity:Length() < .4
                end
            end
        elseif command.agent_bomb_kind == nil and type(slot) == "number" and slot >= 0 and slot <= 3 and slot == math.floor(slot)
            and room:GetRoomShape() == RoomShape.ROOMSHAPE_1x1 and room:IsClear()
            and room:GetAliveEnemiesCount() == 0 and room:IsDoorSlotAllowed(slot)
            and paymentAllowed then
            local inward = { Vector(1,0), Vector(0,1), Vector(-1,0), Vector(0,-1) }
            local spot = room:GetDoorSlotPosition(slot) + inward[slot+1]*40
            bombAllowed = player.Position:Distance(spot) <= 8 and player.Velocity:Length() < .4
        end
        if bombAllowed then
            Agent.bombRequests[command.agent_bomb_id] = true
            Agent.lastBombRequest, Agent.lastBombFrame = command.agent_bomb_id, State.updateCount
        end
    end
    command.use_bomb = bombAllowed
    oldApply(command)
end

function AgentWallWatchdog()
    if Agent.lastWall >= 0 and Isaac.GetTime() - Agent.lastWall > 350 then
        InputExecutor.reset()
    end
    -- MC_POST_UPDATE stops in the pause menu. Keep connection/control-mode
    -- maintenance on render so stopping a controller can release FORCE_AI even
    -- while paused, and a replacement controller can connect before resume.
    local game = Game()
    if game.IsPaused and game:IsPaused() then
        InputExecutor.reset()
        if not State.connected then Network.connect() end
        if State.connected then
            local command = Network.receive()
            if command and command.command then CommandHandler.process(command) end
            if Isaac.GetTime() - Agent.lastPauseStatus >= 500 then
                Agent.lastPauseStatus = Isaac.GetTime()
                Network.send({ type="STATUS", paused=true, frame=State.updateCount,
                               level_id=AgentLevelId(), control_mode=State.controlMode })
            end
        end
    end
end

function AgentWatchdog()
    local room = Game():GetLevel():GetCurrentRoomIndex()
    if not State.connected or State.updateCount >= Agent.deadline or room ~= State.currentRoom then
        InputExecutor.reset()
    end
    if State.updateCount > Agent.pulseFrame then
        InputExecutor.useItem = false; InputExecutor.useBomb = false
        InputExecutor.useCard = false; InputExecutor.usePill = false; InputExecutor.drop = false
    end
    AgentWallWatchdog()
    local player = Isaac.GetPlayer(0)
    local dead = player and player:IsDead() or false
    if dead then InputExecutor.reset() end
    if dead and not Agent.wasDead then EventSystem.emit("PLAYER_DEATH", {}) end
    Agent.wasDead = dead
end

local oldDisconnect = Network.disconnect
Network.disconnect = function()
    InputExecutor.reset()
    Agent.lastSequence, Agent.deadline, Agent.partial = -1, -1, ""
    Agent.pendingCommands, Agent.pendingInput = {}, nil
    oldDisconnect()
end

-- LuaSocket returns nil,error on refusal; pcall success alone is insufficient.
Network.connect = function()
    if State.connected then return true end
    if Isaac.GetTime() - Agent.lastRetryWall < 1000 then return false end
    Agent.lastRetryWall = Isaac.GetTime()
    Network.lastRetryFrame = State.updateCount
    local ok, socket = pcall(require, "socket.core")
    if not ok then
        print("[AstraJev] LuaSocket unavailable; set Steam launch option --luadebug")
        return false
    end
    local tcp = socket.tcp()
    tcp:settimeout(0.01)
    local connected = tcp:connect(Config.HOST, Config.PORT)
    if not connected then tcp:close(); return false end
    tcp:settimeout(0)
    State.socket, State.connected = tcp, true
    Agent.lastSequence, Agent.partial = -1, ""
    Agent.pendingCommands, Agent.pendingInput = {}, nil
    State.controlMode = "MANUAL"
    return true
end

Network.receive = function()
    if not State.connected then return nil end
    -- Drain a bounded batch and apply only the freshest motion. One old command
    -- per update cannot catch up when a pause/search has queued several frames.
    for i = 1, 32 do
        local line, err, partial = State.socket:receive("*l")
        if line then
            line = Agent.partial .. line
            Agent.partial = ""
            local ok, decoded = pcall(json.decode, line)
            if ok and type(decoded) == "table" then
                if decoded.command then
                    -- This agent needs control mode and read-only state queries.
                    if decoded.command == "SET_CONTROL_MODE" or decoded.command == "GET_FULL_STATE"
                       or decoded.command == "GET_CONTROL_MODE" then
                        if #Agent.pendingCommands < 8 then table.insert(Agent.pendingCommands, decoded) end
                    end
                elseif decoded.move then
                    Agent.pendingInput = decoded
                end
            end
        else
            Agent.partial = Agent.partial .. (partial or "")
            if err == "closed" or #Agent.partial > 1048576 then Network.disconnect() end
            break
        end
    end
    if #Agent.pendingCommands > 0 then return table.remove(Agent.pendingCommands, 1) end
    local newest = Agent.pendingInput
    Agent.pendingInput = nil
    return newest
end

Network.send = function(data)
    if not State.connected then return false end
    local payload = json.encode(data) .. "\n"
    local sent = State.socket:send(payload)
    -- Never continue a truncated JSON stream after a failed write.
    if not sent or sent ~= #payload then Network.disconnect(); return false end
    return true
end

AgentAddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, continued)
    Agent.deadline, Agent.lastSequence, Agent.wasDead = -1, -1, false
    Agent.bombRequests, Agent.continued = {}, continued == true
    Agent.lastBombRequest, Agent.lastBombFrame = nil, nil
    Network.lastRetryFrame = -60
end)

AgentAddCallback(ModCallbacks.MC_USE_ITEM, function(_, item)
    EventSystem.emit("RESOURCE_USED", { kind = "item", id = item })
end)
AgentAddCallback(ModCallbacks.MC_USE_PILL, function(_, effect)
    EventSystem.emit("RESOURCE_USED", { kind = "pill", effect = effect })
end)
AgentAddCallback(ModCallbacks.MC_USE_CARD, function(_, card)
    EventSystem.emit("RESOURCE_USED", { kind = "card", id = card })
end)
