"""Build an isolated SocketBridge derivative; never edits the installed game."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build():
    vendor = ROOT / "vendor" / "socketbridge"
    manifest = json.loads((vendor / "UPSTREAM.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((vendor / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Upstream checksum mismatch: {name}")
    code = (vendor / "main.lua").read_text()
    code = code.replace('RegisterMod("SocketBridge", 1)', 'RegisterMod("SocketBridge Astra JEV", 1)')
    code = code.replace('DEBUG = true,', 'DEBUG = false,')
    # Older APIs lack some optional callback enums. Core update/input hooks are
    # required and retain their ordinary registration behavior.
    code = code.replace('mod:AddCallback(', 'AgentAddCallback(')
    helper = '''local function AgentAddCallback(callback, fn, ...)
    if callback ~= nil then mod:AddCallback(callback, fn, ...) end
end

'''
    anchor = '-- MC_POST_UPDATE — main game loop (30 tps, respects pause)'
    assert code.count(anchor) == 1
    addon = (ROOT / "scripts" / "agent_bridge.lua").read_text()
    code = code.replace(anchor, helper + addon + '\n' + anchor)
    code = code.replace('State.updateCount = State.updateCount + 1',
                        'State.updateCount = State.updateCount + 1\n    AgentWatchdog()', 1)
    # One-update pulses for consumables; moving/shooting have a bounded lease.
    code = code.replace('State.renderCount = State.renderCount + 1',
                        'State.renderCount = State.renderCount + 1\n    AgentWallWatchdog()', 1)
    # A stock API can expose max charge via ItemConfig instead of this method.
    code = code.replace('player:GetActiveMaxCharge(slot)',
                        '(Isaac.GetItemConfig():GetCollectible(activeItem).MaxCharges or 0)')
    code = code.replace('player:GetPlayerIndex()', 'player.Index')
    # 733 is outside this Repentance build's valid vanilla collectible IDs;
    # querying it produced garbage inventory counts (up to hundreds of millions).
    code = code.replace('for itemId = 1, 733 do', 'for itemId = 1, CollectibleType.NUM_COLLECTIBLES - 1 do')
    # Invulnerability is not disappearance: jumping bosses and closed enemies
    # remain relevant for movement, anticipation, and knowing combat is ongoing.
    enemy_filter = 'if not entity:IsActiveEnemy(false) or not entity:IsVulnerableEnemy() then'
    assert code.count(enemy_filter) == 1
    code = code.replace(enemy_filter, 'if not entity:IsActiveEnemy(false) then')
    code = code.replace('is_boss = entity:IsBoss(),',
                        'is_boss = entity:IsBoss(),\n            is_vulnerable = entity:IsVulnerableEnemy(),\n            collision_class = entity.EntityCollisionClass,')
    # MC_ENTITY_TAKE_DMG fires before health is subtracted.
    code = code.replace('hp_after = player:GetHearts()', 'hp_before = player:GetHearts()')
    anchor = 'broken_hearts = player:GetBrokenHearts(), extra_lives = player:GetExtraLives(),'
    assert code.count(anchor) == 1
    code = code.replace(anchor, anchor + '\n            damage_cooldown = player:GetDamageCooldown(),')
    # A bullet can explode without ever touching the player. Preserve this
    # distinction without serializing the potentially large projectile bitset.
    anchor = 'falling_accel = proj and proj.FallingAccel or 0,'
    assert code.count(anchor) == 1
    code = code.replace(anchor, anchor + '''
                is_explosive = proj and proj:HasProjectileFlags(ProjectileFlags.EXPLODE) or false,
                spawner_type = entity.SpawnerType,
                spawner_variant = entity.SpawnerVariant,''')
    out = ROOT / 'build' / 'SocketBridge_AstraJev'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'main.lua').write_text(code)
    (out / 'metadata.xml').write_text('''<metadata>
  <name>SocketBridge Astra JEV</name>
  <directory>SocketBridge_AstraJev</directory>
  <description>SocketBridge with bounded input and agent state for Astra + JEV.</description>
  <version>0.1.0</version>
</metadata>
''')
    for name in ('LICENCE', 'UPSTREAM.json'):
        shutil.copy2(vendor / name, out / name)
    archive = ROOT / 'build' / 'SocketBridge_AstraJev.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.iterdir()):
            z.write(file, f'SocketBridge_AstraJev/{file.name}')
    print(archive)
    return out


if __name__ == '__main__':
    build()
