# 安装与运行

## 环境

- Python 3.11+；Windows版Repentance，支持LuaSocket的模组环境。
- 本轮实测版本为Repentance v1.7.9b.J835、普通Magdalena/Normal。未验证其他角色和Repentance+的完整行为。
- 推荐直接从源码目录运行。核心无需第三方Python包，录像依赖单独安装。

## 模组

`python scripts/build_mod.py`校验`vendor/socketbridge/UPSTREAM.json`中的哈希，生成独立模组和zip。找到游戏实际Modding Data Path后使用：

```bash
python scripts/install_mod.py --mods-dir '/path/to/actual/game/mods'
```

脚本遇到同名已安装目录会拒绝覆盖。安装后正常重启游戏，并带`--luadebug`启动。先以`--observe-only`确认握手，再测试`--mode local`。请为自己的存档保留备份。

## API与本地设置

复制`config.example.toml`为`config.local.toml`。Sol端点必须是完整的Chat Completions兼容URL，模型访问取决于所用服务。示例localhost网关需另行部署。

JEV示例使用OpenRouter Decisions与`OPENROUTER_API_KEY`。`--jev-scope off`不请求JEV。环境变量优先于`.secrets/api-keys.json`。Windows配置助手保存到该检出目录；Linux/WSL建议通过终端的隐藏输入设置环境变量，切勿截图密钥输入。

可选EID知识读取：将`ISAAC_EID_DIR`指向自己安装的EID目录。未设置则使用内置名称和简报，不尝试开发者电脑的安装路径。

## Windows与WSL

Windows下使用`python`或`py -3`；WSL下使用`python3`。桥接默认监听127.0.0.1:9527。若跨系统回环不可用，先在Windows运行控制器作对照；不要公开暴露桥接端口。

可选`--pause-on-stop`：Windows用当前解释器运行窗口助手；WSL通过`wslpath`转换脚本路径，从PATH查找`python.exe`。找不到时设置`ISAAC_WINDOWS_PYTHON`为WSL可执行的Windows Python路径。未配置助手会记录失败并继续释放输入；该选项发送Escape，请在正常游戏画面使用，避免已经手动暂停时切换状态。

## 失焦与停止

需要后台运行时，先保存并正常退出游戏，再备份并修改实际存档目录`options.ini`的`PauseOnFocusLost=0`。普通窗口被遮挡与最小化不同，录制时不要最小化。

游戏内F3切回人工控制；停止控制器后输入有期限和看门狗保护。需要重新接管时重新启动live。CLI帮助：`python -m isaac_agent live --help`。
