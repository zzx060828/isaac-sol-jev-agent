# 可选游戏窗口录像

在Windows Python中安装`python -m pip install -r scripts/recording-requirements.txt`，并在PATH提供ffmpeg。然后运行：

```powershell
python scripts/record_wgc_windows.py --output recordings/demo/game.mp4 --stop-file recordings/demo/stop --max-seconds 120 --fps 20
```

输出路径和stop标记必须是新的。脚本使用Windows Graphics Capture捕获游戏窗口表面，无声音；支持遮挡，但游戏不可最小化。实测标题为Binding of Isaac: Repentance；其他版本窗口标题需要核验。

结束后同时检查视频、旁边的JSON元数据、warning、frames_written和进程退出码。编码器退出0不保证整段捕获有效。不要把窗口静帧或旧帧当作游戏持续运行证明。

`capture_game_windows.py OUTPUT.png`可截取单张窗口图。录像、截图、存档和密钥不属于默认源码提交范围。
