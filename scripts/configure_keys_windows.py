"""Native masked dialog; save keys through stdin, never command arguments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

PROJECT = Path(__file__).resolve().parents[1]
SAVE = [sys.executable, str(PROJECT / 'scripts/save_keys.py')]


def backend(values=None, providers=False):
    completed = subprocess.run(SAVE + (['--providers'] if providers else ['--status'] if values is None else []),
        input=None if values is None else json.dumps(values), text=True,
        encoding='utf-8', capture_output=True, timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW)
    if completed.returncode:
        raise RuntimeError('配置保存失败，请确认 Python 和项目目录可用。')
    return json.loads(completed.stdout)


def main():
    providers = backend(providers=True)
    labels = {'OPENAI_API_KEY': 'Sol / OpenAI 兼容接口',
              'TYPESAFE_API_KEY': 'TypeSafe 官方', 'OPENROUTER_API_KEY': 'OpenRouter'}
    # This machine has python.exe in C:\Python314 and Tcl under LOCALAPPDATA.
    # Set paths for this process only when those resources exist.
    version = f'Python{sys.version_info.major}{sys.version_info.minor}'
    tcl_root = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python' / version / 'tcl'
    for variable, relative in [('TCL_LIBRARY', 'tcl8.6'), ('TK_LIBRARY', 'tk8.6')]:
        if (tcl_root / relative).is_dir():
            os.environ.setdefault(variable, str(tcl_root / relative))
    window = tk.Tk()
    window.title('以撒 Agent — API 密钥配置')
    window.geometry('610x390')
    window.resizable(False, False)
    window.attributes('-topmost', True)
    frame = ttk.Frame(window, padding=24)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='配置 Sol 和 JEV', font=('Microsoft YaHei UI', 16)).pack(anchor='w')
    ttk.Label(frame, text='按当前项目配置保存对应平台的密钥。\n输入只保存在此项目的私有文件中，不显示在聊天或日志。',
              font=('Microsoft YaHei UI', 10), wraplength=550).pack(anchor='w', pady=(8, 12))
    entries = {}
    for lane, title in [('astra', 'Sol'), ('jev', 'JEV')]:
        name = providers[lane]
        if name in entries:
            continue
        label = f'{title} — {labels.get(name, name)} API Key'
        ttk.Label(frame, text=label).pack(anchor='w', pady=(7, 3))
        entry = ttk.Entry(frame, show='●', width=72)
        entry.pack(fill='x')
        entries[name] = entry
    status = tk.StringVar(value='已有密钥时，留空会保留原配置。保存不会发起付费模型请求。')
    ttk.Label(frame, textvariable=status, wraplength=550).pack(anchor='w', pady=12)

    def save():
        try:
            result = backend({name: entry.get() for name, entry in entries.items()})
            for entry in entries.values():
                entry.delete(0, 'end')
            missing = [name for name in entries if not result.get(name)]
            if missing:
                status.set('已保存，仍缺少：' + '、'.join(missing))
            else:
                messagebox.showinfo('已保存', '两项密钥已保存。下次启动模型模式会自动读取。\n密钥可用性尚未通过网络验证。', parent=window)
                window.destroy()
        except Exception:
            messagebox.showerror('保存失败', '无法保存配置，请确认 Python 可用且输入中没有空格或换行。', parent=window)

    ttk.Button(frame, text='保存配置', command=save).pack(anchor='e')
    next(iter(entries.values())).focus_set()
    window.mainloop()


if __name__ == '__main__':
    main()
