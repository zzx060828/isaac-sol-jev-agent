"""Move Isaac off the desktop without minimizing; target only its window for menu keys."""
import argparse
import ctypes
from ctypes import wintypes as W
import json
from pathlib import Path
import time

u = ctypes.WinDLL('user32', use_last_error=True)
u.SetProcessDPIAware()
class Rect(ctypes.Structure):
    _fields_ = [(k, W.LONG) for k in ('left', 'top', 'right', 'bottom')]
u.FindWindowW.argtypes=[W.LPCWSTR,W.LPCWSTR]; u.FindWindowW.restype=W.HWND
u.GetForegroundWindow.restype=W.HWND
u.GetShellWindow.restype=W.HWND
u.GetWindowRect.argtypes=[W.HWND,ctypes.POINTER(Rect)]
u.IsIconic.argtypes=[W.HWND]; u.IsWindowVisible.argtypes=[W.HWND]
u.SetWindowPos.argtypes=[W.HWND,W.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,W.UINT]
u.SetForegroundWindow.argtypes=[W.HWND]
u.PostMessageW.argtypes=[W.HWND,W.UINT,W.WPARAM,W.LPARAM]
u.MapVirtualKeyW.argtypes=[W.UINT,W.UINT];u.MapVirtualKeyW.restype=W.UINT

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('action',choices=['status','park','restore','key'])
parser.add_argument('--state',type=Path,default=Path(__file__).resolve().parents[1]/'runs/background-window.json')
parser.add_argument('--key',choices=['Enter','Escape','Space','Up','Down','Left','Right'],default='Enter')
parser.add_argument('--hold-ms',type=int,default=80)
a=parser.parse_args()
h=u.FindWindowW(None,'Binding of Isaac: Repentance')
if not h: raise RuntimeError('Isaac window not found')
r=Rect()
if not u.GetWindowRect(h,ctypes.byref(r)): raise ctypes.WinError(ctypes.get_last_error())
virtual=dict(x=u.GetSystemMetrics(76),y=u.GetSystemMetrics(77),width=u.GetSystemMetrics(78),height=u.GetSystemMetrics(79))
if a.action=='park':
    if u.IsIconic(h):
        u.ShowWindowAsync.argtypes=[W.HWND,ctypes.c_int]
        u.ShowWindowAsync(h,4)  # SW_SHOWNOACTIVATE: restore without taking focus.
        deadline=time.monotonic()+2
        while u.IsIconic(h) and time.monotonic()<deadline:time.sleep(.05)
        if u.IsIconic(h):raise RuntimeError('Game did not restore without activation')
        u.GetWindowRect(h,ctypes.byref(r))
    a.state.parent.mkdir(parents=True,exist_ok=True)
    if not a.state.exists():
        a.state.write_text(json.dumps(dict(hwnd=h,rect=[r.left,r.top,r.right,r.bottom])))
    # SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE. No global keyboard injection.
    if not u.SetWindowPos(h,None,virtual['x']+virtual['width']+64,virtual['y']+64,0,0,0x15):
        raise ctypes.WinError(ctypes.get_last_error())
    if u.GetForegroundWindow()==h:
        u.SetForegroundWindow(u.GetShellWindow())
elif a.action=='restore':
    saved=json.loads(a.state.read_text())
    left,top,right,bottom=saved['rect']
    width,height=right-left,bottom-top
    if left>=virtual['x']+virtual['width'] or top>=virtual['y']+virtual['height'] or right<=virtual['x'] or bottom<=virtual['y']:
        left,top=virtual['x']+64,virtual['y']+64
    if not u.SetWindowPos(h,None,left,top,width,height,0x14):
        raise ctypes.WinError(ctypes.get_last_error())
elif a.action=='key':
    if not 50<=a.hold_ms<=1000: raise ValueError('Invalid hold interval')
    vk={'Enter':13,'Escape':27,'Space':32,'Up':38,'Down':40,'Left':37,'Right':39}[a.key]
    scan=u.MapVirtualKeyW(vk,0)
    bits=1|(scan<<16)|(0x1000000 if a.key in ('Up','Down','Left','Right') else 0)
    if not u.PostMessageW(h,0x100,vk,bits): raise ctypes.WinError(ctypes.get_last_error())
    try: time.sleep(a.hold_ms/1000)
    finally:
        if not u.PostMessageW(h,0x101,vk,bits|0xc0000000): raise ctypes.WinError(ctypes.get_last_error())
u.GetWindowRect(h,ctypes.byref(r))
print(json.dumps(dict(action=a.action,hwnd=h,foreground=u.GetForegroundWindow()==h,
                     minimized=bool(u.IsIconic(h)),visible=bool(u.IsWindowVisible(h)),
                     rect=[r.left,r.top,r.right,r.bottom],virtual_desktop=virtual)))
