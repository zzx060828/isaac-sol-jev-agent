"""Capture one Isaac window frame without foreground activation or desktop pixels."""
import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys
import threading

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'.tools/recording-python'))
import cv2
from windows_capture import WindowsCapture

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
a=p.parse_args()
if a.output.exists():p.error('Use a new output filename')
u=ctypes.WinDLL('user32',use_last_error=True)
u.FindWindowW.argtypes=[wintypes.LPCWSTR,wintypes.LPCWSTR];u.FindWindowW.restype=wintypes.HWND
u.IsIconic.argtypes=[wintypes.HWND]
h=u.FindWindowW(None,'Binding of Isaac: Repentance')
if not h or u.IsIconic(h):raise RuntimeError('Isaac window missing or minimized')
ready=threading.Event();data={}
capture=WindowsCapture(window_hwnd=h,cursor_capture=False,draw_border=False)
@capture.event
def on_frame_arrived(frame,control):
    if not ready.is_set():data['frame']=frame.frame_buffer.copy();ready.set()
@capture.event
def on_closed():pass
control=capture.start_free_threaded()
try:
    if not ready.wait(8):raise RuntimeError('No game frame received')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(a.output),data['frame']):raise RuntimeError('Snapshot save failed')
finally:
    control.stop();control.wait()
print('Saved game window frame',flush=True)
