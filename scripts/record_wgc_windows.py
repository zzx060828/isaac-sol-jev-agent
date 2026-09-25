"""Capture Isaac's own surface with WGC, independent of desktop occlusion."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools' / 'recording-python'))
import cv2
import numpy as np
from windows_capture import WindowsCapture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stop-file', required=True, type=Path)
    parser.add_argument('--max-seconds', type=float, default=1200)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--verify-snapshots', action='store_true')
    parser.add_argument('--include-window-frame', action='store_true', help='Keep title bar and borders')
    args = parser.parse_args()
    if (args.output.exists() or args.stop_file.exists()
            or not 0 < args.max_seconds <= 3600 or not 1 <= args.fps <= 60):
        raise ValueError('Use a new output/stop path and valid duration/frame rate')
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    user32.SetProcessDPIAware()
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    hwnd = user32.FindWindowW(None, 'Binding of Isaac: Repentance')
    if not hwnd:
        raise RuntimeError('Isaac window not found')
    if user32.IsIconic(hwnd):
        raise RuntimeError('Isaac is minimized; restore its window before recording')
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.GetForegroundWindow.restype = wintypes.HWND
    dwm = ctypes.WinDLL('dwmapi')
    dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]

    def client_picture(picture):
        if args.include_window_frame:
            return picture
        client, bounds, origin = wintypes.RECT(), wintypes.RECT(), wintypes.POINT()
        if (not user32.GetClientRect(hwnd, ctypes.byref(client))
                or not user32.ClientToScreen(hwnd, ctypes.byref(origin))
                or dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(bounds), ctypes.sizeof(bounds)) != 0):
            raise RuntimeError('Cannot determine game client area')
        x, y = origin.x-bounds.left, origin.y-bounds.top
        width, height = client.right, client.bottom
        if x < 0 or y < 0 or x+width > picture.shape[1] or y+height > picture.shape[0]:
            raise RuntimeError('Captured window geometry does not match its client area')
        return picture[y:y+height, x:x+width]
    encoder = shutil.which('ffmpeg')
    if not encoder:
        raise RuntimeError('FFmpeg not found on Windows PATH')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock, closed, ready = threading.Lock(), threading.Event(), threading.Event()
    state = {'image': None, 'updates': 0, 'received_at': None}
    capture = WindowsCapture(window_hwnd=hwnd, cursor_capture=False, draw_border=False,
                             minimum_update_interval=round(1000/args.fps))

    @capture.event
    def on_frame_arrived(frame, capture_control):
        with lock:
            state['image'] = client_picture(frame.frame_buffer).copy()
            state['updates'] += 1
            state['received_at'] = time.monotonic()
        ready.set()

    @capture.event
    def on_closed():
        closed.set()

    control = capture.start_free_threaded()
    process = None
    frames, samples = 0, []
    report = {'backend': 'windows_graphics_capture', 'window_hwnd': hwnd, 'fps': args.fps,
              'audio': False, 'capture_package': 'windows-capture==2.0.1',
              'client_area_only': not args.include_window_frame}
    try:
        if not ready.wait(10):
            raise RuntimeError('No WGC frame received within 10 seconds')
        with lock:
            height, width = state['image'].shape[:2]
        report.update(width=width, height=height, started_at_unix=time.time())
        command = [encoder, '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo',
                   '-pixel_format', 'bgra', '-video_size', f'{width}x{height}',
                   '-framerate', str(args.fps), '-i', 'pipe:0', '-an',
                   '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
                   '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p', '-n', str(args.output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'recording': str(args.output), **report}), flush=True)
        started = time.monotonic()
        while not closed.is_set() and not args.stop_file.exists() and time.monotonic()-started < args.max_seconds:
            with lock:
                picture = state['image']
                age = time.monotonic()-state['received_at']
            if age > 10:
                report['warning'] = 'No fresh WGC surface for 10 seconds; recording stopped'
                break
            if picture.shape[:2] != (height, width):
                ratio = min(width/picture.shape[1], height/picture.shape[0])
                resized = cv2.resize(picture, (max(1, round(picture.shape[1]*ratio)),
                                              max(1, round(picture.shape[0]*ratio))))
                picture = np.zeros((height, width, 4), dtype=np.uint8)
                y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
                picture[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
            process.stdin.write(picture.tobytes())
            if frames % args.fps == 0:
                samples.append(hashlib.sha256(picture[::16, ::16].tobytes()).hexdigest())
                with (args.output.parent / 'window-state.jsonl').open('a') as audit:
                    audit.write(json.dumps({'time': time.time(), 'frame': frames,
                        'game_foreground': user32.GetForegroundWindow() == hwnd,
                        'minimized': bool(user32.IsIconic(hwnd)), 'surface_age': round(age, 3)})+'\n')
                if args.verify_snapshots:
                    cv2.imwrite(str(args.output.parent / f'wgc-{frames//args.fps:03d}.png'), picture)
            frames += 1
            time.sleep(max(0, started+frames/args.fps-time.monotonic()))
    finally:
        try:
            control.stop()
            control.wait()
        finally:
            if process is not None:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    report['encoder_exit_code'] = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    report['encoder_exit_code'] = process.wait()
                    report['warning'] = 'Encoder did not finalize within 20 seconds'
            report.update(frames_written=frames, surface_updates=state['updates'],
                          unique_sampled_frames=len(set(samples)), finished_at_unix=time.time())
            args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(report, indent=2))
    if report.get('encoder_exit_code') != 0 or frames == 0 or report.get('warning'):
        raise RuntimeError('Recording did not complete successfully')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
