import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

from .demo import demo, replay
from .models import Models, configuration
from .runtime import Log, serve


def main():
    parser = argparse.ArgumentParser(description="Isaac + Sol + JEV prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Synthetic offline smoke test (no API key needed)")
    d.add_argument("--frames", type=int, default=150)
    r = sub.add_parser("replay", help="Replay agent JSONL bridge frames using local rules")
    r.add_argument("source")
    live = sub.add_parser("live", help="Accept the game mod's TCP connection")
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--port", type=int, default=9527)
    live.add_argument("--mode", choices=("local", "jev", "hybrid"), default="hybrid")
    live.add_argument('--jev-scope', choices=('all', 'tactics', 'off'), default='all',
                      help='Hybrid ablation: all JEV, combat tactics only, or Sol + local only')
    live.add_argument('--isolated-memory', action='store_true',
                      help='Keep map, run history and planner backoff inside this new run directory')
    live.add_argument('--evaluation-batch', type=Path, help='Frozen batch.json to validate before control')
    live.add_argument('--evaluation-case', help='Registered case ID, consumed once even if the run fails')
    live.add_argument('--installed-lua', type=Path, help='Actual installed main.lua; required for evaluation')
    local_config = Path(__file__).resolve().parents[1] / 'config.local.toml'
    live.add_argument("--config", default=str(local_config) if local_config.is_file() else None)
    live.add_argument("--seconds", type=float)
    live.add_argument("--until-run-end", action="store_true", help="Release control on death or leaving the run")
    live.add_argument("--stop-after-floor", type=int, choices=range(1, 13),
                      help="Stop after the selected floor's boss room clears and its exit opens")
    live.add_argument("--stop-file", help="Gracefully stop when this local file appears")
    live.add_argument("--pause-on-stop", action="store_true", help="WSL Windows trial: pause the game before releasing control")
    live.add_argument("--observe-only", action="store_true", help="Read state with manual control and no model calls")
    for command in (d, r, live):
        command.add_argument("--run-dir", help="New, unused output directory")
    stats = sub.add_parser("stats", help="Summarize a run's decisions and errors")
    stats.add_argument("run_dir")
    args = parser.parse_args()
    if args.command == "stats":
        summarize(args.run_dir)
        return
    try:
        models = None
        registration = None
        if args.command == "live":
            if args.seconds is not None and args.seconds <= 0:
                raise ValueError("--seconds must be positive")
            config = configuration(args.config)
            if bool(args.evaluation_batch) != bool(args.evaluation_case):
                raise ValueError('Provide both --evaluation-batch and --evaluation-case')
            if args.evaluation_batch:
                from .evaluation import bind_case
                registration = bind_case(Path(__file__).resolve().parents[1], args.evaluation_batch,
                                         args.evaluation_case, config, args)
            models = Models(config, "local" if args.observe_only else args.mode, args.jev_scope)
        if args.command == "demo" and args.frames <= 0:
            raise ValueError("--frames must be positive")
        directory = args.run_dir or str(Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        log = Log(directory)
        log.isolated_memory = args.command == 'live' and args.isolated_memory
        log.evaluation = registration
        try:
            if registration:
                from .evaluation import claim_case
                claim_case(args.evaluation_batch, registration, directory)
                log.write('evaluation_registered', **registration)
            log.write("start", command=args.command, mode=models.mode if models else "local")
            if args.command == "demo":
                asyncio.run(demo(log, args.frames))
            elif args.command == "replay":
                asyncio.run(replay(args.source, log))
            else:
                log.record_trial(models.config, dict(command='live', mode=models.mode, seconds=args.seconds,
                    jev_scope=models.jev_scope, isolated_memory=log.isolated_memory,
                    evaluation_batch=registration['batch_fingerprint'] if registration else None,
                    evaluation_case=args.evaluation_case,
                    observe_only=args.observe_only, until_run_end=args.until_run_end,
                    stop_after_floor=args.stop_after_floor, pause_on_stop=args.pause_on_stop,
                    port=args.port, stop_file_enabled=bool(args.stop_file)))
                asyncio.run(serve(models, log, args.host, args.port, args.seconds, args.observe_only,
                                  args.until_run_end, args.stop_after_floor, args.pause_on_stop, args.stop_file))
        except KeyboardInterrupt:
            print("Stopped; game input lease expires automatically.")
        finally:
            if registration:
                try:
                    bind_case(Path(__file__).resolve().parents[1], args.evaluation_batch,
                              args.evaluation_case, config, args)
                    log.write('evaluation_end_checked', source_and_lua_verified=True,
                              start_verified=bool(log.counts.get('evaluation_start_verified')))
                except (ValueError, OSError) as exc:
                    log.write('evaluation_end_checked', source_and_lua_verified=False,
                              reason=type(exc).__name__, start_verified=bool(log.counts.get('evaluation_start_verified')))
            log.close()
        print(f"Run saved: {directory}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


def summarize(directory):
    counts, latency, overrides, inputs = {}, {"decision": [], "plan": []}, 0, 0
    with (Path(directory) / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            event = row["event"]
            counts[event] = counts.get(event, 0) + 1
            if event in latency:
                latency[event].append(row["latency_ms"])
            if event == "input":
                inputs += 1
                overrides += int(row["shield_override"])
    report = {"events": counts, "shield_overrides": overrides, "inputs": inputs}
    for lane, values in latency.items():
        if values:
            values.sort()
            report[lane + "_latency_ms"] = {"p50": values[len(values) // 2], "p95": values[min(len(values) - 1, int(len(values) * .95))]}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
