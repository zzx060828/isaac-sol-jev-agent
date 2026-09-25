#!/usr/bin/env bash
set -euo pipefail
agent_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ss -ltnH 'sport = :9527' | grep -q .; then
    echo 'A bridge is already listening on port 9527; keeping it.'
    exit 0
fi
agent_run="runs/live-$(date +%Y%m%d-%H%M%S)"
if tmux has-session -t isaac-agent 2>/dev/null; then
    tmux new-window -d -t isaac-agent -n controller -c "$agent_root" "python3 -u -m isaac_agent live --mode hybrid --until-run-end --run-dir $agent_run"
else
    tmux new-session -d -s isaac-agent -c "$agent_root" "python3 -u -m isaac_agent live --mode hybrid --until-run-end --run-dir $agent_run"
fi
echo "Agent started; logs: $agent_root/$agent_run"
