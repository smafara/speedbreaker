#!/usr/bin/env bash
# SpeedBreaker. GPL-3.0-or-later (see COPYING).
#
#   scripts/replay/metal_check.sh [--record] [<capture name>...]
#
# The replays (replay_check.sh) on the Metal backend, checked two ways:
#  1. against Metal's own baselines (set "metal"; --record makes them from
#     this run): every frame bit for bit, so a change that moves what Metal
#     draws shows; the presented picture too (this machine's window);
#  2. against the Vulkan baselines, within float rounding (tolerance 40, 6% of
#     a frame's pixels; Metal's shaders round differently, a world frame
#     differs by up to ~4% of its pixels): the two backends still draw the
#     same game. The presented picture isn't compared (the window's size).
# Run from the repository root, with build/main built with -DSB_METAL=ON.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SB_REPLAY_ENV="NFSMW_BACKEND=metal ${SB_REPLAY_ENV:-}"
if [ "${1:-}" = "--record" ]; then
    shift
    exec scripts/replay/replay_check.sh --record --set metal "$@"
fi
status=0
echo "#### against Metal's baselines"
scripts/replay/replay_check.sh --set metal "$@" || status=1
echo "#### against Vulkan's baselines, within float rounding"
scripts/replay/replay_check.sh --tolerance 40 --max-bad-percent 6 --ignore presented.ppm "$@" || status=1
exit $status
