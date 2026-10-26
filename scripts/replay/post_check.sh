#!/usr/bin/env bash
# SpeedBreaker. GPL-3.0-or-later (see COPYING).
#
#   scripts/replay/post_check.sh [--record] [<capture name>...]
#
# replay_check.sh under the settings that choose the presenter's
# post-processing paths (anti-aliasing, bilinear / bicubic / FSR 1 scaling,
# the tone table), each with its own set of baselines, so that a change to
# those passes shows in the picture of every path, not only the default's.
# Settings are set by environment so the machine's settings file doesn't matter.
set -euo pipefail
cd "$(dirname "$0")/../.."
base="NFSMW_AA=1 NFSMW_SCALING=bicubic NFSMW_SHARPEN=0.3 NFSMW_BRIGHTNESS=0 NFSMW_DISPLAY_GAMMA=0"
variants=(
    "plain|"
    "aa-off|NFSMW_AA=0"
    "linear|NFSMW_SCALING=linear NFSMW_SHARPEN=0"
    "fsr1|NFSMW_SCALING=fsr1 NFSMW_SHARPEN=0.8"
    "tone|NFSMW_BRIGHTNESS=2 NFSMW_DISPLAY_GAMMA=1"
    "fsr1-tone|NFSMW_SCALING=fsr1 NFSMW_BRIGHTNESS=-2 NFSMW_DISPLAY_GAMMA=1"
)
status=0
for v in "${variants[@]}"; do
    name=${v%%|*}
    extra=${v#*|}
    echo "#### $name"
    SB_REPLAY_ENV="$base $extra" scripts/replay/replay_check.sh --set "post-$name" "$@" || status=1
done
exit $status
