#!/usr/bin/env bash
# SpeedBreaker. GPL-3.0-or-later (see COPYING).
#
#   scripts/replay/replay_check.sh [--record] [--tolerance N] [--max-bad-percent P] [<capture name>...]
#
# Replays GPU captures (NFSMW_CAPTURE, gpu/trace.h; DECISIONS.md D7) and
# compares every replayed frame with the recorded baseline. Captures are
# build/captures/<name>.sbtrace, baselines build/baselines/frames/<name>/
# (both hold game memory or game images: never commit them). With no name,
# all captures there. --record replaces the baseline with this run's frames.
# Run from the repository root, with build/main built. Exit status 0 if all
# passed.
#   SB_REPLAY_ENV="NAME=value ..."  extra environment for the replay (a backend)
set -euo pipefail
cd "$(dirname "$0")/../.."

record=0
args=()
names=()
while [ $# -gt 0 ]; do
    case "$1" in
        --record) record=1 ;;
        --tolerance|--max-bad-percent) args+=("$1" "$2"); shift ;;
        *) names+=("$1") ;;
    esac
    shift
done
if [ ${#names[@]} -eq 0 ]; then
    for f in build/captures/*.sbtrace; do
        [ -e "$f" ] && names+=("$(basename "$f" .sbtrace)")
    done
fi
[ ${#names[@]} -gt 0 ] || { echo "no captures in build/captures" >&2; exit 2; }

binary=build/main/runtime/SpeedBreaker
work=$(mktemp -d "${TMPDIR:-/tmp}/replay_check.XXXXXX")
trap 'rm -rf "$work"' EXIT
status=0
for name in "${names[@]}"; do
    capture="build/captures/$name.sbtrace"
    [ -f "$capture" ] || { echo "$name: no $capture" >&2; status=1; continue; }
    out="$work/$name"
    mkdir -p "$out"
    # No timeout(1) on macOS: perl's alarm.
    env ${SB_REPLAY_ENV:-} NFSMW_REPLAY="$capture" NFSMW_REPLAY_EXIT=1 NFSMW_DUMP_FRAMES="$out" NFSMW_DUMP_EVERY=1 \
        perl -e 'alarm 300; exec @ARGV' "$binary" >"$work/$name.log" 2>&1 || {
        echo "$name: the replay failed (log: $work/$name.log)"; trap - EXIT; status=1; continue; }
    base="build/baselines/frames/$name"
    if [ $record -eq 1 ]; then
        rm -rf "$base"
        mkdir -p "$base"
        cp "$out"/*.ppm "$base/"
        echo "$name: recorded $(ls "$base" | wc -l | tr -d ' ') frames"
    elif [ ! -d "$base" ]; then
        echo "$name: no baseline (run with --record)"; status=1
    else
        echo "== $name"
        python3 scripts/replay/compare_frames.py "$base" "$out" ${args[@]+"${args[@]}"} || status=1
    fi
done
exit $status
