#!/usr/bin/env python3
# SpeedBreaker. GPL-3.0-or-later (see COPYING).
#
#   compare_frames.py <baseline dir> <new dir> [--tolerance N] [--max-bad-percent P] [--ignore NAME]...
#
# Compares the PPM frames (NFSMW_DUMP_FRAMES, see replay_check.sh) of two
# runs, by file name. A pixel is bad when any channel differs by more than
# --tolerance (default 0: bit-identical); a frame fails when more than
# --max-bad-percent of its pixels are bad (default 0). Prints one line per
# frame and exits 1 if any frame fails or the sets of frames differ. --ignore
# leaves a file out of both sets (presented.ppm, whose size is the window's).
import argparse
import os
import sys


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, size, depth, pixels = data.split(b"\n", 3)
    if magic != b"P6" or depth != b"255":
        raise ValueError(f"{path}: not a P6 255 PPM")
    width, height = map(int, size.split())
    if len(pixels) != width * height * 3:
        raise ValueError(f"{path}: truncated")
    return width, height, pixels


def compare(a, b, tolerance, width):
    """(bad pixels, largest channel difference, mean absolute difference)."""
    if a == b:
        return 0, 0, 0.0
    bad = largest = total = 0
    stride = width * 3
    for row in range(0, len(a), stride):
        ra, rb = a[row:row + stride], b[row:row + stride]
        if ra == rb:
            continue
        for i in range(0, stride, 3):
            if ra[i:i + 3] == rb[i:i + 3]:
                continue
            d = (abs(ra[i] - rb[i]), abs(ra[i + 1] - rb[i + 1]), abs(ra[i + 2] - rb[i + 2]))
            total += sum(d)
            largest = max(largest, *d)
            if max(d) > tolerance:
                bad += 1
    return bad, largest, total / len(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("new")
    parser.add_argument("--tolerance", type=int, default=0)
    parser.add_argument("--max-bad-percent", type=float, default=0.0)
    parser.add_argument("--ignore", action="append", default=[])
    args = parser.parse_args()

    names = lambda d: sorted(n for n in os.listdir(d) if n.endswith(".ppm") and n not in args.ignore)
    base, new = names(args.baseline), names(args.new)
    failed = False
    if base != new:
        print(f"FAIL: frames differ: baseline {len(base)}, new {len(new)}")
        failed = True
    for name in sorted(set(base) & set(new)):
        wa, ha, a = read_ppm(os.path.join(args.baseline, name))
        wb, hb, b = read_ppm(os.path.join(args.new, name))
        if (wa, ha) != (wb, hb):
            print(f"FAIL {name}: size {wa}x{ha} vs {wb}x{hb}")
            failed = True
            continue
        bad, largest, mean = compare(a, b, args.tolerance, wa)
        percent = 100.0 * bad / (wa * ha)
        ok = percent <= args.max_bad_percent
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {bad} bad pixels ({percent:.3f}%), largest difference {largest}, mean {mean:.4f}")
    print("FAILED" if failed else "PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
