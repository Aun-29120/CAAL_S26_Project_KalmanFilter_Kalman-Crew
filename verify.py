#!/usr/bin/env python3
"""
verify.py — Numerical verification of MS3 (assembly) vs MS2 (C++) output.

Produces the verification table required by Section 6 of the assignment:
  - Average error per joint
  - Average error per state component
  - Max and min absolute error across all entries
"""

import sys
import csv
import math

NUM_JOINTS = 23
STATE_PER_JOINT = 12
STATE_DIM = NUM_JOINTS * STATE_PER_JOINT  # 276
TOLERANCE = 1e-9

JOINT_NAMES = [
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
]

COMPONENT_NAMES = ["px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"]


def load_states(path):
    """Load CSV output file, return list of list-of-floats (one per frame)."""
    states = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            vals = [float(v) for v in row[1:]]  # skip frame index
            assert len(vals) == STATE_DIM, f"Expected {STATE_DIM} cols, got {len(vals)}"
            states.append(vals)
    return states


def verify(ms2_path, ms3_path):
    print(f"MS2 (C++) file : {ms2_path}")
    print(f"MS3 (ASM) file : {ms3_path}")
    print()

    ms2 = load_states(ms2_path)
    ms3 = load_states(ms3_path)

    assert len(ms2) == len(ms3), f"Frame count mismatch: {len(ms2)} vs {len(ms3)}"
    T = len(ms2)
    print(f"Frames: {T}")
    print(f"State dimension: {STATE_DIM}")
    print()

    # Compute absolute errors
    global_max = 0.0
    global_min = float('inf')
    global_sum = 0.0
    count = 0
    violations = 0

    # Per-joint accumulators
    joint_sum = [0.0] * NUM_JOINTS
    joint_count = [0] * NUM_JOINTS

    # Per-component accumulators (across all joints)
    comp_sum = [0.0] * STATE_PER_JOINT
    comp_count = [0] * STATE_PER_JOINT

    for t in range(T):
        for i in range(STATE_DIM):
            err = abs(ms2[t][i] - ms3[t][i])
            global_max = max(global_max, err)
            global_min = min(global_min, err)
            global_sum += err
            count += 1
            if err > TOLERANCE:
                violations += 1

            j = i // STATE_PER_JOINT
            c = i % STATE_PER_JOINT
            joint_sum[j] += err
            joint_count[j] += 1
            comp_sum[c] += err
            comp_count[c] += 1

    global_avg = global_sum / count

    print("="*65)
    print("GLOBAL STATISTICS")
    print("="*65)
    print(f"  Average absolute error : {global_avg:.3e}")
    print(f"  Maximum absolute error : {global_max:.3e}")
    print(f"  Minimum absolute error : {global_min:.3e}")
    print(f"  Tolerance (epsilon)    : {TOLERANCE:.0e}")
    print(f"  Violations (> eps)     : {violations} / {count}")
    if violations == 0:
        print("  ✓ ALL ENTRIES WITHIN TOLERANCE")
    else:
        print("  ✗ SOME ENTRIES EXCEED TOLERANCE")
    print()

    print("="*65)
    print("AVERAGE ABSOLUTE ERROR PER JOINT")
    print("="*65)
    print(f"  {'Joint':<20s} {'Avg Error':>12s}")
    print(f"  {'-'*20} {'-'*12}")
    for j in range(NUM_JOINTS):
        avg = joint_sum[j] / joint_count[j] if joint_count[j] > 0 else 0
        print(f"  {JOINT_NAMES[j]:<20s} {avg:12.3e}")
    print()

    print("="*65)
    print("AVERAGE ABSOLUTE ERROR PER STATE COMPONENT")
    print("="*65)
    print(f"  {'Component':<12s} {'Avg Error':>12s}")
    print(f"  {'-'*12} {'-'*12}")
    for c in range(STATE_PER_JOINT):
        avg = comp_sum[c] / comp_count[c] if comp_count[c] > 0 else 0
        print(f"  {COMPONENT_NAMES[c]:<12s} {avg:12.3e}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 verify.py <ms2_output.csv> <ms3_output_asm.csv>")
        print("Example: python3 verify.py lkf_output.csv output/lkf_output_asm.csv")
        sys.exit(1)
    verify(sys.argv[1], sys.argv[2])
