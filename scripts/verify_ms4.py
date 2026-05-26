#!/usr/bin/env python3
"""
verify_ms4.py — Numerical verification for Milestone 4.

Compares MS4 vector output against MS3 scalar reference for both LKF and EKF.
Produces every table required by Section 7 of the MS4 assignment:

  1. Average absolute error per joint, both filters.
  2. Average absolute error per state component, both filters.
  3. Maximum / minimum absolute error across the whole output, both filters.
  4. Direct comparison: MS3 scalar error vs MS4 vector error for the same
     joint and state component (proving vectorisation didn't degrade accuracy).

Tolerance: 1e-9 (per assignment).

Usage:
  python3 verify_ms4.py \\
      --lkf-ref  LKF_asm_output.csv \\
      --lkf-vec  LKF_vector_output.csv \\
      --ekf-ref  EKF_asm_output.csv \\
      --ekf-vec  EKF_vector_output.csv

Optional: --ms2-lkf and --ms2-ekf to add a third column to Table 4 showing
MS2 C++ vs MS4 vector. If omitted, Table 4 just shows MS3 scalar baseline
errors (which were ~0 for LKF and ~10^-14 for EKF in MS3).
"""

import argparse
import csv
import math
import os
import sys

# ---- Constants matching the assignment ----
NUM_JOINTS      = 23
STATE_PER_JOINT = 12
STATE_DIM       = NUM_JOINTS * STATE_PER_JOINT  # 276
TOLERANCE       = 1e-9

JOINT_NAMES = [
    "pelvis", "L5", "L3", "T12", "T8", "neck", "head",
    "shoulderRight", "upperArmRight", "forearmRight", "handRight",
    "shoulderLeft", "upperArmLeft", "forearmLeft", "handLeft",
    "upperLegRight", "lowerLegRight", "footRight", "toeRight",
    "upperLegLeft", "lowerLegLeft", "footLeft", "toeLeft",
]

COMPONENT_NAMES = ["px", "vx", "ax", "jx",
                   "py", "vy", "ay", "jy",
                   "pz", "vz", "az", "jz"]


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_states(path):
    """Load a state CSV (one row per frame, frame_idx + 276 doubles).
    Returns list of lists; each inner list has STATE_DIM floats.
    Raises FileNotFoundError or ValueError on any structural problem."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    states = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        for row_idx, row in enumerate(reader, start=1):
            try:
                vals = [float(v) for v in row[1:]]   # skip frame index
            except ValueError as e:
                raise ValueError(
                    f"{path} row {row_idx}: cannot parse floats ({e})")
            if len(vals) != STATE_DIM:
                raise ValueError(
                    f"{path} row {row_idx}: expected {STATE_DIM} cols, "
                    f"got {len(vals)}")
            states.append(vals)
    return states


# ─────────────────────────────────────────────────────────────────────────────
# Per-filter error analysis
# ─────────────────────────────────────────────────────────────────────────────

class FilterStats:
    """Accumulates per-joint, per-component, per-cell, and global error statistics
    for one filter (LKF or EKF) across all frames and state components."""

    def __init__(self, name):
        self.name = name
        self.global_max  = 0.0
        self.global_min  = float("inf")
        self.global_sum  = 0.0
        self.count       = 0
        self.violations  = 0
        self.joint_sum   = [0.0] * NUM_JOINTS
        self.joint_count = [0]   * NUM_JOINTS
        self.comp_sum    = [0.0] * STATE_PER_JOINT
        self.comp_count  = [0]   * STATE_PER_JOINT
        # Per-cell (joint, component) accumulators for Table 4
        self.cell_sum    = [0.0] * STATE_DIM   # flat: j * 12 + c
        self.cell_count  = [0]   * STATE_DIM

    @property
    def global_avg(self):
        return self.global_sum / self.count if self.count else 0.0

    def passed(self):
        return self.violations == 0

    def cell_avg(self, j, c):
        idx = j * STATE_PER_JOINT + c
        return (self.cell_sum[idx] / self.cell_count[idx]
                if self.cell_count[idx] else 0.0)


def compare(ref_states, vec_states, name):
    """Compute FilterStats from two aligned state lists. Asserts equal length."""
    if len(ref_states) != len(vec_states):
        raise ValueError(
            f"{name}: frame count mismatch "
            f"(ref={len(ref_states)}, vec={len(vec_states)})")

    stats = FilterStats(name)
    T = len(ref_states)

    for t in range(T):
        ref_row = ref_states[t]
        vec_row = vec_states[t]
        for i in range(STATE_DIM):
            err = abs(ref_row[i] - vec_row[i])
            if err > stats.global_max:
                stats.global_max = err
            if err < stats.global_min:
                stats.global_min = err
            stats.global_sum += err
            stats.count += 1
            if err > TOLERANCE:
                stats.violations += 1
            j = i // STATE_PER_JOINT
            c = i %  STATE_PER_JOINT
            stats.joint_sum[j] += err
            stats.joint_count[j] += 1
            stats.comp_sum[c] += err
            stats.comp_count[c] += 1
            stats.cell_sum[i] += err
            stats.cell_count[i] += 1

    return stats, T


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printers
# ─────────────────────────────────────────────────────────────────────────────

def hr(width=78):
    print("=" * width)


def section(title, width=78):
    print()
    hr(width)
    print(title)
    hr(width)


def print_global_table(lkf, ekf, T):
    """Table 3: max/min global statistics for both filters side by side."""
    section("GLOBAL STATISTICS  (MS4 vector vs MS3 scalar reference)")
    print(f"  Frames per filter   : {T}")
    print(f"  State dimension     : {STATE_DIM}")
    print(f"  Entries per filter  : {T * STATE_DIM}")
    print(f"  Tolerance (epsilon) : {TOLERANCE:.0e}")
    print()
    print(f"  {'Metric':<26s} {'LKF':>15s} {'EKF':>15s}")
    print(f"  {'-'*26} {'-'*15} {'-'*15}")
    print(f"  {'Average abs error':<26s} "
          f"{lkf.global_avg:>15.3e} {ekf.global_avg:>15.3e}")
    print(f"  {'Maximum abs error':<26s} "
          f"{lkf.global_max:>15.3e} {ekf.global_max:>15.3e}")
    print(f"  {'Minimum abs error':<26s} "
          f"{lkf.global_min:>15.3e} {ekf.global_min:>15.3e}")
    print(f"  {'Violations (> eps)':<26s} "
          f"{lkf.violations:>15d} {ekf.violations:>15d}")
    print()
    print(f"  LKF result: {'PASS — within tolerance' if lkf.passed() else 'FAIL'}")
    print(f"  EKF result: {'PASS — within tolerance' if ekf.passed() else 'FAIL'}")


def print_per_joint_table(lkf, ekf):
    """Table 1: per-joint average error, both filters."""
    section("AVERAGE ABSOLUTE ERROR PER JOINT")
    print(f"  {'Joint':<20s} {'LKF':>14s} {'EKF':>14s}")
    print(f"  {'-'*20} {'-'*14} {'-'*14}")
    for j in range(NUM_JOINTS):
        lkf_avg = lkf.joint_sum[j] / lkf.joint_count[j] if lkf.joint_count[j] else 0.0
        ekf_avg = ekf.joint_sum[j] / ekf.joint_count[j] if ekf.joint_count[j] else 0.0
        print(f"  {JOINT_NAMES[j]:<20s} {lkf_avg:>14.3e} {ekf_avg:>14.3e}")


def print_per_component_table(lkf, ekf):
    """Table 2: per-component average error, both filters."""
    section("AVERAGE ABSOLUTE ERROR PER STATE COMPONENT")
    print(f"  {'Component':<12s} {'LKF':>14s} {'EKF':>14s}")
    print(f"  {'-'*12} {'-'*14} {'-'*14}")
    for c in range(STATE_PER_JOINT):
        lkf_avg = lkf.comp_sum[c] / lkf.comp_count[c] if lkf.comp_count[c] else 0.0
        ekf_avg = ekf.comp_sum[c] / ekf.comp_count[c] if ekf.comp_count[c] else 0.0
        print(f"  {COMPONENT_NAMES[c]:<12s} {lkf_avg:>14.3e} {ekf_avg:>14.3e}")


def print_direct_comparison(lkf_stats, ekf_stats,
                            lkf_ms3_path, ekf_ms3_path,
                            ms2_lkf_path, ms2_ekf_path):
    """Table 4: direct comparison MS3 scalar error vs MS4 vector error.

    The "MS3 error" column shows what the MS3 scalar implementation reported
    against its OWN reference (typically the MS2 C++ output, when paths are
    provided). The "MS4 error" column shows MS4 vector vs MS3 scalar.

    If the MS2 C++ paths aren't provided, we just show the MS4 error
    column (which is the primary deliverable anyway)."""
    section("DIRECT COMPARISON: MS3 SCALAR ERROR  vs  MS4 VECTOR ERROR")

    if ms2_lkf_path and os.path.exists(ms2_lkf_path):
        print(f"  MS3 scalar errors computed from: {lkf_ms3_path} vs {ms2_lkf_path}")
        print(f"                                   {ekf_ms3_path} vs {ms2_ekf_path}")
        try:
            ms2_lkf = load_states(ms2_lkf_path)
            ms3_lkf_states = load_states(lkf_ms3_path)
            ms2_ekf = load_states(ms2_ekf_path)
            ms3_ekf_states = load_states(ekf_ms3_path)
            ms3_lkf_stats, _ = compare(ms2_lkf, ms3_lkf_states, "MS3-LKF-vs-MS2")
            ms3_ekf_stats, _ = compare(ms2_ekf, ms3_ekf_states, "MS3-EKF-vs-MS2")
            have_ms3_baseline = True
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: could not load MS2 references ({e})")
            print(f"  Showing MS4 vector errors only.")
            have_ms3_baseline = False
    else:
        print("  (No MS2 C++ paths supplied — showing MS4 vector errors only.")
        print("   Pass --ms2-lkf and --ms2-ekf to add the MS3-scalar-vs-MS2")
        print("   baseline column for full assignment Table 4.)")
        have_ms3_baseline = False

    print()
    if have_ms3_baseline:
        print(f"  {'Filter':<8s} {'Joint':<18s} {'Comp':<6s} "
              f"{'MS3 err':>13s} {'MS4 err':>13s} {'verdict':>10s}")
        print(f"  {'-'*8} {'-'*18} {'-'*6} {'-'*13} {'-'*13} {'-'*10}")
    else:
        print(f"  {'Filter':<8s} {'Joint':<18s} {'Comp':<6s} {'MS4 err':>13s} {'verdict':>10s}")
        print(f"  {'-'*8} {'-'*18} {'-'*6} {'-'*13} {'-'*10}")

    # Show one sample row per joint group: pelvis (px), upperArmRight (vx),
    # footRight (ax) [LKF fallback in EKF], plus all-component summary rows.
    sample_pairs = [
        (0,  0),   # pelvis,        px
        (0,  1),   # pelvis,        vx
        (0,  4),   # pelvis,        py
        (0,  8),   # pelvis,        pz
        (8,  0),   # upperArmRight, px
        (17, 0),   # footRight,     px (EKF LKF-fallback joint)
        (22, 0),   # toeLeft,       px (EKF LKF-fallback joint)
    ]

    def cell(stats, j, c):
        return stats.cell_avg(j, c)

    for (j, c) in sample_pairs:
        for fname, fstats, ms3stats in (
            ("LKF", lkf_stats, (ms3_lkf_stats if have_ms3_baseline else None)),
            ("EKF", ekf_stats, (ms3_ekf_stats if have_ms3_baseline else None)),
        ):
            jname = JOINT_NAMES[j]
            cname = COMPONENT_NAMES[c]
            ms4_err = cell(fstats, j, c)
            verdict = "PASS" if ms4_err <= TOLERANCE else "FAIL"
            if have_ms3_baseline and ms3stats is not None:
                ms3_err = cell(ms3stats, j, c)
                print(f"  {fname:<8s} {jname:<18s} {cname:<6s} "
                      f"{ms3_err:>13.3e} {ms4_err:>13.3e} {verdict:>10s}")
            else:
                print(f"  {fname:<8s} {jname:<18s} {cname:<6s} "
                      f"{ms4_err:>13.3e} {verdict:>10s}")


def print_summary(lkf, ekf):
    """One-line bottom-of-report summary."""
    section("MS4 VERIFICATION SUMMARY")
    overall_pass = lkf.passed() and ekf.passed()
    print(f"  LKF vector: avg={lkf.global_avg:.3e}, max={lkf.global_max:.3e}, "
          f"violations={lkf.violations}")
    print(f"  EKF vector: avg={ekf.global_avg:.3e}, max={ekf.global_max:.3e}, "
          f"violations={ekf.violations}")
    print()
    if overall_pass:
        print("  >> OVERALL: PASS — vectorisation preserved numerical correctness.")
    else:
        print("  >> OVERALL: FAIL — at least one entry exceeds tolerance.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Verify MS4 vector outputs vs MS3 scalar reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--lkf-ref", required=True,
                   help="MS3 scalar LKF output CSV (reference)")
    p.add_argument("--lkf-vec", required=True,
                   help="MS4 vector LKF output CSV")
    p.add_argument("--ekf-ref", required=True,
                   help="MS3 scalar EKF output CSV (reference)")
    p.add_argument("--ekf-vec", required=True,
                   help="MS4 vector EKF output CSV")
    p.add_argument("--ms2-lkf", default=None,
                   help="(optional) MS2 C++ LKF output for Table 4 baseline")
    p.add_argument("--ms2-ekf", default=None,
                   help="(optional) MS2 C++ EKF output for Table 4 baseline")
    args = p.parse_args()

    print()
    hr()
    print(" Kalman Filter — Milestone 4 Numerical Verification")
    print(f" MS3 scalar reference:  {args.lkf_ref}")
    print(f"                        {args.ekf_ref}")
    print(f" MS4 vector outputs:    {args.lkf_vec}")
    print(f"                        {args.ekf_vec}")
    hr()

    # Load and compare LKF
    try:
        lkf_ref = load_states(args.lkf_ref)
        lkf_vec = load_states(args.lkf_vec)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR loading LKF data: {e}", file=sys.stderr)
        sys.exit(1)
    lkf_stats, T_lkf = compare(lkf_ref, lkf_vec, "LKF")

    # Load and compare EKF
    try:
        ekf_ref = load_states(args.ekf_ref)
        ekf_vec = load_states(args.ekf_vec)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR loading EKF data: {e}", file=sys.stderr)
        sys.exit(1)
    ekf_stats, T_ekf = compare(ekf_ref, ekf_vec, "EKF")

    if T_lkf != T_ekf:
        print(f"\nWARNING: LKF and EKF have different frame counts "
              f"(LKF={T_lkf}, EKF={T_ekf})")

    # Print all tables
    print_global_table(lkf_stats, ekf_stats, T_lkf)
    print_per_joint_table(lkf_stats, ekf_stats)
    print_per_component_table(lkf_stats, ekf_stats)
    print_direct_comparison(lkf_stats, ekf_stats,
                            args.lkf_ref, args.ekf_ref,
                            args.ms2_lkf, args.ms2_ekf)
    print_summary(lkf_stats, ekf_stats)

    sys.exit(0 if (lkf_stats.passed() and ekf_stats.passed()) else 1)


if __name__ == "__main__":
    main()
