#!/usr/bin/env python3
"""
perf_ms4.py — Performance analysis for Milestone 4.

Measures the speedup of vectorised LKF/EKF (MS4) over scalar LKF/EKF (MS3),
plus a static instruction-count reduction analysis on the math kernels.

Two metrics required by Section 8 of the assignment:

  1. Instruction count reduction
     For each vectorised kernel (mat_mul, mat_add, mat_sub, mat_transpose,
     zero_mem, ldl_solve / lu_solve), we disassemble both the scalar (MS3)
     and vector (MS4) binaries and count the instructions that make up
     each kernel. We report counts and the reduction ratio.

  2. Speedup
     Wall-clock runtime ratio scalar / vector, measured by running each
     binary under qemu-riscv64 with the same input dataset.

Limitations (worth knowing):
  - Static counts measure the SIZE of the kernel code, not the number of
    instructions executed at runtime. The vector kernel runs FEWER iterations
    of its inner loop because each vector instruction processes multiple
    elements at once, so the dynamic reduction is much larger than the static
    one. We approximate the dynamic reduction by accounting for VL.
  - QEMU emulates each vector instruction in software, so the wall-clock
    ratio underestimates the speedup you'd see on real RVV hardware.
    On real hardware, expect substantially better speedups.

Usage:
  python3 perf_ms4.py \\
      --lkf-scalar ./lkf_scalar \\
      --ekf-scalar ./ekf_scalar \\
      --lkf-vector ./lkf_vector \\
      --ekf-vector ./ekf_vector

Optional flags:
  --runs N          Number of timing runs to average (default 3)
  --skip-ekf        Skip EKF timing/disassembly (useful for quick checks)
  --vlen 128        VLEN in bits, used to compute dynamic count estimates
  --output FILE     Write the report to FILE in addition to stdout
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import time


# ─────────────────────────────────────────────────────────────────────────────
# Vector instruction recognition
# ─────────────────────────────────────────────────────────────────────────────
# Every RVV mnemonic starts with 'v' followed by a letter, and is either a
# config op (vsetvli, vsetivli, vsetvl) or contains at least one '.'
# (vle64.v, vse64.v, vfadd.vv, vfmacc.vf, vmv.v.i, vsse64.v, vfsqrt.v, ...).
# No scalar RISC-V mnemonic starts with 'v', so the prefix alone is reliable.

VECTOR_CONFIG = {"vsetvli", "vsetivli", "vsetvl"}


def is_vector_mnemonic(mnemonic):
    """True if a disassembly mnemonic is an RVV instruction.

    Reliable rule: starts with 'v' + alpha char. RISC-V scalar ISA has no
    mnemonics starting with 'v', so this gives zero false positives.
    """
    if len(mnemonic) < 2 or mnemonic[0] != "v":
        return False
    if not mnemonic[1].isalpha():
        return False
    # config ops have no dots; arithmetic/load/store/move ops always have
    # at least one. Either way, the v + alpha prefix is sufficient.
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Disassembly
# ─────────────────────────────────────────────────────────────────────────────

def find_objdump():
    """Locate the riscv64 objdump binary."""
    for cand in ["riscv64-linux-gnu-objdump",
                 "riscv64-unknown-linux-gnu-objdump",
                 "riscv64-unknown-elf-objdump"]:
        try:
            r = subprocess.run([cand, "--version"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return cand
        except FileNotFoundError:
            continue
    sys.exit("ERROR: no riscv64 objdump found in PATH "
             "(install gcc-riscv64-linux-gnu).")


def disassemble(binary, objdump):
    """Return objdump -d output for the given binary."""
    if not os.path.isfile(binary):
        return None
    r = subprocess.run([objdump, "-d", binary],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"objdump failed on {binary}: {r.stderr}")
    return r.stdout


def parse_kernels(disasm, target_names, function_boundaries):
    """Extract instructions for each named kernel from objdump -d output.

    Strategy: walk the file once, collecting (start_addr, name) pairs from
    every `<symbol>:` header. Then for each target kernel, gather all
    instructions from its start address to the start of the next symbol
    listed in function_boundaries (or EOF).

    function_boundaries is the set of all real function entry points, both
    target kernels and any other helper functions (my_sqrt, h_joint, main,
    etc.). This is needed because in non-stripped object files local labels
    inside a kernel show up as their own `<symbol>:` blocks but are NOT
    function boundaries — they're inside the parent kernel.

    Returns dict {kernel_name: [(addr, mnemonic, raw_line), ...]}.
    """
    sym_re = re.compile(r"^([0-9a-f]+)\s+<([^>]+)>:")
    insn_re = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+([a-z][a-z0-9_.]*)")

    symbols = []
    instructions = []
    for line in disasm.splitlines():
        msym = sym_re.match(line)
        if msym:
            symbols.append((int(msym.group(1), 16), msym.group(2)))
            continue
        minsn = insn_re.match(line)
        if minsn:
            instructions.append((int(minsn.group(1), 16),
                                 minsn.group(3), line.rstrip()))

    symbols.sort()

    # For each target kernel, find its range [start, end). end is the address
    # of the next symbol that is a real function entry point (not a local
    # label inside this kernel).
    ranges = {}
    for i, (addr, name) in enumerate(symbols):
        if name in target_names:
            end = None
            for j in range(i + 1, len(symbols)):
                if (symbols[j][1] in function_boundaries
                        and symbols[j][1] != name):
                    end = symbols[j][0]
                    break
            ranges[name] = (addr, end)

    out = {n: [] for n in target_names}
    for (addr, mnemonic, raw) in instructions:
        for name, (start, end) in ranges.items():
            if addr >= start and (end is None or addr < end):
                out[name].append((addr, mnemonic, raw))
                break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel counts
# ─────────────────────────────────────────────────────────────────────────────

# Symbols we care about for instruction-count comparison. Names match the
# globally-exported labels in lkf_asm.s / ekf_asm.s and lkf_vector.s / ekf_vector.s.
KERNELS_LKF = ["zero_mem", "mat_add", "mat_sub", "mat_transpose",
               "mat_mul", "ldl_solve"]
KERNELS_EKF = ["zero_mem", "mat_add", "mat_sub", "mat_transpose",
               "mat_mul", "lu_solve"]

# Real function entry points across LKF and EKF (used as range terminators
# when bucketing instructions into kernels). Local labels INSIDE these
# functions are NOT in this list, so they're correctly attributed to their
# parent function.
FUNCTION_BOUNDARIES_LKF = set(KERNELS_LKF) | {"main"}
FUNCTION_BOUNDARIES_EKF = set(KERNELS_EKF) | {
    "my_sqrt", "atan_core", "my_atan2", "h_joint", "jac_joint",
    "do_fallback", "main"
}


def kernel_stats(kernels, names):
    """For each kernel name, return (total_instr, vector_instr, scalar_instr)."""
    out = {}
    for n in names:
        body = kernels.get(n, [])
        total = len(body)
        v = sum(1 for (_, m, _) in body if is_vector_mnemonic(m))
        s = total - v
        out[n] = (total, v, s)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Wall-clock timing
# ─────────────────────────────────────────────────────────────────────────────

def run_once(qemu, qemu_cpu, binary):
    """Run one binary under QEMU, return wall-clock seconds.

    QEMU is invoked with -cpu rv64,v=true,vlen=128 for vector binaries; for
    scalar it doesn't matter (the binary won't issue vector instructions),
    but we still pass the same -cpu so toolchain feature differences don't
    affect timing."""
    if not os.path.isfile(binary):
        return None
    cmd = [qemu, "-cpu", qemu_cpu, binary]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    if r.returncode != 0:
        sys.stderr.write(f"WARNING: {binary} returned non-zero exit "
                         f"({r.returncode}); stderr:\n{r.stderr}\n")
        return None
    return t1 - t0


def time_runs(qemu, qemu_cpu, binary, n_runs):
    """Return (mean, stdev) of n_runs wall-clock measurements, or (None, None)
    if the binary doesn't exist."""
    times = []
    for i in range(n_runs):
        t = run_once(qemu, qemu_cpu, binary)
        if t is None:
            return None, None
        times.append(t)
        print(f"    run {i+1}/{n_runs}: {t:.2f} s")
    mean = statistics.fmean(times)
    sd   = statistics.pstdev(times) if len(times) > 1 else 0.0
    return mean, sd


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def hr(width=78):
    return "=" * width


def section(title, width=78):
    return f"\n{hr(width)}\n{title}\n{hr(width)}"


def fmt_int_count(s_total, s_vec, s_scalar):
    return f"{s_total:>6d} ({s_vec} vec / {s_scalar} scalar)"


def report_instruction_counts(scalar_kernels, vector_kernels, names, label,
                              vlen_doubles, lines):
    """Append per-kernel static instruction-count comparison + dynamic estimate."""
    lmul_factor = vlen_doubles * 4   # LMUL=m4 means VL = 4 * VLEN/64 doubles
    lines.append(section(f"INSTRUCTION COUNT — {label}"))
    lines.append(f"  VLEN = {vlen_doubles*64} bits → {vlen_doubles} doubles per "
                 f"vector register at e64 (m1)")
    lines.append(f"  At LMUL=m4 each vector op covers {lmul_factor} doubles.")
    lines.append("")
    s_kstats = kernel_stats(scalar_kernels, names)
    v_kstats = kernel_stats(vector_kernels, names)

    lines.append(f"  {'Kernel':<16s} {'MS3 scalar':<25s} "
                 f"{'MS4 vector':<25s} {'Static reduction':>18s}")
    lines.append(f"  {'-'*16} {'-'*25} {'-'*25} {'-'*18}")

    total_s = total_v = 0
    for n in names:
        s_t, s_v, s_s = s_kstats[n]
        v_t, v_v, v_s = v_kstats[n]
        total_s += s_t
        total_v += v_t
        if s_t == 0:
            lines.append(f"  {n:<16s}  (not found in scalar binary)")
            continue
        if v_t == 0:
            lines.append(f"  {n:<16s}  (not found in vector binary)")
            continue
        ratio = s_t / v_t
        lines.append(
            f"  {n:<16s} "
            f"{fmt_int_count(s_t, s_v, s_s):<25s} "
            f"{fmt_int_count(v_t, v_v, v_s):<25s} "
            f"{ratio:>14.2f}x   ")
    if total_s and total_v:
        lines.append(f"  {'-'*16} {'-'*25} {'-'*25} {'-'*18}")
        lines.append(f"  {'TOTAL':<16s} {total_s:>23d}   {total_v:>23d}   "
                     f"{total_s/total_v:>14.2f}x   ")

    # Dynamic reduction model:
    # Each elementwise kernel processes N elements. In scalar form, the inner
    # loop body of size B_s runs N times, giving N * B_s dynamic instructions.
    # In vector form, each strip-mined iteration handles VL=lmul_factor=8
    # elements, so the body of size B_v runs ceil(N/VL) times, giving
    # ceil(N/VL) * B_v dynamic instructions.
    #
    # The reduction ratio for large N (>> VL) is:
    #     (N * B_s) / (ceil(N/VL) * B_v)  ~= (B_s * VL) / B_v
    #
    # We separate the kernel body from any preamble/cleanup (saving registers,
    # zeroing C, returning) by just dividing both bodies' instruction counts
    # by the same factor — this is approximate but captures the right order
    # of magnitude. The exact value depends on N and per-call structure.
    lines.append("")
    lines.append("  Dynamic reduction estimate (per element, large-N limit):")
    lines.append(f"  formula:  reduction ~= (scalar_body * VL) / vector_body, "
                 f"with VL = {lmul_factor}")
    lines.append(f"  {'Kernel':<16s} {'Scalar body':>13s} {'Vector body':>13s} "
                 f"{'Est. reduction':>17s}")
    lines.append(f"  {'-'*16} {'-'*13} {'-'*13} {'-'*17}")
    for n in names:
        s_t, s_v, s_s = s_kstats[n]
        v_t, v_v, v_s = v_kstats[n]
        if s_t == 0 or v_t == 0:
            continue
        if v_t == 0:
            continue
        # Approximate: scalar body contains all instructions; vector body
        # contains all instructions, but vector ops do VL elements at once.
        est = (s_t * lmul_factor) / v_t
        lines.append(f"  {n:<16s} {s_t:>13d} {v_t:>13d} {est:>15.2f}x")
    lines.append("")
    lines.append("  This estimate is an upper bound for elementwise kernels")
    lines.append("  (mat_add, mat_sub, zero_mem). For mat_mul and ldl_solve")
    lines.append("  the realised speedup is lower because of remaining scalar")
    lines.append("  decision logic (zero-skip in mat_mul, pivoting + factorise")
    lines.append("  loops in ldl_solve / lu_solve).")


def report_runtimes(timings, lines):
    """Append walltime comparison table."""
    lines.append(section("RUNTIME — wall-clock under QEMU"))
    lines.append(f"  {'Filter':<10s} {'MS3 scalar (s)':>20s} "
                 f"{'MS4 vector (s)':>20s} {'Speedup':>12s}")
    lines.append(f"  {'-'*10} {'-'*20} {'-'*20} {'-'*12}")

    overall_s = 0.0
    overall_v = 0.0
    for filt in ("LKF", "EKF"):
        sm, ss = timings.get(f"{filt}_scalar", (None, None))
        vm, vs = timings.get(f"{filt}_vector", (None, None))
        if sm is None and vm is None:
            continue
        if sm is None:
            lines.append(f"  {filt:<10s} {'(skipped)':>20s} "
                         f"{vm:>20.3f} {'-':>12s}")
            continue
        if vm is None:
            lines.append(f"  {filt:<10s} {sm:>20.3f} "
                         f"{'(skipped)':>20s} {'-':>12s}")
            continue
        speedup = sm / vm
        overall_s += sm
        overall_v += vm
        lines.append(f"  {filt:<10s} "
                     f"{sm:>14.3f} ± {ss:>5.3f}   "
                     f"{vm:>14.3f} ± {vs:>5.3f}   "
                     f"{speedup:>10.2f}x")

    if overall_s > 0 and overall_v > 0:
        lines.append(f"  {'-'*10} {'-'*20} {'-'*20} {'-'*12}")
        lines.append(f"  {'TOTAL':<10s} {overall_s:>20.3f} "
                     f"{overall_v:>20.3f} {overall_s/overall_v:>10.2f}x")
    lines.append("")
    lines.append("  Note: QEMU emulates RVV in software; real hardware will be")
    lines.append("  faster. The instruction-count reduction is the more reliable")
    lines.append("  indicator of vectorisation effectiveness on real RVV CPUs.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="MS4 performance analysis (runtime + instruction count).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--lkf-scalar", required=True,
                   help="Path to MS3 scalar LKF binary")
    p.add_argument("--ekf-scalar", required=True,
                   help="Path to MS3 scalar EKF binary")
    p.add_argument("--lkf-vector", required=True,
                   help="Path to MS4 vector LKF binary")
    p.add_argument("--ekf-vector", required=True,
                   help="Path to MS4 vector EKF binary")
    p.add_argument("--qemu", default="qemu-riscv64",
                   help="QEMU binary (default qemu-riscv64)")
    p.add_argument("--qemu-cpu", default="rv64,v=true,vlen=128",
                   help="QEMU -cpu argument (default 'rv64,v=true,vlen=128')")
    p.add_argument("--runs", type=int, default=3,
                   help="Number of timing runs per binary (default 3)")
    p.add_argument("--skip-ekf", action="store_true",
                   help="Skip EKF timing and disassembly")
    p.add_argument("--vlen", type=int, default=128,
                   help="VLEN in bits (default 128)")
    p.add_argument("--output", default=None,
                   help="Also write report to this file")
    args = p.parse_args()

    vlen_doubles = args.vlen // 64    # doubles per vreg at e64,m1
    if vlen_doubles < 1:
        sys.exit(f"ERROR: vlen={args.vlen} too small for e64 doubles.")

    objdump = find_objdump()
    lines = []
    lines.append(hr())
    lines.append(" Kalman Filter — Milestone 4 Performance Analysis")
    lines.append(f" Scalar binaries: {args.lkf_scalar}, {args.ekf_scalar}")
    lines.append(f" Vector binaries: {args.lkf_vector}, {args.ekf_vector}")
    lines.append(f" QEMU: {args.qemu}  -cpu {args.qemu_cpu}")
    lines.append(f" Runs per binary: {args.runs}")
    lines.append(hr())

    # ---- Disassembly + instruction counts ----
    print("[1/2] Disassembling binaries and counting kernel instructions...")
    s_lkf_dis = disassemble(args.lkf_scalar, objdump)
    v_lkf_dis = disassemble(args.lkf_vector, objdump)
    if s_lkf_dis is None:
        lines.append(f"\n  WARNING: scalar LKF binary not found ({args.lkf_scalar})")
    if v_lkf_dis is None:
        lines.append(f"\n  WARNING: vector LKF binary not found ({args.lkf_vector})")
    if s_lkf_dis and v_lkf_dis:
        s_k = parse_kernels(s_lkf_dis, KERNELS_LKF, FUNCTION_BOUNDARIES_LKF)
        v_k = parse_kernels(v_lkf_dis, KERNELS_LKF, FUNCTION_BOUNDARIES_LKF)
        report_instruction_counts(s_k, v_k, KERNELS_LKF, "LKF",
                                  vlen_doubles, lines)

    if not args.skip_ekf:
        s_ekf_dis = disassemble(args.ekf_scalar, objdump)
        v_ekf_dis = disassemble(args.ekf_vector, objdump)
        if s_ekf_dis is None:
            lines.append(f"\n  WARNING: scalar EKF binary not found "
                         f"({args.ekf_scalar})")
        if v_ekf_dis is None:
            lines.append(f"\n  WARNING: vector EKF binary not found "
                         f"({args.ekf_vector})")
        if s_ekf_dis and v_ekf_dis:
            s_k = parse_kernels(s_ekf_dis, KERNELS_EKF, FUNCTION_BOUNDARIES_EKF)
            v_k = parse_kernels(v_ekf_dis, KERNELS_EKF, FUNCTION_BOUNDARIES_EKF)
            report_instruction_counts(s_k, v_k, KERNELS_EKF, "EKF",
                                      vlen_doubles, lines)

    # ---- Walltime ----
    print(f"[2/2] Timing each binary {args.runs} times under QEMU...")
    timings = {}
    for label, path in (
        ("LKF_scalar", args.lkf_scalar),
        ("LKF_vector", args.lkf_vector),
        ("EKF_scalar", args.ekf_scalar),
        ("EKF_vector", args.ekf_vector),
    ):
        if args.skip_ekf and label.startswith("EKF"):
            continue
        if not os.path.isfile(path):
            print(f"  Skipping {label}: {path} not found")
            timings[label] = (None, None)
            continue
        print(f"  Timing {label} ({path}):")
        m, s = time_runs(args.qemu, args.qemu_cpu, path, args.runs)
        timings[label] = (m, s)
    report_runtimes(timings, lines)

    # ---- Final emission ----
    report = "\n".join(lines) + "\n"
    print()
    print(report)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
