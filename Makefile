# =============================================================================
# Makefile — Kalman Filter Milestone 4  (RISC-V Vector Assembly)
#
# Cross-compiles vectorised LKF/EKF assembly with -march=rv64gcv,
# links statically with the C I/O helpers, runs under qemu-riscv64
# with V extension enabled.
#
# Common usage:
#   make                Build both vector binaries
#   make lkf_vec        Build LKF vector binary only
#   make ekf_vec        Build EKF vector binary only
#   make run_lkf_vec    Build + run LKF (writes LKF_vector_output.csv)
#   make run_ekf_vec    Build + run EKF (writes EKF_vector_output.csv)
#   make pipeline       Full LKF → EKF run in sequence
#   make verify         Run numerical verification vs MS3 scalar reference
#   make perf           Run performance analysis (runtime + instruction count)
#   make all_check      pipeline + verify + perf
#   make clean          Remove build artifacts
#   make clean_all      Remove build artifacts AND output CSVs
#
# Optional MS3 scalar build (for direct speedup measurement):
#   make lkf_scalar     Build MS3 scalar LKF (requires MS3_SRC path)
#   make ekf_scalar     Build MS3 scalar EKF
#   make run_lkf_scalar Build + run scalar LKF (writes LKF_asm_output.csv)
#   make run_ekf_scalar Build + run scalar EKF
#
# Configurable variables (override on command line, e.g. make MS3_SRC=...):
#   MS3_SRC   Path to MS3 src/ directory (for scalar baseline build)
#             Defaults to ../MS3_Submission/src
# =============================================================================

# ---- Toolchain ----
CC      := riscv64-linux-gnu-gcc
QEMU    := qemu-riscv64
PYTHON  := python3

# ---- Flags ----
# Static link, optimise C helper (assembly is not affected by -O level),
# warn about everything but suppress the harmless fgets warnings.
CFLAGS    := -O2 -Wall -static -Wno-unused-result
ASFLAGS_V := -march=rv64gcv
ASFLAGS_S := -march=rv64gc

# QEMU: rv64 base + V extension at minimum legal VLEN (128 bits = 2 doubles)
QEMU_CPU  := rv64,v=true,vlen=128
QEMU_RUN  := $(QEMU) -cpu $(QEMU_CPU)
QEMU_RUN_S := $(QEMU)

# ---- Sources / paths ----
MS3_SRC   ?= ../MS3_Submission/src
IO_SRC    := io_helpers.c

# Vector sources (this milestone)
LKF_VEC_SRC := lkf_vector.s
EKF_VEC_SRC := ekf_vector.s

# Vector binaries
LKF_VEC := lkf_vector
EKF_VEC := ekf_vector

# Scalar sources (MS3, optional rebuild for baseline measurement)
LKF_SCALAR_SRC := $(MS3_SRC)/lkf_asm.s
EKF_SCALAR_SRC := $(MS3_SRC)/ekf_asm.s

# Scalar binaries
LKF_SCALAR := lkf_scalar
EKF_SCALAR := ekf_scalar

# Reference outputs (MS3 scalar runs produce these)
LKF_REF := LKF_asm_output.csv
EKF_REF := EKF_asm_output.csv

# Vector outputs (this milestone produces these)
LKF_OUT := LKF_vector_output.csv
EKF_OUT := EKF_vector_output.csv

# Scripts
VERIFY_SCRIPT := verify_ms4.py
PERF_SCRIPT   := perf_ms4.py

# ---- Phony targets ----
.PHONY: all clean clean_all \
        lkf_vec ekf_vec run_lkf_vec run_ekf_vec \
        lkf_scalar ekf_scalar run_lkf_scalar run_ekf_scalar \
        pipeline verify perf all_check help

# ─────────────────────────────────────────────────────────────────────────────
# Default target
# ─────────────────────────────────────────────────────────────────────────────
all: lkf_vec ekf_vec

help:
	@echo "MS4 Makefile targets:"
	@echo "  make              Build both vector binaries"
	@echo "  make pipeline     Run LKF + EKF vector binaries in sequence"
	@echo "  make verify       Compare vector output vs MS3 scalar reference"
	@echo "  make perf         Measure runtime + instruction count vs scalar"
	@echo "  make all_check    pipeline + verify + perf"
	@echo "  make clean        Delete binaries and *.o"
	@echo "  make clean_all    clean + delete vector output CSVs"

# ─────────────────────────────────────────────────────────────────────────────
# VECTOR (MS4) — primary build targets
#
# We invoke gcc directly (not as) because gcc handles -march=rv64gcv
# uniformly across .s and .c, and links libc/libm in one step.
# ─────────────────────────────────────────────────────────────────────────────

$(LKF_VEC): $(LKF_VEC_SRC) $(IO_SRC)
	$(CC) $(ASFLAGS_V) $(CFLAGS) -o $@ $^

$(EKF_VEC): $(EKF_VEC_SRC) $(IO_SRC)
	$(CC) $(ASFLAGS_V) $(CFLAGS) -o $@ $^

lkf_vec: $(LKF_VEC)
ekf_vec: $(EKF_VEC)

# Run targets — produce CSV outputs in the working directory
run_lkf_vec: $(LKF_VEC)
	$(QEMU_RUN) ./$(LKF_VEC)

run_ekf_vec: $(EKF_VEC) $(LKF_OUT)
	$(QEMU_RUN) ./$(EKF_VEC)

# Pipeline: LKF first (its output is consumed by EKF's fallback step)
pipeline: run_lkf_vec
	@echo
	@echo ">>> LKF vector done, starting EKF vector..."
	@echo
	$(QEMU_RUN) ./$(EKF_VEC)
	@echo
	@echo ">>> Pipeline complete. Outputs: $(LKF_OUT), $(EKF_OUT)"

# Force-build LKF output if EKF (or verify) needs it but it isn't there
$(LKF_OUT): $(LKF_VEC)
	$(QEMU_RUN) ./$(LKF_VEC)

# Force-build EKF output if verify needs it but it isn't there
$(EKF_OUT): $(EKF_VEC) $(LKF_OUT)
	$(QEMU_RUN) ./$(EKF_VEC)

# ─────────────────────────────────────────────────────────────────────────────
# SCALAR (MS3) — optional rebuild for direct speedup measurement
#
# Only works if MS3_SRC points to a directory with lkf_asm.s and ekf_asm.s.
# If the path is wrong, the rule fails clearly with a missing-file error.
# ─────────────────────────────────────────────────────────────────────────────

$(LKF_SCALAR): $(LKF_SCALAR_SRC) $(IO_SRC)
	$(CC) $(ASFLAGS_S) $(CFLAGS) -o $@ $^

$(EKF_SCALAR): $(EKF_SCALAR_SRC) $(IO_SRC)
	$(CC) $(ASFLAGS_S) $(CFLAGS) -o $@ $^

lkf_scalar: $(LKF_SCALAR)
ekf_scalar: $(EKF_SCALAR)

run_lkf_scalar: $(LKF_SCALAR)
	$(QEMU_RUN_S) ./$(LKF_SCALAR)

run_ekf_scalar: $(EKF_SCALAR) $(LKF_REF)
	$(QEMU_RUN_S) ./$(EKF_SCALAR)

# Build the LKF scalar reference output if missing (used as input to scalar EKF
# via the lkf_output_ref.csv path inside the MS3 binary). We rename if needed.
$(LKF_REF): $(LKF_SCALAR)
	$(QEMU_RUN_S) ./$(LKF_SCALAR)

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

verify: $(LKF_OUT) $(EKF_OUT)
	$(PYTHON) $(VERIFY_SCRIPT) \
	    --lkf-ref  $(LKF_REF) \
	    --lkf-vec  $(LKF_OUT) \
	    --ekf-ref  $(EKF_REF) \
	    --ekf-vec  $(EKF_OUT)

# ─────────────────────────────────────────────────────────────────────────────
# Performance analysis
# ─────────────────────────────────────────────────────────────────────────────

perf:
	$(PYTHON) $(PERF_SCRIPT) \
	    --lkf-scalar ./$(LKF_SCALAR) \
	    --ekf-scalar ./$(EKF_SCALAR) \
	    --lkf-vector ./$(LKF_VEC) \
	    --ekf-vector ./$(EKF_VEC) \
	    --qemu       "$(QEMU)" \
	    --qemu-cpu   "$(QEMU_CPU)"

all_check: pipeline verify perf

# ─────────────────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────────────────

clean:
	rm -f $(LKF_VEC) $(EKF_VEC) $(LKF_SCALAR) $(EKF_SCALAR) *.o

clean_all: clean
	rm -f $(LKF_OUT) $(EKF_OUT) qemu_*.log perf_results.txt
