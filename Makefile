# =============================================================================
# Makefile — Kalman Filter Milestone 3  (RISC-V Assembly)
#
# Cross-compiles assembly + C helper, links statically, runs under QEMU.
#
# Usage:
#   make lkf          Build LKF binary
#   make run_lkf      Build + run LKF
#   make clean        Remove build artifacts
# =============================================================================

CC      = riscv64-linux-gnu-gcc
AS      = riscv64-linux-gnu-as
LD      = riscv64-linux-gnu-gcc
QEMU    = qemu-riscv64

CFLAGS  = -O2 -Wall -static
ASFLAGS = -march=rv64gc

# Data files (adjust paths as needed)
NOISY   = NoisyValues.csv

.PHONY: all clean lkf ekf run_lkf run_ekf run_all dirs

all: lkf ekf

dirs:
	@mkdir -p build output

# ── LKF ──
build/lkf_asm.o: src/lkf_asm.s | dirs
	$(AS) $(ASFLAGS) -o $@ $<

build/io_helpers.o: src/io_helpers.c | dirs
	$(CC) $(CFLAGS) -c -o $@ $<

build/lkf: build/lkf_asm.o build/io_helpers.o
	$(LD) $(CFLAGS) -o $@ $^ -lm

lkf: build/lkf

run_lkf: build/lkf
	$(QEMU) ./build/lkf

# ── EKF ──
build/ekf_asm.o: src/ekf_asm.s | dirs
	$(AS) $(ASFLAGS) -o $@ $<

build/ekf: build/ekf_asm.o build/io_helpers.o
	$(LD) $(CFLAGS) -o $@ $^ -lm

ekf: build/ekf

run_ekf: build/ekf
	$(QEMU) ./build/ekf

run_all: run_lkf run_ekf

clean:
	rm -rf build output
