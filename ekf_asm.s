# =============================================================
#  ekf_asm.s — RISC-V scalar assembly for EKF
#  Milestone 3: Kalman Filter
#
#  Matches Milestone 2 C++ ekf.cpp exactly:
#    sigma_j = 10.0, R = diag(4.0, 0.36, 0.0049)
#    LU solver with partial pivoting
#    A&S minimax atan2, innovation gating
#    Joseph-form covariance, LKF fallback
# =============================================================

.equ NUM_JOINTS,       23
.equ STATE_PER_JOINT,  12
.equ MEAS_PER_JOINT,   3
.equ N,                276
.equ M,                69
.equ NN,               76176
.equ MM,               4761
.equ NM,               19044
.equ NUM_PTRS,         33

.equ PT_F,         0
.equ PT_Q,         8
.equ PT_R,        16
.equ PT_x,        24
.equ PT_P,        32
.equ PT_Ft,       40
.equ PT_xp,       48
.equ PT_Pp,       56
.equ PT_FP,       64
.equ PT_FPFt,     72
.equ PT_S,        80
.equ PT_Scopy,    88
.equ PT_Hk,       96
.equ PT_Hkt,     104
.equ PT_PHkt,    112
.equ PT_PHktt,   120
.equ PT_Ktsol,   128
.equ PT_K,       136
.equ PT_hpred,   144
.equ PT_zsph,    152
.equ PT_innov,   160
.equ PT_Ky,      168
.equ PT_KHk,     176
.equ PT_IKH,     184
.equ PT_IKHt,    192
.equ PT_IKHP,    200
.equ PT_IKHPIt,  208
.equ PT_KR,      216
.equ PT_KRKt,    224
.equ PT_Kt,      232
.equ PT_Imat,    240
.equ PT_states,  248
.equ PT_FILE,    256

    .section .rodata
noisy_path:     .string "NoisyValues.csv"
true_path:      .string "TrueValues.csv"
lkf_path:       .string "lkf_output_ref.csv"
out_path:       .string "EKF_asm_output.csv"
msg_reading:    .string "[EKF-ASM] Reading datasets...\n"
msg_ts:         .string "[EKF-ASM] Timesteps: %d\n"
msg_building:   .string "[EKF-ASM] Building matrices...\n"
msg_running:    .string "[EKF-ASM] Running filter...\n"
msg_progress:   .string "[EKF-ASM] t=%d/%d\n"
msg_fallback:   .string "[EKF-ASM] Applying LKF fallback for joint %d\n"
msg_nolkf:      .string "[EKF-ASM] LKF file not found, skipping fallback\n"
msg_writing:    .string "[EKF-ASM] Writing output...\n"
msg_done:       .string "[EKF-ASM] Done.\n"

    .section .text

# ==============================================================
#             SHARED UTILITY FUNCTIONS (same as LKF)
# ==============================================================
    .globl zero_mem
zero_mem:
    li      t0, 0
    fcvt.d.w ft0, zero
zm_loop:
    bge     t0, a1, zm_done
    slli    t1, t0, 3
    add     t2, a0, t1
    fsd     ft0, 0(t2)
    addi    t0, t0, 1
    j       zm_loop
zm_done:
    ret

    .globl mat_add
mat_add:
    li      t0, 0
ma_loop:
    bge     t0, a3, ma_done
    slli    t1, t0, 3
    add     t2, a0, t1
    add     t3, a1, t1
    add     t4, a2, t1
    fld     ft0, 0(t2)
    fld     ft1, 0(t3)
    fadd.d  ft2, ft0, ft1
    fsd     ft2, 0(t4)
    addi    t0, t0, 1
    j       ma_loop
ma_done:
    ret

    .globl mat_sub
mat_sub:
    li      t0, 0
ms_loop:
    bge     t0, a3, ms_done
    slli    t1, t0, 3
    add     t2, a0, t1
    add     t3, a1, t1
    add     t4, a2, t1
    fld     ft0, 0(t2)
    fld     ft1, 0(t3)
    fsub.d  ft2, ft0, ft1
    fsd     ft2, 0(t4)
    addi    t0, t0, 1
    j       ms_loop
ms_done:
    ret

    .globl mat_transpose
mat_transpose:
    li      t0, 0
mt_o:
    bge     t0, a2, mt_d
    li      t1, 0
mt_i:
    bge     t1, a3, mt_nr
    mul     t2, t0, a3
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, a0, t2
    mul     t4, t1, a2
    add     t4, t4, t0
    slli    t4, t4, 3
    add     t5, a1, t4
    fld     ft0, 0(t3)
    fsd     ft0, 0(t5)
    addi    t1, t1, 1
    j       mt_i
mt_nr:
    addi    t0, t0, 1
    j       mt_o
mt_d:
    ret

    .globl mat_mul
mat_mul:
    addi    sp, sp, -48
    sd      s0, 0(sp)
    sd      s1, 8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4
    mv      s5, a5
    mul     t0, s3, s5
    li      t1, 0
    fcvt.d.w ft3, zero
mmz:
    bge     t1, t0, mmzd
    slli    t2, t1, 3
    add     t3, s2, t2
    fsd     ft3, 0(t3)
    addi    t1, t1, 1
    j       mmz
mmzd:
    li      t0, 0
mmi:
    bge     t0, s3, mmd
    li      t1, 0
mmk:
    bge     t1, s4, mmni
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s0, t2
    fld     ft0, 0(t3)
    fmv.d.x ft7, zero
    feq.d   a6, ft0, ft7
    bnez    a6, mmnk
    mul     t4, t1, s5
    mul     t5, t0, s5
    li      t6, 0
mmj:
    bge     t6, s5, mmnk
    add     a6, t4, t6
    slli    a6, a6, 3
    add     a7, s1, a6
    fld     ft1, 0(a7)
    add     a6, t5, t6
    slli    a6, a6, 3
    add     a7, s2, a6
    fld     ft2, 0(a7)
    fmadd.d ft2, ft0, ft1, ft2
    fsd     ft2, 0(a7)
    addi    t6, t6, 1
    j       mmj
mmnk:
    addi    t1, t1, 1
    j       mmk
mmni:
    addi    t0, t0, 1
    j       mmi
mmd:
    ld      s0, 0(sp)
    ld      s1, 8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    addi    sp, sp, 48
    ret

# ==============================================================
#       my_sqrt(fa0) -> fa0   (leaf, no calls)
# ==============================================================
    .globl my_sqrt
my_sqrt:
    fmv.d.x ft0, zero
    fle.d   t0, fa0, ft0
    beqz    t0, sq_pos
    fmv.d   fa0, ft0
    ret
sq_pos:
    fsqrt.d ft1, fa0
    fdiv.d  ft2, fa0, ft1
    fadd.d  ft2, ft1, ft2
    li      t0, 0x3FE0000000000000
    fmv.d.x ft3, t0
    fmul.d  fa0, ft2, ft3
    ret

# ==============================================================
#       atan_core(fa0) -> fa0  (leaf, |x|<=1)
#       A&S minimax polynomial, uses fmadd.d
# ==============================================================
    .globl atan_core
atan_core:
    fmul.d  ft0, fa0, fa0       # x²
    # Horner: t = a9
    li      t0, 0x3F9555CBE46F80C1
    fmv.d.x ft1, t0             # a9
    # t = a7 + a9*x²
    li      t0, 0xBFB5CB46BACF7447
    fmv.d.x ft2, t0             # a7
    fmadd.d ft1, ft1, ft0, ft2
    # t = a5 + t*x²
    li      t0, 0x3FC70EDC3BD59924
    fmv.d.x ft2, t0             # a5
    fmadd.d ft1, ft1, ft0, ft2
    # t = a3 + t*x²
    li      t0, 0xBFD523A08398A655
    fmv.d.x ft2, t0             # a3
    fmadd.d ft1, ft1, ft0, ft2
    # t = a1 + t*x²
    li      t0, 0x3FEFFEE6FB4C3C19
    fmv.d.x ft2, t0             # a1
    fmadd.d ft1, ft1, ft0, ft2
    # result = x * t
    fmul.d  fa0, fa0, ft1
    ret

# ==============================================================
#       my_atan2(fa0=y, fa1=x) -> fa0
#       Calls atan_core. Saves ra, fs2-fs6.
# ==============================================================
    .globl my_atan2
my_atan2:
    addi    sp, sp, -64
    sd      ra, 56(sp)
    fsd     fs2, 48(sp)
    fsd     fs3, 40(sp)
    fsd     fs4, 32(sp)
    fsd     fs5, 24(sp)
    fsd     fs6, 16(sp)

    fmv.d   fs2, fa0            # y
    fmv.d   fs3, fa1            # x
    li      t0, 0x400921FB54442D18
    fmv.d.x fs4, t0             # PI
    li      t0, 0x3FF921FB54442D18
    fmv.d.x fs5, t0             # PI/2
    fmv.d.x ft0, zero

    feq.d   t0, fs3, ft0
    beqz    t0, at2_xnz
    feq.d   t0, fs2, ft0
    bnez    t0, at2_rz
    flt.d   t0, fs2, ft0
    bnez    t0, at2_np2
    fmv.d   fa0, fs5
    j       at2_ep
at2_np2:
    fneg.d  fa0, fs5
    j       at2_ep
at2_rz:
    fmv.d   fa0, ft0
    j       at2_ep

at2_xnz:
    fdiv.d  ft1, fs2, fs3       # ratio
    fabs.d  ft2, ft1
    li      t0, 0x3FF0000000000000
    fmv.d.x ft3, t0
    fle.d   t0, ft2, ft3
    beqz    t0, at2_big
    fmv.d   fa0, ft1
    call    atan_core
    j       at2_quad
at2_big:
    fmv.d   fs6, ft1            # save ratio
    li      t0, 0x3FF0000000000000
    fmv.d.x ft3, t0
    fdiv.d  fa0, ft3, fs6
    call    atan_core
    fmv.d.x ft0, zero
    flt.d   t0, fs6, ft0
    bnez    t0, at2_ns
    fsub.d  fa0, fs5, fa0
    j       at2_quad
at2_ns:
    fneg.d  ft5, fs5
    fsub.d  fa0, ft5, fa0

at2_quad:
    fmv.d.x ft0, zero
    flt.d   t0, ft0, fs3        # x > 0?
    bnez    t0, at2_ep
    fle.d   t0, ft0, fs2        # y >= 0?
    beqz    t0, at2_sp
    fadd.d  fa0, fa0, fs4
    j       at2_ep
at2_sp:
    fsub.d  fa0, fa0, fs4

at2_ep:
    fld     fs6, 16(sp)
    fld     fs5, 24(sp)
    fld     fs4, 32(sp)
    fld     fs3, 40(sp)
    fld     fs2, 48(sp)
    ld      ra, 56(sp)
    addi    sp, sp, 64
    ret

# ==============================================================
#  lu_solve(A, B, X, n, m) — LU with partial pivoting
#  Matches Matrix::solve_lu() from ekf.cpp
#  A is COPIED internally (original not modified).
# ==============================================================
    .globl lu_solve
lu_solve:
    addi    sp, sp, -112
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)
    sd      s6, 56(sp)
    sd      s7, 64(sp)
    sd      s8, 72(sp)
    sd      s9, 80(sp)
    sd      s10, 88(sp)
    sd      s11, 96(sp)
    mv      s0, a0              # A (original)
    mv      s1, a1              # B
    mv      s2, a2              # X
    mv      s3, a3              # n
    mv      s4, a4              # m

    # Copy A -> LU
    mul     a0, s3, s3
    slli    a0, a0, 3
    call    malloc
    mv      s5, a0              # s5 = LU
    mul     t0, s3, s3
    li      t1, 0
lca:
    bge     t1, t0, lcad
    slli    t2, t1, 3
    add     t3, s0, t2
    add     t4, s5, t2
    fld     ft0, 0(t3)
    fsd     ft0, 0(t4)
    addi    t1, t1, 1
    j       lca
lcad:
    # Allocate pivot array (n ints, stored as 4 bytes)
    slli    a0, s3, 2
    call    malloc
    mv      s6, a0
    li      t0, 0
lip:
    bge     t0, s3, lipd
    slli    t1, t0, 2
    add     t2, s6, t1
    sw      t0, 0(t2)
    addi    t0, t0, 1
    j       lip
lipd:
    # Allocate PB and y
    mul     a0, s3, s4
    slli    a0, a0, 3
    call    malloc
    mv      s7, a0              # PB
    mul     a0, s3, s4
    slli    a0, a0, 3
    call    malloc
    mv      s8, a0              # y (forward sub result)

    # ---- LU factorisation with partial pivoting ----
    li      s9, 0               # col
lkl:
    bge     s9, s3, lkd
    # Find pivot
    mul     t0, s9, s3
    add     t0, t0, s9
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)
    fsgnjx.d ft1, ft0, ft0      # |A[col][col]|
    mv      s10, s9             # maxRow = col
    addi    t0, s9, 1
lps:
    bge     t0, s3, lpd
    mul     t1, t0, s3
    add     t1, t1, s9
    slli    t1, t1, 3
    add     t2, s5, t1
    fld     ft2, 0(t2)
    fsgnjx.d ft3, ft2, ft2      # |A[row][col]|
    fle.d   t3, ft3, ft1
    bnez    t3, lpn
    fmv.d   ft1, ft3
    mv      s10, t0
lpn:
    addi    t0, t0, 1
    j       lps
lpd:
    beq     s10, s9, lns
    # Swap piv
    slli    t0, s9, 2
    add     t1, s6, t0
    slli    t0, s10, 2
    add     t2, s6, t0
    lw      t3, 0(t1)
    lw      t4, 0(t2)
    sw      t4, 0(t1)
    sw      t3, 0(t2)
    # Swap rows in LU
    li      t0, 0
lsl:
    bge     t0, s3, lns
    mul     t1, s9, s3
    add     t1, t1, t0
    slli    t1, t1, 3
    add     t2, s5, t1
    mul     t3, s10, s3
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s5, t3
    fld     ft4, 0(t2)
    fld     ft5, 0(t4)
    fsd     ft5, 0(t2)
    fsd     ft4, 0(t4)
    addi    t0, t0, 1
    j       lsl
lns:
    # Elimination
    addi    s11, s9, 1
lei:
    bge     s11, s3, led
    mul     t0, s11, s3
    add     t0, t0, s9
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)          # A[row][col]
    mul     t2, s9, s3
    add     t2, t2, s9
    slli    t2, t2, 3
    add     t3, s5, t2
    fld     ft1, 0(t3)          # pivot
    fdiv.d  ft0, ft0, ft1       # factor
    fsd     ft0, 0(t1)
    addi    t4, s9, 1
lej:
    bge     t4, s3, leni
    mul     t5, s11, s3
    add     t5, t5, t4
    slli    t5, t5, 3
    add     t6, s5, t5
    fld     ft2, 0(t6)
    mul     a6, s9, s3
    add     a6, a6, t4
    slli    a6, a6, 3
    add     a7, s5, a6
    fld     ft3, 0(a7)
    fnmsub.d ft2, ft0, ft3, ft2
    fsd     ft2, 0(t6)
    addi    t4, t4, 1
    j       lej
leni:
    addi    s11, s11, 1
    j       lei
led:
    addi    s9, s9, 1
    j       lkl
lkd:
    # ---- Permute B -> PB ----
    li      t0, 0
lpi:
    bge     t0, s3, lpdd
    slli    t1, t0, 2
    add     t2, s6, t1
    lw      t3, 0(t2)           # piv[i]
    li      t4, 0
lpj:
    bge     t4, s4, lpni
    mul     t5, t3, s4
    add     t5, t5, t4
    slli    t5, t5, 3
    add     t6, s1, t5
    fld     ft0, 0(t6)
    mul     a6, t0, s4
    add     a6, a6, t4
    slli    a6, a6, 3
    add     a7, s7, a6
    fsd     ft0, 0(a7)
    addi    t4, t4, 1
    j       lpj
lpni:
    addi    t0, t0, 1
    j       lpi
lpdd:
    # ---- Forward sub: L*y = PB ----
    li      t0, 0
lfi:
    bge     t0, s3, lfd
    li      t1, 0
lfj:
    bge     t1, s4, lfni
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s7, t2
    fld     ft0, 0(t3)          # PB[i][j]
    li      t4, 0
lfk:
    bge     t4, t0, lfs
    mul     t5, t0, s3
    add     t5, t5, t4
    slli    t5, t5, 3
    add     t6, s5, t5
    fld     ft1, 0(t6)          # L[i][k]
    mul     a6, t4, s4
    add     a6, a6, t1
    slli    a6, a6, 3
    add     a7, s8, a6
    fld     ft2, 0(a7)          # y[k][j]
    fnmsub.d ft0, ft1, ft2, ft0
    addi    t4, t4, 1
    j       lfk
lfs:
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s8, t2
    fsd     ft0, 0(t3)
    addi    t1, t1, 1
    j       lfj
lfni:
    addi    t0, t0, 1
    j       lfi
lfd:
    # ---- Back sub: U*X = y ----
    addi    t0, s3, -1
lbi:
    bltz    t0, lbd
    li      t1, 0
lbj:
    bge     t1, s4, lbni
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s8, t2
    fld     ft0, 0(t3)          # y[i][j]
    addi    t4, t0, 1
lbk:
    bge     t4, s3, lbdv
    mul     t5, t0, s3
    add     t5, t5, t4
    slli    t5, t5, 3
    add     t6, s5, t5
    fld     ft1, 0(t6)          # U[i][k]
    mul     a6, t4, s4
    add     a6, a6, t1
    slli    a6, a6, 3
    add     a7, s2, a6
    fld     ft2, 0(a7)          # X[k][j]
    fnmsub.d ft0, ft1, ft2, ft0
    addi    t4, t4, 1
    j       lbk
lbdv:
    mul     t5, t0, s3
    add     t5, t5, t0
    slli    t5, t5, 3
    add     t6, s5, t5
    fld     ft1, 0(t6)          # U[i][i]
    fdiv.d  ft0, ft0, ft1
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s2, t2
    fsd     ft0, 0(t3)
    addi    t1, t1, 1
    j       lbj
lbni:
    addi    t0, t0, -1
    j       lbi
lbd:
    # Free temps
    mv      a0, s5
    call    free
    mv      a0, s6
    call    free
    mv      a0, s7
    call    free
    mv      a0, s8
    call    free
    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    ld      s5, 48(sp)
    ld      s6, 56(sp)
    ld      s7, 64(sp)
    ld      s8, 72(sp)
    ld      s9, 80(sp)
    ld      s10, 88(sp)
    ld      s11, 96(sp)
    addi    sp, sp, 112
    ret

# ==============================================================
#  h_joint(px=fa0, py=fa1, pz=fa2, out=a0)
#  out[0]=r, out[1]=theta, out[2]=phi
#  Calls my_sqrt, my_atan2
# ==============================================================
    .globl h_joint
h_joint:
    addi    sp, sp, -48
    sd      ra, 40(sp)
    sd      s0, 32(sp)
    fsd     fs2, 24(sp)
    fsd     fs3, 16(sp)
    fsd     fs4, 8(sp)
    fsd     fs5, 0(sp)
    mv      s0, a0
    fmv.d   fs2, fa0             # px
    fmv.d   fs3, fa1             # py
    fmv.d   fs4, fa2             # pz
    # rxy² = px²+py²
    fmul.d  ft0, fs2, fs2
    fmadd.d ft0, fs3, fs3, ft0
    fmv.d   fa0, ft0
    call    my_sqrt
    fmv.d   fs5, fa0             # rxy
    # r² = rxy²+pz²
    fmul.d  ft0, fs2, fs2
    fmadd.d ft0, fs3, fs3, ft0
    fmadd.d ft0, fs4, fs4, ft0
    fmv.d   fa0, ft0
    call    my_sqrt
    fsd     fa0, 0(s0)          # r
    # theta = atan2(py, px)
    fmv.d   fa0, fs3
    fmv.d   fa1, fs2
    call    my_atan2
    fsd     fa0, 8(s0)
    # phi = atan2(pz, rxy)
    fmv.d   fa0, fs4
    fmv.d   fa1, fs5
    call    my_atan2
    fsd     fa0, 16(s0)
    fld     fs5, 0(sp)
    fld     fs4, 8(sp)
    fld     fs3, 16(sp)
    fld     fs2, 24(sp)
    ld      s0, 32(sp)
    ld      ra, 40(sp)
    addi    sp, sp, 48
    ret

# ==============================================================
#  jac_joint(px=fa0, py=fa1, pz=fa2, J=a0)
#  Fills 3×12 Jacobian (36 doubles). Uses fsqrt.d directly.
# ==============================================================
    .globl jac_joint
jac_joint:
    mv      t0, a0
    # Zero J (36 doubles)
    li      t1, 36
    fmv.d.x ft0, zero
jz_lp:
    blez    t1, jz_d
    fsd     ft0, 0(t0)
    addi    t0, t0, 8
    addi    t1, t1, -1
    j       jz_lp
jz_d:
    # rxy² = px²+py²
    fmul.d  ft1, fa0, fa0
    fmadd.d ft1, fa1, fa1, ft1
    fsqrt.d ft2, ft1             # rxy
    fmadd.d ft3, fa2, fa2, ft1   # r²
    fsqrt.d ft4, ft3             # r
    # Guard
    li      t0, 0x3D719799812DEA11
    fmv.d.x ft5, t0             # 1e-12
    flt.d   t0, ft4, ft5
    bnez    t0, jac_ret
    flt.d   t0, ft2, ft5
    bnez    t0, jac_ret
    # Row 0: range
    fdiv.d  ft5, fa0, ft4       # px/r
    fsd     ft5, 0(a0)          # J[0]
    fdiv.d  ft5, fa1, ft4       # py/r
    fsd     ft5, 32(a0)         # J[4]
    fdiv.d  ft5, fa2, ft4       # pz/r
    fsd     ft5, 64(a0)         # J[8]
    # Row 1: theta
    fneg.d  ft5, fa1
    fdiv.d  ft5, ft5, ft1       # -py/rxy²
    fsd     ft5, 96(a0)         # J[12]
    fdiv.d  ft5, fa0, ft1       # px/rxy²
    fsd     ft5, 128(a0)        # J[16]
    # Row 2: phi
    fmul.d  ft6, ft3, ft2       # r²*rxy
    fmul.d  ft5, fa0, fa2
    fneg.d  ft5, ft5
    fdiv.d  ft5, ft5, ft6       # -px*pz/(r²*rxy)
    fsd     ft5, 192(a0)        # J[24]
    fmul.d  ft5, fa1, fa2
    fneg.d  ft5, ft5
    fdiv.d  ft5, ft5, ft6       # -py*pz/(r²*rxy)
    fsd     ft5, 224(a0)        # J[28]
    fdiv.d  ft5, ft2, ft3       # rxy/r²
    fsd     ft5, 256(a0)        # J[32]
jac_ret:
    ret

# ==============================================================
#                         MAIN
# ==============================================================
    .globl main
main:
    addi    sp, sp, -80
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)
    sd      s6, 56(sp)
    sd      s7, 64(sp)

    # ---- Read CSVs ----
    la      a0, msg_reading
    call    printf

    la      a0, noisy_path
    addi    a1, sp, 72
    addi    a2, sp, 76
    call    read_csv
    mv      s0, a0              # s0 = noisy data
    lw      s1, 72(sp)          # s1 = T
    lw      s2, 76(sp)          # s2 = cols (69)

    la      a0, true_path
    addi    a1, sp, 72
    addi    a2, sp, 76
    call    read_csv
    mv      s5, a0              # s5 = true data

    la      a0, msg_ts
    mv      a1, s1
    call    printf

    # ---- Allocate pointer table ----
    li      a0, NUM_PTRS*8
    call    malloc
    mv      s3, a0

    la      a0, msg_building
    call    printf

    # ---- Allocate matrices ----
    li      a0, NN*8
    call    malloc
    sd      a0, PT_F(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Q(s3)
    li      a0, MM*8
    call    malloc
    sd      a0, PT_R(s3)
    li      a0, N*8
    call    malloc
    sd      a0, PT_x(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_P(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Ft(s3)
    li      a0, N*8
    call    malloc
    sd      a0, PT_xp(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Pp(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_FP(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_FPFt(s3)
    li      a0, MM*8
    call    malloc
    sd      a0, PT_S(s3)
    li      a0, MM*8
    call    malloc
    sd      a0, PT_Scopy(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Hk(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Hkt(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_PHkt(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_PHktt(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Ktsol(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_K(s3)
    li      a0, M*8
    call    malloc
    sd      a0, PT_hpred(s3)
    li      a0, M*8
    call    malloc
    sd      a0, PT_zsph(s3)
    li      a0, M*8
    call    malloc
    sd      a0, PT_innov(s3)
    li      a0, N*8
    call    malloc
    sd      a0, PT_Ky(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_KHk(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKH(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKHt(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKHP(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKHPIt(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_KR(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_KRKt(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Kt(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Imat(s3)
    # states_out: T * N doubles
    mv      a0, s1
    li      t0, N
    mul     a0, a0, t0
    slli    a0, a0, 3
    call    malloc
    sd      a0, PT_states(s3)

    # ==== BUILD F (same as LKF) ====
    ld      a0, PT_F(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_F(s3)
    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0
    fmul.d  ft1, ft0, ft0
    li      t0, 0x3FE0000000000000
    fmv.d.x ft8, t0
    fmul.d  ft2, ft8, ft1
    fmul.d  ft3, ft1, ft0
    li      t0, 0x3FC5555555555555
    fmv.d.x ft9, t0
    fmul.d  ft4, ft9, ft3
    li      t0, 0x3FF0000000000000
    fmv.d.x ft5, t0
    li      t0, 0
eF_jt:
    li      t1, NUM_JOINTS
    bge     t0, t1, eF_done
    li      t2, 0
eF_ax:
    li      t3, 3
    bge     t2, t3, eF_njt
    li      t4, 12
    mul     t4, t0, t4
    slli    t5, t2, 2
    add     t4, t4, t5
    li      a1, N
    mul     a2, t4, a1
    add     a3, a2, t4
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)
    addi    a3, t4, 1
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft0, 0(a4)
    addi    a3, t4, 2
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft2, 0(a4)
    addi    a3, t4, 3
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft4, 0(a4)
    addi    a5, t4, 1
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)
    addi    a3, a5, 1
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft0, 0(a4)
    addi    a3, a5, 2
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft2, 0(a4)
    addi    a5, t4, 2
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)
    addi    a3, a5, 1
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft0, 0(a4)
    addi    a5, t4, 3
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)
    addi    t2, t2, 1
    j       eF_ax
eF_njt:
    addi    t0, t0, 1
    j       eF_jt
eF_done:

    # ==== BUILD Q (sigma_j=10, s2=100) — same structure, different s2 ====
    ld      a0, PT_Q(s3)
    li      a1, NN
    call    zero_mem
    li      a0, 128
    call    malloc
    mv      s4, a0
    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0
    fmul.d  ft1, ft0, ft0
    fmul.d  ft2, ft1, ft0
    fmul.d  ft3, ft2, ft0
    fmul.d  ft4, ft3, ft0
    fmul.d  ft5, ft4, ft0
    li      t0, 0x4059000000000000
    fmv.d.x fs0, t0             # s2 = 100.0
    li      t0, 0x3FF0000000000000
    fmv.d.x ft9, t0
    li      t0, 0x4042000000000000
    fmv.d.x ft8, t0
    li      t0, 0x4028000000000000
    fmv.d.x ft7, t0
    li      t0, 0x4018000000000000
    fmv.d.x ft6, t0
    fdiv.d  fa0, ft5, ft8
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 0(s4)
    fdiv.d  fa0, ft4, ft7
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 8(s4)
    fdiv.d  fa0, ft3, ft6
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 16(s4)
    fdiv.d  fa0, ft2, ft6
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 24(s4)
    fld     fa0, 8(s4)
    fsd     fa0, 32(s4)
    li      t0, 0x4010000000000000
    fmv.d.x fa1, t0
    fdiv.d  fa0, ft3, fa1
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 40(s4)
    li      t0, 0x4000000000000000
    fmv.d.x fa1, t0
    fdiv.d  fa0, ft2, fa1
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 48(s4)
    fdiv.d  fa0, ft1, fa1
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 56(s4)
    fld     fa0, 16(s4)
    fsd     fa0, 64(s4)
    fld     fa0, 48(s4)
    fsd     fa0, 72(s4)
    fmul.d  fa0, ft1, fs0
    fsd     fa0, 80(s4)
    fmul.d  fa0, ft0, fs0
    fsd     fa0, 88(s4)
    fld     fa0, 24(s4)
    fsd     fa0, 96(s4)
    fld     fa0, 56(s4)
    fsd     fa0, 104(s4)
    fld     fa0, 88(s4)
    fsd     fa0, 112(s4)
    fmul.d  fa0, ft9, fs0
    fsd     fa0, 120(s4)

    ld      a5, PT_Q(s3)
    li      t0, 0
eQ_jt:
    li      t1, NUM_JOINTS
    bge     t0, t1, eQ_done
    li      t2, 0
eQ_ax:
    li      t3, 3
    bge     t2, t3, eQ_njt
    li      t4, 0
eQ_i:
    li      t5, 4
    bge     t4, t5, eQ_nax
    li      t6, 0
eQ_j:
    li      a1, 4
    bge     t6, a1, eQ_ni
    slli    a2, t4, 2
    add     a2, a2, t6
    slli    a2, a2, 3
    add     a3, s4, a2
    fld     ft10, 0(a3)
    li      a1, 12
    mul     a2, t0, a1
    slli    a3, t2, 2
    add     a2, a2, a3
    add     a2, a2, t4
    mul     a3, t0, a1
    slli    a4, t2, 2
    add     a3, a3, a4
    add     a3, a3, t6
    li      a1, N
    mul     a4, a2, a1
    add     a4, a4, a3
    slli    a4, a4, 3
    add     a4, a5, a4
    fsd     ft10, 0(a4)
    addi    t6, t6, 1
    j       eQ_j
eQ_ni:
    addi    t4, t4, 1
    j       eQ_i
eQ_nax:
    addi    t2, t2, 1
    j       eQ_ax
eQ_njt:
    addi    t0, t0, 1
    j       eQ_jt
eQ_done:
    mv      a0, s4
    call    free

    # ==== BUILD R (spherical: var_r, var_th, var_phi per joint) ====
    ld      a0, PT_R(s3)
    li      a1, MM
    call    zero_mem
    ld      s4, PT_R(s3)
    li      t0, 0x4010000000000000
    fmv.d.x ft0, t0             # var_r = 4.0
    li      t0, 0x3FD70A3D70A3D70A
    fmv.d.x ft1, t0             # var_th = 0.36
    li      t0, 0x3F741205BC01A36F
    fmv.d.x ft2, t0             # var_phi = 0.0049
    li      t1, 0
eR_lp:
    li      t2, NUM_JOINTS
    bge     t1, t2, eR_done
    li      t3, 3
    mul     t4, t1, t3          # j*3
    li      a1, M
    # R[j*3][j*3] = var_r
    mul     a2, t4, a1
    add     a2, a2, t4
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    # R[j*3+1][j*3+1] = var_th
    addi    a4, t4, 1
    mul     a2, a4, a1
    add     a2, a2, a4
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft1, 0(a3)
    # R[j*3+2][j*3+2] = var_phi
    addi    a4, t4, 2
    mul     a2, a4, a1
    add     a2, a2, a4
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft2, 0(a3)
    addi    t1, t1, 1
    j       eR_lp
eR_done:

    # ==== BUILD I_mat ====
    ld      a0, PT_Imat(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_Imat(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
eI_lp:
    li      t2, N
    bge     t1, t2, eI_done
    li      a1, N
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       eI_lp
eI_done:

    # ==== INIT x from TRUE frame[0] ====
    ld      a0, PT_x(s3)
    li      a1, N
    call    zero_mem
    ld      s4, PT_x(s3)
    li      t0, 0
eix_lp:
    li      t1, NUM_JOINTS
    bge     t0, t1, eix_done
    li      a1, 3
    mul     a2, t0, a1
    slli    a2, a2, 3
    add     a3, s5, a2          # s5 = true data
    fld     ft0, 0(a3)
    fld     ft1, 8(a3)
    fld     ft2, 16(a3)
    li      a1, 12
    mul     a2, t0, a1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    fsd     ft1, 32(a3)
    fsd     ft2, 64(a3)
    addi    t0, t0, 1
    j       eix_lp
eix_done:

    # ==== INIT P = I ====
    ld      a0, PT_P(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_P(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
eiP_lp:
    li      t2, N
    bge     t1, t2, eiP_done
    li      a1, N
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       eiP_lp
eiP_done:

    # Ft
    ld      a0, PT_F(s3)
    ld      a1, PT_Ft(s3)
    li      a2, N
    li      a3, N
    call    mat_transpose

    # Free true data (no longer needed)
    mv      a0, s5
    call    free

    # ==== FILTER LOOP ====
    la      a0, msg_running
    call    printf

    li      s4, 0
efl:
    bge     s4, s1, efl_done

    # -- PREDICT --
    ld      a0, PT_F(s3)
    ld      a1, PT_x(s3)
    ld      a2, PT_xp(s3)
    li      a3, N
    li      a4, N
    li      a5, 1
    call    mat_mul

    ld      a0, PT_F(s3)
    ld      a1, PT_P(s3)
    ld      a2, PT_FP(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    ld      a0, PT_FP(s3)
    ld      a1, PT_Ft(s3)
    ld      a2, PT_FPFt(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    ld      a0, PT_FPFt(s3)
    ld      a1, PT_Q(s3)
    ld      a2, PT_Pp(s3)
    li      a3, NN
    call    mat_add

    # -- h_pred = h_full(x_pred) --
    ld      s6, PT_xp(s3)
    ld      s7, PT_hpred(s3)
    li      t0, 0
eh_lp:
    li      t1, NUM_JOINTS
    bge     t0, t1, eh_done
    li      t2, 12
    mul     t2, t0, t2
    slli    t2, t2, 3
    add     t3, s6, t2
    fld     fa0, 0(t3)         # px
    fld     fa1, 32(t3)        # py (offset 4*8)
    fld     fa2, 64(t3)        # pz (offset 8*8)
    li      t2, 3
    mul     t2, t0, t2
    slli    t2, t2, 3
    add     a0, s7, t2         # &h_pred[j*3]
    # Save t0 (loop counter) across call
    addi    sp, sp, -16
    sd      t0, 0(sp)
    call    h_joint
    ld      t0, 0(sp)
    addi    sp, sp, 16
    addi    t0, t0, 1
    j       eh_lp
eh_done:

    # -- z_sph = cartesian_to_spherical(noisy[t]) --
    mv      a0, s4
    mv      a1, s2
    mul     a2, a0, a1
    slli    a2, a2, 3
    add     s6, s0, a2          # &noisy[t*cols]
    ld      s7, PT_zsph(s3)
    li      t0, 0
ez_lp:
    li      t1, NUM_JOINTS
    bge     t0, t1, ez_done
    li      t2, 3
    mul     t2, t0, t2
    slli    t2, t2, 3
    add     t3, s6, t2
    fld     fa0, 0(t3)
    fld     fa1, 8(t3)
    fld     fa2, 16(t3)
    add     a0, s7, t2
    addi    sp, sp, -16
    sd      t0, 0(sp)
    call    h_joint
    ld      t0, 0(sp)
    addi    sp, sp, 16
    addi    t0, t0, 1
    j       ez_lp
ez_done:

    # -- Innovation gating --
    ld      a0, PT_zsph(s3)
    ld      a1, PT_hpred(s3)
    li      t0, 0x4018000000000000
    fmv.d.x ft8, t0             # GATE_R = 6.0
    li      t0, 0x3FF0000000000000
    fmv.d.x ft9, t0             # GATE_ANG = 1.0
    li      t0, 0
eg_lp:
    li      t1, NUM_JOINTS
    bge     t0, t1, eg_done
    li      t2, 3
    mul     t3, t0, t2
    slli    t3, t3, 3
    add     t4, a0, t3
    add     t5, a1, t3
    fld     ft0, 0(t4)         # z_sph[j*3]
    fld     ft1, 0(t5)         # h_pred[j*3]
    fsub.d  ft2, ft0, ft1      # dr
    fabs.d  ft2, ft2
    fld     ft3, 8(t4)
    fld     ft4, 8(t5)
    fsub.d  ft5, ft3, ft4      # dth
    fabs.d  ft5, ft5
    fld     ft3, 16(t4)
    fld     ft4, 16(t5)
    fsub.d  ft6, ft3, ft4      # dph
    fabs.d  ft6, ft6
    # Check gates
    flt.d   t2, ft8, ft2       # GATE_R < |dr| → gated
    bnez    t2, eg_gate
    flt.d   t2, ft9, ft5
    bnez    t2, eg_gate
    flt.d   t2, ft9, ft6
    bnez    t2, eg_gate
    j       eg_next
eg_gate:
    fld     ft0, 0(t5)
    fsd     ft0, 0(t4)
    fld     ft0, 8(t5)
    fsd     ft0, 8(t4)
    fld     ft0, 16(t5)
    fsd     ft0, 16(t4)
eg_next:
    addi    t0, t0, 1
    j       eg_lp
eg_done:

    # -- Hk = H_full_jacobian(x_pred) --
    ld      a0, PT_Hk(s3)
    li      a1, NM
    call    zero_mem
    ld      s6, PT_xp(s3)
    ld      s7, PT_Hk(s3)
    li      t0, 0
ej_lp:
    li      t1, NUM_JOINTS
    bge     t0, t1, ej_done
    # Load joint positions from x_pred
    li      t2, 12
    mul     t2, t0, t2
    slli    t2, t2, 3
    add     t3, s6, t2
    fld     fa0, 0(t3)
    fld     fa1, 32(t3)
    fld     fa2, 64(t3)
    # Temp 3×12 Jacobian on stack
    addi    sp, sp, -304        # 36*8=288 for Jac + 16 for t0 + padding
    sd      t0, 296(sp)
    mv      a0, sp              # J temp at sp
    call    jac_joint
    ld      t0, 296(sp)
    # Copy 3×12 block into Hk at rows j*3..j*3+2, cols j*12..j*12+11
    li      t2, 0               # row 0..2
ej_r:
    li      t3, 3
    bge     t2, t3, ej_rd
    li      t4, 0               # col 0..11
ej_c:
    li      t5, 12
    bge     t4, t5, ej_cr
    # src: sp[r*12+c]
    mul     t5, t2, t5
    add     t5, t5, t4
    slli    t5, t5, 3
    add     t6, sp, t5
    fld     ft0, 0(t6)
    # dst: Hk[(j*3+r)*N + j*12+c]
    li      a1, 3
    mul     a2, t0, a1
    add     a2, a2, t2          # j*3+r
    li      a1, N
    mul     a3, a2, a1          # (j*3+r)*N
    li      a1, 12
    mul     a4, t0, a1
    add     a4, a4, t4          # j*12+c
    add     a3, a3, a4
    slli    a3, a3, 3
    add     a4, s7, a3
    fsd     ft0, 0(a4)
    addi    t4, t4, 1
    j       ej_c
ej_cr:
    addi    t2, t2, 1
    j       ej_r
ej_rd:
    addi    sp, sp, 304
    addi    t0, t0, 1
    j       ej_lp
ej_done:

    # -- PHkt = P_pred * Hk^T --
    ld      a0, PT_Hk(s3)
    ld      a1, PT_Hkt(s3)
    li      a2, M
    li      a3, N
    call    mat_transpose

    ld      a0, PT_Pp(s3)
    ld      a1, PT_Hkt(s3)
    ld      a2, PT_PHkt(s3)
    li      a3, N
    li      a4, N
    li      a5, M
    call    mat_mul

    # -- S = Hk * PHkt + R --
    ld      a0, PT_Hk(s3)
    ld      a1, PT_PHkt(s3)
    ld      a2, PT_S(s3)
    li      a3, M
    li      a4, N
    li      a5, M
    call    mat_mul

    ld      a0, PT_S(s3)
    ld      a1, PT_R(s3)
    ld      a2, PT_S(s3)
    li      a3, MM
    call    mat_add

    # -- Solve S * Kt_sol = PHkt^T --
    ld      a0, PT_PHkt(s3)
    ld      a1, PT_PHktt(s3)
    li      a2, N
    li      a3, M
    call    mat_transpose

    ld      a0, PT_S(s3)
    ld      a1, PT_PHktt(s3)
    ld      a2, PT_Ktsol(s3)
    li      a3, M
    li      a4, N
    call    lu_solve

    # -- K = Kt_sol^T --
    ld      a0, PT_Ktsol(s3)
    ld      a1, PT_K(s3)
    li      a2, M
    li      a3, N
    call    mat_transpose

    # -- innov = z_gated - h_pred --
    ld      a0, PT_zsph(s3)
    ld      a1, PT_hpred(s3)
    ld      a2, PT_innov(s3)
    li      a3, M
    call    mat_sub

    # -- x = x_pred + K*innov --
    ld      a0, PT_K(s3)
    ld      a1, PT_innov(s3)
    ld      a2, PT_Ky(s3)
    li      a3, N
    li      a4, M
    li      a5, 1
    call    mat_mul

    ld      a0, PT_xp(s3)
    ld      a1, PT_Ky(s3)
    ld      a2, PT_x(s3)
    li      a3, N
    call    mat_add

    # -- Joseph form P update --
    ld      a0, PT_K(s3)
    ld      a1, PT_Hk(s3)
    ld      a2, PT_KHk(s3)
    li      a3, N
    li      a4, M
    li      a5, N
    call    mat_mul

    ld      a0, PT_Imat(s3)
    ld      a1, PT_KHk(s3)
    ld      a2, PT_IKH(s3)
    li      a3, NN
    call    mat_sub

    ld      a0, PT_IKH(s3)
    ld      a1, PT_IKHt(s3)
    li      a2, N
    li      a3, N
    call    mat_transpose

    ld      a0, PT_IKH(s3)
    ld      a1, PT_Pp(s3)
    ld      a2, PT_IKHP(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    ld      a0, PT_IKHP(s3)
    ld      a1, PT_IKHt(s3)
    ld      a2, PT_IKHPIt(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    ld      a0, PT_K(s3)
    ld      a1, PT_R(s3)
    ld      a2, PT_KR(s3)
    li      a3, N
    li      a4, M
    li      a5, M
    call    mat_mul

    ld      a0, PT_K(s3)
    ld      a1, PT_Kt(s3)
    li      a2, N
    li      a3, M
    call    mat_transpose

    ld      a0, PT_KR(s3)
    ld      a1, PT_Kt(s3)
    ld      a2, PT_KRKt(s3)
    li      a3, N
    li      a4, M
    li      a5, N
    call    mat_mul

    ld      a0, PT_IKHPIt(s3)
    ld      a1, PT_KRKt(s3)
    ld      a2, PT_P(s3)
    li      a3, NN
    call    mat_add

    # -- Store state in output array --
    ld      a0, PT_states(s3)
    mv      a1, s4
    li      t0, N
    mul     a1, a1, t0
    slli    a1, a1, 3
    add     a0, a0, a1          # &states[t*N]
    ld      a1, PT_x(s3)
    li      t0, 0
est_lp:
    li      t1, N
    bge     t0, t1, est_done
    slli    t2, t0, 3
    add     t3, a1, t2
    add     t4, a0, t2
    fld     ft0, 0(t3)
    fsd     ft0, 0(t4)
    addi    t0, t0, 1
    j       est_lp
est_done:

    # -- Progress --
    li      t0, 500
    rem     t1, s4, t0
    bnez    t1, enp
    la      a0, msg_progress
    mv      a1, s4
    mv      a2, s1
    call    printf
enp:
    addi    s4, s4, 1
    j       efl

efl_done:

    # ==== LKF FALLBACK ====
    addi    sp, sp, -16
    sd      zero, 0(sp)         # out_rows
    la      a0, lkf_path
    addi    a1, sp, 0
    call    read_state_csv
    mv      s5, a0              # lkf states (or NULL)
    lw      s6, 0(sp)           # lkf T
    addi    sp, sp, 16

    beqz    s5, efb_skip
    bne     s6, s1, efb_skip

    # Fallback joints: 10, 17, 18, 21, 22
    # For each: copy 12 doubles per frame from LKF to EKF states
    li      t0, 10
    la      a0, msg_fallback
    mv      a1, t0
    call    printf
    li      t0, 10
    call    do_fallback

    li      t0, 17
    la      a0, msg_fallback
    mv      a1, t0
    call    printf
    li      t0, 17
    call    do_fallback

    li      t0, 18
    la      a0, msg_fallback
    mv      a1, t0
    call    printf
    li      t0, 18
    call    do_fallback

    li      t0, 21
    la      a0, msg_fallback
    mv      a1, t0
    call    printf
    li      t0, 21
    call    do_fallback

    li      t0, 22
    la      a0, msg_fallback
    mv      a1, t0
    call    printf
    li      t0, 22
    call    do_fallback

    j       efb_write

efb_skip:
    la      a0, msg_nolkf
    call    printf

efb_write:
    # ==== Write output ====
    la      a0, msg_writing
    call    printf
    la      a0, out_path
    call    open_output
    mv      s6, a0              # FILE*
    ld      a1, PT_states(s3)
    mv      a2, s1              # T
    li      a3, N
    mv      a0, s6
    call    write_all_states
    mv      a0, s6
    call    close_output

    la      a0, msg_done
    call    printf

    li      a0, 0
    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    ld      s5, 48(sp)
    ld      s6, 56(sp)
    ld      s7, 64(sp)
    addi    sp, sp, 80
    ret

# ==============================================================
#  do_fallback: t0 = joint index
#  Copies 12 state values per frame from LKF(s5) to EKF states(PT_states)
#  Uses s1=T, s3=ptbl, s5=lkf_states
# ==============================================================
do_fallback:
    ld      a0, PT_states(s3)   # ekf states
    # offset = joint * 12
    li      t1, 12
    mul     t2, t0, t1          # joint * 12
    li      t3, 0               # frame counter
dfb_f:
    bge     t3, s1, dfb_done
    # src = lkf_states + frame*N + joint*12
    li      t4, N
    mul     t5, t3, t4
    add     t5, t5, t2          # frame*N + joint*12
    slli    t5, t5, 3
    add     t6, s5, t5          # &lkf[frame*N + joint*12]
    add     a1, a0, t5          # &ekf[frame*N + joint*12]
    # Copy 12 doubles
    li      t4, 0
dfb_c:
    li      t5, 12
    bge     t4, t5, dfb_nf
    slli    a2, t4, 3
    add     a3, t6, a2
    add     a4, a1, a2
    fld     ft0, 0(a3)
    fsd     ft0, 0(a4)
    addi    t4, t4, 1
    j       dfb_c
dfb_nf:
    addi    t3, t3, 1
    j       dfb_f
dfb_done:
    ret
