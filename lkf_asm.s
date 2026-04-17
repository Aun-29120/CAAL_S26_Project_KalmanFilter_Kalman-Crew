# =============================================================
#  lkf_asm.s — RISC-V scalar assembly for LKF
#  Milestone 3: Kalman Filter
#
#  Matches Milestone 2 C++ implementation exactly:
#    sigma_j = 0.5,  sigma_r = 0.3593
#    LDL^T solver,   Joseph-form covariance update
#
#  Register plan for main:
#    s0 = noisy data pointer (flat array)
#    s1 = T (number of timesteps)
#    s2 = csv_cols (69)
#    s3 = ptbl (pointer table — all matrix pointers)
#    s4 = t (loop counter)
#
#  Pointer table indices (ptbl[i] at offset i*8):
#    0=F  1=Q  2=H  3=Ht  4=R  5=x  6=P  7=Ft
#    8=x_pred  9=P_pred  10=FP  11=FPFt  12=S
#    13=HP  14=HPHt  15=PHt  16=PHt_t  17=Kt_sol
#    18=K  19=z  20=Hx  21=y  22=Ky  23=KH
#    24=IKH  25=IKHt  26=IKH_P  27=IKH_P_It
#    28=KR  29=KRKt  30=Kt  31=I_mat  32=FILE*
#    33=ldl_D  34=ldl_v
# =============================================================

.equ NUM_JOINTS,       23
.equ STATE_PER_JOINT,  12
.equ MEAS_PER_JOINT,   3
.equ N,                276
.equ M,                69
.equ NN,               76176
.equ MM,               4761
.equ NM,               19044
.equ NUM_PTRS,         35

# Pointer table offsets (index * 8)
.equ PT_F,         0
.equ PT_Q,         8
.equ PT_H,        16
.equ PT_Ht,       24
.equ PT_R,        32
.equ PT_x,        40
.equ PT_P,        48
.equ PT_Ft,       56
.equ PT_xp,       64
.equ PT_Pp,       72
.equ PT_FP,       80
.equ PT_FPFt,     88
.equ PT_S,        96
.equ PT_HP,      104
.equ PT_HPHt,    112
.equ PT_PHt,     120
.equ PT_PHtt,    128
.equ PT_Ktsol,   136
.equ PT_K,       144
.equ PT_z,       152
.equ PT_Hx,      160
.equ PT_y,       168
.equ PT_Ky,      176
.equ PT_KH,      184
.equ PT_IKH,     192
.equ PT_IKHt,    200
.equ PT_IKHP,    208
.equ PT_IKHPIt,  216
.equ PT_KR,      224
.equ PT_KRKt,    232
.equ PT_Kt,      240
.equ PT_Imat,    248
.equ PT_FILE,    256
.equ PT_D,       264
.equ PT_v,       272

    .section .rodata
csv_path:       .string "NoisyValues.csv"
out_path:       .string "LKF_asm_output.csv"
msg_reading:    .string "[LKF-ASM] Reading dataset...\n"
msg_ts:         .string "[LKF-ASM] Timesteps: %d, Cols: %d\n"
msg_building:   .string "[LKF-ASM] Building matrices...\n"
msg_running:    .string "[LKF-ASM] Running filter...\n"
msg_progress:   .string "[LKF-ASM] t=%d/%d\n"
msg_done:       .string "[LKF-ASM] Done. Output saved.\n"

    .section .text

# ==============================================================
#                 UTILITY: zero_mem(ptr, count)
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

# ==============================================================
#                    MATH FUNCTIONS
# ==============================================================

# --- mat_add(A, B, C, size) ---
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

# --- mat_sub(A, B, C, size) ---
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

# --- mat_transpose(A, B, rows, cols) ---
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

# --- mat_mul(A, B, C, rowsA, colsA, colsB) ---
#   Uses fmadd.d for inner multiply-accumulate
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
    # Zero C
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
    # Load A[i*cA + k]
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s0, t2
    fld     ft0, 0(t3)
    # Skip if zero (sparse optimization)
    fmv.d.x ft7, zero
    feq.d   a6, ft0, ft7
    bnez    a6, mmnk
    # Inner j loop
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
#  ldl_solve(S, B, X, n, m, D, v)
#
#  Solve S*X = B via LDL^T decomposition.
#  S is n×n SPD (MODIFIED in-place: lower tri → L).
#  D is n doubles workspace, v is n doubles workspace.
#
#  Matches Matrix::solve() from MS2 lkf.cpp exactly.
#
#  Register map:
#    s0=S  s1=B  s2=X  s3=n  s4=m  s5=D  s6=v
#    s7,s8,s9,s10,s11 = temps for row offsets
# ==============================================================
    .globl ldl_solve
ldl_solve:
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
    mv      s0, a0              # S
    mv      s1, a1              # B
    mv      s2, a2              # X
    mv      s3, a3              # n
    mv      s4, a4              # m
    mv      s5, a5              # D
    mv      s6, a6              # v

    # Build 1e-14 floor constant in fs1
    li      t0, 0x3D06849B86A12B9B
    fmv.d.x fs1, t0

    # ============ STEP 1: LDL^T Factorisation ============
    # For j = 0..n-1:
    #   For i = 0..j-1: v[i] = S[j*n+i] * D[i]
    #   D[j] = S[j*n+j] - sum(S[j*n+i]*v[i], i=0..j-1)
    #   if |D[j]| < 1e-14: D[j] = 1e-14
    #   For k = j+1..n-1:
    #     S[k*n+j] = (S[k*n+j] - sum(S[k*n+i]*v[i], i=0..j-1)) / D[j]

    li      s7, 0               # j = 0
ldl_j:
    bge     s7, s3, ldl_fdone

    # --- v[i] = S[j*n+i] * D[i] for i=0..j-1 ---
    li      s8, 0               # i = 0
    mul     s9, s7, s3          # j * n
ldl_v:
    bge     s8, s7, ldl_vd
    add     t0, s9, s8          # j*n + i
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)          # S[j][i] = L[j][i]
    slli    t1, s8, 3
    add     t1, s5, t1
    fld     ft1, 0(t1)          # D[i]
    fmul.d  ft2, ft0, ft1       # v[i] = L[j][i]*D[i]
    slli    t1, s8, 3
    add     t1, s6, t1
    fsd     ft2, 0(t1)
    addi    s8, s8, 1
    j       ldl_v
ldl_vd:

    # --- D[j] = S[j][j] - sum(L[j][i]*v[i]) ---
    add     t0, s9, s7          # j*n + j
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft5, 0(t0)          # accumulator = S[j][j]
    li      s8, 0
ldl_ds:
    bge     s8, s7, ldl_dsd
    add     t0, s9, s8
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)          # L[j][i]
    slli    t1, s8, 3
    add     t1, s6, t1
    fld     ft1, 0(t1)          # v[i]
    fnmsub.d ft5, ft0, ft1, ft5 # D[j] -= L[j][i]*v[i]
    addi    s8, s8, 1
    j       ldl_ds
ldl_dsd:

    # Floor |D[j]| < 1e-14
    fabs.d  ft6, ft5
    flt.d   t0, ft6, fs1
    beqz    t0, ldl_nf
    fmv.d   ft5, fs1
ldl_nf:
    slli    t0, s7, 3
    add     t0, s5, t0
    fsd     ft5, 0(t0)          # store D[j]

    # --- L[k][j] for k = j+1..n-1 ---
    addi    s10, s7, 1          # k = j+1
ldl_lk:
    bge     s10, s3, ldl_lkd
    mul     s11, s10, s3        # k*n
    add     t0, s11, s7         # k*n + j
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft6, 0(t0)          # S[k][j]
    li      s8, 0               # i = 0
ldl_lks:
    bge     s8, s7, ldl_lksd
    add     t0, s11, s8         # k*n + i
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)          # L[k][i]
    slli    t1, s8, 3
    add     t1, s6, t1
    fld     ft1, 0(t1)          # v[i]
    fnmsub.d ft6, ft0, ft1, ft6
    addi    s8, s8, 1
    j       ldl_lks
ldl_lksd:
    fdiv.d  ft6, ft6, ft5       # / D[j]
    mul     t0, s10, s3
    add     t0, t0, s7
    slli    t0, t0, 3
    add     t0, s0, t0
    fsd     ft6, 0(t0)          # S[k*n+j] = L[k][j]
    addi    s10, s10, 1
    j       ldl_lk
ldl_lkd:

    addi    s7, s7, 1
    j       ldl_j
ldl_fdone:

    # ============ STEP 2: Forward sub  L*y = B  → y in X ============
    li      s7, 0               # col
ldl_fc:
    bge     s7, s4, ldl_fcd
    li      s8, 0               # i
ldl_fi:
    bge     s8, s3, ldl_fid
    # X[i*m+col] = B[i*m+col]
    mul     t0, s8, s4
    add     t0, t0, s7
    slli    t0, t0, 3
    add     t1, s1, t0
    fld     ft5, 0(t1)          # B[i][col]
    mul     s9, s8, s3          # i*n
    li      s10, 0              # k
ldl_fk:
    bge     s10, s8, ldl_fkd
    add     t0, s9, s10         # i*n + k
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)          # L[i][k]
    mul     t0, s10, s4
    add     t0, t0, s7
    slli    t0, t0, 3
    add     t0, s2, t0
    fld     ft1, 0(t0)          # X[k][col]
    fnmsub.d ft5, ft0, ft1, ft5
    addi    s10, s10, 1
    j       ldl_fk
ldl_fkd:
    mul     t0, s8, s4
    add     t0, t0, s7
    slli    t0, t0, 3
    add     t0, s2, t0
    fsd     ft5, 0(t0)          # X[i][col] = y[i]
    addi    s8, s8, 1
    j       ldl_fi
ldl_fid:
    addi    s7, s7, 1
    j       ldl_fc
ldl_fcd:

    # ============ STEP 3: Diagonal solve  X[i][c] /= D[i] ============
    li      s7, 0               # i
ldl_di:
    bge     s7, s3, ldl_did
    slli    t0, s7, 3
    add     t0, s5, t0
    fld     ft5, 0(t0)          # D[i]
    li      s8, 0               # col
ldl_dc:
    bge     s8, s4, ldl_dcd
    mul     t0, s7, s4
    add     t0, t0, s8
    slli    t0, t0, 3
    add     t0, s2, t0
    fld     ft0, 0(t0)
    fdiv.d  ft0, ft0, ft5
    fsd     ft0, 0(t0)
    addi    s8, s8, 1
    j       ldl_dc
ldl_dcd:
    addi    s7, s7, 1
    j       ldl_di
ldl_did:

    # ============ STEP 4: Back sub  L^T*x = z  → x in X ============
    li      s7, 0               # col
ldl_bc:
    bge     s7, s4, ldl_bcd
    addi    s8, s3, -1          # i = n-1
ldl_bi:
    bltz    s8, ldl_bid
    mul     t0, s8, s4
    add     t0, t0, s7
    slli    t0, t0, 3
    add     t1, s2, t0
    fld     ft5, 0(t1)          # X[i][col]
    addi    s10, s8, 1          # k = i+1
ldl_bk:
    bge     s10, s3, ldl_bkd
    mul     t0, s10, s3
    add     t0, t0, s8          # k*n + i
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)          # L[k][i]
    mul     t0, s10, s4
    add     t0, t0, s7
    slli    t0, t0, 3
    add     t0, s2, t0
    fld     ft1, 0(t0)          # X[k][col]
    fnmsub.d ft5, ft0, ft1, ft5
    addi    s10, s10, 1
    j       ldl_bk
ldl_bkd:
    fsd     ft5, 0(t1)
    addi    s8, s8, -1
    j       ldl_bi
ldl_bid:
    addi    s7, s7, 1
    j       ldl_bc
ldl_bcd:

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
#                         MAIN
# ==============================================================
    .globl main
main:
    addi    sp, sp, -64
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)

    # ---- Read CSV ----
    la      a0, msg_reading
    call    printf
    la      a0, csv_path
    addi    a1, sp, 48
    addi    a2, sp, 52
    call    read_csv
    mv      s0, a0              # s0 = noisy data (flat)
    lw      s1, 48(sp)          # s1 = T
    lw      s2, 52(sp)          # s2 = cols
    la      a0, msg_ts
    mv      a1, s1
    mv      a2, s2
    call    printf

    # ---- Allocate pointer table ----
    li      a0, NUM_PTRS*8
    call    malloc
    mv      s3, a0

    la      a0, msg_building
    call    printf

    # ---- Allocate all matrices ----
    # F (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_F(s3)
    # Q (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Q(s3)
    # H (NM)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_H(s3)
    # Ht (NM)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Ht(s3)
    # R (MM)
    li      a0, MM*8
    call    malloc
    sd      a0, PT_R(s3)
    # x (N)
    li      a0, N*8
    call    malloc
    sd      a0, PT_x(s3)
    # P (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_P(s3)
    # Ft (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Ft(s3)
    # x_pred (N)
    li      a0, N*8
    call    malloc
    sd      a0, PT_xp(s3)
    # P_pred (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Pp(s3)
    # FP (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_FP(s3)
    # FPFt (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_FPFt(s3)
    # S (MM)
    li      a0, MM*8
    call    malloc
    sd      a0, PT_S(s3)
    # HP (MN)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_HP(s3)
    # HPHt (MM)
    li      a0, MM*8
    call    malloc
    sd      a0, PT_HPHt(s3)
    # PHt (NM)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_PHt(s3)
    # PHt_t (MN)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_PHtt(s3)
    # Kt_sol (MN)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Ktsol(s3)
    # K (NM)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_K(s3)
    # z (M)
    li      a0, M*8
    call    malloc
    sd      a0, PT_z(s3)
    # Hx (M)
    li      a0, M*8
    call    malloc
    sd      a0, PT_Hx(s3)
    # y (M)
    li      a0, M*8
    call    malloc
    sd      a0, PT_y(s3)
    # Ky (N)
    li      a0, N*8
    call    malloc
    sd      a0, PT_Ky(s3)
    # KH (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_KH(s3)
    # IKH (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKH(s3)
    # IKHt (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKHt(s3)
    # IKH_P (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKHP(s3)
    # IKH_P_It (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKHPIt(s3)
    # KR (NM)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_KR(s3)
    # KRKt (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_KRKt(s3)
    # Kt (MN)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Kt(s3)
    # I_mat (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Imat(s3)
    # LDL workspace: D (M doubles)
    li      a0, M*8
    call    malloc
    sd      a0, PT_D(s3)
    # LDL workspace: v (M doubles)
    li      a0, M*8
    call    malloc
    sd      a0, PT_v(s3)

    # ==== BUILD F ====
    ld      a0, PT_F(s3)
    li      a1, NN
    call    zero_mem

    ld      s4, PT_F(s3)

    # FP constants
    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0             # DT = 0.01
    fmul.d  ft1, ft0, ft0       # DT^2
    li      t0, 0x3FE0000000000000
    fmv.d.x ft8, t0             # 0.5
    fmul.d  ft2, ft8, ft1       # DT^2/2
    fmul.d  ft3, ft1, ft0       # DT^3
    li      t0, 0x3FC5555555555555
    fmv.d.x ft9, t0             # 1/6
    fmul.d  ft4, ft9, ft3       # DT^3/6
    li      t0, 0x3FF0000000000000
    fmv.d.x ft5, t0             # 1.0

    li      t0, 0               # jt = 0
bF_jt:
    li      t1, NUM_JOINTS
    bge     t0, t1, bF_done
    li      t2, 0               # axis = 0
bF_ax:
    li      t3, 3
    bge     t2, t3, bF_njt
    # base = jt*12 + axis*4
    li      t4, 12
    mul     t4, t0, t4
    slli    t5, t2, 2
    add     t4, t4, t5

    # Row 0: [b][b]=1, [b][b+1]=dt, [b][b+2]=dt2/2, [b][b+3]=dt3/6
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

    # Row 1
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

    # Row 2
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

    # Row 3
    addi    a5, t4, 3
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)

    addi    t2, t2, 1
    j       bF_ax
bF_njt:
    addi    t0, t0, 1
    j       bF_jt
bF_done:

    # ==== BUILD Q (matching C++ exactly: dt2..dt6 then q[i][j]*s2) ====
    ld      a0, PT_Q(s3)
    li      a1, NN
    call    zero_mem

    # Allocate q[4][4] = 16 doubles
    li      a0, 128
    call    malloc
    mv      s4, a0              # s4 = q table

    # Compute dt powers exactly like C++:
    # dt2=dt*dt, dt3=dt2*dt, dt4=dt3*dt, dt5=dt4*dt, dt6=dt5*dt
    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0             # dt = 0.01
    fmul.d  ft1, ft0, ft0       # dt2 = dt*dt
    fmul.d  ft2, ft1, ft0       # dt3 = dt2*dt
    fmul.d  ft3, ft2, ft0       # dt4 = dt3*dt
    fmul.d  ft4, ft3, ft0       # dt5 = dt4*dt
    fmul.d  ft5, ft4, ft0       # dt6 = dt5*dt

    # sigma_j^2 = 0.25
    li      t0, 0x3FD0000000000000
    fmv.d.x fs0, t0

    # Load division constants
    li      t0, 0x3FF0000000000000
    fmv.d.x ft9, t0             # 1.0
    li      t0, 0x4042000000000000
    fmv.d.x ft8, t0             # 36.0
    li      t0, 0x4028000000000000
    fmv.d.x ft7, t0             # 12.0
    li      t0, 0x4018000000000000
    fmv.d.x ft6, t0             # 6.0
    # Also need 4.0, 2.0
    # ft0=dt ft1=dt2 ft2=dt3 ft3=dt4 ft4=dt5 ft5=dt6
    # ft6=6.0 ft7=12.0 ft8=36.0 ft9=1.0

    # Build q[4][4] matching C++ exactly: s2 * {dt6/36, dt5/12, ...}
    # Row 0: {s2*dt6/36, s2*dt5/12, s2*dt4/6, s2*dt3/6}
    fdiv.d  fa0, ft5, ft8       # dt6/36
    fmul.d  fa0, fa0, fs0       # * s2
    fsd     fa0, 0(s4)          # q[0][0]

    fdiv.d  fa0, ft4, ft7       # dt5/12
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 8(s4)          # q[0][1]

    fdiv.d  fa0, ft3, ft6       # dt4/6
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 16(s4)         # q[0][2]

    fdiv.d  fa0, ft2, ft6       # dt3/6
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 24(s4)         # q[0][3]

    # Row 1: {s2*dt5/12, s2*dt4/4, s2*dt3/2, s2*dt2/2}
    fld     fa0, 8(s4)          # reuse q[0][1] for q[1][0] (symmetric)
    fsd     fa0, 32(s4)         # q[1][0]

    li      t0, 0x4010000000000000
    fmv.d.x fa1, t0             # 4.0
    fdiv.d  fa0, ft3, fa1       # dt4/4
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 40(s4)         # q[1][1]

    li      t0, 0x4000000000000000
    fmv.d.x fa1, t0             # 2.0
    fdiv.d  fa0, ft2, fa1       # dt3/2
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 48(s4)         # q[1][2]

    fdiv.d  fa0, ft1, fa1       # dt2/2
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 56(s4)         # q[1][3]

    # Row 2: {s2*dt4/6, s2*dt3/2, s2*dt2, s2*dt}
    fld     fa0, 16(s4)         # reuse q[0][2]
    fsd     fa0, 64(s4)         # q[2][0]

    fld     fa0, 48(s4)         # reuse q[1][2]
    fsd     fa0, 72(s4)         # q[2][1]

    fmul.d  fa0, ft1, fs0       # s2*dt2
    fsd     fa0, 80(s4)         # q[2][2]

    fmul.d  fa0, ft0, fs0       # s2*dt
    fsd     fa0, 88(s4)         # q[2][3]

    # Row 3: {s2*dt3/6, s2*dt2/2, s2*dt, s2*1.0}
    fld     fa0, 24(s4)         # reuse q[0][3]
    fsd     fa0, 96(s4)         # q[3][0]

    fld     fa0, 56(s4)         # reuse q[1][3]
    fsd     fa0, 104(s4)        # q[3][1]

    fld     fa0, 88(s4)         # reuse q[2][3]
    fsd     fa0, 112(s4)        # q[3][2]

    fmul.d  fa0, ft9, fs0       # s2*1.0
    fsd     fa0, 120(s4)        # q[3][3]

    # Now fill Q using the precomputed q[4][4] table
    ld      a5, PT_Q(s3)
    li      t0, 0               # jt
bQ_jt:
    li      t1, NUM_JOINTS
    bge     t0, t1, bQ_done
    li      t2, 0               # axis
bQ_ax:
    li      t3, 3
    bge     t2, t3, bQ_njt
    li      t4, 0               # i
bQ_i:
    li      t5, 4
    bge     t4, t5, bQ_nax
    li      t6, 0               # j
bQ_j:
    li      a1, 4
    bge     t6, a1, bQ_ni
    # Load q[i][j] from table
    slli    a2, t4, 2           # i*4
    add     a2, a2, t6          # i*4+j
    slli    a2, a2, 3           # byte offset
    add     a3, s4, a2
    fld     ft10, 0(a3)         # q[i][j] (already includes s2)

    # row = jt*12+axis*4+i,  col = jt*12+axis*4+j
    li      a1, 12
    mul     a2, t0, a1
    slli    a3, t2, 2
    add     a2, a2, a3
    add     a2, a2, t4          # row

    mul     a3, t0, a1
    slli    a4, t2, 2
    add     a3, a3, a4
    add     a3, a3, t6          # col

    li      a1, N
    mul     a4, a2, a1
    add     a4, a4, a3
    slli    a4, a4, 3
    add     a4, a5, a4
    fsd     ft10, 0(a4)

    addi    t6, t6, 1
    j       bQ_j
bQ_ni:
    addi    t4, t4, 1
    j       bQ_i
bQ_nax:
    addi    t2, t2, 1
    j       bQ_ax
bQ_njt:
    addi    t0, t0, 1
    j       bQ_jt
bQ_done:
    mv      a0, s4
    call    free

    # ==== BUILD H ====
    ld      a0, PT_H(s3)
    li      a1, NM
    call    zero_mem
    ld      s4, PT_H(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft5, t0
    li      t0, 0
bH_lp:
    li      t1, NUM_JOINTS
    bge     t0, t1, bH_done
    li      a1, N
    li      t2, 3
    mul     t3, t0, t2
    mul     t4, t3, a1
    li      t2, 12
    mul     t5, t0, t2
    # H[(j*3)*N + j*12] = 1
    add     t6, t4, t5
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)
    # H[(j*3+1)*N + j*12+4] = 1
    addi    t6, t3, 1
    mul     t6, t6, a1
    addi    a2, t5, 4
    add     t6, t6, a2
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)
    # H[(j*3+2)*N + j*12+8] = 1
    addi    t6, t3, 2
    mul     t6, t6, a1
    addi    a2, t5, 8
    add     t6, t6, a2
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)
    addi    t0, t0, 1
    j       bH_lp
bH_done:

    # ==== BUILD R (diagonal, sigma_r^2 = 0.3593^2 = 0.12909649) ====
    ld      a0, PT_R(s3)
    li      a1, MM
    call    zero_mem
    ld      s4, PT_R(s3)
    li      t0, 0x3FC0863BD94A08B8
    fmv.d.x ft0, t0             # sigma_r^2
    li      t1, 0
bR_lp:
    li      t2, M
    bge     t1, t2, bR_done
    li      a1, M
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       bR_lp
bR_done:

    # ==== BUILD I_mat (N×N identity) ====
    ld      a0, PT_Imat(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_Imat(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
bI_lp:
    li      t2, N
    bge     t1, t2, bI_done
    li      a1, N
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       bI_lp
bI_done:

    # ==== INIT x from first measurement ====
    ld      a0, PT_x(s3)
    li      a1, N
    call    zero_mem
    ld      s4, PT_x(s3)
    li      t0, 0
ix_lp:
    li      t1, NUM_JOINTS
    bge     t0, t1, ix_done
    li      a1, 3
    mul     a2, t0, a1
    slli    a2, a2, 3
    add     a3, s0, a2
    fld     ft0, 0(a3)
    fld     ft1, 8(a3)
    fld     ft2, 16(a3)
    li      a1, 12
    mul     a2, t0, a1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)          # px at offset 0
    fsd     ft1, 32(a3)         # py at offset 4*8=32
    fsd     ft2, 64(a3)         # pz at offset 8*8=64
    addi    t0, t0, 1
    j       ix_lp
ix_done:

    # ==== INIT P = I ====
    ld      a0, PT_P(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_P(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
iP_lp:
    li      t2, N
    bge     t1, t2, iP_done
    li      a1, N
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       iP_lp
iP_done:

    # ==== Compute Ht = H^T ====
    ld      a0, PT_H(s3)
    ld      a1, PT_Ht(s3)
    li      a2, M
    li      a3, N
    call    mat_transpose

    # ==== Compute Ft = F^T ====
    ld      a0, PT_F(s3)
    ld      a1, PT_Ft(s3)
    li      a2, N
    li      a3, N
    call    mat_transpose

    # ==== Open output ====
    la      a0, out_path
    call    open_output
    sd      a0, PT_FILE(s3)

    # ==== FILTER LOOP ====
    la      a0, msg_running
    call    printf

    li      s4, 0
filter_loop:
    bge     s4, s1, filter_done

    # -- PREDICTION: x_pred = F * x --
    ld      a0, PT_F(s3)
    ld      a1, PT_x(s3)
    ld      a2, PT_xp(s3)
    li      a3, N
    li      a4, N
    li      a5, 1
    call    mat_mul

    # -- FP = F * P --
    ld      a0, PT_F(s3)
    ld      a1, PT_P(s3)
    ld      a2, PT_FP(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    # -- FPFt = FP * Ft --
    ld      a0, PT_FP(s3)
    ld      a1, PT_Ft(s3)
    ld      a2, PT_FPFt(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    # -- P_pred = FPFt + Q --
    ld      a0, PT_FPFt(s3)
    ld      a1, PT_Q(s3)
    ld      a2, PT_Pp(s3)
    li      a3, NN
    call    mat_add

    # -- Load z from noisy[t] --
    mv      a0, s4
    mv      a1, s2
    mul     a2, a0, a1
    slli    a2, a2, 3
    add     a2, s0, a2
    ld      a3, PT_z(s3)
    li      t0, 0
lz_lp:
    li      t1, M
    bge     t0, t1, lz_done
    slli    t2, t0, 3
    add     t3, a2, t2
    add     t4, a3, t2
    fld     ft0, 0(t3)
    fsd     ft0, 0(t4)
    addi    t0, t0, 1
    j       lz_lp
lz_done:

    # -- PHt = P_pred * Ht --  (must come BEFORE S, matching C++ order)
    ld      a0, PT_Pp(s3)
    ld      a1, PT_Ht(s3)
    ld      a2, PT_PHt(s3)
    li      a3, N
    li      a4, N
    li      a5, M
    call    mat_mul

    # -- HPHt = H * PHt --   (H * (P*H^T), same grouping as C++)
    ld      a0, PT_H(s3)
    ld      a1, PT_PHt(s3)
    ld      a2, PT_HPHt(s3)
    li      a3, M
    li      a4, N
    li      a5, M
    call    mat_mul

    # -- S = HPHt + R --
    ld      a0, PT_HPHt(s3)
    ld      a1, PT_R(s3)
    ld      a2, PT_S(s3)
    li      a3, MM
    call    mat_add

    # -- PHt_t = PHt^T (M×N) --
    ld      a0, PT_PHt(s3)
    ld      a1, PT_PHtt(s3)
    li      a2, N
    li      a3, M
    call    mat_transpose

    # -- Solve S * Kt_sol = PHt_t  via LDL^T --
    ld      a0, PT_S(s3)
    ld      a1, PT_PHtt(s3)
    ld      a2, PT_Ktsol(s3)
    li      a3, M
    li      a4, N
    ld      a5, PT_D(s3)
    ld      a6, PT_v(s3)
    call    ldl_solve

    # -- K = Kt_sol^T (N×M) --
    ld      a0, PT_Ktsol(s3)
    ld      a1, PT_K(s3)
    li      a2, M
    li      a3, N
    call    mat_transpose

    # -- Hx = H * x_pred --
    ld      a0, PT_H(s3)
    ld      a1, PT_xp(s3)
    ld      a2, PT_Hx(s3)
    li      a3, M
    li      a4, N
    li      a5, 1
    call    mat_mul

    # -- y = z - Hx --
    ld      a0, PT_z(s3)
    ld      a1, PT_Hx(s3)
    ld      a2, PT_y(s3)
    li      a3, M
    call    mat_sub

    # -- Ky = K * y --
    ld      a0, PT_K(s3)
    ld      a1, PT_y(s3)
    ld      a2, PT_Ky(s3)
    li      a3, N
    li      a4, M
    li      a5, 1
    call    mat_mul

    # -- x = x_pred + Ky --
    ld      a0, PT_xp(s3)
    ld      a1, PT_Ky(s3)
    ld      a2, PT_x(s3)
    li      a3, N
    call    mat_add

    # -- KH = K * H --
    ld      a0, PT_K(s3)
    ld      a1, PT_H(s3)
    ld      a2, PT_KH(s3)
    li      a3, N
    li      a4, M
    li      a5, N
    call    mat_mul

    # -- IKH = I - KH --
    ld      a0, PT_Imat(s3)
    ld      a1, PT_KH(s3)
    ld      a2, PT_IKH(s3)
    li      a3, NN
    call    mat_sub

    # -- IKHt = IKH^T --
    ld      a0, PT_IKH(s3)
    ld      a1, PT_IKHt(s3)
    li      a2, N
    li      a3, N
    call    mat_transpose

    # -- IKH_P = IKH * P_pred --
    ld      a0, PT_IKH(s3)
    ld      a1, PT_Pp(s3)
    ld      a2, PT_IKHP(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    # -- IKH_P_It = IKH_P * IKHt --
    ld      a0, PT_IKHP(s3)
    ld      a1, PT_IKHt(s3)
    ld      a2, PT_IKHPIt(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    # -- KR = K * R --
    ld      a0, PT_K(s3)
    ld      a1, PT_R(s3)
    ld      a2, PT_KR(s3)
    li      a3, N
    li      a4, M
    li      a5, M
    call    mat_mul

    # -- Kt = K^T --
    ld      a0, PT_K(s3)
    ld      a1, PT_Kt(s3)
    li      a2, N
    li      a3, M
    call    mat_transpose

    # -- KRKt = KR * Kt --
    ld      a0, PT_KR(s3)
    ld      a1, PT_Kt(s3)
    ld      a2, PT_KRKt(s3)
    li      a3, N
    li      a4, M
    li      a5, N
    call    mat_mul

    # -- P = IKH_P_It + KRKt --
    ld      a0, PT_IKHPIt(s3)
    ld      a1, PT_KRKt(s3)
    ld      a2, PT_P(s3)
    li      a3, NN
    call    mat_add

    # -- Write row --
    ld      a0, PT_FILE(s3)
    ld      a1, PT_x(s3)
    li      a2, N
    call    write_row

    # -- Progress every 500 steps --
    li      t0, 500
    rem     t1, s4, t0
    bnez    t1, no_print
    la      a0, msg_progress
    mv      a1, s4
    mv      a2, s1
    call    printf
no_print:

    addi    s4, s4, 1
    j       filter_loop

filter_done:
    ld      a0, PT_FILE(s3)
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
    addi    sp, sp, 64
    ret
