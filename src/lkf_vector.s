# =============================================================
#  lkf_vector.s — RISC-V VECTOR (RVV 1.0) assembly for LKF
#  Milestone 4: Vectorised Linear Kalman Filter
#
#  Built directly on top of MS3 scalar lkf_asm.s. All algorithmic
#  structure (pointer table, F/Q/H/R construction, filter loop,
#  Joseph-form covariance update, LDL^T solver) is identical to MS3.
#  The math kernels mat_add, mat_sub, mat_transpose, mat_mul,
#  zero_mem and large parts of ldl_solve are replaced with strip-mined
#  RVV kernels using e64,m4 vtype.
#
#  Numerical guarantee: per-lane operation order in every kernel
#  matches the MS3 scalar code exactly, so vector output is bit-
#  identical to MS3 scalar within the LKF (which itself is bit-
#  identical to the MS2 C++ reference).
#
#  Register plan for main (unchanged from MS3):
#    s0 = noisy data pointer (flat array)
#    s1 = T (number of timesteps)
#    s2 = csv_cols (69)
#    s3 = ptbl (pointer table — all matrix pointers)
#    s4 = t (loop counter / temp)
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
out_path:       .string "LKF_vector_output.csv"
msg_reading:    .string "[LKF-VEC] Reading dataset...\n"
msg_ts:         .string "[LKF-VEC] Timesteps: %d, Cols: %d\n"
msg_building:   .string "[LKF-VEC] Building matrices...\n"
msg_running:    .string "[LKF-VEC] Running filter (RVV 1.0)...\n"
msg_progress:   .string "[LKF-VEC] t=%d/%d\n"
msg_done:       .string "[LKF-VEC] Done. Output saved to LKF_vector_output.csv\n"

    .section .text

# ==============================================================
#         VECTORISED UTILITY: zero_mem(ptr=a0, count=a1)
#   Writes count doubles of 0.0 to ptr.
#   Uses vmv.v.i with imm=0 (e64 zero is bit-identical to +0.0).
# ==============================================================
    .globl zero_mem
zero_mem:
zmv_lp:
    beqz    a1, zmv_done
    vsetvli t0, a1, e64, m4, ta, ma
    vmv.v.i v0, 0
    vse64.v v0, (a0)
    slli    t1, t0, 3
    add     a0, a0, t1
    sub     a1, a1, t0
    j       zmv_lp
zmv_done:
    ret

# ==============================================================
#                    MATH FUNCTIONS (vectorised)
# ==============================================================

# --- mat_add(A=a0, B=a1, C=a2, size=a3) : C = A + B ---
#   Strip-mined elementwise vfadd.vv. Bit-identical to scalar
#   because each lane performs the same single fadd as the scalar
#   loop on the same operand pair.
    .globl mat_add
mat_add:
mav_lp:
    beqz    a3, mav_done
    vsetvli t0, a3, e64, m4, ta, ma
    vle64.v v0, (a0)
    vle64.v v4, (a1)
    vfadd.vv v8, v0, v4
    vse64.v v8, (a2)
    slli    t1, t0, 3
    add     a0, a0, t1
    add     a1, a1, t1
    add     a2, a2, t1
    sub     a3, a3, t0
    j       mav_lp
mav_done:
    ret

# --- mat_sub(A=a0, B=a1, C=a2, size=a3) : C = A - B ---
    .globl mat_sub
mat_sub:
msv_lp:
    beqz    a3, msv_done
    vsetvli t0, a3, e64, m4, ta, ma
    vle64.v v0, (a0)
    vle64.v v4, (a1)
    vfsub.vv v8, v0, v4
    vse64.v v8, (a2)
    slli    t1, t0, 3
    add     a0, a0, t1
    add     a1, a1, t1
    add     a2, a2, t1
    sub     a3, a3, t0
    j       msv_lp
msv_done:
    ret

# --- mat_transpose(A=a0 [rows*cols], B=a1 [cols*rows], rows=a2, cols=a3) ---
#   Outer loop over rows of A. Per-row: contiguous vle64.v from
#   A[i][j..], strided vsse64.v into B[j..][i] with byte stride rows*8.
#   Pure data movement: bit-identical.
    .globl mat_transpose
mat_transpose:
    li      t0, 0                   # i = 0
mtv_io:
    bge     t0, a2, mtv_id
    # src ptr = &A[i*cols] = a0 + i*cols*8
    mul     t1, t0, a3
    slli    t1, t1, 3
    add     t2, a0, t1              # t2 = src
    # dst start = &B[0*rows + i] = a1 + i*8 ; stride = rows*8
    slli    t1, t0, 3
    add     t3, a1, t1              # t3 = dst
    slli    t4, a2, 3               # t4 = stride bytes
    mv      t5, a3                  # t5 = j_remaining
mtv_jl:
    beqz    t5, mtv_jd
    vsetvli t6, t5, e64, m4, ta, ma
    vle64.v v0, (t2)                # contiguous load from row i
    vsse64.v v0, (t3), t4           # strided store to column i of B
    slli    a4, t6, 3
    add     t2, t2, a4              # advance src by VL doubles
    mul     a4, t6, t4
    add     t3, t3, a4              # advance dst by VL strides
    sub     t5, t5, t6
    j       mtv_jl
mtv_jd:
    addi    t0, t0, 1
    j       mtv_io
mtv_id:
    ret

# --- mat_mul(A=a0, B=a1, C=a2, rowsA=a3, colsA=a4, colsB=a5) : C = A*B ---
#   i-k-j ordering with vector rank-1 update on the inner j loop.
#   Per element C[i][j], the FMA chain is identical to scalar:
#       C[i][j] = sum_k A[i][k] * B[k][j]
#   evaluated left-to-right via vfmacc.vf which broadcasts the scalar
#   A[i][k] across the vector chunk and fuses with B[k][j..].
#   Zero-skip optimization preserved for sparse F and H.
#   Bit-identical to scalar mat_mul.
#
#   Register allocation (callee-saved s0..s5):
#     s0=A  s1=B  s2=C  s3=rowsA  s4=colsA  s5=colsB
#   Inner: t0=i, t1=k, t4=&C[i][0], t5=&B[k][0], t6=j_remaining
#          ft0=A[i][k]  v0=C[i][j..] (m4)  v4=B[k][j..] (m4)
    .globl mat_mul
mat_mul:
    addi    sp, sp, -48
    sd      s0,  0(sp)
    sd      s1,  8(sp)
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

    # ---- Zero C: rowsA*colsB doubles (vectorised) ----
    mul     a3, s3, s5
    mv      a4, s2
mmv_zlp:
    beqz    a3, mmv_zd
    vsetvli t1, a3, e64, m4, ta, ma
    vmv.v.i v0, 0
    vse64.v v0, (a4)
    slli    t2, t1, 3
    add     a4, a4, t2
    sub     a3, a3, t1
    j       mmv_zlp
mmv_zd:

    # ---- i loop ----
    li      t0, 0
mmv_i:
    bge     t0, s3, mmv_d
    # ---- k loop ----
    li      t1, 0
mmv_k:
    bge     t1, s4, mmv_ni
    # Load scalar A[i*colsA + k] into ft0
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s0, t2
    fld     ft0, 0(t3)
    # Sparse-skip: if A[i][k] == 0.0, no contribution to row i.
    fmv.d.x ft7, zero
    feq.d   a6, ft0, ft7
    bnez    a6, mmv_nk
    # &C[i*colsB] base
    mul     t4, t0, s5
    slli    t4, t4, 3
    add     t4, s2, t4
    # &B[k*colsB] base
    mul     t5, t1, s5
    slli    t5, t5, 3
    add     t5, s1, t5
    # j_remaining = colsB
    mv      t6, s5
mmv_j:
    beqz    t6, mmv_nk
    vsetvli a7, t6, e64, m4, ta, ma
    vle64.v v0, (t4)                # C[i][j..]
    vle64.v v4, (t5)                # B[k][j..]
    vfmacc.vf v0, ft0, v4           # v0 += A[i][k] * v4
    vse64.v v0, (t4)
    slli    a3, a7, 3
    add     t4, t4, a3
    add     t5, t5, a3
    sub     t6, t6, a7
    j       mmv_j
mmv_nk:
    addi    t1, t1, 1
    j       mmv_k
mmv_ni:
    addi    t0, t0, 1
    j       mmv_i
mmv_d:
    ld      s0,  0(sp)
    ld      s1,  8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    addi    sp, sp, 48
    ret

# ==============================================================
#  ldl_solve(S=a0, B=a1, X=a2, n=a3, m=a4, D=a5, v=a6)
#
#  Solve S*X = B via LDL^T decomposition. S is n×n SPD (modified
#  in-place; lower triangle becomes L). B and X are n×m, row-major.
#
#  In LKF: n=M=69, m=N=276. The m direction (276 cols) is where
#  vectorisation pays off; the n direction (69) stays scalar where
#  there are tight serial dependencies.
#
#  Vectorisation strategy:
#    Step 1 (factorisation): scalar (preserved verbatim from MS3)
#    Step 2 (forward sub):   vectorised across columns
#    Step 3 (diag solve):    vectorised across columns
#    Step 4 (back sub):      vectorised across columns
#
#  Per-lane operation order in steps 2/3/4 matches the scalar
#  inner loop exactly, so the result is bit-identical.
#
#  Register map:
#    s0=S  s1=B  s2=X  s3=n  s4=m  s5=D  s6=v
#    s7..s11 = loop indices and offsets
#    fs1 = 1e-14 floor constant (Step 1)
# ==============================================================
    .globl ldl_solve
ldl_solve:
    addi    sp, sp, -112
    sd      ra,   0(sp)
    sd      s0,   8(sp)
    sd      s1,  16(sp)
    sd      s2,  24(sp)
    sd      s3,  32(sp)
    sd      s4,  40(sp)
    sd      s5,  48(sp)
    sd      s6,  56(sp)
    sd      s7,  64(sp)
    sd      s8,  72(sp)
    sd      s9,  80(sp)
    sd      s10, 88(sp)
    sd      s11, 96(sp)
    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4
    mv      s5, a5
    mv      s6, a6

    # 1e-14 numerical floor constant
    li      t0, 0x3D06849B86A12B9B
    fmv.d.x fs1, t0

    # ============ STEP 1: LDL^T Factorisation (SCALAR) ============
    # For j = 0..n-1:
    #   v[i] = S[j*n+i] * D[i]    for i = 0..j-1
    #   D[j] = S[j][j] - sum L[j][i]*v[i]
    #   if |D[j]| < 1e-14: D[j] = 1e-14
    #   For k = j+1..n-1:
    #     L[k][j] = (S[k][j] - sum L[k][i]*v[i]) / D[j]
    li      s7, 0
ldl_j:
    bge     s7, s3, ldl_fdone

    # --- v[i] = S[j*n+i] * D[i] for i = 0..j-1 ---
    li      s8, 0
    mul     s9, s7, s3
ldl_v:
    bge     s8, s7, ldl_vd
    add     t0, s9, s8
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)
    slli    t1, s8, 3
    add     t1, s5, t1
    fld     ft1, 0(t1)
    fmul.d  ft2, ft0, ft1
    slli    t1, s8, 3
    add     t1, s6, t1
    fsd     ft2, 0(t1)
    addi    s8, s8, 1
    j       ldl_v
ldl_vd:

    # --- D[j] = S[j][j] - sum L[j][i]*v[i] ---
    add     t0, s9, s7
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft5, 0(t0)
    li      s8, 0
ldl_ds:
    bge     s8, s7, ldl_dsd
    add     t0, s9, s8
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)
    slli    t1, s8, 3
    add     t1, s6, t1
    fld     ft1, 0(t1)
    fnmsub.d ft5, ft0, ft1, ft5
    addi    s8, s8, 1
    j       ldl_ds
ldl_dsd:

    fabs.d  ft6, ft5
    flt.d   t0, ft6, fs1
    beqz    t0, ldl_nf
    fmv.d   ft5, fs1
ldl_nf:
    slli    t0, s7, 3
    add     t0, s5, t0
    fsd     ft5, 0(t0)              # store D[j]

    # --- L[k][j] for k = j+1..n-1 ---
    addi    s10, s7, 1
ldl_lk:
    bge     s10, s3, ldl_lkd
    mul     s11, s10, s3
    add     t0, s11, s7
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft6, 0(t0)
    li      s8, 0
ldl_lks:
    bge     s8, s7, ldl_lksd
    add     t0, s11, s8
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)
    slli    t1, s8, 3
    add     t1, s6, t1
    fld     ft1, 0(t1)
    fnmsub.d ft6, ft0, ft1, ft6
    addi    s8, s8, 1
    j       ldl_lks
ldl_lksd:
    fdiv.d  ft6, ft6, ft5
    mul     t0, s10, s3
    add     t0, t0, s7
    slli    t0, t0, 3
    add     t0, s0, t0
    fsd     ft6, 0(t0)
    addi    s10, s10, 1
    j       ldl_lk
ldl_lkd:

    addi    s7, s7, 1
    j       ldl_j
ldl_fdone:

    # ============ STEP 2: Forward sub  L*y = B  → y in X ============
    # Vectorised across cols.
    # For i = 0..n-1:
    #   For col_base = 0..m step VL:
    #     accum_vec = B[i][col_base..col_base+VL-1]   ; vle64 from &B[i*m+col_base]
    #     For k = 0..i-1:
    #       L_ik   = scalar L[i][k] = S[i*n+k]
    #       X_kvec = X[k][col_base..]                 ; vle64 from &X[k*m+col_base]
    #       accum_vec -= L_ik * X_kvec                ; vfnmsac.vf
    #     X[i][col_base..] = accum_vec                ; vse64
    li      s7, 0                   # i = 0
ldl_fi:
    bge     s7, s3, ldl_fid
    li      s8, 0                   # col_base = 0
    mul     s9, s7, s3              # i*n   (for L[i][k] = S[i*n+k])
ldl_fc:
    bge     s8, s4, ldl_fcd
    sub     a4, s4, s8              # col_remaining = m - col_base
    vsetvli a5, a4, e64, m4, ta, ma
    # Load initial accum_vec from &B[i*m + col_base]
    mul     a3, s7, s4
    add     a3, a3, s8
    slli    a3, a3, 3
    add     t6, s1, a3
    vle64.v v0, (t6)
    # Inner k loop (k = 0..i-1)
    li      s10, 0
ldl_fk:
    bge     s10, s7, ldl_fkd
    # scalar L[i][k] = S[i*n + k]
    add     t0, s9, s10
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)
    # vector X[k][col_base..] from &X[k*m + col_base]
    mul     a3, s10, s4
    add     a3, a3, s8
    slli    a3, a3, 3
    add     t0, s2, a3
    vle64.v v4, (t0)
    # accum_vec -= L[i][k] * X[k][col_base..]
    vfnmsac.vf v0, ft0, v4
    addi    s10, s10, 1
    j       ldl_fk
ldl_fkd:
    # Store X[i][col_base..] = accum_vec
    mul     a3, s7, s4
    add     a3, a3, s8
    slli    a3, a3, 3
    add     t0, s2, a3
    vse64.v v0, (t0)
    add     s8, s8, a5
    j       ldl_fc
ldl_fcd:
    addi    s7, s7, 1
    j       ldl_fi
ldl_fid:

    # ============ STEP 3: Diagonal solve  X[i][col] /= D[i] ============
    # For each row i, broadcast scalar D[i] and do strip-mined
    # vfdiv.vf across the m-element row of X.
    li      s7, 0                   # i = 0
ldl_di:
    bge     s7, s3, ldl_did
    slli    t0, s7, 3
    add     t0, s5, t0
    fld     ft8, 0(t0)              # D[i]
    # &X[i*m]
    mul     t0, s7, s4
    slli    t0, t0, 3
    add     t1, s2, t0
    mv      t2, s4                  # remaining = m
ldl_dv:
    beqz    t2, ldl_dvd
    vsetvli t3, t2, e64, m4, ta, ma
    vle64.v v0, (t1)
    vfdiv.vf v0, v0, ft8
    vse64.v v0, (t1)
    slli    t4, t3, 3
    add     t1, t1, t4
    sub     t2, t2, t3
    j       ldl_dv
ldl_dvd:
    addi    s7, s7, 1
    j       ldl_di
ldl_did:

    # ============ STEP 4: Back sub  L^T*x = z  → x in X ============
    # Vectorised across cols. For i = n-1 downto 0:
    #   For col_base = 0..m step VL:
    #     accum_vec = X[i][col_base..]   (currently holds z[i] from Step 3)
    #     For k = i+1..n-1:
    #       L_ki   = scalar L[k][i] = S[k*n+i]
    #       X_kvec = X[k][col_base..]
    #       accum_vec -= L_ki * X_kvec
    #     X[i][col_base..] = accum_vec
    addi    s7, s3, -1              # i = n - 1
ldl_bi:
    bltz    s7, ldl_bid
    li      s8, 0                   # col_base = 0
ldl_bc:
    bge     s8, s4, ldl_bcd
    sub     a4, s4, s8
    vsetvli a5, a4, e64, m4, ta, ma
    # Load accum_vec from &X[i*m + col_base]
    mul     a3, s7, s4
    add     a3, a3, s8
    slli    a3, a3, 3
    add     t6, s2, a3              # &X[i][col_base]
    vle64.v v0, (t6)
    # Inner k loop (k = i+1..n-1)
    addi    s10, s7, 1
ldl_bk:
    bge     s10, s3, ldl_bkd
    # scalar L[k][i] = S[k*n + i]
    mul     t0, s10, s3
    add     t0, t0, s7
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)
    # vector X[k][col_base..] from &X[k*m + col_base]
    mul     a3, s10, s4
    add     a3, a3, s8
    slli    a3, a3, 3
    add     t0, s2, a3
    vle64.v v4, (t0)
    vfnmsac.vf v0, ft0, v4
    addi    s10, s10, 1
    j       ldl_bk
ldl_bkd:
    # Store X[i][col_base..]
    vse64.v v0, (t6)
    add     s8, s8, a5
    j       ldl_bc
ldl_bcd:
    addi    s7, s7, -1
    j       ldl_bi
ldl_bid:

    ld      ra,   0(sp)
    ld      s0,   8(sp)
    ld      s1,  16(sp)
    ld      s2,  24(sp)
    ld      s3,  32(sp)
    ld      s4,  40(sp)
    ld      s5,  48(sp)
    ld      s6,  56(sp)
    ld      s7,  64(sp)
    ld      s8,  72(sp)
    ld      s9,  80(sp)
    ld      s10, 88(sp)
    ld      s11, 96(sp)
    addi    sp, sp, 112
    ret


# ==============================================================
#                            MAIN
#   Identical structure to MS3 scalar lkf_asm.s — the vectorised
#   kernels are drop-in replacements with the same ABI.
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
    mv      s0, a0
    lw      s1, 48(sp)
    lw      s2, 52(sp)
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
    li      a0, NN*8
    call    malloc
    sd      a0, PT_F(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Q(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_H(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Ht(s3)
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
    li      a0, NM*8
    call    malloc
    sd      a0, PT_HP(s3)
    li      a0, MM*8
    call    malloc
    sd      a0, PT_HPHt(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_PHt(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_PHtt(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Ktsol(s3)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_K(s3)
    li      a0, M*8
    call    malloc
    sd      a0, PT_z(s3)
    li      a0, M*8
    call    malloc
    sd      a0, PT_Hx(s3)
    li      a0, M*8
    call    malloc
    sd      a0, PT_y(s3)
    li      a0, N*8
    call    malloc
    sd      a0, PT_Ky(s3)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_KH(s3)
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
    li      a0, M*8
    call    malloc
    sd      a0, PT_D(s3)
    li      a0, M*8
    call    malloc
    sd      a0, PT_v(s3)

    # ==== BUILD F (constant-jerk 4×4 blocks, 23 joints × 3 axes) ====
    ld      a0, PT_F(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_F(s3)

    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0                 # DT = 0.01
    fmul.d  ft1, ft0, ft0           # DT^2
    li      t0, 0x3FE0000000000000
    fmv.d.x ft8, t0                 # 0.5
    fmul.d  ft2, ft8, ft1           # DT^2/2
    fmul.d  ft3, ft1, ft0           # DT^3
    li      t0, 0x3FC5555555555555
    fmv.d.x ft9, t0                 # 1/6
    fmul.d  ft4, ft9, ft3           # DT^3/6
    li      t0, 0x3FF0000000000000
    fmv.d.x ft5, t0                 # 1.0

    li      t0, 0
bF_jt:
    li      t1, NUM_JOINTS
    bge     t0, t1, bF_done
    li      t2, 0
bF_ax:
    li      t3, 3
    bge     t2, t3, bF_njt
    li      t4, 12
    mul     t4, t0, t4
    slli    t5, t2, 2
    add     t4, t4, t5

    # Row 0 of 4x4 block
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

    # ==== BUILD Q (matching MS2 C++ arithmetic order) ====
    ld      a0, PT_Q(s3)
    li      a1, NN
    call    zero_mem

    li      a0, 128
    call    malloc
    mv      s4, a0                  # q[4][4] table

    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0                 # dt
    fmul.d  ft1, ft0, ft0           # dt2
    fmul.d  ft2, ft1, ft0           # dt3
    fmul.d  ft3, ft2, ft0           # dt4
    fmul.d  ft4, ft3, ft0           # dt5
    fmul.d  ft5, ft4, ft0           # dt6

    li      t0, 0x3FD0000000000000
    fmv.d.x fs0, t0                 # sigma_j^2 = 0.25

    li      t0, 0x3FF0000000000000
    fmv.d.x ft9, t0                 # 1.0
    li      t0, 0x4042000000000000
    fmv.d.x ft8, t0                 # 36.0
    li      t0, 0x4028000000000000
    fmv.d.x ft7, t0                 # 12.0
    li      t0, 0x4018000000000000
    fmv.d.x ft6, t0                 # 6.0

    # Row 0
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

    # Row 1
    fld     fa0, 8(s4)
    fsd     fa0, 32(s4)
    li      t0, 0x4010000000000000
    fmv.d.x fa1, t0                 # 4.0
    fdiv.d  fa0, ft3, fa1
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 40(s4)
    li      t0, 0x4000000000000000
    fmv.d.x fa1, t0                 # 2.0
    fdiv.d  fa0, ft2, fa1
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 48(s4)
    fdiv.d  fa0, ft1, fa1
    fmul.d  fa0, fa0, fs0
    fsd     fa0, 56(s4)

    # Row 2
    fld     fa0, 16(s4)
    fsd     fa0, 64(s4)
    fld     fa0, 48(s4)
    fsd     fa0, 72(s4)
    fmul.d  fa0, ft1, fs0
    fsd     fa0, 80(s4)
    fmul.d  fa0, ft0, fs0
    fsd     fa0, 88(s4)

    # Row 3
    fld     fa0, 24(s4)
    fsd     fa0, 96(s4)
    fld     fa0, 56(s4)
    fsd     fa0, 104(s4)
    fld     fa0, 88(s4)
    fsd     fa0, 112(s4)
    fmul.d  fa0, ft9, fs0
    fsd     fa0, 120(s4)

    # Fill Q from q[4][4] table
    ld      a5, PT_Q(s3)
    li      t0, 0
bQ_jt:
    li      t1, NUM_JOINTS
    bge     t0, t1, bQ_done
    li      t2, 0
bQ_ax:
    li      t3, 3
    bge     t2, t3, bQ_njt
    li      t4, 0
bQ_i:
    li      t5, 4
    bge     t4, t5, bQ_nax
    li      t6, 0
bQ_j:
    li      a1, 4
    bge     t6, a1, bQ_ni
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
    add     t6, t4, t5
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)
    addi    t6, t3, 1
    mul     t6, t6, a1
    addi    a2, t5, 4
    add     t6, t6, a2
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)
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

    # ==== BUILD R (diagonal, sigma_r^2 = 0.12909649) ====
    ld      a0, PT_R(s3)
    li      a1, MM
    call    zero_mem
    ld      s4, PT_R(s3)
    li      t0, 0x3FC0863BD94A08B8
    fmv.d.x ft0, t0
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
    fsd     ft0, 0(a3)
    fsd     ft1, 32(a3)
    fsd     ft2, 64(a3)
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

    # -- PHt = P_pred * Ht --
    ld      a0, PT_Pp(s3)
    ld      a1, PT_Ht(s3)
    ld      a2, PT_PHt(s3)
    li      a3, N
    li      a4, N
    li      a5, M
    call    mat_mul

    # -- HPHt = H * PHt --
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

    # -- PHt_t = PHt^T --
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

    # -- K = Kt_sol^T --
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
