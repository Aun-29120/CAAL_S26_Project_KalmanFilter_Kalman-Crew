#!/usr/bin/env python3
"""
make_plots.py — Generate all plots for Milestone 3 report.

1. Position time-series (pelvis): True vs Noisy vs LKF vs EKF
2. Velocity, Acceleration, Jerk (pelvis): LKF vs EKF
3. Per-joint RMSE bar chart
4. MS2 vs MS3 comparison overlay (numerical verification visual)
"""

import csv
import sys
import os
import math

# Try importing matplotlib; if unavailable, print message
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib/numpy not available. Install with:")
    print("  pip3 install matplotlib numpy")
    sys.exit(1)

NUM_JOINTS = 23
STATE_PER_JOINT = 12
DT = 0.01

JOINT_NAMES = [
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
]


def load_state_csv(path):
    """Returns (T, 276) numpy array of states."""
    data = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([float(v) for v in row[1:]])
    return np.array(data)


def load_meas_csv(path):
    """Returns (T, 23, 3) numpy array of positions."""
    data = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            vals = [float(v) for v in row]
            frame = np.array(vals).reshape(NUM_JOINTS, 3)
            data.append(frame)
    return np.array(data)


def extract_joint(states, j, comp):
    """Extract component from joint j. comp: 0=px,1=vx,...,11=jz"""
    return states[:, j * STATE_PER_JOINT + comp]


def plot_position_timeseries(true_pos, noisy_pos, lkf, ekf, joint_idx, outdir):
    """Figure 1: Position x,y,z for one joint."""
    T = lkf.shape[0]
    t = np.arange(T) * DT
    labels = ['$p_x$', '$p_y$', '$p_z$']
    comps = [0, 4, 8]  # px, py, pz offsets

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Position Estimates: {JOINT_NAMES[joint_idx]}', fontsize=14)

    for i, (ax, label, c) in enumerate(zip(axes, labels, comps)):
        lkf_pos = extract_joint(lkf, joint_idx, c)
        ekf_pos = extract_joint(ekf, joint_idx, c)
        ax.plot(t, noisy_pos[:, joint_idx, i], color='0.75', lw=0.3, label='Noisy')
        ax.plot(t, true_pos[:, joint_idx, i], color='green', lw=1.5, label='True')
        ax.plot(t, lkf_pos, color='navy', lw=1.0, label='LKF')
        ax.plot(t, ekf_pos, color='red', lw=0.8, ls='--', label='EKF')
        ax.set_ylabel(f'{label} (m)')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig1_position.png'), dpi=150)
    plt.close()
    print("  Saved fig1_position.png")


def plot_derivatives(lkf, ekf, joint_idx, outdir):
    """Figures 3-5: Velocity, Acceleration, Jerk."""
    T = lkf.shape[0]
    t = np.arange(T) * DT

    for deriv_name, offsets, unit, fignum in [
        ('Velocity',     [1,5,9],   'm/s',    'fig3_velocity'),
        ('Acceleration', [2,6,10],  'm/s²',   'fig4_acceleration'),
        ('Jerk',         [3,7,11],  'm/s³',   'fig5_jerk'),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f'{deriv_name} Estimates: {JOINT_NAMES[joint_idx]}', fontsize=14)
        axis_labels = ['x', 'y', 'z']
        for i, (ax, off) in enumerate(zip(axes, offsets)):
            lkf_d = extract_joint(lkf, joint_idx, off)
            ekf_d = extract_joint(ekf, joint_idx, off)
            ax.plot(t, lkf_d, color='navy', lw=1.0, label='LKF')
            ax.plot(t, ekf_d, color='red', lw=0.8, ls='--', label='EKF')
            ax.set_ylabel(f'{axis_labels[i]} ({unit})')
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{fignum}.png'), dpi=150)
        plt.close()
        print(f"  Saved {fignum}.png")


def plot_rmse_bar(true_pos, noisy_pos, lkf, ekf, outdir):
    """Figure 6: Per-joint RMSE bar chart."""
    T = lkf.shape[0]
    rmse_noisy = np.zeros(NUM_JOINTS)
    rmse_lkf = np.zeros(NUM_JOINTS)
    rmse_ekf = np.zeros(NUM_JOINTS)

    for j in range(NUM_JOINTS):
        for ax_i, comp in enumerate([0, 4, 8]):
            diff_n = noisy_pos[:, j, ax_i] - true_pos[:, j, ax_i]
            diff_l = extract_joint(lkf, j, comp) - true_pos[:, j, ax_i]
            diff_e = extract_joint(ekf, j, comp) - true_pos[:, j, ax_i]
            rmse_noisy[j] += np.mean(diff_n**2)
            rmse_lkf[j]   += np.mean(diff_l**2)
            rmse_ekf[j]   += np.mean(diff_e**2)
        rmse_noisy[j] = np.sqrt(rmse_noisy[j] / 3) * 1000  # mm
        rmse_lkf[j]   = np.sqrt(rmse_lkf[j]   / 3) * 1000
        rmse_ekf[j]   = np.sqrt(rmse_ekf[j]   / 3) * 1000

    x = np.arange(NUM_JOINTS)
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, rmse_noisy, w, label=f'Noisy (mean {np.mean(rmse_noisy):.0f} mm)', color='0.7')
    ax.bar(x,     rmse_lkf,   w, label=f'LKF (mean {np.mean(rmse_lkf):.0f} mm)', color='navy')
    ax.bar(x + w, rmse_ekf,   w, label=f'EKF (mean {np.mean(rmse_ekf):.0f} mm)', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Position RMSE (mm)')
    ax.set_title('Per-Joint Position RMSE: Noisy vs LKF vs EKF')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig6_rmse_bar.png'), dpi=150)
    plt.close()
    print("  Saved fig6_rmse_bar.png")


def plot_ms2_vs_ms3(ms2_states, ms3_states, label, joint_idx, outdir):
    """Comparison overlay: MS2 (C++) vs MS3 (ASM) for visual verification."""
    T = ms2_states.shape[0]
    t = np.arange(T) * DT

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f'MS2 (C++) vs MS3 (ASM) — {label} — {JOINT_NAMES[joint_idx]}', fontsize=13)
    for i, (ax, comp, name) in enumerate(zip(axes, [0,4,8], ['$p_x$','$p_y$','$p_z$'])):
        ms2_v = extract_joint(ms2_states, joint_idx, comp)
        ms3_v = extract_joint(ms3_states, joint_idx, comp)
        ax.plot(t, ms2_v, color='navy', lw=1.5, label='MS2 (C++)')
        ax.plot(t, ms3_v, color='red', lw=0.8, ls='--', label='MS3 (ASM)')
        ax.set_ylabel(f'{name} (m)')
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fname = f'fig_compare_{label.lower()}.png'
    plt.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 make_plots.py <true.csv> <noisy.csv> "
              "<lkf_asm.csv> <ekf_asm.csv> [lkf_ms2.csv] [ekf_ms2.csv]")
        sys.exit(1)

    true_path  = sys.argv[1]
    noisy_path = sys.argv[2]
    lkf_path   = sys.argv[3]
    ekf_path   = sys.argv[4]
    lkf_ms2    = sys.argv[5] if len(sys.argv) > 5 else None
    ekf_ms2    = sys.argv[6] if len(sys.argv) > 6 else None

    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)

    print("Loading data...")
    true_pos  = load_meas_csv(true_path)
    noisy_pos = load_meas_csv(noisy_path)
    lkf = load_state_csv(lkf_path)
    ekf = load_state_csv(ekf_path)

    joint = 0  # pelvis

    print("Generating plots...")
    plot_position_timeseries(true_pos, noisy_pos, lkf, ekf, joint, outdir)
    plot_derivatives(lkf, ekf, joint, outdir)
    plot_rmse_bar(true_pos, noisy_pos, lkf, ekf, outdir)

    # MS2 vs MS3 comparison
    if lkf_ms2 and os.path.exists(lkf_ms2):
        lkf_ms2_data = load_state_csv(lkf_ms2)
        plot_ms2_vs_ms3(lkf_ms2_data, lkf, 'LKF', joint, outdir)
    if ekf_ms2 and os.path.exists(ekf_ms2):
        ekf_ms2_data = load_state_csv(ekf_ms2)
        plot_ms2_vs_ms3(ekf_ms2_data, ekf, 'EKF', joint, outdir)

    print("Done!")


if __name__ == "__main__":
    main()
