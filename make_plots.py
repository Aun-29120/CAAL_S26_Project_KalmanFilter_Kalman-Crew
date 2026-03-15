import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

OUT = '/mnt/user-data/outputs/figures'
os.makedirs(OUT, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────
lkf   = pd.read_csv('/mnt/user-data/uploads/lkf_output__1_.csv')
ekf   = pd.read_csv('/mnt/user-data/uploads/ekf_output__1_.csv')
true_ = pd.read_csv('/mnt/user-data/uploads/3D_Full_Body_Humain_Gait_Walking_Dataset__True_Values_.csv')
noisy = pd.read_csv('/mnt/user-data/uploads/3D_Full_Body_Humain_Gait_Walking_Dataset__Noisy_Values_.csv')

JOINTS = ['pelvis','L5','L3','T12','T8','neck','head',
          'shoulderRight','upperArmRight','forearmRight','handRight',
          'shoulderLeft','upperArmLeft','forearmLeft','handLeft',
          'upperLegRight','lowerLegRight','footRight','toeRight',
          'upperLegLeft','lowerLegLeft','footLeft','toeLeft']

DT = 0.01
T  = len(lkf)
t  = np.arange(T) * DT

# ── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'       : 'DejaVu Serif',
    'font.size'         : 10,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.linewidth'    : 0.8,
    'axes.labelsize'    : 10,
    'xtick.major.width' : 0.7,
    'ytick.major.width' : 0.7,
    'xtick.labelsize'   : 8.5,
    'ytick.labelsize'   : 8.5,
    'lines.linewidth'   : 1.4,
    'legend.frameon'    : True,
    'legend.framealpha' : 0.85,
    'legend.edgecolor'  : '#cccccc',
    'legend.fontsize'   : 8.5,
    'figure.dpi'        : 180,
    'savefig.dpi'       : 180,
})

# Palette — high contrast, works when printed
C_TRUE  = '#145a32'   # dark green
C_NOISY = '#aab7b8'   # light grey
C_LKF   = '#154360'   # dark navy
C_EKF   = '#b03a2e'   # dark red

JOINT = 'pelvis'

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Position time-series: True / Noisy / LKF / EKF
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8.5, 7), sharex=True,
                         gridspec_kw={'hspace': 0.08})
fig.subplots_adjust(top=0.93, bottom=0.09, left=0.10, right=0.97)

ylabels = [r'$p_x$ (m)', r'$p_y$ (m)', r'$p_z$ (m)']
ax_keys = ['x', 'y', 'z']

for i, (ax_lbl, ylab) in enumerate(zip(ax_keys, ylabels)):
    ax = axes[i]
    ax.plot(t, noisy[f'{JOINT}_{ax_lbl}'],   color=C_NOISY, lw=0.6, alpha=0.85, label='Noisy',  zorder=1)
    ax.plot(t, true_[f'{JOINT}_{ax_lbl}'],   color=C_TRUE,  lw=1.8,             label='True',   zorder=4)
    ax.plot(t, lkf[f'{JOINT}_p{ax_lbl}'],    color=C_LKF,   lw=1.3,             label='LKF',    zorder=3)
    ax.plot(t, ekf[f'{JOINT}_p{ax_lbl}'],    color=C_EKF,   lw=1.1, ls='--',    label='EKF',    zorder=2)
    ax.set_ylabel(ylab)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, prune='both'))
    ax.tick_params(axis='x', labelbottom=(i==2))

axes[0].legend(ncol=4, loc='upper right', bbox_to_anchor=(1.0, 1.35),
               fontsize=8.5, handlelength=2.2)
axes[2].set_xlabel('Time (s)')
fig.suptitle('Figure 1 — Position Estimates: Pelvis Joint', fontsize=11, fontweight='bold', y=0.98)

fig.savefig(f'{OUT}/fig1_position.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig1_position.png', bbox_inches='tight')
plt.close()
print('✓ fig1_position')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — True vs Noisy vs Estimated (zoomed view, cleaner comparison)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8.5, 7), sharex=True,
                         gridspec_kw={'hspace': 0.08})
fig.subplots_adjust(top=0.93, bottom=0.09, left=0.10, right=0.97)

for i, (ax_lbl, ylab) in enumerate(zip(ax_keys, ylabels)):
    ax = axes[i]
    # Shade noisy region
    ax.fill_between(t, noisy[f'{JOINT}_{ax_lbl}'], true_[f'{JOINT}_{ax_lbl}'],
                    color=C_NOISY, alpha=0.25, label='Noise band')
    ax.plot(t, noisy[f'{JOINT}_{ax_lbl}'],  color=C_NOISY, lw=0.5, alpha=0.7)
    ax.plot(t, true_[f'{JOINT}_{ax_lbl}'],  color=C_TRUE,  lw=2.0,         label='True',  zorder=4)
    ax.plot(t, lkf[f'{JOINT}_p{ax_lbl}'],   color=C_LKF,   lw=1.4,         label='LKF',   zorder=3)
    ax.plot(t, ekf[f'{JOINT}_p{ax_lbl}'],   color=C_EKF,   lw=1.1, ls='--',label='EKF',   zorder=2)
    ax.set_ylabel(ylab)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, prune='both'))
    ax.tick_params(axis='x', labelbottom=(i==2))

axes[0].legend(ncol=4, loc='upper right', bbox_to_anchor=(1.0, 1.35),
               fontsize=8.5, handlelength=2.2)
axes[2].set_xlabel('Time (s)')
fig.suptitle('Figure 2 — True vs Noisy vs Estimated Position: Pelvis Joint',
             fontsize=11, fontweight='bold', y=0.98)

fig.savefig(f'{OUT}/fig2_true_noisy_estimated.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig2_true_noisy_estimated.png', bbox_inches='tight')
plt.close()
print('✓ fig2_true_noisy_estimated')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Velocity
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8.5, 7), sharex=True,
                         gridspec_kw={'hspace': 0.08})
fig.subplots_adjust(top=0.93, bottom=0.09, left=0.11, right=0.97)

vylabels = [r'$v_x$ (m/s)', r'$v_y$ (m/s)', r'$v_z$ (m/s)']
for i, (ax_lbl, ylab) in enumerate(zip(ax_keys, vylabels)):
    ax = axes[i]
    ax.axhline(0, color='#cccccc', lw=0.7, zorder=0)
    ax.plot(t, lkf[f'{JOINT}_v{ax_lbl}'], color=C_LKF, lw=1.4,         label='LKF')
    ax.plot(t, ekf[f'{JOINT}_v{ax_lbl}'], color=C_EKF, lw=1.1, ls='--',label='EKF')
    ax.set_ylabel(ylab)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune='both'))
    ax.tick_params(axis='x', labelbottom=(i==2))

axes[0].legend(ncol=2, loc='upper right', bbox_to_anchor=(1.0, 1.35),
               fontsize=8.5, handlelength=2.2)
axes[2].set_xlabel('Time (s)')
fig.suptitle('Figure 3 — Velocity Estimates: Pelvis Joint',
             fontsize=11, fontweight='bold', y=0.98)

fig.savefig(f'{OUT}/fig3_velocity.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig3_velocity.png', bbox_inches='tight')
plt.close()
print('✓ fig3_velocity')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Acceleration
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8.5, 7), sharex=True,
                         gridspec_kw={'hspace': 0.08})
fig.subplots_adjust(top=0.93, bottom=0.09, left=0.11, right=0.97)

aylabels = [r'$a_x$ (m/s²)', r'$a_y$ (m/s²)', r'$a_z$ (m/s²)']
for i, (ax_lbl, ylab) in enumerate(zip(ax_keys, aylabels)):
    ax = axes[i]
    ax.axhline(0, color='#cccccc', lw=0.7, zorder=0)
    ax.plot(t, lkf[f'{JOINT}_a{ax_lbl}'], color=C_LKF, lw=1.4,         label='LKF')
    ax.plot(t, ekf[f'{JOINT}_a{ax_lbl}'], color=C_EKF, lw=1.1, ls='--',label='EKF')
    ax.set_ylabel(ylab)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune='both'))
    ax.tick_params(axis='x', labelbottom=(i==2))

axes[0].legend(ncol=2, loc='upper right', bbox_to_anchor=(1.0, 1.35),
               fontsize=8.5, handlelength=2.2)
axes[2].set_xlabel('Time (s)')
fig.suptitle('Figure 4 — Acceleration Estimates: Pelvis Joint',
             fontsize=11, fontweight='bold', y=0.98)

fig.savefig(f'{OUT}/fig4_acceleration.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig4_acceleration.png', bbox_inches='tight')
plt.close()
print('✓ fig4_acceleration')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Jerk
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8.5, 7), sharex=True,
                         gridspec_kw={'hspace': 0.08})
fig.subplots_adjust(top=0.93, bottom=0.09, left=0.12, right=0.97)

jylabels = [r'$j_x$ (m/s³)', r'$j_y$ (m/s³)', r'$j_z$ (m/s³)']
for i, (ax_lbl, ylab) in enumerate(zip(ax_keys, jylabels)):
    ax = axes[i]
    ax.axhline(0, color='#cccccc', lw=0.7, zorder=0)
    ax.plot(t, lkf[f'{JOINT}_j{ax_lbl}'], color=C_LKF, lw=1.4,         label='LKF')
    ax.plot(t, ekf[f'{JOINT}_j{ax_lbl}'], color=C_EKF, lw=1.1, ls='--',label='EKF')
    ax.set_ylabel(ylab)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune='both'))
    ax.tick_params(axis='x', labelbottom=(i==2))

axes[0].legend(ncol=2, loc='upper right', bbox_to_anchor=(1.0, 1.35),
               fontsize=8.5, handlelength=2.2)
axes[2].set_xlabel('Time (s)')
fig.suptitle('Figure 5 — Jerk Estimates: Pelvis Joint',
             fontsize=11, fontweight='bold', y=0.98)

fig.savefig(f'{OUT}/fig5_jerk.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig5_jerk.png', bbox_inches='tight')
plt.close()
print('✓ fig5_jerk')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Per-joint RMSE bar chart
# ─────────────────────────────────────────────────────────────────────────────
lkf_rmse, ekf_rmse, noisy_rmse = [], [], []
for j in JOINTS:
    tx = true_[f'{j}_x'].values; ty = true_[f'{j}_y'].values; tz = true_[f'{j}_z'].values
    nx = noisy[f'{j}_x'].values; ny = noisy[f'{j}_y'].values; nz = noisy[f'{j}_z'].values
    lx = lkf[f'{j}_px'].values;  ly = lkf[f'{j}_py'].values;  lz = lkf[f'{j}_pz'].values
    ex = ekf[f'{j}_px'].values;  ey = ekf[f'{j}_py'].values;  ez = ekf[f'{j}_pz'].values
    noisy_rmse.append(np.sqrt(np.mean((nx-tx)**2+(ny-ty)**2+(nz-tz)**2))*1000)
    lkf_rmse.append  (np.sqrt(np.mean((lx-tx)**2+(ly-ty)**2+(lz-tz)**2))*1000)
    ekf_rmse.append  (np.sqrt(np.mean((ex-tx)**2+(ey-ty)**2+(ez-tz)**2))*1000)

print(f'  Global RMSE — Noisy: {np.mean(noisy_rmse):.1f} mm | '
      f'LKF: {np.mean(lkf_rmse):.1f} mm | EKF: {np.mean(ekf_rmse):.1f} mm')

# Short joint labels
SHORT = ['pelvis','L5','L3','T12','T8','neck','head',
         'shldr-R','uArm-R','fArm-R','hand-R',
         'shldr-L','uArm-L','fArm-L','hand-L',
         'uLeg-R','lLeg-R','foot-R','toe-R',
         'uLeg-L','lLeg-L','foot-L','toe-L']

x_pos = np.arange(len(JOINTS))
w = 0.27

fig, ax = plt.subplots(figsize=(13, 4.5))
fig.subplots_adjust(bottom=0.22, left=0.07, right=0.97, top=0.88)

b1 = ax.bar(x_pos - w,   noisy_rmse, w, color=C_NOISY, label=f'Noisy  (mean {np.mean(noisy_rmse):.0f} mm)', alpha=0.9, edgecolor='#888', linewidth=0.4)
b2 = ax.bar(x_pos,       lkf_rmse,   w, color=C_LKF,   label=f'LKF    (mean {np.mean(lkf_rmse):.0f} mm)',   edgecolor='#0a2340', linewidth=0.4)
b3 = ax.bar(x_pos + w,   ekf_rmse,   w, color=C_EKF,   label=f'EKF    (mean {np.mean(ekf_rmse):.0f} mm)',   edgecolor='#7b241c', linewidth=0.4, alpha=0.9)

ax.set_xticks(x_pos)
ax.set_xticklabels(SHORT, rotation=42, ha='right', fontsize=8)
ax.set_ylabel('Position RMSE (mm)', fontsize=10)
ax.set_title('Figure 6 — Per-Joint Position RMSE: Noisy vs LKF vs EKF',
             fontsize=11, fontweight='bold', pad=10)
ax.legend(fontsize=9, loc='upper right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(f'{OUT}/fig6_rmse_per_joint.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig6_rmse_per_joint.png', bbox_inches='tight')
plt.close()
print('✓ fig6_rmse_per_joint')

print(f'\nAll 6 figures saved to {OUT}')
