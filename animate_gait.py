import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa

OUT_DIR = '/mnt/user-data/outputs/animations_v2'
os.makedirs(OUT_DIR, exist_ok=True)

JOINTS = [
    'pelvis','L5','L3','T12','T8','neck','head',
    'shoulderRight','upperArmRight','forearmRight','handRight',
    'shoulderLeft','upperArmLeft','forearmLeft','handLeft',
    'upperLegRight','lowerLegRight','footRight','toeRight',
    'upperLegLeft','lowerLegLeft','footLeft','toeLeft'
]
JI = {j: i for i, j in enumerate(JOINTS)}

BONES = [
    ('pelvis','L5'), ('L5','L3'), ('L3','T12'), ('T12','T8'),
    ('T8','neck'), ('neck','head'),
    ('T8','shoulderRight'), ('shoulderRight','upperArmRight'),
    ('upperArmRight','forearmRight'), ('forearmRight','handRight'),
    ('T8','shoulderLeft'), ('shoulderLeft','upperArmLeft'),
    ('upperArmLeft','forearmLeft'), ('forearmLeft','handLeft'),
    ('pelvis','upperLegRight'), ('upperLegRight','lowerLegRight'),
    ('lowerLegRight','footRight'), ('footRight','toeRight'),
    ('pelvis','upperLegLeft'), ('upperLegLeft','lowerLegLeft'),
    ('lowerLegLeft','footLeft'), ('footLeft','toeLeft'),
]

# High contrast colors on white background
# Spine = deep blue, Right arm = bright orange, Left arm = bright green,
# Right leg = crimson red, Left leg = purple
def bone_color(j1, j2):
    combo = j1 + j2
    if 'Right' in combo and ('Leg' in combo or 'foot' in combo.lower() or 'toe' in combo.lower()):
        return '#e63946'   # vivid red
    if 'Left' in combo and ('Leg' in combo or 'foot' in combo.lower() or 'toe' in combo.lower()):
        return '#9b2226'   # deep red-purple
    if 'Right' in combo and ('Arm' in combo or 'hand' in combo.lower() or 'shoulder' in combo.lower()):
        return '#f77f00'   # vivid orange
    if 'Left' in combo and ('Arm' in combo or 'hand' in combo.lower() or 'shoulder' in combo.lower()):
        return '#fcbf49'   # yellow-orange
    return '#023e8a'       # deep navy blue for spine

def load_true(path):
    df = pd.read_csv(path)
    pos = np.zeros((len(df), len(JOINTS), 3))
    for ji, j in enumerate(JOINTS):
        pos[:, ji, 0] = df[f'{j}_x'].values
        pos[:, ji, 1] = df[f'{j}_y'].values
        pos[:, ji, 2] = df[f'{j}_z'].values
    return pos

def load_filter(path):
    df = pd.read_csv(path)
    pos = np.zeros((len(df), len(JOINTS), 3))
    for ji, j in enumerate(JOINTS):
        pos[:, ji, 0] = df[f'{j}_px'].values
        pos[:, ji, 1] = df[f'{j}_py'].values
        pos[:, ji, 2] = df[f'{j}_pz'].values
    return pos

def make_gif(pos, title, subtitle, out_path, step=5, fps=18):
    frames_idx = list(range(0, len(pos), step))
    N = len(frames_idx)

    fig = plt.figure(figsize=(6, 6.5), dpi=110)
    ax  = fig.add_subplot(111, projection='3d')

    # White background — maximum contrast
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor('#f0f0f0')
        pane.set_edgecolor('#cccccc')
        pane.set_alpha(0.4)
    ax.grid(True, color='#dddddd', linewidth=0.5)
    ax.tick_params(colors='#444', labelsize=6)

    # Axis limits — tight around skeleton
    cx = (pos[:,:,0].min() + pos[:,:,0].max()) / 2
    cy = (pos[:,:,1].min() + pos[:,:,1].max()) / 2
    cz = (pos[:,:,2].min() + pos[:,:,2].max()) / 2
    xspan = (pos[:,:,0].max() - pos[:,:,0].min()) * 0.6
    yspan = (pos[:,:,1].max() - pos[:,:,1].min()) * 0.6
    zspan = (pos[:,:,2].max() - pos[:,:,2].min()) * 0.7
    span  = max(xspan, yspan)
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(cz - zspan * 0.3, cz + zspan * 1.1)

    ax.set_xlabel('X (m)', fontsize=7, color='#333')
    ax.set_ylabel('Y (m)', fontsize=7, color='#333')
    ax.set_zlabel('Z (m)', fontsize=7, color='#333')

    # Title block
    fig.text(0.5, 0.98, title,    ha='center', va='top',
             color='#111111', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.95, subtitle, ha='center', va='top',
             color='#555555', fontsize=8)

    # Legend patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color='#023e8a', lw=2.5, label='Spine'),
        Line2D([0],[0], color='#f77f00', lw=2.5, label='Right arm'),
        Line2D([0],[0], color='#fcbf49', lw=2.5, label='Left arm'),
        Line2D([0],[0], color='#e63946', lw=2.5, label='Right leg'),
        Line2D([0],[0], color='#9b2226', lw=2.5, label='Left leg'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=6.5, framealpha=0.7, edgecolor='#ccc')

    # Pre-build line objects
    lines = []
    for (j1, j2) in BONES:
        ln, = ax.plot([], [], [], '-', color=bone_color(j1, j2),
                      lw=2.8, solid_capstyle='round', alpha=0.95)
        lines.append(ln)

    # Joint dots — black on white background = very visible
    scat = ax.scatter([], [], [], c='#222222', s=20,
                      depthshade=False, zorder=6)
    ttxt = fig.text(0.97, 0.03, '', color='#666', fontsize=7, ha='right')

    def update(fi):
        t = frames_idx[fi]
        p = pos[t]
        for ln, (j1, j2) in zip(lines, BONES):
            i1, i2 = JI[j1], JI[j2]
            ln.set_data([p[i1,0], p[i2,0]], [p[i1,1], p[i2,1]])
            ln.set_3d_properties([p[i1,2], p[i2,2]])
        scat._offsets3d = (p[:,0], p[:,1], p[:,2])
        # Slow full 360° rotation
        ax.view_init(elev=20, azim=30 + (fi / N) * 360)
        ttxt.set_text(f't = {t*0.01:.1f} s')
        return lines + [scat, ttxt]

    ani = animation.FuncAnimation(fig, update, frames=N,
                                  interval=1000//fps, blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(out_path)}  ({N} frames)')

# ── Load ───────────────────────────────────────────────────────────────────
print('Loading...')
pos_true = load_true('/mnt/user-data/uploads/3D_Full_Body_Humain_Gait_Walking_Dataset__True_Values_.csv')
pos_lkf  = load_filter('/mnt/user-data/uploads/lkf_output__1_.csv')
pos_ekf  = load_filter('/mnt/user-data/uploads/ekf_output__1_.csv')
print(f'  True {pos_true.shape} | LKF {pos_lkf.shape} | EKF {pos_ekf.shape}')

print('Rendering gait_true.gif ...')
make_gif(pos_true, 'Ground Truth', '3D Full-Body Gait — True Positions',
         f'{OUT_DIR}/gait_true.gif')

print('Rendering gait_lkf.gif ...')
make_gif(pos_lkf, 'LKF Estimates', 'Linear Kalman Filter  |  σⱼ = 0.5 m/s³  |  RMSE = 205.5 mm',
         f'{OUT_DIR}/gait_lkf.gif')

print('Rendering gait_ekf.gif ...')
make_gif(pos_ekf, 'EKF Estimates', 'Extended Kalman Filter  |  σⱼ = 10.0 m/s³  |  RMSE = 221.7 mm',
         f'{OUT_DIR}/gait_ekf.gif')

print('\nAll done.')
