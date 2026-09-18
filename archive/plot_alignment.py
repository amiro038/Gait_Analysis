"""Visual validation of the MOVE4D -> Theia alignment."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import align_m4d_to_theia as A

# bones drawn in the overlay, in Theia vocabulary
BONES = [
    ("l_thigh", "l_shank"), ("l_shank", "l_foot"), ("l_foot", "l_toes"),
    ("r_thigh", "r_shank"), ("r_shank", "r_foot"), ("r_foot", "r_toes"),
    ("l_uarm", "l_larm"), ("r_uarm", "r_larm"),
    ("l_thigh", "r_thigh"), ("l_uarm", "r_uarm"),
    ("pelvis", "thorax"), ("thorax", "head"),
]
FITTED = set(A.TRANSLATION_JOINTS)

C_THEIA = "#1f4e79"
C_M4D = "#c0392b"


def _skeleton(ax, pos, color, label, i, j, style):
    for a, b in BONES:
        if a in pos and b in pos:
            ax.plot([pos[a][i], pos[b][i]], [pos[a][j], pos[b][j]],
                    style, color=color, lw=1.6, zorder=2)
    fx = [pos[k][i] for k in pos if k in FITTED]
    fy = [pos[k][j] for k in pos if k in FITTED]
    ux = [pos[k][i] for k in pos if k not in FITTED]
    uy = [pos[k][j] for k in pos if k not in FITTED]
    ax.scatter(fx, fy, s=26, color=color, zorder=3, label=label)
    ax.scatter(ux, uy, s=26, facecolors="none", edgecolors=color,
               zorder=3, linewidths=1.2)


def make_plot(pairs, T, outpath):
    tp, mp = pairs[0]
    th = A.Trial(tp, A.JOINT_MAP.keys())
    m4 = A.Trial(mp, A.JOINT_MAP.values())
    m4.pos = {tk: m4.pos[mk] for tk, mk in A.JOINT_MAP.items() if mk in m4.pos}
    m4.resolved = {tk: m4.resolved[mk] for tk, mk in A.JOINT_MAP.items()
                   if mk in m4.resolved}

    pt = th.mean_pose()
    # undo the canonicalisation so we can apply T in raw file coordinates
    Cinv = m4.C.T
    pm_raw = {k: Cinv @ v for k, v in m4.mean_pose().items()}
    pm = {k: A.apply_transform(T, v) for k, v in pm_raw.items()}

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], hspace=0.28, wspace=0.26)

    for col, (i, j, name) in enumerate([(0, 2, "front (X-Z)"),
                                        (1, 2, "side (Y-Z)"),
                                        (0, 1, "top (X-Y)")]):
        ax = fig.add_subplot(gs[0, col])
        _skeleton(ax, pt, C_THEIA, "Theia3D", i, j, "-")
        _skeleton(ax, pm, C_M4D, "MOVE4D (transformed)", i, j, "--")
        ax.set_title(f"{name}", fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # --- per-joint position residual --------------------------------
    ax = fig.add_subplot(gs[1, 0:2])
    keys = [k for k in pt if k in pm]
    keys.sort(key=lambda k: -np.linalg.norm(pt[k] - pm[k]))
    vals = [np.linalg.norm(pt[k] - pm[k]) * 1000 for k in keys]
    cols = [C_THEIA if k in FITTED else "#999999" for k in keys]
    ax.bar(range(len(keys)), vals, color=cols)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("residual (mm)", fontsize=9)
    ax.set_title("Per-joint position residual  "
                 "(solid = fitted, grey = excluded landmark)", fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    # --- per-direction angular residual -----------------------------
    d, p, dg, info = A.build_observations(th, m4)
    R = T[:3, :3] @ m4.C.T
    ax = fig.add_subplot(gs[1, 2])
    seen, labs, res = set(), [], []
    for lab, u, v, w in d:
        if lab in seen:
            continue
        seen.add(lab)
        labs.append(lab)
        res.append(np.degrees(np.arccos(np.clip(np.dot(u, R @ v), -1, 1))))
    order = np.argsort(res)[::-1]
    ax.barh([labs[i] for i in order][::-1], [res[i] for i in order][::-1],
            color=C_M4D)
    ax.set_xlabel("angular residual (deg)", fontsize=9)
    ax.set_title("Per-direction angular residual", fontsize=10)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    fig.suptitle("MOVE4D -> Theia3D alignment check", fontsize=12, y=0.97)
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
