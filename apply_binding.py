# -*- coding: utf-8 -*-
"""
=============================================================================
 STEP 2 of 2 -- replay a binding over a trial to get mesh vertex positions
=============================================================================

Reads the .npz written by bind_mesh_to_bones.py and a Visual3D metrics export
of any trial, and reconstructs where every mesh vertex was, frame by frame.

    python apply_binding.py TRIAL_metrics.csv
    python apply_binding.py TRIAL_metrics.csv --plot            # 3D viewer
    python apply_binding.py TRIAL_metrics.csv --plot-frames 0,120,240
    python apply_binding.py TRIAL_metrics.csv --plot-frames 0,120 --feet
    python apply_binding.py TRIAL_metrics.csv --vertices --obj

HOW IT WORKS
------------
If the trial carries the segment's `<Side>_Foot_Global_4x4` signal, that IS
the pose -- it is read straight out and used. Nothing is fitted, so there is
no reconstruction error to report.

Otherwise the pose is fitted from the stored landmarks by Kabsch, least
squares over all of them, so extra landmarks improve the result rather than
being ignored.

When both are available the script fits the landmarks anyway and reports how
far the two disagree. That is a free end-to-end check on the whole chain: two
independent routes to the same pose.

WHAT COMES OUT
--------------
`poses` (frames, segments, 4, 4) is always written and is tiny -- it is the
whole result, since vertices are one matrix multiply away. `--vertices` also
stores the (frames, vertices, 3) array and `--obj` writes one .obj per frame.
Both get large fast: 10000 vertices over 500 frames is 60 MB as float32.

Frames where the pose is missing come back as NaN rather than stopping the
run, and interior gaps keep their frame numbering.

This script is deliberately standalone -- it shares no imports with
bind_mesh_to_bones.py, so you can hand it and the .npz to someone else. The
metrics reader below is a copy of the one there; fix bugs in both.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


# %%==========================================================================
#  CONFIG
# ============================================================================

BINDING_FILE = "foot_mesh_binding.npz"
TRIAL_METRICS_CSV = "D05_C01_SKS_metrics.csv"

SAVE_VERTICES = False
WRITE_OBJ_SEQUENCE = False
OBJ_DIR = "posed_frames"
OBJ_STRIDE = 1
OUT_SUFFIX = "_posed"
VERBOSE = True

SHAPE_TOLERANCE_MM = 5.0

# --- 3D check figure --------------------------------------------------------
# Draws the whole body from the export -- every joint and segment position,
# the skeleton linking them -- with the posed mesh on top and each driven
# segment's own axes as arrows. If the mesh sits on the skeleton's feet and
# its axes follow the ankle, the binding is right. This is the check to
# trust: it reads the same arrays the rest of the script works with, so
# nothing can be lost in a file format on the way out.
MESH_POINT_STRIDE = 6        # plot every Nth mesh vertex (10208 is a lot)
AXIS_ARROW_M = 0.12          # length of the segment-axis arrows
PLOT_ELEV, PLOT_AZIM = 18, -62
# "body" frames the whole person; "feet" crops to the bound meshes, which is
# where you can actually judge whether the binding sits right.
PLOT_ZOOM = "body"

# Joint centres (blue) and segment origins (green), exactly as Visual3D names
# them. Anything missing from the export is skipped silently.
SKELETON_JOINTS = [
    "Head_Position", "Trunk_Position", "Low_Back_Position", "Pelvis_Position",
    "Left_Shoulder_Position", "Left_Elbow_Position", "Left_Wrist_Position",
    "Right_Shoulder_Position", "Right_Elbow_Position", "Right_Wrist_Position",
    "Left_Hip_Position", "Left_Knee_Position", "Left_Ankle_Position",
    "Left_Toes_Position", "Left_Heel_Position",
    "Right_Hip_Position", "Right_Knee_Position", "Right_Ankle_Position",
    "Right_Toes_Position", "Right_Heel_Position",
]
SKELETON_SEGMENTS = [
    "Left_Upper_Arm_Position", "Left_Forearm_Position", "Left_Hand_Position",
    "Right_Upper_Arm_Position", "Right_Forearm_Position", "Right_Hand_Position",
    "Left_Thigh_Position", "Left_Shank_Position", "Left_Foot_Position",
    "Right_Thigh_Position", "Right_Shank_Position", "Right_Foot_Position",
]
SKELETON_LINKS = [
    ("Head_Position", "Trunk_Position"),
    ("Trunk_Position", "Low_Back_Position"),
    ("Low_Back_Position", "Pelvis_Position"),
    ("Trunk_Position", "Left_Shoulder_Position"),
    ("Trunk_Position", "Right_Shoulder_Position"),
    ("Right_Shoulder_Position", "Right_Upper_Arm_Position"),
    ("Right_Upper_Arm_Position", "Right_Elbow_Position"),
    ("Right_Elbow_Position", "Right_Forearm_Position"),
    ("Right_Forearm_Position", "Right_Wrist_Position"),
    ("Right_Wrist_Position", "Right_Hand_Position"),
    ("Left_Shoulder_Position", "Left_Upper_Arm_Position"),
    ("Left_Upper_Arm_Position", "Left_Elbow_Position"),
    ("Left_Elbow_Position", "Left_Forearm_Position"),
    ("Left_Forearm_Position", "Left_Wrist_Position"),
    ("Left_Wrist_Position", "Left_Hand_Position"),
    ("Pelvis_Position", "Right_Hip_Position"),
    ("Right_Hip_Position", "Right_Thigh_Position"),
    ("Right_Thigh_Position", "Right_Knee_Position"),
    ("Right_Knee_Position", "Right_Shank_Position"),
    ("Right_Shank_Position", "Right_Ankle_Position"),
    ("Right_Ankle_Position", "Right_Foot_Position"),
    ("Right_Foot_Position", "Right_Toes_Position"),
    ("Right_Ankle_Position", "Right_Heel_Position"),
    ("Pelvis_Position", "Left_Hip_Position"),
    ("Left_Hip_Position", "Left_Thigh_Position"),
    ("Left_Thigh_Position", "Left_Knee_Position"),
    ("Left_Knee_Position", "Left_Shank_Position"),
    ("Left_Shank_Position", "Left_Ankle_Position"),
    ("Left_Ankle_Position", "Left_Foot_Position"),
    ("Left_Foot_Position", "Left_Toes_Position"),
    ("Left_Ankle_Position", "Left_Heel_Position"),
]


# %%==========================================================================
#  VISUAL3D METRICS READER  (copy of the one in bind_mesh_to_bones.py)
# ============================================================================

def read_metrics(path):
    """-> (vectors {name: (F,3)}, matrices {name: (F,4,4)}, frame ids)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh]
    if len(rows) < 6:
        raise ValueError(f"{path}: too short to be a Visual3D metrics export")

    names, comps = rows[1][1:], rows[4][1:]
    width = len(names)
    vals, items = [], []
    for r in rows[5:]:
        cells = r[1:]
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        vals.append([float(c) if c.strip() else np.nan for c in cells[:width]])
        items.append(r[0].strip())
    data = np.array(vals, float)

    blank = np.all(np.isnan(data), axis=1)
    last = len(data)
    while last > 0 and blank[last - 1]:        # trailing padding only
        last -= 1
    data, items = data[:last], items[:last]
    if not len(data):
        raise ValueError(f"{path}: no frames with data")

    by_name = {}
    for i, (n, c) in enumerate(zip(names, comps)):
        by_name.setdefault(n, {})[c.strip().upper()] = data[:, i]
    vectors, matrices = {}, {}
    for n, ch in by_name.items():
        if {"X", "Y", "Z"} <= set(ch):
            vectors[n] = np.stack([ch["X"], ch["Y"], ch["Z"]], axis=1)
        elif all(str(k) in ch for k in range(16)):
            matrices[n] = np.stack([ch[str(k)] for k in range(16)],
                                   axis=1).reshape(-1, 4, 4)
    return vectors, matrices, items


# %%==========================================================================
#  POSE RECOVERY
# ============================================================================

def kabsch(P, Q):
    """Rigid (R, t) taking P onto Q, both (K, 3). Least squares over all K."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((Q - cq).T @ (P - cp))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, cq - R @ cp


def poses_from_landmarks(local, world):
    """local (K,3), world (F,K,3) -> poses (F,4,4). Bad frames come back NaN."""
    out = np.full((len(world), 4, 4), np.nan)
    for i, W in enumerate(world):
        if not np.isfinite(W).all():
            continue
        R, t = kabsch(local, W)
        out[i, :3, :3] = R
        out[i, :3, 3] = t
        out[i, 3, :] = (0.0, 0.0, 0.0, 1.0)
    return out


def orthonormalise(R):
    U, _, Vt = np.linalg.svd(R)
    D = np.ones(U.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


def rotation_gap_deg(A, B):
    """Angle between two stacks of rotations, in degrees."""
    tr = np.trace(np.einsum("fij,fkj->fik", A, B), axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


# %%==========================================================================
#  3D CHECK FIGURE
# ============================================================================

def _mesh_points(z, poses, frame, stride=None):
    """Posed mesh vertices for one frame, subsampled. -> (M,3) or None."""
    stride = MESH_POINT_STRIDE if stride is None else stride
    v_local, seg_of = z["vertices_local"], z["vertex_segment"]
    idx = np.arange(0, len(v_local), max(1, stride))
    out = np.full((len(idx), 3), np.nan)
    for k in range(poses.shape[1]):
        P = poses[frame, k]
        if not np.isfinite(P).all():
            continue
        m = seg_of[idx] == k
        if m.any():
            out[m] = v_local[idx][m] @ P[:3, :3].T + P[:3, 3]
    return out


def _frame_extent(vectors, z, poses, zoom=None):
    """One fixed set of axis limits for the whole trial, so nothing jumps."""
    zoom = PLOT_ZOOM if zoom is None else zoom
    pts = []
    if zoom != "feet":
        for n in SKELETON_JOINTS + SKELETON_SEGMENTS:
            if n in vectors:
                pts.append(vectors[n])
    else:
        # _Position only. Matching on the substring alone would also pull in
        # Ankle_Joint_Acc and Foot_Seg_Angle, whose values run to thousands
        # and blow the axis limits to nonsense.
        for n in SKELETON_JOINTS + SKELETON_SEGMENTS:
            if n in vectors and any(k in n for k in
                                    ("Ankle", "Toes", "Heel", "Foot")):
                pts.append(vectors[n])
    for f in range(0, len(poses), max(1, len(poses) // 25)):
        mp = _mesh_points(z, poses, f, stride=40)
        if mp is not None:
            pts.append(mp)
    P = np.vstack(pts)
    P = P[np.isfinite(P).all(axis=1)]
    lo, hi = P.min(0), P.max(0)
    ctr = (lo + hi) / 2
    r = float((hi - lo).max()) / 2 * 1.08
    return ctr, max(r, 0.2)


def draw_3d_frame(ax, vectors, z, poses, frame, meta, ctr=None, r=None,
                  title=None):
    """One frame: skeleton, segment origins, posed mesh, segment axes."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    ax.clear()
    pos = {n: vectors[n][frame] for n in vectors
           if frame < len(vectors[n]) and np.isfinite(vectors[n][frame]).all()}

    links = [[pos[a], pos[b]] for a, b in SKELETON_LINKS
             if a in pos and b in pos]
    if links:
        ax.add_collection3d(Line3DCollection(links, colors="0.35", linewidths=1.4))

    J = np.array([pos[n] for n in SKELETON_JOINTS if n in pos])
    S = np.array([pos[n] for n in SKELETON_SEGMENTS if n in pos])
    if len(J):
        ax.scatter(J[:, 0], J[:, 1], J[:, 2], c="#1f4e79", s=26, depthshade=False)
    if len(S):
        ax.scatter(S[:, 0], S[:, 1], S[:, 2], c="#2e8b57", s=26, depthshade=False)

    MP = _mesh_points(z, poses, frame)
    if MP is not None:
        ok = np.isfinite(MP).all(axis=1)
        if ok.any():
            ax.scatter(MP[ok, 0], MP[ok, 1], MP[ok, 2], c="#c0392b", s=1.2,
                       alpha=0.5, depthshade=False)

    # each driven segment's own axes -- this is the orientation the mesh rides
    for k, seg in enumerate(meta["segments"]):
        P = poses[frame, k]
        if not np.isfinite(P).all():
            continue
        o = P[:3, 3]
        for axis, colour in zip(range(3), ("#e74c3c", "#27ae60", "#2980b9")):
            d = P[:3, axis] * AXIS_ARROW_M
            ax.plot([o[0], o[0] + d[0]], [o[1], o[1] + d[1]],
                    [o[2], o[2] + d[2]], color=colour, lw=2)

    if ctr is not None:
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=PLOT_ELEV, azim=PLOT_AZIM)
    ax.set_title(title or f"frame {frame}", fontsize=10)


def _legend(fig):
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color="#1f4e79", marker="o", ls="none", label="joint centre"),
        Line2D([], [], color="#2e8b57", marker="o", ls="none", label="segment origin"),
        Line2D([], [], color="#c0392b", marker="o", ls="none", ms=4,
               label="posed mesh"),
        Line2D([], [], color="#e74c3c", label="segment X"),
        Line2D([], [], color="#27ae60", label="segment Y"),
        Line2D([], [], color="#2980b9", label="segment Z"),
    ], loc="lower center", ncol=6, fontsize=8, frameon=False)


def save_3d_figure(result, frames=None, path=None, zoom=None, verbose=True):
    """A static grid of frames. Works headless; good for a report."""
    import matplotlib.pyplot as plt

    vectors, z, poses, meta = (result["vectors"], result["binding_npz"],
                               result["poses"], result["meta"])
    n = len(poses)
    if not frames:
        frames = [int(round(x)) for x in np.linspace(0, n - 1, 6)]
    frames = [f for f in frames if 0 <= f < n]
    ctr, r = _frame_extent(vectors, z, poses, zoom)

    cols = min(3, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    fig = plt.figure(figsize=(5.2 * cols, 4.6 * rows))
    for i, f in enumerate(frames):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        draw_3d_frame(ax, vectors, z, poses, f, meta, ctr, r)
    _legend(fig)
    fig.suptitle(f"{result['trial']}   mesh on the Theia skeleton"
                 + ("   [feet]" if (zoom or PLOT_ZOOM) == "feet" else ""),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = path or (os.path.splitext(result["out"])[0]
                    + ("_feet" if (zoom or PLOT_ZOOM) == "feet" else "")
                    + "_check.png")
    fig.savefig(path, dpi=130)
    if verbose:
        print(f"  wrote {path}")
    return fig, path


def show_3d(result, frame=0, zoom=None):
    """Interactive viewer with a frame slider. Needs a GUI backend.

    In Spyder: Preferences > IPython console > Graphics > Backend: Automatic.
    With the inline backend there is no slider, so use save_3d_figure instead.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    vectors, z, poses, meta = (result["vectors"], result["binding_npz"],
                               result["poses"], result["meta"])
    ctr, r = _frame_extent(vectors, z, poses, zoom)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes((0.02, 0.12, 0.96, 0.84), projection="3d")
    sax = fig.add_axes((0.12, 0.04, 0.76, 0.03))
    slider = Slider(sax, "frame", 0, len(poses) - 1, valinit=frame, valstep=1)

    def redraw(val):
        draw_3d_frame(ax, vectors, z, poses, int(slider.val), meta, ctr, r,
                      title=f"{result['trial']}   frame {int(slider.val)} "
                            f"of {len(poses) - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(redraw)
    redraw(frame)
    _legend(fig)
    fig._binding_slider = slider          # keep a reference alive
    plt.show()
    return fig, slider


# %%==========================================================================
#  MAIN
# ============================================================================

def load_binding(path=None):
    path = BINDING_FILE if path is None else path
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run bind_mesh_to_bones.py")
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["meta"])), z


def apply_binding(trial_csv=None, binding_file=None, save_vertices=None,
                  write_obj=None, verbose=None):
    trial_csv = TRIAL_METRICS_CSV if trial_csv is None else trial_csv
    save_vertices = SAVE_VERTICES if save_vertices is None else save_vertices
    write_obj = WRITE_OBJ_SEQUENCE if write_obj is None else write_obj
    verbose = VERBOSE if verbose is None else verbose

    meta, z = load_binding(binding_file)
    v_local, n_local = z["vertices_local"], z["normals_local"]
    seg_of, lm_local = z["vertex_segment"], z["landmarks_local"]
    seg_names = meta["segments"]

    vectors, matrices, items = read_metrics(trial_csv)

    if verbose:
        W = 74
        print("=" * W)
        print("  APPLY BINDING")
        print("=" * W)
        print(f"\n  binding : {os.path.basename(binding_file or BINDING_FILE)}"
              f"  (built from {meta['static_metrics_csv']})")
        print(f"  trial   : {os.path.basename(trial_csv)}  "
              f"({len(items)} frames)")
        print(f"  meshes  : {', '.join(m['name'] for m in meta['meshes'])}"
              f"  ({len(v_local)} vertices)")

    poses = np.full((len(items), len(seg_names), 4, 4), np.nan)
    stats = []
    for k, s in enumerate(seg_names):
        sig = meta["pose_signal"].get(s)
        names = meta["landmark_names"].get(s, [])
        have_lms = len(names) >= 3 and all(n in vectors for n in names)

        fitted = None
        if have_lms:
            world = np.stack([vectors[n] for n in names], axis=1)
            fitted = poses_from_landmarks(lm_local[k, :len(names)], world)

        if sig and sig in matrices:
            P = matrices[sig].copy()
            P[:, :3, :3] = orthonormalise(P[:, :3, :3])   # strip numeric drift
            P[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
            poses[:, k] = P
            used = sig
        elif fitted is not None:
            poses[:, k] = fitted
            used = "landmarks"
        else:
            raise KeyError(
                f"{s}: the trial has neither '{sig}' nor the landmarks "
                f"{names}. It must carry what the binding was built from.")

        st = dict(segment=s, source=used,
                  n_good=int(np.isfinite(poses[:, k, 3, 3]).sum()))
        if fitted is not None and used != "landmarks":
            ok = np.isfinite(fitted[:, 3, 3]) & np.isfinite(poses[:, k, 3, 3])
            if ok.any():
                st["cross_check_deg"] = float(np.mean(rotation_gap_deg(
                    poses[ok][:, k, :3, :3], fitted[ok][:, :3, :3])))
                st["cross_check_mm"] = float(np.mean(np.linalg.norm(
                    poses[ok][:, k, :3, 3] - fitted[ok][:, :3, 3],
                    axis=1)) * 1000)
        if have_lms:
            world = np.stack([vectors[n] for n in names], axis=1)
            loc = lm_local[k, :len(names)]
            ref_d, trial_d = [], []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    ref_d.append(np.linalg.norm(loc[i] - loc[j]))
                    trial_d.append(np.nanmean(np.linalg.norm(
                        world[:, i] - world[:, j], axis=1)))
            st["shape_mismatch_mm"] = float(np.max(np.abs(
                np.array(ref_d) - np.array(trial_d))) * 1000)
        stats.append(st)

    verts = norms = None
    if save_vertices or write_obj:
        verts = np.full((len(items), len(v_local), 3), np.nan, dtype=np.float32)
        norms = np.full((len(items), len(n_local), 3), np.nan, dtype=np.float32)
        for k in range(len(seg_names)):
            m = seg_of == k
            if not m.any():
                continue
            Vl, Nl = v_local[m], n_local[m]
            for i, P in enumerate(poses[:, k]):
                if np.isfinite(P).all():
                    R = P[:3, :3]
                    verts[i, m] = (Vl @ R.T + P[:3, 3]).astype(np.float32)
                    norms[i, m] = (Nl @ R.T).astype(np.float32)

    stem = os.path.splitext(os.path.basename(trial_csv))[0]
    out_npz = stem + OUT_SUFFIX + ".npz"
    payload = dict(poses=poses, segments=np.array(seg_names),
                   frames=np.array(items), vertex_segment=seg_of,
                   meta=np.array(json.dumps(dict(
                       trial=os.path.basename(trial_csv), binding=meta,
                       stats=stats))))
    if save_vertices:
        payload["vertices"] = verts
    np.savez_compressed(out_npz, **payload)

    n_obj = write_obj_sequence(meta, verts, norms, stem) if write_obj else 0

    if verbose:
        print("\n" + "-" * 74)
        print("  POSE RECOVERY")
        print("-" * 74)
        for st in stats:
            print(f"\n    {st['segment']:12s} {st['n_good']:5d}/{len(items)} "
                  f"frames | source: {st['source']}")
            if st["source"] != "landmarks":
                print("      read straight from the export -- nothing fitted,")
                print("      so there is no reconstruction error here at all.")
            if "cross_check_deg" in st:
                print(f"      cross-check vs an independent landmark fit: "
                      f"{st['cross_check_deg']:.3f} deg, "
                      f"{st['cross_check_mm']:.3f} mm")
            if "shape_mismatch_mm" in st:
                print(f"      landmark geometry vs the binding: "
                      f"{st['shape_mismatch_mm']:.2f} mm", end="")
                print("   *** MISMATCH -- different subject or model? ***"
                      if st["shape_mismatch_mm"] > SHAPE_TOLERANCE_MM
                      else "   OK, same body")
        print("\n" + "-" * 74)
        print(f"  wrote {out_npz}  ({os.path.getsize(out_npz) / 1e6:.1f} MB)")
        print("    poses          (frames, segments, 4, 4)   always")
        if save_vertices:
            print("    vertices       (frames, vertices, 3)     float32")
        else:
            print("    vertices       not stored -- pass --vertices, or "
                  "rebuild with")
            print("                   vertices_at_frame() below (one matmul)")
        if n_obj:
            print(f"  wrote {n_obj} .obj files to {OBJ_DIR}/")
        print("-" * 74)
    return dict(poses=poses, vertices=verts, normals=norms, stats=stats,
                frames=items, out=out_npz, vectors=vectors, matrices=matrices,
                binding_npz=z, meta=meta,
                trial=os.path.basename(trial_csv))


def vertices_at_frame(z, poses, frame):
    """World vertices for one frame from the compact `poses` array."""
    v_local, seg_of = z["vertices_local"], z["vertex_segment"]
    out = np.full_like(v_local, np.nan)
    for k in range(poses.shape[1]):
        P = poses[frame, k]
        if np.isfinite(P).all():
            m = seg_of == k
            out[m] = v_local[m] @ P[:3, :3].T + P[:3, 3]
    return out


def write_obj_sequence(meta, verts, norms, stem):
    """One .obj per frame, reusing each mesh's original faces and comments."""
    os.makedirs(OBJ_DIR, exist_ok=True)
    n = 0
    for i in range(0, len(verts), OBJ_STRIDE):
        if not np.isfinite(verts[i]).any():
            continue
        for m in meta["meshes"]:
            lines = list(m["lines"])
            sl = slice(m["start"], m["start"] + m["count"])
            V = verts[i][sl]
            for j, li in enumerate(m["v_at"]):
                lines[li] = "v %.6f %.6f %.6f" % tuple(V[j])
            if m["has_normals"] and norms is not None:
                N = norms[i][sl]
                for j, li in enumerate(m["n_at"]):
                    lines[li] = "vn %.6f %.6f %.6f" % tuple(N[j])
            path = os.path.join(
                OBJ_DIR, f"{stem}_{os.path.splitext(m['name'])[0]}_{i:05d}.obj")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            n += 1
    return n


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    files = [a for a in argv if not a.startswith("--")]
    res = apply_binding(files[0] if files else None,
                        save_vertices="--vertices" in argv,
                        write_obj="--obj" in argv)

    frames = None
    for a in argv:
        if a.startswith("--plot-frames"):
            spec = a.split("=", 1)[1] if "=" in a else ""
            if not spec:
                i = argv.index(a)
                spec = argv[i + 1] if i + 1 < len(argv) else ""
            frames = [int(x) for x in spec.replace(",", " ").split() if
                      x.lstrip("-").isdigit()]
    zoom = "feet" if "--feet" in argv else None
    if frames is not None:
        save_3d_figure(res, frames, zoom=zoom)
    elif "--plot" in argv:
        show_3d(res, zoom=zoom)
    return res


if __name__ == "__main__":
    RESULT = main()
