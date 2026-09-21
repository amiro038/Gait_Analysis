# -*- coding: utf-8 -*-
"""
=============================================================================
 STEP 2 of 2 -- replay a binding over a trial to get mesh vertex positions
=============================================================================

Reads the .npz written by build_foot_binding.py and a Visual3D metrics export
of any trial, and reconstructs where every mesh vertex was, frame by frame.

Press Run in Spyder (or F5) and it uses the CONFIG block below -- no
arguments needed. SHOW_3D_VIEWER / SAVE_3D_FIGURE / SAVE_3D_VIDEO there
decide what you get to look at: a slider window, a PNG grid of frames, and a
movie of the whole trial. From a terminal you can override with flags:

    python apply_binding.py TRIAL_metrics.csv
    python apply_binding.py TRIAL_metrics.csv --plot            # 3D viewer
    python apply_binding.py TRIAL_metrics.csv --plot-frames 0,120,240
    python apply_binding.py TRIAL_metrics.csv --plot-frames 0,120 --feet
    python apply_binding.py TRIAL_metrics.csv --video          # movie only
    python apply_binding.py TRIAL_metrics.csv --video --feet
    python apply_binding.py TRIAL_metrics.csv --no-video
    python apply_binding.py TRIAL_metrics.csv --no-plot
    python apply_binding.py TRIAL_metrics.csv --vertices
    python apply_binding.py TRIAL_metrics.csv --h5

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
`poses` (frames, segments, 4, 4) is the whole result and it is tiny. The
binding is rigid -- every vertex rides exactly one segment -- so a vertex
position is exactly

    p(t) = R_seg(t) @ v_local + t_seg(t)

and the (frames, vertices, 3) array is an *algebraically exact* function of
`poses`. Storing it is storing the same information over and over: for a
3000-frame trial of 10208 vertices, 0.25 MB of poses against 368 MB of baked
positions, a factor of about 1500, and the rebuild agrees to the last bit
rather than to some tolerance.

So do not store vertices. Call `vertex_tracks()` on the small .npz and ask
for the vertices you need:

    import apply_binding as ab
    V, idx = ab.vertex_tracks("TRIAL_posed.npz", mesh="left")

`V` is (frames, len(idx), 3) in Theia world metres, `idx` the global vertex
index of each column. Units are metres and the frame is Theia's, the same one
the skeleton is in, so the array drops straight into any analysis that already
works on the export's joint positions.

Measured on a 3000-frame trial: the whole mesh rebuilds in 0.69 s, one foot
in 0.65 s, 100 vertices in 15 ms, a single vertex in 0.5 ms. Reading the
368 MB baked array off disk takes 0.09 s, so it only wins if you genuinely
want every vertex of every frame at once -- and it costs 368 MB to do it.

`--vertices` (into the .npz) and `--h5` (a standalone HDF5 beside the trial)
still exist for handing a baked array to something that cannot do the matrix
multiply itself. They are off by default and should stay that way.

Frames where the pose is missing come back as NaN rather than stopping the
run, and interior gaps keep their frame numbering.

This script is deliberately standalone -- it shares no imports with
build_foot_binding.py, so you can hand it and the .npz to someone else. The
metrics reader below is a copy of the one there; fix bugs in both.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

import numpy as np


# %%==========================================================================
#  CONFIG
# ============================================================================

BINDING_FILE = "foot_mesh_binding.npz"
TRIAL_METRICS_CSV = "D05_C01_SKS_metrics.csv"

# Baked vertex positions. Both of these write the (frames, vertices, 3) array
# that `poses` already determines exactly, so both are redundant by ~1500x and
# both are off. Turn one on only to hand the array to something that cannot
# rebuild it -- a collaborator without the binding, or a tool that reads
# arrays and not matrices. For your own analysis use vertex_tracks().
SAVE_VERTICES = False        # also put the full array in the _posed.npz
SAVE_MESH_H5 = False         # ... or write it beside the trial CSV as HDF5
H5_SUFFIX = "_mesh.h5"
H5_DTYPE = "float32"         # float64 doubles the file for sub-micron gains
H5_COMPRESS = "gzip"         # gzip+shuffle buys ~24% on float coordinates and
                             # costs ~20x the write time; None is the other
                             # sensible choice. Measured, not guessed.
OUT_SUFFIX = "_posed"
# Where the _posed.npz goes. None puts it beside the trial CSV, so an
# analysis script can find it from the trial path alone and it does not
# matter which directory you happened to run from -- which matters with
# absolute trial paths and Spyder, where the working directory is rarely
# the data folder. Set "." to keep it in the working directory instead.
OUT_DIR = None
VERBOSE = True

SHAPE_TOLERANCE_MM = 5.0

# --- 3D check figure --------------------------------------------------------
# Draws the whole body from the export -- every joint and segment position,
# the skeleton linking them -- with the posed mesh on top and each driven
# segment's own axes as arrows. If the mesh sits on the skeleton's feet and
# its axes follow the ankle, the binding is right. This is the check to
# trust: it reads the same arrays the rest of the script works with, so
# nothing can be lost in a file format on the way out.
#
# Spyder's Run button (and %runfile) passes no command-line arguments, so the
# --plot flags never fire that way. These switches are what decide whether a
# figure appears when you just run the file.
SHOW_3D_VIEWER = True        # open the interactive frame-slider window
SAVE_3D_FIGURE = True        # also write a PNG grid next to the output .npz
PLOT_FRAMES = None           # frames for the PNG; None = 6 across the trial
VIEWER_START_FRAME = 0

# A movie of the whole trial -- the mesh moving with the skeleton, which is
# the check a still cannot give you: a binding can look right in one frame
# and swim, lag or flip halfway through a stride. Costs about a minute for a
# 500-frame trial, so raise VIDEO_STRIDE if you just want a quick look.
SAVE_3D_VIDEO = True
VIDEO_STRIDE = 2             # render every Nth frame
VIDEO_FPS = 25
VIDEO_DPI = 110
FIG_W_IN, FIG_H_IN = 9.0, 7.2
VIDEO_PROGRESS_SEC = 5       # seconds between progress lines
GIF_MEMORY_CAP_MB = 700      # gif only: thin the clip to stay under this
VIDEO_FORMAT = "auto"        # "auto" -> mp4 if ffmpeg is there, else gif
VIDEO_ZOOM = None            # None follows PLOT_ZOOM; "feet" crops to the mesh
VIDEO_FOLLOW = False         # re-centre on the mesh each frame (overground)
VIDEO_SPIN_DEG = 0.0         # total camera rotation across the clip

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
#  VISUAL3D METRICS READER  (copy of the one in build_foot_binding.py)
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
    """Nearest rotation to each 3x3 in a stack, missing frames left NaN.

    The NaN guard is not decoration: a trial with a gap in its
    `<Side>_Foot_Global_4x4` signal puts a NaN block into this stack, and
    np.linalg.svd raises "SVD did not converge" on it rather than returning
    NaN. Without this the whole run dies on one dropped frame.
    """
    R = np.asarray(R, dtype=float)
    flat = R.reshape(-1, *R.shape[-2:])
    out = np.full_like(flat, np.nan)
    ok = np.isfinite(flat).all(axis=(1, 2))
    if ok.any():
        U, _, Vt = np.linalg.svd(flat[ok])
        D = np.ones(U.shape[:-1])
        D[..., -1] = np.sign(np.linalg.det(U @ Vt))
        out[ok] = (U * D[..., None, :]) @ Vt
    return out.reshape(R.shape)


def rotation_gap_deg(A, B):
    """Angle between two stacks of rotations, in degrees."""
    tr = np.trace(np.einsum("fij,fkj->fik", A, B), axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def segment_blocks(seg_of):
    """Contiguous runs of one segment: [(segment, start, stop), ...].

    The meshes are laid out back to back, so this is normally one run per
    mesh and the pose can then be applied to a whole slice at once. If the
    numbering really is interleaved it degrades to more runs and still gives
    the right answer, just with less work done per call.
    """
    seg = np.asarray(seg_of).ravel()
    if not seg.size:
        return []
    cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    return [(int(seg[s]), int(s), int(e)) for s, e in
            zip(np.concatenate(([0], cut)), np.concatenate((cut, [seg.size])))]


def apply_poses(poses, v_local, blocks, dtype=np.float64):
    """(frames, segments, 4, 4) x (V, 3) -> (frames, V, 3) world positions.

    One batched matrix multiply per run rather than one per frame per
    segment. That is the whole job -- the rigid binding means there is no
    blending to do -- and it is about 6.5x faster than looping frames and
    writes into the output dtype directly instead of building float64 and
    converting, which halves the peak memory.

    Frames whose pose is missing come back NaN, and are given the identity
    before the multiply so a NaN never reaches BLAS.
    """
    v_local = np.asarray(v_local, dtype=dtype)
    out = np.empty((len(poses), len(v_local), 3), dtype=dtype)
    for k, s, e in blocks:
        P = poses[:, k]
        bad = ~np.isfinite(P).all(axis=(1, 2))
        R = P[:, :3, :3].astype(dtype, copy=True)
        t = P[:, :3, 3].astype(dtype, copy=True)
        if bad.any():
            R[bad] = np.eye(3, dtype=dtype)
            t[bad] = 0.0
        # (1, n, 3) @ (frames, 3, 3) -> (frames, n, 3), i.e. v @ R.T + t
        out[:, s:e] = v_local[s:e][None] @ R.transpose(0, 2, 1) + t[:, None, :]
        if bad.any():
            out[bad, s:e] = np.nan
    return out


def apply_poses_selected(poses, v_local, seg_of, idx, dtype=np.float64):
    """Same, for an arbitrary selection of global vertex indices."""
    idx = np.asarray(idx, dtype=np.intp).ravel()
    v_local = np.asarray(v_local, dtype=dtype)
    seg = np.asarray(seg_of).ravel()[idx]
    out = np.empty((len(poses), idx.size, 3), dtype=dtype)
    for k in np.unique(seg):
        col = np.flatnonzero(seg == k)
        P = poses[:, int(k)]
        bad = ~np.isfinite(P).all(axis=(1, 2))
        R = P[:, :3, :3].astype(dtype, copy=True)
        t = P[:, :3, 3].astype(dtype, copy=True)
        if bad.any():
            R[bad] = np.eye(3, dtype=dtype)
            t[bad] = 0.0
        block = v_local[idx[col]][None] @ R.transpose(0, 2, 1) + t[:, None, :]
        if bad.any():
            block[bad] = np.nan
        out[:, col] = block
    return out


# %%==========================================================================
#  3D CHECK FIGURE
# ============================================================================

def _mesh_points(z, poses, frame, stride=None):
    """Posed mesh vertices for one frame, subsampled. -> (M,3) or None."""
    stride = MESH_POINT_STRIDE if stride is None else stride
    v_local, seg_of = z["vertices_local"], z["vertex_segment"]
    idx = np.arange(0, len(v_local), max(1, stride))
    return apply_poses_selected(poses[frame:frame + 1], v_local, seg_of, idx)[0]


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


def _ffmpeg_works(exe, timeout=10):
    """True if the binary answers `-version`. Guards against a broken or
    hung ffmpeg, which otherwise stalls the first grab_frame() silently."""
    import subprocess
    try:
        r = subprocess.run([exe, "-version"], capture_output=True,
                           timeout=timeout)
        return r.returncode == 0
    except Exception:                                   # noqa: BLE001
        return False


def _video_writer(fmt, fps):
    """Pick a writer. mp4 needs ffmpeg; gif only needs Pillow, so it always
    works. Returns (writer, extension, how it was found)."""
    from matplotlib import animation, rcParams

    if fmt in ("auto", "mp4"):
        exe = rcParams.get("animation.ffmpeg_path", "ffmpeg")
        how = "ffmpeg on PATH"
        if not (exe and (os.path.isfile(exe) or shutil.which(exe))):
            exe = None
            try:                       # pip install imageio-ffmpeg
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
                how = "imageio-ffmpeg"
            except Exception:          # noqa: BLE001
                pass
        if exe and not _ffmpeg_works(exe):
            # A binary that is present but broken blocks on the first frame
            # rather than raising, which looks exactly like a slow render.
            # Better to find out now, in one second, than at 0% forever.
            print(f"  [{exe} did not answer --version; ignoring it]")
            exe = None
        if exe:
            rcParams["animation.ffmpeg_path"] = exe
            return animation.FFMpegWriter(fps=fps, bitrate=2400), ".mp4", how
        if fmt == "mp4":
            raise RuntimeError(
                "no ffmpeg found -- pip install imageio-ffmpeg, or set "
                "VIDEO_FORMAT = 'gif'")

    return animation.PillowWriter(fps=fps), ".gif", "Pillow"


def save_3d_video(result, path=None, stride=None, fps=None, zoom=None,
                  follow=None, spin_deg=None, verbose=True):
    """The trial as a movie: mesh and skeleton together, frame by frame.

    A still says the binding is right in that pose. Only the movie says it
    stays right -- a mesh that swims against the foot, lags the skeleton, or
    flips at midstance shows up here and nowhere else.

    Writes .mp4 when ffmpeg is available and .gif otherwise; `pip install
    imageio-ffmpeg` is the easiest way to get mp4 with no system install.

    Rendering goes through a bare Agg figure, never pyplot. A pyplot figure
    under a GUI backend puts a real window behind every frame and each draw
    then pays the GUI's costs; this way the interactive backend you need for
    show_3d does not slow the movie down.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    stride = VIDEO_STRIDE if stride is None else stride
    fps = VIDEO_FPS if fps is None else fps
    follow = VIDEO_FOLLOW if follow is None else follow
    spin_deg = VIDEO_SPIN_DEG if spin_deg is None else spin_deg
    zoom = (VIDEO_ZOOM or PLOT_ZOOM) if zoom is None else zoom

    vectors, z, poses, meta = (result["vectors"], result["binding_npz"],
                               result["poses"], result["meta"])
    stride = max(1, int(stride))
    if not len(poses):
        raise ValueError("no frames to render")
    ctr, r = _frame_extent(vectors, z, poses, zoom)

    writer, ext, how = _video_writer(VIDEO_FORMAT, fps)
    path = path or (os.path.splitext(result["out"])[0]
                    + ("_feet" if zoom == "feet" else "") + ext)

    # Pillow's gif writer keeps every rendered frame in memory and only
    # encodes at the end, so a long trial can ask for gigabytes and spend the
    # whole time swapping. Thin the clip until it fits instead of wedging.
    asked = stride
    if ext == ".gif":
        mb = FIG_W_IN * VIDEO_DPI * FIG_H_IN * VIDEO_DPI * 4 / 1e6   # per frame
        while (len(range(0, len(poses), stride)) * mb > GIF_MEMORY_CAP_MB
               and stride < len(poses)):
            stride += 1

    frames = list(range(0, len(poses), stride))
    if ext == ".gif" and verbose:
        print(f"  gif writer: ffmpeg was not found, and Pillow holds every "
              f"frame in memory")
        print(f"       until the end -- ~{len(frames) * mb:.0f} MB for this "
              f"clip at stride {stride}.")
        if stride != asked:
            print(f"       Raised the stride from {asked} to {stride} to stay "
                  f"under {GIF_MEMORY_CAP_MB:.0f} MB.")
        print(f"       `pip install imageio-ffmpeg` gets you mp4 instead: "
              f"faster, smaller, no ceiling.")

    fig = Figure(figsize=(FIG_W_IN, FIG_H_IN))
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.02, 0.06, 0.96, 0.90), projection="3d")
    _legend(fig)
    azim0 = PLOT_AZIM

    if verbose:
        print(f"  rendering {len(frames)} frames -> {os.path.basename(path)}"
              f"  [{how}]", flush=True)

    t0 = time.perf_counter()
    last = [t0]
    with writer.saving(fig, path, VIDEO_DPI):
        for n, f in enumerate(frames):
            c = ctr
            if follow:
                mp = _mesh_points(z, poses, f, stride=40)
                if mp is not None:
                    mp = mp[np.isfinite(mp).all(axis=1)]
                    if len(mp):
                        c = (mp.min(0) + mp.max(0)) / 2
            draw_3d_frame(ax, vectors, z, poses, f, meta, c, r,
                          title=f"{result['trial']}   frame {f} "
                                f"of {len(poses) - 1}")
            if spin_deg:
                ax.view_init(elev=PLOT_ELEV,
                             azim=azim0 + spin_deg * n / len(frames))
            writer.grab_frame()

            # Progress on a clock, not on a frame count: the first frame tells
            # you the rate, then a line every few seconds tells you it is still
            # moving. A percentage that only updates 8 times looks identical to
            # a hang when each step takes minutes.
            now = time.perf_counter()
            if verbose and (n == 0 or now - last[0] >= VIDEO_PROGRESS_SEC
                            or n == len(frames) - 1):
                per = (now - t0) / (n + 1)
                eta = per * (len(frames) - n - 1)
                print(f"    {n + 1:5d}/{len(frames)}  "
                      f"{100 * (n + 1) // len(frames):3d}%  "
                      f"{per:.2f} s/frame  eta {eta:4.0f} s", flush=True)
                last[0] = now

    if verbose:
        print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB, "
              f"{len(frames) / fps:.1f} s clip, "
              f"{time.perf_counter() - t0:.0f} s to render)")
    return path


def show_3d(result, frame=0, zoom=None):
    """Interactive viewer with a frame slider. Needs a GUI backend.

    In Spyder: Preferences > IPython console > Graphics > Backend: Automatic.
    With the inline backend there is no slider, so use save_3d_figure instead.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    backend = matplotlib.get_backend()
    if "inline" in backend.lower() or backend.lower() == "agg":
        print(f"  [backend is {backend}: the frame slider will not respond."
              f" Spyder: Preferences > IPython console > Graphics >"
              f" Backend: Automatic, then restart the kernel]")

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

def posed_path(trial_csv=None, out_dir=None):
    """Where apply_binding() writes, and vertex_tracks() looks for, the .npz.

    One place, so the writer and the readers cannot disagree about it.
    """
    trial_csv = TRIAL_METRICS_CSV if trial_csv is None else trial_csv
    out_dir = OUT_DIR if out_dir is None else out_dir
    stem = os.path.splitext(os.path.basename(trial_csv))[0]
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(trial_csv))
    return os.path.join(out_dir, stem + OUT_SUFFIX + ".npz")


def load_binding(path=None):
    path = BINDING_FILE if path is None else path
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run build_foot_binding.py")
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["meta"])), z


def apply_binding(trial_csv=None, binding_file=None, save_vertices=None,
                  verbose=None):
    trial_csv = TRIAL_METRICS_CSV if trial_csv is None else trial_csv
    save_vertices = SAVE_VERTICES if save_vertices is None else save_vertices
    verbose = VERBOSE if verbose is None else verbose

    meta, z = load_binding(binding_file)
    v_local = z["vertices_local"]
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

        # The whole 4x4, not just [3, 3]: the branch above forces row 3 to
        # (0, 0, 0, 1) on every frame, so [3, 3] is finite even where the
        # pose is missing and counting it reports a full trial every time.
        st = dict(segment=s, source=used,
                  n_good=int(np.isfinite(poses[:, k]).all(axis=(1, 2)).sum()))
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

    verts = None
    if save_vertices:
        verts = apply_poses(poses, v_local, segment_blocks(seg_of),
                            dtype=np.float32)

    out_npz = posed_path(trial_csv)
    # The binding's mesh entries carry the entire .obj file text -- `lines`
    # plus the v/vn line numbers -- so the builder can rewrite the meshes.
    # Nothing downstream of here reads it: vertex_tracks() wants only
    # name/start/count, and export_posed_fbx.py reads the binding file
    # itself. It was 1.5 MB in every trial's .npz, an order of magnitude
    # more than the poses it travelled with, so it does not get copied.
    slim = dict(meta)
    slim["meshes"] = [{k: v for k, v in m.items()
                       if k not in ("lines", "v_at", "n_at")}
                      for m in meta["meshes"]]
    payload = dict(poses=poses, segments=np.array(seg_names),
                   frames=np.array(items), vertex_segment=seg_of,
                   meta=np.array(json.dumps(dict(
                       trial=os.path.basename(trial_csv), binding=slim,
                       stats=stats))))
    if save_vertices:
        payload["vertices"] = verts
    np.savez_compressed(out_npz, **payload)

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
            print("    vertices       not stored -- they are exactly")
            print("                   R_seg(t) @ v_local + t_seg(t), so "
                  "rebuild what you need:")
            print(f"                   ab.vertex_tracks({out_npz!r}"
                  ", mesh='left')")
        print("-" * 74)
    return dict(poses=poses, vertices=verts, stats=stats,
                frames=items, out=out_npz, vectors=vectors, matrices=matrices,
                binding_npz=z, meta=meta,
                trial=os.path.basename(trial_csv),
                trial_path=os.path.abspath(trial_csv))


def vertices_at_frame(z, poses, frame):
    """World vertices for one frame from the compact `poses` array."""
    v_local, seg_of = z["vertices_local"], z["vertex_segment"]
    return apply_poses(poses[frame:frame + 1], v_local,
                       segment_blocks(seg_of))[0]


def vertex_tracks(source=None, vertices=None, mesh=None, binding=None,
                  dtype=np.float64):
    """Vertex positions over a whole trial, in Theia world metres.

    This is the entry point for analysis. The full array is (frames, 10208, 3)
    -- 368 MB for a 3000-frame trial -- so select what you actually need and
    it stays small: one vertex over 484 frames is 12 kB, and costs 0.5 ms.

        import apply_binding as ab
        V, idx = ab.vertex_tracks("D05_C01_SKS_metrics_posed.npz")
        V, idx = ab.vertex_tracks(res, mesh="left")          # one foot
        V, idx = ab.vertex_tracks(res, vertices=[0, 1500])   # two vertices

    `source` is either the dict apply_binding() returned or the path to a
    _posed.npz (default: TRIAL_METRICS_CSV's). `mesh` matches a name from the
    binding -- "left", "right", "both", or the full .obj filename. `vertices`
    is explicit global indices and wins over `mesh`.

    Returns (V, idx): V is (frames, len(idx), 3) in `dtype`, NaN on frames
    whose pose was missing; idx is the global vertex index of each column, so
    you can carry a selection between trials and know what you are looking at.

    `dtype=np.float32` halves the memory and is still ~0.1 um at metre scale,
    which is four orders below the binding's own accuracy.
    """
    if isinstance(source, dict):                       # apply_binding() result
        poses, z = source["poses"], source["binding_npz"]
    else:
        if source is None:
            source = posed_path()
        posed = np.load(source, allow_pickle=False)
        poses = posed["poses"]
        _, z = load_binding(binding)

    v_local, seg_of = z["vertices_local"], z["vertex_segment"]
    meta = json.loads(str(z["meta"]))

    if vertices is not None:
        idx = np.asarray(vertices, dtype=np.intp).ravel()
    elif mesh is not None:
        hits = [m for m in meta["meshes"]
                if mesh == m["name"] or mesh.lower() in m["name"].lower()]
        if len(hits) != 1:
            raise ValueError(
                f"mesh={mesh!r} matched {[m['name'] for m in hits]}; "
                f"available: {[m['name'] for m in meta['meshes']]}")
        idx = np.arange(hits[0]["start"], hits[0]["start"] + hits[0]["count"])
    else:
        idx = np.arange(len(v_local))

    if idx.size and (idx.min() < 0 or idx.max() >= len(v_local)):
        raise IndexError(f"vertex index out of range 0..{len(v_local) - 1}")

    if idx.size == len(v_local) and np.array_equal(idx, np.arange(len(v_local))):
        out = apply_poses(poses, v_local, segment_blocks(seg_of), dtype=dtype)
    else:
        out = apply_poses_selected(poses, v_local, seg_of, idx, dtype=dtype)
    return out, idx


def mesh_dataframe(source=None, binding=None, dtype=None):
    """Every vertex position as a DataFrame: index = frame, columns a
    MultiIndex of (segment, vertex, axis).

    Built in memory on demand -- it is not written to disk, because it is a
    view of `poses` and nothing more. Pass `source` a _posed.npz path or the
    dict apply_binding() returned.

        df = ab.mesh_dataframe("TRIAL_posed.npz")
        left  = df["left_foot"]                  # (frames, vertices*3)
        v9000 = df[("right_foot", 9000)]         # x, y, z of one vertex
        xs    = df.xs("X", axis=1, level="axis")

    Positions are Theia world metres. `df.attrs` carries the trial name, the
    units, the coordinate system and the binding's provenance, so the file
    says what it is without the script that wrote it.
    """
    try:
        import pandas as pd
    except ImportError as exc:                          # noqa: BLE001
        raise ImportError(
            "mesh_dataframe needs pandas -- pip install pandas, or use "
            "vertex_tracks() which is numpy only") from exc

    dtype = H5_DTYPE if dtype is None else dtype
    V, idx = vertex_tracks(source, binding=binding, dtype=np.dtype(dtype))

    if isinstance(source, dict):
        z, trial = source["binding_npz"], source["trial"]
    else:
        _, z = load_binding(binding)
        trial = os.path.basename(str(source)) if source else TRIAL_METRICS_CSV
    meta = json.loads(str(z["meta"]))
    seg_of, seg_names = z["vertex_segment"], meta["segments"]

    cols = pd.MultiIndex.from_tuples(
        [(seg_names[seg_of[v]], int(v), a) for v in idx for a in "XYZ"],
        names=["segment", "vertex", "axis"])
    df = pd.DataFrame(V.reshape(len(V), -1), columns=cols)
    # Sort the columns so the MultiIndex is lexsorted: the global vertex
    # numbering interleaves the two feet, and without this every
    # df[("right_foot", 9000)] lookup warns and scans.
    df = df.sort_index(axis=1)
    df.index.name = "frame"
    df.attrs = {"trial": trial, "units": "m",
                "coordinate_system": meta.get("coordinate_system", "theia"),
                "segments": list(seg_names),
                "static_metrics_csv": meta.get("static_metrics_csv"),
                "pose_signal": meta.get("pose_signal"),
                "created_by": "apply_binding.py"}
    return df


def save_mesh_h5(source=None, trial_csv=None, path=None, binding=None,
                 dtype=None, compress=None, chunk_frames=256, verbose=True):
    """Write every vertex position beside the trial CSV, as HDF5.

    Only worth doing to hand the array to something that cannot rebuild it
    from `poses`. It is ~1500x the size of the _posed.npz for exactly the
    same information, so reach for vertex_tracks() first.

    HDF5 rather than a pickle for four reasons that all bite in practice:

      * A pickle is executable. Loading one runs whatever is inside it, so it
        is not something to accept from anyone else or to archive for years.
      * A pickle is Python-and-pandas-only, and a pandas pickle is tied to
        the version that wrote it. MATLAB, R and Visual3D all read HDF5.
      * HDF5 is chunked, so `f["vertices"][:, 9000, :]` reads one vertex
        track in ~3 ms without touching the other 368 MB. `pd.read_pickle`
        has to deserialise the whole file to give you one column.
      * The units, the coordinate frame and the binding's provenance ride in
        the file as attributes, so it says what it is on its own.

    Written in frame blocks, so peak memory is the block and not the trial.
    """
    import h5py

    dt = np.dtype(H5_DTYPE if dtype is None else dtype)
    compress = H5_COMPRESS if compress is None else compress

    if isinstance(source, dict):
        poses, z = source["poses"], source["binding_npz"]
        trial = source.get("trial")
        src_path = trial_csv or source.get("trial_path") or TRIAL_METRICS_CSV
    else:
        if source is None:
            source = posed_path()
        poses = np.load(source, allow_pickle=False)["poses"]
        _, z = load_binding(binding)
        trial = os.path.basename(str(source))
        src_path = trial_csv or TRIAL_METRICS_CSV

    v_local, seg_of = z["vertices_local"], z["vertex_segment"]
    meta = json.loads(str(z["meta"]))
    blocks = segment_blocks(seg_of)
    n_f, n_v = len(poses), len(v_local)

    if path is None:
        d = os.path.dirname(os.path.abspath(src_path))
        stem = os.path.splitext(os.path.basename(src_path))[0]
        path = os.path.join(d, stem + H5_SUFFIX)

    kw = dict(chunks=(min(chunk_frames, n_f), min(64, n_v), 3))
    if compress:
        kw.update(compression=compress, shuffle=True)
        if compress == "gzip":
            kw["compression_opts"] = 4

    with h5py.File(path, "w") as fh:
        d = fh.create_dataset("vertices", shape=(n_f, n_v, 3), dtype=dt, **kw)
        for i0 in range(0, n_f, chunk_frames):
            # float64 for the multiply, then one rounding on the way to disk.
            # The block bounds the cost, so this is the accurate order for
            # free rather than a tradeoff.
            sl = slice(i0, min(i0 + chunk_frames, n_f))
            d[sl] = apply_poses(poses[sl], v_local, blocks,
                                dtype=np.float64).astype(dt, copy=False)
        d.attrs["units"] = "m"
        d.attrs["axes"] = "frame, vertex, xyz"
        d.attrs["coordinate_system"] = meta.get("coordinate_system", "theia")
        d.attrs["trial"] = str(trial)
        d.attrs["created_by"] = "apply_binding.py"
        fh.create_dataset("vertex_segment", data=np.asarray(seg_of))
        fh.create_dataset("segments",
                          data=np.array(meta["segments"], dtype="S"))
        fh.attrs["binding"] = json.dumps(meta)

    if verbose:
        print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB, "
              f"{n_f} frames x {n_v} vertices, {dt.name}"
              f"{', ' + compress if compress else ''})")
    return path


def load_mesh_h5(trial_csv, vertices=None, frames=None):
    """The other half: hand it the trial CSV path, get the array back.

        V = ab.load_mesh_h5("D05_C01_SKS_metrics.csv")
        V = ab.load_mesh_h5(trial, vertices=[9000])   # one track, no full read

    `vertices` and `frames` are read straight out of the chunked dataset, so
    a selection never pays for the rest of the file.
    """
    import h5py

    d = os.path.dirname(os.path.abspath(trial_csv))
    stem = os.path.splitext(os.path.basename(trial_csv))[0]
    path = os.path.join(d, stem + H5_SUFFIX)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run apply_binding on "
            f"{os.path.basename(trial_csv)} with SAVE_MESH_H5 = True, or "
            f"rebuild from the _posed.npz with vertex_tracks()")
    with h5py.File(path, "r") as fh:
        d = fh["vertices"]
        fsel = slice(None) if frames is None else frames
        if vertices is None:
            return d[fsel]
        return d[fsel, np.asarray(vertices, dtype=np.intp).ravel(), :]


def _plot_frames_arg(argv):
    """--plot-frames 0,120,240  or  --plot-frames=0,120,240

    Returns (frames, consumed) where `consumed` is the index of the value
    argument, if the value was passed separately. That index has to come out
    of the positional list, or `0,120,240` gets taken for the trial file.
    """
    for i, a in enumerate(argv):
        if not a.startswith("--plot-frames"):
            continue
        spec, consumed = (a.split("=", 1)[1] if "=" in a else ""), None
        if not spec and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            spec, consumed = argv[i + 1], i + 1
        return [int(x) for x in spec.replace(",", " ").split()
                if x.lstrip("-").isdigit()], consumed
    return None, None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cli_frames, consumed = _plot_frames_arg(argv)
    files = [a for i, a in enumerate(argv)
             if not a.startswith("--") and i != consumed]
    res = apply_binding(files[0] if files else None,
                        save_vertices="--vertices" in argv)

    if SAVE_MESH_H5 or "--h5" in argv:
        try:
            save_mesh_h5(res)
        except Exception as exc:                        # noqa: BLE001
            print(f"  [mesh HDF5 skipped: {type(exc).__name__}: {exc}]")

    # CONFIG decides by default; flags override it. Running the file from
    # Spyder hands us an empty argv, which is exactly the case where the
    # CONFIG switches have to be the ones in charge.
    frames = cli_frames if cli_frames is not None else PLOT_FRAMES
    save_png = SAVE_3D_FIGURE or cli_frames is not None
    viewer = SHOW_3D_VIEWER or "--plot" in argv
    video = SAVE_3D_VIDEO or "--video" in argv
    if "--no-plot" in argv:
        save_png = viewer = video = False
    elif cli_frames is not None and "--plot" not in argv:
        viewer = False            # --plot-frames on its own means the PNG
    if "--no-video" in argv:
        video = False
    if "--video" in argv:         # asking for the movie means only the movie
        save_png = viewer = False
    zoom = "feet" if "--feet" in argv else None

    for want, fn, what in ((save_png, lambda: save_3d_figure(res, frames,
                                                            zoom=zoom), "PNG"),
                           (video, lambda: save_3d_video(res, zoom=zoom),
                            "video"),
                           (viewer, lambda: show_3d(res, VIEWER_START_FRAME,
                                                    zoom), "viewer")):
        if not want:
            continue
        try:
            fn()
        except Exception as exc:                        # noqa: BLE001
            print(f"  [3D {what} skipped: {type(exc).__name__}: {exc}]")
    return res


if __name__ == "__main__":
    RESULT = main()
