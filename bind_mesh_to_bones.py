# -*- coding: utf-8 -*-
"""
=============================================================================
 STEP 1 of 2 -- bind the foot meshes to their Theia segments
=============================================================================

Takes the scan-derived foot meshes (already in Theia coordinates, from
transform_obj_to_theia.py) plus a STATIC-POSE Visual3D metrics export, works
out where each mesh sits on each foot segment, and stores that relationship
in one .npz file.

`apply_binding.py` then replays it over any trial to get vertex positions.

WHAT THE RELATIONSHIP IS
------------------------
Three Visual3D signals per foot -- Ankle_Position, Foot_Position and
Toes_Position -- form a rigid triangle that tracks the foot segment. They
are all derived from the same segment pose, so they move with the bone
exactly: on D05_C01 their pairwise distances hold to +-0.01 mm across every
frame, and the rotation recovered from them matches the segment orientation
in Theia's own FBX to 0.01 deg mean, 0.03 deg max.

So the binding is: express every mesh vertex in a frame built from those
three landmarks. Replaying is then a Kabsch fit of the stored landmarks onto
the trial's landmarks, once per frame.

Note that Foot_Position is NOT on the ankle-to-toes line -- it sits 25.15 mm
off it, constant to 0.02 mm -- which is exactly why three points are enough.
Three collinear points would fix only 5 of the 6 degrees of freedom and leave
inversion/eversion undetermined.

WHICH MESH GOES WITH WHICH SEGMENT
----------------------------------
Assigned per VERTEX, not per file, by proximity to each segment's landmarks.
That is what makes `both_feet.obj` work: its left and right halves bind to
different segments and move independently, which a per-file assignment would
get wrong.

RIGIDITY
--------
Each mesh region is bound rigidly to one segment. The foot's arch does not
flatten and the toes do not flex -- Theia has no toe orientation to drive
them with. The report says how many vertices sit distal to the MTP joint so
you can see what that affects. If you need toe motion, export more landmarks
from Visual3D (see LANDMARKS below) and add a toes segment.

Needs numpy. Run in Spyder with F5, or: python bind_mesh_to_bones.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

import numpy as np


# %%==========================================================================
#  CONFIG
# ============================================================================

# Visual3D metrics export of the STATIC trial -- the pose the meshes are in.
STATIC_METRICS_CSV = "D05_C01_a_pose_metrics.csv"

# Meshes to bind, in THEIA coordinates (run transform_obj_to_theia.py first).
MESH_FILES = ["left_feet_theia.obj", "right_feet_theia.obj",
              "both_feet_theia.obj"]

# One entry per rigid segment. The landmarks must be Visual3D signal names
# present in the export, and there must be at least three non-collinear ones.
# Adding more (if you export them) improves the fit and costs nothing here.
SEGMENTS = {
    "left_foot":  ["Left_Ankle_Position", "Left_Foot_Position",
                   "Left_Toes_Position"],
    "right_foot": ["Right_Ankle_Position", "Right_Foot_Position",
                   "Right_Toes_Position"],
}

# Which landmark marks the MTP joint, per segment -- used only to report how
# many vertices are distal to it and therefore cannot flex.
MTP_LANDMARK = {"left_foot": "Left_Toes_Position",
                "right_foot": "Right_Toes_Position"}

OUT_FILE = "foot_mesh_binding.npz"
VERBOSE = True

# A static trial's first and last frames are usually filter edge transients.
# On D05_C01 they throw the landmarks 20-40 mm off, and averaging them into
# the reference pose drags the whole binding with them. Frames whose landmark
# configuration sits further than this from the median one are dropped.
FRAME_TOLERANCE_MM = 8.0


# %%==========================================================================
#  VISUAL3D METRICS READER
# ============================================================================
# Five header rows: file path, signal name, signal type, ORIGINAL, component.
# Then one row per frame, with the frame number in column 0. Visual3D pads the
# block with all-NaN rows, which are dropped.
#
# NOTE: apply_binding.py carries its own copy of this reader so that it stays
# standalone. Fix bugs in both.

def read_metrics(path):
    """Visual3D tab-delimited export -> {signal: (frames, 3)} plus frame ids."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh]
    if len(rows) < 6:
        raise ValueError(f"{path}: too short to be a Visual3D metrics export")

    names, comps = rows[1][1:], rows[4][1:]
    body = rows[5:]
    width = len(names)

    vals, items = [], []
    for r in body:
        cells = r[1:]
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        row = [float(c) if c.strip() else np.nan for c in cells[:width]]
        if np.all(np.isnan(row)):
            continue                      # Visual3D's trailing padding
        vals.append(row)
        items.append(r[0].strip())
    if not vals:
        raise ValueError(f"{path}: no frames with data")
    data = np.array(vals, float)

    signals = {}
    for i, (n, c) in enumerate(zip(names, comps)):
        signals.setdefault(n, {})[c.strip().upper()] = data[:, i]
    out = {}
    for n, ch in signals.items():
        if {"X", "Y", "Z"} <= set(ch):
            out[n] = np.stack([ch["X"], ch["Y"], ch["Z"]], axis=1)
    return out, items


def landmark_block(signals, names):
    """Stack named signals into (frames, n_landmarks, 3)."""
    missing = [n for n in names if n not in signals]
    if missing:
        raise KeyError(f"signals not in the export: {missing}. "
                       f"Available: {sorted(signals)[:8]} ...")
    return np.stack([signals[n] for n in names], axis=1)


# %%==========================================================================
#  OBJ READER
# ============================================================================

def read_obj(path):
    """-> (lines, vertices (N,3), normals (M,3), v_line_idx, n_line_idx)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    verts, norms, v_at, n_at = [], [], [], []
    for i, line in enumerate(lines):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
            v_at.append(i)
        elif line.startswith("vn "):
            norms.append([float(x) for x in line.split()[1:4]])
            n_at.append(i)
    if not verts:
        raise ValueError(f"{path}: no vertices")
    return (lines, np.array(verts, float),
            np.array(norms, float) if norms else np.zeros((0, 3)), v_at, n_at)


# %%==========================================================================
#  THE SEGMENT FRAME
# ============================================================================

def segment_frame(L):
    """Orthonormal frame from a segment's landmarks. L is (n_landmarks, 3).

    Origin at the first landmark (the joint centre), long axis towards the
    last, third axis from whatever is left over. Any consistent convention
    would do -- replaying uses Kabsch on the landmarks themselves, so the
    world positions do not depend on this choice. It exists so the stored
    binding is something you can read and reason about.
    """
    origin = L[0]
    e1 = L[-1] - origin
    e1 = e1 / np.linalg.norm(e1)
    off = L[1] - origin
    e2 = off - (off @ e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-6:
        raise ValueError("segment landmarks are collinear: they fix only 5 of "
                         "6 degrees of freedom, leaving the roll about the "
                         "long axis undetermined. Export a landmark off that "
                         "axis.")
    e2 = e2 / n2
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1), origin      # (R, origin)


def frame_quality(L):
    """How well-conditioned is this landmark set? Returns a dict."""
    X = L - L.mean(0)
    sv = np.linalg.svd(X, compute_uv=False)
    sv = np.pad(sv, (0, max(0, 3 - len(sv))))[:3]
    e1 = L[-1] - L[0]
    e1 = e1 / np.linalg.norm(e1)
    off = L[1] - L[0]
    lever = float(np.linalg.norm(off - (off @ e1) * e1))
    # a 1 mm landmark error tilts the frame about the long axis by ~1/lever
    roll_per_mm = float(np.degrees(np.arctan2(0.001, max(lever, 1e-9))))
    return dict(singular_values_mm=(sv * 1000).tolist(),
                off_axis_lever_mm=lever * 1000,
                roll_deg_per_mm_noise=roll_per_mm,
                spans_3d=bool(sv[2] * 1000 > 0.5 or lever * 1000 > 1.0))


# %%==========================================================================
#  BINDING
# ============================================================================

def good_frames(blocks):
    """Drop static frames whose landmarks sit far from the median pose.

    Uses every segment at once, so one shared verdict covers them all and the
    reference poses stay mutually consistent.
    """
    allblk = np.concatenate(list(blocks.values()), axis=1)   # (F, sum K, 3)
    med = np.median(allblk, axis=0)
    dev = np.linalg.norm(allblk - med, axis=2).mean(axis=1)
    keep = np.where(dev <= FRAME_TOLERANCE_MM / 1000.0)[0]
    if len(keep) == 0:                       # never leave ourselves nothing
        keep = np.arange(len(allblk))
    dropped = [(int(i), float(dev[i] * 1000))
               for i in range(len(allblk)) if i not in keep]
    return keep, dropped, dev * 1000


def bind(static_csv=None, mesh_files=None, verbose=None):
    """Build the binding. Returns the dict that gets saved."""
    static_csv = STATIC_METRICS_CSV if static_csv is None else static_csv
    mesh_files = MESH_FILES if mesh_files is None else mesh_files
    verbose = VERBOSE if verbose is None else verbose

    signals, items = read_metrics(static_csv)
    seg_names = list(SEGMENTS)

    # --- reference landmark positions, over the usable static frames -----
    blocks = {s: landmark_block(signals, SEGMENTS[s]) for s in seg_names}
    keep, dropped, dev_mm = good_frames(blocks)

    ref_world, quality, per_frame = {}, {}, {}
    for s in seg_names:
        blk = blocks[s][keep]                              # (F, K, 3)
        per_frame[s] = blk
        ref_world[s] = blk.mean(axis=0)
        quality[s] = frame_quality(ref_world[s])

    # --- rigidity: do the landmarks actually move as one body? -----------
    rigid = {}
    for s in seg_names:
        blk = per_frame[s]
        K = blk.shape[1]
        sd = []
        for i in range(K):
            for j in range(i + 1, K):
                sd.append(np.linalg.norm(blk[:, i] - blk[:, j], axis=1).std())
        rigid[s] = float(np.max(sd) * 1000) if sd else 0.0

    # --- meshes -----------------------------------------------------------
    meshes, all_v, all_n, all_seg = [], [], [], []
    cursor = 0
    for path in mesh_files:
        if not os.path.exists(path):
            print(f"  !! {path}: not found, skipped")
            continue
        lines, V, N, v_at, n_at = read_obj(path)

        # assign every vertex to its nearest segment
        cen = np.stack([ref_world[s].mean(0) for s in seg_names])
        d = np.linalg.norm(V[:, None, :] - cen[None, :, :], axis=2)
        seg_idx = d.argmin(axis=1)

        meshes.append(dict(name=os.path.basename(path), start=cursor,
                           count=len(V), lines=lines, v_at=v_at, n_at=n_at,
                           has_normals=bool(len(N))))
        all_v.append(V)
        all_n.append(N if len(N) == len(V) else np.zeros((len(V), 3)))
        all_seg.append(seg_idx)
        cursor += len(V)

    if not meshes:
        raise ValueError("no meshes were read")
    V = np.vstack(all_v)
    Nrm = np.vstack(all_n)
    seg_of = np.concatenate(all_seg)

    # --- express vertices, normals and landmarks in segment-local frames --
    v_local = np.zeros_like(V)
    n_local = np.zeros_like(Nrm)
    lm_local = np.zeros((len(seg_names), max(len(SEGMENTS[s]) for s in seg_names), 3))
    for k, s in enumerate(seg_names):
        R, origin = segment_frame(ref_world[s])
        m = seg_of == k
        v_local[m] = (V[m] - origin) @ R
        n_local[m] = Nrm[m] @ R
        lm_local[k, :len(SEGMENTS[s])] = (ref_world[s] - origin) @ R

    # --- diagnostics ------------------------------------------------------
    report = dict(segments=seg_names, quality=quality, rigidity_mm=rigid,
                  n_frames=len(items), n_used=int(len(keep)),
                  dropped_frames=dropped,
                  frame_deviation_mm=[round(float(d), 2) for d in dev_mm])
    report["vertices_per_segment"] = {s: int((seg_of == k).sum())
                                      for k, s in enumerate(seg_names)}
    # how close does each segment's joint centre sit to its own mesh?
    report["ankle_inside_mesh_mm"] = {}
    report["distal_to_mtp"] = {}
    for k, s in enumerate(seg_names):
        m = seg_of == k
        if not m.any():
            continue
        ank = ref_world[s][0]
        report["ankle_inside_mesh_mm"][s] = float(
            np.linalg.norm(V[m] - ank, axis=1).min() * 1000)
        mtp_name = MTP_LANDMARK.get(s)
        if mtp_name in SEGMENTS[s]:
            mtp = ref_world[s][SEGMENTS[s].index(mtp_name)]
            axis = mtp - ank
            axis = axis / np.linalg.norm(axis)
            beyond = ((V[m] - mtp) @ axis) > 0
            report["distal_to_mtp"][s] = [int(beyond.sum()), int(m.sum())]

    binding = dict(
        meta=dict(
            format_version=1,
            created=str(date.today()),
            static_metrics_csv=os.path.basename(static_csv),
            meshes=[dict(name=m["name"], start=m["start"], count=m["count"],
                         has_normals=m["has_normals"], lines=m["lines"],
                         v_at=m["v_at"], n_at=m["n_at"]) for m in meshes],
            segments=seg_names,
            landmark_names={s: SEGMENTS[s] for s in seg_names},
            landmark_counts={s: len(SEGMENTS[s]) for s in seg_names},
            units="metres",
            coordinate_system="Theia3D global (Z-up), same as the metrics export",
            report=report),
        vertices_local=v_local, normals_local=n_local,
        vertex_segment=seg_of.astype(np.int16),
        landmarks_local=lm_local,
        reference_landmarks_world=np.stack(
            [np.pad(ref_world[s], ((0, lm_local.shape[1] - len(SEGMENTS[s])), (0, 0)))
             for s in seg_names]),
    )
    if verbose:
        print_report(binding)
    return binding


def save_binding(binding, path=None):
    path = OUT_FILE if path is None else path
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(binding["meta"])),
        vertices_local=binding["vertices_local"],
        normals_local=binding["normals_local"],
        vertex_segment=binding["vertex_segment"],
        landmarks_local=binding["landmarks_local"],
        reference_landmarks_world=binding["reference_landmarks_world"])
    return path


# %%==========================================================================
#  REPORT
# ============================================================================

def print_report(b):
    W = 74
    meta, rep = b["meta"], b["meta"]["report"]
    print("=" * W)
    print("  BIND FOOT MESHES TO THEIA SEGMENTS")
    print("=" * W)
    print(f"\n  static pose: {meta['static_metrics_csv']}  "
          f"({rep['n_frames']} frames with data, {rep['n_used']} used)")
    for i, d in rep["dropped_frames"]:
        print(f"      dropped frame {i}: landmarks {d:.1f} mm from the median "
              f"pose (filter edge transient)")
    print(f"      per-frame spread about the median: "
          f"{', '.join('%.1f' % d for d in rep['frame_deviation_mm'])} mm")
    print(f"  meshes:      {', '.join(m['name'] for m in meta['meshes'])}")
    print(f"  total vertices: {len(b['vertices_local'])}")

    print("\n" + "-" * W)
    print("  CAN THESE LANDMARKS DEFINE A POSE?")
    print("-" * W)
    for s in meta["segments"]:
        q = rep["quality"][s]
        sv = q["singular_values_mm"]
        print(f"\n    {s}   ({', '.join(meta['landmark_names'][s])})")
        print(f"      spread of the landmarks   "
              f"[{sv[0]:7.2f} {sv[1]:6.2f} {sv[2]:5.2f}] mm")
        print(f"      off-axis lever arm        {q['off_axis_lever_mm']:7.2f} mm"
              "   <- this is what fixes the roll")
        print(f"      roll error per mm of landmark noise  "
              f"{q['roll_deg_per_mm_noise']:.2f} deg")
        if not q["spans_3d"]:
            print("      *** COLLINEAR: these fix only 5 of 6 DOF. "
                  "Inversion/eversion")
            print("          is undetermined. Export a landmark off the long "
                  "axis. ***")
        print(f"      rigidity: pairwise distances vary by "
              f"{rep['rigidity_mm'][s]:.3f} mm over the static frames")
        if rep["rigidity_mm"][s] > 2.0:
            print("      ! these landmarks do not move as one rigid body. "
                  "They may not")
            print("        all come from the same segment -- check the signal "
                  "names.")

    print("\n" + "-" * W)
    print("  MESH TO SEGMENT ASSIGNMENT")
    print("-" * W)
    for s in meta["segments"]:
        n = rep["vertices_per_segment"].get(s, 0)
        print(f"    {s:12s} {n:6d} vertices", end="")
        if s in rep["ankle_inside_mesh_mm"]:
            print(f" | joint centre sits {rep['ankle_inside_mesh_mm'][s]:.1f} mm "
                  f"inside the surface", end="")
        print()
    bad = [s for s, v in rep["ankle_inside_mesh_mm"].items() if v > 80]
    if bad:
        print(f"    ! {bad}: the joint centre is far from the mesh. Are the "
              f"meshes really")
        print("      in Theia coordinates? Run transform_obj_to_theia.py first.")

    print("\n" + "-" * W)
    print("  WHAT THIS BINDING CANNOT DO")
    print("-" * W)
    print("    Each region is rigid. The arch will not flatten and the toes")
    print("    will not flex, because Theia exports no toe orientation to")
    print("    drive them with.")
    for s, (beyond, total) in rep["distal_to_mtp"].items():
        print(f"      {s:12s} {beyond:5d} of {total} vertices "
              f"({100.0 * beyond / max(total, 1):.0f}%) are distal to the MTP "
              f"joint")
    print("    Those are the ones a rigid binding holds still relative to the")
    print("    rest of the foot. Fine if you want skin position in the lab")
    print("    frame; not fine if you are after toe kinematics.")
    print("=" * W)


# %%==========================================================================
#  RUN
# ============================================================================

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    csv_path = argv[0] if argv else None
    binding = bind(csv_path)
    out = save_binding(binding)
    print(f"\n  wrote {out}  "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")
    print("  next:  python apply_binding.py TRIAL_metrics.csv")
    return binding


if __name__ == "__main__":
    BINDING = main()
