# -*- coding: utf-8 -*-
"""
=============================================================================
 STEP 1 of 2 -- bind the foot meshes to their Theia segments
=============================================================================

Takes the scan-derived foot meshes (already in Theia coordinates, from
transform_obj_to_theia.py) plus a STATIC-POSE Visual3D metrics export, works
out where each mesh sits on each foot segment, and stores that relationship
in one .npz file.

`apply_binding.py` then replays it over any trial, and
`export_posed_fbx.py` writes an animated FBX you can look at.

WHAT DRIVES IT
--------------
If the export carries `<Side>_Foot_Global_4x4`, that IS the segment pose:
a 4x4 homogeneous matrix, row-major, whose translation is the ankle joint
centre and whose rotation block is the foot segment's global orientation. It
is used directly -- nothing is fitted, nothing is reconstructed. Verified on
D05_C01: its translation equals `<Side>_Ankle_Position` to 0.000 mm, its
rotation block matches the segment orientation in Theia's own FBX to 0.01 deg
mean / 0.03 deg max, and every other foot landmark sits at a FIXED position
in its frame to +-0.01 mm across all 484 frames of a movement trial.

Without it, the script falls back to fitting a pose from three landmarks
(ankle, foot COM, heel) by Kabsch. That works -- three non-collinear points
determine a rigid pose -- but it is an estimate rather than the thing itself,
and a landmark error of 1 mm becomes roughly 2.3 deg of inversion/eversion.

DO NOT put the toes in the fallback set. `<Side>_Toes_Position` is the toes
segment's origin, and it is NOT rigid to the foot: between the A-pose and the
SKS trial it moves 29 mm in the foot's own frame on the left side. It would
drag the fitted foot pose with it.

WHICH MESH GOES WITH WHICH SEGMENT
----------------------------------
Assigned per VERTEX, not per file, by proximity to each segment. That is what
makes `both_feet.obj` work: its left and right halves bind to different
segments and move independently.

RIGIDITY
--------
Each mesh region is bound rigidly to one segment. The arch does not flatten
and the toes do not flex. The report says how many vertices sit distal to the
MTP joint so you can see what that affects.

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
STATIC_METRICS_CSV = "D05_C01_apose_metrics_v2.csv"

# Meshes to bind, in THEIA coordinates (run transform_obj_to_theia.py first).
MESH_FILES = ["left_feet_theia.obj", "right_feet_theia.obj",
              "both_feet_theia.obj"]

# One entry per rigid segment.
#   pose      -- a 4x4 signal. Used directly when present. Preferred.
#   landmarks -- three or more signals, used only if `pose` is missing.
#                They must be rigid to the segment and not collinear.
SEGMENTS = {
    "left_foot": dict(
        pose="Left_Foot_Global_4x4",
        landmarks=["Left_Ankle_Position", "Left_Foot_Position",
                   "Left_Heel_Position"]),
    "right_foot": dict(
        pose="Right_Foot_Global_4x4",
        landmarks=["Right_Ankle_Position", "Right_Foot_Position",
                   "Right_Heel_Position"]),
}

# Reported only, to say how much of each mesh a rigid binding holds still.
MTP_LANDMARK = {"left_foot": "Left_Toes_Position",
                "right_foot": "Right_Toes_Position"}

# A static trial's first and last frames are usually filter edge transients.
# Frames whose landmarks sit further than this from the median pose are
# dropped from the reference.
FRAME_TOLERANCE_MM = 8.0

OUT_FILE = "foot_mesh_binding.npz"
VERBOSE = True


# %%==========================================================================
#  VISUAL3D METRICS READER
# ============================================================================
# Five header rows: file path, signal name, signal type, ORIGINAL, component.
# Then one row per frame with the frame number in column 0. Three-component
# signals come back as (F, 3); sixteen-component ones as (F, 4, 4), row-major.
#
# Trailing all-NaN rows are Visual3D's padding and are dropped. Interior gaps
# are KEPT as NaN, so frame numbering is never silently compacted.
#
# NOTE: apply_binding.py carries its own copy of this reader so it stays
# standalone. Fix bugs in both.

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
            flat = np.stack([ch[str(k)] for k in range(16)], axis=1)
            matrices[n] = flat.reshape(-1, 4, 4)            # row-major
    return vectors, matrices, items


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
#  POSES
# ============================================================================

def orthonormalise(R):
    """Nearest rotation matrix. Works on (...,3,3)."""
    U, _, Vt = np.linalg.svd(R)
    D = np.ones(U.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


def landmark_frame(L):
    """Orthonormal frame from landmarks (K,3): origin, long axis, remainder."""
    origin = L[0]
    e1 = L[-1] - origin
    e1 = e1 / np.linalg.norm(e1)
    off = L[1] - origin
    e2 = off - (off @ e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-6:
        raise ValueError("landmarks are collinear: they fix only 5 of 6 "
                         "degrees of freedom, leaving roll about the long "
                         "axis undetermined. Use a landmark off that axis.")
    e2 = e2 / n2
    return np.stack([e1, e2, np.cross(e1, e2)], axis=1), origin


def landmark_quality(L):
    X = L - L.mean(0)
    sv = np.linalg.svd(X, compute_uv=False)
    sv = np.pad(sv, (0, max(0, 3 - len(sv))))[:3]
    e1 = L[-1] - L[0]
    e1 = e1 / np.linalg.norm(e1)
    off = L[1] - L[0]
    lever = float(np.linalg.norm(off - (off @ e1) * e1))
    return dict(singular_values_mm=(sv * 1000).tolist(),
                off_axis_lever_mm=lever * 1000,
                roll_deg_per_mm_noise=float(np.degrees(np.arctan2(0.001,
                                                                 max(lever, 1e-9)))),
                spans_3d=bool(lever * 1000 > 1.0))


# %%==========================================================================
#  BINDING
# ============================================================================

def good_frames(blocks):
    """Drop static frames whose landmarks sit far from the median pose."""
    allblk = np.concatenate(list(blocks.values()), axis=1)
    med = np.median(allblk, axis=0)
    dev = np.linalg.norm(allblk - med, axis=2).mean(axis=1)
    keep = np.where(dev <= FRAME_TOLERANCE_MM / 1000.0)[0]
    if len(keep) == 0:
        keep = np.arange(len(allblk))
    dropped = [(int(i), float(dev[i] * 1000))
               for i in range(len(allblk)) if i not in keep]
    return keep, dropped, dev * 1000


def bind(static_csv=None, mesh_files=None, verbose=None):
    static_csv = STATIC_METRICS_CSV if static_csv is None else static_csv
    mesh_files = MESH_FILES if mesh_files is None else mesh_files
    verbose = VERBOSE if verbose is None else verbose

    vectors, matrices, items = read_metrics(static_csv)
    seg_names = list(SEGMENTS)

    # --- which frames to average over ------------------------------------
    blocks = {}
    for s in seg_names:
        lms = [n for n in SEGMENTS[s].get("landmarks", []) if n in vectors]
        if lms:
            blocks[s] = np.stack([vectors[n] for n in lms], axis=1)
    if blocks:
        keep, dropped, dev_mm = good_frames(blocks)
    else:
        keep, dropped, dev_mm = np.arange(len(items)), [], np.zeros(len(items))

    # --- the reference pose per segment ----------------------------------
    ref_R, ref_t, source, quality, lm_used = {}, {}, {}, {}, {}
    for s in seg_names:
        spec = SEGMENTS[s]
        pose_sig = spec.get("pose")
        if pose_sig and pose_sig in matrices:
            T = matrices[pose_sig][keep]
            ref_R[s] = orthonormalise(T[:, :3, :3].mean(axis=0))
            ref_t[s] = T[:, :3, 3].mean(axis=0)
            source[s] = pose_sig
            lm_used[s] = [n for n in spec.get("landmarks", []) if n in vectors]
        else:
            lms = [n for n in spec.get("landmarks", []) if n in vectors]
            if len(lms) < 3:
                raise KeyError(
                    f"{s}: no '{pose_sig}' signal and fewer than three usable "
                    f"landmarks. Export the 4x4, or name three rigid, "
                    f"non-collinear landmarks.")
            L = np.stack([vectors[n] for n in lms], axis=1)[keep].mean(axis=0)
            ref_R[s], ref_t[s] = landmark_frame(L)
            source[s] = "landmarks"
            lm_used[s] = lms
            quality[s] = landmark_quality(L)

    # --- how rigid are the landmarks, and where do they sit? -------------
    lm_local, lm_rigid = {}, {}
    for s in seg_names:
        lms = lm_used[s]
        if not lms:
            continue
        P = np.stack([vectors[n] for n in lms], axis=1)[keep]      # (F,K,3)
        loc = np.einsum("ji,fkj->fki", ref_R[s], P - ref_t[s])
        lm_local[s] = loc.mean(axis=0)
        lm_rigid[s] = float(loc.std(axis=0).max() * 1000)

    # --- meshes ----------------------------------------------------------
    meshes, all_v, all_n, all_seg = [], [], [], []
    cursor = 0
    for path in mesh_files:
        if not os.path.exists(path):
            print(f"  !! {path}: not found, skipped")
            continue
        lines, V, N, v_at, n_at = read_obj(path)
        cen = np.stack([ref_t[s] for s in seg_names])
        seg_idx = np.linalg.norm(V[:, None, :] - cen[None, :, :],
                                 axis=2).argmin(axis=1)
        meshes.append(dict(name=os.path.basename(path), start=cursor,
                           count=len(V), lines=lines, v_at=v_at, n_at=n_at,
                           has_normals=bool(len(N))))
        all_v.append(V)
        all_n.append(N if len(N) == len(V) else np.zeros((len(V), 3)))
        all_seg.append(seg_idx)
        cursor += len(V)
    if not meshes:
        raise ValueError("no meshes were read")
    V, Nrm = np.vstack(all_v), np.vstack(all_n)
    seg_of = np.concatenate(all_seg)

    # --- vertices into each segment's own frame --------------------------
    v_local, n_local = np.zeros_like(V), np.zeros_like(Nrm)
    for k, s in enumerate(seg_names):
        m = seg_of == k
        v_local[m] = (V[m] - ref_t[s]) @ ref_R[s]
        n_local[m] = Nrm[m] @ ref_R[s]

    # --- diagnostics ------------------------------------------------------
    report = dict(segments=seg_names, source=source, quality=quality,
                  landmark_rigidity_mm=lm_rigid,
                  landmarks_in_segment_frame_mm={
                      s: (lm_local[s] * 1000).round(2).tolist()
                      for s in lm_local},
                  n_frames=len(items), n_used=int(len(keep)),
                  dropped_frames=dropped,
                  frame_deviation_mm=[round(float(d), 2) for d in dev_mm],
                  vertices_per_segment={s: int((seg_of == k).sum())
                                        for k, s in enumerate(seg_names)},
                  ankle_inside_mesh_mm={}, distal_to_mtp={})
    for k, s in enumerate(seg_names):
        m = seg_of == k
        if not m.any():
            continue
        report["ankle_inside_mesh_mm"][s] = float(
            np.linalg.norm(V[m] - ref_t[s], axis=1).min() * 1000)
        mtp_name = MTP_LANDMARK.get(s)
        if mtp_name in vectors:
            mtp = vectors[mtp_name][keep].mean(axis=0)
            axis = mtp - ref_t[s]
            axis = axis / np.linalg.norm(axis)
            beyond = ((V[m] - mtp) @ axis) > 0
            report["distal_to_mtp"][s] = [int(beyond.sum()), int(m.sum())]

    K = max((len(lm_used[s]) for s in seg_names), default=0)
    binding = dict(
        meta=dict(format_version=2, created=str(date.today()),
                  static_metrics_csv=os.path.basename(static_csv),
                  meshes=[dict(name=m["name"], start=m["start"],
                               count=m["count"], has_normals=m["has_normals"],
                               lines=m["lines"], v_at=m["v_at"],
                               n_at=m["n_at"]) for m in meshes],
                  segments=seg_names,
                  pose_signal={s: SEGMENTS[s].get("pose") for s in seg_names},
                  pose_source=source,
                  landmark_names={s: lm_used[s] for s in seg_names},
                  units="metres",
                  coordinate_system="Theia3D global (Z-up), as the export",
                  report=report),
        vertices_local=v_local, normals_local=n_local,
        vertex_segment=seg_of.astype(np.int16),
        reference_R=np.stack([ref_R[s] for s in seg_names]),
        reference_t=np.stack([ref_t[s] for s in seg_names]),
        landmarks_local=np.stack([
            np.pad(lm_local.get(s, np.zeros((0, 3))),
                   ((0, K - len(lm_used[s])), (0, 0))) for s in seg_names])
        if K else np.zeros((len(seg_names), 0, 3)),
    )
    if verbose:
        print_report(binding)
    return binding


def save_binding(binding, path=None):
    path = OUT_FILE if path is None else path
    np.savez_compressed(
        path, meta=np.array(json.dumps(binding["meta"])),
        vertices_local=binding["vertices_local"],
        normals_local=binding["normals_local"],
        vertex_segment=binding["vertex_segment"],
        reference_R=binding["reference_R"], reference_t=binding["reference_t"],
        landmarks_local=binding["landmarks_local"])
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
          f"({rep['n_frames']} frames, {rep['n_used']} used)")
    for i, d in rep["dropped_frames"]:
        print(f"      dropped frame {i}: landmarks {d:.1f} mm from the median "
              f"pose (edge transient)")
    print(f"  meshes: {', '.join(m['name'] for m in meta['meshes'])}  "
          f"({len(b['vertices_local'])} vertices)")

    print("\n" + "-" * W)
    print("  WHAT DRIVES EACH SEGMENT")
    print("-" * W)
    for s in meta["segments"]:
        src = rep["source"][s]
        if src != "landmarks":
            print(f"\n    {s}:  {src}")
            print("      the exported segment pose, used directly. Nothing is")
            print("      fitted, so there is no reconstruction error at all.")
        else:
            print(f"\n    {s}:  FITTED from landmarks "
                  f"({', '.join(meta['landmark_names'][s])})")
            q = rep["quality"].get(s, {})
            if q:
                sv = q["singular_values_mm"]
                print(f"      landmark spread   [{sv[0]:7.2f} {sv[1]:6.2f} "
                      f"{sv[2]:5.2f}] mm")
                print(f"      off-axis lever    {q['off_axis_lever_mm']:7.2f} mm"
                      "   <- this is what fixes the roll")
                print(f"      roll error per mm of landmark noise  "
                      f"{q['roll_deg_per_mm_noise']:.2f} deg")
                if not q["spans_3d"]:
                    print("      *** COLLINEAR -- inversion/eversion is "
                          "undetermined. ***")
            print(f"      export {meta['pose_signal'][s]} from Visual3D and "
                  "this becomes exact.")
        if s in rep["landmark_rigidity_mm"]:
            names = meta["landmark_names"][s]
            loc = rep["landmarks_in_segment_frame_mm"][s]
            print("      landmarks in this segment's frame (mm), and how "
                  "far they drift:")
            for nm, p in zip(names, loc):
                print(f"        {nm:26s} [{p[0]:7.1f} {p[1]:7.1f} {p[2]:7.1f}]")
            print(f"        max drift over the static frames "
                  f"{rep['landmark_rigidity_mm'][s]:.2f} mm", end="")
            print("   (rigid)" if rep["landmark_rigidity_mm"][s] < 2.0
                  else "   ! not rigid to this segment")

    print("\n" + "-" * W)
    print("  MESH TO SEGMENT ASSIGNMENT")
    print("-" * W)
    for s in meta["segments"]:
        n = rep["vertices_per_segment"].get(s, 0)
        line = f"    {s:12s} {n:6d} vertices"
        if s in rep["ankle_inside_mesh_mm"]:
            line += (f" | joint centre sits "
                     f"{rep['ankle_inside_mesh_mm'][s]:.1f} mm inside the "
                     f"surface")
        print(line)
    bad = [s for s, v in rep["ankle_inside_mesh_mm"].items() if v > 80]
    if bad:
        print(f"    ! {bad}: joint centre far from the mesh. Are the meshes in")
        print("      Theia coordinates? Run transform_obj_to_theia.py first.")

    print("\n" + "-" * W)
    print("  WHAT THIS BINDING CANNOT DO")
    print("-" * W)
    print("    Each region is rigid: the arch will not flatten and the toes")
    print("    will not flex.")
    for s, (beyond, total) in rep["distal_to_mtp"].items():
        print(f"      {s:12s} {beyond:5d} of {total} vertices "
              f"({100.0 * beyond / max(total, 1):.0f}%) are distal to the MTP "
              f"joint")
    print("=" * W)


# %%==========================================================================
#  RUN
# ============================================================================

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    binding = bind(argv[0] if argv else None)
    out = save_binding(binding)
    print(f"\n  wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    print("  next:  python apply_binding.py TRIAL_metrics.csv")
    return binding


if __name__ == "__main__":
    BINDING = main()
