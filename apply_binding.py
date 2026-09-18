# -*- coding: utf-8 -*-
"""
=============================================================================
 STEP 2 of 2 -- replay a binding over a trial to get mesh vertex positions
=============================================================================

Reads the .npz written by bind_mesh_to_bones.py and a Visual3D metrics export
of any trial, and reconstructs where every mesh vertex was, frame by frame.

    python apply_binding.py TRIAL_metrics.csv
    python apply_binding.py TRIAL_metrics.csv --vertices --obj

HOW IT WORKS
------------
The binding stores each segment's landmarks and every vertex in that
segment's local frame. For each trial frame it fits the stored landmarks onto
the trial's landmarks (Kabsch, least squares over all of them) to get a 4x4
pose, then carries the vertices through it. Using a least-squares fit rather
than reconstructing the frame axis by axis means extra landmarks, if you ever
export them, improve the result instead of being ignored.

WHAT COMES OUT
--------------
`poses` (frames, segments, 4, 4) is always written and is tiny -- it is the
whole result, since vertices are one matrix multiply away. Pass --vertices to
also store the (frames, vertices, 3) array, and --obj to write one .obj per
frame. Both get large fast: 5000 vertices over 3000 frames is 180 MB as
float32, and 3000 .obj files is not a thing you want by accident.

Frames where a landmark is missing come back as NaN rather than stopping the
run.

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
TRIAL_METRICS_CSV = "D05_C01_a_pose_metrics.csv"   # default if none given

SAVE_VERTICES = False      # (frames, vertices, 3) -- large
WRITE_OBJ_SEQUENCE = False # one .obj per frame -- larger
OBJ_DIR = "posed_frames"
OBJ_STRIDE = 1             # write every Nth frame only
OUT_SUFFIX = "_posed"
VERBOSE = True

# A trial whose landmark triangle does not match the one in the binding is a
# different subject, a different model, or the wrong file.
RIGIDITY_TOLERANCE_MM = 5.0


# %%==========================================================================
#  VISUAL3D METRICS READER  (copy of the one in bind_mesh_to_bones.py)
# ============================================================================

def read_metrics(path):
    """Visual3D tab-delimited export -> {signal: (frames, 3)} plus frame ids."""
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
    return ({n: np.stack([ch["X"], ch["Y"], ch["Z"]], axis=1)
             for n, ch in signals.items() if {"X", "Y", "Z"} <= set(ch)}, items)


# %%==========================================================================
#  POSE RECOVERY
# ============================================================================

def kabsch(P, Q):
    """Rigid (R, t) taking P onto Q, both (K, 3). Least squares over all K."""
    cp, cq = P.mean(0), Q.mean(0)
    A, B = P - cp, Q - cq
    U, _, Vt = np.linalg.svd(B.T @ A)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, cq - R @ cp


def poses_from_landmarks(local, world):
    """local (K,3) reference, world (F,K,3) per frame -> poses (F,4,4).

    Frames with any missing landmark come back as NaN.
    """
    F = len(world)
    out = np.full((F, 4, 4), np.nan)
    for i in range(F):
        W = world[i]
        if not np.isfinite(W).all():
            continue
        R, t = kabsch(local, W)
        out[i, :3, :3] = R
        out[i, :3, 3] = t
        out[i, 3, :] = (0.0, 0.0, 0.0, 1.0)
    return out


def residuals(local, world, poses):
    """Per-frame RMS gap between the posed reference landmarks and the trial's."""
    out = np.full(len(world), np.nan)
    for i, P in enumerate(poses):
        if not np.isfinite(P).all():
            continue
        pred = local @ P[:3, :3].T + P[:3, 3]
        out[i] = np.sqrt(((pred - world[i]) ** 2).sum(1).mean())
    return out


# %%==========================================================================
#  MAIN
# ============================================================================

def load_binding(path=None):
    path = BINDING_FILE if path is None else path
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run bind_mesh_to_bones.py")
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return meta, z


def apply_binding(trial_csv=None, binding_file=None, save_vertices=None,
                  write_obj=None, verbose=None):
    trial_csv = TRIAL_METRICS_CSV if trial_csv is None else trial_csv
    save_vertices = SAVE_VERTICES if save_vertices is None else save_vertices
    write_obj = WRITE_OBJ_SEQUENCE if write_obj is None else write_obj
    verbose = VERBOSE if verbose is None else verbose

    meta, z = load_binding(binding_file)
    v_local = z["vertices_local"]
    n_local = z["normals_local"]
    seg_of = z["vertex_segment"]
    lm_local = z["landmarks_local"]
    seg_names = meta["segments"]

    signals, items = read_metrics(trial_csv)

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

    # ---- a pose per segment per frame -------------------------------
    poses = np.full((len(items), len(seg_names), 4, 4), np.nan)
    stats = []
    for k, s in enumerate(seg_names):
        names = meta["landmark_names"][s]
        missing = [n for n in names if n not in signals]
        if missing:
            raise KeyError(f"{s}: the trial export is missing {missing}. It "
                           f"must carry the same signals the binding was "
                           f"built from.")
        world = np.stack([signals[n] for n in names], axis=1)     # (F, K, 3)
        loc = lm_local[k, :len(names)]
        poses[:, k] = poses_from_landmarks(loc, world)
        res = residuals(loc, world, poses[:, k])

        # does the trial's landmark triangle match the binding's?
        ref_d, tri_d = [], []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ref_d.append(np.linalg.norm(loc[i] - loc[j]))
                tri_d.append(np.nanmean(np.linalg.norm(world[:, i] - world[:, j],
                                                       axis=1)))
        shape_err = float(np.max(np.abs(np.array(ref_d) - np.array(tri_d))) * 1000)
        stats.append(dict(segment=s, n_good=int(np.isfinite(poses[:, k, 3, 3]).sum()),
                          fit_rms_mm=float(np.nanmean(res) * 1000),
                          fit_max_mm=float(np.nanmax(res) * 1000),
                          shape_mismatch_mm=shape_err))

    # ---- vertices ----------------------------------------------------
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
                    # a rigid pose rotates normals by R, no inverse-transpose
                    norms[i, m] = (Nl @ R.T).astype(np.float32)

    # ---- write -------------------------------------------------------
    stem = os.path.splitext(os.path.basename(trial_csv))[0]
    out_npz = stem + OUT_SUFFIX + ".npz"
    payload = dict(poses=poses, segments=np.array(seg_names),
                   frames=np.array(items),
                   vertex_segment=seg_of,
                   meta=np.array(json.dumps(dict(
                       trial=os.path.basename(trial_csv),
                       binding=meta, stats=stats))))
    if save_vertices:
        payload["vertices"] = verts
    np.savez_compressed(out_npz, **payload)

    n_obj = 0
    if write_obj:
        n_obj = write_obj_sequence(meta, verts, norms, stem)

    if verbose:
        print("\n" + "-" * 74)
        print("  POSE RECOVERY")
        print("-" * 74)
        for st in stats:
            print(f"    {st['segment']:12s} {st['n_good']:5d}/{len(items)} frames "
                  f"| landmark fit RMS {st['fit_rms_mm']:5.2f} mm "
                  f"(max {st['fit_max_mm']:.2f})")
            print(f"      landmark triangle vs the binding: "
                  f"{st['shape_mismatch_mm']:.2f} mm", end="")
            if st["shape_mismatch_mm"] > RIGIDITY_TOLERANCE_MM:
                print("   *** MISMATCH -- different subject or model? ***")
            else:
                print("   OK, same body")
        print("\n    A landmark fit RMS near zero is expected: these signals are")
        print("    derived from one segment pose, so they are rigid by")
        print("    construction. A non-zero value means they are not all from")
        print("    the same segment.")
        print("\n" + "-" * 74)
        print(f"  wrote {out_npz}  ({os.path.getsize(out_npz) / 1e6:.1f} MB)")
        print("    poses          (frames, segments, 4, 4)   always")
        if save_vertices:
            print("    vertices       (frames, vertices, 3)     float32")
        else:
            print("    vertices       not stored -- pass --vertices, or "
                  "rebuild them with")
            print("                   vertices_at_frame() below (one matmul)")
        if n_obj:
            print(f"  wrote {n_obj} .obj files to {OBJ_DIR}/")
        print("-" * 74)
    return dict(poses=poses, vertices=verts, normals=norms, stats=stats,
                frames=items, out=out_npz)


def vertices_at_frame(meta, z, poses, frame):
    """Rebuild world vertices for one frame from the compact `poses` array."""
    v_local, seg_of = z["vertices_local"], z["vertex_segment"]
    out = np.full_like(v_local, np.nan)
    for k in range(poses.shape[1]):
        P = poses[frame, k]
        if not np.isfinite(P).all():
            continue
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
    trial = files[0] if files else None
    return apply_binding(trial,
                         save_vertices="--vertices" in argv,
                         write_obj="--obj" in argv)


if __name__ == "__main__":
    RESULT = main()
