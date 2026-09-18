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
                frames=items, out=out_npz)


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
    return apply_binding(files[0] if files else None,
                         save_vertices="--vertices" in argv,
                         write_obj="--obj" in argv)


if __name__ == "__main__":
    RESULT = main()
