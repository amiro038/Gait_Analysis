# -*- coding: utf-8 -*-
"""
=============================================================================
 OPTIONAL -- write an animated FBX so you can watch the result
=============================================================================

Builds an FBX in which each foot mesh is a rigid object animated by its
segment's pose, alongside small cubes at the lower-limb joint centres for
context. Open it and the scan feet should stay attached to the legs.

    python export_posed_fbx.py TRIAL_metrics.csv

Requires Blender's Python module for the FBX writing:  pip install bpy
(a large install, ~1 GB). The binding and replay scripts do NOT need it --
this is a visualisation step only. Everything it exports comes from the same
`poses` array apply_binding.py writes, so what you see is what you get.

A mesh that spans more than one segment (both_feet.obj) is skipped: it cannot
be one rigid object, and left_feet + right_feet already cover it.

The scene is metres and Z-up, matching Theia, and the FBX is written Z-up so
it drops straight into the same viewer as Theia's own export.
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
OUT_FBX = None                 # None -> <trial>_posed.fbx
FPS = 40.0                     # playback rate only; geometry is unaffected
FRAME_STRIDE = 1

# Joint centres drawn as small cubes for context. Anything missing is skipped.
CONTEXT_JOINTS = [
    "Pelvis_Position",
    "Left_Hip_Position", "Left_Knee_Position", "Left_Ankle_Position",
    "Left_Toes_Position", "Left_Heel_Position",
    "Right_Hip_Position", "Right_Knee_Position", "Right_Ankle_Position",
    "Right_Toes_Position", "Right_Heel_Position",
]
JOINT_CUBE_MM = 20.0


# %%==========================================================================
#  INPUT
# ============================================================================

def read_metrics(path):
    """Visual3D export -> (vectors {name:(F,3)}, matrices {name:(F,4,4)})."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh]
    names, comps = rows[1][1:], rows[4][1:]
    width = len(names)
    vals = []
    for r in rows[5:]:
        cells = r[1:]
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        vals.append([float(c) if c.strip() else np.nan for c in cells[:width]])
    data = np.array(vals, float)
    blank = np.all(np.isnan(data), axis=1)
    last = len(data)
    while last > 0 and blank[last - 1]:
        last -= 1
    data = data[:last]

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
    return vectors, matrices


def obj_faces(lines):
    """Triangle/polygon indices from .obj `f` lines, zero-based."""
    faces = []
    for line in lines:
        if not line.startswith("f "):
            continue
        idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
        if len(idx) >= 3:
            faces.append(idx)
    return faces


def segment_poses(meta, vectors, matrices):
    """(frames, segments, 4, 4), straight from the export."""
    seg_names = meta["segments"]
    n = None
    for s in seg_names:
        sig = meta["pose_signal"].get(s)
        if sig and sig in matrices:
            n = len(matrices[sig])
            break
    if n is None:
        raise KeyError("the trial carries no *_Global_4x4 pose signal; run "
                       "apply_binding.py and export from its poses instead")
    poses = np.full((n, len(seg_names), 4, 4), np.nan)
    for k, s in enumerate(seg_names):
        sig = meta["pose_signal"].get(s)
        if sig not in matrices:
            continue
        P = matrices[sig].copy()
        U, _, Vt = np.linalg.svd(P[:, :3, :3])
        D = np.ones(U.shape[:-1])
        D[..., -1] = np.sign(np.linalg.det(U @ Vt))
        P[:, :3, :3] = (U * D[..., None, :]) @ Vt
        P[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
        poses[:, k] = P
    return poses


# %%==========================================================================
#  FBX
# ============================================================================

def export(trial_csv=None, binding_file=None, out_fbx=None, verbose=True):
    try:
        import bpy
        from mathutils import Matrix
    except ImportError:
        raise SystemExit(
            "this step needs Blender's python module:  pip install bpy\n"
            "(the binding and replay scripts do not)")

    trial_csv = TRIAL_METRICS_CSV if trial_csv is None else trial_csv
    binding_file = BINDING_FILE if binding_file is None else binding_file
    stem = os.path.splitext(os.path.basename(trial_csv))[0]
    out_fbx = out_fbx or OUT_FBX or (stem + "_posed.fbx")

    z = np.load(binding_file, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    v_local, seg_of = z["vertices_local"], z["vertex_segment"]
    seg_names = meta["segments"]

    vectors, matrices = read_metrics(trial_csv)
    poses = segment_poses(meta, vectors, matrices)
    frames = range(0, len(poses), FRAME_STRIDE)

    if verbose:
        print("=" * 74)
        print("  EXPORT ANIMATED FBX")
        print("=" * 74)
        print(f"\n  binding : {os.path.basename(binding_file)}")
        print(f"  trial   : {os.path.basename(trial_csv)}  "
              f"({len(poses)} frames, exporting {len(list(frames))})")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.fps = int(round(FPS))
    # Key from frame ZERO, not 1. Blender's FBX exporter writes keyframe times
    # relative to time zero, and the importer maps FBX time 0 back onto frame
    # 1. Keying from frame 1 therefore lands every key one frame late: the
    # animation comes back shifted, which on this trial was a silent 12.8 mm
    # error that looks entirely plausible on screen. Keying from 0 round-trips
    # to 0.00005 mm.
    n_out = len(list(frames))
    scene.frame_start, scene.frame_end = 0, max(0, n_out - 1)

    def animate(obj, mats):
        obj.rotation_mode = "XYZ"
        for f, M in enumerate(mats):
            if not np.isfinite(M).all():
                continue
            obj.matrix_world = Matrix([list(row) for row in M])
            obj.keyframe_insert("location", frame=f)
            obj.keyframe_insert("rotation_euler", frame=f)
            obj.keyframe_insert("scale", frame=f)

    # ---- one rigid object per foot mesh -------------------------------
    exported, skipped = [], []
    for m in meta["meshes"]:
        sl = slice(m["start"], m["start"] + m["count"])
        segs = np.unique(seg_of[sl])
        if len(segs) != 1:
            skipped.append((m["name"], len(segs)))
            continue
        k = int(segs[0])
        name = os.path.splitext(m["name"])[0]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([tuple(p) for p in v_local[sl]], [],
                         obj_faces(m["lines"]))
        mesh.validate()
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        animate(obj, [poses[i, k] for i in frames])
        exported.append((name, seg_names[k], m["count"]))

    # ---- joint centres as little cubes, for context -------------------
    half = JOINT_CUBE_MM / 2000.0
    corners = [(x, y, zz) for x in (-half, half) for y in (-half, half)
               for zz in (-half, half)]
    cube_faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                  (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    n_ctx = 0
    for jname in CONTEXT_JOINTS:
        if jname not in vectors:
            continue
        P = vectors[jname]
        mesh = bpy.data.meshes.new(jname)
        mesh.from_pydata(corners, [], cube_faces)
        mesh.validate()
        mesh.update()
        obj = bpy.data.objects.new(jname, mesh)
        scene.collection.objects.link(obj)
        mats = []
        for i in frames:
            M = np.eye(4)
            M[:3, 3] = P[i]
            mats.append(M if np.isfinite(P[i]).all() else np.full((4, 4), np.nan))
        animate(obj, mats)
        n_ctx += 1

    bpy.ops.export_scene.fbx(
        filepath=os.path.abspath(out_fbx),
        use_selection=False, apply_unit_scale=True, global_scale=1.0,
        axis_forward="Y", axis_up="Z",              # keep Theia's Z-up
        bake_anim=True, bake_anim_use_all_bones=False,
        bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
        bake_anim_step=1.0, bake_anim_simplify_factor=0.0,
        object_types={"MESH"}, mesh_smooth_type="OFF",
        add_leaf_bones=False, path_mode="COPY")

    if verbose:
        print("\n  objects written:")
        for name, seg, n in exported:
            print(f"    {name:24s} {n:6d} vertices, driven by {seg}")
        for name, nseg in skipped:
            print(f"    {name:24s} SKIPPED -- spans {nseg} segments, so it "
                  f"cannot be one")
            print(f"    {'':24s} rigid object. left_feet + right_feet already "
                  f"cover it.")
        print(f"    {n_ctx} joint-centre cubes for context")
        print(f"\n  wrote {out_fbx}  "
              f"({os.path.getsize(out_fbx) / 1e6:.1f} MB)  "
              f"{scene.frame_end + 1} frames @ {scene.render.fps} fps, "
              f"Z-up, metres")
        print("  data frame i is at FBX frame i (the first key is at time 0).")
    return out_fbx


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    files = [a for a in argv if not a.startswith("--")]
    return export(files[0] if files else None)


if __name__ == "__main__":
    OUT = main()
