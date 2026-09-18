# -*- coding: utf-8 -*-
"""
=============================================================================
 BUILD THE FOOT-MESH BINDING  --  everything up to, and including, the bind
=============================================================================

One script, three steps, one output. Set the paths in CONFIG and press F5.

    1. Solve the MOVE4D -> Theia3D rigid transform from a static A-pose
       recorded by both systems.
    2. Put the segmented .obj foot meshes into Theia coordinates.
    3. Bind them to the Theia foot segments using a Visual3D metrics export
       of that same static pose.

The result is a single file, `foot_mesh_binding.npz`. From then on,
`apply_binding.py` replays it over any trial you like; it shares no imports
with this script, so the two can live apart.

    python build_foot_binding.py          # steps 1-3
    python apply_binding.py TRIAL.csv --plot

STEP 1 IS A PER-SESSION CALIBRATION
-----------------------------------
The MOVE4D and Theia lab frames do not move between trials, so the transform
only has to be solved once. After the first run it is cached in T_FILE and
reused; set FORCE_RESOLVE to redo it. If you already have a T you trust, drop
it in T_MATRIX and step 1 is skipped entirely.

WHAT EACH STEP CHECKS
---------------------
Step 1  mirroring (a reflection cannot be fixed by any rotation), units by
        segment-length ratio, bone-axis consistency per rig, and an
        end-to-end test of T applied in raw file coordinates. It reports the
        heading uncertainty from independent subsets, because the residual
        cannot tell you that.
Step 2  that each mesh really is in raw MOVE4D coordinates (re-centring, a
        unit change and axis flips are all caught), and that the transformed
        soles land on Theia's floor.
Step 3  that the landmarks driving each segment are rigid to it and not
        collinear, and how much of each mesh sits distal to the MTP joint and
        therefore cannot flex under a rigid binding.

Needs numpy. matplotlib only for the optional check figures. No FBX SDK.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import zlib
from datetime import date

import numpy as np


# %%==========================================================================
#  CONFIG -- everything you edit lives here
# ===========================================================================

# Step 1 is a per-SESSION calibration: the MOVE4D and Theia lab frames do not
# move between trials. Solve it once, and every later run reuses T_FILE unless
# you set FORCE_RESOLVE.
REUSE_SAVED_TRANSFORM = True
FORCE_RESOLVE = False


# ---- STEP 1 -- solving the MOVE4D -> Theia3D transform ---------------

THEIA_FBX = "D05_C1_apose_Theia.fbx"

M4D_FBX   = "D05_C1_apose_M4D.fbx"

# Constrain the rotation to be about the vertical. Both systems get their
# vertical from the same physical gravity/lab calibration, so in principle
# only the heading differs. Set False to fit all 3 DOF; the report evaluates
# both either way, so you can see what the extra freedom buys.
ALIGN_VERTICAL_ONLY = True

# The two coordinate systems ARE offset vertically, so t_z is fitted like
# any other component. Set ASSUME_SHARED_FLOOR = True only if you know both
# systems put z = 0 on the same floor plane; it then locks t_z to zero and
# reports what the joint centres implied instead of burying it in the matrix.
#
# Be aware of what a fitted t_z contains. It is the TRUE vertical offset
# between the two calibrations PLUS any systematic vertical bias between the
# two skeletons' joint-centre definitions, and one static pose cannot
# separate them. The floor check below is the closest thing to an
# independent read on it: the MOVE4D scan locates its own floor plane
# directly, so the fitted t_z can be turned into an implied Theia floor
# height and sanity-checked against where Theia's feet sit.
#
# The residual signs are the tell. Suppress a real offset and every vertical
# residual comes out the same sign; absorb it correctly and they scatter
# about zero with only genuine definition differences left standing. The
# report alignment_checks this for you.
ASSUME_SHARED_FLOOR = False

CHECK_FLOOR_FROM_MESH = True

# Bilateral right->left vectors between paired joint centres, added to the
# rotation fit. They are not segment orientations, but they are the best
# HEADING information a standing pose contains, and they come from the same
# joint centres step 6 already uses.
#
# Why they matter: only the HORIZONTAL part of a direction constrains the
# heading, and in a standing pose the limb long axes are nearly vertical
# (horizontal component 0.12-0.17 for the legs, 0.58-0.65 for the A-posed
# arms). A 1 deg error in a thigh axis becomes ~6.7 deg of heading error. A
# right->left vector is horizontal by construction, so its leverage is 1.00,
# and a joint-centre offset that is symmetric between sides -- the usual
# case -- does not rotate it at all. Measured on D05_C1, adding these four
# cuts the leave-one-out heading spread from 2.34 deg to 0.56 deg while
# moving the answer itself by only 0.02 deg.
#
# No weighting is needed. A plain Kabsch already weights each observation's
# heading contribution by its squared horizontal component, so the ML axes
# outvote the long axes ~7:1 for yaw and not at all for tilt, which is
# exactly right.
USE_ML_AXES = True

REPORT_FRAME_CONVENTIONS = True

ALIGN_MAKE_PLOT    = True

# A frame is dropped when its pose sits further from the median pose than
# FACTOR * (the typical frame's distance) + FLOOR. On this trial that removes
# Theia's bind pose (475 mm out) and its last frame (16 mm, a filter edge
# transient) while keeping everything within ~6 mm. The floor stops the rule
# biting on a genuinely still trial, where the typical distance is under 1 mm.
# This assumes a STATIC trial; it is not meant for a walk.
ALIGN_FRAME_FACTOR = 3.0

ALIGN_FRAME_FLOOR_M = 0.002

# Segments whose global orientation drives the rotation (step 5), named by
# their proximal joint. Theia name / MOVE4D name.
ALIGN_SEGMENTS = {
    "l_thigh": ("l_thigh", "LeftHip"),      "r_thigh": ("r_thigh", "RightHip"),
    "l_shank": ("l_shank", "LeftKnee"),     "r_shank": ("r_shank", "RightKnee"),
    "l_uarm":  ("l_uarm",  "LeftShoulder"), "r_uarm":  ("r_uarm",  "RightShoulder"),
    "l_larm":  ("l_larm",  "LeftElbow"),    "r_larm":  ("r_larm",  "RightElbow"),
}

# Joint centres whose positions drive the translation (step 6). These are the
# origins of the segments above: hip, knee, shoulder, elbow.
ALIGN_JOINTS = {
    "l_hip":      ("l_thigh", "LeftHip"),      "r_hip":      ("r_thigh", "RightHip"),
    "l_knee":     ("l_shank", "LeftKnee"),     "r_knee":     ("r_shank", "RightKnee"),
    "l_shoulder": ("l_uarm",  "LeftShoulder"), "r_shoulder": ("r_uarm",  "RightShoulder"),
    "l_elbow":    ("l_larm",  "LeftElbow"),    "r_elbow":    ("r_larm",  "RightElbow"),
}

ALIGN_ML_AXES = [("hip_width", "r_hip", "l_hip"),
           ("knee_width", "r_knee", "l_knee"),
           ("shoulder_width", "r_shoulder", "l_shoulder"),
           ("elbow_width", "r_elbow", "l_elbow")]

# Distal ends, used ONLY to work out which local axis each rig runs its bones
# along, and to report segment lengths. Not fitted on.
ALIGN_SEGMENT_ENDS = {
    "l_thigh": ("l_thigh", "l_shank"),  "r_thigh": ("r_thigh", "r_shank"),
    "l_shank": ("l_shank", "l_foot"),   "r_shank": ("r_shank", "r_foot"),
    "l_uarm":  ("l_uarm",  "l_larm"),   "r_uarm":  ("r_uarm",  "r_larm"),
}

ALIGN_M4D_ENDS = {
    "l_thigh": ("LeftHip", "LeftKnee"),   "r_thigh": ("RightHip", "RightKnee"),
    "l_shank": ("LeftKnee", "LeftAnkle"), "r_shank": ("RightKnee", "RightAnkle"),
    "l_uarm":  ("LeftShoulder", "LeftElbow"),
    "r_uarm":  ("RightShoulder", "RightElbow"),
}



# ---- STEP 2 -- putting the meshes into Theia coordinates -------------

# The .obj files to meshes_to_theia. Output goes next to each input with OUT_SUFFIX
# inserted before the extension, unless OUT_DIR is set.
MESH_FILES_RAW = ["left_feet.obj", "right_feet.obj", "both_feet.obj"]

OUT_SUFFIX = "_theia"

OUT_DIR = None                      # None = alongside the input

# The transform. Either the .npy / .txt written by m4d_to_theia_transform.py,
# or set T_MATRIX directly to a 4x4 array and leave T_FILE as None.
T_FILE = "T_m4d_to_theia.npy"

T_MATRIX = None

# Sanity checks. Both are cheap and both have caught real mistakes.
CHECK_AGAINST_SCAN = "D05_C1_apose_M4D.fbx"   # None to skip

FLOOR_TOLERANCE_MM = 25.0           # how far the sole may sit from the floor

# Draw the result next to Theia's own body model and joint centres. The most
# convincing check there is: if the transform is right, a segmented MOVE4D
# part lands around the matching Theia anatomy.
MESH_MAKE_PLOT = True

COMPARE_WITH_THEIA = "D05_C1_apose_Theia.fbx"     # None to skip the overlay

PLOT_FILE = "obj_transform_check.png"



# ---- STEP 3 -- binding them to the Theia foot segments ---------------

# Visual3D metrics export of the STATIC trial -- the pose the meshes are in.
STATIC_METRICS_CSV = "D05_C01_apose_metrics_v2.csv"

# One entry per rigid segment.
#   pose      -- a 4x4 signal. Used directly when present. Preferred.
#   landmarks -- three or more signals, used only if `pose` is missing.
#                They must be rigid to the segment and not collinear.
BIND_SEGMENTS = {
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
BIND_FRAME_TOLERANCE_MM = 8.0

OUT_FILE = "foot_mesh_binding.npz"

VERBOSE = True



# %%==========================================================================
#  A. BINARY FBX READER  (plumbing -- you should not need to touch this)
# ===========================================================================

_ARRAY_FMT = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b"}


class FbxNode:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name, self.props, self.children = name, props, children

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name):
        return [c for c in self.children if c.name == name]


def _read_prop(buf, off):
    t = buf[off:off + 1].decode("ascii")
    off += 1
    for code, fmt, size in (("Y", "<h", 2), ("I", "<i", 4), ("F", "<f", 4),
                            ("D", "<d", 8), ("L", "<q", 8)):
        if t == code:
            return struct.unpack_from(fmt, buf, off)[0], off + size
    if t == "C":
        return bool(buf[off]), off + 1
    if t in _ARRAY_FMT:                                    # typed array
        n, enc, clen = struct.unpack_from("<III", buf, off)
        off += 12
        raw = buf[off:off + clen]
        off += clen
        if enc == 1:
            raw = zlib.decompress(raw)
        return np.frombuffer(raw, dtype="<" + _ARRAY_FMT[t], count=n), off
    if t in "SR":                                          # string / blob
        n = struct.unpack_from("<I", buf, off)[0]
        off += 4
        raw = buf[off:off + n]
        return (raw.decode("utf-8", "replace") if t == "S" else raw), off + n
    raise ValueError(f"unknown FBX property type {t!r} at offset {off - 1}")


def _read_node(buf, off, version):
    wide = version >= 7500
    end_off, nprops, _ = struct.unpack_from("<QQQ" if wide else "<III", buf, off)
    off += 24 if wide else 12
    sentinel = 25 if wide else 13

    name_len = buf[off]
    off += 1
    if end_off == 0:                       # null record: end of a child list
        return None, off + name_len
    name = buf[off:off + name_len].decode("utf-8", "replace")
    off += name_len

    props = []
    for _ in range(nprops):
        v, off = _read_prop(buf, off)
        props.append(v)

    children = []
    if off < end_off:
        while off < end_off - sentinel:
            child, off = _read_node(buf, off, version)
            if child is None:
                break
            children.append(child)
        off = end_off
    return FbxNode(name, props, children), off


def fbx_load(path):
    """Parse a binary FBX file into its raw node tree. -> (root, version)"""
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf[:21] != b"Kaydara FBX Binary  \x00":
        raise ValueError(f"{path}: not a binary FBX (ASCII FBX unsupported)")
    version = struct.unpack_from("<I", buf, 23)[0]
    off, n, children = 27, len(buf), []
    while off < n - 100:                   # stop before the footer
        node, off = _read_node(buf, off, version)
        if node is None:
            break
        children.append(node)
    return FbxNode("__root__", [], children), version




# %%==========================================================================
#  B. FBX SCENE EVALUATION  (plumbing)
# ===========================================================================

FBX_TIME_UNIT = 46186158000.0          # FBX ktime units per second


# FBX EulerOrder enum -> order the axis rotations are APPLIED in. For column
# vectors the matrix is the reverse product, so XYZ -> Rz @ Ry @ Rx.
_EULER_ORDER = {0: "XYZ", 1: "XZY", 2: "YZX", 3: "YXZ", 4: "ZXY", 5: "ZYX",
                6: "XYZ"}


_TIME_MODE_FPS = {1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0, 6: 30.0,
                  7: 30.0, 8: 29.97, 9: 29.97, 10: 25.0, 11: 24.0, 12: 1000.0,
                  13: 23.976, 15: 96.0, 16: 72.0, 17: 59.94}


def euler_to_matrix(ang_deg, order_enum=0):
    """Euler angles in degrees (..., 3) -> rotation matrices (..., 3, 3)."""
    a = np.radians(np.asarray(ang_deg, dtype=float))
    c, s = np.cos(a), np.sin(a)
    R = np.broadcast_to(np.eye(3), a.shape[:-1] + (3, 3)).copy()
    for axis in _EULER_ORDER.get(int(order_enum), "XYZ"):
        i = "XYZ".index(axis)
        ci, si, one, zero = c[..., i], s[..., i], np.ones_like(c[..., i]), \
            np.zeros_like(c[..., i])
        rows = {"X": [[one, zero, zero], [zero, ci, -si], [zero, si, ci]],
                "Y": [[ci, zero, si], [zero, one, zero], [-si, zero, ci]],
                "Z": [[ci, -si, zero], [si, ci, zero], [zero, zero, one]]}[axis]
        R = np.stack([np.stack(r, axis=-1) for r in rows], axis=-2) @ R
    return R


def _prop(p70, name, default):
    """Read one Properties70 entry as a float scalar or vector."""
    vec = isinstance(default, (list, tuple))
    if p70 is not None:
        for p in p70.children:
            if p.props[0] == name:
                v = p.props[4:]
                return (np.array([float(x) for x in v[:len(default)]])
                        if vec else float(v[0]))
    return np.array(default, dtype=float) if vec else float(default)


class _Joint:
    """One FBX Model node: its static transform parts and its curves."""

    def __init__(self, uid, name, node_type):
        self.uid, self.name, self.type = uid, name, node_type
        self.parent, self.children = None, []
        self.t = np.zeros(3)
        self.r = np.zeros(3)
        self.s = np.ones(3)
        self.pre = np.zeros(3)
        self.post = np.zeros(3)
        self.roff = np.zeros(3)
        self.rpiv = np.zeros(3)
        self.soff = np.zeros(3)
        self.spiv = np.zeros(3)
        self.order = 0
        self.curves = {}

    def _channel(self, prop, static, times):
        out = np.tile(np.asarray(static, dtype=float), (len(times), 1))
        entry = self.curves.get(prop)
        if entry:
            channels, defaults = entry
            for i, ax in enumerate("XYZ"):
                if ax in channels:
                    out[:, i] = channels[ax](times)
                elif ax in defaults:
                    out[:, i] = defaults[ax]
        return out

    def local_matrices(self, times):
        """Local 4x4 transforms over `times` -> (T, 4, 4).

        The Autodesk chain in full:
          T . Roff . Rp . Rpre . R . Rpost^-1 . Rp^-1 . Soff . Sp . S . Sp^-1
        """
        tr = self._channel("Lcl Translation", self.t, times)
        ro = self._channel("Lcl Rotation", self.r, times)
        sc = self._channel("Lcl Scaling", self.s, times)

        Rf = (euler_to_matrix(self.pre) @ euler_to_matrix(ro, self.order)
              @ euler_to_matrix(self.post).T)
        S = np.zeros((len(times), 3, 3))
        S[:, [0, 1, 2], [0, 1, 2]] = sc

        inner = (-self.rpiv + self.soff + self.spiv) - (S @ self.spiv)
        M = np.zeros((len(times), 4, 4))
        M[:, :3, :3] = Rf @ S
        M[:, :3, 3] = (tr + self.roff + self.rpiv
                       + np.einsum("tij,tj->ti", Rf, inner))
        M[:, 3, 3] = 1.0
        return M


class Scene:
    """An FBX file: joint hierarchy, animation curves and global settings."""

    def __init__(self, path):
        self.path = path
        root, self.version = fbx_load(path)
        self.joints, self._curves, self._curve_nodes = {}, {}, {}
        self._objects(root)
        self._connections(root)
        self._globals(root)
        ts = [c.t for c in self._curves.values() if len(c.t)]
        self.times = np.unique(np.concatenate(ts)) if ts else np.array([0.0])
        self.by_name = {}
        for j in self.joints.values():
            self.by_name.setdefault(j.name, j)

    # ------------------------------------------------------------------
    def _objects(self, root):
        objs = root.find("Objects")
        if objs is None:
            raise ValueError(f"{self.path}: no Objects section")
        for m in objs.find_all("Model"):
            j = _Joint(m.props[0], m.props[1].split("\x00")[0], m.props[2])
            p = m.find("Properties70")
            j.t = _prop(p, "Lcl Translation", [0, 0, 0])
            j.r = _prop(p, "Lcl Rotation", [0, 0, 0])
            j.s = _prop(p, "Lcl Scaling", [1, 1, 1])
            j.pre = _prop(p, "PreRotation", [0, 0, 0])
            j.post = _prop(p, "PostRotation", [0, 0, 0])
            j.roff = _prop(p, "RotationOffset", [0, 0, 0])
            j.rpiv = _prop(p, "RotationPivot", [0, 0, 0])
            j.soff = _prop(p, "ScalingOffset", [0, 0, 0])
            j.spiv = _prop(p, "ScalingPivot", [0, 0, 0])
            j.order = int(_prop(p, "RotationOrder", 0))
            self.joints[j.uid] = j

        for c in objs.find_all("AnimationCurve"):
            kt, kv = c.find("KeyTime"), c.find("KeyValueFloat")
            if kt is None or kv is None:
                continue
            curve = _Curve(np.asarray(kt.props[0], float) / FBX_TIME_UNIT,
                           np.asarray(kv.props[0], float))
            self._curves[c.props[0]] = curve

        for cn in objs.find_all("AnimationCurveNode"):
            p = cn.find("Properties70")
            defaults = {q.props[0][2:]: float(q.props[4])
                        for q in (p.children if p is not None else [])
                        if q.props[0].startswith("d|")}
            self._curve_nodes[cn.props[0]] = ({}, defaults)

    def _connections(self, root):
        conns = root.find("Connections")
        for c in (conns.children if conns is not None else []):
            kind, src, dst = c.props[0], c.props[1], c.props[2]
            if kind == "OO" and src in self.joints and dst in self.joints:
                self.joints[src].parent = self.joints[dst]
                self.joints[dst].children.append(self.joints[src])
            elif kind == "OP":
                prop = c.props[3]
                if src in self._curves and dst in self._curve_nodes:
                    if prop.startswith("d|"):
                        self._curve_nodes[dst][0][prop[2:]] = self._curves[src]
                elif src in self._curve_nodes and dst in self.joints:
                    self.joints[dst].curves[prop] = self._curve_nodes[src]

    def _globals(self, root):
        gs = root.find("GlobalSettings")
        p = gs.find("Properties70") if gs is not None else None
        self.up_axis = int(_prop(p, "UpAxis", 1))
        self.up_sign = _prop(p, "UpAxisSign", 1)
        self.metres_per_unit = _prop(p, "UnitScaleFactor", 1.0) / 100.0
        mode = int(_prop(p, "TimeMode", 0))
        custom = _prop(p, "CustomFrameRate", -1.0)
        self.fps = _TIME_MODE_FPS.get(mode, custom if custom > 0 else None)

    # ------------------------------------------------------------------
    def resolve(self, name):
        """Node lookup, tolerant of Theia's 'person_N:' namespace prefix."""
        if name in self.by_name:
            return self.by_name[name]
        hits = sorted(n for n in self.by_name if n.split(":")[-1] == name)
        return self.by_name[hits[0]] if hits else None

    def global_transforms(self, all_nodes=False):
        """Global 4x4 over every key time -> {name: (T,4,4)}.

        Limb nodes only by default; all_nodes=True also returns the mesh
        nodes, which the floor check needs.
        """
        order, seen = [], set()

        def visit(j):
            if j.uid in seen:
                return
            seen.add(j.uid)
            if j.parent is not None:
                visit(j.parent)
            order.append(j)

        for j in self.joints.values():
            if all_nodes or j.type in ("LimbNode", "Root"):
                visit(j)

        out = {}
        for j in order:
            L = j.local_matrices(self.times)
            out[j.uid] = L if j.parent is None else out[j.parent.uid] @ L
        return {j.name: out[j.uid] for j in order
                if all_nodes or j.type == "LimbNode"}


class _Curve:
    def __init__(self, t, v):
        self.t, self.v = t, v

    def __call__(self, times):
        if len(self.t) == 1:
            return np.full(len(times), self.v[0])
        return np.interp(times, self.t, self.v)     # clamped at both ends




# %%==========================================================================
#  C. VISUAL3D METRICS READER  (plumbing)
# ===========================================================================

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
#  D. WAVEFRONT .OBJ I/O  (plumbing)
# ===========================================================================

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


def write_obj(path, lines, verts, norms, v_at, n_at, header=None):
    out = list(lines)
    for k, i in enumerate(v_at):
        out[i] = "v %.6f %.6f %.6f" % tuple(verts[k])
    for k, i in enumerate(n_at):
        out[i] = "vn %.6f %.6f %.6f" % tuple(norms[k])
    if header:
        out = ["# " + h for h in header] + out
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")




# %%==========================================================================
#  E. GEOMETRY HELPERS
# ===========================================================================

def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def orthonormalise(R):
    """Nearest rotation matrix, i.e. strip scale/shear. Works on (..., 3, 3)."""
    U, _, Vt = np.linalg.svd(R)
    D = np.ones(U.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


def yaw_matrix(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_and_tilt(R):
    """Heading about +Z, and how far R tips the vertical away from +Z."""
    return (float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
            float(np.degrees(np.arccos(np.clip((R @ [0, 0, 1.0])[2], -1, 1)))))


def rotation_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def angle_between(U, V):
    """Angle in degrees between corresponding rows of two (N,3) arrays."""
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", U, V), -1, 1)))


def up_axis_to_z(up_axis, up_sign):
    """Right-handed rotation taking a file's declared up-axis onto +Z."""
    s = 1.0 if up_sign >= 0 else -1.0
    if up_axis == 2:
        return np.eye(3) if s > 0 else np.diag([1.0, -1.0, -1.0])
    if up_axis == 1:
        return np.array([[1, 0, 0], [0, 0, -s], [0, s, 0]], float)
    return np.array([[0, 0, -s], [0, 1, 0], [s, 0, 0]], float)


def apply_transform(T, points):
    """Map (..., 3) MOVE4D positions (raw file coords) into Theia coords."""
    p = np.asarray(points, dtype=float)
    return (p.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]).reshape(p.shape)


def transform_orientation(T, R_m4d):
    """Re-express a MOVE4D orientation (..., 3, 3) in Theia coordinates.

    Coordinate system only. This does NOT convert MOVE4D's bone convention
    into Theia's -- see the frame-convention section of the report.
    """
    M = T[:3, :3]
    return (M / np.cbrt(abs(np.linalg.det(M)))) @ np.asarray(R_m4d, float)


def transform_points(T, P):
    """(N, 3) positions -> (N, 3), i.e. p_theia = T @ [p_m4d, 1]."""
    return P @ T[:3, :3].T + T[:3, 3]


def transform_normals(T, N):
    """(N, 3) unit normals -> (N, 3).

    Normals follow the inverse-transpose of the linear block, not the block
    itself. For a pure rotation those are the same thing, so this is belt
    and braces -- but it stays correct if the transform ever carries a scale.
    """
    if len(N) == 0:
        return N
    out = N @ np.linalg.inv(T[:3, :3])
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, n, out=np.zeros_like(out), where=n > 1e-12)




# %%==========================================================================
#  STEP 1 -- SOLVE THE MOVE4D -> THEIA3D TRANSFORM
# ===========================================================================

class Skeleton:
    """One FBX file, reduced to global segment poses in a common frame.

    After construction:
        .pos[node]   (F, 3)     global position, Z-up, metres
        .rot[node]   (F, 3, 3)  global orientation, Z-up
        .p[node]     (3,)       mean over the retained frames
        .R[node]     (3, 3)     mean over the retained frames
    Node names are this file's own, so use .lookup(key) with the keys from
    ALIGN_SEGMENTS / ALIGN_JOINTS to get at them by anatomy.
    """

    def __init__(self, path, system):
        self.path, self.system = path, os.path.basename(path)
        self.which = system                       # "theia" or "m4d"

        # --- step 1: load ---------------------------------------------
        self.scene = Scene(path)

        # --- step 2 + 3: every segment, local -> global ---------------
        G = self.scene.global_transforms()

        # --- step 4: common units and coordinate system ---------------
        # metres, and the declared up-axis rotated onto +Z.
        self.C = up_axis_to_z(self.scene.up_axis, self.scene.up_sign)
        self.scale = self.scene.metres_per_unit
        pos = {n: np.einsum("ij,tj->ti", self.C, M[:, :3, 3] * self.scale)
               for n, M in G.items()}
        rot = {n: np.einsum("ij,tjk->tik", self.C,
                            orthonormalise(M[:, :3, :3])) for n, M in G.items()}

        # --- drop bind / rest-pose frames -----------------------------
        names = sorted(pos)
        P = np.stack([pos[n] for n in names], axis=1)        # (F, N, 3)
        self.keep, self.dropped = self._good_frames(P)
        self.pos = {n: v[self.keep] for n, v in pos.items()}
        self.rot = {n: v[self.keep] for n, v in rot.items()}
        self.p = {n: v.mean(axis=0) for n, v in self.pos.items()}
        self.R = {n: orthonormalise(v.mean(axis=0))
                  for n, v in self.rot.items()}
        self.n_frames = len(self.scene.times)

    @staticmethod
    def _good_frames(P):
        """Keep frames whose POSE is near the median one.

        A bind/rest pose sits tens of centimetres from the data (Theia frame 0
        here is 514 mm out) and would wreck the mean. Comparing shape only --
        centroid removed -- means a subject who drifts is not penalised.
        """
        if P.shape[0] < 3:
            return np.arange(P.shape[0]), []
        centred = P - P.mean(axis=1, keepdims=True)
        dev = np.sqrt(((centred - np.median(centred, axis=0)) ** 2)
                      .sum(-1)).mean(-1)
        limit = (ALIGN_FRAME_FACTOR * np.median(dev)
                 + ALIGN_FRAME_FLOOR_M)
        keep = np.where(dev <= limit)[0]
        if len(keep) < 2:                      # never leave ourselves nothing
            return np.arange(P.shape[0]), []
        return keep, [(int(i), float(dev[i] * 1000))
                      for i in range(P.shape[0]) if i not in keep]

    # ------------------------------------------------------------------
    def lookup(self, key, table):
        """Anatomical key -> this file's node name, or None."""
        name = table[key][0 if self.which == "theia" else 1]
        node = self.scene.resolve(name)
        return node.name if node is not None and node.name in self.p else None

    def position(self, key, table=None):
        return self.p[self.lookup(key, table or ALIGN_JOINTS)]

    def orientation(self, key):
        return self.R[self.lookup(key, ALIGN_SEGMENTS)]

    def raw_position(self, key, table=None):
        """Mean position back in this file's OWN raw coordinates."""
        return self.C.T @ self.position(key, table) / self.scale


def detect_bone_axis(skel):
    """Which signed LOCAL axis does this rig run its bones along?

    Determined from the segments whose two endpoints are both known, by
    expressing the proximal->distal direction in the segment's own frame.
    Returned as (column, sign, worst |dot|). A worst |dot| near 1 means the
    rig is perfectly consistent and the long axis can then be read straight
    off the orientation matrix of ANY segment -- including the forearm, whose
    distal end (the wrist) neither rig exposes as a single clean node.
    """
    table = ALIGN_SEGMENT_ENDS if skel.which == "theia" else ALIGN_M4D_ENDS
    votes = []
    for seg, (a, b) in table.items():
        na, nb = skel.scene.resolve(a), skel.scene.resolve(b)
        node = skel.lookup(seg, ALIGN_SEGMENTS)
        if node is None or na is None or nb is None:
            continue
        local = skel.R[node].T @ unit(skel.p[nb.name] - skel.p[na.name])
        i = int(np.argmax(np.abs(local)))
        votes.append((i, np.sign(local[i]), abs(local[i])))
    if not votes:
        raise ValueError(f"{skel.path}: cannot determine the bone axis")
    col = max({v[0] for v in votes}, key=[v[0] for v in votes].count)
    sign = float(np.sign(sum(v[1] for v in votes if v[0] == col)))
    return col, sign, float(min(v[2] for v in votes if v[0] == col))


def long_axes(skel, axis):
    """Global long-axis unit vector of every fitted segment. -> (N, 3)"""
    col, sign, _ = axis
    return np.array([sign * skel.orientation(s)[:, col] for s in ALIGN_SEGMENTS])


def observations(th, m4, ax_t, ax_m):
    """Every matched direction the rotation is fitted to. -> (U, V, labels)"""
    U = list(long_axes(th, ax_t))
    V = list(long_axes(m4, ax_m))
    labels = [(s, "long axis") for s in ALIGN_SEGMENTS]
    if USE_ML_AXES:
        for name, a, b in ALIGN_ML_AXES:
            U.append(unit(th.position(b) - th.position(a)))
            V.append(unit(m4.position(b) - m4.position(a)))
            labels.append((name, "ML"))
    return np.array(U), np.array(V), labels


def solve_rotation(U, V, vertical_only):
    """Rotation R minimising sum ||u_i - R v_i||^2 over the segment axes.

    Unconstrained this is Wahba/Kabsch: one SVD. Constrained to the vertical
    it has a closed form -- maximising sum u.Rz(th)v over th gives
        th = atan2( sum (u_y v_x - u_x v_y), sum (u_x v_x + u_y v_y) )
    which is the right way to impose the constraint. Fitting 3 DOF and then
    reading the heading off the result is a different, worse estimator: the
    free tilt absorbs skeleton mismatch and drags the heading with it.
    """
    if vertical_only:
        num = float((U[:, 1] * V[:, 0] - U[:, 0] * V[:, 1]).sum())
        den = float((U[:, 0] * V[:, 0] + U[:, 1] * V[:, 1]).sum())
        return yaw_matrix(np.degrees(np.arctan2(num, den)))
    A, _, Bt = np.linalg.svd(U.T @ V)
    return A @ np.diag([1.0, 1.0, np.sign(np.linalg.det(A @ Bt))]) @ Bt


def solve_translation(A, B, R, lock_vertical=None):
    """t minimising sum ||a_i - (R b_i + t)||^2, which is just the mean.

    With lock_vertical, t_z is forced to zero instead of fitted -- for the
    case where both systems put z = 0 on the same floor plane and there is
    therefore no vertical offset to estimate. Either way the vertical offset
    the joint centres imply is returned separately, so it can be reported
    and cross-checked against the floor rather than only living inside the
    matrix.
    """
    lock = ASSUME_SHARED_FLOOR if lock_vertical is None else lock_vertical
    t = (A - B @ R.T).mean(axis=0)
    implied_vertical = float(t[2])
    if lock:
        t = t.copy()
        t[2] = 0.0
    return t, implied_vertical


def body_surface(scene, frame=None):
    """Posed skin vertices in the file's own raw coordinates. -> (N, 3)

    Handles both cases the two files present: a single mesh bound to the
    skeleton by skin clusters (MOVE4D), and rigid meshes parented to segment
    nodes (Theia).
    """
    root, _ = fbx_load(scene.path)
    objs = root.find("Objects")
    conns = root.find("Connections").children
    geo = {}
    for g in objs.find_all("Geometry"):
        v = g.find("Vertices")
        if v is not None and len(v.props[0]):
            geo[g.props[0]] = np.asarray(v.props[0], float).reshape(-1, 3)
    if not geo:
        return None

    G = scene.global_transforms(all_nodes=True)
    frame = len(scene.times) // 2 if frame is None else frame
    uid_name = {j.uid: j.name for j in scene.joints.values()}
    clusters = [d for d in objs.find_all("Deformer")
                if str(d.props[2]) == "Cluster"]

    # ---- skinned mesh (MOVE4D) ---------------------------------------
    if clusters:
        # FBX connects Model(bone) -> Cluster, so src is the bone.
        bone_of = {c.props[2]: uid_name.get(c.props[1]) for c in conns
                   if c.props[1] in uid_name}
        V = max(geo.values(), key=len)
        acc = np.zeros((len(V), 3))
        wsum = np.zeros(len(V))
        for d in clusters:
            bone = bone_of.get(d.props[0])
            idx, wts = d.find("Indexes"), d.find("Weights")
            link = d.find("TransformLink")
            if bone is None or bone not in G or idx is None or link is None:
                continue
            # FBX stores matrices row-major for row vectors; transpose for ours
            TL = np.asarray(link.props[0], float).reshape(4, 4).T
            I = np.asarray(idx.props[0], int)
            W = np.asarray(wts.props[0], float)
            Mx = G[bone][frame] @ np.linalg.inv(TL)
            acc[I] += W[:, None] * (V[I] @ Mx[:3, :3].T + Mx[:3, 3])
            wsum[I] += W
        ok = wsum > 1e-9
        return acc[ok] / wsum[ok, None]

    # ---- rigid meshes parented to segments (Theia) -------------------
    pts = []
    for c in conns:
        if c.props[1] in geo and c.props[2] in uid_name:
            name = uid_name[c.props[2]]
            if name not in G:
                continue
            Mx = G[name][frame]
            pts.append(geo[c.props[1]] @ Mx[:3, :3].T + Mx[:3, 3])
    return np.vstack(pts) if pts else None


def floor_check(skel):
    """Where does the body surface sit relative to z = 0?

    ASSUME_SHARED_FLOOR rests on both systems putting their origin on the
    floor. This measures it: if the lowest skin vertices cluster around zero
    the subject is standing on the floor plane, and the vertical offset
    between the two coordinate systems really is nothing to fit. If the
    surface floats, the assumption is wrong for that file.
    """
    raw = body_surface(skel.scene)
    if raw is None:
        return None
    P = np.einsum("ij,nj->ni", skel.C, raw * skel.scale)     # common Z-up, m
    z = P[:, 2]
    return dict(n_vertices=int(len(P)),
                lowest_mm=float(z.min() * 1000),
                p1_mm=float(np.percentile(z, 1) * 1000),
                highest_mm=float(z.max() * 1000),
                height_m=float(z.max() - z.min()),
                within_5mm_of_floor=int((np.abs(z) < 0.005).sum()),
                within_10mm_of_floor=int((np.abs(z) < 0.010).sum()))


def heading_uncertainty(U, V, labels):
    """Spread of the heading across subsets of the very same data.

    The residual cannot tell you this. Heading is constrained only by the
    HORIZONTAL component of each direction, so an observation that is nearly
    vertical amplifies its own error by 1/leverage when it votes on yaw. The
    honest measure is: refit on independent subsets and see how far apart
    they land. That spread is the uncertainty, whatever the residual says.
    """
    def yaw_of(mask):
        if mask.sum() < 2:
            return None
        return yaw_and_tilt(solve_rotation(U[mask], V[mask], True))[0]

    n = len(U)
    full = yaw_of(np.ones(n, bool))

    loo = []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        y = yaw_of(m)
        if y is not None:
            loo.append((labels[i][0], y))

    kinds = np.array([k for _, k in labels])
    names = np.array([s for s, _ in labels])
    groups = {
        "segment long axes": kinds == "long axis",
        "bilateral ML axes": kinds == "ML",
        "legs": np.array([n_.startswith(("l_thigh", "r_thigh", "l_shank",
                                         "r_shank", "hip_", "knee_"))
                          for n_ in names]),
        "arms": np.array([n_.startswith(("l_uarm", "r_uarm", "l_larm",
                                         "r_larm", "shoulder_", "elbow_"))
                          for n_ in names]),
        "left side": np.array([n_.startswith("l_") for n_ in names]),
        "right side": np.array([n_.startswith("r_") for n_ in names]),
    }
    subsets = []
    for name, mask in groups.items():
        y = yaw_of(mask)
        if y is None:
            continue
        lev = float(np.linalg.norm(V[mask][:, :2], axis=1).mean())
        subsets.append(dict(subset=name, yaw_deg=y, leverage=lev,
                            n=int(mask.sum()), informative=lev >= 0.30))

    loo_vals = [y for _, y in loo]
    good = [s["yaw_deg"] for s in subsets if s["informative"]]
    return dict(
        yaw_deg=full,
        leave_one_out_spread_deg=(float(max(loo_vals) - min(loo_vals))
                                  if len(loo_vals) > 1 else 0.0),
        worst_single_observation=max(loo, key=lambda x: abs(x[1] - full))[0]
        if loo else None,
        subset_spread_deg=float(max(good) - min(good)) if len(good) > 1 else 0.0,
        subsets=subsets,
        leverage=dict(zip(names.tolist(),
                          np.linalg.norm(V[:, :2], axis=1).round(3).tolist())),
    )


def mirror_test(U, V):
    """Is the best-fitting linear map between the segment axes a REFLECTION?

    Kabsch forces a proper rotation by flipping the smallest singular
    direction. The honest test is whether it had to: if the raw SVD product
    has a negative determinant, the data are better explained by a mirror
    than by any rotation, which means one of the files is mirrored. No rigid
    transform can fix that, so it has to be caught rather than fitted.
    """
    A, _, Bt = np.linalg.svd(U.T @ V)
    return float(np.linalg.det(A @ Bt))


def alignment_checks(th, m4, R, t, T, keys):
    """Units, segment lengths, and an end-to-end test of T."""
    lengths = []
    for seg, (a, b) in ALIGN_SEGMENT_ENDS.items():
        ma, mb = ALIGN_M4D_ENDS[seg]
        na, nb = th.scene.resolve(a), th.scene.resolve(b)
        qa, qb = m4.scene.resolve(ma), m4.scene.resolve(mb)
        if None in (na, nb, qa, qb):
            continue
        lt = float(np.linalg.norm(th.p[nb.name] - th.p[na.name]))
        lm = float(np.linalg.norm(m4.p[qb.name] - m4.p[qa.name]))
        lengths.append(dict(segment=seg, theia_m=lt, m4d_m=lm, ratio=lt / lm))

    # end-to-end: push the RAW MOVE4D joints through T and compare with the
    # RAW Theia joints. This is what catches a mistake in the up-axis or unit
    # bookkeeping behind T, which the fitted residual above cannot see.
    raw = apply_transform(T, np.array([m4.raw_position(k) for k in keys]))
    tgt = np.array([th.raw_position(k) for k in keys])
    raw_err = np.linalg.norm(raw - tgt, axis=1) * th.scale * 1000

    return dict(segment_lengths=lengths,
                median_length_ratio=float(np.median([r["ratio"]
                                                     for r in lengths])),
                raw_check_rms_mm=float(np.sqrt((raw_err ** 2).mean())))


def frame_conventions(th, m4, R):
    """The per-segment bone-roll offsets: why the full 3x3 is not fitted on.

        O_s = (R . R_m4d,s)^T . R_theia,s
    If every O_s were the same rotation, the model R_theia = G . R_m4d . O
    would be identifiable from one pose and worth fitting. They are not.
    """
    O = {s: (R @ m4.orientation(s)).T @ th.orientation(s) for s in ALIGN_SEGMENTS}
    labels = list(ALIGN_SEGMENTS)
    pair = np.array([[rotation_angle(O[a].T @ O[b]) for b in labels]
                     for a in labels])
    return dict(offset_deg={s: rotation_angle(O[s]) for s in labels},
                max_disagreement_deg=float(pair.max()),
                mean_disagreement_deg=float(pair[np.triu_indices(len(labels), 1)].mean()))


def solve_alignment(theia_path=None, m4d_path=None, vertical_only=None, verbose=True):
    """Run all six steps. Returns (report, T, theia, m4d)."""
    vertical_only = ALIGN_VERTICAL_ONLY if vertical_only is None else vertical_only

    # --- steps 1-4 ----------------------------------------------------
    th = Skeleton(theia_path or THEIA_FBX, "theia")
    m4 = Skeleton(m4d_path or M4D_FBX, "m4d")

    missing = [k for k in ALIGN_JOINTS
               if th.lookup(k, ALIGN_JOINTS) is None or m4.lookup(k, ALIGN_JOINTS) is None]
    if missing:
        raise ValueError(f"joints not found in one or both files: {missing}. "
                         f"Check the ALIGN_JOINTS / ALIGN_SEGMENTS tables.")

    ax_t, ax_m = detect_bone_axis(th), detect_bone_axis(m4)

    # --- step 5 -------------------------------------------------------
    U, V, labels = observations(th, m4, ax_t, ax_m)
    R = solve_rotation(U, V, vertical_only)

    # --- step 6 -------------------------------------------------------
    keys = list(ALIGN_JOINTS)
    A = np.array([th.position(k) for k in keys])
    B = np.array([m4.position(k) for k in keys])
    t, vertical_gap = solve_translation(A, B, R)

    # --- assemble T in RAW file coordinates ---------------------------
    # p_common = C @ (metres_per_unit * p_raw), so undo both on the Theia side
    T = np.eye(4)
    T[:3, :3] = th.C.T @ R @ m4.C * (m4.scale / th.scale)
    T[:3, 3] = th.C.T @ t / th.scale

    # --- residuals ----------------------------------------------------
    seg_err = angle_between(U, V @ R.T)
    resid = A - (B @ R.T + t)
    pos_err = np.linalg.norm(resid, axis=1) * 1000
    # With the vertical locked, the horizontal residual is what the fit
    # actually minimised; the vertical one is the skeleton disagreement the
    # transform is deliberately no longer absorbing. Quoting them together
    # would hide that.
    horiz_err = np.linalg.norm(resid[:, :2], axis=1) * 1000
    vert_err = resid[:, 2] * 1000

    # the same fit under the other rotation model, for comparison
    R_alt = solve_rotation(U, V, not vertical_only)
    t_alt, _ = solve_translation(A, B, R_alt)
    resid_alt = A - (B @ R_alt.T + t_alt)
    alt = dict(vertical_only=not vertical_only,
               yaw_deg=yaw_and_tilt(R_alt)[0], tilt_deg=yaw_and_tilt(R_alt)[1],
               segment_rms_deg=float(np.sqrt((angle_between(U, V @ R_alt.T) ** 2).mean())),
               # horizontal, because that is what the translation fitted once
               # the vertical is locked; comparing totals would just compare
               # the same skeleton disagreement twice
               position_rms_mm=float(np.sqrt((np.linalg.norm(
                   resid_alt[:, :2], axis=1) ** 2).mean()) * 1000))

    yaw, tilt = yaw_and_tilt(R)
    report = dict(
        T_m4d_to_theia=T.tolist(),
        rotation=R.tolist(), translation_m=t.tolist(),
        vertical_only=vertical_only, yaw_deg=yaw, tilt_deg=tilt,
        total_rotation_deg=rotation_angle(R),
        bone_axis=dict(
            theia=dict(axis=f"{'+' if ax_t[1] > 0 else '-'}{'XYZ'[ax_t[0]]}",
                       worst_dot=ax_t[2]),
            m4d=dict(axis=f"{'+' if ax_m[1] > 0 else '-'}{'XYZ'[ax_m[0]]}",
                     worst_dot=ax_m[2])),
        direction_residual_deg=dict(
            rms=float(np.sqrt((seg_err ** 2).mean())), max=float(seg_err.max()),
            per_direction={lab: dict(residual_deg=float(e), kind=kind)
                           for (lab, kind), e in zip(labels, seg_err)}),
        segment_residual_deg=dict(
            rms=float(np.sqrt((seg_err[:len(ALIGN_SEGMENTS)] ** 2).mean())),
            per_segment=dict(zip(ALIGN_SEGMENTS,
                                 seg_err[:len(ALIGN_SEGMENTS)].round(3).tolist()))),
        position_residual_mm=dict(
            rms=float(np.sqrt((pos_err ** 2).mean())), max=float(pos_err.max()),
            horizontal_rms=float(np.sqrt((horiz_err ** 2).mean())),
            vertical_rms=float(np.sqrt((vert_err ** 2).mean())),
            per_joint=dict(zip(keys, pos_err.round(2).tolist())),
            vertical_per_joint=dict(zip(keys, vert_err.round(2).tolist()))),
        vertical=dict(
            locked=ASSUME_SHARED_FLOOR,
            implied_offset_mm=vertical_gap * 1000,
            residual_signs_agree=bool(np.all(vert_err > 0)
                                      or np.all(vert_err < 0)),
            floor_check=dict(theia=floor_check(th), m4d=floor_check(m4))
            if CHECK_FLOOR_FROM_MESH else None),
        heading_uncertainty=heading_uncertainty(U, V, labels),
        alternative_model=alt,
        frame_conventions=frame_conventions(th, m4, R),
        alignment_checks=dict(mirror_determinant=mirror_test(U, V),
                    **alignment_checks(th, m4, R, t, T, keys)),
    )
    if verbose:
        print_alignment_report(report, th, m4)
    return report, T, th, m4


def print_alignment_report(r, th, m4):
    W = 74
    mode = "vertical-axis only" if r["vertical_only"] else "full 3-DOF"
    print("=" * W)
    print(f"  MOVE4D -> THEIA3D    segment-orientation alignment  [{mode}]")
    print("=" * W)

    print("\n  STEPS 1-4: load, read segments, local->global, common frame")
    for s, lab in ((th, "Theia "), (m4, "MOVE4D")):
        sc = s.scene
        print(f"    {lab} {s.path}")
        print(f"           {len(sc.joints):>3d} nodes, {len(s.pos)} segments, "
              f"{s.n_frames} frames @ {sc.fps or '?'} fps, kept {len(s.keep)}")
        print(f"           up-axis {'XYZ'[sc.up_axis]} -> +Z   |   "
              f"{sc.metres_per_unit:g} m per file unit -> metres")
        for i, d in s.dropped:
            kind = "bind/rest pose" if d > 100 else "edge transient"
            print(f"           dropped frame {i}: {d:.0f} mm from the median "
                  f"pose ({kind}, not data)")

    c = r["alignment_checks"]
    det = c["mirror_determinant"]
    print(f"\n    mirroring      best-fit determinant {det:+.3f}   "
          f"{'OK, a proper rotation' if det > 0 else '*** ONE FILE IS MIRRORED ***'}")
    print(f"    scale          median segment length ratio "
          f"{c['median_length_ratio']:.4f}  (1.0 = same units)")
    for s in c["segment_lengths"]:
        print(f"      {s['segment']:9s} Theia {s['theia_m']:.4f} m | "
              f"M4D {s['m4d_m']:.4f} m | ratio {s['ratio']:.4f}")

    b = r["bone_axis"]
    print("\n  STEP 5: rotation from global segment orientations")
    print(f"    bone long axis   Theia = {b['theia']['axis']} local "
          f"(consistency {b['theia']['worst_dot']:.4f})")
    print(f"                     M4D   = {b['m4d']['axis']} local "
          f"(consistency {b['m4d']['worst_dot']:.4f})")
    dr = r["direction_residual_deg"]
    hu = r["heading_uncertainty"]
    print(f"\n    direction residual   RMS {dr['rms']:.2f} deg, "
          f"max {dr['max']:.2f} deg"
          f"   (segments alone: {r['segment_residual_deg']['rms']:.2f} deg)")
    print(f"      {'direction':16s} {'residual':>9s} {'leverage':>9s}  kind")
    for k, v in sorted(dr["per_direction"].items(),
                       key=lambda x: -x[1]["residual_deg"]):
        print(f"      {k:16s} {v['residual_deg']:6.2f} deg "
              f"{hu['leverage'].get(k, float('nan')):9.3f}  {v['kind']}")
    print("      leverage = horizontal component. Only that part constrains the")
    print("      heading, so a near-vertical axis amplifies its own error.")

    print("\n  STEP 6: translation from joint positions")
    pr = r["position_residual_mm"]
    vt = r["vertical"]
    if vt["locked"]:
        print("    vertical LOCKED (ASSUME_SHARED_FLOOR): t_z = 0, and the")
        print(f"    {vt['implied_offset_mm']:+.1f} mm the joint centres imply "
              f"is reported below instead")
        print("    of being absorbed into the matrix.")
    else:
        print(f"    vertical FITTED: t_z = {vt['implied_offset_mm']:+.1f} mm. "
              f"That figure is the true")
        print("    calibration offset PLUS any systematic vertical bias "
              "between the")
        print("    two skeletons' joint-centre definitions; one static pose "
              "cannot")
        print("    separate them. See the floor check below.")
    print(f"\n    horizontal residual   RMS {pr['horizontal_rms']:5.1f} mm"
          "   <- what the fit minimised")
    print(f"    vertical residual     RMS {pr['vertical_rms']:5.1f} mm"
          + ("   <- NOT fitted, definition gap" if vt["locked"]
             else "   <- what is left after fitting t_z"))
    print(f"    total                 RMS {pr['rms']:5.1f} mm, "
          f"max {pr['max']:.1f} mm")
    print(f"\n      {'joint':11s} {'total':>9s} {'vertical':>9s}")
    for k, v in sorted(pr["per_joint"].items(), key=lambda x: -x[1]):
        print(f"      {k:11s} {v:6.1f} mm {pr['vertical_per_joint'][k]:+8.1f}")

    fc = vt.get("floor_check")
    if fc and (fc["theia"] or fc["m4d"]):
        print("\n    floor check from the body SURFACE (a joint centre is an")
        print("    inference; the sole of a foot is not):")
        for lab, d in (("Theia model", fc["theia"]), ("MOVE4D scan", fc["m4d"])):
            if not d:
                continue
            print(f"      {lab:12s} {d['n_vertices']:6d} verts | lowest "
                  f"{d['lowest_mm']:+7.1f} mm | top {d['highest_mm']:7.1f} mm"
                  f" | {d['within_5mm_of_floor']:5d} within 5 mm of z=0")
        m, th_fc = fc.get("m4d"), fc.get("theia")
        if m:
            on_floor = abs(m["lowest_mm"]) < 20 and m["within_5mm_of_floor"] > 50
            if on_floor:
                print("      -> the scanned subject stands ON the floor, so "
                      "MOVE4D's floor plane")
                print(f"         is z = {m['lowest_mm']:+.1f} mm: its origin "
                      f"is on the floor.")
            else:
                print("      -> the scan does not reach z = 0, so MOVE4D's "
                      "floor cannot be")
                print("         located this way. Treat the rest of this "
                      "block with caution.")
            tz = vt["implied_offset_mm"]
            if on_floor and th_fc:
                # the MOVE4D floor (z=0) maps to z = t_z in Theia coordinates
                clear_fit = th_fc["lowest_mm"] - tz
                print(f"\n      A vertical offset of {tz:+.1f} mm puts Theia's "
                      f"floor at z = {tz:+.1f} mm in")
                print("      Theia coordinates, i.e. its origin that far "
                      "BELOW the floor. Theia's")
                print(f"      lowest model vertex then clears the floor by "
                      f"{clear_fit:+.1f} mm, against")
                print(f"      {th_fc['lowest_mm']:+.1f} mm if the two floors "
                      f"were assumed shared.")
                print("      Theia's meshes are a body MODEL (a ~270-vertex "
                      "primitive per limb),")
                print("      not a measurement, so neither figure is proof -- "
                      "but a simplified")
                if abs(clear_fit) < abs(th_fc["lowest_mm"]):
                    print("      foot primitive sitting the SMALLER distance "
                          "off the floor is the")
                    print("      more plausible of the two, which independently "
                          "supports a real")
                    print("      vertical offset rather than a shared floor.")
                else:
                    print("      foot primitive would have to sit FURTHER off "
                          "the floor for the")
                    print("      fitted offset to hold, which argues the other "
                          "way. Worth a look.")

        # the residual signs are the other tell
        if vt["residual_signs_agree"]:
            print("\n      !! every vertical residual has the same sign. That "
                  "is the signature of")
            print("      a real vertical offset being SUPPRESSED, not a "
                  "definition gap."
                  + ("\n      Set ASSUME_SHARED_FLOOR = False."
                     if vt["locked"] else ""))
        else:
            print("\n      vertical residuals scatter about zero, which is "
                  "what you want:")
            print("      the offset has been absorbed and only genuine "
                  "definition")
            print("      differences are left standing.")

    print("\n" + "-" * W)
    print("  RESULT     p_theia = T @ [p_m4d, 1]   (raw FBX coords both sides)")
    print("-" * W)
    for row in np.array(r["T_m4d_to_theia"]):
        print("    [" + "  ".join(f"{v:10.6f}" for v in row) + "]")
    print(f"\n    heading (yaw)    {r['yaw_deg']:+8.3f} deg")
    print(f"    tilt             {r['tilt_deg']:8.3f} deg"
          + ("   (constrained to zero)" if r["vertical_only"] else ""))
    tt = r["translation_m"]
    print(f"    translation      [{tt[0]:+.4f} {tt[1]:+.4f} {tt[2]:+.4f}] m"
          "   in the common Z-up frame"
          + ("   (t_z locked to 0)" if r["vertical"]["locked"] else ""))
    print(f"\n    end-to-end check on RAW coordinates: "
          f"{c['raw_check_rms_mm']:.1f} mm RMS"
          f"   (must equal {pr['rms']:.1f})")
    if abs(c["raw_check_rms_mm"] - pr["rms"]) > 0.1:
        print("    *** MISMATCH: the up-axis / unit bookkeeping behind T is "
              "wrong. Do not use this matrix. ***")

    hu = r["heading_uncertainty"]
    print("\n" + "-" * W)
    print("  HOW WELL IS THE HEADING DETERMINED?")
    print("-" * W)
    print("    The residual cannot tell you. Refit on independent subsets of")
    print("    the same data and see how far apart they land -- that spread")
    print("    IS the uncertainty.")
    print(f"\n      {'subset':20s} {'yaw':>8s} {'leverage':>9s} {'n':>3s}")
    for s in hu["subsets"]:
        mark = "" if s["informative"] else "   (low leverage, ignore)"
        print(f"      {s['subset']:20s} {s['yaw_deg']:+7.2f}d "
              f"{s['leverage']:9.3f} {s['n']:3d}{mark}")
    print(f"\n    spread over informative subsets   "
          f"{hu['subset_spread_deg']:.2f} deg")
    print(f"    leave-one-observation-out spread  "
          f"{hu['leave_one_out_spread_deg']:.2f} deg"
          f"   (most influential: {hu['worst_single_observation']})")
    print("\n    The two say different things. Leave-one-out being small "
          "means no")
    print("    single observation is driving the answer. The subset spread "
          "being")
    print("    larger means independent PARTS OF THE BODY disagree, which "
          "averaging")
    print("    cannot remove. Quote the larger one. Note that a subset with "
          "low")
    print("    leverage amplifies its own residual by 1/leverage, so some of "
          "that")
    print("    spread is noise rather than bias -- which is exactly why more "
          "than")
    print("    one standing pose, at different headings, is worth capturing.")
    quoted = max(hu["subset_spread_deg"], hu["leave_one_out_spread_deg"])
    err = np.radians(quoted)
    print(f"\n    taking {quoted:.2f} deg, that displaces a point")
    print(f"      1 m from the origin by {err * 1000:5.1f} mm")
    print(f"      3 m from the origin by {err * 3000:5.1f} mm")
    if not USE_ML_AXES:
        print("\n      USE_ML_AXES is off. Turning it on typically cuts this")
        print("      spread several-fold: the segment long axes are nearly")
        print("      vertical and carry little heading information.")

    a = r["alternative_model"]
    other = "vertical-axis only" if a["vertical_only"] else "full 3-DOF"
    print("\n    the other rotation model, for comparison:")
    print(f"      {mode:19s} yaw {r['yaw_deg']:+6.2f}  tilt {r['tilt_deg']:5.2f}"
          f"  dir {dr['rms']:5.2f} deg  horiz {pr['horizontal_rms']:5.1f} mm"
          f"   <- used")
    print(f"      {other:19s} yaw {a['yaw_deg']:+6.2f}  tilt {a['tilt_deg']:5.2f}"
          f"  dir {a['segment_rms_deg']:5.2f} deg  horiz "
          f"{a['position_rms_mm']:5.1f} mm")
    if r["vertical_only"] and a["position_rms_mm"] >= pr["horizontal_rms"] - 0.5:
        print(f"      -> the {a['tilt_deg']:.2f} deg tilt the 3-DOF fit wants "
              f"does not improve the position")
        print("         fit, so it is absorbing skeleton mismatch rather than "
              "a real")
        print("         calibration difference. Keep ALIGN_VERTICAL_ONLY = True.")

    if REPORT_FRAME_CONVENTIONS:
        fc = r["frame_conventions"]
        print("\n" + "-" * W)
        print("  WHY THE FULL 3x3 ORIENTATION IS NOT FITTED ON")
        print("-" * W)
        print("    Per-segment bone-roll offset  O_s = (R . R_m4d,s)^T . "
              "R_theia,s :")
        for s, v in sorted(fc["offset_deg"].items(), key=lambda x: -x[1]):
            print(f"      {s:9s} {v:7.2f} deg")
        print("\n    These would all be EQUAL if the two rigs shared a bone")
        print("    convention, and the full orientations would then be safe "
              "to fit.")
        print(f"    They disagree with each other by "
              f"{fc['mean_disagreement_deg']:.1f} deg on average, "
              f"{fc['max_disagreement_deg']:.1f} deg at worst.")
        print("    So the roll is naming convention, not anatomy. The long "
              "axis --")
        print(f"    which agrees to "
              f"{r['segment_residual_deg']['rms']:.1f} deg RMS above -- is "
              f"the part that means")
        print("    something, and it is what the fit uses.")
    print("=" * W)


def alignment_figure(report, th, m4, path=None):
    """Skeleton overlay after the transform, plus the two residual bars."""
    import matplotlib.pyplot as plt

    R = np.array(report["rotation"])
    t = np.array(report["translation_m"])
    pt = {k: th.position(k) for k in ALIGN_JOINTS}
    pm = {k: R @ m4.position(k) + t for k in ALIGN_JOINTS}
    bones = [("l_hip", "l_knee"), ("r_hip", "r_knee"),
             ("l_shoulder", "l_elbow"), ("r_shoulder", "r_elbow"),
             ("l_hip", "r_hip"), ("l_shoulder", "r_shoulder"),
             ("l_hip", "l_shoulder"), ("r_hip", "r_shoulder")]

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.32, wspace=0.28)
    for col, (i, j, name) in enumerate([(0, 2, "front (X-Z)"),
                                        (1, 2, "side (Y-Z)"),
                                        (0, 1, "top (X-Y)")]):
        ax = fig.add_subplot(gs[0, col])
        for pos, colour, style, lab in ((pt, "#1f4e79", "-", "Theia3D"),
                                        (pm, "#c0392b", "--", "MOVE4D -> Theia")):
            for a, b in bones:
                ax.plot([pos[a][i], pos[b][i]], [pos[a][j], pos[b][j]],
                        style, color=colour, lw=1.7, zorder=2)
            ax.scatter([p[i] for p in pos.values()],
                       [p[j] for p in pos.values()], s=32, color=colour, zorder=3)
            ax.plot([], [], style, color=colour, lw=1.7, label=lab)
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.legend(fontsize=8, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    d = report["segment_residual_deg"]["per_segment"]
    ks = sorted(d, key=d.get)
    ax.barh(ks, [d[k] for k in ks], color="#c0392b")
    ax.set_xlabel("segment orientation residual (deg)", fontsize=9)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    ax = fig.add_subplot(gs[1, 1:])
    pr = report["position_residual_mm"]
    d, dz = pr["per_joint"], pr["vertical_per_joint"]
    ks = sorted(d, key=lambda k: -d[k])
    horiz = [float(np.hypot(d[k], 0) ** 2 - dz[k] ** 2) ** 0.5
             if d[k] ** 2 > dz[k] ** 2 else 0.0 for k in ks]
    x = np.arange(len(ks))
    ax.bar(x - 0.2, horiz, 0.4, color="#1f4e79", label="horizontal (fitted)")
    ax.bar(x + 0.2, [abs(dz[k]) for k in ks], 0.4, color="#aaaaaa",
           label="vertical (" + ("not fitted" if report["vertical"]["locked"]
                                 else "fitted") + ")")
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.legend(fontsize=8)
    ax.set_ylabel("joint position residual (mm)", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    fig.suptitle("MOVE4D -> Theia3D alignment check", fontsize=12, y=0.97)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig




# %%==========================================================================
#  STEP 2 -- PUT THE MESHES INTO THEIA COORDINATES
# ===========================================================================

def load_transform():
    """The 4x4, from T_MATRIX or from T_FILE (.npy or whitespace .txt)."""
    if T_MATRIX is not None:
        T = np.asarray(T_MATRIX, float)
    elif T_FILE and os.path.exists(T_FILE):
        T = (np.load(T_FILE) if T_FILE.endswith(".npy")
             else np.loadtxt(T_FILE))
    else:
        raise FileNotFoundError(
            f"no transform: set T_MATRIX, or run m4d_to_theia_transform.py "
            f"to produce {T_FILE!r}")
    if T.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {T.shape}")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError(f"bottom row of T is {T[3]}, expected [0 0 0 1]")
    return T


def scan_bounds(fbx_path):
    """Bounding box of the MOVE4D scan surface, in raw file coordinates.

    Reuses the skinning from step 1, so there is one implementation of it.
    """
    if not fbx_path or not os.path.exists(fbx_path):
        return None
    surf = body_surface(Scene(fbx_path))
    return None if surf is None else (surf.min(0), surf.max(0))


def check_input_frame(name, V, bounds):
    """Is this mesh really in raw MOVE4D coordinates?

    A mesh segmented out of the scan must sit inside the scan's own bounding
    box. Re-centring, a unit change or an axis flip all show up here, and all
    of them would silently produce a wrong-looking result otherwise.
    """
    if bounds is None:
        return ["scan not available, input frame NOT verified"]
    lo, hi = bounds
    pad = 0.02                                   # 20 mm of slack
    inside = np.all(V.min(0) >= lo - pad) and np.all(V.max(0) <= hi + pad)
    if inside:
        return []
    msgs = [f"{name}: vertices fall OUTSIDE the MOVE4D scan's bounding box, "
            f"so this mesh is probably not in raw MOVE4D coordinates."]
    span_v, span_s = V.max(0) - V.min(0), hi - lo
    ratio = np.median(span_v / np.maximum(span_s, 1e-9))
    if ratio > 10:
        msgs.append(f"  its extent is ~{ratio:.0f}x the scan's -- millimetres "
                    f"rather than metres?")
    elif ratio < 0.1:
        msgs.append(f"  its extent is ~{ratio:.3f}x the scan's -- check the "
                    f"units.")
    else:
        msgs.append("  extent looks right, so it is more likely re-centred "
                    "or axis-flipped than rescaled.")
    return msgs


def check_floor(name, V_theia, T):
    """After transforming, does the sole sit on Theia's floor?

    MOVE4D's floor is z = 0 in its own file, so it lands at z = T[2, 3] in
    Theia coordinates. A foot mesh's lowest vertices should be right there.
    This is the end-to-end test: it fails if the transform was applied to
    the wrong axis convention, or not at all.
    """
    expected = T[2, 3] * 1000.0
    sole = np.percentile(V_theia[:, 2], 0.5) * 1000.0
    off = sole - expected
    ok = abs(off) <= FLOOR_TOLERANCE_MM
    return dict(ok=ok, expected_mm=expected, sole_mm=sole, offset_mm=off)


def theia_reference(fbx_path, bbox, pad=0.05):
    """Theia's own surface and joint centres, clipped to `bbox`. -> (pts, joints)

    Clipping to the transformed mesh's own bounding box keeps the comparison
    readable whatever part of the body was segmented out -- a foot, a hand,
    a whole leg.
    """
    if not fbx_path or not os.path.exists(fbx_path):
        return None, {}
    sc = Scene(fbx_path)
    surf = body_surface(sc)
    lo, hi = bbox[0] - pad, bbox[1] + pad
    if surf is not None:
        inside = np.all((surf >= lo) & (surf <= hi), axis=1)
        surf = surf[inside] if inside.any() else None
    G = sc.global_transforms()
    frame = len(sc.times) // 2
    joints = {}
    for name, M4 in G.items():
        p = M4[frame, :3, 3]
        if np.all((p >= lo) & (p <= hi)):
            joints[name.split(":")[-1]] = p
    return surf, joints


def mesh_placement_figure(results, T, path=None):
    """Three orthographic views of every transformed mesh, in Theia coords."""
    import matplotlib.pyplot as plt

    pts = []
    for r in results:
        V = read_obj(r["output"])[1]
        pts.append(V)
    if not pts:
        return None
    allpts = np.vstack(pts)
    bbox = (allpts.min(0), allpts.max(0))
    ref, joints = theia_reference(COMPARE_WITH_THEIA, bbox)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    views = [(0, 1, "top  (X-Y)"), (1, 2, "side  (Y-Z)"), (0, 2, "front  (X-Z)")]
    step = max(1, len(allpts) // 3000)
    for ax, (i, j, name) in zip(axes, views):
        ax.scatter(allpts[::step, i], allpts[::step, j], s=1.2, c="#c0392b",
                   alpha=0.45, label="MOVE4D mesh, transformed", rasterized=True)
        if ref is not None and len(ref):
            s2 = max(1, len(ref) // 3000)
            ax.scatter(ref[::s2, i], ref[::s2, j], s=1.2, c="#1f4e79",
                       alpha=0.45, label="Theia body model", rasterized=True)
        for nm, p in joints.items():
            ax.plot(p[i], p[j], "k+", ms=11, mew=2)
            ax.annotate(nm, (p[i], p[j]), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        if j == 2:                                  # a vertical view
            ax.axhline(T[2, 3], color="k", ls=":", lw=1)
            ax.annotate(f"Theia floor (z = {T[2, 3] * 1000:+.1f} mm)",
                        (ax.get_xlim()[0], T[2, 3]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, markerscale=6, loc="upper left")
    fig.suptitle("MOVE4D mesh placed in Theia3D coordinates   "
                 "(black + = Theia joint centres)", fontsize=11, y=1.0)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def meshes_to_theia(T=None, paths=None, verbose=True):
    """Transform every .obj in `paths`. Returns a list of result dicts."""
    paths = MESH_FILES_RAW if paths is None else paths
    T = load_transform() if T is None else np.asarray(T, float)
    bounds = scan_bounds(CHECK_AGAINST_SCAN) if CHECK_AGAINST_SCAN else None

    if verbose:
        W = 74
        print("=" * W)
        print("  MOVE4D .obj  ->  THEIA3D coordinates")
        print("=" * W)
        print("\n  T (raw MOVE4D -> raw Theia):")
        for row in T:
            print("    [" + "  ".join(f"{v:10.6f}" for v in row) + "]")
        print("\n  MOVE4D is Y-up, Theia is Z-up, and that conversion is part")
        print("  of T -- so the OUTPUT files are Z-UP. A Y-up viewer will show")
        print("  them lying on their side. That is correct.")
        if bounds is None and CHECK_AGAINST_SCAN:
            print(f"\n  ! {CHECK_AGAINST_SCAN} not readable -- the input frame "
                  f"cannot be verified.")

    results = []
    for path in paths:
        if not os.path.exists(path):
            print(f"\n  !! {path}: not found, skipped")
            continue
        lines, V, N, v_at, n_at = read_obj(path)
        warnings = check_input_frame(os.path.basename(path), V, bounds)

        Vt = transform_points(T, V)
        Nt = transform_normals(T, N)

        stem, ext = os.path.splitext(os.path.basename(path))
        out_dir = OUT_DIR or (os.path.dirname(path) or ".")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, stem + OUT_SUFFIX + ext)
        write_obj(out_path, lines, Vt, Nt, v_at, n_at, header=[
            "Transformed from MOVE4D into Theia3D coordinates (Z-up, metres)",
            "by transform_obj_to_theia.py. p_theia = T @ [p_m4d, 1]",
            "T rows: " + " | ".join(
                " ".join(f"{v:.9f}" for v in row) for row in T)])

        floor = check_floor(os.path.basename(path), Vt, T)
        res = dict(input=path, output=out_path, n_vertices=len(V),
                   n_normals=len(N), warnings=warnings, floor=floor,
                   bbox_in=(V.min(0), V.max(0)),
                   bbox_out=(Vt.min(0), Vt.max(0)))
        results.append(res)

        if verbose:
            print(f"\n  {os.path.basename(path)} -> {os.path.basename(out_path)}")
            print(f"    {len(V)} vertices, {len(N)} normals")
            print(f"    in  (MOVE4D, Y-up)  min {np.round(V.min(0), 4)}  "
                  f"max {np.round(V.max(0), 4)}")
            print(f"    out (Theia,  Z-up)  min {np.round(Vt.min(0), 4)}  "
                  f"max {np.round(Vt.max(0), 4)}")
            for w in warnings:
                print(f"    ! {w}")
            f = floor
            verdict = "OK" if f["ok"] else "*** CHECK THIS ***"
            print(f"    floor check: sole at z = {f['sole_mm']:+.1f} mm, "
                  f"Theia's floor is at {f['expected_mm']:+.1f} mm "
                  f"({f['offset_mm']:+.1f} mm)  {verdict}")
            if not f["ok"]:
                print("      A foot should sit ON the floor. If this is far "
                      "out, either the")
                print("      input was not in raw MOVE4D coordinates or the "
                      "wrong T was used.")

    if results and MESH_MAKE_PLOT:
        try:
            mesh_placement_figure(results, T, PLOT_FILE)
            if verbose:
                print(f"\n  wrote {PLOT_FILE}")
        except ImportError:
            if verbose:
                print("\n  (no matplotlib -- skipping the figure)")

    if verbose and results:
        bad = [r for r in results if not r["floor"]["ok"] or r["warnings"]]
        print("\n" + "-" * 74)
        print(f"  {len(results)} file(s) written"
              + (f", {len(bad)} with warnings" if bad else ", all checks passed"))
        print("-" * 74)
    return results




# %%==========================================================================
#  STEP 3 -- BIND THE MESHES TO THE THEIA FOOT SEGMENTS
# ===========================================================================

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


def good_frames(blocks):
    """Drop static frames whose landmarks sit far from the median pose."""
    allblk = np.concatenate(list(blocks.values()), axis=1)
    med = np.median(allblk, axis=0)
    dev = np.linalg.norm(allblk - med, axis=2).mean(axis=1)
    keep = np.where(dev <= BIND_FRAME_TOLERANCE_MM / 1000.0)[0]
    if len(keep) == 0:
        keep = np.arange(len(allblk))
    dropped = [(int(i), float(dev[i] * 1000))
               for i in range(len(allblk)) if i not in keep]
    return keep, dropped, dev * 1000


def bind_meshes(static_csv=None, mesh_files=None, verbose=None):
    static_csv = STATIC_METRICS_CSV if static_csv is None else static_csv
    verbose = VERBOSE if verbose is None else verbose

    vectors, matrices, items = read_metrics(static_csv)
    seg_names = list(BIND_SEGMENTS)

    # --- which frames to average over ------------------------------------
    blocks = {}
    for s in seg_names:
        lms = [n for n in BIND_SEGMENTS[s].get("landmarks", []) if n in vectors]
        if lms:
            blocks[s] = np.stack([vectors[n] for n in lms], axis=1)
    if blocks:
        keep, dropped, dev_mm = good_frames(blocks)
    else:
        keep, dropped, dev_mm = np.arange(len(items)), [], np.zeros(len(items))

    # --- the reference pose per segment ----------------------------------
    ref_R, ref_t, source, quality, lm_used = {}, {}, {}, {}, {}
    for s in seg_names:
        spec = BIND_SEGMENTS[s]
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
                  pose_signal={s: BIND_SEGMENTS[s].get("pose") for s in seg_names},
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
        print_binding_report(binding)
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


def print_binding_report(b):
    W = 74
    meta, rep = b["meta"], b["meta"]["report"]
    print("=" * W)
    print("  BIND FOOT MESHES TO THEIA BIND_SEGMENTS")
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
#  RUN  --  the three steps, in order
# ============================================================================

def get_transform(verbose=True):
    """Step 1. Solve the MOVE4D -> Theia transform, or reuse a cached one."""
    if T_MATRIX is not None:
        if verbose:
            print("  step 1: using T_MATRIX from CONFIG, nothing solved\n")
        return np.asarray(T_MATRIX, float), None
    if REUSE_SAVED_TRANSFORM and not FORCE_RESOLVE and os.path.exists(T_FILE):
        T = np.load(T_FILE) if T_FILE.endswith(".npy") else np.loadtxt(T_FILE)
        if verbose:
            print(f"  step 1: reusing {T_FILE} (set FORCE_RESOLVE to redo it)\n")
        return T, None

    report, T, th, m4 = solve_alignment(verbose=verbose)
    np.save(os.path.splitext(T_FILE)[0] + ".npy", T)
    np.savetxt(os.path.splitext(T_FILE)[0] + ".txt", T, fmt="%.9f",
               header="4x4: p_theia = T @ [p_m4d; 1]  (raw FBX coords both sides)")
    with open("alignment_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    if ALIGN_MAKE_PLOT:
        try:
            alignment_figure(report, th, m4, "alignment_check.png")
            if verbose:
                print("  wrote alignment_check.png")
        except ImportError:
            pass
    return T, report


def build(verbose=None):
    """Run all three steps. Returns the binding dict."""
    verbose = VERBOSE if verbose is None else verbose
    W = 78
    if verbose:
        print("#" * W)
        print("#  STEP 1 of 3   MOVE4D -> THEIA3D TRANSFORM")
        print("#" * W + "\n")
    T, _ = get_transform(verbose)

    if verbose:
        print("\n" + "#" * W)
        print("#  STEP 2 of 3   MESHES INTO THEIA COORDINATES")
        print("#" * W + "\n")
    moved = meshes_to_theia(T, MESH_FILES_RAW, verbose=verbose)
    theia_meshes = [r["output"] for r in moved]
    if not theia_meshes:
        raise ValueError("step 2 produced no meshes -- check MESH_FILES_RAW")

    if verbose:
        print("\n" + "#" * W)
        print("#  STEP 3 of 3   BIND TO THE THEIA FOOT SEGMENTS")
        print("#" * W + "\n")
    binding = bind_meshes(STATIC_METRICS_CSV, theia_meshes, verbose=verbose)
    out = save_binding(binding, OUT_FILE)

    if verbose:
        print(f"\n  wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
        print("\n  next:  python apply_binding.py TRIAL_metrics.csv --plot")
    return binding


def main(argv=None):
    global FORCE_RESOLVE
    argv = sys.argv[1:] if argv is None else argv
    if "--resolve" in argv:
        FORCE_RESOLVE = True
    return build()


if __name__ == "__main__":
    BINDING = main()
