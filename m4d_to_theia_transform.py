# -*- coding: utf-8 -*-
"""
=============================================================================
 MOVE4D  ->  THEIA3D   rigid transformation, from a static standing trial
=============================================================================

Finds the 4x4 matrix T such that

        p_theia = T @ [p_m4d, 1]          (positions, raw FBX file coordinates)

by aligning the GLOBAL ORIENTATIONS of the thigh, shank, upper-arm and
forearm segments (the "hip, knee, shoulder, elbow" segments), and then the
joint-centre positions.

The script follows six steps, one per section:

    1. load both FBX files
    2. read the kinematic data for every segment
    3. turn the segment rotations from local into global space
    4. put both files in the same units and the same coordinate system
    5. rotation:    global segment orientations as close as possible
    6. translation: joint positions as close as possible

Run it in Spyder: set the two paths in CONFIG and press F5. Afterwards the
namespace holds T, THEIA, M4D and RESULT, and you can do::

    pts_theia = apply_transform(T, pts_m4d)          # (..., 3)
    R_theia   = transform_orientation(T, R_m4d)      # (..., 3, 3)

Needs numpy. matplotlib only for the check figure.


WHY NOT THE AUTODESK FBX PYTHON SDK?
------------------------------------
It is not installable here, and probably not for you either. The SDK is not
on PyPI (`pip install fbx` fails); Autodesk ships it as a manual download
built against specific Python versions, so it breaks whenever you move
Python. Pip-installable readers do exist -- `bpy` (Blender as a module) is
the realistic one -- but bpy is a ~1 GB dependency and its FBX importer
applies its own axis and unit conversion on the way in, which is precisely
the quantity this script is trying to measure. Reading the node records
directly is ~250 lines (sections A and B) and removes both problems. Those
two sections are plumbing: you should not need to touch them.


WHY NOT THE FULL 3x3 SEGMENT ORIENTATION?
-----------------------------------------
Because two thirds of it is naming convention, not anatomy. Measured on
D05_C1, with `REPORT_FRAME_CONVENTIONS = True` printing it every run:

  * The LONG AXIS is consistent and meaningful in both rigs. Theia puts it
    on local -Z for every segment, MOVE4D on local +Y, both to 4 decimal
    places. After alignment the long axes agree to 0.5-3.7 deg. That is
    real anatomy, and it is what this script fits on.

  * The AXIAL ROLL about that axis is pure convention and the two rigs do
    not share it. The residual per-segment offset
    O_s = (R . R_m4d,s)^T . R_theia,s runs 87-117 deg, and the offsets
    differ BETWEEN segments by 5-27 deg within the legs and arms, and by
    161 deg between the left and right forearm (MOVE4D rolls the forearm
    bone with pronation, Theia does not).

So R_theia,s = G . R_m4d,s . O_s, and from a single posture G and the O_s
are confounded. Fitting the full orientations means letting a quantity that
disagrees by 90 deg drive the estimate: the constrained version lands 3 deg
from the long-axis answer and the unconstrained one diverges completely
(yaw +106 deg, tilt +91 deg). The script prints all of this so you can
re-check it on your own data rather than take my word for it.

If you want to use the full orientations honestly, you need two or more
DISTINCT postures. Then the offsets cancel in the relative rotations,
    R_theia,s(t2) . R_theia,s(t1)^T = G . R_m4d,s(t2) . R_m4d,s(t1)^T . G^T
and G is identifiable without knowing any O_s. A single A-pose cannot do it.


CAVEAT on orientations: T[:3,:3] @ R_m4d re-expresses a MOVE4D segment frame
in Theia's COORDINATE SYSTEM. It does not convert MOVE4D's bone convention
into Theia's anatomical convention -- that is the O_s problem above.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import zlib

import numpy as np


# %%==========================================================================
#  CONFIG
# ============================================================================

THEIA_FBX = "D05_C1_apose_Theia.fbx"
M4D_FBX   = "D05_C1_apose_M4D.fbx"

# Constrain the rotation to be about the vertical. Both systems get their
# vertical from the same physical gravity/lab calibration, so in principle
# only the heading differs. Set False to fit all 3 DOF; the report evaluates
# both either way, so you can see what the extra freedom buys.
VERTICAL_ONLY = True

# Segments whose global orientation drives the rotation (step 5), named by
# their proximal joint. Theia name / MOVE4D name.
SEGMENTS = {
    "l_thigh": ("l_thigh", "LeftHip"),      "r_thigh": ("r_thigh", "RightHip"),
    "l_shank": ("l_shank", "LeftKnee"),     "r_shank": ("r_shank", "RightKnee"),
    "l_uarm":  ("l_uarm",  "LeftShoulder"), "r_uarm":  ("r_uarm",  "RightShoulder"),
    "l_larm":  ("l_larm",  "LeftElbow"),    "r_larm":  ("r_larm",  "RightElbow"),
}

# Joint centres whose positions drive the translation (step 6). These are the
# origins of the segments above: hip, knee, shoulder, elbow.
JOINTS = {
    "l_hip":      ("l_thigh", "LeftHip"),      "r_hip":      ("r_thigh", "RightHip"),
    "l_knee":     ("l_shank", "LeftKnee"),     "r_knee":     ("r_shank", "RightKnee"),
    "l_shoulder": ("l_uarm",  "LeftShoulder"), "r_shoulder": ("r_uarm",  "RightShoulder"),
    "l_elbow":    ("l_larm",  "LeftElbow"),    "r_elbow":    ("r_larm",  "RightElbow"),
}

# Distal ends, used ONLY to work out which local axis each rig runs its bones
# along, and to report segment lengths. Not fitted on.
SEGMENT_ENDS = {
    "l_thigh": ("l_thigh", "l_shank"),  "r_thigh": ("r_thigh", "r_shank"),
    "l_shank": ("l_shank", "l_foot"),   "r_shank": ("r_shank", "r_foot"),
    "l_uarm":  ("l_uarm",  "l_larm"),   "r_uarm":  ("r_uarm",  "r_larm"),
}
M4D_ENDS = {
    "l_thigh": ("LeftHip", "LeftKnee"),   "r_thigh": ("RightHip", "RightKnee"),
    "l_shank": ("LeftKnee", "LeftAnkle"), "r_shank": ("RightKnee", "RightAnkle"),
    "l_uarm":  ("LeftShoulder", "LeftElbow"),
    "r_uarm":  ("RightShoulder", "RightElbow"),
}

# A frame is dropped when its pose sits further from the median pose than
# FACTOR * (the typical frame's distance) + FLOOR. On this trial that removes
# Theia's bind pose (475 mm out) and its last frame (16 mm, a filter edge
# transient) while keeping everything within ~6 mm. The floor stops the rule
# biting on a genuinely still trial, where the typical distance is under 1 mm.
# This assumes a STATIC trial; it is not meant for a walk.
FRAME_TOLERANCE_FACTOR = 3.0
FRAME_TOLERANCE_FLOOR_M = 0.002
REPORT_FRAME_CONVENTIONS = True
MAKE_PLOT    = True
SAVE_OUTPUTS = True
OUTDIR       = "."


# %%==========================================================================
#  A. BINARY FBX READER (plumbing -- no need to read this)
# ============================================================================

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
#  B. FBX TRANSFORM EVALUATION (plumbing -- no need to read this)
# ============================================================================

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

    def global_transforms(self):
        """Every limb node's global 4x4 over every key time -> {name: (T,4,4)}."""
        order, seen = [], set()

        def visit(j):
            if j.uid in seen:
                return
            seen.add(j.uid)
            if j.parent is not None:
                visit(j.parent)
            order.append(j)

        for j in self.joints.values():
            if j.type in ("LimbNode", "Root"):
                visit(j)

        out = {}
        for j in order:
            L = j.local_matrices(self.times)
            out[j.uid] = L if j.parent is None else out[j.parent.uid] @ L
        return {j.name: out[j.uid] for j in order if j.type == "LimbNode"}


class _Curve:
    def __init__(self, t, v):
        self.t, self.v = t, v

    def __call__(self, times):
        if len(self.t) == 1:
            return np.full(len(times), self.v[0])
        return np.interp(times, self.t, self.v)     # clamped at both ends


# %%==========================================================================
#  SMALL GEOMETRY HELPERS
# ============================================================================

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


# %%==========================================================================
#  STEPS 1-4:  load, read every segment, local -> global, common frame
# ============================================================================

def up_axis_to_z(up_axis, up_sign):
    """Right-handed rotation taking a file's declared up-axis onto +Z."""
    s = 1.0 if up_sign >= 0 else -1.0
    if up_axis == 2:
        return np.eye(3) if s > 0 else np.diag([1.0, -1.0, -1.0])
    if up_axis == 1:
        return np.array([[1, 0, 0], [0, 0, -s], [0, s, 0]], float)
    return np.array([[0, 0, -s], [0, 1, 0], [s, 0, 0]], float)


class Skeleton:
    """One FBX file, reduced to global segment poses in a common frame.

    After construction:
        .pos[node]   (F, 3)     global position, Z-up, metres
        .rot[node]   (F, 3, 3)  global orientation, Z-up
        .p[node]     (3,)       mean over the retained frames
        .R[node]     (3, 3)     mean over the retained frames
    Node names are this file's own, so use .lookup(key) with the keys from
    SEGMENTS / JOINTS to get at them by anatomy.
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
        limit = (FRAME_TOLERANCE_FACTOR * np.median(dev)
                 + FRAME_TOLERANCE_FLOOR_M)
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
        return self.p[self.lookup(key, table or JOINTS)]

    def orientation(self, key):
        return self.R[self.lookup(key, SEGMENTS)]

    def raw_position(self, key, table=None):
        """Mean position back in this file's OWN raw coordinates."""
        return self.C.T @ self.position(key, table) / self.scale


# %%==========================================================================
#  STEP 5:  ROTATION  -- global segment orientations as close as possible
# ============================================================================

def detect_bone_axis(skel):
    """Which signed LOCAL axis does this rig run its bones along?

    Determined from the segments whose two endpoints are both known, by
    expressing the proximal->distal direction in the segment's own frame.
    Returned as (column, sign, worst |dot|). A worst |dot| near 1 means the
    rig is perfectly consistent and the long axis can then be read straight
    off the orientation matrix of ANY segment -- including the forearm, whose
    distal end (the wrist) neither rig exposes as a single clean node.
    """
    table = SEGMENT_ENDS if skel.which == "theia" else M4D_ENDS
    votes = []
    for seg, (a, b) in table.items():
        na, nb = skel.scene.resolve(a), skel.scene.resolve(b)
        node = skel.lookup(seg, SEGMENTS)
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
    return np.array([sign * skel.orientation(s)[:, col] for s in SEGMENTS])


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


# %%==========================================================================
#  STEP 6:  TRANSLATION  -- joint positions as close as possible
# ============================================================================

def solve_translation(A, B, R):
    """t minimising sum ||a_i - (R b_i + t)||^2, which is just the mean."""
    return (A - B @ R.T).mean(axis=0)


# %%==========================================================================
#  MAIN
# ============================================================================

def solve(theia_path=None, m4d_path=None, vertical_only=None, verbose=True):
    """Run all six steps. Returns (report, T, theia, m4d)."""
    vertical_only = VERTICAL_ONLY if vertical_only is None else vertical_only

    # --- steps 1-4 ----------------------------------------------------
    th = Skeleton(theia_path or THEIA_FBX, "theia")
    m4 = Skeleton(m4d_path or M4D_FBX, "m4d")

    missing = [k for k in JOINTS
               if th.lookup(k, JOINTS) is None or m4.lookup(k, JOINTS) is None]
    if missing:
        raise ValueError(f"joints not found in one or both files: {missing}. "
                         f"Check the JOINTS / SEGMENTS tables.")

    ax_t, ax_m = detect_bone_axis(th), detect_bone_axis(m4)

    # --- step 5 -------------------------------------------------------
    U, V = long_axes(th, ax_t), long_axes(m4, ax_m)
    R = solve_rotation(U, V, vertical_only)

    # --- step 6 -------------------------------------------------------
    keys = list(JOINTS)
    A = np.array([th.position(k) for k in keys])
    B = np.array([m4.position(k) for k in keys])
    t = solve_translation(A, B, R)

    # --- assemble T in RAW file coordinates ---------------------------
    # p_common = C @ (metres_per_unit * p_raw), so undo both on the Theia side
    T = np.eye(4)
    T[:3, :3] = th.C.T @ R @ m4.C * (m4.scale / th.scale)
    T[:3, 3] = th.C.T @ t / th.scale

    # --- residuals ----------------------------------------------------
    seg_err = angle_between(U, V @ R.T)
    pos_err = np.linalg.norm(A - (B @ R.T + t), axis=1) * 1000

    # the same fit under the other rotation model, for comparison
    R_alt = solve_rotation(U, V, not vertical_only)
    t_alt = solve_translation(A, B, R_alt)
    alt = dict(vertical_only=not vertical_only,
               yaw_deg=yaw_and_tilt(R_alt)[0], tilt_deg=yaw_and_tilt(R_alt)[1],
               segment_rms_deg=float(np.sqrt((angle_between(U, V @ R_alt.T) ** 2).mean())),
               position_rms_mm=float(np.sqrt((np.linalg.norm(
                   A - (B @ R_alt.T + t_alt), axis=1) ** 2).mean()) * 1000))

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
        segment_residual_deg=dict(
            rms=float(np.sqrt((seg_err ** 2).mean())), max=float(seg_err.max()),
            per_segment=dict(zip(SEGMENTS, seg_err.round(3).tolist()))),
        position_residual_mm=dict(
            rms=float(np.sqrt((pos_err ** 2).mean())), max=float(pos_err.max()),
            per_joint=dict(zip(keys, pos_err.round(2).tolist()))),
        alternative_model=alt,
        frame_conventions=frame_conventions(th, m4, R),
        checks=dict(mirror_determinant=mirror_test(U, V),
                    **checks(th, m4, R, t, T, keys)),
    )
    if verbose:
        print_report(report, th, m4)
    return report, T, th, m4


# %%==========================================================================
#  CHECKS AND REPORT
# ============================================================================

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


def checks(th, m4, R, t, T, keys):
    """Units, segment lengths, and an end-to-end test of T."""
    lengths = []
    for seg, (a, b) in SEGMENT_ENDS.items():
        ma, mb = M4D_ENDS[seg]
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
    O = {s: (R @ m4.orientation(s)).T @ th.orientation(s) for s in SEGMENTS}
    labels = list(SEGMENTS)
    pair = np.array([[rotation_angle(O[a].T @ O[b]) for b in labels]
                     for a in labels])
    return dict(offset_deg={s: rotation_angle(O[s]) for s in labels},
                max_disagreement_deg=float(pair.max()),
                mean_disagreement_deg=float(pair[np.triu_indices(len(labels), 1)].mean()))


def print_report(r, th, m4):
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

    c = r["checks"]
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
    sr = r["segment_residual_deg"]
    print(f"\n    segment orientation residual   RMS {sr['rms']:.2f} deg, "
          f"max {sr['max']:.2f} deg")
    for s, v in sorted(sr["per_segment"].items(), key=lambda x: -x[1]):
        print(f"      {s:9s} {v:6.2f} deg")

    print("\n  STEP 6: translation from joint positions")
    pr = r["position_residual_mm"]
    print(f"    position residual              RMS {pr['rms']:.1f} mm, "
          f"max {pr['max']:.1f} mm")
    for k, v in sorted(pr["per_joint"].items(), key=lambda x: -x[1]):
        print(f"      {k:11s} {v:6.1f} mm")

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
          "   in the common Z-up frame")
    print(f"\n    end-to-end check on RAW coordinates: "
          f"{c['raw_check_rms_mm']:.1f} mm RMS"
          f"   (must equal {pr['rms']:.1f})")
    if abs(c["raw_check_rms_mm"] - pr["rms"]) > 0.1:
        print("    *** MISMATCH: the up-axis / unit bookkeeping behind T is "
              "wrong. Do not use this matrix. ***")

    a = r["alternative_model"]
    other = "vertical-axis only" if a["vertical_only"] else "full 3-DOF"
    print("\n    the other rotation model, for comparison:")
    print(f"      {mode:19s} yaw {r['yaw_deg']:+6.2f}  tilt {r['tilt_deg']:5.2f}"
          f"  seg {sr['rms']:5.2f} deg  pos {pr['rms']:5.1f} mm   <- used")
    print(f"      {other:19s} yaw {a['yaw_deg']:+6.2f}  tilt {a['tilt_deg']:5.2f}"
          f"  seg {a['segment_rms_deg']:5.2f} deg  pos {a['position_rms_mm']:5.1f} mm")
    if r["vertical_only"] and a["position_rms_mm"] >= pr["rms"] - 0.5:
        print(f"      -> the {a['tilt_deg']:.2f} deg tilt the 3-DOF fit wants "
              f"does not improve the position")
        print("         fit, so it is absorbing skeleton mismatch rather than "
              "a real")
        print("         calibration difference. Keep VERTICAL_ONLY = True.")

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
        print(f"    which agrees to {sr['rms']:.1f} deg RMS above -- is the "
              f"part that means")
        print("    something, and it is what the fit uses.")
    print("=" * W)


# %%==========================================================================
#  USING THE RESULT
# ============================================================================

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


def save_outputs(report, T, outdir=None):
    outdir = OUTDIR if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)
    np.savetxt(os.path.join(outdir, "T_m4d_to_theia.txt"), T, fmt="%.9f",
               header="4x4 homogeneous: p_theia = T @ [p_m4d; 1]  "
                      "(raw FBX file coordinates, both sides)")
    np.save(os.path.join(outdir, "T_m4d_to_theia.npy"), T)
    with open(os.path.join(outdir, "alignment_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)


def make_figure(report, th, m4, path=None):
    """Skeleton overlay after the transform, plus the two residual bars."""
    import matplotlib.pyplot as plt

    R = np.array(report["rotation"])
    t = np.array(report["translation_m"])
    pt = {k: th.position(k) for k in JOINTS}
    pm = {k: R @ m4.position(k) + t for k in JOINTS}
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
    d = report["position_residual_mm"]["per_joint"]
    ks = sorted(d, key=lambda k: -d[k])
    ax.bar(ks, [d[k] for k in ks], color="#1f4e79")
    ax.set_ylabel("joint position residual (mm)", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    fig.suptitle("MOVE4D -> Theia3D alignment check", fontsize=12, y=0.97)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


# %%==========================================================================
#  RUN
# ============================================================================

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    theia, m4d = (argv[0], argv[1]) if len(argv) >= 2 else (None, None)
    vertical = False if "--full" in argv else None

    report, T, th, m4 = solve(theia, m4d, vertical)
    if SAVE_OUTPUTS:
        save_outputs(report, T)
        print(f"\n  wrote T_m4d_to_theia.{{txt,npy}} and alignment_report.json "
              f"to {os.path.abspath(OUTDIR)}")
    if MAKE_PLOT:
        try:
            make_figure(report, th, m4,
                        os.path.join(OUTDIR, "alignment_check.png")
                        if SAVE_OUTPUTS else None)
            if SAVE_OUTPUTS:
                print("  wrote alignment_check.png")
        except ImportError:
            print("  (no matplotlib -- skipping the figure)")
    return report, T, th, m4


if __name__ == "__main__":
    RESULT, T, THEIA, M4D = main()
