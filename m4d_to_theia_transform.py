# -*- coding: utf-8 -*-
"""
=============================================================================
 MOVE4D  ->  THEIA3D   rigid transformation, from a static standing trial
=============================================================================

WHAT THIS DOES
--------------
Finds the 4x4 homogeneous matrix ``T`` such that

        p_theia = T @ [p_m4d, 1]          (positions, raw file coordinates)
        R_theia = T[:3, :3] @ R_m4d       (orientations, see caveat below)

i.e. it puts MOVE4D data into the Theia3D global coordinate system, so that
the *global orientation of the body segments* agrees between the two systems.

The transform is fitted from a static standing (A-pose) trial recorded
simultaneously by both systems.

WHAT IT FITS ON, AND WHY
------------------------
Only the **arm and leg segments** are used: thigh, shank, upper arm, forearm.
Feet, hands, pelvis, trunk and head are deliberately excluded and reported as
*validation* landmarks instead -- the two skeletons define them differently
enough that including them would fit the skeleton mismatch, not the
coordinate-system difference.  (Measured on D05_C1: foot length ratio 0.86,
trunk 1.16, versus 0.94-1.02 for the limb segments.)

Two kinds of observation are used, both built from joint-centre *positions*:

  * segment LONG AXES  (hip->knee, knee->ankle, shoulder->elbow, elbow->wrist)
    These constrain the out-of-plane part of the rotation.
  * bilateral MEDIO-LATERAL AXES (right->left at hip, knee, ankle, shoulder,
    elbow, wrist).  These are horizontal, so they are what actually pins the
    heading (yaw), and they are insensitive to a joint-centre definition
    offset that is symmetric between sides -- which is the usual case.

Everything is a *unit direction between two landmarks*, so the rotation is
translation-invariant and decouples from the translation.

WHY NOT THE FULL SEGMENT ORIENTATION MATRICES?
----------------------------------------------
It is tempting to fit the rotation directly from each segment's 3x3 global
orientation.  Do not.  The two rigs use unrelated local bone-axis
conventions, so each segment carries its own constant offset:

    R_theia,s = G @ R_m4d,s @ O_s          (O_s = per-segment convention)

and from a single posture G and the O_s are perfectly confounded.  The script
measures the O_s and reports them (`--- segment frame offsets ---`).  On
D05_C1 they are 87-117 deg and differ between segments by up to 30 deg, so
even the "shared offset" model (all O_s equal, which *would* be identifiable)
fails: it leaves 8-9 deg of residual and its 3-DOF solution is unstable.

The long axis is the part of a segment frame that both systems define
anatomically rather than by convention, which is why the fit uses it and
ignores the axial roll.

CAVEAT on orientations: ``T[:3,:3] @ R_m4d`` re-expresses a MOVE4D segment
frame in Theia's coordinate system.  It does NOT convert MOVE4D's local bone
convention into Theia's anatomical convention -- that is a separate,
per-segment problem (see the offsets above).

ROTATION MODEL
--------------
``ROTATION_MODEL = "yaw"`` (default) constrains the rotation to be about the
vertical.  Both systems establish their vertical from the same physical
gravity/lab calibration, so in principle only the heading and the origin
differ.  The yaw is solved in closed form under that constraint, which is a
genuinely different (and better) estimator than fitting 3 DOF and then
throwing the tilt away.  Set ``"full"`` to fit all 3 DOF; the report always
evaluates both so you can see whether the extra freedom buys anything.

USAGE (Spyder)
--------------
Set the paths in the CONFIG cell below and press F5.  Afterwards the
namespace holds ``T`` (the 4x4), ``RESULT`` (every number in the report, as
a dict) and ``CONTEXT`` (the loaded trials, observations and the figure), and
``OUTDIR`` holds the same on disk.  Then::

    pts_theia = apply_transform(T, pts_m4d)          # (..., 3)
    R_theia   = transform_orientation(T, R_m4d)      # (..., 3, 3)

Command line also works:  ``python m4d_to_theia_transform.py THEIA.fbx M4D.fbx``

Requires numpy.  matplotlib only for the optional figure.  No FBX SDK.

TRUST BUT VERIFY
----------------
T is fitted in a canonicalised frame (Z-up, metres) and then rebuilt to act
on raw FBX coordinates, which drags in both files' up-axis matrices and unit
scale factors.  The report ends that section with an end-to-end check: raw
MOVE4D landmarks pushed through T and compared against raw Theia landmarks.
On a static trial it MUST reproduce the fitted residual; if it does not, say
so rather than using the matrix.

The single biggest improvement available is to pool a second, DYNAMIC trial
from the same session into ``TRIAL_PAIRS`` -- see the "azimuthal diversity"
diagnostic near the end of the report for why.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import zlib

import numpy as np


# %%==========================================================================
#  1. CONFIG  -- everything you are likely to change lives here
# ============================================================================

# --- input ------------------------------------------------------------------
THEIA_FBX = "D05_C1_apose_Theia.fbx"
M4D_FBX   = "D05_C1_apose_M4D.fbx"

# Pool several simultaneously-recorded pairs into ONE fit by adding them here.
# A dynamic trial (walk / turn) alongside the static one is strongly
# recommended: it is what breaks the A-pose's lack of azimuthal diversity.
TRIAL_PAIRS = [(THEIA_FBX, M4D_FBX)]

# --- model ------------------------------------------------------------------
ROTATION_MODEL   = "yaw"    # "yaw" (recommended) | "full"
INCLUDE_FOREARM  = True     # elbow->wrist long axis and the wrist ML axis
ALLOW_SCALE      = False    # fit a uniform scale factor too (only if the
                            # files genuinely use different units)

# --- observation weighting --------------------------------------------------
# Expected disagreement between the two systems' estimates of each joint
# CENTRE, in mm.  A direction between landmarks a and b then has an angular
# uncertainty of about sqrt(sa^2 + sb^2) / L, and observations are weighted
# by 1 / sigma^2.  This replaces hand-tuned weights with something you can
# argue about on anatomical grounds.
LANDMARK_SIGMA_MM = {
    "l_hip": 20.0, "r_hip": 20.0,          # hip centre: worst-defined
    "l_knee": 12.0, "r_knee": 12.0,
    "l_ankle": 10.0, "r_ankle": 10.0,
    "l_shoulder": 20.0, "r_shoulder": 20.0,
    "l_elbow": 12.0, "r_elbow": 12.0,
    "l_wrist": 12.0, "r_wrist": 12.0,
}
# Irreducible systematic disagreement per direction, added in quadrature.
# Without it, the very long inter-limb ML axes (elbow-to-elbow ~0.7 m,
# wrist-to-wrist ~0.95 m in an A-pose) would swamp everything else.
SYSTEMATIC_FLOOR_DEG = 2.0
# A bilateral right->left vector is far more robust than the landmark sigmas
# suggest, because the two endpoint errors are strongly correlated: if both
# hip centres are displaced anteriorly by the same amount, the direction does
# not rotate at all. Only the left-right ASYMMETRIC part of the error tilts
# it, and that is a fraction of the total. Long axes get no such discount --
# their two endpoints are genuinely independent landmarks.
ML_ASYMMETRY_FRACTION = 0.4

# --- robustness -------------------------------------------------------------
HUBER_DEG = 3.0             # angular residual beyond which a direction is
HUBER_MM  = 20.0            # position residual beyond which a joint is
                            #   progressively down-weighted (IRLS)
N_BOOTSTRAP = 500
RANDOM_SEED = 0

# --- frame quality control --------------------------------------------------
# Stage 1 catches bind/rest poses (tens of cm out). Stage 2 catches filter
# edge transients. Both have an absolute floor as well as a MAD threshold:
# on an 8-frame static trial the MAD is estimated from a handful of numbers
# and will happily reject a frame that is 3 mm from its neighbours, which is
# noise, not an artifact -- averaging over it beats throwing it away.
GROSS_FLOOR_M = 0.05        # a rest/bind pose sits at least this far out
EDGE_FLOOR_M  = 0.010       # never reject a frame closer than this to the median
MIN_FRAMES    = 3

# "static" trials are collapsed to their mean pose; "dynamic" ones are
# resampled onto a common time base. "auto" decides from the residual motion.
TRIAL_MODE = "auto"              # "auto" | "static" | "dynamic"
STATIC_MOTION_THRESHOLD = 0.05   # m of residual motion; a standing trial with
                                 # filtered edges easily shows 20-30 mm

# --- output -----------------------------------------------------------------
OUTDIR       = "."           # where T_m4d_to_theia.{txt,npy}, the JSON
                             # report and the PNG are written
SAVE_OUTPUTS = True
MAKE_PLOT    = True
VERBOSE      = True


# %%==========================================================================
#  2. SKELETON MODEL  -- landmark correspondence and the segments to fit
# ============================================================================
# A landmark spec is either
#     "NodeName"                          -- namespace-tolerant node lookup
#     ("children_of", "NodeName")         -- centroid of that node's children
#     ("centroid", ["A", "B", ...])       -- centroid of several nodes
#
# MOVE4D has no single "wrist" node: the five carpometacarpal roots hang
# directly off *Forearm and sit within ~15 mm of each other, so their centroid
# is a clean and rig-version-tolerant wrist estimate.

LANDMARKS = {
    #  key            Theia3D          MOVE4D
    "l_hip":        ("l_thigh",       "LeftHip"),
    "r_hip":        ("r_thigh",       "RightHip"),
    "l_knee":       ("l_shank",       "LeftKnee"),
    "r_knee":       ("r_shank",       "RightKnee"),
    "l_ankle":      ("l_foot",        "LeftAnkle"),
    "r_ankle":      ("r_foot",        "RightAnkle"),
    "l_shoulder":   ("l_uarm",        "LeftShoulder"),
    "r_shoulder":   ("r_uarm",        "RightShoulder"),
    "l_elbow":      ("l_larm",        "LeftElbow"),
    "r_elbow":      ("r_larm",        "RightElbow"),
    "l_wrist":      ("l_hand",        ("children_of", "LeftForearm")),
    "r_wrist":      ("r_hand",        ("children_of", "RightForearm")),
    # --- not fitted: reported as validation / skeleton-gap diagnostics ---
    "pelvis":       ("pelvis",        "Hips"),
    "thorax":       ("thorax",        "Chest4"),
    "head":         ("head",          "Head"),
    "l_toe":        ("l_toes",        "LeftToe"),
    "r_toe":        ("r_toes",        "RightToe"),
}

# Segment long axes: (label, proximal, distal, is_forearm)
SEGMENT_AXES = [
    ("l_thigh",   "l_hip",      "l_knee",  False),
    ("r_thigh",   "r_hip",      "r_knee",  False),
    ("l_shank",   "l_knee",     "l_ankle", False),
    ("r_shank",   "r_knee",     "r_ankle", False),
    ("l_uarm",    "l_shoulder", "l_elbow", False),
    ("r_uarm",    "r_shoulder", "r_elbow", False),
    ("l_larm",    "l_elbow",    "l_wrist", True),
    ("r_larm",    "r_elbow",    "r_wrist", True),
]

# Bilateral medio-lateral axes: (label, right, left, is_forearm_dependent)
ML_AXES = [
    ("hip_ML",      "r_hip",      "l_hip",      False),
    ("knee_ML",     "r_knee",     "l_knee",     False),
    ("ankle_ML",    "r_ankle",    "l_ankle",    False),
    ("shoulder_ML", "r_shoulder", "l_shoulder", False),
    ("elbow_ML",    "r_elbow",    "l_elbow",    False),
    ("wrist_ML",    "r_wrist",    "l_wrist",    True),
]

# Landmarks used for the translation fit (arm/leg joint centres only).
TRANSLATION_LANDMARKS = ["l_hip", "r_hip", "l_knee", "r_knee",
                         "l_ankle", "r_ankle", "l_shoulder", "r_shoulder",
                         "l_elbow", "r_elbow", "l_wrist", "r_wrist"]

# Reported but never fitted -- these quantify the skeleton definition gap.
VALIDATION_LANDMARKS = ["pelvis", "thorax", "head", "l_toe", "r_toe"]

# Nodes carrying each segment's local frame, for the "why not full
# orientations" diagnostic.
SEGMENT_FRAME_NODES = {
    "l_thigh": ("l_thigh", "LeftHip"),   "r_thigh": ("r_thigh", "RightHip"),
    "l_shank": ("l_shank", "LeftKnee"),  "r_shank": ("r_shank", "RightKnee"),
    "l_uarm":  ("l_uarm", "LeftShoulder"), "r_uarm": ("r_uarm", "RightShoulder"),
    "l_larm":  ("l_larm", "LeftElbow"),  "r_larm":  ("r_larm", "RightElbow"),
}

# Yaw cross-checks: anatomically independent subsets.
YAW_SUBSETS = {
    "leg ML (hip/knee/ankle width)":  ["hip_ML", "knee_ML", "ankle_ML"],
    "arm ML (shoulder/elbow/wrist)":  ["shoulder_ML", "elbow_ML", "wrist_ML"],
    "leg long axes":                  ["l_thigh", "r_thigh", "l_shank", "r_shank"],
    "arm long axes":                  ["l_uarm", "r_uarm", "l_larm", "r_larm"],
}


# %%==========================================================================
#  3. BINARY FBX READER  (versions 7100-7700)
# ============================================================================
# Parses the raw node-record tree only.  No FBX semantics here.

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

    def __repr__(self):
        return (f"<FbxNode {self.name} props={len(self.props)} "
                f"children={len(self.children)}>")


_ARRAY_FMT = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b"}


def _read_prop(buf, off):
    """Read one FBX property. Returns (value, new_offset)."""
    t = buf[off:off + 1].decode("ascii")
    off += 1
    if t == "Y":
        return struct.unpack_from("<h", buf, off)[0], off + 2
    if t == "C":
        return bool(buf[off]), off + 1
    if t == "I":
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if t == "F":
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if t == "D":
        return struct.unpack_from("<d", buf, off)[0], off + 8
    if t == "L":
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if t in _ARRAY_FMT:                                   # typed array
        n, enc, clen = struct.unpack_from("<III", buf, off)
        off += 12
        raw = buf[off:off + clen]
        off += clen
        if enc == 1:
            raw = zlib.decompress(raw)
        return np.frombuffer(raw, dtype="<" + _ARRAY_FMT[t], count=n), off
    if t in "SR":                                         # string / blob
        n = struct.unpack_from("<I", buf, off)[0]
        off += 4
        raw = buf[off:off + n]
        off += n
        return (raw.decode("utf-8", "replace") if t == "S" else raw), off
    raise ValueError(f"unknown FBX property type {t!r} at offset {off - 1}")


def _read_node(buf, off, version):
    """Read one node record. Returns (FbxNode | None, new_offset)."""
    if version >= 7500:
        end_off, nprops, _plen = struct.unpack_from("<QQQ", buf, off)
        off += 24
        sentinel = 25
    else:
        end_off, nprops, _plen = struct.unpack_from("<III", buf, off)
        off += 12
        sentinel = 13

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
    """Parse a binary FBX file. Returns (root_node, version)."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf[:21] != b"Kaydara FBX Binary  \x00":
        raise ValueError(f"{path}: not a binary FBX (ASCII FBX is unsupported)")
    version = struct.unpack_from("<I", buf, 23)[0]
    off, n = 27, len(buf)
    children = []
    while off < n - 100:                   # stop before the footer region
        node, off = _read_node(buf, off, version)
        if node is None:
            break
        children.append(node)
    return FbxNode("__root__", [], children), version


# %%==========================================================================
#  4. FBX SCENE EVALUATION  -- hierarchy, curves, global joint transforms
# ============================================================================

FBX_TIME_UNIT = 46186158000.0          # FBX ktime units per second

# FBX EulerOrder enum -> the order in which the axis rotations are APPLIED.
# For column vectors the matrix is the reverse product, e.g. XYZ -> Rz@Ry@Rx.
_EULER_ORDER = {0: "XYZ", 1: "XZY", 2: "YZX", 3: "YXZ", 4: "ZXY", 5: "ZYX",
                6: "XYZ"}

_TIME_MODE_FPS = {0: None, 1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0,
                  6: 30.0, 7: 30.0, 8: 29.97, 9: 29.97, 10: 25.0, 11: 24.0,
                  12: 1000.0, 13: 23.976, 14: None, 15: 96.0, 16: 72.0,
                  17: 59.94}


def euler_to_matrix(ang_deg, order_enum=0):
    """Euler angles in degrees, shape (..., 3) -> rotation matrices (..., 3, 3).

    Vectorised over the leading axes so a whole trial is one call.
    """
    a = np.radians(np.asarray(ang_deg, dtype=float))
    lead = a.shape[:-1]
    c, s = np.cos(a), np.sin(a)
    R = np.broadcast_to(np.eye(3), lead + (3, 3)).copy()
    for axis in _EULER_ORDER.get(int(order_enum), "XYZ"):
        i = "XYZ".index(axis)
        ci, si = c[..., i], s[..., i]
        one = np.ones_like(ci)
        zero = np.zeros_like(ci)
        if axis == "X":
            rows = [[one, zero, zero], [zero, ci, -si], [zero, si, ci]]
        elif axis == "Y":
            rows = [[ci, zero, si], [zero, one, zero], [-si, zero, ci]]
        else:
            rows = [[ci, -si, zero], [si, ci, zero], [zero, zero, one]]
        M = np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)
        R = M @ R
    return R


def _prop_value(p70, name, default):
    """Read one Properties70 entry as a float scalar or vector."""
    want_vec = isinstance(default, (list, tuple))
    if p70 is not None:
        for p in p70.children:
            if p.props[0] == name:
                vals = p.props[4:]
                if not want_vec:
                    return float(vals[0])
                return np.array([float(v) for v in vals[:len(default)]],
                                dtype=float)
    return np.array(default, dtype=float) if want_vec else float(default)


class Curve:
    """One animation curve (a single scalar channel)."""

    def __init__(self, times, values):
        self.t = np.asarray(times, dtype=float) / FBX_TIME_UNIT
        self.v = np.asarray(values, dtype=float)

    def sample(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if len(self.t) == 0:
            return np.zeros_like(t)
        if len(self.t) == 1:
            return np.full_like(t, self.v[0])
        return np.interp(t, self.t, self.v)      # clamped at both ends


class Joint:
    """One FBX Model node and its transform components."""

    def __init__(self, uid, name, node_type):
        self.uid, self.name, self.type = uid, name, node_type
        self.parent = None
        self.children = []
        self.lcl_t = np.zeros(3)
        self.lcl_r = np.zeros(3)
        self.lcl_s = np.ones(3)
        self.pre_rot = np.zeros(3)
        self.post_rot = np.zeros(3)
        self.rot_off = np.zeros(3)
        self.rot_piv = np.zeros(3)
        self.sca_off = np.zeros(3)
        self.sca_piv = np.zeros(3)
        self.rot_order = 0
        self.inherit_type = 0
        self.curves = {}      # {'Lcl Translation': ({'X': Curve, ...}, defaults)}

    # -- channel sampling ---------------------------------------------------
    def _channel(self, prop, static, times):
        """Evaluate one animatable 3-vector property over `times` -> (T, 3)."""
        out = np.tile(np.asarray(static, dtype=float), (len(times), 1))
        entry = self.curves.get(prop)
        if not entry:
            return out
        channels, defaults = entry
        for i, ax in enumerate("XYZ"):
            if ax in channels:
                out[:, i] = channels[ax].sample(times)
            elif ax in defaults:
                out[:, i] = defaults[ax]
        return out

    # -- local transform ----------------------------------------------------
    def local_matrices(self, times):
        """Local 4x4 transforms over `times` -> (T, 4, 4).

        Autodesk chain, in full:
            T . Roff . Rp . Rpre . R . Rpost^-1 . Rp^-1 . Soff . Sp . S . Sp^-1
        """
        tr = self._channel("Lcl Translation", self.lcl_t, times)
        ro = self._channel("Lcl Rotation", self.lcl_r, times)
        sc = self._channel("Lcl Scaling", self.lcl_s, times)

        R = euler_to_matrix(ro, self.rot_order)                     # (T,3,3)
        Rpre = euler_to_matrix(self.pre_rot, 0)
        Rpost = euler_to_matrix(self.post_rot, 0)
        Rf = Rpre @ R @ Rpost.T                                     # (T,3,3)

        S = np.zeros((len(times), 3, 3))
        S[:, [0, 1, 2], [0, 1, 2]] = sc

        # linear part and translation part of the chain above
        lin = Rf @ S
        inner = (-self.rot_piv + self.sca_off + self.sca_piv)        # (3,)
        inner = inner - (S @ self.sca_piv)                           # (T,3)
        t = tr + self.rot_off + self.rot_piv + np.einsum("tij,tj->ti", Rf, inner)

        M = np.zeros((len(times), 4, 4))
        M[:, :3, :3] = lin
        M[:, :3, 3] = t
        M[:, 3, 3] = 1.0
        return M


class Scene:
    """An FBX file evaluated into a joint hierarchy with animation."""

    def __init__(self, path):
        self.path = path
        self.root_node, self.version = fbx_load(path)
        self.joints = {}
        self._parse_objects()
        self._parse_connections()
        self._read_globals()
        self._time_range()
        self._by_name = {}
        for j in self.joints.values():
            self._by_name.setdefault(j.name, j)

    # ------------------------------------------------------------------
    def _parse_objects(self):
        objs = self.root_node.find("Objects")
        if objs is None:
            raise ValueError(f"{self.path}: no Objects section")
        self.anim_curves, self.curve_nodes = {}, {}

        for m in objs.find_all("Model"):
            uid = m.props[0]
            j = Joint(uid, m.props[1].split("\x00")[0], m.props[2])
            p70 = m.find("Properties70")
            j.lcl_t = _prop_value(p70, "Lcl Translation", [0, 0, 0])
            j.lcl_r = _prop_value(p70, "Lcl Rotation", [0, 0, 0])
            j.lcl_s = _prop_value(p70, "Lcl Scaling", [1, 1, 1])
            j.pre_rot = _prop_value(p70, "PreRotation", [0, 0, 0])
            j.post_rot = _prop_value(p70, "PostRotation", [0, 0, 0])
            j.rot_off = _prop_value(p70, "RotationOffset", [0, 0, 0])
            j.rot_piv = _prop_value(p70, "RotationPivot", [0, 0, 0])
            j.sca_off = _prop_value(p70, "ScalingOffset", [0, 0, 0])
            j.sca_piv = _prop_value(p70, "ScalingPivot", [0, 0, 0])
            j.rot_order = int(_prop_value(p70, "RotationOrder", 0))
            j.inherit_type = int(_prop_value(p70, "InheritType", 0))
            self.joints[uid] = j

        for c in objs.find_all("AnimationCurve"):
            kt, kv = c.find("KeyTime"), c.find("KeyValueFloat")
            if kt is not None and kv is not None:
                self.anim_curves[c.props[0]] = Curve(kt.props[0], kv.props[0])

        for cn in objs.find_all("AnimationCurveNode"):
            p70 = cn.find("Properties70")
            defaults = {}
            if p70 is not None:
                for p in p70.children:
                    if p.props[0].startswith("d|"):
                        defaults[p.props[0][2:]] = float(p.props[4])
            self.curve_nodes[cn.props[0]] = {"defaults": defaults, "channels": {}}

    # ------------------------------------------------------------------
    def _parse_connections(self):
        conns = self.root_node.find("Connections")
        if conns is None:
            return
        for c in conns.children:
            kind, src, dst = c.props[0], c.props[1], c.props[2]
            if kind == "OO":
                if src in self.joints and dst in self.joints:
                    self.joints[src].parent = self.joints[dst]
                    self.joints[dst].children.append(self.joints[src])
            elif kind == "OP":
                prop = c.props[3]
                if src in self.anim_curves and dst in self.curve_nodes:
                    if prop.startswith("d|"):
                        self.curve_nodes[dst]["channels"][prop[2:]] = \
                            self.anim_curves[src]
                elif src in self.curve_nodes and dst in self.joints:
                    cn = self.curve_nodes[src]
                    self.joints[dst].curves[prop] = (cn["channels"],
                                                     cn["defaults"])

    # ------------------------------------------------------------------
    def _read_globals(self):
        gs = self.root_node.find("GlobalSettings")
        p70 = gs.find("Properties70") if gs is not None else None
        g = {}
        for nm, dflt in [("UpAxis", 1), ("UpAxisSign", 1), ("FrontAxis", 2),
                         ("FrontAxisSign", 1), ("CoordAxis", 0),
                         ("CoordAxisSign", 1), ("UnitScaleFactor", 1.0),
                         ("TimeMode", 0), ("CustomFrameRate", -1.0)]:
            g[nm] = _prop_value(p70, nm, dflt)
        self.globals = g
        fps = _TIME_MODE_FPS.get(int(g["TimeMode"]))
        if fps is None:
            fps = g["CustomFrameRate"] if g["CustomFrameRate"] > 0 else None
        self.fps = fps
        # UnitScaleFactor is centimetres per file unit
        self.metres_per_unit = g["UnitScaleFactor"] / 100.0

    def _time_range(self):
        ts = [c.t for c in self.anim_curves.values() if len(c.t)]
        if ts:
            self.key_times = np.unique(np.concatenate(ts))
        else:
            self.key_times = np.array([0.0])
        self.t0, self.t1 = float(self.key_times[0]), float(self.key_times[-1])

    # ------------------------------------------------------------------
    def resolve(self, name):
        """Node lookup tolerant of Theia's 'person_N:' namespace prefix."""
        if name in self._by_name:
            return self._by_name[name]
        hits = sorted((n for n in self._by_name if n.split(":")[-1] == name))
        return self._by_name[hits[0]] if hits else None

    def global_transforms(self, names, times):
        """Global 4x4 transforms for `names` at `times` -> {name: (T,4,4)}.

        Only the requested nodes and their ancestors are evaluated.
        """
        times = np.atleast_1d(np.asarray(times, dtype=float))
        wanted = []
        for n in names:
            j = self.resolve(n) if isinstance(n, str) else n
            if j is not None:
                wanted.append(j)

        order, seen = [], set()

        def visit(j):
            if j.uid in seen:
                return
            seen.add(j.uid)
            if j.parent is not None:
                visit(j.parent)
            order.append(j)

        for j in wanted:
            visit(j)

        glob = {}
        for j in order:
            L = j.local_matrices(times)
            glob[j.uid] = L if j.parent is None else glob[j.parent.uid] @ L
        return {j.name: glob[j.uid] for j in wanted}

    def has_nonunit_scale(self):
        for j in self.joints.values():
            if not np.allclose(j.lcl_s, 1.0, atol=1e-6):
                return True
        return False


# %%==========================================================================
#  5. GEOMETRY HELPERS
# ============================================================================

def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-12, v / n, 0.0)


def orthonormalise(R):
    """Nearest rotation matrix (strips scale/shear). Works on (...,3,3)."""
    U, _, Vt = np.linalg.svd(R)
    D = np.ones(U.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


def up_axis_to_z(up_axis, up_sign):
    """Right-handed rotation taking a file's declared up-axis onto +Z."""
    s = 1.0 if up_sign >= 0 else -1.0
    a = int(up_axis)
    if a == 2:
        return np.eye(3) if s > 0 else np.array([[1, 0, 0], [0, -1, 0],
                                                 [0, 0, -1]], float)
    if a == 1:
        return (np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float) if s > 0
                else np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float))
    return (np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float) if s > 0
            else np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float))


def yaw_matrix(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def decompose_yaw_tilt(R):
    """Heading about +Z, and how far R tips the vertical away from +Z."""
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    tilt = float(np.degrees(np.arccos(np.clip((R @ [0, 0, 1.0])[2], -1, 1))))
    return yaw, tilt


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def angular_residuals_deg(U, V, R):
    """Angle between each u_i and R v_i, in degrees."""
    return np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", U, V @ R.T), -1.0, 1.0)))


# %%==========================================================================
#  6. TRIAL  -- one FBX reduced to canonical (Z-up, metres) landmark positions
# ============================================================================

class Trial:
    """One FBX file, evaluated and quality-controlled."""

    def __init__(self, path, system, label=None):
        if system not in ("theia", "m4d"):
            raise ValueError("system must be 'theia' or 'm4d'")
        self.path = path
        self.system = system
        self.label = label or os.path.basename(path)
        self.scene = Scene(path)
        self.C = up_axis_to_z(self.scene.globals["UpAxis"],
                              self.scene.globals["UpAxisSign"])
        self.scale = self.scene.metres_per_unit      # file units -> metres
        self._evaluate()

    # ------------------------------------------------------------------
    def _spec(self, key):
        spec = LANDMARKS[key][0 if self.system == "theia" else 1]
        return spec

    def _landmark_series(self, key, cache):
        """Canonical (Z-up, metres) position of one landmark -> (T, 3) or None."""
        spec = self._spec(key)
        if isinstance(spec, str):
            nodes = [spec]
        elif spec[0] == "children_of":
            parent = self.scene.resolve(spec[1])
            if parent is None:
                return None
            nodes = [c.name for c in parent.children if c.type == "LimbNode"]
        elif spec[0] == "centroid":
            nodes = list(spec[1])
        else:
            raise ValueError(f"bad landmark spec {spec!r}")

        pts = []
        for n in nodes:
            node = self.scene.resolve(n)
            if node is None or node.name not in cache:
                continue
            pts.append(cache[node.name][:, :3, 3])
        if not pts:
            return None
        p = np.mean(pts, axis=0) * self.scale
        return np.einsum("ij,tj->ti", self.C, p)

    def _needed_nodes(self):
        """Every node name this trial has to evaluate."""
        names = []
        for key in LANDMARKS:
            spec = self._spec(key)
            if isinstance(spec, str):
                names.append(spec)
            elif spec[0] == "children_of":
                p = self.scene.resolve(spec[1])
                if p is not None:
                    names += [c.name for c in p.children if c.type == "LimbNode"]
            elif spec[0] == "centroid":
                names += list(spec[1])
        for seg, pair in SEGMENT_FRAME_NODES.items():
            names.append(pair[0 if self.system == "theia" else 1])
        return names

    # ------------------------------------------------------------------
    def _evaluate(self):
        sc = self.scene
        times = sc.key_times
        cache = sc.global_transforms(self._needed_nodes(), times)

        # -- landmarks -------------------------------------------------
        raw = {}
        self.missing = []
        for key in LANDMARKS:
            series = self._landmark_series(key, cache)
            if series is None:
                self.missing.append(key)
            else:
                raw[key] = series

        fit_keys = [k for k in active_translation_landmarks() if k in raw]
        if len(fit_keys) < 4:
            raise ValueError(f"{self.label}: only {len(fit_keys)} fitting "
                             f"landmarks resolved; check LANDMARKS. "
                             f"Missing: {self.missing}")

        # frame QC uses EVERY resolved landmark, including the ones that are
        # not fitted: the more of the body it sees, the more decisively a
        # bind pose stands out.
        P = np.stack([raw[k] for k in raw], axis=1)          # (F, J, 3)
        self.all_times = times
        self.valid = self._select_frames(P)
        self.times = times[self.valid]
        self.pos = {k: v[self.valid] for k, v in raw.items()}

        # -- segment local frames (diagnostic only) --------------------
        self.frames = {}
        for seg, pair in SEGMENT_FRAME_NODES.items():
            node = pair[0 if self.system == "theia" else 1]
            resolved = sc.resolve(node)
            if resolved is None or resolved.name not in cache:
                continue
            R = orthonormalise(cache[resolved.name][self.valid, :3, :3])
            self.frames[seg] = np.einsum("ij,tjk->tik", self.C, R)

        # -- residual motion within the retained frames ----------------
        # median over landmarks of each landmark's travel: a max statistic
        # would call a trial dynamic because one wrist twitched.
        kept = np.stack([raw[k][self.valid] for k in fit_keys], axis=1)
        if len(kept) > 1:
            travel = np.linalg.norm(np.ptp(kept, axis=0), axis=-1)   # (J,)
            self.motion = float(np.median(travel))
            self.motion_max = float(travel.max())
        else:
            self.motion = self.motion_max = 0.0
        if TRIAL_MODE == "static":
            self.is_static = True
        elif TRIAL_MODE == "dynamic":
            self.is_static = False
        else:
            self.is_static = self.motion < STATIC_MOTION_THRESHOLD
        up = kept[:, :, 2]
        self.span = float(np.median(up.max(1) - up.min(1)))

    # ------------------------------------------------------------------
    def _select_frames(self, P):
        """Drop bind/rest poses and filter edge artifacts.

        Two stages, because they are different failure modes at very
        different magnitudes: a bind pose sits tens of centimetres from the
        data, a filter edge artifact a few millimetres -- and the latter
        survives any threshold loose enough to be safe against real motion.
        Shape only (centroid removed), so translating the subject is fine.
        """
        F = P.shape[0]
        self.dropped_gross, self.dropped_edge = [], []
        if F < MIN_FRAMES:
            return np.arange(F)
        centred = P - P.mean(axis=1, keepdims=True)

        def deviation(idx):
            med = np.median(centred[idx], axis=0)
            return np.sqrt(((centred - med) ** 2).sum(-1)).mean(-1)

        dev = deviation(np.arange(F))                       # stage 1: gross
        mad = np.median(np.abs(dev - np.median(dev)))
        keep = np.where(dev <= np.median(dev) + 6 * max(mad, 1e-4)
                        + GROSS_FLOOR_M)[0]
        if len(keep) < MIN_FRAMES:
            return keep if len(keep) >= 2 else np.arange(F)
        self.dropped_gross = sorted(set(range(F)) - set(keep.tolist()))

        dev = deviation(keep)                               # stage 2: relative
        sub = dev[keep]
        mad = np.median(np.abs(sub - np.median(sub)))
        thr = max(np.median(sub) + 5 * max(mad, 1e-6), EDGE_FLOOR_M)
        keep2 = keep[sub <= thr]
        if len(keep2) >= MIN_FRAMES:
            self.dropped_edge = sorted(set(keep.tolist()) - set(keep2.tolist()))
            return keep2
        return keep

    # ------------------------------------------------------------------
    def mean_pose(self):
        """Mean landmark positions in the common Z-up, metres frame."""
        return {k: v.mean(axis=0) for k, v in self.pos.items()}

    def raw_mean_pose(self):
        """Mean landmark positions in this file's OWN raw coordinates.

        The inverse of the canonicalisation, so it is exactly what a reader
        of the FBX would get. Used to verify T end to end.
        """
        return {k: self.C.T @ v.mean(axis=0) / self.scale
                for k, v in self.pos.items()}

    def mean_frames(self):
        return {k: orthonormalise(v.mean(axis=0)) for k, v in self.frames.items()}

    def sample_at(self, t):
        return {k: np.stack([np.interp(t, self.times, v[:, d])
                             for d in range(3)], axis=-1)
                for k, v in self.pos.items()}


# %%==========================================================================
#  7. OBSERVATIONS  -- unit directions and point pairs, with derived weights
# ============================================================================

def direction_sigma_deg(kind, a, b, length_m):
    """Angular uncertainty of the direction a->b, in degrees.

    Endpoint definition noise sqrt(sa^2+sb^2) spread over the segment length,
    plus an irreducible systematic floor. This is what the fit weights by,
    and it is why the long inter-limb ML axes end up carrying the heading:
    the same 12 mm of landmark ambiguity is 5.6 deg across a 0.12 m hip
    width but 0.7 deg across a 0.95 m wrist-to-wrist span.
    """
    sa = LANDMARK_SIGMA_MM.get(a, 15.0) / 1000.0
    sb = LANDMARK_SIGMA_MM.get(b, 15.0) / 1000.0
    endpoint = np.hypot(sa, sb)
    if kind == "ML":
        endpoint *= ML_ASYMMETRY_FRACTION
    geometric = np.degrees(endpoint / max(length_m, 1e-6))
    return float(np.hypot(geometric, SYSTEMATIC_FLOOR_DEG))


def active_axes():
    """The (label, from, to, kind) list actually used, honouring the flags."""
    out = []
    for lab, a, b, needs_forearm in SEGMENT_AXES:
        if needs_forearm and not INCLUDE_FOREARM:
            continue
        out.append((lab, a, b, "long"))
    for lab, a, b, needs_forearm in ML_AXES:
        if needs_forearm and not INCLUDE_FOREARM:
            continue
        out.append((lab, a, b, "ML"))
    return out


def active_translation_landmarks():
    """Joint centres used for the translation, honouring INCLUDE_FOREARM.

    Turning the forearm off drops the wrist everywhere, not just from the
    direction fit: the wrist is only in the model because the forearm is.
    """
    if INCLUDE_FOREARM:
        return list(TRANSLATION_LANDMARKS)
    return [k for k in TRANSLATION_LANDMARKS if not k.endswith("wrist")]


def build_observations(theia, m4d):
    """Matched unit directions and point pairs for one trial pair.

    Static pairs collapse to the mean pose (the frame grids need not
    overlap).  Dynamic pairs are resampled onto the overlapping part of a
    common time base, which is what handles differing frame rates.
    """
    if theia.is_static and m4d.is_static:
        poses = [(theia.mean_pose(), m4d.mean_pose())]
        info = dict(mode="static", n_samples=1)
    else:
        t0 = max(theia.times[0], m4d.times[0])
        t1 = min(theia.times[-1], m4d.times[-1])
        if t1 <= t0:
            raise ValueError(
                f"{theia.label}/{m4d.label}: dynamic trials whose time ranges "
                f"do not overlap (Theia {theia.times[0]:.3f}-{theia.times[-1]:.3f} s, "
                f"M4D {m4d.times[0]:.3f}-{m4d.times[-1]:.3f} s). Check the sync.")
        fps = min(theia.scene.fps or 60.0, m4d.scene.fps or 60.0)
        n = max(2, int(round((t1 - t0) * fps)) + 1)
        grid = np.linspace(t0, t1, n)
        pt_all, pm_all = theia.sample_at(grid), m4d.sample_at(grid)
        poses = [({k: v[i] for k, v in pt_all.items()},
                  {k: v[i] for k, v in pm_all.items()}) for i in range(n)]
        info = dict(mode="dynamic", n_samples=n, t_start=float(t0),
                    t_end=float(t1))
    info.update(theia_motion=theia.motion, m4d_motion=m4d.motion)

    dirs, pts, val = [], [], []
    for pt, pm in poses:
        for lab, a, b, kind in active_axes():
            if not (a in pt and b in pt and a in pm and b in pm):
                continue
            u, v = pt[b] - pt[a], pm[b] - pm[a]
            lu, lv = np.linalg.norm(u), np.linalg.norm(v)
            if lu < 1e-6 or lv < 1e-6:
                continue
            sigma = direction_sigma_deg(kind, a, b, 0.5 * (lu + lv))
            dirs.append(dict(label=lab, kind=kind, u=u / lu, v=v / lv,
                             w=1.0 / sigma ** 2, sigma_deg=sigma))
        for k in active_translation_landmarks():
            if k in pt and k in pm:
                s = LANDMARK_SIGMA_MM.get(k, 15.0) / 1000.0
                pts.append(dict(label=k, a=pt[k], b=pm[k], w=1.0 / s ** 2))
        for k in VALIDATION_LANDMARKS:
            if k in pt and k in pm:
                val.append(dict(label=k, a=pt[k], b=pm[k]))
    return dirs, pts, val, info


def _arrays(dirs):
    U = np.array([d["u"] for d in dirs])
    V = np.array([d["v"] for d in dirs])
    w = np.array([d["w"] for d in dirs], dtype=float)
    return U, V, w / w.mean()


# %%==========================================================================
#  8. SOLVERS
# ============================================================================

def _irls_weights(res, w0, huber):
    """Huber down-weighting, with a MAD-based scale so it adapts to the data."""
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    scale = max(huber, mad)
    return w0 * np.minimum(1.0, scale / np.maximum(res, 1e-9))


def solve_rotation_full(dirs, n_iter=15):
    """Unconstrained 3-DOF Wahba/Kabsch on unit vectors, robust (IRLS)."""
    U, V, w0 = _arrays(dirs)
    w, R = w0.copy(), np.eye(3)
    for _ in range(n_iter):
        M = (U * w[:, None]).T @ V
        A, _, Bt = np.linalg.svd(M)
        R = A @ np.diag([1.0, 1.0, np.sign(np.linalg.det(A @ Bt))]) @ Bt
        w_new = _irls_weights(angular_residuals_deg(U, V, R), w0, HUBER_DEG)
        if np.allclose(w_new, w, atol=1e-10):
            break
        w = w_new
    return R, angular_residuals_deg(U, V, R), w, w0


def solve_rotation_yaw(dirs, n_iter=15):
    """Rotation constrained to the vertical axis, solved IN CLOSED FORM.

    Maximising sum_i w_i u_i . Rz(th) v_i gives

        th = atan2( sum w (u_y v_x - u_x v_y),  sum w (u_x v_x + u_y v_y) )

    This is the correct constrained estimator.  Fitting 3 DOF and then
    reading the yaw off the result is NOT the same thing: the unconstrained
    tilt is free to absorb skeleton mismatch and it drags the yaw with it.
    On the D05_C1 A-pose that difference is what makes the anatomical
    subsets agree (spread 0.6 deg) instead of disagree (spread 2.7 deg).
    """
    U, V, w0 = _arrays(dirs)
    w = w0.copy()
    R = np.eye(3)
    for _ in range(n_iter):
        num = float((w * (U[:, 1] * V[:, 0] - U[:, 0] * V[:, 1])).sum())
        den = float((w * (U[:, 0] * V[:, 0] + U[:, 1] * V[:, 1])).sum())
        R = yaw_matrix(np.degrees(np.arctan2(num, den)))
        w_new = _irls_weights(angular_residuals_deg(U, V, R), w0, HUBER_DEG)
        if np.allclose(w_new, w, atol=1e-10):
            break
        w = w_new
    return R, angular_residuals_deg(U, V, R), w, w0


def solve_rotation(dirs, model=None):
    model = model or ROTATION_MODEL
    return (solve_rotation_yaw(dirs) if model == "yaw"
            else solve_rotation_full(dirs))


def solve_translation(pts, R, scale=1.0, n_iter=15):
    """Robust t minimising || p_theia - (scale * R p_m4d + t) ||."""
    A = np.array([p["a"] for p in pts])
    B = np.array([p["b"] for p in pts])
    w0 = np.array([p["w"] for p in pts], dtype=float)
    w0 = w0 / w0.mean()
    d = A - scale * (B @ R.T)
    w = w0.copy()
    t = np.average(d, axis=0, weights=w)
    for _ in range(n_iter):
        r = np.linalg.norm(d - t, axis=1) * 1000.0             # mm
        w_new = _irls_weights(r, w0, HUBER_MM)
        t_new = np.average(d, axis=0, weights=w_new)
        if np.allclose(t_new, t, atol=1e-10):
            t, w = t_new, w_new
            break
        t, w = t_new, w_new
    return t, d - t, w, w0


def solve_scale(dirs_pts, R):
    """Uniform scale from the ratio of centred point spreads (only if asked)."""
    A = np.array([p["a"] for p in dirs_pts])
    B = np.array([p["b"] for p in dirs_pts]) @ R.T
    A = A - A.mean(0)
    B = B - B.mean(0)
    return float((A * B).sum() / max((B * B).sum(), 1e-12))


def weighted_rms(res, w):
    return float(np.sqrt(np.average(res ** 2, weights=w)))


# %%==========================================================================
#  9. DIAGNOSTICS
# ============================================================================

def segment_length_table(theia, m4d):
    """Theia/M4D length ratio per segment -- the unit and skeleton check."""
    pt, pm = theia.mean_pose(), m4d.mean_pose()
    rows = []
    extra = [("l_foot", "l_ankle", "l_toe"), ("r_foot", "r_ankle", "r_toe"),
             ("trunk", "pelvis", "thorax")]
    used = {(a, b) for _, a, b, _ in active_axes()}
    for lab, a, b, kind in active_axes():
        rows.append((lab, a, b, kind, True))
    for lab, a, b in extra:
        if (a, b) not in used:
            rows.append((lab, a, b, "excluded", False))
    out = []
    for lab, a, b, kind, fitted in rows:
        if not (a in pt and b in pt and a in pm and b in pm):
            continue
        lt = float(np.linalg.norm(pt[b] - pt[a]))
        lm = float(np.linalg.norm(pm[b] - pm[a]))
        if lt < 1e-6 or lm < 1e-6:
            continue
        out.append(dict(segment=lab, kind=kind, fitted=fitted,
                        theia_m=lt, m4d_m=lm, ratio=lt / lm,
                        sigma_deg=(direction_sigma_deg(kind, a, b,
                                                       0.5 * (lt + lm))
                                   if fitted else None)))
    return out


def handedness(trial):
    """Signed volume of (medio-lateral, vertical, antero-posterior).

    A mirrored file cannot be fixed by any rotation, so this must be caught
    rather than fitted.
    """
    p = trial.mean_pose()
    need = ["l_hip", "r_hip", "l_shoulder", "l_ankle", "l_toe", "r_ankle",
            "r_toe"]
    if not all(k in p for k in need):
        return None
    ml = p["l_hip"] - p["r_hip"]
    up = p["l_shoulder"] - p["l_hip"]
    ap = (p["l_toe"] - p["l_ankle"]) + (p["r_toe"] - p["r_ankle"])
    return float(np.linalg.det(np.stack([ml, up, ap], axis=1)))


def yaw_observability(dirs, weights):
    """Two things that are easy to conflate.

    yaw_information = sum w |horizontal component|^2 -- the Fisher
    information for a rotation about the vertical.  A vertical direction
    contributes nothing, a horizontal one contributes fully.  Parallel
    horizontal vectors each contribute: they are redundant, not degenerate.

    azimuthal_diversity = eigenvalue ratio of the horizontal scatter.  This
    does NOT affect identifiability.  It says whether independent anatomical
    directions exist to cross-check each other, which is the only protection
    against a systematic bias shared by every vector pointing the same way.
    An A-pose is nearly planar, so it is low by construction.
    """
    H = np.array([d["v"][:2] * np.sqrt(max(w, 0.0))
                  for d, w in zip(dirs, weights)])
    if len(H) < 2:
        return None
    ev = np.linalg.eigvalsh(H.T @ H)
    total_w = float(sum(max(w, 0.0) for w in weights))
    info = float((H ** 2).sum())
    return dict(yaw_information=info,
                yaw_information_per_obs=info / max(total_w, 1e-9),
                azimuthal_diversity=float(ev[0] / ev[1]) if ev[1] > 1e-12 else 0.0)


# A subset whose directions are nearly vertical carries almost no heading
# information, so its yaw estimate is noise amplified by 1/leverage. Report
# it, but do not let it inflate the quoted spread.
MIN_YAW_LEVERAGE = 0.30


def subset_yaw(dirs):
    """Yaw from anatomically independent subsets -- the honest uncertainty.

    If the leg widths and the arm widths disagree, that disagreement IS the
    uncertainty, whatever the residuals say. Subsets below MIN_YAW_LEVERAGE
    (mean horizontal component of their direction vectors) are shown for
    information only: in a standing pose the leg long axes are 9 deg off
    vertical, so a 0.5 deg error in them becomes several degrees of yaw.
    """
    out = []
    for name, labels in YAW_SUBSETS.items():
        sub = [d for d in dirs if d["label"] in labels]
        if len({d["label"] for d in sub}) < 2:
            continue
        R, res, w, _ = solve_rotation_yaw(sub)
        yaw, _ = decompose_yaw_tilt(R)
        lever = float(np.mean([np.linalg.norm(d["v"][:2]) for d in sub]))
        out.append(dict(subset=name, yaw_deg=yaw,
                        n_labels=len({d["label"] for d in sub}),
                        mean_horizontal_leverage=lever,
                        informative=lever >= MIN_YAW_LEVERAGE,
                        rms_residual_deg=weighted_rms(res, w)))

    def _spread(rows):
        ys = [o["yaw_deg"] for o in rows]
        return float(max(ys) - min(ys)) if len(ys) > 1 else 0.0

    return out, dict(all_deg=_spread(out),
                     informative_deg=_spread([o for o in out
                                              if o["informative"]]))


def leave_one_out(dirs, pts, model, scale):
    """Refit with each direction label removed in turn."""
    labels = sorted({d["label"] for d in dirs})
    out = []
    for lab in labels:
        sub = [d for d in dirs if d["label"] != lab]
        if len({d["label"] for d in sub}) < 3:
            continue
        R, _, _, _ = solve_rotation(sub, model)
        t, _, _, _ = solve_translation(pts, R, scale)
        yaw, tilt = decompose_yaw_tilt(R)
        out.append(dict(excluded=lab, yaw_deg=yaw, tilt_deg=tilt, t=t.tolist()))
    return out


def bootstrap(dirs, pts, model, scale, n=None, seed=None):
    """Resample direction LABELS with replacement, not observations.

    The dominant error here is a systematic per-segment definition mismatch,
    not independent per-frame noise, so the segment is the unit of
    uncertainty.  The spread of transformed points on the subject is what
    matters downstream -- the translation vector alone understates it when
    the subject is far from the origin and overstates it when near.
    """
    n = N_BOOTSTRAP if n is None else n
    rng = np.random.default_rng(RANDOM_SEED if seed is None else seed)
    labels = sorted({d["label"] for d in dirs})
    by_label = {l: [d for d in dirs if d["label"] == l] for l in labels}
    probes = np.array([p["b"] for p in pts])

    yaws, tilts, ts, moved = [], [], [], []
    for _ in range(n):
        pick = rng.choice(len(labels), size=len(labels), replace=True)
        sub = [d for i in pick for d in by_label[labels[i]]]
        if len({d["label"] for d in sub}) < 3:
            continue
        try:
            R, _, _, _ = solve_rotation(sub, model)
            t, _, _, _ = solve_translation(pts, R, scale, n_iter=6)
        except Exception:
            continue
        y, ti = decompose_yaw_tilt(R)
        yaws.append(y)
        tilts.append(ti)
        ts.append(t)
        moved.append(scale * (probes @ R.T) + t)
    if len(yaws) < 10:
        return None
    ts = np.array(ts)
    moved = np.array(moved)
    dist = np.linalg.norm(moved - moved.mean(axis=0), axis=2)
    return dict(
        n=len(yaws),
        yaw_sd_deg=float(np.std(yaws)),
        yaw_ci95_deg=[float(np.percentile(yaws, 2.5)),
                      float(np.percentile(yaws, 97.5))],
        tilt_sd_deg=float(np.std(tilts)),
        t_sd_mm=(np.std(ts, axis=0) * 1000).tolist(),
        transformed_point_rms_mm=float(np.sqrt((dist ** 2).mean()) * 1000),
        transformed_point_p95_mm=float(np.percentile(dist, 95) * 1000),
    )


def segment_frame_offsets(theia, m4d, R):
    """Per-segment local-frame convention offset  O_s = (R R_m4d)^T R_theia.

    This is the number that says whether fitting on FULL segment
    orientations could ever work.  If the O_s were all equal, the model
    R_theia,s = G R_m4d,s O would be identifiable and worth using.  They are
    not: on D05_C1 they span ~30 deg, so the axial roll is pure convention.
    """
    Ft, Fm = theia.mean_frames(), m4d.mean_frames()
    rows = []
    for seg in sorted(set(Ft) & set(Fm)):
        O = (R @ Fm[seg]).T @ Ft[seg]
        rows.append(dict(segment=seg, offset_deg=rotation_angle_deg(O),
                         O=O))
    if len(rows) < 2:
        return rows, None
    # how far from "all offsets identical" are we?
    Os = [r["O"] for r in rows]
    mean_O = orthonormalise(np.mean(Os, axis=0))
    spread = [rotation_angle_deg(mean_O.T @ O) for O in Os]
    shared = dict(mean_offset_deg=rotation_angle_deg(mean_O),
                  spread_about_mean_deg=float(np.mean(spread)),
                  max_spread_deg=float(np.max(spread)))
    for r in rows:
        r.pop("O")
    return rows, shared


# %%==========================================================================
#  10. MAIN SOLVE
# ============================================================================

def solve_alignment(pairs=None, model=None, verbose=None):
    """Fit the MOVE4D -> Theia3D transform. Returns (report_dict, T, context)."""
    pairs = TRIAL_PAIRS if pairs is None else pairs
    model = ROTATION_MODEL if model is None else model
    verbose = VERBOSE if verbose is None else verbose
    if model not in ("yaw", "full"):
        raise ValueError("ROTATION_MODEL must be 'yaw' or 'full'")

    trials, dirs, pts, vals, infos, lengths = [], [], [], [], [], []
    for theia_path, m4d_path in pairs:
        th = Trial(theia_path, "theia")
        m4 = Trial(m4d_path, "m4d")
        d, p, v, info = build_observations(th, m4)
        info.update(theia_file=th.label, m4d_file=m4.label)
        dirs += d
        pts += p
        vals += v
        infos.append(info)
        lengths.append(segment_length_table(th, m4))
        trials.append((th, m4))

    # ---- sanity checks before fitting anything -----------------------
    warnings = []
    hand = dict(theia=handedness(trials[0][0]), m4d=handedness(trials[0][1]))
    if (hand["theia"] is not None and hand["m4d"] is not None
            and np.sign(hand["theia"]) != np.sign(hand["m4d"])):
        raise ValueError(
            "The two skeletons have OPPOSITE handedness (one file is "
            "mirrored). No rigid transform can fix that -- fix the export.")

    flat_lengths = [r for tbl in lengths for r in tbl if r["fitted"]]
    ratios = np.array([r["ratio"] for r in flat_lengths])
    median_ratio = float(np.median(ratios))
    if not 0.8 < median_ratio < 1.25:
        warnings.append(
            f"median segment length ratio is {median_ratio:.3f}: the two files "
            f"are probably in different units. Set ALLOW_SCALE = True, or fix "
            f"the export.")
    for th, m4 in trials:
        for tr in (th, m4):
            if tr.scene.has_nonunit_scale() and any(
                    j.inherit_type != 0 for j in tr.scene.joints.values()):
                warnings.append(
                    f"{tr.label}: non-unit node scaling with a non-default "
                    f"InheritType; global positions may be slightly off.")

    # ---- rotation ----------------------------------------------------
    R, ang_res, w_rot, w_rot0 = solve_rotation(dirs, model)
    yaw, tilt = decompose_yaw_tilt(R)

    # ---- scale (optional) --------------------------------------------
    scale = solve_scale(pts, R) if ALLOW_SCALE else 1.0
    if ALLOW_SCALE and 0.9 < scale < 1.1:
        warnings.append(
            f"ALLOW_SCALE fitted a scale of {scale:.4f}. A unit mismatch would "
            f"be a round number (2.54, 10, 100, 1000); {scale:.4f} is the two "
            f"skeletons being different SIZES, and scaling it away hides a "
            f"real disagreement instead of aligning coordinate systems. Set "
            f"ALLOW_SCALE = False unless you know the units differ.")

    # ---- translation -------------------------------------------------
    t, pos_res, w_tr, w_tr0 = solve_translation(pts, R, scale)

    # ---- the other rotation model, for comparison --------------------
    alt_model = "full" if model == "yaw" else "yaw"
    R_alt, ang_alt, w_alt, _ = solve_rotation(dirs, alt_model)
    t_alt, res_alt, _, _ = solve_translation(pts, R_alt, scale)

    # ---- per-observation summaries -----------------------------------
    per_direction = {}
    for d, res_i, w, w0 in zip(dirs, ang_res, w_rot, w_rot0):
        per_direction.setdefault(d["label"], []).append(
            dict(res=res_i, w=w, w0=w0, sigma=d["sigma_deg"], kind=d["kind"]))
    per_direction = {
        k: dict(kind=v[0]["kind"],
                mean_deg=float(np.mean([x["res"] for x in v])),
                sigma_deg=float(v[0]["sigma"]),
                prior_weight=float(np.mean([x["w0"] for x in v])),
                robust_factor=float(np.mean([x["w"] / max(x["w0"], 1e-12)
                                             for x in v])),
                n=len(v))
        for k, v in per_direction.items()}

    per_joint = {}
    for p, res_i, w, w0 in zip(pts, pos_res, w_tr, w_tr0):
        per_joint.setdefault(p["label"], []).append(
            dict(d=np.linalg.norm(res_i), vec=res_i, w=w, w0=w0))
    per_joint = {
        k: dict(mean_mm=float(np.mean([x["d"] for x in v]) * 1000),
                vector_mm=(np.mean([x["vec"] for x in v], axis=0) * 1000).tolist(),
                prior_weight=float(np.mean([x["w0"] for x in v])),
                robust_factor=float(np.mean([x["w"] / max(x["w0"], 1e-12)
                                             for x in v])),
                n=len(v))
        for k, v in per_joint.items()}

    validation = {}
    for v in vals:
        d = v["a"] - (scale * R @ v["b"] + t)
        validation.setdefault(v["label"], []).append(d)
    validation = {k: dict(mean_mm=float(np.linalg.norm(np.mean(v, 0)) * 1000),
                          vector_mm=(np.mean(v, 0) * 1000).tolist())
                  for k, v in validation.items()}

    pos_rms = float(np.sqrt((np.linalg.norm(pos_res, axis=1) ** 2).mean()) * 1000)
    pos_rms_alt = float(np.sqrt((np.linalg.norm(res_alt, axis=1) ** 2).mean())
                        * 1000)

    # ---- uncertainty & stability -------------------------------------
    obs = yaw_observability(dirs, w_rot)
    subsets, subset_spread = subset_yaw(dirs)   # spread is a dict
    boot = bootstrap(dirs, pts, model, scale)
    loo = leave_one_out(dirs, pts, model, scale)
    frame_rows, frame_shared = segment_frame_offsets(trials[0][0],
                                                     trials[0][1], R)

    # ---- assemble T in RAW file coordinates --------------------------
    # canonical:  p_can = C @ (metres_per_unit * p_raw)
    # fit:        p_theia_can = scale * R @ p_m4d_can + t
    # therefore:  p_theia_raw = Ct^-1 @ (scale*R @ Cm @ p_m4d_raw * km + t) / kt
    th0, m40 = trials[0]
    km, kt = m40.scale, th0.scale
    M = th0.C.T @ (scale * R @ m40.C) * (km / kt)
    T = np.eye(4)
    T[:3, :3] = M
    T[:3, 3] = th0.C.T @ t / kt

    report = dict(
        T_m4d_to_theia=T.tolist(),
        rotation_canonical=R.tolist(),
        translation_canonical_m=t.tolist(),
        scale=scale,
        rotation_model=model,
        yaw_deg=yaw,
        tilt_deg=tilt,
        total_rotation_deg=rotation_angle_deg(R),
        warnings=warnings,
        handedness=hand,
        vertical_span_m=dict(theia=th0.span, m4d=m40.span),
        segment_lengths=flat_lengths + [r for tbl in lengths for r in tbl
                                        if not r["fitted"]],
        median_length_ratio=median_ratio,
        length_ratio_iqr=float(np.percentile(ratios, 75)
                               - np.percentile(ratios, 25)),
        angular_residual_deg=dict(
            weighted_rms=weighted_rms(ang_res, w_rot),
            mean=float(np.mean(ang_res)), max=float(np.max(ang_res)),
            per_direction=per_direction),
        position_residual=dict(rms_mm=pos_rms, per_joint=per_joint),
        model_comparison=dict(
            chosen=model, chosen_angular_rms_deg=weighted_rms(ang_res, w_rot),
            chosen_position_rms_mm=pos_rms,
            alternative=alt_model,
            alternative_angular_rms_deg=weighted_rms(ang_alt, w_alt),
            alternative_position_rms_mm=pos_rms_alt,
            alternative_yaw_deg=decompose_yaw_tilt(R_alt)[0],
            alternative_tilt_deg=decompose_yaw_tilt(R_alt)[1]),
        validation_landmarks=validation,
        yaw_observability=obs,
        subset_agreement=dict(subsets=subsets, **subset_spread),
        uncertainty=boot,
        leave_one_out=dict(
            runs=loo,
            yaw_spread_deg=float(np.ptp([l["yaw_deg"] for l in loo])) if loo else 0.0,
            t_spread_mm=(float(np.ptp(np.array([l["t"] for l in loo]), axis=0).max())
                         * 1000) if loo else 0.0),
        segment_frame_offsets=dict(per_segment=frame_rows, shared=frame_shared),
        trials=infos,
        n_direction_obs=len(dirs), n_point_obs=len(pts),
    )
    context = dict(trials=trials, dirs=dirs, pts=pts, R=R, t=t, scale=scale)
    chk = verify_transform(T, context)
    # In static mode the check evaluates exactly what was fitted, so the two
    # numbers must agree. In dynamic mode the fit spans many frames while the
    # check uses the mean pose, so they legitimately differ.
    chk["exact_comparison"] = all(i["mode"] == "static" for i in infos)
    report["raw_coordinate_check"] = chk
    if verbose:
        print_report(report, trials)
    return report, T, context


# %%==========================================================================
#  11. REPORT
# ============================================================================

def _rule(title=None, w=76):
    print("-" * w if title is None else "\n" + "-" * w + f"\n  {title}\n"
          + "-" * w)


def print_report(r, trials):
    W = 76
    print("=" * W)
    print("  MOVE4D  ->  THEIA3D   rigid alignment   "
          f"[{r['rotation_model']} model, arms + legs only]")
    print("=" * W)

    for (th, m4), info in zip(trials, r["trials"]):
        print(f"\n  {info['theia_file']}   /   {info['m4d_file']}")
        for tr, lab in ((th, "Theia"), (m4, "M4D  ")):
            sc = tr.scene
            fps = f"{sc.fps:g}" if sc.fps else "?"
            print(f"    {lab}: {len(tr.all_times):>4d} keys @ {fps} fps, "
                  f"kept {len(tr.times):>3d} | up-axis "
                  f"{'XYZ'[int(sc.globals['UpAxis'])]} | "
                  f"{sc.metres_per_unit:g} m/unit | "
                  f"motion {tr.motion * 1000:.1f} mm "
                  f"(max {tr.motion_max * 1000:.1f})")
            if tr.dropped_gross:
                print(f"           dropped {tr.dropped_gross} -- rest/bind pose")
            if tr.dropped_edge:
                print(f"           dropped {tr.dropped_edge} -- filter edge artifact")
            if tr.missing:
                print(f"           MISSING landmarks: {tr.missing}")
        print(f"    -> {info['mode'].upper()}, {info['n_samples']} sample(s)")

    for msg in r["warnings"]:
        print(f"\n  !! WARNING: {msg}")

    # ---------------------------------------------------------------
    _rule("CHECKS")
    h = r["handedness"]
    if h["theia"] is None or h["m4d"] is None:
        print("    handedness       not checked (needs hips, shoulder, "
              "ankles and toes)")
    else:
        print(f"    handedness       Theia {h['theia']:+.4f} | "
              f"M4D {h['m4d']:+.4f}   OK, not mirrored")
    print(f"    vertical span    Theia {r['vertical_span_m']['theia']:.3f} m | "
          f"M4D {r['vertical_span_m']['m4d']:.3f} m   (fitted landmarks)")
    print(f"    length ratio     median {r['median_length_ratio']:.4f} "
          f"(IQR {r['length_ratio_iqr']:.4f}) over fitted segments")
    print("\n      segment         Theia     M4D     ratio   sigma   role")
    for s in r["segment_lengths"]:
        sig = f"{s['sigma_deg']:5.2f}d" if s["sigma_deg"] else ""
        print(f"      {s['segment']:14s} {s['theia_m']:6.4f}  {s['m4d_m']:6.4f}  "
              f"{s['ratio']:7.4f}  {sig:>6s}   "
              f"{s['kind'] if s['fitted'] else 'NOT FITTED'}")

    # ---------------------------------------------------------------
    _rule("SOLUTION")
    T = np.array(r["T_m4d_to_theia"])
    print("    T, raw MOVE4D file coords -> raw Theia file coords:")
    for row in T:
        print("      [" + "  ".join(f"{v:10.6f}" for v in row) + "]")
    print(f"\n    yaw about vertical    {r['yaw_deg']:+9.3f} deg")
    print(f"    residual tilt         {r['tilt_deg']:9.3f} deg"
          + ("   (constrained to zero)" if r["rotation_model"] == "yaw" else ""))
    print(f"    total rotation        {r['total_rotation_deg']:9.3f} deg")
    tc = r["translation_canonical_m"]
    print(f"    translation (m)       [{tc[0]:+.4f} {tc[1]:+.4f} {tc[2]:+.4f}]"
          "   in the common Z-up frame")
    if r["scale"] != 1.0:
        print(f"    scale                 {r['scale']:.6f}")

    # ---------------------------------------------------------------
    _rule("FIT QUALITY  (how well segment orientations agree)")
    a = r["angular_residual_deg"]
    print(f"    direction residual    weighted RMS {a['weighted_rms']:.2f} deg "
          f"| mean {a['mean']:.2f} | max {a['max']:.2f}")
    print("\n      direction        resid   sigma   weight  robust  kind")
    for k, v in sorted(a["per_direction"].items(),
                       key=lambda x: -x[1]["mean_deg"]):
        flag = "  <- rejected by IRLS" if v["robust_factor"] < 0.7 else ""
        print(f"      {k:14s} {v['mean_deg']:6.2f}d {v['sigma_deg']:6.2f}d "
              f"{v['prior_weight']:7.2f} {v['robust_factor']:7.2f}  "
              f"{v['kind']}{flag}")
    print("      weight = 1/sigma^2 from the landmark uncertainties (prior);")
    print("      robust = extra Huber factor the fit applied (1.00 = kept).")

    p = r["position_residual"]
    print(f"\n    position residual     RMS {p['rms_mm']:.1f} mm "
          f"over fitted joint centres")
    print("      joint           resid        residual vector (mm)")
    for k, v in sorted(p["per_joint"].items(), key=lambda x: -x[1]["mean_mm"]):
        vec = "  ".join(f"{c:+6.1f}" for c in v["vector_mm"])
        print(f"      {k:14s} {v['mean_mm']:6.1f} mm   [{vec}]")

    chk = r["raw_coordinate_check"]
    print("\n    end-to-end check, T applied in RAW file coordinates:")
    if chk["exact_comparison"]:
        print(f"      RMS {chk['rms_mm']:.1f} mm, max {chk['max_mm']:.1f} mm"
              f"   (must match the {p['rms_mm']:.1f} mm above)")
        if abs(chk["rms_mm"] - p["rms_mm"]) > 0.1:
            print("      *** MISMATCH -- the up-axis / unit bookkeeping in the")
            print("          assembly of T is wrong. Do not use this matrix. ***")
    else:
        print(f"      RMS {chk['rms_mm']:.1f} mm, max {chk['max_mm']:.1f} mm"
              f"   (mean pose; the fit above spans all frames,")
        print("      so these are close but not identical)")

    mc = r["model_comparison"]
    print("\n    rotation model comparison")
    print(f"      {'model':10s}  {'angular RMS':>12s}  {'position RMS':>13s}")
    print(f"      {mc['chosen']:10s}  {mc['chosen_angular_rms_deg']:9.2f} deg  "
          f"{mc['chosen_position_rms_mm']:10.1f} mm   <- chosen")
    print(f"      {mc['alternative']:10s}  "
          f"{mc['alternative_angular_rms_deg']:9.2f} deg  "
          f"{mc['alternative_position_rms_mm']:10.1f} mm")
    if mc["alternative"] == "full":
        print(f"      the 3-DOF fit wants a tilt of "
              f"{mc['alternative_tilt_deg']:.2f} deg; it is only real if it "
              f"also\n      improves the position RMS by more than a mm or two.")
        if mc["alternative_position_rms_mm"] >= mc["chosen_position_rms_mm"] - 0.5:
            print("      -> it does not. The tilt is absorbing skeleton "
                  "mismatch. Keep 'yaw'.")
        else:
            print("      -> it does. Check both systems' floor/vertical "
                  "calibration, then consider 'full'.")

    v = r["validation_landmarks"]
    if v:
        print("\n    NOT fitted -- how far apart the skeleton definitions are.")
        print("    These are a definition gap, not alignment error. Do not")
        print("    quote them as accuracy.")
        print("      landmark        offset       offset vector (mm)")
        for k, d in sorted(v.items(), key=lambda x: -x[1]["mean_mm"]):
            vec = "  ".join(f"{c:+6.1f}" for c in d["vector_mm"])
            print(f"      {k:14s} {d['mean_mm']:6.1f} mm   [{vec}]")

    # ---------------------------------------------------------------
    _rule("WHY THE FULL SEGMENT ORIENTATIONS ARE NOT USED")
    fo = r["segment_frame_offsets"]
    print("    Local bone-axis convention offset per segment,")
    print("    O_s = (R . R_m4d,s)^T . R_theia,s :")
    for row in fo["per_segment"]:
        print(f"      {row['segment']:10s} {row['offset_deg']:7.2f} deg")
    if fo["shared"]:
        s = fo["shared"]
        print("\n    If every O_s were the same, the model "
              "R_theia = G . R_m4d . O would be")
        print("    identifiable and worth fitting. Spread about the mean "
              "offset:")
        print(f"      mean |O|   {s['mean_offset_deg']:.2f} deg     "
              f"spread {s['spread_about_mean_deg']:.2f} deg "
              f"(max {s['max_spread_deg']:.2f} deg)")
        if s["max_spread_deg"] > 5.0:
            print("      -> the offsets are NOT shared: the axial roll is pure")
            print("         convention and differs per segment. Long axes only.")
        else:
            print("      -> the offsets are nearly shared. A joint (G, O) fit")
            print("         on full orientations could add information here.")

    # ---------------------------------------------------------------
    _rule("OBSERVABILITY & UNCERTAINTY")
    o = r["yaw_observability"]
    if o:
        print(f"    yaw information       {o['yaw_information']:.2f} "
              f"({o['yaw_information_per_obs']:.2f} per unit weight)")
        if o["yaw_information_per_obs"] < 0.10:
            print("      *** yaw barely identifiable: nearly every direction is")
            print("          vertical. Do not trust the heading. ***")
        elif o["yaw_information_per_obs"] < 0.30:
            print("      LOW: the heading rests on a few horizontal vectors.")
        else:
            print("      adequate -- the bilateral ML axes carry it.")
        print(f"    azimuthal diversity   {o['azimuthal_diversity']:.4f}"
              "   (cross-check capacity, NOT identifiability)")
        if o["azimuthal_diversity"] < 0.05:
            print("      LOW, as expected for a single A-pose: the horizontal")
            print("      directions are nearly parallel, so a bias shared by")
            print("      all of them cannot be detected from within this trial.")
            print("      Pool a dynamic trial to fix this (see TRIAL_PAIRS).")

    sa = r["subset_agreement"]
    if sa["subsets"]:
        print("\n    yaw from anatomically independent subsets"
              "  (the most honest number here):")
        print(f"      {'subset':32s} {'yaw':>8s} {'leverage':>9s} {'resid':>8s}")
        for s in sa["subsets"]:
            mark = "" if s["informative"] else "   (low leverage, ignore)"
            print(f"      {s['subset']:32s} {s['yaw_deg']:+7.2f}d "
                  f"{s['mean_horizontal_leverage']:9.2f} "
                  f"{s['rms_residual_deg']:7.2f}d{mark}")
        print(f"      -> spread over informative subsets "
              f"{sa['informative_deg']:.2f} deg"
              f"   (all subsets: {sa['all_deg']:.2f} deg)")

    b = r["uncertainty"]
    if b:
        print(f"\n    bootstrap over segment labels (n={b['n']}):")
        print(f"      yaw   {r['yaw_deg']:+.2f} deg   SD {b['yaw_sd_deg']:.2f}"
              f"   95% CI [{b['yaw_ci95_deg'][0]:+.2f}, "
              f"{b['yaw_ci95_deg'][1]:+.2f}]")
        sd = b["t_sd_mm"]
        print(f"      t     SD [{sd[0]:.1f}, {sd[1]:.1f}, {sd[2]:.1f}] mm")
        print("\n    propagated onto points ON THE SUBJECT -- the number that")
        print("    actually matters downstream:")
        print(f"      RMS spread {b['transformed_point_rms_mm']:.1f} mm"
              f"   |   95th pct {b['transformed_point_p95_mm']:.1f} mm")

    lo = r["leave_one_out"]
    print(f"\n    leave-one-segment-out   yaw spread "
          f"{lo['yaw_spread_deg']:.2f} deg, t spread {lo['t_spread_mm']:.1f} mm")

    # ---------------------------------------------------------------
    _rule("BOTTOM LINE")
    ang = r["angular_residual_deg"]["weighted_rms"]
    pos = r["position_residual"]["rms_mm"]
    unc = b["transformed_point_rms_mm"] if b else float("nan")
    print(f"    Segment orientations agree to {ang:.1f} deg RMS after "
          f"alignment.")
    print(f"    Joint centres agree to {pos:.0f} mm RMS.")
    print(f"    Alignment uncertainty itself is ~{unc:.0f} mm RMS on the "
          f"subject")
    print("    (bootstrap), with independent anatomical subsets of the fit")
    print(f"    disagreeing on the heading by "
          f"{r['subset_agreement']['informative_deg']:.1f} deg.")
    print("    So the skeleton DEFINITIONS, not the alignment, are the limit.")
    print("=" * W)


# %%==========================================================================
#  12. APPLYING THE RESULT
# ============================================================================

def verify_transform(T, context):
    """Round-trip check of T in RAW file coordinates.

    The fit happens in a canonicalised frame (Z-up, metres); T is then
    rebuilt to act on raw FBX coordinates. That rebuild involves the
    up-axis matrices and the unit scale factors of both files, which is
    exactly the sort of bookkeeping that silently goes wrong. So: take the
    raw MOVE4D landmarks, push them through T, and compare with the raw
    Theia landmarks. This must reproduce the fitted residual.
    """
    th, m4 = context["trials"][0]
    pt, pm = th.raw_mean_pose(), m4.raw_mean_pose()
    keys = [k for k in active_translation_landmarks() if k in pt and k in pm]
    moved = apply_transform(T, np.array([pm[k] for k in keys]))
    target = np.array([pt[k] for k in keys])
    # residuals are in Theia file units -> metres -> mm
    res_mm = np.linalg.norm(moved - target, axis=1) * th.scale * 1000.0
    return dict(rms_mm=float(np.sqrt((res_mm ** 2).mean())),
                max_mm=float(res_mm.max()),
                per_landmark_mm={k: float(v) for k, v in zip(keys, res_mm)})


def apply_transform(T, points):
    """Map (..., 3) MOVE4D positions (raw file coords) into Theia coords."""
    p = np.asarray(points, dtype=float)
    return (p.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]).reshape(p.shape)


def transform_orientation(T, R_m4d):
    """Re-express a MOVE4D orientation (..., 3, 3) in Theia coordinates.

    NOTE this only changes the coordinate system. It does NOT convert
    MOVE4D's local bone-axis convention into Theia's -- see the
    'segment frame offsets' section of the report.
    """
    M = T[:3, :3]
    s = float(np.cbrt(abs(np.linalg.det(M))))
    return (M / s) @ np.asarray(R_m4d, dtype=float)


def save_outputs(report, T, outdir=None):
    outdir = OUTDIR if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)
    np.savetxt(os.path.join(outdir, "T_m4d_to_theia.txt"), T, fmt="%.9f",
               header="4x4 homogeneous: p_theia = T @ [p_m4d; 1]  "
                      "(raw FBX file coordinates, both sides)")
    np.save(os.path.join(outdir, "T_m4d_to_theia.npy"), T)
    with open(os.path.join(outdir, "alignment_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return outdir


# %%==========================================================================
#  13. FIGURE
# ============================================================================

BONES = [("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_toe"),
         ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe"),
         ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
         ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
         ("l_hip", "r_hip"), ("l_shoulder", "r_shoulder"),
         ("pelvis", "thorax"), ("thorax", "head")]

C_THEIA, C_M4D = "#1f4e79", "#c0392b"


def make_figure(report, context, path=None):
    """Overlay + residual bars. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    th, m4 = context["trials"][0]
    R, t, s = context["R"], context["t"], context["scale"]
    pt = th.mean_pose()
    pm = {k: s * (R @ v) + t for k, v in m4.mean_pose().items()}
    fitted = set(active_translation_landmarks())

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.30, wspace=0.28)

    def draw(ax, pos, colour, label, i, j, style):
        for a, b in BONES:
            if a in pos and b in pos:
                ax.plot([pos[a][i], pos[b][i]], [pos[a][j], pos[b][j]],
                        style, color=colour, lw=1.6, zorder=2)
        for keys, kw in ((fitted, dict(s=30, color=colour)),
                         (set(pos) - fitted, dict(s=30, facecolors="none",
                                                  edgecolors=colour, lw=1.2))):
            xs = [pos[k][i] for k in pos if k in keys]
            ys = [pos[k][j] for k in pos if k in keys]
            ax.scatter(xs, ys, zorder=3, **kw)
        ax.plot([], [], style, color=colour, lw=1.6, label=label)

    for col, (i, j, name) in enumerate([(0, 2, "front  (X-Z)"),
                                        (1, 2, "side  (Y-Z)"),
                                        (0, 1, "top  (X-Y)")]):
        ax = fig.add_subplot(gs[0, col])
        draw(ax, pt, C_THEIA, "Theia3D", i, j, "-")
        draw(ax, pm, C_M4D, "MOVE4D transformed", i, j, "--")
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    ax = fig.add_subplot(gs[1, 0:2])
    per = dict(report["position_residual"]["per_joint"])
    per.update({k: dict(mean_mm=v["mean_mm"])
                for k, v in report["validation_landmarks"].items()})
    keys = sorted(per, key=lambda k: -per[k]["mean_mm"])
    ax.bar(range(len(keys)), [per[k]["mean_mm"] for k in keys],
           color=[C_THEIA if k in fitted else "#aaaaaa" for k in keys])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("position residual (mm)", fontsize=9)
    ax.set_title("Per-landmark position residual   "
                 "(solid = fitted, grey = NOT fitted, definition gap)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    ax = fig.add_subplot(gs[1, 2])
    pd = report["angular_residual_deg"]["per_direction"]
    ks = sorted(pd, key=lambda k: pd[k]["mean_deg"])
    ax.barh(ks, [pd[k]["mean_deg"] for k in ks],
            color=["#8e44ad" if pd[k]["kind"] == "ML" else C_M4D for k in ks])
    ax.set_xlabel("angular residual (deg)", fontsize=9)
    ax.set_title("Direction residual  (purple = ML, red = long axis)",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    fig.suptitle(f"MOVE4D -> Theia3D alignment check "
                 f"[{report['rotation_model']} model, arms + legs only]",
                 fontsize=12, y=0.97)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


# %%==========================================================================
#  14. RUN
# ============================================================================

def main(argv=None):
    global TRIAL_PAIRS, ROTATION_MODEL
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) >= 2:
        TRIAL_PAIRS = [(argv[0], argv[1])]
    if "--full" in argv:
        ROTATION_MODEL = "full"

    report, T, context = solve_alignment()
    if SAVE_OUTPUTS:
        out = save_outputs(report, T)
        print(f"\n  wrote {out}/T_m4d_to_theia.{{txt,npy}} and "
              f"alignment_report.json")
    if MAKE_PLOT:
        try:
            context["figure"] = make_figure(
                report, context,
                os.path.join(OUTDIR, "alignment_check.png")
                if SAVE_OUTPUTS else None)
            if SAVE_OUTPUTS:
                print(f"  wrote {OUTDIR}/alignment_check.png")
        except ImportError:
            print("  (matplotlib not available -- skipping the figure)")
    return report, T, context


if __name__ == "__main__":
    RESULT, T, CONTEXT = main()

    # --- ready to use -------------------------------------------------
    # pts_theia = apply_transform(T, pts_m4d)          # (..., 3)
    # R_theia   = transform_orientation(T, R_m4d)      # (..., 3, 3)
