"""Evaluate an FBX scene graph into per-frame global joint transforms.

Implements the full Autodesk transform chain, the FBX rotation-order
convention, and linear resampling of animation curves onto an arbitrary
time grid.
"""
import numpy as np
import fbx_raw

FBX_TIME_UNIT = 46186158000.0  # FBX ktime units per second

# FBX EulerOrder enum -> order in which axis rotations are APPLIED.
# Matrix for column vectors is the reverse product, e.g. XYZ -> Rz@Ry@Rx.
_EULER_ORDER = {
    0: "XYZ", 1: "XZY", 2: "YZX", 3: "YXZ", 4: "ZXY", 5: "ZYX", 6: "XYZ",
}

# FBX TimeMode enum -> nominal fps (14 = custom, read CustomFrameRate)
_TIME_MODE_FPS = {
    0: None, 1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0, 6: 30.0,
    7: 30.0, 8: 29.97, 9: 29.97, 10: 25.0, 11: 24.0, 12: 1000.0,
    13: 23.976, 14: None, 15: 96.0, 16: 72.0, 17: 59.94,
}


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_AXIS_FN = {"X": _rx, "Y": _ry, "Z": _rz}


def euler_to_matrix(xyz_deg, order_enum=0):
    """Euler angles (degrees, XYZ-labelled) -> 3x3 rotation matrix."""
    order = _EULER_ORDER.get(order_enum, "XYZ")
    ang = {"X": np.radians(xyz_deg[0]),
           "Y": np.radians(xyz_deg[1]),
           "Z": np.radians(xyz_deg[2])}
    R = np.eye(3)
    for axis in order:                 # applied first .. applied last
        R = _AXIS_FN[axis](ang[axis]) @ R
    return R


def _prop_value(p70, name, default):
    """Read a Properties70 entry as a float vector / scalar."""
    if p70 is None:
        return np.array(default, dtype=float)
    for p in p70.children:
        if p.props[0] == name:
            vals = p.props[4:]
            if len(vals) == 1:
                return float(vals[0])
            return np.array([float(v) for v in vals[:len(default)]], dtype=float)
    return np.array(default, dtype=float) if isinstance(default, (list, tuple)) else default


class Curve:
    """A single animation curve (one scalar channel)."""

    def __init__(self, times, values, default=0.0):
        self.t = np.asarray(times, dtype=float) / FBX_TIME_UNIT
        self.v = np.asarray(values, dtype=float)
        self.default = default

    def sample(self, t):
        if len(self.t) == 0:
            return np.full_like(np.atleast_1d(t), self.default, dtype=float)
        if len(self.t) == 1:
            return np.full_like(np.atleast_1d(t), self.v[0], dtype=float)
        return np.interp(t, self.t, self.v)  # clamped at the ends


class Joint:
    def __init__(self, uid, name, node_type):
        self.uid = uid
        self.name = name
        self.type = node_type
        self.parent = None
        self.children = []
        # static transform components
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
        # animated channels: {'Lcl Translation': {'X': Curve, ...}, ...}
        self.curves = {}

    def local_matrix(self, t):
        """Local transform at time t as a 4x4 matrix."""
        tr = self._chan("Lcl Translation", self.lcl_t, t)
        ro = self._chan("Lcl Rotation", self.lcl_r, t)
        sc = self._chan("Lcl Scaling", self.lcl_s, t)

        R = euler_to_matrix(ro, self.rot_order)
        Rpre = euler_to_matrix(self.pre_rot, 0)
        Rpost = euler_to_matrix(self.post_rot, 0)
        Rfull = Rpre @ R @ Rpost.T

        S = np.diag(sc)

        # World = T . Roff . Rp . Rfull . Rp^-1 . Soff . Sp . S . Sp^-1
        M = np.eye(4)
        M[:3, 3] = tr + self.rot_off + self.rot_piv
        A = np.eye(4)
        A[:3, :3] = Rfull
        A[:3, 3] = -Rfull @ self.rot_piv + self.sca_off + self.sca_piv
        B = np.eye(4)
        B[:3, :3] = S
        B[:3, 3] = -S @ self.sca_piv
        return M @ A @ B

    def _chan(self, prop, static, t):
        c = self.curves.get(prop)
        if not c:
            return np.asarray(static, dtype=float)
        out = np.asarray(static, dtype=float).copy()
        for i, ax in enumerate("XYZ"):
            if ax in c:
                out[i] = float(c[ax].sample(t))
        return out


class Scene:
    def __init__(self, path):
        self.path = path
        self.root_node, self.version = fbx_raw.load(path)
        self.joints = {}
        self._parse_objects()
        self._parse_connections()
        self._read_globals()
        self._time_range()

    # ------------------------------------------------------------------
    def _parse_objects(self):
        objs = self.root_node.find("Objects")
        self.anim_curves = {}
        self.curve_nodes = {}

        for m in objs.find_all("Model"):
            uid = m.props[0]
            name = m.props[1].split("\x00")[0]
            j = Joint(uid, name, m.props[2])
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
            ro = _prop_value(p70, "RotationOrder", 0)
            j.rot_order = int(ro) if np.isscalar(ro) else int(ro[0])
            self.joints[uid] = j

        for c in objs.find_all("AnimationCurve"):
            kt = c.find("KeyTime")
            kv = c.find("KeyValueFloat")
            if kt is None or kv is None:
                continue
            self.anim_curves[c.props[0]] = Curve(kt.props[0], kv.props[0])

        for cn in objs.find_all("AnimationCurveNode"):
            p70 = cn.find("Properties70")
            defaults = {}
            if p70:
                for p in p70.children:
                    nm = p.props[0]
                    if nm.startswith("d|"):
                        defaults[nm[2:]] = float(p.props[4])
            self.curve_nodes[cn.props[0]] = {"defaults": defaults, "channels": {}}

    # ------------------------------------------------------------------
    def _parse_connections(self):
        conns = self.root_node.find("Connections")
        for c in conns.children:
            kind = c.props[0]
            src, dst = c.props[1], c.props[2]

            if kind == "OO":
                if src in self.joints and dst in self.joints:
                    self.joints[src].parent = self.joints[dst]
                    self.joints[dst].children.append(self.joints[src])

            elif kind == "OP":
                prop = c.props[3]
                # curve -> curve-node  (prop is 'd|X' etc.)
                if src in self.anim_curves and dst in self.curve_nodes:
                    if prop.startswith("d|"):
                        self.curve_nodes[dst]["channels"][prop[2:]] = self.anim_curves[src]
                # curve-node -> model  (prop is 'Lcl Translation' etc.)
                elif src in self.curve_nodes and dst in self.joints:
                    self.joints[dst].curves[prop] = self.curve_nodes[src]["channels"]

    # ------------------------------------------------------------------
    def _read_globals(self):
        gs = self.root_node.find("GlobalSettings")
        p70 = gs.find("Properties70") if gs else None
        g = {}
        for nm, dflt in [("UpAxis", 1), ("UpAxisSign", 1), ("FrontAxis", 2),
                         ("FrontAxisSign", 1), ("CoordAxis", 0),
                         ("CoordAxisSign", 1), ("UnitScaleFactor", 1.0),
                         ("TimeMode", 0), ("CustomFrameRate", -1.0)]:
            v = _prop_value(p70, nm, dflt)
            g[nm] = float(v) if np.isscalar(v) else float(v[0])
        self.globals = g

        fps = _TIME_MODE_FPS.get(int(g["TimeMode"]))
        if fps is None:
            fps = g["CustomFrameRate"] if g["CustomFrameRate"] > 0 else None
        self.fps = fps
        self.unit_cm = g["UnitScaleFactor"]  # centimetres per file unit

    def _time_range(self):
        ts = [c.t for c in self.anim_curves.values() if len(c.t)]
        if ts:
            self.t0 = min(t[0] for t in ts)
            self.t1 = max(t[-1] for t in ts)
            uniq = np.unique(np.concatenate(ts))
            self.key_times = uniq
        else:
            self.t0 = self.t1 = 0.0
            self.key_times = np.array([0.0])

    # ------------------------------------------------------------------
    def limb_joints(self):
        return {j.name: j for j in self.joints.values()
                if j.type in ("LimbNode", "Root")}

    def global_transforms(self, times):
        """Evaluate global 4x4 transforms for every joint at each time.

        Returns dict name -> (T,4,4) array.
        """
        times = np.atleast_1d(np.asarray(times, dtype=float))
        out = {}

        order = []
        seen = set()

        def visit(j):
            if j.uid in seen:
                return
            seen.add(j.uid)
            if j.parent is not None:
                visit(j.parent)
            order.append(j)

        for j in self.joints.values():
            visit(j)

        for ti, t in enumerate(times):
            for j in order:
                L = j.local_matrix(t)
                G = L if j.parent is None else out[j.parent.name][ti] @ L
                if j.name not in out:
                    out[j.name] = np.zeros((len(times), 4, 4))
                out[j.name][ti] = G
        return out
