# -*- coding: utf-8 -*-
"""
=============================================================================
 GAIT EVENTS ON THE SPLIT-BELT TREADMILL  --  GRF first, kinematics second
=============================================================================

Replaces 01-04. Set the paths in CONFIG and press F5, or

    python gait_events.py                    # every trial in FORCE_FOLDER
    python gait_events.py STEM [STEM ...]    # just these trials

1. BELT CONTACTS. Each belt's vertical force, with its own drifting baseline
   removed, above FORCE_THRESHOLD_N. Dips shorter than MERGE_GAP_S are joined
   and blips shorter than MIN_CONTACT_S dropped.

2. WHERE THE FEET ARE. The MOVE4D boot meshes are posed on the Theia feet
   (foot_mesh_binding.npz) and reduced to their soles. A sole vertex within
   CONTACT_HEIGHT_MM of the belt surface is "in contact". The force-plate
   frame is registered to Theia's by matching the centre of pressure to the
   one foot in contact during single support, unless FP_TO_THEIA is given.

3. TRUST. A GRF heel strike or toe-off is trusted only if, over
   +/-TRUST_WINDOW_S around it, the belt was carrying one foot, all of that
   foot's contact patch was on the belt with BELT_MARGIN_MM to spare, the
   other foot was nowhere near the belt, and the centre of pressure was under
   the foot. The limb comes from the mesh, not from the belt's name, so a
   clean crossover step is kept and labelled correctly.

4. FALLBACK. Every variable in fallback_detectors.py is calibrated on the
   trusted events of the same limb in the same trial and scored on the
   trusted events it was not calibrated on. FALLBACK_ORDER picks which ones
   fill the gaps.

5. SEQUENCE. Walking goes L HS -> R TO -> R HS -> L TO -> L HS. Between two
   trusted events the sequence says exactly which events are missing and
   roughly when; each one is searched for only in that window -- first
   among GRF events that could not be checked for want of tracking, then
   with the fallback, and only then interpolated. Before the
   first and after the last trusted event the sequence is extended for as
   long as the fallback keeps finding events where they should be.

Per trial, in OUTPUT_FOLDER:

    {trial}_merged_events.csv       event_type, support_limb, frame_100hz,
                                    source -- the same four columns as before.
                                    source is GRF (trusted), GRF_unverified
                                    (a clean-looking belt contact the mesh
                                    could not check because tracking dropped
                                    out), kinematic or interpolated
    {trial}_event_qa.csv            the same events plus how each was obtained
                                    and stance / stride outlier flags
    {trial}_grf_contacts.csv        every belt contact, its label and why each
                                    of its events was or was not trusted
    {trial}_fallback_benchmark.csv  every fallback variable scored
    {trial}_event_registration.json the FP -> Theia registration and floor

and fallback_benchmark_all_trials.csv across everything processed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import apply_binding as ab            # noqa: E402  (orthonormalise, kabsch)
import fallback_detectors as fd       # noqa: E402


# %%==========================================================================
#  CONFIG
# ============================================================================

DATA_FOLDER = Path(
    r"C:\Users\lcour081\Box\Military Project\Data\Analysis\DICE_Treadmill"
)
FORCE_FOLDER = DATA_FOLDER / "FP_renamed"
KINEMATIC_FOLDER = DATA_FOLDER / "Theia_csv_outputs"
OUTPUT_FOLDER = DATA_FOLDER / "gait_event_outputs"
BINDING_FILE = HERE / "foot_mesh_binding.npz"
TRIALS = None                  # None = every .csv in FORCE_FOLDER, or stems

GRF_RATE = 1000
KINEMATIC_RATE = 100

# --- belts -----------------------------------------------------------------
BELT_CHANNEL = {"L": "Left belt", "R": "Right belt"}

# The four corners of each belt's plate in the force-plate frame, millimetres,
# any order -- FORCE_PLATFORM:CORNERS in the C3D. None puts the gap halfway
# between the two belts' centre-of-pressure clouds with BELT_GAP_MM between
# them and prints a warning; enter the real corners as soon as you have them.
BELT_CORNERS_MM = None
# BELT_CORNERS_MM = {
#     "L": [(-850, -1000), (-5, -1000), (-5, 1000), (-850, 1000)],
#     "R": [(5, -1000), (850, -1000), (850, 1000), (5, 1000)],
# }
BELT_GAP_MM = 10.0

# Force-plate metres -> Theia metres, as a 2x3 matrix [[a, b, tx], [c, d, ty]]
# on (x, y, 1). None estimates it per trial and prints the estimate in this
# form, so a registration you trust can be pasted back here.
FP_TO_THEIA = None
ALLOW_MIRROR = False           # True only if the two frames differ in handedness

# --- belt contacts ---------------------------------------------------------
FORCE_FILTER_HZ = 50.0
FORCE_THRESHOLD_N = 20.0       # above the belt's own baseline
BASELINE_WINDOW_S = 10.0
MERGE_GAP_S = 0.030
MIN_CONTACT_S = 0.080
# The force itself must look like a foot landing or leaving: from threshold
# to SHAPE_LEVEL_N within MAX_LOADING_S at a heel strike, and back within
# MAX_UNLOADING_S at a toe-off. On D05 the left belt sometimes keeps
# 100-200 N for up to 300 ms after push-off, with the CoP 11 cm lateral of
# the foot; that toe-off time means nothing, whatever the mesh says.
SHAPE_LEVEL_N = 200.0
MAX_LOADING_S = 0.060          # D05 sample: 10-36 ms
MAX_UNLOADING_S = 0.080        # D05 sample: 34-45 ms, tails up to 315 ms

# --- feet ------------------------------------------------------------------
SOLE_CELL_MM = 10.0            # sole = lowest mesh vertex in each cell
SOLE_MAX_RISE_MM = 35.0        # ... up to this far above the lowest (toe spring)
FLOOR_PERCENTILE = 25          # of each foot's lowest sole point = belt surface
CONTACT_HEIGHT_MM = 15.0
MIN_CONTACT_VERTICES = 3

# --- trust -----------------------------------------------------------------
TRUST_WINDOW_S = 0.050
BELT_MARGIN_MM = 10.0
SHARED_MIN_FRAMES = 2
STRADDLE_MIN_FRAMES = 2
USE_COP_CHECK = True
COP_TOLERANCE_MM = 30.0
COP_MIN_FORCE_N = 150.0
COP_MIN_INSIDE = 0.90
MAX_REGISTRATION_ERROR_MM = 40.0

# --- fallback --------------------------------------------------------------
# Which variables fill the gaps, tried in order. "auto" ranks all of
# fallback_detectors.DETECTORS per limb and event by this trial's own
# benchmark (95th-percentile error, among those that found at least
# AUTO_MIN_FOUND_PCT of the held-out events). A fixed list keeps the method
# identical across trials, which is easier to report.
FALLBACK_ORDER = ["mesh_sole_height", "heel_toe_ap_velocity"]
AUTO_MIN_FOUND_PCT = 95.0
KINEMATIC_FILTER_HZ = 10.0
MAX_GAP_FILL_FRAMES = 10       # tracking gaps bridged before filtering
CALIBRATE_PER_LIMB = True
MIN_CALIBRATION_EVENTS = 10
INTERPOLATE_MISSING = True     # last resort between two trusted events only

# --- output ----------------------------------------------------------------
TRIM_TO_COMPLETE_STANCES = True   # each limb starts on HS and ends on TO
FLAG_RATIO = (0.75, 1.25)         # stance / stride outside this x median
WRITE_BENCHMARK = True


LIMBS = ("L", "R")
OTHER = {"L": "R", "R": "L"}
HS, TO = "heel_strike", "toe_off"
# the order events come in during walking, as (limb, event)
SEQUENCE = [("L", HS), ("R", TO), ("R", HS), ("L", TO)]
POSITION = {key: i for i, key in enumerate(SEQUENCE)}
STEP = GRF_RATE // KINEMATIC_RATE
MIN_SEPARATION = 1.0           # frames between consecutive events


# %%==========================================================================
#  LOADING
# ============================================================================

def find_file(folder, stem, suffix=""):
    """The export spells the trial with a space or an underscore; try both."""
    for name in (stem, stem.replace(" ", "_"), stem.replace("_", " ")):
        path = Path(folder) / f"{name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"no {stem}{suffix} in {folder}")


def load_forces(path):
    """-> {limb: {"fz": (n,), "cop": (n, 2) mm}} for each belt."""
    with open(path, encoding="utf-8-sig") as fh:
        header = next(csv.reader(fh))
    wanted = {(limb, q): header.index(f"{belt}_{q}")
              for limb, belt in BELT_CHANNEL.items()
              for q in ("Force_Z", "COP_X", "COP_Y")}
    columns = sorted(set(wanted.values()))
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=columns,
                      ndmin=2)
    col = {key: data[:, columns.index(i)] for key, i in wanted.items()}
    return {limb: {"fz": col[(limb, "Force_Z")],
                   "cop": np.column_stack([col[(limb, "COP_X")],
                                           col[(limb, "COP_Y")]])}
            for limb in BELT_CHANNEL}


def load_kinematics(path, binding_meta):
    """Only the columns needed, from a Visual3D metrics export.

    -> frames (ITEM numbers), vectors {name: (F, 3)}, poses {limb: (F, 4, 4)}
    """
    segments = binding_meta["segments"]
    seg_of_limb = {"L": segments.index("left_foot"),
                   "R": segments.index("right_foot")}
    pose_signal = {limb: binding_meta["pose_signal"][segments[k]]
                   for limb, k in seg_of_limb.items()}
    landmarks = {limb: binding_meta["landmark_names"][segments[k]]
                 for limb, k in seg_of_limb.items()}
    points = {"Left_Heel_Position", "Right_Heel_Position",
              "Left_Toes_Position", "Right_Toes_Position", "Pelvis_Position"}
    points |= {n for names in landmarks.values() for n in names}

    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        head = [next(fh).rstrip("\r\n").split("\t") for _ in range(5)]
        where = {(n, c.strip().upper()): i
                 for i, (n, c) in enumerate(zip(head[1], head[4])) if i}
        want = []
        for name in sorted(points):
            want += [(name, a) for a in "XYZ"]
        for sig in pose_signal.values():
            want += [(sig, str(i)) for i in range(16)]
        want = [w for w in want if w in where]
        cols = [where[w] for w in want]
        items, rows = [], []
        for line in fh:
            cells = line.rstrip("\r\n").split("\t")
            if not cells or not cells[0].strip():
                continue
            items.append(cells[0].strip())
            rows.append([float(cells[i]) if i < len(cells) and cells[i].strip()
                         else np.nan for i in cols])
    data = np.array(rows, float).reshape(len(rows), len(cols))
    last = len(data)
    while last and np.isnan(data[last - 1]).all():   # trailing padding
        last -= 1
    data, items = data[:last], items[:last]
    frames = np.array([int(float(i)) for i in items])
    column = {w: data[:, j] for j, w in enumerate(want)}

    vectors = {}
    for name in points:
        if all((name, a) in column for a in "XYZ"):
            vectors[name] = np.column_stack([column[(name, a)] for a in "XYZ"])

    poses = {}
    for limb, sig in pose_signal.items():
        if all((sig, str(i)) in column for i in range(16)):
            P = np.stack([column[(sig, str(i))] for i in range(16)],
                         axis=1).reshape(-1, 4, 4)
            P[:, :3, :3] = ab.orthonormalise(P[:, :3, :3])
            P[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
            bad = ~np.isfinite(P[:, :3, :]).all(axis=(1, 2))
            P[bad] = np.nan
            poses[limb] = P
        elif all(n in vectors for n in landmarks[limb]):
            poses[limb] = None          # fitted from landmarks by the caller
        else:
            raise KeyError(f"{path.name}: neither {sig} nor the landmarks "
                           f"{landmarks[limb]} are in the export")
    return frames, vectors, poses, seg_of_limb, landmarks


def load_binding(path):
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["meta"])), z


# %%==========================================================================
#  SIGNAL HELPERS
# ============================================================================

def lowpass(x, cutoff, rate, max_gap=MAX_GAP_FILL_FRAMES):
    """Zero-lag Butterworth that survives NaN.

    Gaps up to max_gap frames are bridged linearly first; anything longer
    stays NaN and each finite stretch is filtered on its own, so one dropout
    no longer wipes out the whole channel (sosfiltfilt spreads a single NaN
    over everything).
    """
    x = np.array(x, float)
    flat = x.reshape(len(x), -1)
    sos = butter(4, cutoff, btype="lowpass", fs=rate, output="sos")
    n = np.arange(len(x))
    for j in range(flat.shape[1]):
        y = flat[:, j]
        ok = np.isfinite(y)
        if ok.sum() < 2:
            continue
        bridged = np.interp(n, n[ok], y[ok])
        missing = ~ok
        edges = np.diff(np.r_[0, missing.astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s > max_gap or s == 0 or e == len(y):
                bridged[s:e] = np.nan
        good = np.isfinite(bridged)
        edges = np.diff(np.r_[0, good.astype(np.int8), 0])
        out = np.full(len(y), np.nan)
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s > 30:
                out[s:e] = sosfiltfilt(sos, bridged[s:e])
            else:
                out[s:e] = bridged[s:e]
        flat[:, j] = out
    return flat.reshape(x.shape)


def belt_baseline(fz):
    """The belt's unloaded reading, per BASELINE_WINDOW_S, and its noise.

    A window in which the belt is never unloaded -- standing on it before
    the belt starts, say -- has no baseline in it at all; its reading is the
    body weight. Those windows are recognised by sitting far above the rest
    and take their baseline from their neighbours instead.
    """
    win = int(BASELINE_WINDOW_S * GRF_RATE)
    centres, levels, quiet = [], [], []
    for s in range(0, len(fz), win):
        seg = fz[s:s + win]
        low = seg[seg <= np.percentile(seg, 20)]
        centres.append(s + len(seg) / 2)
        levels.append(np.median(low))
        quiet.append(low - np.median(low))
    centres, levels = np.array(centres), np.array(levels)
    good = np.abs(levels - np.median(levels)) < FORCE_THRESHOLD_N
    base = np.interp(np.arange(len(fz)), centres[good], levels[good])
    q = np.concatenate([x for x, g in zip(quiet, good) if g])
    noise = 1.4826 * np.median(np.abs(q - np.median(q)))
    return base, noise, float(np.ptp(levels[good]))


def detect_belt_contacts(fz, belt):
    """-> list of contacts on one belt, and the baselined, filtered force."""
    base, noise, drift = belt_baseline(fz)
    sos = butter(4, FORCE_FILTER_HZ, btype="lowpass", fs=GRF_RATE,
                 output="sos")
    force = sosfiltfilt(sos, fz) - base
    above = force > FORCE_THRESHOLD_N
    edges = np.diff(np.r_[0, above.astype(np.int8), 0])
    merged = []
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if merged and s - merged[-1][1] < MERGE_GAP_S * GRF_RATE:
            merged[-1][1] = e
            merged[-1][2] += 1
        else:
            merged.append([s, e, 0])
    contacts = []
    for s, e, m in merged:
        if e - s < MIN_CONTACT_S * GRF_RATE:
            continue
        high = np.flatnonzero(force[s:e] > SHAPE_LEVEL_N)
        contacts.append(dict(
            belt=belt, start=int(s), stop=int(e), dips_merged=m,
            complete_start=s > 0, complete_stop=e < len(fz),
            peak_n=float(force[s:e].max()),
            loading_s=(high[0] if len(high) else e - s) / GRF_RATE,
            unloading_s=(e - s - high[-1] if len(high) else e - s) / GRF_RATE))
    info = dict(baseline_n=float(np.median(base)), noise_n=float(noise),
                baseline_drift_n=drift)
    return contacts, force, info


# %%==========================================================================
#  FEET: SOLE GEOMETRY, CONTACT AND BELT MEMBERSHIP
# ============================================================================

def sole_model(v_local, pose):
    """The underside of the boot, in the foot's own frame.

    Neither "up" nor "forward" is assumed from a naming convention: up is
    world vertical seen from the foot, median over the trial, and forward is
    the mesh's long axis, pointing away from the ankle (the segment origin).
    """
    R = pose[:, :3, :3]
    ok = np.isfinite(R).all(axis=(1, 2))
    up = np.median(R[ok][:, 2, :], axis=0)
    up /= np.linalg.norm(up)
    height = v_local @ up
    flat = v_local - np.outer(height, up)
    _, _, vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
    long_axis = vt[0] - (vt[0] @ up) * up
    long_axis /= np.linalg.norm(long_axis)
    along = v_local @ long_axis
    if abs(along.max()) < abs(along.min()):
        long_axis, along = -long_axis, -along
    across = v_local @ np.cross(up, long_axis)

    cell = SOLE_CELL_MM / 1000
    key = (np.floor(along / cell).astype(np.int64) * 100003
           + np.floor(across / cell).astype(np.int64))
    order = np.lexsort((height, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    idx = order[first]
    idx = idx[height[idx] < height[idx].min() + SOLE_MAX_RISE_MM / 1000]
    a = along[idx]
    rear = a < a.min() + 0.4 * (a.max() - a.min())
    return dict(vertices=v_local[idx], rear=rear, long_axis=long_axis)


def sole_chunks(pose, sole, chunk=4000):
    """Yield (slice, (n, V, 3)) world positions of the sole vertices."""
    for s in range(0, len(pose), chunk):
        P = pose[s:s + chunk]
        world = sole @ P[:, :3, :3].transpose(0, 2, 1) + P[:, None, :3, 3]
        yield slice(s, s + len(P)), world


def polygon_ccw(corners):
    c = np.asarray(corners, float)[:, :2]
    centre = c.mean(0)
    return c[np.argsort(np.arctan2(c[:, 1] - centre[1], c[:, 0] - centre[0]))]


def inside_distance(poly, xy):
    """Signed distance inside a convex CCW polygon (negative outside)."""
    d = np.full(xy.shape[:-1], np.inf)
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        e = q - p
        normal = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        d = np.minimum(d, (xy - p) @ normal)
    return d


def apply_2d(A, xy):
    return xy @ A[:, :2].T + A[:, 2]


class Feet:
    """Per-frame contact state of both boots against both belts."""

    def __init__(self, poses, binding, seg_of_limb):
        v_local, seg = binding["vertices_local"], binding["vertex_segment"]
        self.poses = poses
        self.n = len(next(iter(poses.values())))
        self.valid = {f: np.isfinite(poses[f][:, :3, :]).all(axis=(1, 2))
                      for f in LIMBS}
        self.model = {f: sole_model(v_local[seg == seg_of_limb[f]],
                                    poses[f]) for f in LIMBS}
        self.h = CONTACT_HEIGHT_MM / 1000

        # pass 1: how low each sole gets, and where the belt surface is
        self.min_z, self.rear_z, self.fore_z = {}, {}, {}
        for f in LIMBS:
            sole, rear = self.model[f]["vertices"], self.model[f]["rear"]
            mz, rz, fz = (np.full(self.n, np.nan) for _ in range(3))
            for sl, W in sole_chunks(poses[f], sole):
                z = W[..., 2]
                mz[sl], rz[sl], fz[sl] = (z.min(1), z[:, rear].min(1),
                                          z[:, ~rear].min(1))
            self.min_z[f], self.rear_z[f], self.fore_z[f] = mz, rz, fz
        self.floor = {f: float(np.nanpercentile(self.min_z[f],
                                                FLOOR_PERCENTILE))
                      for f in LIMBS}
        self.height = {f: self.min_z[f] - self.floor[f] for f in LIMBS}

        # forward = the feet's long axis, horizontal, median over the trial
        fwd = np.zeros(3)
        for f in LIMBS:
            R = poses[f][self.valid[f], :3, :3]
            d = R @ self.model[f]["long_axis"]
            d[:, 2] = 0
            fwd += np.median(d / np.linalg.norm(d, axis=1, keepdims=True), 0)
        self.forward = fwd / np.linalg.norm(fwd)

        # pass 2: the contact patch, for registration
        self.n_contact, self.centroid = {}, {}
        for f in LIMBS:
            nc = np.zeros(self.n, int)
            cen = np.full((self.n, 2), np.nan)
            for sl, W in sole_chunks(poses[f], self.model[f]["vertices"]):
                touch = W[..., 2] < self.floor[f] + self.h
                nc[sl] = touch.sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cen[sl] = ((W[..., :2] * touch[..., None]).sum(1)
                               / touch.sum(1)[:, None])
            self.n_contact[f], self.centroid[f] = nc, cen

    def belt_membership(self, belts, cop_theia, margin):
        """Pass 3: per foot, per belt, how many contact vertices touch it
        (within margin), how many are inside it with margin to spare, and how
        far the belt's centre of pressure is from the foot's contact patch."""
        self.n_touch = {f: {} for f in LIMBS}
        self.n_strict = {f: {} for f in LIMBS}
        self.cop_dist = {f: {} for f in LIMBS}
        for f in LIMBS:
            for b in belts:
                self.n_touch[f][b] = np.zeros(self.n, int)
                self.n_strict[f][b] = np.zeros(self.n, int)
                self.cop_dist[f][b] = np.full(self.n, np.inf)
            for sl, W in sole_chunks(self.poses[f], self.model[f]["vertices"]):
                touch = W[..., 2] < self.floor[f] + self.h
                xy = W[..., :2]
                for b, poly in belts.items():
                    d = inside_distance(poly, xy)
                    self.n_touch[f][b][sl] = (touch & (d >= -margin)).sum(1)
                    self.n_strict[f][b][sl] = (touch & (d >= margin)).sum(1)
                    c = cop_theia[b][sl]
                    gap = np.linalg.norm(xy - c[:, None, :], axis=2)
                    gap = np.where(touch, gap, np.inf).min(1)
                    self.cop_dist[f][b][sl] = np.where(np.isfinite(c[:, 0]),
                                                       gap, np.inf)


# %%==========================================================================
#  REGISTRATION AND BELTS
# ============================================================================

def estimate_registration(force100, cop100, feet, belts_fp):
    """Force-plate frame -> Theia frame, from single support.

    The rotation comes from the belt's direction of travel. In the plate
    frame that is the belts' long axis, known from their corners; in Theia
    it is the velocity of the stance foot, which rides the belt. Neither
    involves where the CoP sits under the foot -- at the heel early and the
    toe late, while the contact patch is wherever the sole is low -- and that
    phase-dependent offset would otherwise leak straight into the angle. The
    CoP's own velocity gives a second, rougher angle, reported as a check.

    The translation is the CoP-to-patch offset, taken per foot and then
    averaged over the two feet, so a medial/lateral bias that mirrors
    between the feet cancels. The side-to-side residual is the one that
    matters for the belt gap, and it is reported separately.
    """
    P, Q, V_fp, V_th, foot = [], [], [], [], []
    for b in LIMBS:
        loaded = force100[b] > COP_MIN_FORCE_N
        free = force100[OTHER[b]] < FORCE_THRESHOLD_N
        v_cop = np.gradient(cop100[b], axis=0) / 1000
        for f in LIMBS:
            alone = ((feet.n_contact[f] >= MIN_CONTACT_VERTICES)
                     & (feet.height[OTHER[f]] > 2 * feet.h)
                     & feet.valid[f] & feet.valid[OTHER[f]])
            m = loaded & free & alone & np.isfinite(cop100[b][:, 0])
            # velocities only where the neighbours are single support too
            inner = m & np.r_[False, m[:-1]] & np.r_[m[1:], False]
            v_foot = np.gradient(feet.poses[f][:, :2, 3], axis=0)
            P.append(cop100[b][m] / 1000)
            Q.append(feet.centroid[f][m])
            V_fp.append(v_cop[inner])
            V_th.append(v_foot[inner])
            foot += [f] * int(m.sum())
    P, Q = np.concatenate(P), np.concatenate(Q)
    V_fp, V_th = np.concatenate(V_fp), np.concatenate(V_th)
    foot = np.array(foot)
    if len(P) < 200 or len(V_fp) < 100:
        raise RuntimeError(f"only {len(P)} single-support frames to register "
                           f"the force plates on; set FP_TO_THEIA")

    def unit(v):
        v = np.nansum(v, axis=0)
        return v / np.linalg.norm(v)

    u_cop, u_th = unit(V_fp), unit(V_th)
    axes = []
    for poly in belts_fp.values():
        edges = np.roll(poly, -1, axis=0) - poly
        e = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        axes.append(e / np.linalg.norm(e) * np.sign(e @ u_cop))
    u_fp = unit(np.array(axes))
    cop_check = np.degrees(np.arccos(np.clip(u_fp @ u_cop, -1, 1)))
    n_fp, n_th = np.array([-u_fp[1], u_fp[0]]), np.array([-u_th[1], u_th[0]])
    # which way is left: a proper rotation takes n_fp onto n_th, so the
    # side-to-side positions of CoP and contact patch must agree in sign
    agree = np.corrcoef(P @ n_fp, Q @ n_th)[0, 1]
    mirror = agree < 0
    M = np.eye(2) - (2 * np.outer(n_fp, n_fp) if mirror else 0)
    if mirror and not ALLOW_MIRROR:
        raise RuntimeError("the force-plate frame looks mirrored relative to "
                           "Theia's; check it, then set ALLOW_MIRROR = True")
    phi = np.arctan2(u_th[1], u_th[0]) - np.arctan2(u_fp[1], u_fp[0])
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]) @ M
    offsets = Q - P @ R.T
    per_foot = [np.median(offsets[foot == f], axis=0) for f in LIMBS
                if (foot == f).sum() >= 50]
    t = (np.mean(per_foot, axis=0) if per_foot
         else np.median(offsets, axis=0))
    A = np.column_stack([R, t])
    res = apply_2d(A, P) - Q
    side = np.abs(res @ n_th) * 1000
    along = np.abs(res @ u_th) * 1000
    return A, dict(pairs=int(len(P)), mirror=bool(mirror),
                   belt_axis_vs_cop_track_deg=float(cop_check),
                   side_agreement=float(agree),
                   median_side_residual_mm=float(np.median(side)),
                   p90_side_residual_mm=float(np.percentile(side, 90)),
                   median_along_residual_mm=float(np.median(along)))


def belt_polygons_fp(forces):
    """Belt corners in FP metres, either given or guessed from the CoP."""
    if BELT_CORNERS_MM is not None:
        return {b: polygon_ccw(np.asarray(BELT_CORNERS_MM[b], float) / 1000)
                for b in LIMBS}, False
    x = {}
    for b in LIMBS:
        m = forces[b]["fz"] > COP_MIN_FORCE_N
        x[b] = forces[b]["cop"][m, 0]
    left_is_low = np.median(x["L"]) < np.median(x["R"])
    lo, hi = ("L", "R") if left_is_low else ("R", "L")
    centre = (np.percentile(x[lo], 99) + np.percentile(x[hi], 1)) / 2 / 1000
    g, w, ap = BELT_GAP_MM / 2000, 1.0, 2.0
    boxes = {lo: [(centre - w, -ap), (centre - g, -ap), (centre - g, ap),
                  (centre - w, ap)],
             hi: [(centre + g, -ap), (centre + w, -ap), (centre + w, ap),
                  (centre + g, ap)]}
    return {b: polygon_ccw(boxes[b]) for b in LIMBS}, True


# %%==========================================================================
#  TRUST
# ============================================================================

def label_contacts(contacts, feet, force100):
    """Label every belt contact and decide which of its events to trust."""
    w = int(round(TRUST_WINDOW_S * KINEMATIC_RATE))
    n = feet.n
    events, unverified = [], []
    for c in contacts:
        b = c["belt"]
        c["hs_frame"], c["to_frame"] = c["start"] / STEP, c["stop"] / STEP
        r0 = int(round(c["hs_frame"]))
        r1 = min(int(round(c["to_frame"])), n - 1)
        rows = np.arange(max(r0, 0), r1 + 1)
        c.update(foot="", label="", hs_trusted=False, to_trusted=False,
                 hs_reason="", to_reason="", cop_inside=np.nan)
        if r0 >= n or not len(rows):
            c["label"] = c["hs_reason"] = c["to_reason"] = "no_kinematics"
            continue
        valid = feet.valid["L"][rows] & feet.valid["R"][rows]
        if valid.mean() < 0.5:
            c["label"] = c["hs_reason"] = c["to_reason"] = "no_mesh"
            c["foot"] = b
            for kind, t, ok in ((HS, c["hs_frame"], c["complete_start"]
                                 and c["loading_s"] <= MAX_LOADING_S),
                                (TO, c["to_frame"], c["complete_stop"]
                                 and c["unloading_s"] <= MAX_UNLOADING_S)):
                if ok:
                    unverified.append(_grf_event(c, b, kind, t, "no_mesh"))
            continue
        touching = {f: feet.n_touch[f][b][rows] >= MIN_CONTACT_VERTICES
                    for f in LIMBS}
        f = max(LIMBS, key=lambda k: touching[k].sum())
        g = OTHER[f]
        if not touching[f].any():
            c["label"] = c["hs_reason"] = c["to_reason"] = "no_foot_on_belt"
            continue
        in_contact = feet.n_contact[f][rows] >= MIN_CONTACT_VERTICES
        off_belt = in_contact & (feet.n_strict[f][b][rows]
                                 < feet.n_contact[f][rows])
        if touching[g].sum() >= SHARED_MIN_FRAMES:
            label = "shared"
        elif off_belt.sum() >= STRADDLE_MIN_FRAMES:
            label = "straddle"
        elif f != b:
            label = "crossover"
        else:
            label = "clean"
        alone = (force100[b][rows] > COP_MIN_FORCE_N) & ~touching[g]
        if alone.any():
            c["cop_inside"] = float(np.mean(
                feet.cop_dist[f][b][rows][alone] < COP_TOLERANCE_MM / 1000))
        c["foot"], c["label"] = f, label

        for kind, centre, complete, inner in (
                (HS, r0, c["complete_start"], (0, w)),
                (TO, r1, c["complete_stop"], (-w, 0))):
            win = np.arange(max(centre - w, 0), min(centre + w, n - 1) + 1)
            near = np.arange(max(centre + inner[0], 0),
                             min(centre + inner[1], n - 1) + 1)
            fc = feet.n_contact[f][win] >= MIN_CONTACT_VERTICES
            if not complete:
                reason = "incomplete"
            elif kind == HS and c["loading_s"] > MAX_LOADING_S:
                reason = "slow_loading"
            elif kind == TO and c["unloading_s"] > MAX_UNLOADING_S:
                reason = "slow_unloading"
            elif not (feet.valid[f][win] & feet.valid[g][win]).all():
                reason = "no_mesh"
            elif (feet.n_touch[g][b][win] >= MIN_CONTACT_VERTICES).any():
                reason = "other_foot_on_belt"
            elif not (feet.n_contact[f][near] >= MIN_CONTACT_VERTICES).any():
                reason = "foot_not_in_contact"
            elif (fc & (feet.n_strict[f][b][win]
                        < feet.n_contact[f][win])).any():
                reason = "foot_over_belt_edge"
            elif (USE_COP_CHECK and np.isfinite(c["cop_inside"])
                  and c["cop_inside"] < COP_MIN_INSIDE):
                reason = "cop_outside_foot"
            else:
                reason = "trusted"
            key = "hs" if kind == HS else "to"
            c[f"{key}_reason"] = reason
            c[f"{key}_trusted"] = reason == "trusted"
            t = c["hs_frame"] if kind == HS else c["to_frame"]
            if reason == "trusted":
                events.append(_grf_event(c, f, kind, t, label))
            elif reason == "no_mesh" and label in ("clean", "crossover"):
                # tracking dropped out right at the event: nothing says the
                # event is wrong, only that it cannot be checked
                unverified.append(_grf_event(c, f, kind, t, "no_mesh"))
    events.sort(key=lambda e: e["t"])
    for e in unverified:
        e["source"] = e["method"] = "GRF_unverified"

    # two trusted events of one kind on one foot a few frames apart cannot
    # both be right; keep the first and say so
    kept = []
    for e in events:
        recent = [k for k in kept[-8:] if e["t"] - k["t"] < 10]
        clash = [k for k in recent if k["limb"] == e["limb"]
                 and k["event"] == e["event"]]
        if clash:
            print(f"    ! duplicate trusted {e['limb']} {e['event']} at "
                  f"frame {e['t']:.1f}, dropped")
            continue
        kept.append(e)
    return kept, unverified


def _grf_event(c, limb, kind, t, label):
    return dict(t=t, limb=limb, event=kind, source="GRF", method="GRF",
                grf_sample=c["start"] if kind == HS else c["stop"],
                belt=c["belt"], contact_label=label)


# %%==========================================================================
#  FALLBACK: CALIBRATION, SEARCH AND BENCHMARK
# ============================================================================

class Kinematics:
    """What fallback_detectors.py sees. Everything filtered, NaN kept."""

    def __init__(self, vectors, feet, rate):
        self.rate = rate
        self.forward = feet.forward
        f = lambda x: lowpass(x, KINEMATIC_FILTER_HZ, rate)   # noqa: E731
        side = {"L": "Left", "R": "Right"}
        self.heel = {k: f(vectors[f"{side[k]}_Heel_Position"]) for k in LIMBS}
        self.toe = {k: f(vectors[f"{side[k]}_Toes_Position"]) for k in LIMBS}
        self.pelvis = f(vectors["Pelvis_Position"])
        self.sole_height = {k: f(feet.height[k]) for k in LIMBS}
        self.rear_height = {k: f(feet.rear_z[k] - feet.floor[k])
                            for k in LIMBS}
        self.fore_height = {k: f(feet.fore_z[k] - feet.floor[k])
                            for k in LIMBS}

    def ap(self, x):
        return x @ self.forward

    def vertical(self, x):
        return x[..., 2]

    def velocity(self, x):
        return np.gradient(x, 1 / self.rate, axis=0)


def expected_intervals(anchors):
    """Median time from each sequence position to the next, in frames."""
    gaps = {p: [] for p in range(4)}
    for a, b in zip(anchors[:-1], anchors[1:]):
        if (b["pos"] - a["pos"]) % 4 == 1:
            gaps[a["pos"]].append(b["t"] - a["t"])
    if any(len(v) < 3 for v in gaps.values()):
        raise RuntimeError(
            "too few trusted GRF events to learn the gait timing -- check "
            "the belt corners and the registration")
    med = {p: np.median(v) for p, v in gaps.items()}
    # a whole missing cycle can look adjacent; drop those and redo
    return {p: float(np.median([g for g in v if g < 1.5 * med[p]]))
            for p, v in gaps.items()}


def neighbour_windows(anchors, d):
    """For each trusted event with trusted neighbours on both sides of the
    sequence: the window it must lie in and where it should be."""
    out = {}
    for i in range(1, len(anchors) - 1):
        a, e, b = anchors[i - 1], anchors[i], anchors[i + 1]
        pa = (e["pos"] - 1) % 4
        if a["pos"] != pa or b["pos"] != (e["pos"] + 1) % 4:
            continue
        span, want = b["t"] - a["t"], d[pa] + d[e["pos"]]
        if not 0.6 < span / want < 1.5:
            continue
        p = a["t"] + span * d[pa] / want
        out[i] = (a["t"] + MIN_SEPARATION, b["t"] - MIN_SEPARATION, p)
    return out


def calibrate(spec, truths, windows):
    """Level from the signal at the true events, then the residual bias."""
    kind, direction, x = spec
    t = np.array([e["t"] for e in truths])
    level = np.nan
    if kind == "level":
        at = np.interp(t, np.arange(len(x)), x)
        at = at[np.isfinite(at)]
        if not len(at):
            return None
        level = float(np.median(at))
    offsets = []
    for e, win in zip(truths, windows):
        if win is None:
            continue
        found = fd.find_events(spec, level, win[0], win[1])
        if len(found):
            offsets.append(found[np.argmin(abs(found - win[2]))] - e["t"])
    if len(offsets) < 3:
        return None
    return dict(level=level, bias=float(np.median(offsets)), n=len(truths))


def detector_specs(kin):
    specs = {}
    for name, fn in fd.DETECTORS.items():
        for limb in LIMBS:
            for kind, spec in fn(kin, limb).items():
                specs[(name, limb, kind)] = spec
    return specs


def fit_detectors(specs, anchors, windows):
    """Calibrate every detector on every trusted event."""
    params = {}
    for (name, limb, kind), spec in specs.items():
        idx = [i for i, e in enumerate(anchors) if e["event"] == kind
               and (not CALIBRATE_PER_LIMB or e["limb"] == limb)]
        if len(idx) < MIN_CALIBRATION_EVENTS:      # too few: pool both limbs
            idx = [i for i, e in enumerate(anchors) if e["event"] == kind]
        truths = [anchors[i] for i in idx]
        params[(name, limb, kind)] = calibrate(
            spec, truths, [windows.get(i) for i in idx])
    return params


def fallback_order(bench):
    """-> {(limb, event): [detector, ...]} from FALLBACK_ORDER."""
    keys = [(limb, kind) for limb in LIMBS for kind in (HS, TO)]
    if FALLBACK_ORDER != "auto":
        return {k: list(FALLBACK_ORDER) for k in keys}
    order = {}
    for limb, kind in keys:
        rows = [r for r in bench if r["limb"] == limb and r["event"] == kind
                and r["found_pct"] >= AUTO_MIN_FOUND_PCT and "p95_abs_ms" in r]
        order[(limb, kind)] = [r["detector"] for r in
                               sorted(rows, key=lambda r: r["p95_abs_ms"])]
    return order


def search(specs, params, order, limb, kind, lo, hi, p, unverified=()):
    """The event in (lo, hi): an unverifiable GRF event if there is one,
    else the first detector in the fallback order that finds it."""
    grf = [e for e in unverified if e["limb"] == limb and e["event"] == kind
           and lo < e["t"] < hi]
    if grf:
        return min(grf, key=lambda e: abs(e["t"] - p)), "GRF_unverified"
    for name in order[(limb, kind)]:
        par = params.get((name, limb, kind))
        if par is None:
            continue
        found = fd.find_events(specs[(name, limb, kind)], par["level"],
                               lo + par["bias"], hi + par["bias"])
        found = found - par["bias"]
        found = found[(found > lo) & (found < hi)]
        if len(found):
            return float(found[np.argmin(abs(found - p))]), name
    return None, None


def benchmark(specs, anchors, windows):
    """Score every detector on trusted events it was not calibrated on.

    Each (limb, event) group is split in two by alternating event; each half
    is predicted with the calibration from the other half, from the same
    window the gap filler would have used had that event been missing.
    """
    rows = []
    for (name, limb, kind), spec in specs.items():
        idx = [i for i, e in enumerate(anchors)
               if e["limb"] == limb and e["event"] == kind and i in windows]
        errors, missed = [], 0
        for fold in (0, 1):
            fit = [anchors[i] for j, i in enumerate(idx) if j % 2 != fold]
            test = [i for j, i in enumerate(idx) if j % 2 == fold]
            par = calibrate(spec, fit, [windows[i] for j, i in enumerate(idx)
                                        if j % 2 != fold])
            if par is None:
                missed += len(test)
                continue
            for i in test:
                lo, hi, p = windows[i]
                found = fd.find_events(spec, par["level"], lo + par["bias"],
                                       hi + par["bias"]) - par["bias"]
                found = found[(found > lo) & (found < hi)]
                if not len(found):
                    missed += 1
                    continue
                errors.append(found[np.argmin(abs(found - p))]
                              - anchors[i]["t"])
        err = np.array(errors) * 1000 / KINEMATIC_RATE
        n = len(idx)
        row = dict(detector=name, limb=limb, event=kind, n_trusted=n,
                   found_pct=100 * len(err) / n if n else np.nan)
        if len(err):
            row.update(bias_ms=np.mean(err), sd_ms=np.std(err),
                       mae_ms=np.mean(abs(err)),
                       p95_abs_ms=np.percentile(abs(err), 95),
                       within_10ms_pct=100 * np.mean(abs(err) <= 10),
                       within_20ms_pct=100 * np.mean(abs(err) <= 20))
        rows.append(row)
    return rows


# %%==========================================================================
#  SEQUENCE
# ============================================================================

def build_sequence(anchors, d, specs, params, order, first_row, last_row,
                   unverified=()):
    """Every event, in order: trusted ones kept, the rest searched for."""

    def found_event(found, name, limb, kind, pos):
        if name == "GRF_unverified":
            return dict(found, pos=pos)
        return dict(t=found, limb=limb, event=kind, source="kinematic",
                    method=name, pos=pos)

    cycle = sum(d.values())
    out, dropped = [], []

    def expected(pos, steps):
        return sum(d[(pos + j) % 4] for j in range(steps))

    # keep only anchors that agree with each other on the sequence. Two
    # events that follow each other in the sequence can be any distance
    # apart; one that implies missing events in between cannot come sooner
    # than half the time those events need.
    clean = [anchors[0]]
    for e in anchors[1:]:
        a = clean[-1]
        k = (e["pos"] - a["pos"]) % 4 or 4
        gap = e["t"] - a["t"]
        if gap < MIN_SEPARATION or (k > 1 and gap < 0.5 * expected(a["pos"], k)):
            dropped.append(e)
            continue
        clean.append(e)
    for e in dropped:
        print(f"    ! trusted {e['limb']} {e['event']} at frame {e['t']:.1f} "
              f"is out of sequence with its neighbours, dropped")

    # backwards from the first anchor
    head, t, pos = [], clean[0]["t"], clean[0]["pos"]
    while True:
        pos = (pos - 1) % 4
        p = t - d[pos]
        lo, hi = p - 0.5 * d[(pos - 1) % 4], min(t - MIN_SEPARATION,
                                                 p + 0.5 * d[pos])
        if lo < first_row:
            break
        limb, kind = SEQUENCE[pos]
        found, name = search(specs, params, order, limb, kind, lo, hi, p,
                             unverified)
        if found is None:
            break
        head.append(found_event(found, name, limb, kind, pos))
        t = head[-1]["t"]
    out.extend(reversed(head))

    for a, b in zip(clean[:-1], clean[1:]):
        out.append(a)
        gap = b["t"] - a["t"]
        k = (b["pos"] - a["pos"]) % 4 or 4
        steps = min((k + 4 * m for m in range(int(gap / cycle) + 2)),
                    key=lambda s: abs(np.log(gap / expected(a["pos"], s))))
        total = expected(a["pos"], steps)
        scale = gap / total
        prev_t = a["t"]
        for j in range(1, steps):
            pos = (a["pos"] + j) % 4
            limb, kind = SEQUENCE[pos]
            # predicted from the event just placed, not stretched from the
            # anchor, so a long gap follows the cadence instead of drifting
            p = prev_t + d[(pos - 1) % 4] * scale
            left = b["t"] - expected(pos, steps - j) * scale * 0.5
            lo = max(prev_t + MIN_SEPARATION,
                     p - 0.5 * d[(pos - 1) % 4] * scale)
            hi = min(b["t"] - MIN_SEPARATION, p + 0.5 * d[pos] * scale,
                     max(left, prev_t + 2 * MIN_SEPARATION))
            found, name = search(specs, params, order, limb, kind, lo, hi, p,
                                 unverified)
            if found is not None:
                e = found_event(found, name, limb, kind, pos)
            elif INTERPOLATE_MISSING:
                e = dict(t=max(p, prev_t + MIN_SEPARATION), limb=limb,
                         event=kind, source="interpolated",
                         method="interpolated", pos=pos)
            else:
                continue
            out.append(e)
            prev_t = e["t"]
    out.append(clean[-1])

    # forwards from the last anchor
    t, pos = clean[-1]["t"], clean[-1]["pos"]
    while True:
        p = t + d[pos]
        lo = max(t + MIN_SEPARATION, p - 0.5 * d[pos])
        pos = (pos + 1) % 4
        hi = p + 0.5 * d[pos]
        if hi > last_row:
            break
        limb, kind = SEQUENCE[pos]
        found, name = search(specs, params, order, limb, kind, lo, hi, p,
                             unverified)
        if found is None:
            break
        out.append(found_event(found, name, limb, kind, pos))
        t = out[-1]["t"]
    return out, dropped


def interval_flags(events):
    """Stance and stride of each HS, and whether either is an outlier."""
    for e in events:
        e.update(stance_s=np.nan, stride_s=np.nan, flag="")
    for limb in LIMBS:
        mine = [e for e in events if e["limb"] == limb]
        for i, e in enumerate(mine):
            if e["event"] != HS:
                continue
            nxt = [x for x in mine[i + 1:] if x["event"] == TO][:1]
            hs2 = [x for x in mine[i + 1:] if x["event"] == HS][:1]
            if nxt:
                e["stance_s"] = (nxt[0]["t"] - e["t"]) / KINEMATIC_RATE
            if hs2:
                e["stride_s"] = (hs2[0]["t"] - e["t"]) / KINEMATIC_RATE
    for key in ("stance_s", "stride_s"):
        for limb in LIMBS:
            vals = [e[key] for e in events if e["limb"] == limb
                    and np.isfinite(e[key])]
            if not vals:
                continue
            med = np.median(vals)
            for e in events:
                if (e["limb"] == limb and np.isfinite(e[key]) and not
                        FLAG_RATIO[0] * med <= e[key] <= FLAG_RATIO[1] * med):
                    e["flag"] = (e["flag"] + " " + key[:-2]).strip()


def trim_incomplete(events):
    """Drop a limb's leading toe-off and trailing heel strike."""
    out = list(events)
    for limb in LIMBS:
        mine = [e for e in out if e["limb"] == limb]
        drop = []
        if mine and mine[0]["event"] == TO:
            drop.append(mine[0])
        if mine and mine[-1]["event"] == HS:
            drop.append(mine[-1])
        out = [e for e in out if not any(e is x for x in drop)]
    return out


# %%==========================================================================
#  ONE TRIAL
# ============================================================================

def process_trial(force_path, meta, binding, verbose=True):
    stem = force_path.stem
    kin_path = find_file(KINEMATIC_FOLDER, stem, "_metrics.csv")
    say = print if verbose else (lambda *a, **k: None)
    say(f"\n{stem}")

    forces = load_forces(force_path)
    n_samples = len(forces["L"]["fz"])
    frames, vectors, poses, seg_of_limb, landmarks = load_kinematics(
        kin_path, meta)
    for limb, P in poses.items():
        if P is None:                    # no 4x4 in the export: fit landmarks
            world = np.stack([vectors[n] for n in landmarks[limb]], axis=1)
            k = seg_of_limb[limb]
            poses[limb] = ab.poses_from_landmarks(
                binding["landmarks_local"][k, :len(landmarks[limb])], world)
    n_rows = len(frames)
    say(f"  force {n_samples / GRF_RATE:.1f} s, kinematics "
        f"{n_rows / KINEMATIC_RATE:.1f} s")
    if abs(n_samples / GRF_RATE - n_rows / KINEMATIC_RATE) > 1.0:
        say("  ! the two recordings differ in length by more than 1 s; "
            "events assume they start together")

    # 1. belt contacts
    contacts, force100, cop100 = [], {}, {}
    rows_in_force = np.arange(n_rows) * STEP
    in_force = rows_in_force < n_samples
    for b in LIMBS:
        found, force, info = detect_belt_contacts(forces[b]["fz"], b)
        contacts += found
        say(f"  {BELT_CHANNEL[b]}: {len(found)} contacts, baseline "
            f"{info['baseline_n']:.1f} N (drift {info['baseline_drift_n']:.1f}"
            f" N), noise {info['noise_n']:.1f} N")
        force100[b] = np.full(n_rows, np.nan)
        force100[b][in_force] = force[rows_in_force[in_force]]
        cop100[b] = np.full((n_rows, 2), np.nan)
        cop100[b][in_force] = forces[b]["cop"][rows_in_force[in_force]]
        cop100[b][~(force100[b] > COP_MIN_FORCE_N)] = np.nan
    contacts.sort(key=lambda c: c["start"])

    # 2. feet
    feet = Feet(poses, binding, seg_of_limb)
    say(f"  belt surface at z = {feet.floor['L'] * 1000:.1f} / "
        f"{feet.floor['R'] * 1000:.1f} mm (left / right sole)")

    belts_fp, guessed = belt_polygons_fp(forces)
    if guessed:
        say("  ! BELT_CORNERS_MM not set: belts guessed from the CoP")
    if FP_TO_THEIA is not None:
        A = np.asarray(FP_TO_THEIA, float)
        reg = dict(source="config")
    else:
        A, reg = estimate_registration(force100, cop100, feet, belts_fp)
        reg["source"] = "estimated"
    say(f"  FP -> Theia ({reg['source']}): "
        f"{json.dumps(np.round(A, 5).tolist())}")
    if "median_side_residual_mm" in reg:
        say(f"    CoP vs contact patch over {reg['pairs']} single-support "
            f"frames: side to side median {reg['median_side_residual_mm']:.1f}"
            f" mm (90% {reg['p90_side_residual_mm']:.1f}), along the belt "
            f"{reg['median_along_residual_mm']:.1f} mm; belt axis vs CoP "
            f"track {reg['belt_axis_vs_cop_track_deg']:.1f} deg")
        if reg["p90_side_residual_mm"] > MAX_REGISTRATION_ERROR_MM:
            say("  ! registration is poor; trust labels will be unreliable")

    belts = {b: apply_2d(A, poly) for b, poly in belts_fp.items()}
    cop_theia = {b: apply_2d(A, cop100[b] / 1000) for b in LIMBS}
    feet.belt_membership(belts, cop_theia, BELT_MARGIN_MM / 1000)

    # 3. trust
    anchors, unverified = label_contacts(contacts, feet, force100)
    for e in anchors:
        e["pos"] = POSITION[(e["limb"], e["event"])]
    labels = {}
    for c in contacts:
        labels[c["label"]] = labels.get(c["label"], 0) + 1
    say("  contacts: " + ", ".join(f"{v} {k}" for k, v in
                                    sorted(labels.items())))
    why = {}
    for c in contacts:
        for key in ("hs_reason", "to_reason"):
            if c[key] != "trusted":
                why[c[key]] = why.get(c[key], 0) + 1
    say(f"  trusted GRF events: {len(anchors)} of {2 * len(contacts)}; "
        "not trusted: " + ", ".join(f"{v} {k}" for k, v in
                                    sorted(why.items(), key=lambda x: -x[1])))

    # 4. fallback
    kin = Kinematics(vectors, feet, KINEMATIC_RATE)
    specs = detector_specs(kin)
    d = expected_intervals(anchors)
    windows = neighbour_windows(anchors, d)
    params = fit_detectors(specs, anchors, windows)
    bench = (benchmark(specs, anchors, windows)
             if WRITE_BENCHMARK or FALLBACK_ORDER == "auto" else [])
    order = fallback_order(bench)
    if FALLBACK_ORDER == "auto":
        for (limb, kind), names in order.items():
            say(f"  fallback for {limb} {kind}: "
                f"{', '.join(names[:3]) or 'none qualifies'}")

    # 5. sequence
    valid = feet.valid["L"] & feet.valid["R"]
    first_row = float(np.argmax(valid))
    last_row = float(n_rows - 1 - np.argmax(valid[::-1]))
    events, dropped = build_sequence(anchors, d, specs, params, order,
                                     first_row, last_row, unverified)
    untrimmed = list(events)
    if TRIM_TO_COMPLETE_STANCES:
        events = trim_incomplete(events)
    interval_flags(events)
    for e in events:
        r = int(np.clip(round(e["t"]), 0, n_rows - 1))
        e["frame_100hz"] = int(frames[r])

    # outputs
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_FOLDER / f"{stem}_merged_events.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event_type", "support_limb", "frame_100hz", "source"])
        for e in events:
            w.writerow([e["event"], e["limb"], e["frame_100hz"], e["source"]])

    qa_fields = ["event_type", "support_limb", "frame_100hz", "source",
                 "method", "frame_float", "grf_frame_1000hz", "belt",
                 "contact_label", "stance_s", "stride_s", "flag"]
    with open(OUTPUT_FOLDER / f"{stem}_event_qa.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(qa_fields)
        for e in events:
            w.writerow([e["event"], e["limb"], e["frame_100hz"], e["source"],
                        e["method"], f"{e['t']:.2f}", e.get("grf_sample", ""),
                        e.get("belt", ""), e.get("contact_label", ""),
                        _fmt(e["stance_s"]), _fmt(e["stride_s"]), e["flag"]])

    c_fields = ["belt", "start_1000hz", "stop_1000hz", "start_frame_100hz",
                "stop_frame_100hz", "duration_s", "peak_n", "loading_s",
                "unloading_s", "dips_merged",
                "foot", "label", "hs_trusted", "hs_reason", "to_trusted",
                "to_reason", "cop_inside_fraction"]
    with open(OUTPUT_FOLDER / f"{stem}_grf_contacts.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(c_fields)
        for c in contacts:
            r0 = int(np.clip(round(c["start"] / STEP), 0, n_rows - 1))
            r1 = int(np.clip(round(c["stop"] / STEP), 0, n_rows - 1))
            w.writerow([c["belt"], c["start"], c["stop"], frames[r0],
                        frames[r1], f"{(c['stop'] - c['start']) / GRF_RATE:.3f}",
                        f"{c['peak_n']:.1f}", f"{c['loading_s']:.3f}",
                        f"{c['unloading_s']:.3f}", c["dips_merged"], c["foot"],
                        c["label"], c["hs_trusted"], c["hs_reason"],
                        c["to_trusted"], c["to_reason"],
                        _fmt(c["cop_inside"])])

    if bench:
        _write_rows(OUTPUT_FOLDER / f"{stem}_fallback_benchmark.csv",
                    [dict(trial=stem, **r) for r in bench])

    with open(OUTPUT_FOLDER / f"{stem}_event_registration.json", "w") as fh:
        json.dump(dict(fp_to_theia=A.tolist(), registration=reg,
                       belts_theia={b: p.tolist() for b, p in belts.items()},
                       belts_guessed=guessed, floor_m=feet.floor,
                       forward=feet.forward.tolist(),
                       expected_intervals_frames={
                           f"{SEQUENCE[p][0]} {SEQUENCE[p][1]} -> next": v
                           for p, v in d.items()}), fh, indent=2)

    by_source = {}
    for e in events:
        by_source[e["method"]] = by_source.get(e["method"], 0) + 1
    flagged = sum(bool(e["flag"]) for e in events)
    say(f"  events: {len(events)} ("
        + ", ".join(f"{v} {k}" for k, v in sorted(by_source.items())) + ")")
    say(f"  stance/stride outliers flagged: {flagged}")
    if bench:
        say("  fallback benchmark (held-out trusted events, ms):")
        say("    " + _bench_table(bench).replace("\n", "\n    "))
    say(f"  saved {out}")
    return dict(events=events, untrimmed=untrimmed, contacts=contacts,
                anchors=anchors, unverified=unverified,
                benchmark=bench, registration=(A, reg), feet=feet,
                params=params, dropped=dropped)


def _fmt(x):
    return "" if x is None or not np.isfinite(x) else f"{x:.3f}"


def _write_rows(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v)
                        for k, v in r.items()})


def _bench_table(rows):
    lines = [f"{'detector':34s} {'limb':4s} {'event':11s} {'found%':>6s} "
             f"{'bias':>6s} {'sd':>6s} {'p95':>6s}"]
    for r in sorted(rows, key=lambda r: (r["event"], r.get("p95_abs_ms", 1e9),
                                         r["limb"])):
        lines.append(
            f"{r['detector']:34s} {r['limb']:4s} {r['event']:11s} "
            f"{r['found_pct']:6.1f} {r.get('bias_ms', np.nan):6.1f} "
            f"{r.get('sd_ms', np.nan):6.1f} {r.get('p95_abs_ms', np.nan):6.1f}")
    return "\n".join(lines)


def main(stems=None):
    meta, binding = load_binding(BINDING_FILE)
    stems = stems or TRIALS
    if stems:
        paths = [find_file(FORCE_FOLDER, s, ".csv") for s in stems]
    else:
        paths = sorted(Path(FORCE_FOLDER).glob("*.csv"))
    everything = []
    for path in paths:
        try:
            result = process_trial(path, meta, binding)
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            print(f"\n{path.stem}: SKIPPED -- {exc}")
            continue
        everything += [dict(trial=path.stem, **r)
                       for r in result["benchmark"]]
    if everything:
        _write_rows(OUTPUT_FOLDER / "fallback_benchmark_all_trials.csv",
                    everything)
        print(f"\nsaved {OUTPUT_FOLDER / 'fallback_benchmark_all_trials.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
