# -*- coding: utf-8 -*-
"""
Heel strikes and toe-offs on the DICE split-belt treadmill.

    1. GRF first. Every belt contact is found from the vertical force. Each
       of its two events (heel strike, toe-off) is kept only if, around that
       event, the MOVE4D boot meshes posed on the Theia feet show that belt
       carrying one boot, that boot not also standing on the other belt, the
       other boot off this belt, the centre of pressure under the boot, and a
       force that rises or falls like a foot landing or leaving. Hanging over
       the gap or the edge is fine. The limb comes from the mesh, not from
       the belt's name, so a clean crossover step is kept with the right limb.

    2. Zeni for the rest. Every event the GRF cannot vouch for is taken from
       Zeni et al. (2008): heel strike where the heel is furthest in front of
       the pelvis, toe-off where the toe is furthest behind it. Each one is
       searched for only in the gap the gait sequence leaves for it
       (L HS -> R TO -> R HS -> L TO), and shifted by Zeni's median offset
       from this trial's trusted GRF events.

    3. Only where Zeni finds nothing (a tracking dropout, in practice): a
       clean-looking belt contact the mesh could not check, then a time
       interpolated between the neighbouring events.

Everything is in this one file. Needs numpy, scipy, and two files that sit
next to it: foot_mesh_binding.npz (the boot meshes, from
build_foot_binding.py) and the force-plate parameters from the C3D.

Run it with F5 after setting the paths in CONFIG, or

    python detect_gait_events.py                   # every trial
    python detect_gait_events.py "D05_C1_Treadmill_1.3mpers 108bpm"

Per trial, in OUTPUT_FOLDER:

    {trial}_merged_events.csv   event_type, support_limb, frame_100hz, source
                                source: GRF | kinematic (Zeni) |
                                GRF_unverified | interpolated
    {trial}_event_qa.csv        the same events, how each was found, stance
                                and stride, and a flag on outliers
    {trial}_grf_contacts.csv    every belt contact: which foot the mesh put
                                on it, its label, and why each event was or
                                was not trusted
    {trial}_event_summary.json  plate fit, registration, belt surface, and
                                how far Zeni sits from the trusted GRF events
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent


# %% ==========================================================================
#  CONFIG
# =============================================================================

DATA_FOLDER = Path(
    r"C:\Users\lcour081\Box\Military Project\Data\Analysis\DICE_Treadmill"
)
FORCE_FOLDER = DATA_FOLDER / "FP_renamed"            # {trial}.csv, 1000 Hz
KINEMATIC_FOLDER = DATA_FOLDER / "Theia_csv_outputs"  # {trial}_metrics.csv
OUTPUT_FOLDER = DATA_FOLDER / "gait_event_outputs"
# The participant's foot binding from build_foot_binding.py -- the boot mesh
# itself -- NOT a trial's _posed.npz (that holds poses only, and this script
# reads the poses from the Theia export anyway). {participant} is replaced
# by the start of each trial's name up to the first "_" (D05_C1_... -> D05),
# so each participant gets their own boots. A plain path is used for all.
BINDING_FILE = DATA_FOLDER / "Foot bindings" / "{participant}_foot_mesh_binding.npz"
PLATE_FILE = HERE / "force_plates_DICE_treadmill.txt"
TRIALS = None                  # None = every .csv in FORCE_FOLDER, or a list

GRF_RATE = 1000
KINEMATIC_RATE = 100

# Force channels, and the frame the CoP is in. "plate": each belt's own
# frame, origin at the plate centre (the DICE export: COP = (-My/Fz, Mx/Fz)
# - OFFSET exactly). "lab": already in the frame of the plate corners.
BELT = {"L": "Left belt", "R": "Right belt"}
COP_FRAME = "plate"

# Plate-corner lab frame (m) -> Theia (m), as [[a, b, tx], [c, d, ty]].
# None: estimated per trial from single support and printed in this form,
# with its distance from the identity. "identity": Theia uses the mocap lab
# frame.
LAB_TO_THEIA = None

# Belt contacts from force
FORCE_FILTER_HZ = 50.0
FORCE_THRESHOLD_N = 20.0       # above each belt's own baseline
BASELINE_WINDOW_S = 10.0       # the baseline is re-measured this often
MERGE_GAP_S = 0.030            # dips shorter than this do not split a contact
MIN_CONTACT_S = 0.080          # contacts shorter than this are dropped
SHAPE_LEVEL_N = 200.0          # a real landing reaches this within ...
MAX_LOADING_S = 0.060          # ... this (D05: 10-36 ms)
MAX_UNLOADING_S = 0.080        # and leaves it within this (D05: 34-45 ms;
                               # the left-belt tails run to 315 ms)

# Boots
SOLE_CELL_MM = 10.0            # sole = lowest mesh vertex in each cell
SOLE_MAX_RISE_MM = 35.0        # ... up to this far above the lowest one
CONTACT_HEIGHT_MM = 15.0       # a sole vertex this close to the belt touches
MIN_CONTACT_VERTICES = 3       # this many touching = the boot is down
# Bend each boot at the MTP by Theia's toe angle (a hinge about the foot's
# X axis through <Side>_Toes_Position). Without it the rigid toe cap swings
# 30-40 mm through the belt at every push-off. A positive angle is extension
# with TOE_SIGN = 1 (checked on D05: the toes then lie flat on the floor).
TOE_HINGE = True
TOE_SIGN = 1
TOE_REFERENCE_DEG = 0.0        # toe angle in the static scan (D05: +0.4/-1.4)

# Trust
TRUST_WINDOW_S = 0.050         # checked over +/- this around each event
# A boot is on a plate when at least MIN_CONTACT_VERTICES of its touching
# sole vertices are inside the plate's fitted outline, grown by the outline's
# own uncertainty (the worst corner miss, 8 mm on DICE) plus this. Hanging
# over the gap or the outer edge is fine -- it moves no force anywhere -- so
# only a boot on the wrong plate costs an event.
EXTRA_MARGIN_MM = 0.0
SHARED_MIN_FRAMES = 2
STRADDLE_MIN_FRAMES = 2
COP_TOLERANCE_MM = 30.0        # CoP this close to the contact patch = under it
COP_MIN_FORCE_N = 150.0
COP_MIN_INSIDE = 0.90          # share of the contact the CoP must be under it

# Zeni
KINEMATIC_FILTER_HZ = 10.0
MAX_GAP_FILL_FRAMES = 10       # tracking gaps bridged before filtering
MIN_CALIBRATION_EVENTS = 10    # fewer on one limb: pool both limbs
INTERPOLATE_MISSING = True     # last resort, between two trusted events only

# Output
TRIM_TO_COMPLETE_STANCES = True   # each limb starts on HS and ends on TO
FLAG_RATIO = (0.75, 1.25)         # stance / stride outside this x median


LIMBS = ("L", "R")
OTHER = {"L": "R", "R": "L"}
SIDE = {"L": "Left", "R": "Right"}
HS, TO = "heel_strike", "toe_off"
SEQUENCE = [("L", HS), ("R", TO), ("R", HS), ("L", TO)]   # walking order
POSITION = {key: i for i, key in enumerate(SEQUENCE)}
STEP = GRF_RATE // KINEMATIC_RATE
MIN_SEPARATION = 1.0           # frames between consecutive events


# %% ==========================================================================
#  READING FILES
# =============================================================================

def find_file(folder, stem, suffix=""):
    """The exports spell trials with a space or an underscore; try both."""
    for name in (stem, stem.replace(" ", "_"), stem.replace("_", " ")):
        path = Path(folder) / f"{name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"no {stem}{suffix} in {folder}")


def read_forces(path):
    """-> fz {limb: (n,)}, cop {limb: (n, 2) mm, as exported}."""
    with open(path, encoding="utf-8-sig") as fh:
        header = next(csv.reader(fh))
    names = [f"{BELT[b]}_{q}" for b in LIMBS
             for q in ("Force_Z", "COP_X", "COP_Y")]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2,
                      usecols=[header.index(n) for n in names])
    fz = {b: data[:, 3 * i] for i, b in enumerate(LIMBS)}
    cop = {b: data[:, 3 * i + 1:3 * i + 3] for i, b in enumerate(LIMBS)}
    return fz, cop


def read_plates(path):
    """{limb: {FORCE_PLATE_*: value}} from the C3D plate parameters."""
    blocks, name = {}, None
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue
            if parts[0] == "FORCE_PLATE_NAME":
                name = parts[1].strip()
                blocks[name] = {}
            elif name is not None:
                blocks[name][parts[0]] = float(parts[1])
    return {b: blocks[BELT[b]] for b in LIMBS}


def read_theia(path, pose_names):
    """Only the columns needed, from a Visual3D metrics export.

    -> frames (ITEM numbers), points {name: (F, 3)}, pose {limb: (F, 4, 4)}
    """
    points = ["Pelvis_Position"] + [f"{SIDE[b]}_{p}_Position"
                                    for b in LIMBS for p in ("Heel", "Toes")]
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        head = [next(fh).rstrip("\r\n").split("\t") for _ in range(5)]
        where = {(n, c.strip().upper()): i
                 for i, (n, c) in enumerate(zip(head[1], head[4])) if i}
        want = [(p, a) for p in points for a in "XYZ"]
        want += [(pose_names[b], str(i)) for b in LIMBS for i in range(16)]
        missing = [w for w in want if w not in where]
        if missing:
            raise KeyError(f"{path.name} lacks {sorted({m[0] for m in missing})}")
        # optional: the toe hinge angle, or the toes' own 4x4
        extra = [(f"{SIDE[b]}_Toes_Joint_Angle", "X") for b in LIMBS]
        extra += [(f"{SIDE[b]}_Toes_Global_4x4", str(i)) for b in LIMBS
                  for i in range(16)]
        extra = [w for w in extra if w in where]
        want += extra
        cols = [where[w] for w in want]
        items, rows = [], []
        for line in fh:
            cells = line.rstrip("\r\n").split("\t")
            if not cells[0].strip():
                continue
            items.append(cells[0])
            rows.append([float(cells[i]) if i < len(cells) and cells[i].strip()
                         else np.nan for i in cols])
    data = np.array(rows, float).reshape(len(rows), len(cols))
    last = len(data)
    while last and np.isnan(data[last - 1]).all():     # trailing padding
        last -= 1
    data = data[:last]
    frames = np.array([int(float(i)) for i in items[:last]])

    out = {p: data[:, 3 * k:3 * k + 3] for k, p in enumerate(points)}
    pose = {}
    for k, b in enumerate(LIMBS):
        start = 3 * len(points) + 16 * k
        P = data[:, start:start + 16].reshape(-1, 4, 4).copy()
        P[:, :3, :3] = orthonormalise(P[:, :3, :3])
        P[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
        P[~np.isfinite(P[:, :3, :]).all(axis=(1, 2))] = np.nan
        pose[b] = P
    column = {w: data[:, j] for j, w in enumerate(want)}
    toes = {}
    for b in LIMBS:
        angle = column.get((f"{SIDE[b]}_Toes_Joint_Angle", "X"))
        four = [column.get((f"{SIDE[b]}_Toes_Global_4x4", str(i)))
                for i in range(16)]
        toes[b] = dict(angle=angle, pose=None if any(c is None for c in four)
                       else np.stack(four, axis=1).reshape(-1, 4, 4))
    return frames, out, pose, toes


def binding_for(stem):
    """The binding file for a trial: BINDING_FILE with {participant} filled."""
    return Path(str(BINDING_FILE).replace("{participant}", stem.split("_")[0]))


def read_binding(path):
    """-> boot mesh vertices per limb, in the foot frame, and the name of the
    pose signal that carries each foot."""
    if not Path(path).exists():
        raise FileNotFoundError(f"no foot binding at {path}")
    z = np.load(path, allow_pickle=False)
    if "vertices_local" not in z.files:
        raise KeyError(
            f"{Path(path).name} is not a foot binding -- "
            + ("it is a trial's _posed.npz, which holds poses only. "
               if "poses" in z.files else "")
            + "Set BINDING_FILE to the participant's *_foot_mesh_binding.npz "
            "from build_foot_binding.py")
    meta = json.loads(str(z["meta"]))
    seg = {"L": meta["segments"].index("left_foot"),
           "R": meta["segments"].index("right_foot")}
    verts = {b: z["vertices_local"][z["vertex_segment"] == seg[b]]
             for b in LIMBS}
    pose_names = {b: meta["pose_signal"][meta["segments"][seg[b]]]
                  for b in LIMBS}
    return verts, pose_names


# %% ==========================================================================
#  SMALL MATHS
# =============================================================================

def orthonormalise(R):
    """Nearest rotation to each 3x3; frames with NaN stay NaN."""
    out = np.full_like(R, np.nan)
    ok = np.isfinite(R).all(axis=(1, 2))
    if ok.any():
        U, _, Vt = np.linalg.svd(R[ok])
        D = np.ones(U.shape[:-1])
        D[..., -1] = np.sign(np.linalg.det(U @ Vt))
        out[ok] = (U * D[..., None, :]) @ Vt
    return out


def kabsch(P, Q):
    """Rigid (R, t) with Q ~ R P + t, least squares, P and Q (K, 3)."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((Q - cq).T @ (P - cp))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, cq - R @ cp


def lowpass(x, cutoff, rate):
    """Zero-lag Butterworth that survives NaN: gaps up to MAX_GAP_FILL_FRAMES
    are bridged, longer ones stay NaN, and each finite stretch is filtered
    on its own (a single NaN would otherwise spread over the whole signal)."""
    x = np.array(x, float)
    flat = x.reshape(len(x), -1)
    sos = butter(4, cutoff, btype="lowpass", fs=rate, output="sos")
    n = np.arange(len(x))
    for j in range(flat.shape[1]):
        y = flat[:, j]
        ok = np.isfinite(y)
        if ok.sum() < 2:
            continue
        y = np.interp(n, n[ok], y[ok])
        for s, e in runs(~ok):
            if e - s > MAX_GAP_FILL_FRAMES or s == 0 or e == len(y):
                y[s:e] = np.nan
        out = np.full(len(y), np.nan)
        for s, e in runs(np.isfinite(y)):
            out[s:e] = sosfiltfilt(sos, y[s:e]) if e - s > 30 else y[s:e]
        flat[:, j] = out
    return flat.reshape(x.shape)


def runs(mask):
    """[(start, stop), ...] of the True stretches of a boolean array."""
    edges = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


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


def sole_chunks(pose, sole, chunk=4000):
    """Yield (slice, (n, V, 3)) world positions of the sole vertices; the
    toe cap rides the toes pose when the boot is bent at the MTP."""
    v, toe, toe_pose = sole["v"], sole["toe"], sole["toe_pose"]
    for s in range(0, len(pose), chunk):
        P = pose[s:s + chunk]
        W = v @ P[:, :3, :3].transpose(0, 2, 1) + P[:, None, :3, 3]
        if toe_pose is not None:
            T = toe_pose[s:s + chunk]
            W[:, toe] = (v[toe] @ T[:, :3, :3].transpose(0, 2, 1)
                         + T[:, None, :3, 3])
        yield slice(s, s + len(P)), W


def rot_x(a):
    """Rotations about the foot's X (medio-lateral) axis, a (F,) in rad."""
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = 1.0
    R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = c, -s, s, c
    return R


def toe_hinge(pose, mtp_world, angle, toes4, sole_v):
    """The toes pose, and which sole vertices ride it: the foot pose turned
    by the toe angle about the MTP (fixed in the foot frame), or the export's
    own toes 4x4. -> (toe_pose or None, toe mask, MTP in the foot frame)."""
    none = (None, np.zeros(len(sole_v), bool), None)
    if not TOE_HINGE or (angle is None and toes4 is None):
        return none
    R, t = pose[:, :3, :3], pose[:, :3, 3]
    ok = np.isfinite(pose).all(axis=(1, 2))
    if toes4 is not None:
        mtp_world = toes4[:, :3, 3]
    local = np.einsum("fji,fj->fi", R, mtp_world - t)
    good = ok & np.isfinite(local).all(axis=1)
    if good.sum() < 10:
        return none
    m = np.median(local[good], axis=0)
    up = np.median(R[ok][:, 2, :], axis=0)
    up /= np.linalg.norm(up)
    fwd = m - m[0] * np.array([1.0, 0, 0])
    fwd -= (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    toe = (sole_v - m) @ fwd > 0
    if toes4 is not None:
        rel = np.einsum("fji,fjk->fik", R, orthonormalise(toes4[:, :3, :3]))
    else:
        rel = rot_x(TOE_SIGN * np.radians(angle - TOE_REFERENCE_DEG))
    rel[~np.isfinite(rel).all(axis=(1, 2))] = np.eye(3)     # no angle: straight
    H = np.zeros((len(rel), 4, 4))
    H[:, :3, :3], H[:, :3, 3], H[:, 3, 3] = rel, m - rel @ m, 1.0
    return pose @ H, toe, m


# %% ==========================================================================
#  1. BELT CONTACTS FROM FORCE
# =============================================================================

def belt_baseline(fz):
    """The unloaded reading, re-measured every BASELINE_WINDOW_S. A window
    where the belt is never unloaded (standing before the belt starts) sits
    far above the rest and borrows its neighbours' baseline instead."""
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
    return base, float(noise)


def find_contacts(fz, belt):
    """-> contacts on one belt, and its filtered force above baseline."""
    base, noise = belt_baseline(fz)
    sos = butter(4, FORCE_FILTER_HZ, btype="lowpass", fs=GRF_RATE,
                 output="sos")
    force = sosfiltfilt(sos, fz) - base
    merged = []
    for s, e in runs(force > FORCE_THRESHOLD_N):
        if merged and s - merged[-1][1] < MERGE_GAP_S * GRF_RATE:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    contacts = []
    for s, e in merged:
        if e - s < MIN_CONTACT_S * GRF_RATE:
            continue
        high = np.flatnonzero(force[s:e] > SHAPE_LEVEL_N)
        contacts.append(dict(
            belt=belt, start=int(s), stop=int(e),
            complete_start=s > 0, complete_stop=e < len(fz),
            peak_n=float(force[s:e].max()),
            loading_s=(high[0] if len(high) else e - s) / GRF_RATE,
            unloading_s=(e - s - high[-1] if len(high) else e - s) / GRF_RATE))
    return contacts, force, dict(baseline_n=float(np.median(base)),
                                 noise_n=noise)


# %% ==========================================================================
#  2. BELTS, BOOTS AND WHERE THEY ARE
# =============================================================================

def fit_plate(p):
    """The nominal LENGTH x WIDTH rectangle fitted rigidly onto the four
    measured corners (lab = R @ (x, y, 0) + t, mm). The corners' miss in
    the horizontal plane is how uncertain the outline is."""
    L, W = p["FORCE_PLATE_LENGTH"], p["FORCE_PLATE_WIDTH"]
    names = ("POSX_POSY", "NEGX_POSY", "NEGX_NEGY", "POSX_NEGY")
    measured = np.array([[p[f"FORCE_PLATE_CORNER_{c}_{a}"] for a in "XYZ"]
                         for c in names])
    nominal = np.array([[W / 2 if c.startswith("POSX") else -W / 2,
                         L / 2 if c.endswith("POSY") else -L / 2, 0.0]
                        for c in names])
    R, t = kabsch(nominal, measured)
    fitted = nominal @ R.T + t
    return dict(R=R, t=t, width=W, length=L, outline=fitted[:, :2],
                corner_miss_mm=np.linalg.norm((measured - fitted)[:, :2],
                                              axis=1),
                pitch_deg=float(np.degrees(np.arcsin(R[2, 1]))))


def boot_sole(v_local, pose):
    """The underside of the boot, in the foot frame, and its long axis.
    "Up" is world vertical seen from the foot (median over the trial);
    forward is the mesh's long axis, pointing away from the ankle."""
    R = pose[:, :3, :3]
    up = np.median(R[np.isfinite(R).all(axis=(1, 2))][:, 2, :], axis=0)
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
    idx = order[np.r_[True, key[order][1:] != key[order][:-1]]]
    idx = idx[height[idx] < height[idx].min() + SOLE_MAX_RISE_MM / 1000]
    return v_local[idx], long_axis


def walking_direction(pose, long_axis):
    """Horizontal unit vector the boots point along, median over the trial."""
    total = np.zeros(3)
    for b in LIMBS:
        R = pose[b][np.isfinite(pose[b][:, :3, :3]).all(axis=(1, 2)), :3, :3]
        d = R @ long_axis[b]
        d[:, 2] = 0
        total += np.median(d / np.linalg.norm(d, axis=1, keepdims=True), 0)
    return total / np.linalg.norm(total)


def fit_belt_surface(pose, soles, forward):
    """Belt height under a point = floor[foot] + slope * (distance along the
    walking direction). The DICE plates are pitched ~0.8 deg, 13 mm over a
    stance. Fitted to each frame's lowest sole point where it sits on the
    surface; one slope, one offset per boot to absorb a pose bias in either."""
    low_z, low_s = {}, {}
    for b in LIMBS:
        low_z[b] = np.full(len(pose[b]), np.nan)
        low_s[b] = np.full(len(pose[b]), np.nan)
        for sl, W in sole_chunks(pose[b], soles[b]):
            z = np.where(np.isfinite(W[..., 2]), W[..., 2], np.inf)
            pick = W[np.arange(len(W)), z.argmin(1)]
            low_z[b][sl] = pick[:, 2]
            low_s[b][sl] = pick[:, :2] @ forward[:2]
    floor = {b: float(np.nanpercentile(low_z[b], 25)) for b in LIMBS}
    slope, keep = 0.0, 0.020
    for _ in range(4):
        A, z = [], []
        for k, b in enumerate(LIMBS):
            h = low_z[b] - floor[b] - slope * low_s[b]
            m = np.isfinite(h) & (np.abs(h) < keep)
            cols = np.zeros((m.sum(), 3))
            cols[:, k], cols[:, 2] = 1, low_s[b][m]
            A.append(cols)
            z.append(low_z[b][m])
        a_l, a_r, slope = np.linalg.lstsq(np.vstack(A), np.concatenate(z),
                                          rcond=None)[0]
        floor, keep = {"L": float(a_l), "R": float(a_r)}, 0.008
    return dict(floor=floor, slope=float(slope), forward=forward)


def above_belt(W, surface, b):
    """Height of world points W (..., 3) above the belt surface."""
    s = W[..., :2] @ surface["forward"][:2]
    return W[..., 2] - surface["floor"][b] - surface["slope"] * s


def boot_contact(pose, soles, surface):
    """Per frame and boot: lowest sole point above the belt, how many sole
    vertices touch, and where the touching patch is (Theia xy)."""
    h = CONTACT_HEIGHT_MM / 1000
    height, n_touch, patch = {}, {}, {}
    for b in LIMBS:
        n = len(pose[b])
        height[b], n_touch[b] = np.full(n, np.nan), np.zeros(n, int)
        patch[b] = np.full((n, 2), np.nan)
        for sl, W in sole_chunks(pose[b], soles[b]):
            hz = above_belt(W, surface, b)
            touch = hz < h
            height[b][sl] = hz.min(1)
            n_touch[b][sl] = touch.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                patch[b][sl] = ((W[..., :2] * touch[..., None]).sum(1)
                                / touch.sum(1)[:, None])
    return height, n_touch, patch


def register_lab_to_theia(force100, cop100, pose, height, n_touch, patch,
                          belts_lab):
    """Plate-corner lab frame -> Theia, from single support.

    Rotation: the belt's direction of travel -- the plates' long axis in the
    lab, the stance boot's velocity in Theia. Translation: the CoP-to-patch
    offset per boot, averaged over the two boots so a medial/lateral bias
    that mirrors between them cancels."""
    P, Q, V_lab, V_theia, who = [], [], [], [], []
    for b in LIMBS:
        loaded = force100[b] > COP_MIN_FORCE_N
        free = force100[OTHER[b]] < FORCE_THRESHOLD_N
        v_cop = np.gradient(cop100[b], axis=0)
        for f in LIMBS:
            alone = ((n_touch[f] >= MIN_CONTACT_VERTICES)
                     & (height[OTHER[f]] > 2 * CONTACT_HEIGHT_MM / 1000))
            m = loaded & free & alone & np.isfinite(cop100[b][:, 0])
            inner = m & np.r_[False, m[:-1]] & np.r_[m[1:], False]
            P.append(cop100[b][m])
            Q.append(patch[f][m])
            V_lab.append(v_cop[inner])
            V_theia.append(np.gradient(pose[f][:, :2, 3], axis=0)[inner])
            who += [f] * int(m.sum())
    P, Q, who = np.concatenate(P), np.concatenate(Q), np.array(who)
    V_lab, V_theia = np.concatenate(V_lab), np.concatenate(V_theia)
    if len(P) < 200 or len(V_lab) < 100:
        raise RuntimeError(f"only {len(P)} single-support frames to register "
                           f"the plates on; set LAB_TO_THEIA")

    def unit(v):
        v = np.nansum(v, axis=0)
        return v / np.linalg.norm(v)

    u_cop, u_theia = unit(V_lab), unit(V_theia)
    axes = []
    for poly in belts_lab.values():
        edges = np.roll(poly, -1, axis=0) - poly
        e = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        axes.append(e / np.linalg.norm(e) * np.sign(e @ u_cop))
    u_lab = unit(np.array(axes))
    n_lab = np.array([-u_lab[1], u_lab[0]])
    n_theia = np.array([-u_theia[1], u_theia[0]])
    if np.corrcoef(P @ n_lab, Q @ n_theia)[0, 1] < 0:
        raise RuntimeError("the plate frame looks mirrored relative to Theia")
    phi = np.arctan2(u_theia[1], u_theia[0]) - np.arctan2(u_lab[1], u_lab[0])
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    offsets = Q - P @ R.T
    per_boot = [np.median(offsets[who == f], axis=0) for f in LIMBS
                if (who == f).sum() >= 50]
    t = np.mean(per_boot, axis=0) if per_boot else np.median(offsets, axis=0)
    A = np.column_stack([R, t])
    res = apply_2d(A, P) - Q
    side = np.abs(res @ n_theia) * 1000
    return A, dict(single_support_frames=int(len(P)),
                   side_residual_median_mm=float(np.median(side)),
                   side_residual_p90_mm=float(np.percentile(side, 90)),
                   along_residual_median_mm=float(
                       np.median(np.abs(res @ u_theia)) * 1000))


def boot_on_belts(pose, soles, surface, belts, cop_theia, margin):
    """Per frame, boot and belt: how many touching sole vertices are on the
    belt (inside its outline grown by `margin`), and the distance from the
    belt's CoP to the boot's touching patch."""
    h = CONTACT_HEIGHT_MM / 1000
    on, cop_gap = {f: {} for f in LIMBS}, {f: {} for f in LIMBS}
    for f in LIMBS:
        n = len(pose[f])
        for b in LIMBS:
            on[f][b] = np.zeros(n, int)
            cop_gap[f][b] = np.full(n, np.inf)
        for sl, W in sole_chunks(pose[f], soles[f]):
            touch = above_belt(W, surface, f) < h
            xy = W[..., :2]
            for b, poly in belts.items():
                d = inside_distance(poly, xy)
                on[f][b][sl] = (touch & (d >= -margin)).sum(1)
                c = cop_theia[b][sl]
                gap = np.where(touch, np.linalg.norm(xy - c[:, None, :],
                                                     axis=2), np.inf).min(1)
                cop_gap[f][b][sl] = np.where(np.isfinite(c[:, 0]), gap,
                                             np.inf)
    return on, cop_gap


# %% ==========================================================================
#  3. WHICH GRF EVENTS TO TRUST
# =============================================================================

def judge_contacts(contacts, valid, n_touch, on, cop_gap, force100):
    """Label every contact and decide, per event, whether to trust it.

    -> trusted events, and 'unverified' ones: a clean-looking contact whose
    event fell in a tracking dropout, so the mesh could not check it."""
    w = int(round(TRUST_WINDOW_S * KINEMATIC_RATE))
    n = len(valid["L"])
    down = {f: n_touch[f] >= MIN_CONTACT_VERTICES for f in LIMBS}
    trusted, unverified = [], []

    def event(c, limb, kind, label):
        return dict(t=c["start"] / STEP if kind == HS else c["stop"] / STEP,
                    limb=limb, event=kind, source="GRF", method="GRF",
                    grf_sample=c["start"] if kind == HS else c["stop"],
                    belt=c["belt"], contact_label=label)

    for c in contacts:
        b = c["belt"]
        r0 = int(round(c["start"] / STEP))
        r1 = min(int(round(c["stop"] / STEP)), n - 1)
        rows = np.arange(max(r0, 0), r1 + 1)
        c.update(foot="", label="", hs_trusted=False, to_trusted=False,
                 hs_reason="", to_reason="", cop_inside=np.nan)
        force_ok = {HS: c["complete_start"] and c["loading_s"] <= MAX_LOADING_S,
                    TO: c["complete_stop"]
                    and c["unloading_s"] <= MAX_UNLOADING_S}
        if r0 >= n or not len(rows):
            c["label"] = c["hs_reason"] = c["to_reason"] = "no_kinematics"
            continue
        if (valid["L"][rows] & valid["R"][rows]).mean() < 0.5:
            c["label"] = c["hs_reason"] = c["to_reason"] = "no_mesh"
            c["foot"] = b
            unverified += [event(c, b, k, "no_mesh") for k in (HS, TO)
                           if force_ok[k]]
            continue

        # which boot is on this belt, and is it also on the other one
        o = OTHER[b]
        touching = {f: on[f][b][rows] >= MIN_CONTACT_VERTICES for f in LIMBS}
        f = max(LIMBS, key=lambda k: touching[k].sum())
        g = OTHER[f]
        if not touching[f].any():
            c["label"] = c["hs_reason"] = c["to_reason"] = "no_foot_on_belt"
            continue
        on_both = on[f][o][rows] >= MIN_CONTACT_VERTICES
        if touching[g].sum() >= SHARED_MIN_FRAMES:
            label = "shared"
        elif on_both.sum() >= STRADDLE_MIN_FRAMES:
            label = "straddle"
        elif f != b:
            label = "crossover"
        else:
            label = "clean"
        alone = (force100[b][rows] > COP_MIN_FORCE_N) & ~touching[g]
        if alone.any():
            c["cop_inside"] = float(np.mean(
                cop_gap[f][b][rows][alone] < COP_TOLERANCE_MM / 1000))
        c["foot"], c["label"] = f, label

        # each event on its own, over +/- w frames around it
        for kind, centre, near in ((HS, r0, (0, w)), (TO, r1, (-w, 0))):
            win = np.arange(max(centre - w, 0), min(centre + w, n - 1) + 1)
            close = np.arange(max(centre + near[0], 0),
                              min(centre + near[1], n - 1) + 1)
            if not (c["complete_start"] if kind == HS else c["complete_stop"]):
                reason = "incomplete"
            elif kind == HS and c["loading_s"] > MAX_LOADING_S:
                reason = "slow_loading"
            elif kind == TO and c["unloading_s"] > MAX_UNLOADING_S:
                reason = "slow_unloading"
            elif not (valid[f][win] & valid[g][win]).all():
                reason = "no_mesh"
            elif (on[g][b][win] >= MIN_CONTACT_VERTICES).any():
                reason = "other_foot_on_belt"
            elif not down[f][close].any():
                reason = "foot_not_in_contact"
            elif (on[f][o][win] >= MIN_CONTACT_VERTICES).any():
                reason = "foot_on_other_belt"
            elif (np.isfinite(c["cop_inside"])
                  and c["cop_inside"] < COP_MIN_INSIDE):
                reason = "cop_outside_foot"
            else:
                reason = "trusted"
            key = "hs" if kind == HS else "to"
            c[f"{key}_reason"], c[f"{key}_trusted"] = reason, reason == "trusted"
            if reason == "trusted":
                trusted.append(event(c, f, kind, label))
            elif reason == "no_mesh" and label in ("clean", "crossover"):
                unverified.append(event(c, f, kind, "no_mesh"))

    for e in unverified:
        e["source"] = e["method"] = "GRF_unverified"
    trusted.sort(key=lambda e: e["t"])
    kept = []                          # one foot cannot land twice in 10 frames
    for e in trusted:
        if any(k["limb"] == e["limb"] and k["event"] == e["event"]
               and e["t"] - k["t"] < 10 for k in kept[-8:]):
            print(f"    ! duplicate trusted {e['limb']} {e['event']} at frame "
                  f"{e['t']:.1f}, dropped")
            continue
        kept.append(e)
    for e in kept + unverified:
        e["pos"] = POSITION[(e["limb"], e["event"])]
    return kept, unverified


# %% ==========================================================================
#  4. ZENI
# =============================================================================

def zeni_signals(points, forward):
    """Forward distance heel-to-pelvis (heel strike at its maximum) and
    minus toe-to-pelvis (toe-off at its maximum), per limb, filtered."""
    f = lambda x: lowpass(x, KINEMATIC_FILTER_HZ, KINEMATIC_RATE)  # noqa: E731
    pelvis = f(points["Pelvis_Position"])
    sig = {}
    for b in LIMBS:
        heel = f(points[f"{SIDE[b]}_Heel_Position"])
        toe = f(points[f"{SIDE[b]}_Toes_Position"])
        sig[(b, HS)] = (heel - pelvis) @ forward
        sig[(b, TO)] = -((toe - pelvis) @ forward)
    return sig


def peaks_between(x, lo, hi):
    """Sub-frame local maxima of x strictly inside (lo, hi), in frames."""
    i0, i1 = max(int(np.floor(lo)), 1), min(int(np.ceil(hi)), len(x) - 2)
    if i1 < i0:
        return np.empty(0)
    mid, left, right = x[i0:i1 + 1], x[i0 - 1:i1], x[i0 + 1:i1 + 2]
    idx = np.flatnonzero((mid >= left) & (mid > right))
    curve = left[idx] - 2 * mid[idx] + right[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(curve < 0, 0.5 * (left[idx] - right[idx]) / curve, 0)
    t = i0 + idx + np.clip(shift, -0.5, 0.5)
    t = t[np.isfinite(t)]
    return t[(t > lo) & (t < hi)]


def zeni_find(sig, offset, lo, hi, p):
    """The Zeni event nearest p in (lo, hi), offset-corrected, or None."""
    found = peaks_between(sig, lo + offset, hi + offset) - offset
    found = found[(found > lo) & (found < hi)]
    return float(found[np.argmin(abs(found - p))]) if len(found) else None


def typical_intervals(anchors):
    """Median frames from each place in the sequence to the next."""
    gaps = {p: [] for p in range(4)}
    for a, b in zip(anchors[:-1], anchors[1:]):
        if (b["pos"] - a["pos"]) % 4 == 1:
            gaps[a["pos"]].append(b["t"] - a["t"])
    if any(len(v) < 3 for v in gaps.values()):
        raise RuntimeError("too few trusted GRF events to learn the gait "
                           "timing -- check the plates and the registration")
    med = {p: np.median(v) for p, v in gaps.items()}
    return {p: float(np.median([g for g in v if g < 1.5 * med[p]]))
            for p, v in gaps.items()}


def zeni_offsets(sig, anchors, d):
    """Zeni's median timing offset from the trusted GRF events, per limb and
    event, found exactly as a missing event would be: in the window between
    its two trusted neighbours. Also returns the residuals left after the
    offset -- how far Zeni is likely to be off."""
    found = {key: [] for key in sig}
    for i in range(1, len(anchors) - 1):
        a, e, b = anchors[i - 1], anchors[i], anchors[i + 1]
        pa = (e["pos"] - 1) % 4
        if a["pos"] != pa or b["pos"] != (e["pos"] + 1) % 4:
            continue
        span, want = b["t"] - a["t"], d[pa] + d[e["pos"]]
        if not 0.6 < span / want < 1.5:
            continue
        t = zeni_find(sig[(e["limb"], e["event"])], 0.0,
                      a["t"] + MIN_SEPARATION, b["t"] - MIN_SEPARATION,
                      a["t"] + span * d[pa] / want)
        if t is not None:
            found[(e["limb"], e["event"])].append(t - e["t"])
    offset, report = {}, {}
    for (limb, kind), diffs in found.items():
        if len(diffs) < MIN_CALIBRATION_EVENTS:       # pool the two limbs
            diffs = found[("L", kind)] + found[("R", kind)]
        if not diffs:
            raise RuntimeError(f"Zeni found no {kind} near any trusted one")
        offset[(limb, kind)] = float(np.median(diffs))
        err = (np.array(diffs) - offset[(limb, kind)]) * 1000 / KINEMATIC_RATE
        report[f"{limb} {kind}"] = dict(
            n=len(diffs), offset_ms=offset[(limb, kind)] * 1000
            / KINEMATIC_RATE, mae_ms=float(np.mean(np.abs(err))),
            p95_ms=float(np.percentile(np.abs(err), 95)))
    return offset, report


# %% ==========================================================================
#  5. THE FULL SEQUENCE
# =============================================================================

def build_sequence(anchors, unverified, d, sig, offset, first, last):
    """Every event in walking order: trusted GRF kept, the gaps filled with
    Zeni, then unverified GRF, then (between trusted events) interpolation."""
    cycle = sum(d.values())

    def span(pos, steps):
        return sum(d[(pos + j) % 4] for j in range(steps))

    def fill(pos, lo, hi, p, allow_interpolation):
        limb, kind = SEQUENCE[pos]
        t = zeni_find(sig[(limb, kind)], offset[(limb, kind)], lo, hi, p)
        if t is not None:
            return dict(t=t, limb=limb, event=kind, source="kinematic",
                        method="zeni", pos=pos)
        grf = [e for e in unverified if e["pos"] == pos and lo < e["t"] < hi]
        if grf:
            return dict(min(grf, key=lambda e: abs(e["t"] - p)))
        if allow_interpolation and INTERPOLATE_MISSING:
            return dict(t=p, limb=limb, event=kind, source="interpolated",
                        method="interpolated", pos=pos)
        return None

    # trusted events that contradict the sequence are dropped: two events
    # that follow each other may be any distance apart, but one implying
    # missing events cannot come sooner than half the time they need
    clean = [anchors[0]]
    for e in anchors[1:]:
        a = clean[-1]
        k = (e["pos"] - a["pos"]) % 4 or 4
        gap = e["t"] - a["t"]
        if gap < MIN_SEPARATION or (k > 1 and gap < 0.5 * span(a["pos"], k)):
            print(f"    ! trusted {e['limb']} {e['event']} at frame "
                  f"{e['t']:.1f} is out of sequence, dropped")
            continue
        clean.append(e)

    # backwards from the first trusted event, while Zeni keeps finding them
    head, t, pos = [], clean[0]["t"], clean[0]["pos"]
    while True:
        pos = (pos - 1) % 4
        p = t - d[pos]
        lo, hi = p - 0.5 * d[(pos - 1) % 4], min(t - MIN_SEPARATION,
                                                 p + 0.5 * d[pos])
        if lo < first or (e := fill(pos, lo, hi, p, False)) is None:
            break
        head.append(e)
        t = e["t"]
    out = head[::-1]

    # between trusted events: the sequence says what is missing, and when
    for a, b in zip(clean[:-1], clean[1:]):
        out.append(a)
        gap = b["t"] - a["t"]
        k = (b["pos"] - a["pos"]) % 4 or 4
        steps = min((k + 4 * m for m in range(int(gap / cycle) + 2)),
                    key=lambda s: abs(np.log(gap / span(a["pos"], s))))
        scale = gap / span(a["pos"], steps)
        t_prev = a["t"]
        for j in range(1, steps):
            pos = (a["pos"] + j) % 4
            p = t_prev + d[(pos - 1) % 4] * scale
            lo = max(t_prev + MIN_SEPARATION,
                     p - 0.5 * d[(pos - 1) % 4] * scale)
            hi = min(b["t"] - MIN_SEPARATION, p + 0.5 * d[pos] * scale,
                     max(b["t"] - span(pos, steps - j) * scale * 0.5,
                         t_prev + 2 * MIN_SEPARATION))
            e = fill(pos, lo, hi, max(p, t_prev + MIN_SEPARATION), True)
            if e is not None:
                out.append(e)
                t_prev = e["t"]
    out.append(clean[-1])

    # forwards from the last trusted event
    t, pos = clean[-1]["t"], clean[-1]["pos"]
    while True:
        p = t + d[pos]
        lo = max(t + MIN_SEPARATION, p - 0.5 * d[pos])
        pos = (pos + 1) % 4
        hi = p + 0.5 * d[pos]
        if hi > last or (e := fill(pos, lo, hi, p, False)) is None:
            break
        out.append(e)
        t = e["t"]
    return out


def trim_and_flag(events):
    """Drop a limb's leading toe-off and trailing heel strike, then give each
    heel strike its stance and stride and flag outliers against the median."""
    if TRIM_TO_COMPLETE_STANCES:
        for b in LIMBS:
            mine = [e for e in events if e["limb"] == b]
            drop = ([mine[0]] if mine and mine[0]["event"] == TO else []) + \
                   ([mine[-1]] if mine and mine[-1]["event"] == HS else [])
            events = [e for e in events if not any(e is x for x in drop)]
    for e in events:
        e.update(stance_s=np.nan, stride_s=np.nan, flag="")
    for b in LIMBS:
        mine = [e for e in events if e["limb"] == b]
        for i, e in enumerate(mine):
            if e["event"] != HS:
                continue
            nxt_to = next((x for x in mine[i + 1:] if x["event"] == TO), None)
            nxt_hs = next((x for x in mine[i + 1:] if x["event"] == HS), None)
            if nxt_to:
                e["stance_s"] = (nxt_to["t"] - e["t"]) / KINEMATIC_RATE
            if nxt_hs:
                e["stride_s"] = (nxt_hs["t"] - e["t"]) / KINEMATIC_RATE
        for key in ("stance_s", "stride_s"):
            vals = [e[key] for e in mine if np.isfinite(e[key])]
            if not vals:
                continue
            med = np.median(vals)
            for e in mine:
                if np.isfinite(e[key]) and not (
                        FLAG_RATIO[0] * med <= e[key] <= FLAG_RATIO[1] * med):
                    e["flag"] = (e["flag"] + " " + key[:-2]).strip()
    return events


# %% ==========================================================================
#  ONE TRIAL, TOP TO BOTTOM
# =============================================================================

def process_trial(force_path, boots, pose_names, plates, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    stem = force_path.stem
    say(f"\n{stem}")

    # --- read -----------------------------------------------------------
    fz, cop = read_forces(force_path)
    frames, points, pose, toes = read_theia(
        find_file(KINEMATIC_FOLDER, stem, "_metrics.csv"), pose_names)
    n_rows, n_samples = len(frames), len(fz["L"])
    say(f"  force {n_samples / GRF_RATE:.1f} s, kinematics "
        f"{n_rows / KINEMATIC_RATE:.1f} s")
    if abs(n_samples / GRF_RATE - n_rows / KINEMATIC_RATE) > 1.0:
        say("  ! the recordings differ in length by over 1 s; they are "
            "assumed to start together")
    valid = {b: np.isfinite(pose[b][:, :3, :]).all(axis=(1, 2)) for b in LIMBS}

    # --- plates, CoP into the lab frame ----------------------------------
    plate_report, belts_lab = {}, {}
    for b in LIMBS:
        pl = plates[b]
        loaded = fz[b] > COP_MIN_FORCE_N + 20
        if COP_FRAME == "plate":
            on_plate = ((np.abs(cop[b][loaded, 0]) <= pl["width"] / 2)
                        & (np.abs(cop[b][loaded, 1]) <= pl["length"] / 2))
            cop[b] = (np.column_stack([cop[b], np.zeros(n_samples)])
                      @ pl["R"].T + pl["t"])[:, :2]
        else:
            on_plate = inside_distance(polygon_ccw(pl["outline"]),
                                       cop[b][loaded]) >= 0
        belts_lab[b] = polygon_ccw(pl["outline"] / 1000)
        plate_report[b] = dict(corner_miss_mm=pl["corner_miss_mm"].round(1)
                               .tolist(), pitch_deg=round(pl["pitch_deg"], 2),
                               loaded_cop_on_plate=float(on_plate.mean()))
        say(f"  {BELT[b]}: corners {plate_report[b]['corner_miss_mm']} mm off "
            f"the {pl['width']:g} x {pl['length']:g} rectangle, "
            f"{100 * on_plate.mean():.2f}% of loaded CoP on the plate")
        if on_plate.mean() < 0.99:
            say(f"  ! the {BELT[b]} CoP is not where the plate is; check "
                f"COP_FRAME")
    margin = (EXTRA_MARGIN_MM + max(pl["corner_miss_mm"].max()
                                   for pl in plates.values())) / 1000

    # --- 1. belt contacts ------------------------------------------------
    contacts, force100, cop100 = [], {}, {}
    rows_at = np.arange(n_rows) * STEP
    has_force = rows_at < n_samples
    for b in LIMBS:
        found, force, info = find_contacts(fz[b], b)
        contacts += found
        say(f"  {BELT[b]}: {len(found)} contacts, baseline "
            f"{info['baseline_n']:.1f} N, noise {info['noise_n']:.1f} N")
        force100[b] = np.full(n_rows, np.nan)
        force100[b][has_force] = force[rows_at[has_force]]
        cop100[b] = np.full((n_rows, 2), np.nan)
        cop100[b][has_force] = cop[b][rows_at[has_force]] / 1000
        cop100[b][~(force100[b] > COP_MIN_FORCE_N)] = np.nan
    contacts.sort(key=lambda c: c["start"])

    # --- 2. boots: surface, contact, registration, belts ------------------
    soles, long_axis = {}, {}
    for b in LIMBS:
        sole_v, long_axis[b] = boot_sole(boots[b], pose[b])
        toe_pose, toe, mtp = toe_hinge(
            pose[b], points[f"{SIDE[b]}_Toes_Position"], toes[b]["angle"],
            toes[b]["pose"], sole_v)
        soles[b] = dict(v=sole_v, toe=toe, toe_pose=toe_pose)
        say(f"  {SIDE[b]} boot: " + (
            f"{toe.sum()} of {len(sole_v)} sole points bend at the MTP"
            if toe_pose is not None else "no toe angle in the export, rigid"))
    forward = walking_direction(pose, long_axis)
    surface = fit_belt_surface(pose, soles, forward)
    say(f"  belt surface {surface['floor']['L'] * 1000:.1f} / "
        f"{surface['floor']['R'] * 1000:.1f} mm under the left / right boot, "
        f"rising {np.degrees(np.arctan(surface['slope'])):+.2f} deg forward")
    height, n_touch, patch = boot_contact(pose, soles, surface)

    if LAB_TO_THEIA is None:
        A, reg = register_lab_to_theia(force100, cop100, pose, height,
                                       n_touch, patch, belts_lab)
        say(f"  lab -> Theia (estimated): {json.dumps(np.round(A, 5).tolist())}")
        say(f"    CoP vs contact patch: side to side "
            f"{reg['side_residual_median_mm']:.1f} mm median "
            f"({reg['side_residual_p90_mm']:.1f} at 90%); "
            f"{np.degrees(np.arctan2(A[1, 0], A[0, 0])):+.2f} deg and "
            f"{np.linalg.norm(A[:, 2]) * 1000:.0f} mm from the identity")
    elif LAB_TO_THEIA == "identity":
        A, reg = np.array([[1.0, 0, 0], [0, 1.0, 0]]), {}
    else:
        A, reg = np.asarray(LAB_TO_THEIA, float), {}
    belts = {b: apply_2d(A, poly) for b, poly in belts_lab.items()}
    cop_theia = {b: apply_2d(A, cop100[b]) for b in LIMBS}
    on, cop_gap = boot_on_belts(pose, soles, surface, belts, cop_theia,
                                margin)

    # --- 3. trust ---------------------------------------------------------
    anchors, unverified = judge_contacts(contacts, valid, n_touch, on,
                                         cop_gap, force100)
    why = {}
    for c in contacts:
        for key in ("hs_reason", "to_reason"):
            if c[key] != "trusted":
                why[c[key]] = why.get(c[key], 0) + 1
    say(f"  trusted GRF events: {len(anchors)} of {2 * len(contacts)}; not: "
        + ", ".join(f"{v} {k}" for k, v in sorted(why.items(),
                                                   key=lambda x: -x[1])))

    # --- 4. Zeni, timed against the trusted events -------------------------
    sig = zeni_signals(points, forward)
    d = typical_intervals(anchors)
    offset, zeni_report = zeni_offsets(sig, anchors, d)
    for key, r in zeni_report.items():
        say(f"  Zeni {key:15s} offset {r['offset_ms']:+5.1f} ms, then "
            f"{r['mae_ms']:.1f} ms mean / {r['p95_ms']:.1f} ms p95 "
            f"from GRF (n={r['n']})")

    # --- 5. every event ---------------------------------------------------
    both = valid["L"] & valid["R"]
    events = build_sequence(anchors, unverified, d, sig, offset,
                            float(np.argmax(both)),
                            float(n_rows - 1 - np.argmax(both[::-1])))
    untrimmed = list(events)
    events = trim_and_flag(events)
    for e in events:
        e["frame_100hz"] = int(frames[int(np.clip(round(e["t"]), 0,
                                                  n_rows - 1))])

    # --- write --------------------------------------------------------------
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_FOLDER / f"{stem}_merged_events.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event_type", "support_limb", "frame_100hz", "source"])
        for e in events:
            w.writerow([e["event"], e["limb"], e["frame_100hz"], e["source"]])

    with open(OUTPUT_FOLDER / f"{stem}_event_qa.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event_type", "support_limb", "frame_100hz", "source",
                    "method", "frame_float", "grf_frame_1000hz", "belt",
                    "contact_label", "stance_s", "stride_s", "flag"])
        for e in events:
            w.writerow([e["event"], e["limb"], e["frame_100hz"], e["source"],
                        e["method"], f"{e['t']:.2f}", e.get("grf_sample", ""),
                        e.get("belt", ""), e.get("contact_label", ""),
                        fmt(e["stance_s"]), fmt(e["stride_s"]), e["flag"]])

    with open(OUTPUT_FOLDER / f"{stem}_grf_contacts.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["belt", "start_1000hz", "stop_1000hz", "start_frame_100hz",
                    "stop_frame_100hz", "duration_s", "peak_n", "loading_s",
                    "unloading_s", "foot", "label", "hs_trusted", "hs_reason",
                    "to_trusted", "to_reason", "cop_inside_fraction"])
        for c in contacts:
            r0 = int(np.clip(round(c["start"] / STEP), 0, n_rows - 1))
            r1 = int(np.clip(round(c["stop"] / STEP), 0, n_rows - 1))
            w.writerow([c["belt"], c["start"], c["stop"], frames[r0],
                        frames[r1], f"{(c['stop'] - c['start']) / GRF_RATE:.3f}",
                        f"{c['peak_n']:.1f}", f"{c['loading_s']:.3f}",
                        f"{c['unloading_s']:.3f}", c["foot"], c["label"],
                        c["hs_trusted"], c["hs_reason"], c["to_trusted"],
                        c["to_reason"], fmt(c["cop_inside"])])

    with open(OUTPUT_FOLDER / f"{stem}_event_summary.json", "w") as fh:
        json.dump(dict(plates=plate_report, margin_mm=margin * 1000,
                       lab_to_theia=A.tolist(), registration=reg,
                       belt_surface=dict(
                           floor_m=surface["floor"],
                           slope_deg=float(np.degrees(np.arctan(
                               surface["slope"])))),
                       walking_direction=forward.tolist(),
                       zeni_vs_grf=zeni_report), fh, indent=2)

    counts = {}
    for e in events:
        counts[e["method"]] = counts.get(e["method"], 0) + 1
    say(f"  events: {len(events)} ("
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + "); "
        f"{sum(bool(e['flag']) for e in events)} stance/stride outliers "
        f"flagged")
    say(f"  saved {out}")
    return dict(events=events, untrimmed=untrimmed, contacts=contacts,
                anchors=anchors, registration=(A, reg), surface=surface,
                height=height,
                zeni=zeni_report)


def fmt(x):
    return "" if x is None or not np.isfinite(x) else f"{x:.3f}"


# %% ==========================================================================
#  RUN
# =============================================================================

def main(stems=None):
    plates = {b: fit_plate(p) for b, p in read_plates(PLATE_FILE).items()}
    stems = stems or TRIALS
    paths = ([find_file(FORCE_FOLDER, s, ".csv") for s in stems] if stems
             else sorted(Path(FORCE_FOLDER).glob("*.csv")))
    bindings = {}                     # one read per participant
    for path in paths:
        try:
            where = binding_for(path.stem)
            if where not in bindings:
                bindings[where] = read_binding(where)
            boots, pose_names = bindings[where]
            process_trial(path, boots, pose_names, plates)
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            print(f"\n{path.stem}: SKIPPED -- {exc}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
