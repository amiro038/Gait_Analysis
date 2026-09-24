"""
One trial, loaded once, with everything the metrics need.

    the events       from detect_gait_events.py (event_qa.csv: sub-frame
                     times, how each was found, which belt carried it)
    the steps        one row per heel strike: its toe-off, the next heel
                     strike, and the other foot's events in between
    the kinematics   the Visual3D metrics export
    the boots        the participant's MOVE4D boot soles posed on the Theia
                     feet, bent at the MTP, with heights above the pitched
                     belt -- the SAME boot geometry detect_gait_events.py
                     uses, imported from it, so events and metrics agree
    the forces       GRF on the body per belt, baseline removed, turned into
                     Theia's axes through the plate corners and the
                     lab-to-Theia registration the event detection found
    the load         weighed on the plates in the quiet standing that opens
                     every trial: total weight minus body weight
    the system CoM   Theia's body CoM with the load added, the load riding
                     on the trunk where the standing centre of pressure puts
                     it
    the CoM velocity markers below 0.5 Hz, integrated GRF above

ROWS, NOT FRAME NUMBERS. Everything indexes the export by row (0 = first
frame). Events carry sub-frame rows (event_qa's frame_float), so stance and
stride times are not rounded to 10 ms. The force sample of row r is r * 10,
as in the event detection.

THE LOAD. Theia's Whole_body_COG is the body the markerless model sees; it
knows nothing of the mass of a pack or a vest. The plates do. During the
opening quiet standing the vertical GRF is the whole system's weight, and
the centre of pressure sits under the whole system's CoM (static
equilibrium), so

    system mass       m = W / g
    load mass         m_load = m - body mass (participants.csv)
    load, horizontal  (m * CoP - m_body * CoM_body) / m_load

The load's height cannot be seen standing still; it is placed at
Trunk_Position + LOAD_HEIGHT_M, and on the midline (LOAD_CENTRED) because
left-right the CoP places it only to the registration error x m / m_load. The load point is then carried with the
trunk (a frame from Low_Back to Neck), and the system CoM is the mass-
weighted mean of body and load. The GRF acts on the system, so the fused
velocity uses the system mass, and every margin of stability, pendulum
length and CoM work uses the system CoM.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter, sosfiltfilt

import detect_gait_events as dge

G = 9.81
FS = dge.KINEMATIC_RATE
FS_FORCE = dge.GRF_RATE
STEP = dge.STEP
LIMBS = dge.LIMBS
OTHER = dge.OTHER
SIDE = dge.SIDE
HS, TO = dge.HS, dge.TO

WARMUP_S = 60.0                # after the first event: belt ramp, settling
FORCE_FILTER_HZ = 50.0         # GRF for peaks and loading rate
ANTIALIAS_HZ = 40.0            # before taking the force down to 100 Hz
CROSSOVER_HZ = 0.5             # markers below, integrated GRF above
MARKER_SAVGOL = (11, 3)        # window, order: CoM smoothing + derivative
HANDRAIL_N = 15.0              # N above a rail's baseline that is contact

# quiet standing
QS_SEARCH_S = 30.0             # looked for in the first this-many seconds
QS_WINDOW_S = 2.0              # steadiest window of this length
QS_MAX_CV = 0.02               # total vertical force SD / mean
QS_MIN_SHARE = 0.15            # each belt carries at least this share
QS_MAX_FOOT_TRAVEL = 0.02      # m a heel may move across the window
LOAD_MIN_KG = 1.0              # less than this is no load
LOAD_HEIGHT_M = 0.0            # load CoM above Trunk_Position
# Packs and vests sit on the midline. Left-right, the standing CoP only
# locates the load to the plate registration's error times m / m_load
# (~5x: 4 mm of registration is 20 mm of load), so it is centred unless
# this is False
LOAD_CENTRED = True
LOAD_MAX_OFFSET_M = 0.40       # horizontal load-to-trunk distance believed


# %% ==========================================================================
#  THE EXPORT
# =============================================================================

def read_export(path):
    """A Visual3D metrics export -> (frames, {name: (F, k) float32},
    {name: component labels}). Trailing all-empty rows are dropped."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        head = [next(fh).rstrip("\r\n").split("\t") for _ in range(5)]
    data = pd.read_csv(path, sep="\t", header=None, skiprows=5,
                       encoding_errors="replace", low_memory=False)
    data = data.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    last = len(data)
    while last and np.isnan(data[last - 1, 1:]).all():
        last -= 1
    data = data[:last]
    groups, comps = {}, {}
    for i, (name, comp) in enumerate(zip(head[1], head[4])):
        if i == 0 or not name.strip():
            continue
        groups.setdefault(name, []).append(i)
        comps.setdefault(name, []).append(comp.strip().upper())
    columns = {n: data[:, idx] for n, idx in groups.items()}
    return data[:, 0].astype(int), columns, comps


def foot_pose(columns, name):
    """(F, 4, 4) from a *_Global_4x4 signal, rotation orthonormalised."""
    P = columns[name].astype(float).reshape(-1, 4, 4).copy()
    P[:, :3, :3] = dge.orthonormalise(P[:, :3, :3])
    P[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
    P[~np.isfinite(P[:, :3, :]).all(axis=(1, 2))] = np.nan
    return P


def at(x, rows):
    """x (F, ...) at fractional rows, by linear interpolation."""
    rows = np.asarray(rows, float)
    i = np.clip(np.floor(rows).astype(int), 0, len(x) - 2)
    w = np.clip(rows - i, 0, 1).reshape(rows.shape + (1,) * (x.ndim - 1))
    return x[i] * (1 - w) + x[i + 1] * w


def fill_gaps(x):
    """Linear fill of NaN, column by column; -> (filled, gap mask)."""
    x = np.array(x, float)
    gap = ~np.isfinite(x).all(axis=1) if x.ndim > 1 else ~np.isfinite(x)
    n = np.arange(len(x))
    flat = x.reshape(len(x), -1)
    for j in range(flat.shape[1]):
        ok = np.isfinite(flat[:, j])
        if ok.sum() >= 2:
            flat[~ok, j] = np.interp(n[~ok], n[ok], flat[ok, j])
    return flat.reshape(x.shape), gap


def widen(mask, reach):
    return np.convolve(mask, np.ones(2 * reach + 1), "same") > 0


# %% ==========================================================================
#  THE TRIAL
# =============================================================================

class Trial:
    """Attributes are filled by load_trial(); the methods are small views."""

    fs = FS

    def kin(self, name):
        """A three-component signal as (F, 3) float, or None."""
        if name not in self.columns or len(self.comps[name]) < 3:
            return None
        return self.columns[name][:, :3].astype(float)

    def kin_walker(self, name):
        """The same, in the walker's (ML, AP, VT): ML along self.lateral
        (to the left), AP along the walking direction, VT up."""
        x = self.kin(name)
        if x is None:
            return None
        R = np.column_stack([np.r_[self.lateral, 0], np.r_[self.forward, 0],
                             [0, 0, 1]])
        return x @ R

    def angle(self, name, axis=0):
        """One component of an angle signal (deg), or None."""
        if name not in self.columns:
            return None
        return self.columns[name][:, axis].astype(float)


def load_trial(stem, folders, participant, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    t = Trial()
    t.stem, t.participant = stem, participant
    say(f"\n{stem}")

    # --- events and kinematics --------------------------------------------
    t.frames, t.columns, t.comps = read_export(
        dge.find_file(folders["kinematic"], stem, "_metrics.csv"))
    t.n = len(t.frames)
    t.events = read_events(folders["events"], stem, t.frames)
    say(f"  {t.n / FS:.0f} s of kinematics, {len(t.events)} events "
        f"({(t.events['source'] == 'GRF').mean():.0%} trusted GRF)")

    summary_path = Path(folders["events"]) / f"{stem}_event_summary.json"
    summary = (json.loads(summary_path.read_text())
               if summary_path.exists() else {})

    # --- steps, and the belt they walk on -------------------------------------
    t.steps = build_steps(t.events)
    first = t.events["row"].min()
    t.steps["steady"] = t.steps["hs"] >= first + WARMUP_S * FS
    t.ankle = {b: t.kin(f"{SIDE[b]}_Ankle_Position") for b in LIMBS}
    t.heel = {b: t.kin(f"{SIDE[b]}_Heel_Position") for b in LIMBS}
    t.forward, t.belt_speed = belt_motion(t, say)
    t.lateral = np.array([-t.forward[1], t.forward[0]])

    # --- boots -------------------------------------------------------------
    boots, pose_names = dge.read_binding(
        Path(str(folders["binding"]).replace("{participant}",
                                             stem.split("_")[0])))
    t.pose = {b: foot_pose(t.columns, pose_names[b]) for b in LIMBS}
    t.boots = Boots(t, boots, summary)
    n_sole = {b: len(t.boots.sole[b]["v"]) for b in LIMBS}
    say(f"  boots: {n_sole['L']} / {n_sole['R']} sole points; "
        f"belt {t.boots.surface['floor']['L'] * 1000:.1f} / "
        f"{t.boots.surface['floor']['R'] * 1000:.1f} mm, "
        f"{np.degrees(np.arctan(t.boots.surface['slope'])):+.2f} deg")

    # --- forces ------------------------------------------------------------
    plates = {b: dge.fit_plate(p)
              for b, p in dge.read_plates(folders["plates"]).items()}
    A = np.asarray(summary.get("lab_to_theia", [[1, 0, 0], [0, 1, 0]]), float)
    if "lab_to_theia" not in summary:
        say("  ! no event summary: the plates are taken to be in Theia's "
            "frame")
    read_forces(t, dge.find_file(folders["force"], stem, ".csv"), plates, A,
                say)

    # --- load, system CoM, velocity -------------------------------------------
    t.body_mass = float(participant.get("body_mass_kg", np.nan))
    t.load = quiet_standing(t, say)
    t.mass = t.load["system_mass_kg"]
    system_com(t, say)
    fuse_velocity(t, say)
    return t


# %% ==========================================================================
#  EVENTS -> STEPS
# =============================================================================

def read_events(folder, stem, frames):
    """event_qa.csv if there is one (sub-frame rows, belt, contact), else
    merged_events.csv (whole frames, source only)."""
    folder = Path(folder)
    qa = folder / f"{stem}_event_qa.csv"
    if qa.exists():
        ev = pd.read_csv(qa)
        ev["row"] = ev["frame_float"].astype(float)
        ev["grf_sample"] = pd.to_numeric(ev["grf_frame_1000hz"],
                                         errors="coerce")
    else:
        ev = pd.read_csv(dge.find_file(folder, stem, "_merged_events.csv"))
        where = pd.Series(np.arange(len(frames)), index=frames)
        ev["row"] = where.reindex(ev["frame_100hz"]).to_numpy(float)
        ev["grf_sample"] = np.nan
        for c in ("method", "belt", "contact_label"):
            ev[c] = ""
        print(f"  ! no {qa.name}: event times are whole frames and the "
              f"kinetics cannot tell which belt a step was on")
    ev = ev.rename(columns={"support_limb": "limb"})
    ev["grf_sample"] = ev["grf_sample"].fillna(ev["row"] * STEP)
    return (ev.dropna(subset=["row"]).sort_values("row")
            .reset_index(drop=True))


def build_steps(ev):
    """One row per heel strike, rows as floats.

        hs, to, next_hs        this foot: contact, lift-off, next contact
        contra_to, contra_hs   the other foot's lift-off and contact that
                               follow hs (they close the initial double
                               support and open the terminal one)
        prev_contra_hs         the other foot's contact before hs: the step
                               that ENDS at hs runs from there
    """
    def after(rows, x):
        i = np.searchsorted(rows, x, side="right")
        return np.where(i < len(rows), rows[np.minimum(i, len(rows) - 1)],
                        np.nan)

    def before(rows, x):
        i = np.searchsorted(rows, x, side="left") - 1
        return np.where(i >= 0, rows[np.maximum(i, 0)], np.nan)

    get = {(b, k): ev[(ev["limb"] == b) & (ev["event_type"] == k)]
           for b in LIMBS for k in (HS, TO)}
    rows = {key: g["row"].to_numpy() for key, g in get.items()}
    out = []
    for b in LIMBS:
        o = OTHER[b]
        h = get[(b, HS)]
        s = pd.DataFrame(dict(limb=b, hs=h["row"].to_numpy()))
        s["to"] = after(rows[(b, TO)], s["hs"])
        s["next_hs"] = after(rows[(b, HS)], s["hs"])
        s.loc[s["to"] > s["next_hs"], "to"] = np.nan   # a missed toe-off
        s["contra_to"] = after(rows[(o, TO)], s["hs"])
        s["contra_hs"] = after(rows[(o, HS)], s["hs"])
        s["prev_contra_hs"] = before(rows[(o, HS)], s["hs"])
        s["hs_source"] = h["source"].to_numpy()
        s["hs_belt"] = h["belt"].to_numpy()
        s["hs_sample"] = h["grf_sample"].to_numpy()
        s["label"] = h["contact_label"].to_numpy()
        tos = get[(b, TO)].set_index("row")
        s["to_source"] = tos["source"].reindex(s["to"]).to_numpy()
        s["to_belt"] = tos["belt"].reindex(s["to"]).to_numpy()
        s["to_sample"] = tos["grf_sample"].reindex(s["to"]).to_numpy()
        out.append(s)
    steps = pd.concat(out).sort_values("hs").reset_index(drop=True)
    steps["hs_frame"] = np.round(steps["hs"]).astype(int)
    return steps


# %% ==========================================================================
#  BOOTS
# =============================================================================

class Boots:
    """The soles of the MOVE4D boots, posed per frame.

    Kept per frame, for every frame (cheap): the lowest sole point's height
    above the belt and which vertex it is, and the sole's extent along the
    belt's direction of travel and across it. Whole-sole positions are rebuilt on
    demand for the few frames a figure or a swing needs.
    """

    def __init__(self, trial, verts, summary):
        pose = trial.pose
        self.sole, self.long_axis = {}, {}
        for b in LIMBS:
            sole_v, self.long_axis[b] = dge.boot_sole(verts[b], pose[b])
            angle = trial.angle(f"{SIDE[b]}_Toes_Joint_Angle")
            toes4 = (foot_pose(trial.columns, f"{SIDE[b]}_Toes_Global_4x4")
                     if f"{SIDE[b]}_Toes_Global_4x4" in trial.columns
                     else None)
            toe_pose, toe, _ = dge.toe_hinge(
                pose[b], trial.kin(f"{SIDE[b]}_Toes_Position"), angle, toes4,
                sole_v)
            self.sole[b] = dict(v=sole_v, toe=toe, toe_pose=toe_pose)
        self.pose = pose
        s = summary.get("belt_surface")
        if s and "walking_direction" in summary:
            fwd = np.asarray(summary["walking_direction"], float)
            self.surface = dict(floor=s["floor_m"], forward=fwd,
                                slope=float(np.tan(np.radians(s["slope_deg"]))))
        else:
            fwd = dge.walking_direction(pose, self.long_axis)
            self.surface = dge.fit_belt_surface(pose, self.sole, fwd)
        fwd2, lat2 = trial.forward, trial.lateral
        n = len(pose["L"])
        self.low_z, self.low_idx, self.ext = {}, {}, {}
        for b in LIMBS:
            self.low_z[b] = np.full(n, np.nan)
            self.low_idx[b] = np.zeros(n, int)
            self.ext[b] = {k: np.full(n, np.nan) for k in
                           ("fwd_max", "fwd_min", "lat_max", "lat_min")}
            for sl, W in dge.sole_chunks(pose[b], self.sole[b]):
                h = self.height(b, W)
                hz = np.where(np.isfinite(h), h, np.inf)
                self.low_idx[b][sl] = hz.argmin(1)
                self.low_z[b][sl] = np.where(np.isfinite(h).any(1),
                                             hz.min(1), np.nan)
                pf, pl = W[..., :2] @ fwd2, W[..., :2] @ lat2
                self.ext[b]["fwd_max"][sl] = pf.max(1)
                self.ext[b]["fwd_min"][sl] = pf.min(1)
                self.ext[b]["lat_max"][sl] = pl.max(1)
                self.ext[b]["lat_min"][sl] = pl.min(1)

    def height(self, b, W):
        """Height above the belt surface of world points W (..., 3)."""
        return dge.above_belt(W, self.surface, b)

    def world(self, b, rows):
        """Sole vertex positions (len(rows), V, 3) at integer rows."""
        rows = np.asarray(rows, int)
        P = self.pose[b][rows]
        s = self.sole[b]
        part = dict(s, toe_pose=None if s["toe_pose"] is None
                    else s["toe_pose"][rows])
        return next(dge.sole_chunks(P, part, chunk=len(rows) + 1))[1]


# %% ==========================================================================
#  FORCES
# =============================================================================

def read_forces(t, path, plates, A, say):
    """GRF on the body per belt, in Theia's axes, at 1000 Hz.

    The export's forces and moments are in each plate's own frame, where
    the CoP is (-My/Fz, Mx/Fz) - offset (detect_gait_events.COP_FRAME).
    The plate's rotation from its fitted corners takes them to the lab,
    then the lab-to-Theia registration to Theia. Whether the export gives
    the force ON the plate or ON the body is not documented, so the sign is
    the one that makes the vertical force hold the walker up.
    """
    head = pd.read_csv(path, nrows=0).columns
    want = [f"{dge.BELT[b]}_{q}_{a}" for b in LIMBS
            for q, axes in (("Force", "XYZ"), ("Moment", "XYZ"), ("COP", "XY"))
            for a in axes]
    rails = [c for c in head if "handrail_Force" in c]
    data = pd.read_csv(path, usecols=[c for c in want + rails if c in head])
    t.grf, t.grf_raw, t.cop, t.free_moment, t.baseline = {}, {}, {}, {}, {}
    sos = butter(4, FORCE_FILTER_HZ, fs=FS_FORCE, output="sos")
    for b in LIMBS:
        belt, pl = dge.BELT[b], plates[b]
        col = lambda q: (data[f"{belt}_{q}"].to_numpy(float)  # noqa: E731
                         if f"{belt}_{q}" in data else None)
        F = np.column_stack([col(f"Force_{a}") for a in "XYZ"])
        base, _ = dge.belt_baseline(F[:, 2])
        loaded = sosfiltfilt(sos, F[:, 2]) - base > dge.FORCE_THRESHOLD_N
        F[:, 2] -= base
        F[:, :2] -= np.median(F[~loaded, :2], axis=0)
        lab = F @ pl["R"].T
        sign = 1.0 if np.mean(lab[loaded, 2]) > 0 else -1.0
        lab *= sign
        theia = np.column_stack([lab[:, :2] @ A[:, :2].T, lab[:, 2]])
        t.grf_raw[b] = theia
        t.grf[b] = sosfiltfilt(sos, theia, axis=0)
        t.baseline[b] = float(np.median(base))

        cop = np.column_stack([col("COP_X"), col("COP_Y")])
        cop_lab = (np.column_stack([cop, np.zeros(len(cop))]) @ pl["R"].T
                   + pl["t"])[:, :2] / 1000 if dge.COP_FRAME == "plate" \
            else cop / 1000
        cop_theia = dge.apply_2d(A, cop_lab)
        cop_theia[~(F[:, 2] > dge.COP_MIN_FORCE_N)] = np.nan
        t.cop[b] = cop_theia

        M = [col(f"Moment_{a}") for a in "XYZ"]
        if any(m is None for m in M):
            t.free_moment[b] = np.full(len(F), np.nan)
        else:
            Fz = np.where(loaded, F[:, 2] + base, np.nan)
            ox = np.nanmedian(cop[:, 0] + 1000 * M[1] / Fz)
            oy = np.nanmedian(cop[:, 1] - 1000 * M[0] / Fz)
            rx, ry = (cop[:, 0] - ox) / 1000, (cop[:, 1] - oy) / 1000
            fm = M[2] - (rx * F[:, 1] - ry * F[:, 0])
            t.free_moment[b] = np.where(loaded, fm, np.nan)
    t.n_force = len(data)
    t.handrail = np.zeros(len(data), bool)
    for c in rails:
        x = data[c].to_numpy(float)
        t.handrail |= np.abs(x - np.median(x)) > HANDRAIL_N
    say(f"  forces: {t.n_force / FS_FORCE:.0f} s, baselines "
        f"{t.baseline['L']:.0f} / {t.baseline['R']:.0f} N"
        + (f", handrail touched {t.handrail.mean():.1%} of the time"
           if t.handrail.any() else ""))


# %% ==========================================================================
#  BELT SPEED, LOAD, SYSTEM CoM, VELOCITY
# =============================================================================

def belt_motion(t, say, flat=(0.25, 0.55)):
    """The belt's direction of travel and speed, from the stance heels
    during foot-flat: they ride the belt, so their velocity IS the belt's.

    This, not the way the boots point, is the walking direction: boots toe
    out, and a degree or two of it would leak step length into step width.
    -> (forward unit vector in Theia xy, speed m/s)
    """
    vel = []
    for s in t.steps.itertuples():
        if not np.isfinite(s.to) or s.to <= s.hs:
            continue
        a = int(s.hs + flat[0] * (s.to - s.hs))
        b = int(s.hs + flat[1] * (s.to - s.hs))
        xy = t.heel[s.limb][a:b, :2]
        if b - a >= 5 and np.isfinite(xy).all():
            vel.append(np.polyfit(np.arange(len(xy)) / FS, xy, 1)[0])
    vel = np.array(vel)
    back = np.median(vel / np.linalg.norm(vel, axis=1, keepdims=True), 0)
    forward = -back / np.linalg.norm(back)
    speed = -vel @ forward
    say(f"  belt: {np.median(speed):.3f} m/s from {len(vel)} stance feet "
        f"(IQR {np.percentile(speed, 25):.3f}-{np.percentile(speed, 75):.3f}), "
        f"travel along ({forward[0]:+.3f}, {forward[1]:+.3f})")
    return forward, float(np.median(speed))


def trunk_frame(t):
    """(F, 3, 3) columns: to the right, forward, and up along
    Low_Back -> Neck (right-handed)."""
    lo, hi = t.kin("Low_Back_Position"), t.kin("Neck_Position")
    n = t.n
    up = np.tile([0.0, 0, 1], (n, 1))
    if lo is not None and hi is not None:
        u = hi - lo
        ok = np.isfinite(u).all(1)
        up[ok] = u[ok] / np.linalg.norm(u[ok], axis=1, keepdims=True)
    right = -np.r_[t.lateral, 0]
    x = right - (up @ right)[:, None] * up
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.stack([x, np.cross(up, x), up], axis=2)


def quiet_standing(t, say):
    """Weigh the walker plus load in the opening quiet standing.

    The steadiest QS_WINDOW_S before the first step, with both belts
    loaded and the heels still. -> dict describing it."""
    fz = {b: t.grf[b][:, 2] for b in LIMBS}
    total = fz["L"] + fz["R"]
    first = int(t.events["row"].min() * STEP)
    end = min(first, int(QS_SEARCH_S * FS_FORCE), len(total))
    win = int(QS_WINDOW_S * FS_FORCE)

    def heels_still(rows):
        """Both heels in the same place at the start and end of the window
        (medians over 0.2 s, so tracking noise does not count)."""
        k = FS // 5
        for b in LIMBS:
            h = t.heel[b][rows]
            travel = np.linalg.norm(np.nanmedian(h[-k:], 0)
                                    - np.nanmedian(h[:k], 0))
            if not travel < QS_MAX_FOOT_TRAVEL:
                return False
        return True

    best, best_cv = None, np.inf
    for s in range(0, max(end - win, 0), FS_FORCE // 10):
        seg = total[s:s + win]
        share = fz["L"][s:s + win] / seg
        rows = slice(s // STEP, (s + win) // STEP)
        cv = np.std(seg) / np.mean(seg) if np.mean(seg) > 200 else np.inf
        if (cv < min(best_cv, QS_MAX_CV) and share.min() > QS_MIN_SHARE
                and share.max() < 1 - QS_MIN_SHARE
                and heels_still(rows)):
            best, best_cv = s, cv
    steady = t.steps.loc[t.steps["steady"], "hs"]
    walk = slice(int(steady.min() * STEP), int(steady.max() * STEP)) \
        if len(steady) else slice(first, len(total))
    walking_weight = float(np.mean(total[walk]))

    out = dict(found=best is not None, start_s=np.nan, stop_s=np.nan,
               cv=best_cv, walking_weight_n=walking_weight,
               body_mass_kg=t.body_mass, load_kg=np.nan,
               load_offset_local=np.full(3, np.nan))
    if best is None:
        say("  ! no quiet standing found at the start: the system mass is "
            "taken from the mean vertical GRF while walking, and the load "
            "cannot be placed")
        out.update(weight_n=walking_weight,
                   system_mass_kg=walking_weight / G)
    else:
        sl = slice(best, best + win)
        W = float(np.mean(total[sl]))
        out.update(start_s=best / FS_FORCE, stop_s=(best + win) / FS_FORCE,
                   weight_n=W, system_mass_kg=W / G)
        with np.errstate(invalid="ignore"):
            cop = np.nansum([t.cop[b][sl] * fz[b][sl, None] for b in LIMBS],
                            axis=0) / total[sl, None]
        out["cop_xy"] = np.nanmean(cop, axis=0)
        out["rows"] = (best // STEP, (best + win) // STEP)
        say(f"  quiet standing {out['start_s']:.1f}-{out['stop_s']:.1f} s: "
            f"{W:.0f} N = {W / G:.1f} kg (walking mean "
            f"{walking_weight / W - 1:+.1%})")
        if abs(walking_weight / W - 1) > 0.02:
            say("  ! the walking mean GRF and the standing weight differ by "
                "over 2%: plate drift, or the standing was not still")
    m = out["system_mass_kg"]
    if np.isfinite(t.body_mass):
        out["load_kg"] = m - t.body_mass
        say(f"  body {t.body_mass:.1f} kg -> load {out['load_kg']:+.1f} kg")
        if out["load_kg"] < -LOAD_MIN_KG:
            say("  ! the plates weigh LESS than the body mass entered in "
                "participants.csv -- check the entry or the plate zero")
    else:
        say("  ! no body mass in participants.csv: the load is unknown and "
            "the CoM is Theia's body CoM")
    return out


def system_com(t, say):
    """Theia's body CoM with the load riding on the trunk."""
    body = t.kin("Whole_body_COG")
    t.com_body = body
    L = t.load
    t.com = body.copy()
    t.load_position = None
    if not (L["found"] and np.isfinite(L["load_kg"])
            and L["load_kg"] >= LOAD_MIN_KG):
        return
    trunk = t.kin("Trunk_Position")
    if trunk is None:
        say("  ! no Trunk_Position: the load cannot be carried; CoM is the "
            "body's")
        return
    a, b = L["rows"]
    m, ml, mb = t.mass, L["load_kg"], t.body_mass
    xy = (m * L["cop_xy"] - mb * np.nanmean(body[a:b, :2], 0)) / ml
    z = np.nanmean(trunk[a:b, 2]) + LOAD_HEIGHT_M
    R = trunk_frame(t)
    p = np.r_[xy, z]
    offset = np.nanmedian(np.einsum("fji,fj->fi", R[a:b],
                                    p - trunk[a:b]), axis=0)
    if LOAD_CENTRED:
        offset[0] = 0.0
    horizontal = np.hypot(*(p[:2] - np.nanmean(trunk[a:b, :2], 0)))
    if horizontal > LOAD_MAX_OFFSET_M:
        say(f"  ! the load would sit {horizontal * 1000:.0f} mm from the "
            f"trunk; not believed -- placed on the trunk instead")
        offset[:2] = 0.0
    L["load_offset_local"] = offset
    t.load_position = trunk + np.einsum("fij,j->fi", R, offset)
    t.com = (mb * body + ml * t.load_position) / m
    say(f"  load CoM {offset[1] * 1000:+.0f} mm forward of the trunk, "
        + ("centred left-right, " if LOAD_CENTRED else
           f"{offset[0] * 1000:+.0f} mm to the right, ")
        + f"{LOAD_HEIGHT_M * 1000:+.0f} mm up (assumed); system CoM "
        f"{np.nanmedian(np.linalg.norm(t.com - body, axis=1)) * 1000:.0f} mm "
        f"from the body's")


def complementary(v_markers, acc, fs=FS, crossover=None):
    """lowpass(v_markers) + highpass(trapezoidal integral of acc), the same
    second-order Butterworth on both, so nothing is counted twice."""
    v_f = np.zeros_like(acc)
    v_f[1:] = np.cumsum((acc[:-1] + acc[1:]) / 2, axis=0) / fs
    v_f -= v_f.mean(0)
    b, a = butter(2, crossover or CROSSOVER_HZ, fs=fs)
    return (filtfilt(b, a, v_markers, axis=0) + v_f
            - filtfilt(b, a, v_f, axis=0))


def fuse_velocity(t, say):
    """v = lowpass(d/dt markers) + highpass(integral of GRF / m).

    Same cutoff on both branches, so the two transfer functions sum to one.
    Frames where the CoM was not tracked come back NaN (widened by the
    differentiator's reach) rather than as interpolated data."""
    com, gap = fill_gaps(t.com)
    win, order = MARKER_SAVGOL
    v_mk = savgol_filter(com, win, order, deriv=1, delta=1 / FS, axis=0)

    total = t.grf_raw["L"] + t.grf_raw["R"]
    acc = (total - total.mean(0)) / t.mass
    aa = butter(4, ANTIALIAS_HZ, fs=FS_FORCE, output="sos")
    acc = sosfiltfilt(aa, acc, axis=0)
    rows = np.arange(t.n) * STEP
    a100 = np.full((t.n, 3), np.nan)
    have = rows < len(acc)
    a100[have] = acc[rows[have]]
    rail = np.zeros(t.n, bool)
    rail[have] = t.handrail[rows[have]]
    if (~have).mean() > 0.05:
        say(f"  ! {(~have).mean():.0%} of frames have no force: markers "
            f"alone there")
    a100, _ = fill_gaps(a100)

    # the force axes against the markers, 0.5-5 Hz where both are good
    band = butter(2, [0.5, 5.0], btype="band", fs=FS, output="sos")
    a_mk = savgol_filter(com, win, order, deriv=2, delta=1 / FS, axis=0)
    t.fusion_check = {}
    for k, name in enumerate(("X", "Y", "Z")):
        r = np.corrcoef(sosfiltfilt(band, a_mk[:, k]),
                        sosfiltfilt(band, a100[:, k]))[0, 1]
        t.fusion_check[name] = float(r)
        if k < 2 and r < -0.5:
            say(f"  ! GRF {name} runs against the CoM acceleration "
                f"(r = {r:.2f}): flipped. Check the force export's axes")
            a100[:, k] *= -1
            for b in LIMBS:
                t.grf[b][:, k] *= -1
                t.grf_raw[b][:, k] *= -1
    say("  GRF vs marker CoM acceleration (0.5-5 Hz): r = "
        + " / ".join(f"{v:.2f}" for v in t.fusion_check.values()))
    if min(t.fusion_check.values()) < 0.7:
        say("  ! weak agreement: the force axes or the mass are off, and "
            "the fused velocity inherits it")

    v = complementary(v_mk, a100)
    bad = widen(gap, win // 2)
    v[bad] = np.nan
    t.com_vel = v
    t.com_vel_markers = np.where(bad[:, None], np.nan, v_mk)
    t.com_vel_belt = v + np.r_[t.forward, 0] * t.belt_speed
    t.handrail_rows = rail
    d = 1000 * np.nanstd(v - v_mk, axis=0)
    say(f"  fused CoM velocity differs from markers alone by "
        f"{d[0]:.0f} / {d[1]:.0f} / {d[2]:.0f} mm/s RMS; belt-frame speed "
        f"{np.nanmean(t.com_vel_belt @ np.r_[t.forward, 0]):.3f} m/s")
