"""
Kinetics per stance, from the instrumented split belts.

ONLY TRUSTED STANCES. A stance is measured only when BOTH its heel strike
and its toe-off were trusted GRF events of the same belt contact (see
detect_gait_events.py): one boot on that belt, the other boot off it, the
CoP under the boot. That same record names the belt, so a clean crossover
step is read from the belt it actually landed on. Everything else is NaN,
not an estimate: a crossed or shared contact carries two feet's force.

    impulses       braking and propulsive (the AP force along the walking
                   direction, split by sign), vertical
    peaks          F1 (weight acceptance), trough, F2 (push-off), peak
                   braking and propulsive force
    loading rate   slope of the vertical force between 20% and 80% of F1
    CoP            range along and across the walking direction (15 Hz)
    free moment    peak |vertical torque| not explained by the horizontal
                   forces at the CoP
    CoM work       individual limbs method (Donelan, Kram & Kuo 2002): each
                   leg's GRF dotted with the system CoM velocity in the belt
                   frame, integrated over collision (hs -> other toe-off,
                   negative), rebound and preload (single support, positive
                   and negative) and push-off (other heel strike -> toe-off,
                   positive)

Forces are reported per body weight (_bw, the participant's own weight)
and per total weight (_tw, body + load, as weighed standing). Load raises
the first; the second shows whether the walker's pattern changed beyond
carrying more.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .trial import FS_FORCE, G, at

LOADING_BAND = (0.20, 0.80)
COP_FILTER_HZ = 15.0
WAVE_POINTS = 101


def trusted(st):
    return (st.hs_source == "GRF" and st.to_source == "GRF"
            and isinstance(st.hs_belt, str) and st.hs_belt in ("L", "R")
            and st.hs_belt == st.to_belt)


def phase_work(power, a, b):
    seg = power[max(a, 0):max(b, 0)]
    return (float(np.sum(np.clip(seg, 0, None)) / FS_FORCE),
            float(np.sum(np.clip(seg, None, 0)) / FS_FORCE))


def stance_kinetics(t):
    s = t.steps
    bw = t.body_mass * G if np.isfinite(t.body_mass) else np.nan
    tw = t.load["weight_n"]
    sos = butter(4, COP_FILTER_HZ, fs=FS_FORCE, output="sos")
    fwd = np.r_[t.forward, 0]
    rows = []
    t.waves = {"vgrf": [], "apgrf": [], "limb": [], "hs": []}
    for st in s.itertuples():
        r = {}
        rows.append(r)
        r["kinetics_trusted"] = trusted(st)
        if not r["kinetics_trusted"]:
            continue
        a, b = int(round(st.hs_sample)), int(round(st.to_sample))
        if b - a < 100 or b > t.n_force:
            continue
        F = t.grf[st.hs_belt][a:b]
        ap, vt = F @ fwd, F[:, 2]
        dt = 1 / FS_FORCE
        imp = dict(braking_impulse=np.sum(np.clip(ap, None, 0)) * dt,
                   propulsive_impulse=np.sum(np.clip(ap, 0, None)) * dt,
                   vertical_impulse=np.sum(vt) * dt)
        mid = len(vt) // 2
        i1 = int(np.argmax(vt[:mid]))
        i2 = mid + int(np.argmax(vt[mid:]))
        peaks = dict(f1=vt[i1], f2=vt[i2],
                     trough=vt[i1:i2 + 1].min() if i2 > i1 else np.nan,
                     peak_braking=-ap.min(), peak_propulsive=ap.max())
        lo = int(np.argmax(vt[:i1 + 1] >= LOADING_BAND[0] * vt[i1]))
        hi = int(np.argmax(vt[:i1 + 1] >= LOADING_BAND[1] * vt[i1]))
        rate = (np.polyfit(np.arange(lo, hi + 1) * dt, vt[lo:hi + 1], 1)[0]
                if hi - lo >= 3 else np.nan)
        # impulses in BW*s, forces in BW, loading rate in BW/s
        for name, v in (list(imp.items()) + list(peaks.items())
                        + [("loading_rate", rate)]):
            unit = ("_s" if "impulse" in name
                    else "_per_s" if name == "loading_rate" else "")
            r[f"{name}_bw{unit}"] = v / bw
            r[f"{name}_tw{unit}"] = v / tw

        cop = t.cop[st.hs_belt][a:b]
        cop = cop[np.isfinite(cop).all(1)]        # where the foot is loaded
        if len(cop) > 30:
            cop = sosfiltfilt(sos, cop, axis=0)
            r["cop_ap_range_mm"] = 1000 * np.ptp(cop @ t.forward)
            r["cop_ml_range_mm"] = 1000 * np.ptp(cop @ t.lateral)
        fm = t.free_moment[st.hs_belt][a:b]
        fm = fm[np.isfinite(fm)]
        if len(fm) > 30:
            r["free_moment_peak_nm"] = float(np.max(np.abs(
                sosfiltfilt(sos, fm))))

        # individual limbs CoM work
        v = at(t.com_vel_belt, np.arange(a, b) / 10.0)
        if np.isfinite(v).all():
            power = np.sum(F * v, axis=1)
            k = lambda row: int(round(row * 10)) - a  # noqa: E731
            ok = st.hs < st.contra_to < st.contra_hs < st.to
            r["com_work_pos_j"], r["com_work_neg_j"] = phase_work(power, 0,
                                                                  len(power))
            if ok:
                _, r["collision_j"] = phase_work(power, 0, k(st.contra_to))
                r["rebound_j"], r["preload_j"] = phase_work(
                    power, k(st.contra_to), k(st.contra_hs))
                r["push_off_j"], _ = phase_work(power, k(st.contra_hs),
                                                len(power))
            for key in [x for x in list(r) if x.endswith("_j")]:
                r[key.replace("_j", "_j_kg")] = r[key] / t.mass

        pct = np.linspace(0, 1, WAVE_POINTS)
        x = np.linspace(0, 1, len(vt))
        t.waves["vgrf"].append(np.interp(pct, x, vt / tw))
        t.waves["apgrf"].append(np.interp(pct, x, ap / tw))
        t.waves["limb"].append(st.limb)
        t.waves["hs"].append(st.hs)
    keys = sorted({k for r in rows for k in r})
    for k in keys:
        s[k] = [r.get(k, np.nan) for r in rows]
    s["kinetics_trusted"] = s["kinetics_trusted"].fillna(False).astype(bool)
    return s
