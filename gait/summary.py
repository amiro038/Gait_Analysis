"""
From a trial's step table to one long table of results.

Every row is one number: trial, metric, limb (L, R or both), statistic,
value, its 95% confidence interval, how many strides it rests on, the
bootstrap block length, and a note. One row per number keeps the table the
same shape for every trial, so the trials stack into all_trials_summary.csv
and go straight into a mixed model (participant as a random effect,
condition as fixed) without reshaping.
"""

import numpy as np
import pandas as pd

from . import lds, uncertainty as unc, variability as var
from .trial import OTHER, at

# per-step metrics summarised as mean, SD and CV (steady walking only)
STEP_METRICS = [
    "stride_s", "stance_s", "swing_s", "step_s", "cadence_spm",
    "stance_pct", "double_support_pct", "single_support_pct",
    "step_length_m", "step_width_m", "stride_length_m", "stride_speed_ms",
    "walk_ratio",
    "mos_ml_contact", "mos_ml_min", "mos_ap_contact", "mos_ap_min",
    "mfc_m", "moi_peak_mm", "tri_s",
    "braking_impulse_bw_s", "propulsive_impulse_bw_s", "vertical_impulse_bw_s",
    "f1_bw", "trough_bw", "f2_bw", "loading_rate_bw_per_s",
    "f1_tw", "f2_tw", "propulsive_impulse_tw_s", "braking_impulse_tw_s",
    "peak_braking_bw", "peak_propulsive_bw",
    "cop_ap_range_mm", "cop_ml_range_mm", "free_moment_peak_nm",
    "collision_j_kg", "rebound_j_kg", "preload_j_kg", "push_off_j_kg",
    "com_work_pos_j_kg", "com_work_neg_j_kg",
    "trunk_lean_mean_deg", "trunk_lean_rom_deg", "pelvis_tilt_mean_deg",
    "hip_rom_deg", "knee_rom_deg", "ankle_rom_deg",
]
# stride series for DFA and entropy (one limb's consecutive strides)
SERIES = [
    "stride_s", "stance_s", "swing_s", "double_support_pct",
    "step_length_m", "step_width_m", "stride_length_m", "stride_speed_ms",
    "mfc_m", "mos_ml_contact", "mos_ml_min", "mos_ap_contact",
    "propulsive_impulse_bw_s", "braking_impulse_bw_s", "f1_bw",
    "trunk_lean_mean_deg",
]
SYMMETRY = ["stance_s", "swing_s", "step_length_m", "mfc_m",
            "mos_ml_contact", "propulsive_impulse_bw_s", "f1_bw", "tri_s"]


def row(metric, value, limb="both", statistic="value", ci=(np.nan, np.nan),
        n=np.nan, block=np.nan, note=""):
    return dict(metric=metric, limb=limb, statistic=statistic,
                value=float(value) if value is not None else np.nan,
                ci_low=ci[0], ci_high=ci[1], n=n, block=block, note=note)


def steady_steps(t, limb=None):
    s = t.steps[t.steps["steady"]]
    if limb is not None:
        s = s[s["limb"] == limb]
    return s.sort_values("hs")


def descriptors(t):
    ev = t.events
    s = steady_steps(t)
    L = t.load
    out = [row("belt_speed_ms", t.belt_speed),
           row("system_mass_kg", t.mass,
               note="weighed in quiet standing" if L["found"]
               else "mean walking GRF (no quiet standing found)"),
           row("load_kg", L["load_kg"]),
           row("quiet_standing_cv", L["cv"]),
           row("events_trusted_grf_pct", 100 * (ev["source"] == "GRF").mean()),
           row("events_zeni_pct", 100 * (ev["source"] == "kinematic").mean()),
           row("events_interpolated_pct",
               100 * (ev["source"] == "interpolated").mean()),
           row("steps_steady", len(s)),
           row("stances_kinetics_pct", 100 * s["kinetics_trusted"].mean()),
           row("swings_without_mfc_pct",
               100 * (s["mfc_minima"] == 0).mean()),
           row("mfc_below_belt_pct", 100 * (s["mfc_m"] < 0).mean()),
           row("handrail_steps_pct", 100 * s["handrail"].mean())]
    for axis, r in t.fusion_check.items():
        out.append(row(f"grf_vs_marker_acc_r_{axis}", r))
    for k, v in zip("xyz", L["load_offset_local"]):
        out.append(row(f"load_offset_{k}_m", v,
                       note="trunk frame: x right, y forward, z up"))
    return out


def step_summaries(t):
    out = []
    for limb in ("L", "R", "both"):
        s = steady_steps(t, None if limb == "both" else limb)
        for m in STEP_METRICS:
            if m not in s:
                continue
            res = unc.series_summary(s[m].to_numpy(float),
                                     tag=f"{t.stem}{m}{limb}")
            for stat, r in res.items():
                if np.isfinite(r["value"]):
                    out.append(row(m, r["value"], limb, stat,
                                   (r["ci_low"], r["ci_high"]), r["n"],
                                   r["block"]))
    return out


def symmetry(t):
    out = []
    for m in SYMMETRY:
        if m not in t.steps:
            continue
        mean = {b: steady_steps(t, b)[m].mean() for b in ("L", "R")}
        if all(np.isfinite(v) and v != 0 for v in mean.values()):
            out.append(row(f"symmetry_angle_{m}",
                           var.symmetry_angle(mean["L"], mean["R"]),
                           statistic="pct",
                           note="0 = symmetric, positive = right larger"))
    return out


def series_for(t, limb, n):
    s = steady_steps(t, limb)
    return s.iloc[:n] if n else s


def variability(t, n):
    out = []
    for limb in ("L", "R"):
        s = series_for(t, limb, n)
        cols = [c for c in SERIES if c in s]
        for r in var.stride_series_metrics(s[cols], tag=f"{t.stem}{limb}"):
            if r["metric"] == "dfa_alpha":
                note = (f"boxes {r['boxes']}, R2 {r['r2']:.3f}, GPH "
                        f"{r['gph']:.2f}, shuffled "
                        f"{r['shuffled']:.2f}, bias {r['bias']:+.3f}"
                        + (", metronome-paced" if r["cued"] else ""))
                out.append(row(f"dfa_alpha_{r['series']}", r["value"], limb,
                               ci=(r["ci_low"], r["ci_high"]), n=r["n"],
                               note=note))
            else:
                sweep = ", ".join(f"{k} {r[k]:.3f}" for k in r
                                  if k.startswith("r0"))
                out.append(row(f"sample_entropy_{r['series']}", r["value"],
                               limb, n=r["n"],
                               note=f"m {var.ENT_M}, r {var.ENT_R}; sweep "
                                    + sweep))

        # goal-equivalent manifold, on the same strides
        g = var.gem_decompose(s["stride_s"].to_numpy(float),
                              s["stride_length_m"].to_numpy(float))
        if g["n"] >= 4 * var.DFA_MIN_BOX:
            for comp in ("parallel", "perpendicular"):
                res = unc.series_summary(g[comp], ("sd",),
                                         tag=f"{t.stem}gem{comp}{limb}")
                r = res["sd"]
                out.append(row(f"gem_{comp}_sd_pct", 100 * r["value"], limb,
                               ci=(100 * r["ci_low"], 100 * r["ci_high"]),
                               n=r["n"], block=r["block"],
                               note="% of the mean stride"))
                out.append(row(f"gem_{comp}_lag1", g[f"{comp}_lag1"], limb,
                               n=g["n"]))
                a = var.dfa_alpha(g[comp])
                lo, hi, _, _ = unc.dfa_interval(a, g["n"], var.dfa_alpha,
                                                tag=f"{t.stem}{comp}{limb}")
                out.append(row(f"gem_{comp}_dfa_alpha", a, limb, ci=(lo, hi),
                               n=g["n"]))
        out.extend(foot_placement(t, limb, s))
    return out


def foot_placement(t, limb, s):
    """Where the other foot lands across the walking direction, predicted
    from the CoM's position and velocity at this foot's midstance, all
    relative to this (stance) ankle."""
    s = s[np.isfinite(s["to"]) & np.isfinite(s["contra_hs"])]
    if len(s) < 20:
        return []
    lat = np.r_[t.lateral, 0]
    mid = (s["hs"] + 0.5 * (s["to"] - s["hs"])).to_numpy()
    ank = at(t.ankle[limb], mid)
    z = (at(t.com, mid) - ank) @ lat
    v = at(t.com_vel, mid) @ lat
    y = (at(t.heel[OTHER[limb]], s["contra_hs"].to_numpy()) - ank) @ lat
    fit = var.foot_placement_model(z, v, y)
    ok = np.isfinite(z) & np.isfinite(v) & np.isfinite(y)
    z, v, y = z[ok], v[ok], y[ok]
    block = unc.optimal_block_length(y)
    ci = unc.bootstrap_rows(len(y), lambda i: var.foot_placement_model(
        z[i], v[i], y[i])["r2"], block, n_boot=400, tag=f"{t.stem}fp{limb}")
    return [row("foot_placement_r2", fit["r2"], limb, ci=ci, n=fit["n"],
                block=block,
                note=f"gains {fit['gain_position']:+.2f} (position), "
                     f"{fit['gain_velocity']:+.3f} s (velocity)")]


def trunk(t, limb, n):
    sig = var.trunk_signal(t)
    if sig is None:
        return [], None
    s = series_for(t, limb, n)
    s = s[np.isfinite(s["next_hs"])]
    strides = list(zip(s["hs"], s["next_hs"]))
    per, values, mse = var.trunk_metrics(t, sig, strides)
    out = [row(k, v, note=f"from {sig['name']}") for k, v in values.items()]
    for col in [c for c in per if c.startswith(("hr_", "ihr_"))]:
        res = unc.series_summary(per[col].to_numpy(float), ("mean",),
                                 tag=f"{t.stem}{col}")["mean"]
        out.append(row(f"harmonic_ratio_{col}", res["value"], limb, "mean",
                       (res["ci_low"], res["ci_high"]), res["n"],
                       res["block"],
                       note=f"per stride from {sig['name']}, "
                            f"k^{sig['power']} weighting"))
        out.append(row(f"harmonic_ratio_{col}", per[col].median(), limb,
                       "median", n=len(per)))
    return out, mse


def run_lds(t, limb, n):
    hs = series_for(t, limb, None)["hs"].to_numpy()
    summ, windows, curves = lds.trial_lds(t, hs, tag=t.stem)
    out = [row(r["metric"] + "_" + r["series"], r["value"], limb,
               ci=(r["ci_low"], r["ci_high"]), n=r["n"], block=r["block"],
               note=f"mean of {r['windows']} windows of {r['n']} strides"
                    + (", primary" if r["primary"] else ""))
           for r in summ]
    return out, pd.DataFrame(windows), curves


def summarise(t, series_length, lds_limb="R", run_lds_=True):
    """-> (long summary DataFrame, extras dict for files and figures)"""
    rows = descriptors(t) + step_summaries(t) + symmetry(t)
    rows += variability(t, series_length)
    trunk_rows, mse = trunk(t, lds_limb, series_length)
    rows += trunk_rows
    extras = dict(mse=mse, lds_windows=pd.DataFrame(), lds_curves={})
    if run_lds_:
        lds_rows, extras["lds_windows"], extras["lds_curves"] = run_lds(
            t, lds_limb, series_length)
        rows += lds_rows
    out = pd.DataFrame(rows)
    out.insert(0, "trial", t.stem)
    parts = t.stem.split("_")
    out.insert(1, "participant", parts[0])
    out.insert(2, "condition", parts[1] if len(parts) > 1 else "")
    out["series_length"] = series_length or np.nan
    return out, extras
