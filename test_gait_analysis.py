"""End-to-end test for gait_analysis.py on a synthetic trial.

synthetic_trial.py writes a trial whose answers are known: a walker of 80 kg
carrying a 20 kg load 150 mm behind the trunk, a quiet standing to start, a
belt at 1.3 m/s, GRF that is exactly the system CoM's dynamics, boots that
dip to a known clearance in mid-swing. detect_gait_events.py finds the
events, then the whole analysis runs on it, and the checks are that

  * the load is weighed and placed from the quiet standing, and the system
    CoM and its fused velocity match the truth,
  * belt speed, stance, step width and stride speed match the truth,
  * MFC matches the clearance built into every swing,
  * the margin of stability equals the one computed from the TRUE CoM and
    velocity on the same boots,
  * the kinetics add up: the vertical impulse carries the weight, braking
    cancels propulsion, the free moment built in is found, the limbs' CoM
    work nets to zero,
  * the summary has every family of metric, with intervals around the
    values, and all the files are written.

Run it with `python test_gait_analysis.py` from anywhere (about 2 minutes).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import detect_gait_events as dg  # noqa: E402
import synthetic_trial as syn  # noqa: E402
import gait_analysis as ga  # noqa: E402
G = ga.G

WORK = Path(tempfile.mkdtemp(prefix="gait_analysis_test_"))
truth = syn.make_trial(WORK, duration=200.0, special=False)
name = truth["name"]

dg.FORCE_FOLDER, dg.KINEMATIC_FOLDER = truth["force_dir"], truth["kin_dir"]
dg.OUTPUT_FOLDER, dg.PLATE_FILE = WORK / "events", truth["plates"]
binding = dg.HERE / "foot_mesh_binding.npz"
plates = {b: dg.fit_plate(p) for b, p in dg.read_plates(dg.PLATE_FILE).items()}
boots, pose_names = dg.read_binding(binding)
dg.process_trial(truth["force_dir"] / f"{name}.csv", boots, pose_names,
                 plates, verbose=False)

# a short trial, so shorter windows than a real one would use
ga.WARMUP_S = 10.0
ga.LDS_N_STRIDES = 50
ga.DFA_MIN_BOX = 8
ga.FOLDERS = dict(force=truth["force_dir"], kinematic=truth["kin_dir"],
                   events=WORK / "events", binding=binding,
                   plates=truth["plates"])
ga.OUTPUT_FOLDER = WORK / "analysis"
people = ga.read_participants(WORK / "participants.csv")
t, table = ga.analyse_trial(name, people["S01"], None)

failures = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def value(metric, limb="both", statistic="value"):
    r = table[(table["metric"] == metric) & (table["limb"] == limb)
              & (table["statistic"] == statistic)]
    return r.iloc[0] if len(r) else None


print("\nchecks")
s = t.steps[t.steps["steady"]]

# --- load, CoM, velocity ------------------------------------------------------
L = t.load
off = L["load_offset_local"]
true_off = truth["load_offset"]
check(abs(t.mass - truth["system_mass"]) < 0.5
      and abs(L["load_kg"] - truth["load_mass"]) < 0.5,
      f"weighed {t.mass:.2f} kg in quiet standing, load {L['load_kg']:.2f} kg "
      f"(true {truth['system_mass']:.0f} / {truth['load_mass']:.0f})")
check(abs(off[1] - true_off[1]) < 0.025,
      f"load placed {1000 * off[1]:+.0f} mm forward of the trunk (true "
      f"{1000 * true_off[1]:+.0f})")
err = np.linalg.norm(t.com - truth["com"], axis=1)
check(np.nanmedian(err) < 0.008,
      f"system CoM {1000 * np.nanmedian(err):.1f} mm from the truth (median); "
      f"Theia's body CoM is {1000 * np.nanmedian(np.linalg.norm(t.com_body - truth['com'], axis=1)):.0f} mm off")
ok = np.isfinite(t.com_vel).all(1)
v_err = 1000 * np.sqrt(np.mean((t.com_vel[ok] - truth["com_velocity"][ok]) ** 2, 0))
m_err = 1000 * np.sqrt(np.nanmean((t.com_vel_markers[ok]
                                   - truth["com_velocity"][ok]) ** 2, 0))
check(np.all(v_err < 25) and np.all(v_err <= m_err),
      f"fused CoM velocity error {v_err.round(1)} mm/s RMS, markers alone "
      f"{m_err.round(1)}")
check(abs(t.belt_speed - truth["belt_speed"]) < 0.013,
      f"belt speed {t.belt_speed:.3f} m/s (true {truth['belt_speed']})")

# --- spatiotemporal -------------------------------------------------------------


def true_stance(st, limb=None):
    return min(truth["stances"][limb or st.limb],
               key=lambda x: abs(x["hs"] * 100 - st.hs))


stance_err, width_err = [], []
for st in s.itertuples():
    mine = true_stance(st)
    other = [x for x in truth["stances"][ga.OTHER[st.limb]]
             if x["hs"] * 100 < st.hs][-1]
    stance_err.append(st.stance_s - (mine["to"] - mine["hs"]))
    width_err.append(st.step_width_m - abs(mine["x"] - other["x"]))
stance_err, width_err = 1000 * np.array(stance_err), 1000 * np.array(width_err)
check(abs(np.nanmean(stance_err)) < 2 and np.nanstd(stance_err) < 5,
      f"stance {np.nanmean(stance_err):+.1f} +- {np.nanstd(stance_err):.1f} ms "
      f"from the truth")
check(np.nanmedian(np.abs(width_err)) < 3,
      f"step width {1000 * s['step_width_m'].mean():.1f} mm, "
      f"{np.nanmedian(np.abs(width_err)):.1f} mm from the truth (median)")
check(abs(s["stride_speed_ms"].mean() - truth["belt_speed"]) < 0.01,
      f"stride speed over the belt {s['stride_speed_ms'].mean():.3f} m/s")

# --- clearance and stability ---------------------------------------------------
sw = np.linspace(0.2, 0.9, 2001)
mfc_err = []
for st in s.itertuples():
    mine = true_stance(st)
    if np.isfinite(st.mfc_m):
        mfc_err.append(st.mfc_m - syn.clearance_profile(sw, mine["mfc"]).min())
mfc_err = 1000 * np.array(mfc_err)
check(len(mfc_err) > 0.9 * len(s) and np.median(np.abs(mfc_err)) < 2.0,
      f"MFC found in {len(mfc_err)} of {len(s)} swings, "
      f"{np.median(mfc_err):+.1f} mm from the built-in clearance (median)")

# the margin again, from the TRUE CoM and velocity on the same boots
diff = []
fwd = np.r_[t.forward, 0]
for st in s.itertuples():
    if not np.isfinite(st.mos_ml_contact):
        continue
    r = int(np.ceil(st.hs))
    o = ga.OTHER[st.limb]
    side = np.sign((t.ankle[st.limb][r, :2] - t.ankle[o][r, :2]) @ t.lateral)
    edge = (t.boots.ext[st.limb]["lat_max"][r] if side > 0
            else -t.boots.ext[st.limb]["lat_min"][r])
    c = truth["com"][r]
    v = truth["com_velocity"][r] + fwd * truth["belt_speed"]
    w0 = np.sqrt(G / np.linalg.norm(c - t.ankle[st.limb][r]))
    diff.append(st.mos_ml_contact - (edge - side * ((c[:2] + v[:2] / w0)
                                                    @ t.lateral)))
diff = 1000 * np.abs(diff)
check(np.median(diff) < 3.0 and 0 < s["mos_ml_contact"].mean() < 0.2,
      f"ML margin at contact {1000 * s['mos_ml_contact'].mean():.0f} mm, "
      f"{np.median(diff):.1f} mm from the true-CoM margin (median)")
check((s["mos_ap_min"] < 0).mean() > 0.9,
      f"AP margin goes negative in single support in "
      f"{100 * (s['mos_ap_min'] < 0).mean():.0f}% of stances")

# --- kinetics ---------------------------------------------------------------------
k = s[s["kinetics_trusted"]]
check(len(k) > 0.9 * len(s), f"{len(k)} of {len(s)} steady stances trusted "
      f"for kinetics")
per_stride = (k["vertical_impulse_tw_s"] / k["stride_s"]).mean()
check(abs(per_stride - 0.5) < 0.015,
      f"each limb's vertical impulse carries {100 * per_stride:.1f}% of the "
      f"weight over a stride (50%)")
net = (k["propulsive_impulse_bw_s"] + k["braking_impulse_bw_s"]).mean()
check(abs(net) < 0.1 * k["propulsive_impulse_bw_s"].mean(),
      f"braking and propulsion cancel: net {net:+.4f} BW s")
peak_fz = k[["f1_tw", "f2_tw"]].max(axis=1) * L["weight_n"]
ratio = (k["free_moment_peak_nm"] / (0.005 * peak_fz)).median()
check(abs(ratio - 1) < 0.1, f"free moment {ratio:.2f} x the one built in")
work = (k["com_work_pos_j"] + k["com_work_neg_j"]).mean()
check(abs(work) < 0.1 * k["com_work_pos_j"].mean(),
      f"CoM work per stance {k['com_work_pos_j'].mean():.1f} J positive, "
      f"{k['com_work_neg_j'].mean():.1f} J negative, net {work:+.2f} J")

# --- the summary -------------------------------------------------------------------
m = value("step_width_m", "both", "mean")
check(m is not None and m["ci_low"] <= m["value"] <= m["ci_high"]
      and m["block"] >= 1,
      f"step width mean {m['value']:.4f} m, 95% CI [{m['ci_low']:.4f}, "
      f"{m['ci_high']:.4f}], block {m['block']:.0f}")
fam = {"dfa_alpha_": "DFA", "sample_entropy_": "sample entropy",
       "lds_lambda_S_": "LDS", "foot_placement_r2": "foot placement",
       "harmonic_ratio_": "harmonic ratio", "gem_": "GEM",
       "symmetry_angle_": "symmetry", "step_regularity": "regularity",
       "mse_area": "multiscale entropy"}
missing = [v for k_, v in fam.items()
           if not table["metric"].str.startswith(k_).any()]
check(not missing, f"summary has every family ({len(table)} rows)"
      + (f"; missing {missing}" if missing else ""))
d = table[table["metric"].str.startswith("dfa_alpha_")]
width = d["ci_high"] - d["ci_low"]
check(len(d) and (width > 0).all() and (width < 0.6).all(),
      f"{len(d)} DFA exponents with intervals {width.min():.2f}-"
      f"{width.max():.2f} wide")
lam = table[table["metric"].str.startswith("lds_lambda_S")]
check(len(lam) and np.isfinite(lam["ci_low"]).all(),
      f"{len(lam)} LDS exponents with bootstrap intervals")
out = WORK / "analysis"
files = [f"{name}_{x}" for x in ("steps.csv", "summary.csv", "waveforms.npz",
                                 "lds_windows.csv", "qa.png",
                                 "variability.png")]
check(all((out / f).exists() for f in files), "all the per-trial files")
built = ga.build_tables(out)
wide = pd.read_csv(out / "all_trials_metrics.csv")
dictionary = pd.read_csv(out / "metric_dictionary.csv")
missing = [c for c in ga.KEY_METRICS if c not in wide]
check(len(wide) == 1 and not missing
      and set(dictionary["column"]) == set(wide.columns[3:])
      and dictionary["methods_section"].notna().all(),
      f"one row per trial, {wide.shape[1] - 3} metric columns, each in the "
      f"dictionary with its methods section"
      + (f"; key columns missing {missing}" if missing else ""))
check(abs(wide.loc[0, "load_kg"] - t.load["load_kg"]) < 1e-9
      and abs(wide.loc[0, "step_width_m_L"]
              - s.loc[s["limb"] == "L", "step_width_m"].mean()) < 1e-9,
      "the table's values are the trial's")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}"
      f"  (work dir {WORK})")
sys.exit(1 if failures else 0)
