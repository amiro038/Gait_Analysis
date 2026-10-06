"""End-to-end test for Gait_analysis_all_metrics.py (version 4) on a
synthetic trial laid out exactly as the Box folders are:

    DICE_Treadmill/
        FP_renamed/{trial}.csv
        Theia_csv_outputs/{trial}_metrics.csv
        gait_event_outputs/{trial}_merged_events.csv (+ _event_qa, _summary)
        Foot bindings/S01_foot_mesh_binding.npz
        force_plates_DICE_treadmill.txt
        participants.csv

synthetic_trial.py writes a trial whose answers are known (80 kg walker,
20 kg load, belt at 1.3 m/s, boots that dip to ~20 mm in mid-swing). Then
Theia's toe angle is CORRUPTED: 25 deg of toe flexion during every swing,
while the boot itself stays rigid. A toe hinge driven by that angle (version
3) pushes the toe cap through the belt; the contact-constrained hinge of
version 4 must ignore it. The checks:

  * belt speed, load, stride time, stance, step width, stride speed: truth
  * MFC within 4 mm of the clearance built into every swing, and no swing
    through the belt, despite the corrupted toe angle
  * the toe-angle hinge DOES go through the belt on the same data (so the
    test would catch a regression to it)
  * every swing gets a trip-risk window, and TRI is in Schulz's range
  * the vertical impulse carries the weight; outputs are written

Run it with `python test_gait_analysis_all_metrics.py` (about 4 minutes).
"""
import io
import runpy
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import detect_gait_events as dg  # noqa: E402
import synthetic_trial as syn  # noqa: E402

TOE_FLEXION_DEG = 25.0
WORK = Path(tempfile.mkdtemp(prefix="all_metrics_test_"))
base = WORK / "DICE_Treadmill"
base.mkdir()
truth = syn.make_trial(base, duration=300.0, special=False)
name = truth["name"]
(base / "Foot bindings").mkdir()
shutil.copy(HERE / "foot_mesh_binding.npz",
            base / "Foot bindings" / "S01_foot_mesh_binding.npz")
shutil.copy(truth["plates"], base / "force_plates_DICE_treadmill.txt")

# --- corrupt Theia's toe angle: flexion through every swing -------------------
kin = truth["kin_dir"] / f"{name}_metrics.csv"
lines = kin.read_text().split("\n")
names = lines[1].split("\t")
table = np.loadtxt(io.StringIO("\n".join(lines[5:])), delimiter="\t")
for limb, side in (("L", "Left"), ("R", "Right")):
    col = names.index(f"{side}_Toes_Joint_Angle")
    swing = (~truth["on_belt"][limb]) & (truth["t_k"] > truth["standing"])
    k = np.ones(10) / 10
    shape = np.convolve(np.convolve(swing.astype(float), k, "same"), k, "same")
    table[:, col] -= TOE_FLEXION_DEG * np.clip(shape, 0, 1)
np.savetxt(kin, table, delimiter="\t", header="\n".join(lines[:5]),
           comments="", fmt=["%d"] + ["%.6f"] * (table.shape[1] - 1))

# --- step 1, then the analysis ---------------------------------------------------
dg.FORCE_FOLDER, dg.KINEMATIC_FOLDER = truth["force_dir"], truth["kin_dir"]
dg.OUTPUT_FOLDER = base / "gait_event_outputs"
dg.PLATE_FILE = base / "force_plates_DICE_treadmill.txt"
plates = {b: dg.fit_plate(p) for b, p in dg.read_plates(dg.PLATE_FILE).items()}
boots, pose_names = dg.read_binding(base / "Foot bindings"
                                    / "S01_foot_mesh_binding.npz")
dg.process_trial(truth["force_dir"] / f"{name}.csv", boots, pose_names,
                 plates, verbose=False)

events_csv = dg.OUTPUT_FOLDER / f"{name}_merged_events.csv"
sys.argv = [str(HERE / "Gait_analysis_all_metrics.py"), str(events_csv),
            "--no-figures"]
log = io.StringIO()
with redirect_stdout(log):
    ns = runpy.run_path(sys.argv[0], run_name="__main__")
(WORK / "log.txt").write_text(log.getvalue())

failures = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


g = ns["gait_event_data"]
st = g[g["steady"]]
check(abs(ns["belt_speed"] - truth["belt_speed"]) < 0.01,
      f"belt speed {ns['belt_speed']:.4f} m/s (truth {truth['belt_speed']})")
check(abs(ns["load_kg"] - truth["load_mass"]) < 1.0,
      f"load {ns['load_kg']:.2f} kg (truth {truth['load_mass']})")
check(abs(st["stride_time"].mean() - truth["cycle"]) < 0.01,
      f"stride time {st['stride_time'].mean():.4f} s (truth ~{truth['cycle']})")
stance = 100 * st["stance_time"].mean() / st["stride_time"].mean()
check(abs(stance - 100 * truth["stance_frac"]) < 2,
      f"stance {stance:.1f}% (truth {100 * truth['stance_frac']:.0f}%)")
check(abs(st["step_width"].mean() - 0.20) < 0.02,
      f"step width {st['step_width'].mean():.4f} m (truth 0.20)")
check(abs(st["stride_velocity"].mean() / truth["belt_speed"] - 1) < 0.02,
      f"stride speed over the belt {st['stride_velocity'].mean():.4f} m/s")

mfc_true = np.mean([s["mfc"] for b in "LR" for s in truth["stances"][b]
                    if not s["standing"]])
mfc = st["minimum_foot_clearance"]
check(abs(mfc.mean() - mfc_true) < 0.004 and mfc.notna().mean() > 0.95,
      f"MFC {1000 * mfc.mean():.1f} mm (truth {1000 * mfc_true:.1f}) in "
      f"{100 * mfc.notna().mean():.0f}% of swings, despite "
      f"{TOE_FLEXION_DEG:.0f} deg of false toe flexion")
check((st["swing_min_clearance"].dropna() > -0.002).all(),
      f"no swing through the belt: lowest "
      f"{1000 * st['swing_min_clearance'].min():.1f} mm")
below_v3 = np.mean([np.mean(ns["clear_theia_toe"][b][ns["swing_rows"][b]] < 0)
                    for b in "LR"])
check(below_v3 > 0,
      f"the toe-angle hinge (v3) on the same data: {100 * below_v3:.1f}% of "
      f"swing frames below the belt")

check((st["tri_window_s"] > 0.05).mean() > 0.95,
      f"trip-risk window in every swing: median {st['tri_window_s'].median():.3f} s")
tri = st["trip_risk_integral"].mean()
expected = 3.18 * truth["belt_speed"] / 0.90 - 2.02
check(0.5 * expected < tri < 1.5 * expected,
      f"TRI {tri:.2f} s (Schulz 2017 at this speed: ~{expected:.2f} s)")
trusted = st[st["kinetics_trusted"].astype(bool)]
vi = (trusted["vertical_impulse_tw"] / trusted["stride_time"]).mean()
check(abs(vi - 0.5) < 0.03, f"vertical impulse {vi:.3f} TW per stride s (0.5)")
check(abs(ns["system_mass"] - truth["system_mass"]) < 1.0,
      f"system mass {ns['system_mass']:.2f} kg (truth {truth['system_mass']})")
lit = ns["literature_check"]
check(len(lit) > 40 and lit["verdict"].ne("not computed").mean() > 0.9,
      f"literature check: {len(lit)} metrics, "
      f"{int(lit['verdict'].eq('not computed').sum())} not computed")
out = base / "gait_analysis_outputs"
check(all((out / f).exists() for f in (f"{name}_all_metrics_steps.csv",
                                       f"{name}_all_metrics_summary.csv",
                                       "all_trials_all_metrics.csv")),
      "outputs written")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}"
      f"  (work dir {WORK})")
sys.exit(1 if failures else 0)
