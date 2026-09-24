"""End-to-end test for detect_gait_events.py on a synthetic treadmill trial.

The real boot meshes from foot_mesh_binding.npz walk on a split-belt
treadmill whose forces are computed from where each sole actually touches:

  * the force-plate frame is rotated 90 deg and shifted from Theia's, and the
    registration has to find that from the data alone,
  * as in the DICE export, each belt's CoP is in that plate's own frame (X
    flipped, origin at the plate centre), the plates are described by
    measured corners a few mm off their nominal rectangle, and the treadmill
    is pitched 0.84 deg,
  * step L12 crosses onto the right belt, L20 and R30 straddle the gap and
    R40 crosses onto the left belt -- every belt contact they touch is wrong
    about something -- while L33 hangs 20 mm over the 40 mm gap without
    reaching the other belt, which costs nothing,
  * after step L25's toe-off the left belt keeps a decaying 150 N load for
    250 ms well lateral of the foot, as seen on D05's left belt,
  * the boots bend at the MTP: at push-off the foot tips heel-up over toes
    that stay flat on the belt, and the export carries the toe angle,
  * the trial opens with a quiet standing, belt still, as every DICE trial
    does -- and no phantom steps may come out of it,
  * both belts have their own baseline, slow drift and noise,
  * the kinematics lose 20 frames to a tracking dropout.

The trial itself comes from synthetic_trial.py, shared with
test_gait_analysis.py.

It checks that

  * the registration and the belt's pitch are recovered, and the bent boot
    stays on the belt where a rigid one goes through it,
  * no event the pipeline calls GRF is more than half a frame from the truth,
    none of them comes from a step that was crossed or straddled, and the
    overhanging step keeps its GRF events,
  * every true event in the covered range comes out once, with the right
    limb and type, within 3 frames, and the sequence alternates,
  * every event is trusted GRF or Zeni (GRF_unverified only in the
    dropout), and the Zeni-filled ones are within a few frames,
  * the merged file has exactly the four columns the metrics script reads.

Run it with `python test_detect_gait_events.py` from anywhere.
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import detect_gait_events as dg  # noqa: E402
import synthetic_trial as syn  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="gait_events_test_"))
FS_K = syn.FS_K
truth_trial = syn.make_trial(WORK, duration=60.0)
trial = truth_trial["name"]
stances, on_belt = truth_trial["stances"], truth_trial["on_belt"]
force_dir, kin_dir = truth_trial["force_dir"], truth_trial["kin_dir"]
MESH, POSE_NAMES = dg.read_binding(dg.HERE / "foot_mesh_binding.npz")
YAW, INCLINE, A_TRUE = syn.YAW, syn.INCLINE, syn.A_TRUE
SPECIAL, OVERHANG = syn.SPECIAL, syn.OVERHANG

# --- run ---------------------------------------------------------------------
dg.FORCE_FOLDER, dg.KINEMATIC_FOLDER = force_dir, kin_dir
dg.OUTPUT_FOLDER = WORK / "out"
dg.PLATE_FILE = truth_trial["plates"]
plates = {b: dg.fit_plate(p) for b, p in dg.read_plates(dg.PLATE_FILE).items()}
res = dg.process_trial(force_dir / f"{trial}.csv", MESH, POSE_NAMES, plates)

failures = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


print("\nchecks")
A, reg = res["registration"]
rot_err = np.degrees(abs(np.arctan2(A[1, 0], A[0, 0]) - YAW))
shift_err = np.linalg.norm(A[:, 2] - A_TRUE[:, 2]) * 1000
check(rot_err < 1.0 and shift_err < 15,
      f"registration: {rot_err:.2f} deg, {shift_err:.1f} mm from the truth")
pitch = np.degrees(np.arctan(res["surface"]["slope"]))
check(abs(pitch - np.degrees(INCLINE)) < 0.1,
      f"belt pitch {pitch:.2f} deg (true {np.degrees(INCLINE):.2f})")
low = {b: np.nanmin(res["height"][b][on_belt[b]]) * 1000
       for b in ("L", "R")}
dg.TOE_HINGE = False
rigid = dg.process_trial(force_dir / f"{trial}.csv", MESH, POSE_NAMES,
                         plates, verbose=False)
dg.TOE_HINGE = True
low_rigid = {b: np.nanmin(rigid["height"][b][on_belt[b]]) * 1000
             for b in ("L", "R")}
check(min(low.values()) > -8 and max(low_rigid.values()) < -10,
      f"deepest sole point in stance: bent {low['L']:.0f} / "
      f"{low['R']:.0f} mm, rigid {low_rigid['L']:.0f} / {low_rigid['R']:.0f} mm")

truth = []
for limb in ("L", "R"):
    for s in stances[limb]:
        if not s["standing"]:
            truth.append((s["hs"] * FS_K, limb, dg.HS, s["n"]))
        truth.append((s["to"] * FS_K, limb, dg.TO, s["n"]))
events = res["untrimmed"]
lo_t, hi_t = events[0]["t"] - 2, events[-1]["t"] + 2
truth = [x for x in truth if lo_t <= x[0] <= hi_t]


def nearest_truth(e):
    same = [x for x in truth if x[1] == e["limb"] and x[2] == e["event"]]
    return min(same, key=lambda x: abs(x[0] - e["t"]))


grf = [e for e in events if e["source"] == "GRF"]
worst = max(abs(nearest_truth(e)[0] - e["t"]) for e in grf)
check(worst <= 0.5, f"{len(grf)} GRF events, worst {worst:.2f} frames off")
unv = [e for e in events if e["source"] == "GRF_unverified"]
worst = max((abs(nearest_truth(e)[0] - e["t"]) for e in unv), default=0)
check(unv and worst <= 1.0, f"{len(unv)} GRF events used through the "
      f"tracking dropout, worst {worst:.2f} frames off")
tail_to = [e for e in grf if e["limb"] == "L" and e["event"] == dg.TO
           and nearest_truth(e)[3] == 25]
check(not tail_to, "the toe-off with the slow unloading tail is not GRF")
over = [e for e in grf if (e["limb"], nearest_truth(e)[3]) == OVERHANG]
check(len(over) == 2, f"the step hanging over the gap keeps {len(over)} of "
      f"its 2 GRF events")
special_grf = [e for e in grf if (e["limb"], nearest_truth(e)[3]) in SPECIAL]
check(not special_grf,
      f"no GRF event from a crossed / straddled step ({len(special_grf)})")

errors, missing = [], []
for x in truth:
    same = [e for e in events if e["limb"] == x[1] and e["event"] == x[2]]
    d = min((abs(e["t"] - x[0]) for e in same), default=np.inf)
    (errors if d <= 3 else missing).append(d)
check(not missing, f"{len(truth)} true events recovered, {len(missing)} "
      f"missed; worst {max(errors):.2f} frames")
check(len(events) == len(truth),
      f"{len(events)} events out for {len(truth)} true ones")
order = [dg.POSITION[(e["limb"], e["event"])] for e in events]
check(all((b - a) % 4 == 1 for a, b in zip(order[:-1], order[1:])),
      "events alternate L HS, R TO, R HS, L TO")
final = res["events"]
for limb in ("L", "R"):
    kinds = [e["event"] for e in final if e["limb"] == limb]
    check(kinds[0] == dg.HS and kinds[-1] == dg.TO and all(
        a != b for a, b in zip(kinds[:-1], kinds[1:])),
        f"{limb}: trimmed output runs HS, TO, HS ... TO")
sources = {e["source"] for e in events}
check("interpolated" not in sources, f"sources used: {sorted(sources)}")
labels = {c["label"] for c in res["contacts"]}
check({"shared", "straddle"} <= labels, f"contact labels: {sorted(labels)}")

with open(dg.OUTPUT_FOLDER / f"{trial}_merged_events.csv") as fh:
    head = next(csv.reader(fh))
check(head == ["event_type", "support_limb", "frame_100hz", "source"],
      f"merged file columns {head}")
methods = {e["method"] for e in events}
check(methods <= {"GRF", "zeni", "GRF_unverified"},
      f"events found by: {sorted(methods)}")
zeni = [e for e in events if e["method"] == "zeni"]
worst = max(abs(nearest_truth(e)[0] - e["t"]) for e in zeni)
check(zeni and all(e["source"] == "kinematic" for e in zeni) and worst <= 3,
      f"{len(zeni)} events filled by Zeni (source 'kinematic'), worst "
      f"{worst:.2f} frames off")
check(set(res["zeni"]) == {"L heel_strike", "L toe_off", "R heel_strike",
                           "R toe_off"},
      "Zeni offset and error reported for every limb and event")
check((dg.OUTPUT_FOLDER / f"{trial}_event_summary.json").exists()
      and (dg.OUTPUT_FOLDER / f"{trial}_grf_contacts.csv").exists(),
      "wrote the summary and contact files")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}"
      f"  (work dir {WORK})")
sys.exit(1 if failures else 0)
