"""End-to-end test for gait_events.py on a synthetic treadmill trial.

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
    about something,
  * after step L25's toe-off the left belt keeps a decaying 150 N load for
    250 ms well lateral of the foot, as seen on D05's left belt,
  * both belts have their own baseline, slow drift and noise,
  * the kinematics lose 20 frames to a tracking dropout.

It checks that

  * the registration and the belt's pitch are recovered,
  * no event the pipeline calls GRF is more than a frame from the truth, and
    none of them comes from a step that was crossed or straddled,
  * every true event in the covered range comes out once, with the right
    limb and type, within 3 frames, and the sequence alternates,
  * the merged file has exactly the four columns the metrics script reads.

Run it with `python test_gait_events.py` from anywhere.
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gait_events as ge  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="gait_events_test_"))
rng = np.random.default_rng(7)

FS_K, FS_F = 100, 1000
DURATION = 60.0
CYCLE, STANCE = 1.10, 0.682
BELT_SPEED = 1.3
Y_AT_HS = 0.30
GAP = 0.010                                   # m, belts at |x| > 5 mm
FLOOR_T = 0.012                               # Theia z of the belt surface
YAW = np.radians(90.0)
INCLINE = np.radians(0.84)                    # treadmill pitch, front up
PLATE_W, PLATE_L = 559.0, 1778.0              # mm
PLATE_CX = {"L": -(GAP * 1000 + PLATE_W) / 2, "R": (GAP * 1000 + PLATE_W) / 2}
SHIFT = np.array([0.40, -0.20])
A_TRUE = np.array([[np.cos(YAW), -np.sin(YAW), SHIFT[0]],
                   [np.sin(YAW), np.cos(YAW), SHIFT[1]]])
LATERAL = {"L": -0.10, "R": 0.10}
SPECIAL = {("L", 12): 0.07, ("L", 20): -0.02,        # crossover, straddle
           ("R", 30): 0.015, ("R", 40): -0.075}      # straddle, crossover

meta, binding = ge.load_binding(ge.BINDING_FILE)
v_all, seg = binding["vertices_local"], binding["vertex_segment"]
MESH = {"L": v_all[seg == meta["segments"].index("left_foot")],
        "R": v_all[seg == meta["segments"].index("right_foot")]}


def smooth(s):
    s = np.clip(s, 0, 1)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


# --- the truth: every stance of every foot ----------------------------------
stances = {}
for limb, lag in (("L", 0.0), ("R", CYCLE / 2)):
    out = []
    for n in range(-3, int(DURATION / CYCLE) + 4):
        hs = n * CYCLE + lag + 1.0 + rng.normal(0, 0.012)
        to = hs + STANCE + rng.normal(0, 0.010)
        out.append(dict(n=n, hs=hs, to=to,
                        x=SPECIAL.get((limb, n), LATERAL[limb])))
    stances[limb] = out

t_k = np.arange(int(DURATION * FS_K)) / FS_K


def foot_track(limb):
    """Per kinematic frame: origin (x, y, z) and pitch, treadmill frame."""
    st = stances[limb]
    x, y, pitch, clear = (np.zeros(len(t_k)) for _ in range(4))
    stance_now = np.zeros(len(t_k), bool)
    for i, s in enumerate(st):
        m = (t_k >= s["hs"]) & (t_k < s["to"])
        tau = (t_k[m] - s["hs"]) / (s["to"] - s["hs"])
        stance_now[m] = True
        x[m] = s["x"]
        y[m] = Y_AT_HS - BELT_SPEED * (t_k[m] - s["hs"])
        pitch[m] = (np.radians(15) * (1 - smooth(tau / 0.15))
                    - np.radians(35) * smooth((tau - 0.6) / 0.4))
        if i + 1 < len(st):
            nxt = st[i + 1]
            m = (t_k >= s["to"]) & (t_k < nxt["hs"])
            sw = (t_k[m] - s["to"]) / (nxt["hs"] - s["to"])
            y_to = Y_AT_HS - BELT_SPEED * (s["to"] - s["hs"])
            y[m] = y_to + (Y_AT_HS - y_to) * smooth(sw)
            x[m] = s["x"] + (nxt["x"] - s["x"]) * smooth(sw)
            pitch[m] = np.radians(-35 + 50 * smooth(sw))
            clear[m] = 0.06 * np.sin(np.pi * sw)
    v = MESH[limb]
    c, s = np.cos(pitch), np.sin(pitch)
    lowest = (np.outer(s, v[:, 1]) + np.outer(c, v[:, 2])).min(1)
    z = clear - lowest                       # treadmill floor at z = 0
    return x, y, z, pitch, stance_now


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = 1
    R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = c, -s, s, c
    return R


RZ = np.array([[np.cos(YAW), -np.sin(YAW), 0], [np.sin(YAW), np.cos(YAW), 0],
               [0, 0, 1]])
RI = rot_x(np.array([INCLINE]))[0]           # treadmill -> lab


def to_theia(p):
    """Treadmill frame -> Theia, through the pitched lab frame."""
    q = p @ RI.T @ RZ.T
    q[..., :2] += SHIFT
    q[..., 2] += FLOOR_T
    return q


poses, split, cop_xy = {}, {}, {}
for limb in ("L", "R"):
    x, y, z, pitch, on = foot_track(limb)
    R_t = rot_x(pitch)
    origin = np.column_stack([x, y, z])
    frac = {b: np.zeros(len(t_k)) for b in ("L", "R")}
    cen = {b: np.full((len(t_k), 2), np.nan) for b in ("L", "R")}
    for c0 in range(0, len(t_k), 2000):
        sl = slice(c0, c0 + 2000)
        world = (MESH[limb] @ R_t[sl].transpose(0, 2, 1)
                 + origin[sl, None, :])
        touch = (world[..., 2] < 0.002) & on[sl, None]
        wx = world[..., 0]
        tot = touch.sum(1)
        for b, m in (("L", wx < -GAP / 2), ("R", wx > GAP / 2)):
            w = (touch & (m | (abs(wx) <= GAP / 2))) * np.where(m, 1.0, 0.5)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac[b][sl] = np.where(tot > 0, w.sum(1) / tot, 0.0)
                cen[b][sl] = ((world[..., :2] * w[..., None]).sum(1)
                              / w.sum(1)[:, None])
    split[limb], cop_xy[limb] = frac, cen

    P = np.zeros((len(t_k), 4, 4))
    P[:, :3, :3] = RZ @ RI @ R_t
    P[:, :3, 3] = to_theia(origin) + rng.normal(0, 0.001, origin.shape)
    P[:, 3, 3] = 1
    poses[limb] = P

# --- forces, 1000 Hz ---------------------------------------------------------
t_f = np.arange(int(DURATION * FS_F)) / FS_F
k_of_f = np.minimum(np.arange(len(t_f)) // (FS_F // FS_K), len(t_k) - 1)
fz = {b: np.zeros(len(t_f)) for b in ("L", "R")}
cop_num = {b: np.zeros((len(t_f), 2)) for b in ("L", "R")}
for limb in ("L", "R"):
    load = np.zeros(len(t_f))
    for s in stances[limb]:
        m = (t_f >= s["hs"]) & (t_f < s["to"])
        tau = (t_f[m] - s["hs"]) / (s["to"] - s["hs"])
        load[m] = 750 * (1.05 * np.sin(np.pi * tau)
                         + 0.15 * np.sin(3 * np.pi * tau))
    for b in ("L", "R"):
        part = load * split[limb][b][k_of_f]
        fz[b] += part
        c = np.nan_to_num(cop_xy[limb][b][k_of_f])
        cop_num[b] += c * part[:, None]

# the D05 left-belt tail: something other than the foot, after push-off
TAIL = [s for s in stances["L"] if s["n"] == 25][0]
m = (t_f >= TAIL["to"]) & (t_f < TAIL["to"] + 0.25)
tail = 150 * (1 - (t_f[m] - TAIL["to"]) / 0.25)
fz["L"][m] += tail
cop_num["L"][m] += np.array([-0.25, 0.08]) * tail[:, None]

header = ["", "Left belt_SAMPLE", "Left belt_TIME"]
for belt in ("Left belt", "Right belt"):
    header += [f"{belt}_{q}" for q in ("Force_X", "Force_Y", "Force_Z",
                                       "Moment_X", "Moment_Y", "Moment_Z",
                                       "COP_X", "COP_Y", "COP_Z")]
force_dir, kin_dir = WORK / "FP_renamed", WORK / "Theia_csv_outputs"
force_dir.mkdir()
kin_dir.mkdir()
cols = {}
plate_lines = []
for b, belt, base in (("L", "Left belt", 8.0), ("R", "Right belt", 7.0)):
    true = fz[b].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        cop = cop_num[b] / true[:, None] * 1000
    # into the plate's own frame: origin at its centre, X pointing to -x
    cop = np.column_stack([PLATE_CX[b] - cop[:, 0], cop[:, 1]])
    cop[true < 20] = 0
    cop += rng.normal(0, 1.0, cop.shape)
    plate_lines.append(f"FORCE_PLATE_NAME\t{belt}")
    for corner in ("POSX_POSY", "NEGX_POSY", "NEGX_NEGY", "POSX_NEGY"):
        xl = PLATE_W / 2 * (1 if corner.startswith("POSX") else -1)
        yl = PLATE_L / 2 * (1 if corner.endswith("POSY") else -1)
        lab = RI @ np.array([PLATE_CX[b] - xl, yl, 0.0])
        lab[:2] += rng.normal(0, 4.0, 2)            # measured, not nominal
        plate_lines += [f"FORCE_PLATE_CORNER_{corner}_{a}\t{v:.6f}"
                        for a, v in zip("XYZ", lab)]
    plate_lines += [f"FORCE_PLATE_LENGTH\t{PLATE_L:g}",
                    f"FORCE_PLATE_WIDTH\t{PLATE_W:g}", ""]
    cols[f"{belt}_Force_Z"] = (true + base + 2 * np.sin(2 * np.pi * t_f / 40)
                               + rng.normal(0, 3.0, len(t_f)))
    cols[f"{belt}_COP_X"], cols[f"{belt}_COP_Y"] = cop[:, 0], cop[:, 1]
(WORK / "plates.txt").write_text("\n".join(plate_lines))
trial = "S01_Treadmill_1.3mpers 108bpm"
with open(force_dir / f"{trial}.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(header)
    for i in range(len(t_f)):
        w.writerow([i, i + 1, f"{t_f[i]:.3f}"]
                   + [f"{cols[h][i]:.4f}" if h in cols else "0"
                      for h in header[3:]])

# --- kinematics, a Visual3D-style export -------------------------------------
pelvis_t = np.column_stack([np.zeros(len(t_k)),
                            0.03 * np.sin(2 * np.pi * t_k / 20),
                            0.95 + 0.02 * np.sin(4 * np.pi * t_k / CYCLE)])
signals = {"Pelvis_Position": to_theia(pelvis_t)}
for limb, side in (("L", "Left"), ("R", "Right")):
    P = poses[limb]
    for name, local in (("Heel", (0, -0.045, -0.09)), ("Toes", (0, 0.20, -0.085)),
                        ("Ankle", (0, 0, 0)), ("Foot", (0, 0.065, 0))):
        signals[f"{side}_{name}_Position"] = (P[:, :3, :3] @ np.array(local)
                                             + P[:, :3, 3])
    signals[f"{side}_Foot_Global_4x4"] = P.reshape(len(P), 16)
DROP = slice(3000, 3020)
names, comps, columns = [], [], []
for name, arr in signals.items():
    k = arr.shape[1]
    labels = "XYZ" if k == 3 else [str(i) for i in range(16)]
    a = arr.copy()
    a[DROP] = np.nan
    for j in range(k):
        names.append(name)
        comps.append(labels[j])
        columns.append(a[:, j])
with open(kin_dir / f"{trial}_metrics.csv", "w") as fh:
    fh.write("\t".join(["", *["synthetic.c3d"] * len(names)]) + "\n")
    fh.write("\t".join(["", *names]) + "\n")
    fh.write("\t".join(["", *["LINK_MODEL_BASED"] * len(names)]) + "\n")
    fh.write("\t".join(["", *["ORIGINAL"] * len(names)]) + "\n")
    fh.write("\t".join(["ITEM", *comps]) + "\n")
    for i in range(len(t_k)):
        fh.write("\t".join([str(i + 1)] + ["" if not np.isfinite(c[i])
                                          else f"{c[i]:.6f}"
                                          for c in columns]) + "\n")

# --- run ---------------------------------------------------------------------
ge.FORCE_FOLDER, ge.KINEMATIC_FOLDER = force_dir, kin_dir
ge.OUTPUT_FOLDER = WORK / "out"
ge.FORCE_PLATE_FILE = WORK / "plates.txt"
res = ge.process_trial(force_dir / f"{trial}.csv", meta, binding)

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
pitch = np.degrees(np.arctan(res["feet"].slope))
check(abs(pitch - np.degrees(INCLINE)) < 0.1,
      f"belt pitch {pitch:.2f} deg (true {np.degrees(INCLINE):.2f})")

truth = []
for limb in ("L", "R"):
    for s in stances[limb]:
        truth.append((s["hs"] * FS_K, limb, ge.HS, s["n"]))
        truth.append((s["to"] * FS_K, limb, ge.TO, s["n"]))
events = res["untrimmed"]
lo_t, hi_t = events[0]["t"] - 2, events[-1]["t"] + 2
truth = [x for x in truth if lo_t <= x[0] <= hi_t]


def nearest_truth(e):
    same = [x for x in truth if x[1] == e["limb"] and x[2] == e["event"]]
    return min(same, key=lambda x: abs(x[0] - e["t"]))


grf = [e for e in events if e["source"] == "GRF"]
worst = max(abs(nearest_truth(e)[0] - e["t"]) for e in grf)
check(worst <= 1.0, f"{len(grf)} GRF events, worst {worst:.2f} frames off")
unv = [e for e in events if e["source"] == "GRF_unverified"]
worst = max((abs(nearest_truth(e)[0] - e["t"]) for e in unv), default=0)
check(unv and worst <= 1.0, f"{len(unv)} GRF events used through the "
      f"tracking dropout, worst {worst:.2f} frames off")
tail_to = [e for e in grf if e["limb"] == "L" and e["event"] == ge.TO
           and nearest_truth(e)[3] == 25]
check(not tail_to, "the toe-off with the slow unloading tail is not GRF")
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
order = [ge.POSITION[(e["limb"], e["event"])] for e in events]
check(all((b - a) % 4 == 1 for a, b in zip(order[:-1], order[1:])),
      "events alternate L HS, R TO, R HS, L TO")
final = res["events"]
for limb in ("L", "R"):
    kinds = [e["event"] for e in final if e["limb"] == limb]
    check(kinds[0] == ge.HS and kinds[-1] == ge.TO and all(
        a != b for a, b in zip(kinds[:-1], kinds[1:])),
        f"{limb}: trimmed output runs HS, TO, HS ... TO")
sources = {e["source"] for e in events}
check("interpolated" not in sources, f"sources used: {sorted(sources)}")
labels = {c["label"] for c in res["contacts"]}
check({"shared", "straddle"} <= labels, f"contact labels: {sorted(labels)}")

with open(ge.OUTPUT_FOLDER / f"{trial}_merged_events.csv") as fh:
    head = next(csv.reader(fh))
check(head == ["event_type", "support_limb", "frame_100hz", "source"],
      f"merged file columns {head}")
bench = {(r["detector"], r["limb"], r["event"]): r for r in res["benchmark"]}
order = ge.fallback_order(res["benchmark"])
ge.FALLBACK_ORDER = "auto"
auto = ge.fallback_order(res["benchmark"])
ge.FALLBACK_ORDER = order[("L", ge.HS)]
check(all(v and set(v) <= set(ge.fd.DETECTORS) for v in auto.values()),
      f"auto fallback order: L HS {auto[('L', ge.HS)][:2]}, "
      f"L TO {auto[('L', ge.TO)][:2]}")
check(len(bench) == 2 * 2 * len(ge.fd.DETECTORS),
      f"benchmark rows for all {len(ge.fd.DETECTORS)} detectors")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}"
      f"  (work dir {WORK})")
sys.exit(1 if failures else 0)
