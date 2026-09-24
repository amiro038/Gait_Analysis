"""
A synthetic DICE treadmill trial whose answers are known, for the tests.

The real boot meshes from foot_mesh_binding.npz walk on a split-belt
treadmill, and every file the pipeline reads is written the way the DICE
exports are:

  * a quiet standing to start, belt still, then the belt ramps to speed over
    a second and walking begins. The walker carries a LOAD whose mass and
    position on the trunk are known;
  * the force plates see the SYSTEM (body + load) CoM's dynamics: the total
    GRF is m (a + g) of a known CoM trajectory, shared between the feet by
    which is on the belt, with the CoP under each boot's contact patch;
  * forces, moments and CoP are in each plate's own frame (X flipped, Z
    down, origin at the plate centre, force ON the plate), the plates are
    described by measured corners a few mm off their nominal rectangle, the
    treadmill is pitched 0.84 deg, and the plate frame is rotated 90 deg and
    shifted from Theia's;
  * optionally, awkward steps: L12 crosses onto the right belt, L20 and R30
    straddle the gap, R40 crosses onto the left belt, L33 hangs 20 mm over
    the gap, and after L25's toe-off the left belt keeps a decaying load;
  * the boots bend at the MTP, and the lowest point of the swinging boot
    dips to a known minimum foot clearance in mid-swing;
  * stride times are persistent (fractional Gaussian noise), belts have
    their own baseline, drift and noise, and the kinematics lose 20 frames
    to a tracking dropout;
  * the Theia export carries what the analysis reads: poses, joint centres,
    Whole_body_COG (the BODY's CoM, without the load), trunk, joint angles.

    truth = make_trial(folder, duration=60)

writes FP_renamed/, Theia_csv_outputs/, plates.txt and participants.csv in
`folder` and returns the truth.
"""

from pathlib import Path

import numpy as np

import detect_gait_events as dg
from gait_analysis import fractional_gaussian_noise

FS_K, FS_F = 100, 1000
G = 9.81
GAP = 0.040
FLOOR_T = 0.012                     # Theia z of the belt surface
YAW = np.radians(90.0)
INCLINE = np.radians(0.84)
PLATE_W, PLATE_L = 559.0, 1778.0    # mm
PLATE_CX = {"L": -(GAP * 1000 + PLATE_W) / 2, "R": (GAP * 1000 + PLATE_W) / 2}
SHIFT = np.array([0.40, -0.20])
A_TRUE = np.array([[np.cos(YAW), -np.sin(YAW), SHIFT[0]],
                   [np.sin(YAW), np.cos(YAW), SHIFT[1]]])
LATERAL = {"L": -0.10, "R": 0.10}
Y_AT_HS = 0.30
SPECIAL = {("L", 12): 0.07, ("L", 20): -0.02,        # crossover, straddle
           ("R", 30): 0.015, ("R", 40): -0.075}      # straddle, crossover
OVERHANG = ("L", 33)
TAIL = ("L", 25)
MTP = np.array([0.0, 0.15, -0.04])  # in the foot frame
RAMP_S = 0.10                       # a foot takes its share over this long

RZ = np.array([[np.cos(YAW), -np.sin(YAW), 0], [np.sin(YAW), np.cos(YAW), 0],
               [0, 0, 1]])


def smooth(s):
    s = np.clip(s, 0, 1)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def rot_x(a):
    a = np.atleast_1d(a)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = 1
    R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = c, -s, s, c
    return R


RI = rot_x(INCLINE)[0]              # treadmill -> plate-corner lab frame
# plate axes in the lab: X flipped, Y along the belt, Z down
R_PLATE = np.column_stack([RI @ [-1, 0, 0], RI @ [0, 1, 0], RI @ [0, 0, -1]])


def to_theia(p):
    """Treadmill frame -> Theia, through the pitched lab frame."""
    q = p @ RI.T @ RZ.T
    q[..., :2] += SHIFT
    q[..., 2] += FLOOR_T
    return q


def bend(v, toe_angle):
    """Boot vertices (V, 3) with the toe cap turned by toe_angle (n,) about
    the MTP, extension positive -> (n, V, 3), still in the foot frame."""
    out = np.repeat(v[None], len(toe_angle), axis=0)
    toe = v[:, 1] > MTP[1]
    c, s = np.cos(toe_angle)[:, None], np.sin(toe_angle)[:, None]
    d = v[toe] - MTP
    out[:, toe, 1] = MTP[1] + c * d[:, 1] - s * d[:, 2]
    out[:, toe, 2] = MTP[2] + s * d[:, 1] + c * d[:, 2]
    return out


def tracking_noise(rng, shape, sd=0.001, cutoff=6.0):
    """Tracking noise as Theia leaves it: filtered, so smooth frame to
    frame, with an SD of `sd` metres."""
    from scipy.signal import butter, sosfiltfilt
    x = sosfiltfilt(butter(2, cutoff, fs=FS_K, output="sos"),
                    rng.normal(size=shape), axis=0)
    return sd * x / x.std(axis=0)


def ramp_in(s):
    return np.sqrt(np.clip(s, 0, 1))


def clearance_profile(sw, mfc):
    """Lowest sole point above the belt through a swing (0..1): a first peak
    after toe-off, a dip to about `mfc` near 57% of swing, then landing."""
    return np.sqrt(np.sin(np.pi * np.clip(sw, 0, 1))) * (
        mfc + 0.35 * (sw - 0.55) ** 2)


def make_trial(folder, duration=60.0, standing=3.0, body_mass=80.0,
               load_mass=20.0, load_back=0.15, special=True, seed=7,
               name="S01_C1_Treadmill_1.3mpers 108bpm", cycle=1.10,
               stance_frac=0.62, belt_speed=1.3, mfc=0.020, hurst=0.8):
    folder = Path(folder)
    rng = np.random.default_rng(seed)
    mesh, _ = dg.read_binding(dg.HERE / "foot_mesh_binding.npz")
    t_k = np.arange(int(duration * FS_K)) / FS_K
    t_f = np.arange(int(duration * FS_F)) / FS_F
    m_sys = body_mass + load_mass

    # --- when each foot is down --------------------------------------------
    n_strides = int((duration - standing) / cycle) + 4
    T = cycle + 0.012 * fractional_gaussian_noise(n_strides, hurst, rng)
    hs_l = standing + 0.8 + np.r_[0, np.cumsum(T[:-1])]
    stances = {"L": [], "R": []}
    for n, (h, Tn) in enumerate(zip(hs_l, T)):
        for limb, hs in (("L", h), ("R", h + Tn / 2 + rng.normal(0, 0.005))):
            x = LATERAL[limb] + rng.normal(0, 0.008)
            if special:
                x = SPECIAL.get((limb, n), x)
                if (limb, n) == OVERHANG:
                    x = -GAP / 2 + 0.020 - mesh["L"][:, 0].max()
            stances[limb].append(dict(
                n=n, hs=hs, to=hs + stance_frac * Tn + rng.normal(0, 0.008),
                x=x, y_land=Y_AT_HS, standing=False,
                mfc=mfc + rng.normal(0, 0.003)))
    # the standing "stance": from the start to the first lift-off
    for limb in ("L", "R"):
        first = stances[limb][0]
        stances[limb].insert(0, dict(
            n=-1, hs=-1.0, to=first["hs"] - (1 - stance_frac) * cycle,
            x=LATERAL[limb], y_land=0.0, standing=True,
            mfc=mfc + rng.normal(0, 0.003)))

    def belt(t):
        """Belt travel since the start: still, then a 1 s ramp to speed."""
        u = np.clip(t - standing, 0, None)
        return belt_speed * np.where(u < 1.0, u ** 2 / 2, u - 0.5)

    def foot_track(limb):
        st = stances[limb]
        x, y, pitch, clear, toe = (np.zeros(len(t_k)) for _ in range(5))
        down = np.zeros(len(t_k), bool)
        for i, s in enumerate(st):
            m = (t_k >= s["hs"]) & (t_k < s["to"])
            down[m] = True
            x[m] = s["x"]
            hs0 = max(s["hs"], 0.0)
            y[m] = s["y_land"] - (belt(t_k[m]) - belt(hs0))
            if s["standing"]:
                tau = 1 - (s["to"] - t_k[m]) / (stance_frac * cycle)
                pitch[m] = -np.radians(35) * smooth((tau - 0.6) / 0.4)
            else:
                tau = (t_k[m] - s["hs"]) / (s["to"] - s["hs"])
                pitch[m] = (np.radians(15) * (1 - smooth(tau / 0.15))
                            - np.radians(35) * smooth((tau - 0.6) / 0.4))
            toe[m] = np.clip(-pitch[m], 0, None)
            if i + 1 < len(st):
                nxt = st[i + 1]
                m = (t_k >= s["to"]) & (t_k < nxt["hs"])
                sw = (t_k[m] - s["to"]) / (nxt["hs"] - s["to"])
                y_to = s["y_land"] - (belt(s["to"]) - belt(hs0))
                y[m] = y_to + (Y_AT_HS - y_to) * smooth(sw)
                x[m] = s["x"] + (nxt["x"] - s["x"]) * smooth(sw)
                pitch[m] = np.radians(-35 + 50 * smooth(sw))
                clear[m] = clearance_profile(sw, s["mfc"])
                toe[m] = np.radians(35) * (1 - smooth(sw / 0.3))
        lowest = np.zeros(len(t_k))
        c, s_ = np.cos(pitch), np.sin(pitch)
        for c0 in range(0, len(t_k), 1000):
            sl = slice(c0, c0 + 1000)
            vb = bend(mesh[limb], toe[sl])
            lowest[sl] = (s_[sl, None] * vb[..., 1]
                          + c[sl, None] * vb[..., 2]).min(1)
        return x, y, clear - lowest, pitch, down, toe

    # --- boots, and which belt carries each ----------------------------------
    poses, split, patch, toe_angle, on_belt = {}, {}, {}, {}, {}
    for limb in ("L", "R"):
        x, y, z, pitch, down, toe_angle[limb] = foot_track(limb)
        on_belt[limb] = down
        R_t = rot_x(pitch)
        origin = np.column_stack([x, y, z])
        frac = {b: np.zeros(len(t_k)) for b in ("L", "R")}
        cen = {b: np.full((len(t_k), 2), np.nan) for b in ("L", "R")}
        for c0 in range(0, len(t_k), 2000):
            sl = slice(c0, c0 + 2000)
            world = (np.einsum("nij,nvj->nvi", R_t[sl],
                               bend(mesh[limb], toe_angle[limb][sl]))
                     + origin[sl, None, :])
            touch = (world[..., 2] < 0.002) & down[sl, None]
            wx = world[..., 0]
            tot = (touch & (abs(wx) > GAP / 2)).sum(1)
            for b, m in (("L", wx < -GAP / 2), ("R", wx > GAP / 2)):
                w = (touch & m) * 1.0
                with np.errstate(invalid="ignore", divide="ignore"):
                    frac[b][sl] = np.where(tot > 0, w.sum(1) / tot, 0.0)
                    cen[b][sl] = ((world[..., :2] * w[..., None]).sum(1)
                                  / w.sum(1)[:, None])
        split[limb], patch[limb] = frac, cen
        P = np.zeros((len(t_k), 4, 4))
        P[:, :3, :3] = RZ @ RI @ R_t
        P[:, :3, 3] = to_theia(origin) + tracking_noise(rng, origin.shape)
        P[:, 3, 3] = 1
        poses[limb] = P

    # --- the system CoM, treadmill frame, 1000 Hz ------------------------------
    k_of_f = np.minimum(np.arange(len(t_f)) // (FS_F // FS_K), len(t_k) - 1)
    stand_rows = t_k < standing - 0.2
    c_stand = np.nanmean([np.nanmean(np.nansum(
        [np.nan_to_num(patch[f][b][stand_rows]) * split[f][b][stand_rows, None]
         for b in ("L", "R")], axis=0), axis=0) for f in ("L", "R")], axis=0)
    # standing still, the CoM is straight above the CoP -- vertically, not
    # perpendicular to the pitched belt: 0.95 m up the belt's normal would
    # put it 14 mm off
    c_stand[1] += 0.95 * np.tan(INCLINE)
    hs_all = np.array([s["hs"] for s in stances["L"][1:]])
    phase = np.interp(t_f, hs_all, np.arange(len(hs_all)))
    mid = stance_frac / 2
    ramp = smooth((t_f - standing) / 1.5)
    c_sys = np.column_stack([
        c_stand[0] - 0.025 * np.cos(2 * np.pi * (phase - mid)) * ramp,
        c_stand[1] + 0.012 * np.sin(4 * np.pi * phase) * ramp
        + 0.01 * np.sin(2 * np.pi * t_f / 23) * ramp,
        0.95 + 0.020 * np.cos(4 * np.pi * (phase - mid)) * ramp])
    c_theia = to_theia(c_sys)
    v_theia = np.gradient(c_theia, 1 / FS_F, axis=0)
    a_theia = np.gradient(v_theia, 1 / FS_F, axis=0)
    F_total = m_sys * (a_theia + [0, 0, G])

    # --- each foot's share of it, then each belt's ----------------------------
    raw = {}
    for limb in ("L", "R"):
        r = np.zeros(len(t_f))
        for s in stances[limb]:
            m = (t_f >= s["hs"]) & (t_f < s["to"])
            # a foot takes load steeply at contact and sheds it steeply at
            # lift-off, as a real one does
            rise = 1.0 if s["standing"] else ramp_in((t_f[m] - s["hs"]) / RAMP_S)
            r[m] = rise * ramp_in((s["to"] - t_f[m]) / RAMP_S)
        raw[limb] = r
    both = raw["L"] + raw["R"]
    F_belt = {b: np.zeros((len(t_f), 3)) for b in ("L", "R")}
    cop_num = {b: np.zeros((len(t_f), 2)) for b in ("L", "R")}
    for limb in ("L", "R"):
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(both > 0, raw[limb] / both, 0.0)
        # a foot that lands between two frames is on the belt from its heel
        # strike, not from the next frame: its load goes to the contact
        # patch of the frame after
        k = k_of_f.copy()
        airborne = (split[limb]["L"] + split[limb]["R"])[k] == 0
        k[airborne & (share > 0)] = np.minimum(k[airborne & (share > 0)] + 1,
                                               len(t_k) - 1)
        for b in ("L", "R"):
            part = share * split[limb][b][k]
            F_belt[b] += part[:, None] * F_total
            cop_num[b] += (np.nan_to_num(patch[limb][b][k])
                           * (part * F_total[:, 2])[:, None])
    if special:
        tail = [s for s in stances["L"] if s["n"] == TAIL[1]][0]
        m = (t_f >= tail["to"]) & (t_f < tail["to"] + 0.25)
        extra = 150 * (1 - (t_f[m] - tail["to"]) / 0.25)
        F_belt["L"][m, 2] += extra
        cop_num["L"][m] += np.array([-0.25, 0.08]) * extra[:, None]

    # --- into the plate frame and out to the export ---------------------------
    cols = {}
    header = ["", "Left belt_SAMPLE", "Left belt_TIME"]
    plate_lines = []
    for b, belt_name, base in (("L", "Left belt", 8.0), ("R", "Right belt", 7.0)):
        Ft = F_belt[b]
        fz_true = Ft[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            cop = cop_num[b] / fz_true[:, None] * 1000         # treadmill mm
        cop = np.column_stack([PLATE_CX[b] - cop[:, 0], cop[:, 1]])
        cop[fz_true < 20] = 0
        F_lab = Ft @ RZ                                    # Theia -> lab
        Fp = -F_lab @ R_PLATE                              # ON the plate
        free = 0.005 * np.clip(fz_true, 0, None)           # N m, known
        My = -cop[:, 0] * Fp[:, 2] / 1000
        Mx = cop[:, 1] * Fp[:, 2] / 1000
        Mz = free + (cop[:, 0] * Fp[:, 1] - cop[:, 1] * Fp[:, 0]) / 1000
        drift = 2 * np.sin(2 * np.pi * t_f / 40)
        cols[f"{belt_name}_Force_X"] = Fp[:, 0] + 2.0 + rng.normal(0, 1.5, len(t_f))
        cols[f"{belt_name}_Force_Y"] = Fp[:, 1] - 1.5 + rng.normal(0, 1.5, len(t_f))
        cols[f"{belt_name}_Force_Z"] = Fp[:, 2] + base + drift + rng.normal(0, 3.0, len(t_f))
        cols[f"{belt_name}_Moment_X"] = Mx + rng.normal(0, 0.2, len(t_f))
        cols[f"{belt_name}_Moment_Y"] = My + rng.normal(0, 0.2, len(t_f))
        cols[f"{belt_name}_Moment_Z"] = Mz + rng.normal(0, 0.1, len(t_f))
        noisy = cop + rng.normal(0, 1.0, cop.shape)
        cols[f"{belt_name}_COP_X"], cols[f"{belt_name}_COP_Y"] = noisy.T
        cols[f"{belt_name}_COP_Z"] = np.zeros(len(t_f))
        header += [f"{belt_name}_{q}" for q in
                   ("Force_X", "Force_Y", "Force_Z", "Moment_X", "Moment_Y",
                    "Moment_Z", "COP_X", "COP_Y", "COP_Z")]
        plate_lines.append(f"FORCE_PLATE_NAME\t{belt_name}")
        for corner in ("POSX_POSY", "NEGX_POSY", "NEGX_NEGY", "POSX_NEGY"):
            xl = PLATE_W / 2 * (1 if corner.startswith("POSX") else -1)
            yl = PLATE_L / 2 * (1 if corner.endswith("POSY") else -1)
            lab = RI @ np.array([PLATE_CX[b] - xl, yl, 0.0])
            lab[:2] += rng.normal(0, 4.0, 2)            # measured, not nominal
            plate_lines += [f"FORCE_PLATE_CORNER_{corner}_{a}\t{v:.6f}"
                            for a, v in zip("XYZ", lab)]
        plate_lines += [f"FORCE_PLATE_LENGTH\t{PLATE_L:g}",
                        f"FORCE_PLATE_WIDTH\t{PLATE_W:g}", ""]
    force_dir = folder / "FP_renamed"
    kin_dir = folder / "Theia_csv_outputs"
    force_dir.mkdir(parents=True, exist_ok=True)
    kin_dir.mkdir(parents=True, exist_ok=True)
    table = np.column_stack([np.arange(len(t_f)), np.arange(len(t_f)) + 1, t_f]
                            + [cols[h] for h in header[3:]])
    np.savetxt(force_dir / f"{name}.csv", table, delimiter=",",
               header=",".join(header), comments="",
               fmt=["%d", "%d", "%.3f"] + ["%.4f"] * (len(header) - 3))
    (folder / "plates.txt").write_text("\n".join(plate_lines))

    # --- the Theia export ------------------------------------------------------
    rows = np.arange(len(t_k)) * (FS_F // FS_K)
    com_sys = c_theia[rows]
    fwd = (RZ @ RI @ [0, 1, 0])[:2]
    fwd /= np.linalg.norm(fwd)
    lat = np.array([-fwd[1], fwd[0]])
    trunk = com_sys + to_theia(np.array([0, 0.02, 0.30])) - to_theia(np.zeros(3))
    lean = np.radians(5.0) + np.radians(1.0) * np.sin(4 * np.pi * phase[rows])
    up_t = np.column_stack([np.zeros(len(t_k)), np.sin(lean), np.cos(lean)])
    up = up_t @ RI.T @ RZ.T                               # treadmill -> Theia
    low_back = trunk - 0.25 * up
    neck = trunk + 0.25 * up
    # the load: load_back metres behind the trunk at standing, then carried
    # in the trunk frame exactly as gait/trial.py builds it
    x_t = -np.r_[lat, 0] - (up @ -np.r_[lat, 0])[:, None] * up
    x_t /= np.linalg.norm(x_t, axis=1, keepdims=True)
    R_t = np.stack([x_t, np.cross(up, x_t), up], axis=2)
    p0 = trunk[0] - load_back * np.r_[fwd, 0]
    offset = R_t[0].T @ (p0 - trunk[0])
    load_pos = trunk + np.einsum("fij,j->fi", R_t, offset)
    com_body = (m_sys * com_sys - load_mass * load_pos) / body_mass

    signals = {"Pelvis_Position": trunk - 0.30 * up,
               "Whole_body_COG": com_body + tracking_noise(rng, com_body.shape,
                                                           0.0015),
               "Trunk_Position": trunk, "Low_Back_Position": low_back,
               "Neck_Position": neck,
               "Trunk_Linear_Velocity": np.gradient(trunk, 1 / FS_K, axis=0)
               + rng.normal(0, 0.01, trunk.shape),
               "Thorax_Seg_Angle": np.column_stack(
                   [np.degrees(lean), np.zeros(len(t_k)), np.zeros(len(t_k))]),
               "Pelvis_Seg_Angle": np.column_stack(
                   [8 + 2 * np.sin(4 * np.pi * phase[rows]),
                    np.zeros(len(t_k)), np.zeros(len(t_k))])}
    for limb, side in (("L", "Left"), ("R", "Right")):
        P = poses[limb]
        for part, local in (("Heel", (0, -0.045, -0.09)), ("Toes", MTP),
                            ("Ankle", (0, 0, 0)), ("Foot", (0, 0.065, 0))):
            signals[f"{side}_{part}_Position"] = (P[:, :3, :3] @ np.array(local)
                                                 + P[:, :3, 3])
        signals[f"{side}_Foot_Global_4x4"] = P.reshape(len(P), 16)
        signals[f"{side}_Toes_Joint_Angle"] = np.column_stack(
            [np.degrees(toe_angle[limb]), np.zeros(len(P)), np.zeros(len(P))])
        ph = 2 * np.pi * (phase[rows] + (0.5 if limb == "R" else 0.0))
        for joint, amp, mean in (("Hip", 20, 10), ("Knee", 30, 25),
                                 ("Ankle", 12, 0)):
            signals[f"{side}_{joint}_Joint_Angle"] = np.column_stack(
                [mean + amp * np.cos(ph), np.zeros(len(P)), np.zeros(len(P))])
    # a 20-frame tracking dropout over the right heel strike nearest 30 s
    k = min(stances["R"], key=lambda s: abs(s["hs"] - 30.0))
    drop = slice(int(round(k["hs"] * FS_K)) - 8, int(round(k["hs"] * FS_K)) + 12)
    names, comps, columns = [], [], []
    for sig, arr in signals.items():
        labels = "XYZ" if arr.shape[1] == 3 else [str(i) for i in range(16)]
        a = arr.copy()
        a[drop] = np.nan
        for j in range(arr.shape[1]):
            names.append(sig)
            comps.append(labels[j])
            columns.append(a[:, j])
    head = ["\t".join(["", *["synthetic.c3d"] * len(names)]),
            "\t".join(["", *names]),
            "\t".join(["", *["LINK_MODEL_BASED"] * len(names)]),
            "\t".join(["", *["ORIGINAL"] * len(names)]),
            "\t".join(["ITEM", *comps])]
    np.savetxt(kin_dir / f"{name}_metrics.csv",
               np.column_stack([np.arange(1, len(t_k) + 1)] + columns),
               delimiter="\t", header="\n".join(head), comments="",
               fmt=["%d"] + ["%.6f"] * len(columns))
    (folder / "participants.csv").write_text(
        "participant,body_mass_kg,leg_length_m,height_m,foot_length_m\n"
        f"{name.split('_')[0]},{body_mass},0.90,1.78,\n")

    return dict(name=name, folder=folder, force_dir=force_dir, kin_dir=kin_dir,
                plates=folder / "plates.txt", mesh=mesh, stances=stances,
                on_belt=on_belt, standing=standing, body_mass=body_mass,
                load_mass=load_mass, system_mass=m_sys, load_offset=offset,
                com=com_sys, com_velocity=v_theia[rows], forward=fwd,
                belt_speed=belt_speed, cycle=cycle, stance_frac=stance_frac,
                mfc=mfc, t_k=t_k, special=special, dropout=drop)
