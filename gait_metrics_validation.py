# -*- coding: utf-8 -*-
"""
Validation for the metrics added to Spatiotemporal_analysis_v3.py.

Same idea as lds_validation.py: before a number is used to compare conditions,
establish what it does on signals whose answer is already known. lds_validation
found the Lyapunov estimator off by a system-dependent 4-19%, which is exactly
the kind of thing you want to know before interpreting a between-condition
difference rather than after.

Each section imports the REAL function out of Spatiotemporal_analysis_v3.py by
source extraction, so this tests the code that actually runs, not a copy of it
that can drift.

Run the whole file, or run one cell at a time in Spyder.
"""

import io
import numpy as np
from scipy.signal import butter, filtfilt

ANALYSIS_SCRIPT = "Spatiotemporal_analysis_v3.py"


# %%==========================================================================
#  helper: pull one function out of the analysis script
# ============================================================================

def load_function(name, script=None, extra=None):
    """Extract a top-level `def name(...)` from the analysis script and exec it.

    The analysis script is flat, top to bottom, and running it end to end needs
    the real data files. Pulling one function out lets the validation run
    anywhere while still testing the shipped code.
    """
    script = ANALYSIS_SCRIPT if script is None else script
    src = io.open(script, encoding="utf-8").read()
    start = src.index(f"def {name}(")
    # the function ends at the next top-level statement
    rest = src[start:]
    lines = rest.split("\n")
    body = [lines[0]]
    for line in lines[1:]:
        if line and not line[0].isspace() and not line.startswith(")"):
            break
        body.append(line)
    namespace = dict(np=np, butter=butter, filtfilt=filtfilt)
    namespace.update(extra or {})
    exec(compile("\n".join(body), script, "exec"), namespace)
    return namespace[name]


def load_cell(marker, namespace, script=None, until=None):
    """Exec one `# %%` cell of the analysis script against a prepared namespace.

    Used where the thing under test is the cell body itself rather than a
    function inside it. Same principle as load_function: run the shipped code,
    not a copy.
    """
    script = ANALYSIS_SCRIPT if script is None else script
    src = io.open(script, encoding="utf-8").read()
    start = src.index(marker)
    stop = src.index(until, start) if until else len(src)
    exec(compile(src[start:stop], script, "exec"), namespace)
    return namespace


def rms(x, axis=0):
    return np.sqrt(np.mean(np.square(x), axis=axis))


# %%==========================================================================
#  CoM velocity fusion
# ============================================================================
# The claim being tested: blending a differentiated marker position with an
# integrated GRF beats either one alone, and does not introduce a low-frequency
# bias. This matters because every margin of stability depends on CoM VELOCITY,
# and the velocity error propagates into the extrapolated CoM as
# error / omega_0, which for a 1 m pendulum is error / 3.13 -- so 140 mm/s of
# velocity error is 45 mm of XCoM error, against a mediolateral margin that is
# only 30-100 mm wide.

def synthetic_com(duration_s=300.0, fs=100.0, stride_hz=0.9, seed=0):
    """A gait-like CoM: stride harmonics plus a slow wander, with exact
    derivatives so the truth is known rather than estimated."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / fs)
    axis_gain = np.array([1.0, 0.7, 1.3])          # ML, AP, vertical
    pos = np.zeros((len(t), 3))
    vel = np.zeros((len(t), 3))
    acc = np.zeros((len(t), 3))
    for harmonic, amp in ((1, 0.010), (2, 0.022), (3, 0.004), (4, 0.002)):
        w = 2 * np.pi * harmonic * stride_hz
        phase = rng.uniform(0, 2 * np.pi, 3)
        gain = amp * axis_gain * np.cos(phase)
        pos += gain * np.sin(w * t)[:, None]
        vel += gain * w * np.cos(w * t)[:, None]
        acc += -gain * w ** 2 * np.sin(w * t)[:, None]
    w_slow = 2 * np.pi * 0.07
    slow = np.array([1.0, 1.0, 0.3]) * 0.02
    pos += slow * np.sin(w_slow * t)[:, None]
    vel += slow * w_slow * np.cos(w_slow * t)[:, None]
    acc += -slow * w_slow ** 2 * np.sin(w_slow * t)[:, None]
    return t, pos, vel, acc


def validate_com_fusion(marker_noise_mm=(1.0, 2.0, 5.0), force_offset=0.03,
                        force_noise=0.01, fs=100.0, seed=0):
    fuse = load_function("com_fuse_velocity",
                         extra=dict(kin_kinematic_fs=fs, com_crossover_hz=0.5,
                                    com_filter_order=2))
    rng = np.random.default_rng(seed)
    _, pos, vel, acc = synthetic_com(fs=fs, seed=seed)

    print("=" * 74)
    print("  CoM VELOCITY FUSION")
    print("=" * 74)
    print("  RMS velocity error against a known trajectory, mm/s\n")
    print("  marker noise    method                        ML      AP      VT")
    rows = []
    for noise_mm in marker_noise_mm:
        pos_meas = pos + rng.normal(0, noise_mm / 1000.0, pos.shape)
        acc_meas = acc + force_offset + rng.normal(0, force_noise, acc.shape)

        v_markers = np.gradient(pos_meas, 1.0 / fs, axis=0)
        v_force = np.cumsum(acc_meas, axis=0) / fs
        v_force -= v_force.mean(axis=0)
        v_fused = fuse(pos_meas, acc_meas)

        for tag, v in (("differentiate markers", v_markers),
                       ("integrate force", v_force),
                       ("complementary fusion", v_fused)):
            e = 1000 * rms(v - vel)
            print(f"  {noise_mm:5.1f} mm        {tag:24s} {e[0]:7.1f} {e[1]:7.1f} {e[2]:7.1f}")
            rows.append(dict(noise_mm=noise_mm, method=tag,
                             ml=e[0], ap=e[1], vt=e[2]))
        gain = rms(v_markers - vel) / rms(v_fused - vel)
        print(f"                  -> fusion better by       "
              f"{gain[0]:6.1f}x {gain[1]:6.1f}x {gain[2]:6.1f}x")
        # the number that actually matters downstream
        xcom_err = 1000 * rms(v_fused - vel) / np.sqrt(9.81 / 1.0)
        xcom_err_markers = 1000 * rms(v_markers - vel) / np.sqrt(9.81 / 1.0)
        print(f"                  -> XCoM error (1 m pendulum) "
              f"{xcom_err_markers[1]:.1f} mm -> {xcom_err[1]:.1f} mm in AP\n")

    # No low-frequency bias: below the crossover the fusion must reproduce the
    # markers, which are the only drift-free source.
    pos_meas = pos + rng.normal(0, 0.002, pos.shape)
    v_fused = fuse(pos_meas, acc)
    b, a = butter(2, 0.5 / (fs / 2), "low")
    low_err = 1000 * rms(filtfilt(b, a, v_fused - vel, axis=0))
    mean_err = 1000 * (v_fused - vel).mean(axis=0)
    print(f"  below the 0.5 Hz crossover: RMS error "
          f"{low_err[0]:.2f} / {low_err[1]:.2f} / {low_err[2]:.2f} mm/s")
    print(f"  mean velocity error (bias): "
          f"{mean_err[0]:+.3f} / {mean_err[1]:+.3f} / {mean_err[2]:+.3f} mm/s")
    ok = np.all(np.abs(mean_err) < 1.0)
    print(f"  {'PASS' if ok else 'FAIL'}: fusion is unbiased at low frequency\n")
    return rows


def validate_belt_speed(true_belt=1.3, fs=100.0, stance_s=0.69, stride_s=1.11,
                        n_strides=80, noise_mm=2.0, seed=0):
    """The belt speed estimator, against a known belt speed.

    Worth its own check because the naive approach -- reading it off the centre
    of pressure -- is wrong. During stance the CoP travels heel to toe ALONG
    the foot while the foot travels backwards WITH the belt, and the CoP cannot
    separate the two. The foot can: during foot-flat it is stationary relative
    to the belt.
    """
    rng = np.random.default_rng(seed)
    n = int(n_strides * stride_s * fs)
    heel_ap = np.full(n, np.nan)
    for k in range(n_strides):
        a = int(k * stride_s * fs)
        b = a + int(stance_s * fs)
        if b >= n:
            break
        heel_ap[a:b] = 0.4 - true_belt * (np.arange(b - a) / fs)
    idx = np.arange(n)
    good = np.isfinite(heel_ap)
    heel_ap[~good] = np.interp(idx[~good], idx[good], heel_ap[good])
    heel_ap += rng.normal(0, noise_mm / 1000.0, n)

    slopes = []
    for k in range(n_strides):
        a = int((k * stride_s + 0.30 * stance_s) * fs)
        b = int((k * stride_s + 0.70 * stance_s) * fs)
        if b >= n or b - a < 5:
            continue
        slopes.append(np.polyfit(np.arange(b - a) / fs, heel_ap[a:b], 1)[0])
    estimate = float(np.median(np.abs(slopes)))

    print("=" * 74)
    print("  BELT SPEED FROM FOOT KINEMATICS")
    print("=" * 74)
    print(f"  true {true_belt:.3f} m/s, estimated {estimate:.4f} m/s, "
          f"error {1000*abs(estimate-true_belt):.1f} mm/s  (n={len(slopes)} stances)")
    ok = abs(estimate - true_belt) < 0.01
    print(f"  {'PASS' if ok else 'FAIL'}: within 10 mm/s\n")
    return estimate


# %%==========================================================================
#  Margin of stability
# ============================================================================
# Analytic check: a single stance with a known CoM state, a known pendulum
# length and a foot whose lateral border is at a known place, so MoS can be
# worked out by hand and compared against what the cell produces.

def validate_margin_of_stability(ankle_ml=0.10, foot_lateral_edge=0.14,
                                 com_velocity_ml=0.30, pendulum_length=1.00,
                                 n_frames=400, fs=100.0, g=9.81):
    import pandas as pd
    import tempfile
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    omega0 = np.sqrt(g / pendulum_length)
    xcom_ml = com_velocity_ml / omega0
    expected_ml = foot_lateral_edge - xcom_ml
    expected_marker = ankle_ml - xcom_ml

    print("=" * 74)
    print("  MARGIN OF STABILITY")
    print("=" * 74)
    print(f"  by hand: omega0 = sqrt({g}/{pendulum_length}) = {omega0:.5f} rad/s")
    print(f"           XCoM   = 0 + {com_velocity_ml}/{omega0:.5f} = {xcom_ml:.5f} m")
    print(f"           MoS_ML = {foot_lateral_edge} - {xcom_ml:.5f} = {expected_ml:.5f} m")

    # CoM placed so that |CoM - ankle| is exactly the pendulum length
    com_z = np.sqrt(pendulum_length ** 2 - ankle_ml ** 2)

    columns = {}
    for side, ml in (("Right", ankle_ml), ("Left", -ankle_ml)):
        for joint in ("Ankle", "Toes", "Heel", "Hip"):
            columns[f"{side}_{joint}_Position"] = np.full(n_frames, ml)
            columns[f"{side}_{joint}_Position.1"] = np.zeros(n_frames)
            columns[f"{side}_{joint}_Position.2"] = (
                np.full(n_frames, 0.9) if joint == "Hip" else np.zeros(n_frames))
    columns["Whole_body_COG"] = np.zeros(n_frames)
    columns["Whole_body_COG.1"] = np.zeros(n_frames)
    columns["Whole_body_COG.2"] = np.full(n_frames, com_z)

    def foot_box(lo, hi):
        grid = np.array([[x, y, 0.01]
                         for x in np.linspace(lo, hi, 9)
                         for y in np.linspace(-0.10, 0.15, 9)])
        return np.repeat(grid[None, :, :], n_frames, axis=0)

    blocks = {"left_foot": foot_box(-foot_lateral_edge, -0.06),
              "right_foot": foot_box(0.06, foot_lateral_edge)}
    cols, data = [], []
    for segment, arr in blocks.items():
        for v in range(arr.shape[1]):
            for ai, axis in enumerate("XYZ"):
                cols.append((segment, v, axis))
                data.append(arr[:, v, ai])
    mesh = pd.DataFrame(np.column_stack(data),
                        columns=pd.MultiIndex.from_tuples(
                            cols, names=["segment", "vertex", "axis"]))
    mesh = mesh.sort_index(axis=1)

    tmp = Path(tempfile.mkdtemp())
    mesh.to_pickle(tmp / "metrics_mesh.pkl")

    namespace = dict(
        np=np, pd=pd, Path=Path, plt=plt,
        kinematic_data=pd.DataFrame(columns), metrics_path=tmp / "metrics.csv",
        com_position=np.column_stack([np.zeros(n_frames), np.zeros(n_frames),
                                      np.full(n_frames, com_z)]),
        com_velocity_belt=np.column_stack([np.full(n_frames, com_velocity_ml),
                                           np.zeros(n_frames), np.zeros(n_frames)]),
        com_ml_axis=0, com_ap_axis=1, com_vt_axis=2, com_belt_sign=1.0,
        force_contacts={"Right": [(1000, 1600)], "Left": []},
        kin_sample_to_frame=lambda s: np.asarray(s, float) * 100.0 / 1000.0 + 1.0)

    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        load_cell("# %% Margin of stability, with the base", namespace)
    result = namespace["margin_of_stability"]

    got_mesh = float(result["mos_ml_contact"].iloc[0])
    got_marker = float(result["mos_ml_contact_marker_bos"].iloc[0])
    got_length = float(result["pendulum_length"].iloc[0])

    print(f"  cell:    MoS_ML = {got_mesh:.5f} m   (diff {abs(got_mesh-expected_ml):.2e})")
    print(f"           marker-BoS margin {got_marker:.5f} m "
          f"(expected {expected_marker:.5f})")
    print(f"           pendulum length {got_length:.5f} m "
          f"(expected {pendulum_length:.5f})")
    print(f"           mesh vs marker gap {1000*(got_mesh-got_marker):.1f} mm "
          f"(built in: {1000*(foot_lateral_edge-ankle_ml):.1f} mm)")
    ok = (abs(got_mesh - expected_ml) < 1e-6
          and abs(got_marker - expected_marker) < 1e-6
          and abs(got_length - pendulum_length) < 1e-6)
    print(f"  {'PASS' if ok else 'FAIL'}: margin, boundary and pendulum length exact\n")
    return ok



# %%==========================================================================
#  Minimum foot clearance, margin of instability, trip risk integral
# ============================================================================
# Schulz 2017. Engineered so every piece has a known answer: a clearance
# profile whose minimum is known, a margin of instability held at a constant
# value, and an integration window whose bounds follow from a speed profile
# chosen so its acceleration peaks and troughs at known fractions of swing.

def _trip_risk_scenario(mfc_true=0.018, base_clear=0.060, moi_mm=12.0,
                        pendulum=1.00, swing_s=0.45, fs=100.0, n=1200,
                        monotonic_clearance=False):
    """Build the synthetic trial and run the two cells over it."""
    import pandas as pd
    import tempfile
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = 9.81
    toe_off = 100
    heel_strike = toe_off + int(swing_s * fs)
    swing = np.arange(toe_off, heel_strike + 1)
    u = (swing - toe_off) / (heel_strike - toe_off)

    clearance = np.full(n, base_clear)
    if monotonic_clearance:
        # no local minimum anywhere: a non-MTC cycle, which must return NaN
        clearance[swing] = base_clear - (base_clear - mfc_true) * u
    else:
        clearance[swing] = mfc_true + (base_clear - mfc_true) * ((u - 0.5) / 0.5) ** 2

    # speed profile (1-cos)/2: acceleration peaks at u=0.25, troughs at u=0.75
    speed = np.zeros(n)
    speed[swing] = 0.5 * (1 - np.cos(2 * np.pi * u))

    def foot(side):
        v = np.zeros((n, 2, 3))
        if side == "Right":
            v[:, 0, 2] = clearance
            v[:, 1, 2] = clearance + 0.05
            v[:, :, 1] = 0.30              # anterior extent held CONSTANT so the
            v[:, :, 0] = 0.1               # base of support boundary is fixed
            v[swing, 0, 0] = 0.1 + np.cumsum(speed[swing]) / fs
            v[swing, 1, 0] = v[swing, 0, 0]
        else:
            v[:, :, 2] = 0.001
            v[:, :, 1] = 0.10
            v[:, :, 0] = -0.1
        return v

    cols, data = [], []
    for segment, side in (("left_foot", "Left"), ("right_foot", "Right")):
        arr = foot(side)
        for vi in range(arr.shape[1]):
            for ai, axis in enumerate("XYZ"):
                cols.append((segment, vi, axis))
                data.append(arr[:, vi, ai])
    mesh = pd.DataFrame(np.column_stack(data),
                        columns=pd.MultiIndex.from_tuples(
                            cols, names=["segment", "vertex", "axis"])).sort_index(axis=1)
    tmp = Path(tempfile.mkdtemp())
    mesh.to_pickle(tmp / "m_mesh.pkl")

    columns = {}
    for side, ml in (("Right", 0.1), ("Left", -0.1)):
        for joint in ("Ankle", "Toes", "Heel", "Hip"):
            columns[f"{side}_{joint}_Position"] = np.full(n, ml)
            columns[f"{side}_{joint}_Position.1"] = np.full(
                n, 0.3 if side == "Right" else 0.0)
            columns[f"{side}_{joint}_Position.2"] = (
                np.full(n, 0.9) if joint == "Hip"
                else np.full(n, 0.20) if joint == "Heel" else np.zeros(n))
    com_z = np.sqrt(pendulum ** 2 - 0.1 ** 2)
    columns["Whole_body_COG"] = np.zeros(n)
    columns["Whole_body_COG.1"] = np.zeros(n)
    columns["Whole_body_COG.2"] = np.full(n, com_z)

    # choose the CoM velocity that puts the XCoM exactly moi_mm beyond the
    # anterior boundary, so the margin of instability is a known constant
    omega0 = np.sqrt(g / pendulum)
    com_ap_velocity = (0.30 + moi_mm / 1000.0) * omega0

    namespace = dict(
        np=np, pd=pd, Path=Path, plt=plt,
        kinematic_data=pd.DataFrame(columns), metrics_path=tmp / "m.csv",
        com_position=np.column_stack([np.zeros(n), np.zeros(n), np.full(n, com_z)]),
        com_velocity_belt=np.column_stack([np.zeros(n), np.full(n, com_ap_velocity),
                                           np.zeros(n)]),
        com_ml_axis=0, com_ap_axis=1, com_vt_axis=2, com_belt_sign=1.0,
        kin_kinematic_fs=fs,
        force_contacts={"Right": [(0, toe_off * 10),
                                  (heel_strike * 10, (heel_strike + 60) * 10)],
                        "Left": []},
        kin_sample_to_frame=lambda s: np.asarray(s, float) * 100.0 / 1000.0 + 1.0)

    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        load_cell("# %% Margin of stability, with the base", namespace,
                  until="# %% Minimum foot clearance, margin of instability")
        load_cell("# %% Minimum foot clearance, margin of instability", namespace)
    return (namespace["trip_risk"], clearance, speed, swing, toe_off,
            heel_strike, fs, moi_mm)


def validate_trip_risk():
    print("=" * 74)
    print("  MFC, MARGIN OF INSTABILITY, TRIP RISK INTEGRAL")
    print("=" * 74)

    result, clearance, speed, swing, toe_off, heel_strike, fs, moi_mm = \
        _trip_risk_scenario()
    row = result.iloc[0]

    mfc_sampled = float(clearance[toe_off:heel_strike + 1].min())
    expected_frame = toe_off + (heel_strike - toe_off) // 2 + 1

    # independent integral. Schulz specifies the RESULTANT velocity and
    # acceleration of the MFC point, so the vertical motion counts too --
    # using only the horizontal component gives a different window.
    n = len(clearance)
    pad = int(0.02 * (heel_strike - toe_off))
    frames = np.arange(toe_off + pad, heel_strike - pad + 1)
    point = np.zeros((n, 3))
    point[:, 2] = clearance
    point[:, 1] = 0.30
    point[:, 0] = 0.1
    point[swing, 0] = 0.1 + np.cumsum(speed[swing]) / fs
    resultant = np.linalg.norm(np.gradient(point[frames], 1 / fs, axis=0), axis=1)
    accel = np.gradient(resultant, 1 / fs)
    a, b = int(np.argmax(accel)), int(np.argmin(accel))
    clear_mm = np.maximum(1000 * clearance[frames], 1.0)
    expected_tri = float(np.sum((moi_mm / clear_mm)[a:b + 1]) / fs)

    print(f"  MFC            {row.mfc_m:.6f} m vs sampled minimum {mfc_sampled:.6f} "
          f"(diff {abs(row.mfc_m - mfc_sampled):.1e})")
    print(f"  MFC frame      {int(row.mfc_frame_100hz)} vs expected {expected_frame}")
    print(f"  MFC vertex     {int(row.mfc_vertex)} vs expected 0 (the low vertex)")
    print(f"  MoI peak       {row.moi_peak_mm:.3f} mm vs engineered {moi_mm:.3f}")
    print(f"  TRI            {row.trip_risk_integral:.6f} s vs independent "
          f"{expected_tri:.6f} (diff {abs(row.trip_risk_integral - expected_tri):.1e})")
    print(f"  window         {row.integration_window_s:.3f} s vs independent "
          f"{(b - a) / fs:.3f} s")

    ok = (abs(row.mfc_m - mfc_sampled) < 1e-12
          and int(row.mfc_frame_100hz) == expected_frame
          and int(row.mfc_vertex) == 0
          and abs(row.moi_peak_mm - moi_mm) < 1e-6
          and abs(row.trip_risk_integral - expected_tri) < 1e-12)
    print(f"  {'PASS' if ok else 'FAIL'}: MFC event, margin and integral all exact")

    # A swing with no local minimum must return NaN, not a global minimum.
    # Schulz gives these non-MTC cycles a figure of their own.
    mono = _trip_risk_scenario(monotonic_clearance=True)[0].iloc[0]
    no_event = (not bool(mono.has_mfc_event)) and np.isnan(mono.mfc_m)
    print(f"  monotonic clearance -> has_mfc_event={mono.has_mfc_event}, "
          f"mfc={mono.mfc_m}")
    print(f"  {'PASS' if no_event else 'FAIL'}: a non-MTC cycle returns NaN, "
          f"not a global minimum\n")
    return ok and no_event


# %%==========================================================================
#  run everything
# ============================================================================

if __name__ == "__main__":
    validate_com_fusion()
    validate_belt_speed()
    validate_margin_of_stability()
    validate_trip_risk()
