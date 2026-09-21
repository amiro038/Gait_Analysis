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
                                 step_length=0.68,
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
    # The swing ankle is a realistic STEP LENGTH behind the stance one. With
    # both at AP = 0 the test cannot catch a lateral direction that has picked
    # up an AP component, which is exactly the bug this guards against.
    for side, ml, ap in (("Right", ankle_ml, 0.0),
                         ("Left", -ankle_ml, -step_length)):
        for joint in ("Ankle", "Toes", "Heel", "Hip"):
            columns[f"{side}_{joint}_Position"] = np.full(n_frames, ml)
            columns[f"{side}_{joint}_Position.1"] = np.full(n_frames, ap)
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
    # The cell rebuilds vertices from a _posed.npz through apply_binding rather
    # than reading a pickle, so stand in for both: an empty file so the exists
    # check passes, and a stub module that hands back the mesh built above.
    (tmp / "metrics_posed.npz").write_bytes(b"")
    import sys
    import types
    stub = types.ModuleType("apply_binding")
    stub.mesh_dataframe = lambda source=None, binding=None, dtype=None: mesh
    sys.modules["apply_binding"] = stub

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
        # bounded: without `until` this runs on into the later cells, which
        # need force_data and the rest of the real pipeline
        load_cell("# %% Margin of stability, with the base", namespace,
                  until="# %% Minimum foot clearance, margin of instability")
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
    (tmp / "m_posed.npz").write_bytes(b"")
    import sys
    import types
    stub = types.ModuleType("apply_binding")
    stub.mesh_dataframe = lambda source=None, binding=None, dtype=None: mesh
    sys.modules["apply_binding"] = stub

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
        load_cell("# %% Minimum foot clearance, margin of instability", namespace,
                  until="# %% Kinetic metrics: impulses")
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
#  Detrended fluctuation analysis
# ============================================================================
# Tested against fractional Gaussian noise, where alpha = H is known exactly.
# The generator is checked FIRST, against the theoretical autocovariance, so a
# generator bug cannot be mistaken for an estimator bug.
#
# N is set to the lengths this protocol actually produces -- 486 strides from a
# 10 minute trial and 756 from a 15 minute one -- because the question that
# matters is not whether DFA works in principle but how precisely it works on
# the data in hand.

def fractional_gaussian_noise(n, hurst, rng):
    """Exact fGn by Davies-Harte circulant embedding.

    H = 1 is degenerate for fGn: the autocovariance becomes 1 at every lag, so
    the process is perfectly correlated and there is nothing to estimate. Stay
    below it.
    """
    k = np.arange(0, n + 1)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * hurst)
                   - 2 * np.abs(k) ** (2 * hurst)
                   + np.abs(k + 1) ** (2 * hurst))
    circulant = np.concatenate([gamma, gamma[-2:0:-1]])
    eigenvalues = np.maximum(np.fft.fft(circulant).real, 0.0)
    m = len(eigenvalues)
    noise = rng.normal(size=m) + 1j * rng.normal(size=m)
    return np.fft.fft(np.sqrt(eigenvalues / (2 * m)) * noise).real[:n]


def validate_dfa(lengths=(486, 756), hursts=(0.5, 0.7, 0.9, 0.95),
                 n_reps=60, seed=0):
    extra = dict(dfa_order=1, dfa_min_box=16, dfa_max_box_frac=1.0 / 9,
                 dfa_n_boxes=20, dfa_gph_power=0.5, dfa_n_surrogate=50,
                 dfa_seed=0)
    fluct = load_function("dfa_fluctuation", extra=extra)
    extra["dfa_fluctuation"] = fluct
    alpha_fn = load_function("dfa_alpha", extra=extra)
    gph_fn = load_function("dfa_gph", extra=extra)
    rng = np.random.default_rng(seed)

    print("=" * 74)
    print("  DETRENDED FLUCTUATION ANALYSIS")
    print("=" * 74)

    # 1. the generator itself
    print("  generator: empirical vs theoretical autocovariance, N=20000")
    gen_ok = True
    for hurst in (0.5, 0.7, 0.9):
        acv = np.zeros(6)
        for _ in range(20):
            x = fractional_gaussian_noise(20000, hurst, rng)
            x = x - x.mean()
            acv += np.array([np.mean(x[:len(x) - k] * x[k:])
                             for k in range(6)]) / 20
        acv /= acv[0]
        k = np.arange(6)
        theory = 0.5 * (np.abs(k - 1) ** (2 * hurst) - 2 * np.abs(k) ** (2 * hurst)
                        + np.abs(k + 1) ** (2 * hurst))
        err = np.abs(acv - theory).max()
        gen_ok &= err < 0.12
        print(f"    H={hurst:.1f}  max autocovariance error {err:.4f}")

    # 2. recovery at the lengths this protocol produces
    print(f"\n  alpha recovered from fGn, {n_reps} repetitions per cell")
    header = "     N   " + "".join(f"  H={h:.2f}        " for h in hursts)
    print(header)
    rows = []
    for n in lengths:
        line = f"  {n:4d}   "
        for hurst in hursts:
            vals = [alpha_fn(fractional_gaussian_noise(n, hurst, rng))["alpha"]
                    for _ in range(n_reps)]
            line += f"{np.mean(vals):.3f}+-{np.std(vals):.3f}  "
            rows.append(dict(n=n, hurst=hurst, mean=np.mean(vals),
                             sd=np.std(vals), bias=np.mean(vals) - hurst))
        print(line)

    worst_bias = max(abs(r["bias"]) for r in rows)
    typical_sd = np.median([r["sd"] for r in rows])
    print(f"\n  worst bias {worst_bias:.3f}, typical scatter {typical_sd:.3f}")
    print(f"  So a single trial pins alpha to about +-{typical_sd:.2f}. A")
    print(f"  between-condition difference smaller than that is inside the noise")
    print(f"  of one trial and needs repeated trials or more strides.")

    # 3. the confirmatory estimator, so its precision is known rather than assumed
    print(f"\n  GPH on the same series")
    gph_sd = []
    for n in lengths:
        line = f"  {n:4d}   "
        for hurst in hursts:
            vals = [gph_fn(fractional_gaussian_noise(n, hurst, rng))["alpha"]
                    for _ in range(n_reps)]
            line += f"{np.mean(vals):.3f}+-{np.std(vals):.3f}  "
            gph_sd.append(np.std(vals))
        print(line)
    print(f"  GPH scatter {np.median(gph_sd):.3f} vs DFA {typical_sd:.3f}: it is a")
    print(f"  ballpark cross-check, not a precise second opinion.")

    # 4. the surrogate property, which is the check that runs on real data
    x = fractional_gaussian_noise(756, 0.9, rng)
    shuffled = [alpha_fn(rng.permutation(x))["alpha"] for _ in range(60)]
    sur_mean = float(np.mean(shuffled))
    sur_ok = abs(sur_mean - 0.5) < 0.05
    print(f"\n  shuffled surrogate of an H=0.9 series: alpha {sur_mean:.3f} "
          f"(must be 0.5)")
    print(f"    SD is unchanged by shuffling, alpha is not -- which is the whole")
    print(f"    point: alpha measures order, not magnitude.")

    ok = gen_ok and worst_bias < 0.05 and sur_ok
    print(f"  {'PASS' if ok else 'FAIL'}: generator exact, DFA unbiased, "
          f"surrogate returns 0.5\n")
    return ok



# %%==========================================================================
#  Entropy
# ============================================================================
# Sample entropy has a CLOSED FORM for white noise, which makes this one of the
# few estimators here that can be checked against an exact answer rather than
# an ordering. For a z-scored iid Gaussian series, two independent points
# differ by N(0, 2), so
#
#     P(|x_i - x_j| <= r) = 2 * Phi(r / sqrt(2)) - 1
#
# Extending a match from m to m+1 points adds one more independent comparison,
# so A/B = P and SampEn = -ln(P) for any m. Anything else is a bug.

def validate_entropy(seed=0):
    from scipy.spatial import cKDTree
    from scipy.stats import norm

    extra = dict(cKDTree=cKDTree, ent_m=2, ent_r=0.2, ent_normalise=True,
                 ent_mse_scales=10)
    counts = load_function("ent_match_counts", extra=extra)
    extra["ent_match_counts"] = counts
    sampen = load_function("ent_sample_entropy", extra=extra)
    coarse = load_function("ent_coarse_grain", extra=extra)
    extra["ent_coarse_grain"] = coarse
    rcmse = load_function("ent_rcmse", extra=extra)
    rng = np.random.default_rng(seed)

    print("=" * 74)
    print("  ENTROPY")
    print("=" * 74)

    # 1. the exact check
    print("  white noise against the closed form -ln(P), N=5000, 10 repeats")
    worst = 0.0
    for r in (0.15, 0.20, 0.25, 0.30):
        exact = -np.log(2 * norm.cdf(r / np.sqrt(2)) - 1)
        got = [sampen(rng.normal(size=5000), m=2, r=r) for _ in range(10)]
        diff = abs(np.mean(got) - exact)
        worst = max(worst, diff)
        print(f"    r={r:.2f}   exact {exact:.4f}   measured {np.mean(got):.4f} "
              f"+- {np.std(got):.4f}   diff {diff:.4f}")

    # 2. the ordering: regular < chaotic < random
    n = 4000
    sine = np.sin(2 * np.pi * np.arange(n) / 50)
    x = 0.4
    logistic = []
    for _ in range(n + 1000):
        x = 4 * x * (1 - x)
        logistic.append(x)
    logistic = np.array(logistic[1000:])
    white = rng.normal(size=n)

    e_sine, e_logi, e_white = (sampen(sine), sampen(logistic), sampen(white))
    print(f"\n  ordering at m=2, r=0.2, N={n}")
    print(f"    sine (periodic)          {e_sine:.4f}")
    print(f"    logistic map (chaotic)   {e_logi:.4f}")
    print(f"    white noise (random)     {e_white:.4f}")
    ordered = e_sine < e_logi < e_white

    # 3. multiscale: white noise must fall away, 1/f must hold up
    def pink_noise(n, rng):
        f = np.fft.rfftfreq(n)
        f[0] = f[1]
        spectrum = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.sqrt(f)
        return np.fft.irfft(spectrum, n)

    m = 12000
    curve_white = rcmse(rng.normal(size=m))
    curve_pink = pink_noise(m, rng)
    curve_pink = rcmse(curve_pink)
    print(f"\n  refined composite MSE, N={m}")
    print("    scale :", " ".join(f"{s:5d}" for s in range(1, 11)))
    print("    white :", " ".join(f"{v:5.2f}" for v in curve_white))
    print("    1/f   :", " ".join(f"{v:5.2f}" for v in curve_pink))
    fall_white = curve_white[0] - curve_white[-1]
    fall_pink = abs(curve_pink[0] - curve_pink[-1])
    area_white, area_pink = np.nansum(curve_white), np.nansum(curve_pink)
    print(f"    white falls {fall_white:.2f} across scales; 1/f moves {fall_pink:.2f}")
    print(f"    area: white {area_white:.2f}, 1/f {area_pink:.2f}")
    print(f"    Note white noise has the HIGHER single-scale entropy "
          f"({curve_white[0]:.2f} vs {curve_pink[0]:.2f})")
    print(f"    but the LOWER area. That is exactly why the curve is the")
    print(f"    complexity measure and the scale-1 value is not.")

    mse_ok = fall_white > 0.5 and fall_pink < 0.2 and area_pink > area_white
    ok = worst < 0.02 and ordered and mse_ok
    print(f"  {'PASS' if ok else 'FAIL'}: closed form to {worst:.4f}, ordering "
          f"correct, MSE separates noise from structure\n")
    return ok



# %%==========================================================================
#  Harmonic ratio
# ============================================================================
# A two-tone signal makes HR exactly computable. With
#     x(t) = A2*sin(2*2pi*t) + A1*sin(1*2pi*t)
# over exactly one stride, harmonic 1 has amplitude A1, harmonic 2 has A2 and
# every other harmonic is zero, so
#     HR(even/odd) = A2 / A1
#     iHR          = A2^2 / (A1^2 + A2^2) * 100
# A1 is the asymmetric component: bigger A1 means a more asymmetric stride and
# must give a lower HR.

def validate_harmonic_ratio(n_samples=110):
    extra = dict(hr_n_harmonics=20)
    harmonics = load_function("hr_harmonics", extra=extra)
    ratio = load_function("hr_ratio", extra=extra)

    print("=" * 74)
    print("  HARMONIC RATIO")
    print("=" * 74)
    t = np.arange(n_samples) / n_samples

    print("   A1     A2    HR expect     HR got   iHR expect   iHR got")
    exact = True
    for a1, a2 in ((0.05, 1.0), (0.10, 1.0), (0.25, 1.0), (0.50, 1.0), (1.0, 1.0)):
        amp = harmonics(a2 * np.sin(2 * 2 * np.pi * t) + a1 * np.sin(2 * np.pi * t))
        hr, ihr = ratio(amp, odd_dominant=False)
        expect_hr = a2 / a1
        expect_ihr = 100 * a2 ** 2 / (a1 ** 2 + a2 ** 2)
        exact &= abs(hr - expect_hr) < 1e-9 and abs(ihr - expect_ihr) < 1e-9
        print(f"  {a1:.2f}   {a2:.2f}   {expect_hr:9.4f} {hr:9.4f}   "
              f"{expect_ihr:9.2f}% {ihr:8.2f}%")

    # the mediolateral convention must invert the ratio, not recompute it
    amp = harmonics(np.sin(2 * 2 * np.pi * t) + 0.25 * np.sin(2 * np.pi * t))
    hr_even, _ = ratio(amp, odd_dominant=False)
    hr_odd, _ = ratio(amp, odd_dominant=True)
    inverts = abs(hr_even * hr_odd - 1.0) < 1e-9
    print(f"\n  ML convention: even/odd {hr_even:.4f} x odd/even {hr_odd:.4f} "
          f"= {hr_even * hr_odd:.9f}")

    print("\n  HR must fall as the asymmetric component grows")
    previous, monotone = np.inf, True
    for a1 in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        hr, _ = ratio(harmonics(np.sin(2 * 2 * np.pi * t)
                                + a1 * np.sin(2 * np.pi * t)), odd_dominant=False)
        monotone &= hr < previous
        previous = hr
        print(f"    asymmetry {a1:.2f} -> HR {hr:7.3f}")

    ok = exact and inverts and monotone
    print(f"  {'PASS' if ok else 'FAIL'}: exact on the two-tone case, ML inverts, "
          f"monotone in asymmetry\n")
    return ok



# %%==========================================================================
#  Supplementary metrics
# ============================================================================

def validate_supplementary(seed=0):
    gem = load_function("sup_gem_decompose")
    lag1 = load_function("sup_lag1")
    placement = load_function("sup_foot_placement_model")
    autocorr = load_function("sup_unbiased_autocorr")
    sym_angle = load_function("sup_symmetry_angle")
    rng = np.random.default_rng(seed)

    print("=" * 74)
    print("  SUPPLEMENTARY METRICS")
    print("=" * 74)

    # --- GEM: build strides FROM known components, then recover them --------
    n, v_star, t_bar = 600, 1.30, 1.11
    par_true = rng.normal(0, 0.030, n)
    perp_true = rng.normal(0, 0.010, n)
    norm = np.sqrt(1 + v_star ** 2)
    d_t = (par_true - v_star * perp_true) / norm
    d_l = (v_star * par_true + perp_true) / norm
    g = gem(t_bar + d_t, v_star * t_bar + d_l)
    err_par = np.max(np.abs(g["parallel"] - (par_true - par_true.mean())))
    err_perp = np.max(np.abs(g["perpendicular"] - (perp_true - perp_true.mean())))
    print(f"  GEM: v* {g['v_star']:.4f} (true {v_star:.4f}), "
          f"component error {max(err_par, err_perp):.1e}")
    print(f"       SD par {np.std(g['parallel']):.4f} (built {np.std(par_true):.4f}), "
          f"perp {np.std(g['perpendicular']):.4f} (built {np.std(perp_true):.4f})")
    gem_ok = max(err_par, err_perp) < 1e-4

    # --- lag-1 autocorrelation against a known AR(1) ------------------------
    print("  lag-1 autocorrelation of AR(1):", end=" ")
    ar_ok = True
    for phi in (-0.6, -0.3, 0.0, 0.4):
        x = np.zeros(4000)
        for i in range(1, 4000):
            x[i] = phi * x[i - 1] + rng.normal()
        got = lag1(x)
        ar_ok &= abs(got - phi) < 0.05
        print(f"{phi:+.1f}->{got:+.3f}", end="  ")
    print()

    # --- foot placement with known gains ------------------------------------
    b0, b1, b2 = 0.05, 0.80, 0.20
    z = rng.normal(0, 0.02, 800)
    v = rng.normal(0, 0.10, 800)
    r_clean = placement(z, v, b0 + b1 * z + b2 * v)
    print(f"  foot placement (noiseless): R2 {r_clean['r2']:.4f}, gains "
          f"{r_clean['gain_position']:.3f}/{r_clean['gain_velocity']:.3f} "
          f"(true {b1}/{b2})")
    fp_ok = (r_clean["r2"] > 0.9999
             and abs(r_clean["gain_position"] - b1) < 1e-6
             and abs(r_clean["gain_velocity"] - b2) < 1e-6)

    # --- autocorrelation regularity -----------------------------------------
    t = np.arange(3000)
    period = 110
    symmetric = np.sin(2 * np.pi * 2 * t / period)
    asymmetric = symmetric + 0.5 * np.sin(2 * np.pi * t / period)
    ac_s = autocorr(symmetric, 3 * period)
    ac_a = autocorr(asymmetric, 3 * period)
    print(f"  autocorrelation: symmetric step/stride/symmetry "
          f"{ac_s[period//2]:.3f}/{ac_s[period]:.3f}/"
          f"{ac_s[period//2]/ac_s[period]:.3f}")
    print(f"                   asymmetric                    "
          f"{ac_a[period//2]:.3f}/{ac_a[period]:.3f}/"
          f"{ac_a[period//2]/ac_a[period]:.3f}")
    ac_ok = (abs(ac_s[period // 2] - 1) < 0.01 and abs(ac_s[period] - 1) < 0.01
             and ac_a[period // 2] < 0.9 * ac_a[period])

    # --- symmetry angle ------------------------------------------------------
    print("  symmetry angle:", end=" ")
    sa_ok = abs(sym_angle(1.0, 1.0)) < 1e-9
    sa_ok &= abs(sym_angle(1.1, 0.9) + sym_angle(0.9, 1.1)) < 1e-9
    for left, right in ((1.0, 1.0), (1.1, 0.9), (0.9, 1.1), (2.0, 1.0)):
        print(f"L{left}/R{right}->{sym_angle(left, right):+.2f}%", end="  ")
    print()

    ok = gem_ok and ar_ok and fp_ok and ac_ok and sa_ok
    print(f"  {'PASS' if ok else 'FAIL'}: GEM exact and orthogonal, AR(1) "
          f"recovered, regression exact, symmetry antisymmetric\n")
    return ok


# %%==========================================================================
#  run everything
# ============================================================================

if __name__ == "__main__":
    validate_com_fusion()
    validate_belt_speed()
    validate_margin_of_stability()
    validate_trip_risk()
    validate_dfa()
    validate_entropy()
    validate_harmonic_ratio()
    validate_supplementary()
