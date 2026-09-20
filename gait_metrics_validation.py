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
#  run everything
# ============================================================================

if __name__ == "__main__":
    validate_com_fusion()
    validate_belt_speed()
