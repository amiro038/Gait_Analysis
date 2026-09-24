# -*- coding: utf-8 -*-
"""
Validation of the gait/ metrics on signals whose answers are known.

Before a number is used to compare conditions, establish what it does when
the answer is known. Every section imports the SHIPPED function from gait/
and runs it on a constructed signal: the CoM velocity fusion, the margin of
stability, MFC and the trip risk integral, DFA and its confidence interval,
the block bootstrap, entropy, the harmonic ratio, regularity, the goal-
equivalent manifold, foot placement and the symmetry angle.

The Lyapunov exponent has its own script, lds_validation.py; the whole
pipeline end to end is test_gait_analysis.py.

Run the whole file (about a minute), or one cell at a time in Spyder.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gait import stability, uncertainty as unc, variability as var  # noqa: E402
from gait.trial import complementary  # noqa: E402

RESULTS = {}


def rms(x, axis=0):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def verdict(name, ok, text):
    RESULTS[name] = ok
    print(f"  {'PASS' if ok else 'FAIL'}: {text}\n")


# %%==========================================================================
#  CoM velocity fusion
# ============================================================================
# Blending a differentiated marker CoM with an integrated GRF must beat
# either alone and must not bias the low frequencies. It matters because
# velocity error reaches the extrapolated CoM as error / omega_0.

def synthetic_com(duration_s=300.0, fs=100.0, stride_hz=0.9, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / fs)
    gain_axis = np.array([1.0, 0.7, 1.3])
    pos, vel, acc = (np.zeros((len(t), 3)) for _ in range(3))
    for h, amp in ((1, 0.010), (2, 0.022), (3, 0.004), (4, 0.002)):
        w = 2 * np.pi * h * stride_hz
        g = amp * gain_axis * np.cos(rng.uniform(0, 2 * np.pi, 3))
        pos += g * np.sin(w * t)[:, None]
        vel += g * w * np.cos(w * t)[:, None]
        acc -= g * w ** 2 * np.sin(w * t)[:, None]
    w = 2 * np.pi * 0.07
    slow = np.array([1.0, 1.0, 0.3]) * 0.02
    pos += slow * np.sin(w * t)[:, None]
    vel += slow * w * np.cos(w * t)[:, None]
    acc -= slow * w ** 2 * np.sin(w * t)[:, None]
    return pos, vel, acc


def validate_com_fusion(fs=100.0, seed=0):
    from scipy.signal import savgol_filter
    rng = np.random.default_rng(seed)
    pos, vel, acc = synthetic_com(fs=fs, seed=seed)
    print("=" * 74 + "\n  CoM VELOCITY FUSION\n" + "=" * 74)
    print("  RMS velocity error, mm/s        ML      AP      VT")
    for noise_mm in (1.0, 2.0, 5.0):
        p = pos + rng.normal(0, noise_mm / 1000, pos.shape)
        a = acc + 0.03 + rng.normal(0, 0.01, acc.shape)   # offset + noise
        v_mk = savgol_filter(p, 11, 3, deriv=1, delta=1 / fs, axis=0)
        v = complementary(v_mk, a, fs)
        e_mk, e_f = 1000 * rms(v_mk - vel), 1000 * rms(v - vel)
        print(f"  {noise_mm:.0f} mm markers      alone {e_mk.round(1)}")
        print(f"                     fused {e_f.round(1)}")
    v = complementary(savgol_filter(pos + rng.normal(0, 0.002, pos.shape),
                                    11, 3, deriv=1, delta=1 / fs, axis=0),
                      acc, fs)
    b, a_ = butter(2, 0.5, fs=fs)
    bias = 1000 * (v - vel).mean(0)
    low = 1000 * rms(filtfilt(b, a_, v - vel, axis=0))
    print(f"  below 0.5 Hz: {low.round(2)} mm/s RMS, mean {bias.round(3)}")
    verdict("fusion", np.all(np.abs(bias) < 1.0) and np.all(e_f < e_mk),
            "fusion beats markers alone and is unbiased at low frequency")


# %%==========================================================================
#  Margin of stability
# ============================================================================
# One stance with a known CoM state, pendulum and lateral boot edge, so the
# margin is worked out by hand and compared with stability.margin_of_stability.
# The other foot is a step length BEHIND, so a lateral direction that picked
# up an AP component would show.

def fake_trial(n, steps, ankle, com, vel_belt, ext, fwd=(0.0, 1.0)):
    fwd = np.array(fwd)
    return SimpleNamespace(
        steps=pd.DataFrame(steps), ankle=ankle, com=com, com_vel_belt=vel_belt,
        boots=SimpleNamespace(ext=ext), forward=fwd,
        lateral=np.array([-fwd[1], fwd[0]]), n=n,
        handrail_rows=np.zeros(n, bool), participant={})


def validate_margin_of_stability(edge=0.14, ankle_ml=0.10, v_ml=0.30,
                                 length=1.0, step=0.68, g=9.81, n=400):
    w0 = np.sqrt(g / length)
    expected = edge - v_ml / w0
    print("=" * 74 + "\n  MARGIN OF STABILITY\n" + "=" * 74)
    print(f"  by hand: MoS_ML = {edge} - {v_ml}/{w0:.5f} = {expected:.5f} m")
    z = np.sqrt(length ** 2 - ankle_ml ** 2)
    com = np.tile([0.0, 0.0, z], (n, 1))
    ankle = {"R": np.tile([ankle_ml, 0.0, 0.0], (n, 1)),
             "L": np.tile([-ankle_ml, -step, 0.0], (n, 1))}
    vel = np.tile([v_ml, 0.0, 0.0], (n, 1))
    ext = {b: {"lat_max": np.full(n, -0.06 if b == "R" else edge),
               "lat_min": np.full(n, -edge if b == "R" else 0.06),
               "fwd_max": np.full(n, 0.2), "fwd_min": np.full(n, -0.1)}
           for b in "LR"}
    t = fake_trial(n, [dict(limb="R", hs=100.0, to=160.0, contra_to=110.0,
                            contra_hs=150.0, next_hs=210.0)],
                   ankle, com, vel, ext)
    s = stability.margin_of_stability(t).iloc[0]
    print(f"  shipped: MoS_ML = {s.mos_ml_contact:.5f} m, pendulum "
          f"{s.pendulum_m:.5f} m, ankle-boundary margin "
          f"{s.mos_ml_contact_ankle:.5f} (expected {ankle_ml - v_ml / w0:.5f})")
    verdict("mos", abs(s.mos_ml_contact - expected) < 1e-9
            and abs(s.pendulum_m - length) < 1e-9
            and abs(s.mos_ml_contact_ankle - (ankle_ml - v_ml / w0)) < 1e-9,
            "margin, boundary and pendulum exact")


# %%==========================================================================
#  MFC, margin of instability, trip risk integral (Schulz 2017)
# ============================================================================
# A swing built so every piece has a known answer: a clearance with a known
# mid-swing minimum at the FRONT vertex, a margin of instability held at a
# constant, and an integration window set by a speed profile whose
# acceleration peaks and troughs at known places.

def trip_scenario(mfc=0.018, base=0.060, moi_mm=12.0, length=1.0,
                  n=100, monotonic=False, g=9.81):
    u = np.arange(n) / (n - 1)
    clear = (base - (base - mfc) * u if monotonic
             else mfc + (base - mfc) * ((u - 0.5) / 0.5) ** 2)
    speed = 0.5 * (1 - np.cos(2 * np.pi * u))
    y = np.cumsum(speed) / 100
    W = np.zeros((n, 2, 3))
    W[:, 0] = np.column_stack([np.full(n, 0.1), y + 0.1, clear])   # front
    W[:, 1] = np.column_stack([np.full(n, 0.1), y - 0.1, clear + 0.05])
    boots = SimpleNamespace(
        world=lambda b, rows: W[rows], height=lambda b, P: P[..., 2],
        sole={"R": {"toe": np.array([True, False])}},
        ext={b: {"fwd_max": np.full(n, 0.30)} for b in "LR"})
    z = np.sqrt(length ** 2 - 0.1 ** 2)
    w0 = np.sqrt(g / length)
    t = SimpleNamespace(
        boots=boots, forward=np.array([0.0, 1.0]),
        com=np.tile([0.0, 0.0, z], (n, 1)),
        ankle={"L": np.tile([-0.1, 0.0, 0.0], (n, 1))},
        com_vel_belt=np.tile([0.0, (0.30 + moi_mm / 1000) * w0, 0.0], (n, 1)))
    return stability.swing_clearance(t, "R", "L", np.arange(n)), W, clear


def validate_trip_risk(moi_mm=12.0):
    print("=" * 74 + "\n  MFC, MARGIN OF INSTABILITY, TRIP RISK\n" + "=" * 74)
    r, W, clear = trip_scenario(moi_mm=moi_mm)
    p = W[:, 0]
    sp = np.linalg.norm(np.gradient(p, 0.01, axis=0), axis=1)
    acc = np.gradient(sp, 0.01)
    a, b = sorted((int(np.argmax(acc)), int(np.argmin(acc))))
    tri = np.sum((moi_mm / np.maximum(1000 * clear, 1.0))[a:b + 1]) / 100
    print(f"  MFC {r['mfc_m']:.6f} m (sampled minimum {clear.min():.6f}), "
          f"vertex {r['mfc_vertex']} (front, on the toes: {r['mfc_on_toes']})")
    print(f"  MoI peak {r['moi_peak_mm']:.4f} mm (built {moi_mm})")
    print(f"  TRI {r['tri_s']:.6f} s (independent {tri:.6f})")
    mono, _, _ = trip_scenario(monotonic=True)
    print(f"  monotonic clearance: MFC {mono['mfc_m']}, "
          f"{mono['mfc_minima']} minima")
    verdict("trip", abs(r["mfc_m"] - clear.min()) < 1e-12
            and r["mfc_vertex"] == 0 and abs(r["moi_peak_mm"] - moi_mm) < 1e-6
            and abs(r["tri_s"] - tri) < 1e-12 and np.isnan(mono["mfc_m"]),
            "MFC, MoI and TRI exact; a swing without a minimum gives NaN")


# %%==========================================================================
#  DFA and its confidence interval
# ============================================================================
# fGn has alpha = H exactly. N is what this protocol gives: 486 strides from
# a 10 minute trial, 756 from 15 minutes. Then the parametric bootstrap
# interval is checked for COVERAGE: a 95% interval must contain the true
# alpha in about 95% of trials.

def validate_dfa(lengths=(486, 756), hursts=(0.5, 0.7, 0.9), reps=60,
                 coverage_trials=60, seed=0):
    rng = np.random.default_rng(seed)
    print("=" * 74 + "\n  DETRENDED FLUCTUATION ANALYSIS\n" + "=" * 74)
    worst = 0.0
    for n in lengths:
        line = f"  N={n}  "
        for h in hursts:
            a = [var.dfa_alpha(unc.fractional_gaussian_noise(n, h, rng))
                 for _ in range(reps)]
            worst = max(worst, abs(np.mean(a) - h))
            line += f"H={h}: {np.mean(a):.3f}+-{np.std(a):.3f}   "
        print(line)
    x = unc.fractional_gaussian_noise(756, 0.9, rng)
    shuffled = np.mean([var.dfa_alpha(rng.permutation(x)) for _ in range(40)])
    print(f"  shuffled H=0.9 series: alpha {shuffled:.3f} (must be 0.5)")

    h, n = 0.75, 486
    hits, widths = 0, []
    for k in range(coverage_trials):
        a = var.dfa_alpha(unc.fractional_gaussian_noise(n, h, rng))
        lo, hi, _, _ = unc.dfa_interval(a, n, var.dfa_alpha, n_boot=100,
                                        tag=f"cov{k}")
        hits += lo <= h <= hi
        widths.append(hi - lo)
    cover = hits / coverage_trials
    print(f"  95% interval at N={n}, H={h}: covers the truth in "
          f"{100 * cover:.0f}% of {coverage_trials} trials, "
          f"{np.mean(widths):.2f} wide")
    print("  A between-condition difference in alpha smaller than about half")
    print("  that width is inside one trial's uncertainty.")
    verdict("dfa", worst < 0.05 and abs(shuffled - 0.5) < 0.05
            and 0.85 <= cover <= 1.0,
            "DFA unbiased, shuffle gives 0.5, interval covers ~95%")


# %%==========================================================================
#  Block bootstrap
# ============================================================================
# A persistent AR(1) series with a known mean of zero. A 95% interval for the
# mean must contain zero in ~95% of series. Resampling single strides (block
# 1) ignores the correlation and covers far less; the Politis-White block
# restores it.

def validate_block_bootstrap(phi=0.6, n=400, trials=200, seed=1):
    rng = np.random.default_rng(seed)
    print("=" * 74 + "\n  BLOCK BOOTSTRAP\n" + "=" * 74)
    hit_block = hit_iid = 0
    blocks = []
    for k in range(trials):
        e = rng.normal(size=n + 100)
        x = np.zeros(n + 100)
        for i in range(1, n + 100):
            x[i] = phi * x[i - 1] + e[i]
        x = x[100:]
        r = unc.series_summary(x, ("mean",), n_boot=500, tag=f"b{k}")["mean"]
        hit_block += r["ci_low"] <= 0 <= r["ci_high"]
        blocks.append(r["block"])
        idx = unc.block_indices(n, 1, 500, unc.rng_for(f"i{k}"))
        lo, hi = unc.percentile_ci(x[idx].mean(1))
        hit_iid += lo <= 0 <= hi
    white = unc.optimal_block_length(rng.normal(size=n))
    print(f"  AR(1) phi={phi}, N={n}: block length {np.median(blocks):.0f} "
          f"(white noise: {white})")
    print(f"  95% interval covers the true mean: blocks "
          f"{100 * hit_block / trials:.0f}%, single strides "
          f"{100 * hit_iid / trials:.0f}%")
    verdict("bootstrap", hit_block / trials >= 0.88
            and hit_iid / trials < 0.85 and white <= 2,
            "blocks restore the coverage that single-stride resampling loses")


# %%==========================================================================
#  Entropy
# ============================================================================
# For z-scored white noise SampEn has a closed form, -ln(2 Phi(r/sqrt 2) - 1)
# for any m. Then the ordering regular < chaotic < random, and the multiscale
# curve: white noise falls away with scale, 1/f noise holds.

def validate_entropy(seed=0):
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    print("=" * 74 + "\n  ENTROPY\n" + "=" * 74)
    worst = 0.0
    for r in (0.15, 0.20, 0.25):
        exact = -np.log(2 * norm.cdf(r / np.sqrt(2)) - 1)
        got = np.mean([var.sample_entropy(rng.normal(size=5000), 2, r)
                       for _ in range(8)])
        worst = max(worst, abs(got - exact))
        print(f"  r={r:.2f}: exact {exact:.4f}, measured {got:.4f}")
    n = 4000
    sine = np.sin(2 * np.pi * np.arange(n) / 50)
    x, logistic = 0.4, []
    for _ in range(n + 1000):
        x = 4 * x * (1 - x)
        logistic.append(x)
    e = [var.sample_entropy(s) for s in
         (sine, np.array(logistic[1000:]), rng.normal(size=n))]
    print(f"  sine {e[0]:.3f} < logistic {e[1]:.3f} < white {e[2]:.3f}")

    def pink(n):
        f = np.fft.rfftfreq(n)
        f[0] = f[1]
        return np.fft.irfft((rng.normal(size=len(f))
                             + 1j * rng.normal(size=len(f))) / np.sqrt(f), n)
    cw, cp = var.rcmse(rng.normal(size=12000), 10), var.rcmse(pink(12000), 10)
    print(f"  RCMSE white {cw[0]:.2f} -> {cw[-1]:.2f}, 1/f {cp[0]:.2f} -> "
          f"{cp[-1]:.2f}")
    verdict("entropy", worst < 0.02 and e[0] < e[1] < e[2]
            and cw[0] - cw[-1] > 0.5 and abs(cp[0] - cp[-1]) < 0.2,
            "closed form, ordering, and MSE separates noise from structure")


# %%==========================================================================
#  Harmonic ratio
# ============================================================================
# x = A2 sin(2 w t) + A1 sin(w t) over one stride: HR = A2 / A1 exactly and
# iHR = A2^2 / (A1^2 + A2^2). ML inverts the ratio. Weighting by k^power
# from a velocity must equal the ratio of the differentiated signal.

def validate_harmonic_ratio(n=110):
    print("=" * 74 + "\n  HARMONIC RATIO\n" + "=" * 74)
    t = np.arange(n) / n
    exact = True
    for a1 in (0.05, 0.25, 1.0):
        hr, ihr = var.harmonic_ratio(var.harmonics(
            np.sin(4 * np.pi * t) + a1 * np.sin(2 * np.pi * t)))
        exact &= abs(hr - 1 / a1) < 1e-9 and \
            abs(ihr - 100 / (1 + a1 ** 2)) < 1e-9
        print(f"  A1={a1}: HR {hr:.4f} (exact {1 / a1:.4f}), iHR {ihr:.2f}%")
    amp = var.harmonics(np.sin(4 * np.pi * t) + 0.25 * np.sin(2 * np.pi * t))
    inv = var.harmonic_ratio(amp)[0] * var.harmonic_ratio(amp, True)[0]
    vel = np.cos(4 * np.pi * t) / 2 + 0.25 * np.cos(2 * np.pi * t)
    acc = -np.sin(4 * np.pi * t) * 2 * np.pi - 0.25 * np.sin(2 * np.pi * t) * 2 * np.pi
    spec = var.harmonic_ratio(var.harmonics(vel, power=1))[0]
    direct = var.harmonic_ratio(var.harmonics(acc))[0]
    print(f"  ML convention inverts: product {inv:.9f}; from velocity with "
          f"k-weighting {spec:.6f} vs from acceleration {direct:.6f}")
    verdict("hr", exact and abs(inv - 1) < 1e-9 and abs(spec - direct) < 1e-9,
            "exact on the two-tone case, ML inverts, k-weighting exact")


# %%==========================================================================
#  Regularity, GEM, foot placement, symmetry
# ============================================================================

def validate_supplementary(seed=0):
    rng = np.random.default_rng(seed)
    print("=" * 74 + "\n  REGULARITY, GEM, FOOT PLACEMENT, SYMMETRY\n" + "=" * 74)
    # a symmetric signal whose stride is 6% longer than the lag it is read
    # at: fixed lags miss the peak, the peak search does not
    period, assumed = 116.6, 110
    t = np.arange(4000)
    sym = np.sin(4 * np.pi * t / period)
    reg = var.regularity(var.unbiased_autocorr(sym, 200), assumed)
    fixed = var.unbiased_autocorr(sym, 200)[assumed // 2]
    asym = var.regularity(var.unbiased_autocorr(
        sym + 0.5 * np.sin(2 * np.pi * t / period), 200), assumed)
    print(f"  symmetric: step {reg['step_regularity']:.3f}, stride "
          f"{reg['stride_regularity']:.3f} (value at the fixed lag {fixed:.3f})")
    print(f"  asymmetric: symmetry {asym['symmetry']:.3f} (below 1)")
    reg_ok = (reg["step_regularity"] > 0.99 and reg["stride_regularity"] > 0.99
              and asym["symmetry"] < 0.9)

    # GEM: strides built from known dimensionless components
    par, perp = rng.normal(0, 0.03, 600), rng.normal(0, 0.01, 600)
    T = 1.1 * (1 + (par - perp) / np.sqrt(2))
    L = 1.43 * (1 + (par + perp) / np.sqrt(2))
    g = var.gem_decompose(T, L)
    r_par = np.corrcoef(g["parallel"], par)[0, 1]
    r_perp = np.corrcoef(g["perpendicular"], perp)[0, 1]
    print(f"  GEM: v* {g['v_star']:.3f}, recovered components r = "
          f"{r_par:.4f} / {r_perp:.4f}, SD {g['parallel_sd']:.4f} / "
          f"{g['perpendicular_sd']:.4f} (built {par.std():.4f} / "
          f"{perp.std():.4f})")
    gem_ok = r_par > 0.999 and r_perp > 0.99

    phis = []
    for phi in (-0.5, 0.0, 0.5):
        x = np.zeros(4000)
        for i in range(1, 4000):
            x[i] = phi * x[i - 1] + rng.normal()
        phis.append(abs(var.lag1(x) - phi) < 0.05)
    zc, vc = rng.normal(0, 0.02, 800), rng.normal(0, 0.1, 800)
    fp = var.foot_placement_model(zc, vc, 0.05 + 0.8 * zc + 0.2 * vc)
    print(f"  foot placement: R2 {fp['r2']:.6f}, gains "
          f"{fp['gain_position']:.4f} / {fp['gain_velocity']:.4f} (0.8 / 0.2)")
    sa = var.symmetry_angle(1.1, 0.9) + var.symmetry_angle(0.9, 1.1)
    verdict("supplementary", reg_ok and gem_ok and all(phis)
            and fp["r2"] > 0.9999 and abs(fp["gain_position"] - 0.8) < 1e-9
            and abs(var.symmetry_angle(1, 1)) < 1e-12 and abs(sa) < 1e-12,
            "regularity peaks found, GEM recovered, AR(1), regression, "
            "symmetry antisymmetric")


# %%==========================================================================

if __name__ == "__main__":
    validate_com_fusion()
    validate_margin_of_stability()
    validate_trip_risk()
    validate_dfa()
    validate_block_bootstrap()
    validate_entropy()
    validate_harmonic_ratio()
    validate_supplementary()
    bad = [k for k, v in RESULTS.items() if not v]
    print("ALL PASSED" if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
