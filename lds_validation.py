# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 2026

@author: amiro038
"""

###############################################################################
# Checks that sit alongside gait/lds.py but are not part of a per trial run.
#
# Part 1  does the Rosenstein implementation recover known exponents?
#         Run this once after changing gait/lds.py. It does not touch your
#         data. The divergence is gait.lds.divergence itself -- the shipped
#         code, not a copy of it.
# Part 2  AMI and FNN, the estimators for picking tau and dE. These belong in
#         the dataset level parameter script, not in a per trial analysis: tau
#         and dE have to be FIXED across every trial you compare, so run these
#         over the whole dataset, take the median, and set TAU and DE in
#         gait/lds.py.
###############################################################################

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gait import lds  # noqa: E402

# =============================================================================
# %% shared: divergence curve, from the shipped code
# =============================================================================

def divergence_curve(Y, theiler, n_lags, seed=0):
    """Rosenstein mean log divergence (gait.lds.divergence), and the mean log
    distance between far-apart points: the attractor's size."""
    rng = np.random.default_rng(seed)
    sums, counts = lds.divergence(Y, theiler, n_lags + 1)
    curve = sums.sum(0) / counts.sum(0)
    M = len(Y)
    p1, p2 = rng.integers(0, M, 20000), rng.integers(0, M, 20000)
    far = np.abs(p1 - p2) > theiler
    attractor = np.mean(np.log(np.linalg.norm(Y[p1[far]] - Y[p2[far]], axis=1)))
    return curve, attractor


def headroom_fit(curve, attractor, fs, h_lo=0.15, h_hi=0.25):
    """
    Fit the slope in a band of HEADROOM, the fraction of the distance the curve
    has travelled from its starting separation to the attractor size:

        h(i) = (curve[i] - curve[0]) / (attractor - curve[0])

    A fixed low band lands after the neighbour selection transient but before
    saturation bends the curve down, on any system and at any data length,
    with no per system constant. Nothing here was chosen by checking it against
    the reference exponents below.
    """
    h = (curve - curve[0]) / (attractor - curve[0])
    idx = np.flatnonzero((h >= h_lo) & (h <= h_hi))
    t = idx / fs
    return np.polyfit(t, curve[idx], 1)[0], idx[0] / fs, idx[-1] / fs

# =============================================================================
# %% Part 1: does the implementation recover known exponents?
# =============================================================================
# WHAT THIS CLAIMS: that the divergence and fitting machinery is correct.
# WHAT IT DOES NOT CLAIM: that Rosenstein recovers a true Lyapunov exponent
# accurately. It does not, and that is a property of the method.
#
# Expected at n = 60000 (every point a reference, as in the gait analysis):
#     Lorenz   0.880   vs 0.9056   ( -3%)
#     Rossler  0.0606  vs 0.0714   (-15%)
#
# So the estimator is off by a system dependent 4-19% and no fit window removes
# that for both systems at once. Windows DO exist that reproduce either
# reference exactly, but they sit in completely different places for the two
# systems, so hitting a reference means tuning to an answer you already know.
# Two systems are checked rather than one because a single system can be hit by
# luck. The pass band is +-30%: it detects a BROKEN implementation, it is not an
# accuracy claim. For calibration, fitting from t = 0 (inside the transient)
# returns +67% on Lorenz.
#
# The bias is finite size. On Lorenz a post transient fit gives +17% at
# n = 20000 and about +2% at n = 80000-160000, because more data means closer
# neighbours and more headroom. 150 strides of gait is firmly in the small n
# regime, which is why lambda from the main script is an operational measure of
# short term divergence rather than a converged exponent.

def integrate_rk4(deriv, s0, n, dt, transient=5000):
    s = np.array(s0, float)
    out = np.empty((n, len(s)))
    for i in range(transient + n):
        k1 = deriv(s)
        k2 = deriv(s + dt * k1 / 2)
        k3 = deriv(s + dt * k2 / 2)
        k4 = deriv(s + dt * k3)
        s = s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if i >= transient:
            out[i - transient] = s
    return out


lorenz  = lambda s: np.array([10 * (s[1] - s[0]),
                              s[0] * (28 - s[2]) - s[1],
                              s[0] * s[1] - 8 / 3 * s[2]])
rossler = lambda s: np.array([-s[1] - s[2],
                              s[0] + 0.2 * s[1],
                              0.2 + s[2] * (s[0] - 5.7)])

val_n = 60000
val_results = []

for val_name, val_deriv, val_dt, val_theiler, val_lags, val_ref in [
        ('Lorenz  (sigma=10, rho=28, beta=8/3)', lorenz,  0.01,  75, 1000, 0.9056),
        ('Rossler (a=b=0.2, c=5.7)',             rossler, 0.05, 120, 1500, 0.0714)]:

    val_Y = integrate_rk4(val_deriv, [1.0, 1.0, 1.0], val_n, val_dt)
    val_curve, val_attr = divergence_curve(val_Y, val_theiler, val_lags)
    val_lam, val_lo, val_hi = headroom_fit(val_curve, val_attr, 1.0 / val_dt)
    val_err = 100 * (val_lam - val_ref) / val_ref

    val_results.append({'system': val_name,
                        'fit_window': f'{val_lo:.2f}-{val_hi:.2f} time units',
                        'estimate': val_lam, 'reference': val_ref,
                        'error_pct': val_err,
                        'verdict': 'PASS' if abs(val_err) <= 30 else 'CHECK PIPELINE'})

val_results = pd.DataFrame(val_results)
print(val_results.to_string(index=False))
if (val_results['verdict'] == 'CHECK PIPELINE').any():
    print(" ! a known exponent system is more than 30% out, which is outside the "
          "expected\n   bias for this estimator. Check the code before trusting "
          "any gait result")

# --- what does the pipeline return when there is NO divergence at all? --------
# A periodic signal has lambda = 0. Run it through the gait analysis itself
# (gait.lds.window_lambda: the same normalisation, embedding, stride count,
# fit windows and stride bootstrap) and whatever comes back is the noise
# floor: it is entirely the neighbour selection transient. Compare it against
# your real lambda_S, it is not small.

val_sps, val_strides = lds.SAMPLES_PER_STRIDE, lds.N_STRIDES
val_hs = np.arange(val_strides + 1) * 110.0          # 1.1 s strides at 100 Hz
val_t = np.arange(int(val_hs[-1]) + 1) / 110.0        # in strides
val_clean = np.sin(2 * np.pi * val_t) + 0.3 * np.sin(4 * np.pi * val_t)
val_rng = np.random.default_rng(0)

print("\n Noise floor at the gait settings (true lambda = 0):")
for val_noise in (0.001, 0.01, 0.05):
    val_x = val_clean + val_rng.normal(0, val_noise * np.std(val_clean),
                                       len(val_clean))
    val_r = lds.window_lambda(val_x[:, None], val_hs, lds.DE, "none")
    val_lo, val_hi = np.percentile(val_r["reps_S"], [2.5, 97.5])
    print(f"   {val_noise*100:4g}% noise: lambda_S = {val_r['lambda_S']:.4f} "
          f"[{val_lo:.4f}, {val_hi:.4f}] per stride, lambda_L = "
          f"{val_r['lambda_L']:+.4f} (must be ~0 for a periodic signal)")

# =============================================================================
# %% Part 2: AMI and FNN, for the dataset level tau / dE script
# =============================================================================
# Run these over EVERY trial, take the median, then set TAU and DE in
# gait/lds.py. Do not fit them per trial.
#
# IMPORTANT: tau comes out in samples of whatever series you feed in. The main
# script analyses time normalised strides, so feed these the SAME time
# normalised signal or the value will not transfer.

def average_mutual_information(x, max_lag, n_bins=32):
    """AMI against lag. Equiprobable bins: equal width bins on a skewed channel
    put most of the data in a few bins and make the curve noisy."""
    edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
    ami = np.zeros(max_lag + 1)
    for t in range(max_lag + 1):
        H, _, _ = np.histogram2d(x[:len(x) - t], x[t:], bins=[edges, edges])
        P = H / H.sum()
        Px, Py = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        ami[t] = np.sum(P[nz] * np.log2(P[nz] / (Px @ Py)[nz]))
    return ami


def first_ami_minimum(ami, smooth=3):
    """First local minimum. The left comparison is strict, so a flat stretch of
    the curve does not trigger it."""
    a = np.convolve(ami, np.ones(smooth) / smooth, mode='same')
    for t in range(1, len(a) - 1):
        if a[t] < a[t - 1] and a[t] <= a[t + 1]:
            return t
    return int(np.argmin(a[1:]) + 1)


def false_nearest_neighbours(x, tau, max_dim=12, rtol=15.0, atol=2.0, theiler=100):
    """
    Global FNN, WITH a Theiler exclusion.

    The exclusion is not optional. At 100 Hz the trajectory is heavily
    oversampled relative to the stride, so the plain nearest neighbour is the
    temporally adjacent sample, whose extra coordinate differs by a negligible
    amount by construction. Without the exclusion the false neighbour fraction
    is driven to zero at every dimension and the diagnostic says nothing.
    """
    sd = np.std(x)
    frac = np.full(max_dim, np.nan)
    for d in range(1, max_dim + 1):
        M = len(x) - d * tau
        if M < 500:
            continue
        Y = np.column_stack([x[k * tau: k * tau + M] for k in range(d)])
        dist, idx = cKDTree(Y).query(Y, k=min(M, 200))
        nn = np.full(M, -1)
        r = np.full(M, np.nan)
        me = np.arange(M)
        for col in range(1, idx.shape[1]):
            cand = idx[:, col]
            take = (nn < 0) & (np.abs(cand - me) > theiler)
            nn[take] = cand[take]
            r[take] = dist[take, col]
        ok = np.flatnonzero((nn >= 0) & (r > 0))
        extra = np.abs(x[ok + d * tau] - x[nn[ok] + d * tau])
        frac[d - 1] = np.mean((extra / r[ok] > rtol) |
                              (np.sqrt(r[ok] ** 2 + extra ** 2) / sd > atol))
    return frac


def fnn_knee(frac, threshold=0.10, delta=0.02):
    """First dimension below `threshold` that then stops changing."""
    for k in range(len(frac) - 1):
        if frac[k] < threshold and abs(frac[k + 1] - frac[k]) < delta:
            return k + 1
    return int(np.flatnonzero(np.isfinite(frac))[-1] + 1)

# Example, on the synthetic periodic signal above. Swap in a real time
# normalised trunk velocity channel to use it for real.
val_ami = average_mutual_information(val_clean, 100)
val_fnn = false_nearest_neighbours(val_clean, first_ami_minimum(val_ami))
print(f"\n Example on the synthetic signal: tau = {first_ami_minimum(val_ami)}, "
      f"dE = {fnn_knee(val_fnn)}")

plt.figure(figsize=(9.5, 3.4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(val_ami)), val_ami, color='#2a78d6', linewidth=2)
plt.axvline(first_ami_minimum(val_ami), color='#eb6834', ls='--', linewidth=2,
            label=f'tau = {first_ami_minimum(val_ami)}')
plt.xlabel('lag (samples)')
plt.ylabel('AMI (bits)')
plt.title('Average mutual information', fontsize=10)
plt.legend(fontsize=8, frameon=False)
plt.grid(alpha=0.15)

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(val_fnn) + 1), 100 * val_fnn, 'o-', color='#2a78d6',
         linewidth=2, markersize=5)
plt.axvline(fnn_knee(val_fnn), color='#eb6834', ls='--', linewidth=2,
            label=f'dE = {fnn_knee(val_fnn)}')
plt.xlabel('embedding dimension')
plt.ylabel('false nearest neighbours (%)')
plt.title('Global FNN (Theiler corrected)', fontsize=10)
plt.legend(fontsize=8, frameon=False)
plt.grid(alpha=0.15)
plt.tight_layout()
