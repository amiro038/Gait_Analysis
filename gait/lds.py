"""
Local dynamic stability: the Rosenstein maximum Lyapunov exponent, as in
Bruijn's LocalDynamicStability toolbox (makestatelocal.m + lds_calc.m).

    1. Take N_STRIDES consecutive strides (heel strike to heel strike of one
       limb) and time-normalise the whole block to 100 samples per stride.
    2. Delay-embed each signal (tau samples, dE copies) into a state space.
    3. For every point, find its nearest neighbour outside +-half a stride,
       follow both for WS strides, and average ln(distance) over all pairs.
    4. lambda_S is the slope of that curve over 0-0.5 stride, lambda_L over
       4-10 strides.

TWO ADDITIONS, both about how sure one can be of lambda

WINDOWS. lambda depends on the number of strides, so every estimate uses the
same N_STRIDES. A long trial holds several such windows; each gets its own
lambda, and the trial value is their mean -- more stable than one window
without changing what is estimated.

BOOTSTRAP. Every point of the curve is an average over reference points, and
each reference point belongs to a stride. The curve is stored per stride
(sums and counts), so strides can be resampled -- in blocks of consecutive
strides, the block length set by the Politis-White rule on the per-stride
slopes -- and the curve and its slope recomputed each time. The spread is the
interval. It captures how much lambda depends on which strides happened to be
walked; it does not capture the estimator's own bias (lds_validation.py
measures that on systems with a known exponent).

Reference: Rosenstein, Collins & De Luca (1993) Physica D 65, 117-134;
Bruijn et al. (2013) J R Soc Interface 10, 20120999.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from . import uncertainty as unc

N_STRIDES = 150
SAMPLES_PER_STRIDE = 100
TAU = 10                       # normalised samples
DE = 5
WS = 10                        # strides of divergence followed
FIT = {"S": (0.0, 0.5), "L": (4.0, 10.0)}
MAX_WINDOWS = 4                # windows of N_STRIDES per trial, at most
N_BOOT = 500

# name: (signals as (column, axis), embedding dimension, scale)
#   axis 0/1/2 = ML/AP/VT (the walker's frame). dE None = no delay embedding
#   (the signals ARE the state). scale "zscore" divides each channel by its
#   SD -- needed when units differ, but then no longer comparable between
#   conditions unless a fixed reference SD is used.
STATE_SPACES = {
    "trunkVel_ML": ([("Trunk_Linear_Velocity", 0)], DE, "none"),
    "trunkVel_AP": ([("Trunk_Linear_Velocity", 1)], DE, "none"),
    "trunkVel_VT": ([("Trunk_Linear_Velocity", 2)], DE, "none"),
    "trunkVel_3D": ([("Trunk_Linear_Velocity", a) for a in range(3)], 2,
                    "none"),
}
PRIMARY = "trunkVel_AP"


def state_space(signals, hs, dE, scale):
    """signals (F, c) rows of the trial; hs the heel-strike rows (n+1 of
    them). -> (X, stride index of every row of X)."""
    n = len(hs) - 1
    a, b = int(round(hs[0])), int(round(hs[-1]))
    block = signals[a:b + 1]
    if not np.isfinite(block).all():
        return None, None
    n_samples = n * SAMPLES_PER_STRIDE
    t_new = np.linspace(a, b, n_samples)
    norm = CubicSpline(np.arange(a, b + 1), block, axis=0)(t_new)
    if scale == "zscore":
        norm = norm / norm.std(axis=0)
    d = dE or 1
    rows = n_samples - TAU * (d - 1)
    X = np.hstack([norm[k * TAU:k * TAU + rows] for k in range(d)])
    stride = np.clip(np.searchsorted(hs, t_new[:rows], side="right") - 1,
                     0, n - 1)
    return X, stride


def divergence(X, theiler, n_lags, group=None, n_groups=1):
    """Rosenstein: every point's nearest neighbour more than `theiler`
    samples away in time, both followed for n_lags samples.

    ln(distance) is summed per group of reference points (a group is a
    stride in the gait analysis), so the mean curve is sums.sum(0) /
    counts.sum(0) and a bootstrap can resample the groups.
    -> sums, counts (n_groups, n_lags)"""
    n = len(X)
    group = np.zeros(n, int) if group is None else group
    _, nn = cKDTree(X).query(X, k=2 * theiler + 2)
    far = np.abs(nn - np.arange(n)[:, None]) > theiler
    nbr = nn[np.arange(n), np.argmax(far, axis=1)]
    sums = np.zeros((n_groups, n_lags))
    counts = np.zeros((n_groups, n_lags))
    lags = np.arange(n_lags)
    for c in range(0, n, 400):
        i = np.arange(c, min(c + 400, n))
        a, b = i[:, None] + lags, nbr[i][:, None] + lags
        ok = (a < n) & (b < n)
        d = np.linalg.norm(X[np.minimum(a, n - 1)] - X[np.minimum(b, n - 1)],
                           axis=2)
        ok &= d > 0
        ln = np.log(np.where(ok, d, 1.0))
        np.add.at(sums, group[i], np.where(ok, ln, 0.0))
        np.add.at(counts, group[i], ok)
    return sums, counts


def divergence_by_stride(X, stride, n_strides):
    """Divergence over WS strides, neighbours at least half a stride away,
    summed per reference stride."""
    return divergence(X, SAMPLES_PER_STRIDE // 2, WS * SAMPLES_PER_STRIDE,
                      stride, n_strides)


def fit_lambdas(curve):
    """Slopes over the fit windows, indexed as lds_calc.m."""
    out = {}
    for tag, (w0, w1) in FIT.items():
        i0 = max(int(round(w0 * SAMPLES_PER_STRIDE)), 1)
        i1 = min(int(round(w1 * SAMPLES_PER_STRIDE)), len(curve))
        t = np.arange(i0, i1 + 1) / SAMPLES_PER_STRIDE
        y = curve[i0 - 1:i1]
        ok = np.isfinite(y)
        if ok.sum() < 3:
            out[tag], out[f"R2_{tag}"] = np.nan, np.nan
            continue
        p = np.polyfit(t[ok], y[ok], 1)
        out[tag] = float(p[0])
        out[f"R2_{tag}"] = float(np.corrcoef(t[ok], y[ok])[0, 1] ** 2)
    return out


def window_lambda(signals, hs, dE, scale, tag=""):
    """lambda_S and lambda_L for one window of strides, with the stride-block
    bootstrap replicates. -> dict or None."""
    X, stride = state_space(signals, hs, dE, scale)
    if X is None:
        return None
    n = len(hs) - 1
    sums, counts = divergence_by_stride(X, stride, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = sums.sum(0) / counts.sum(0)
        per_stride = sums / counts
    fit = fit_lambdas(curve)
    # block length from how correlated the strides' own short-term slopes are
    i1 = int(FIT["S"][1] * SAMPLES_PER_STRIDE)
    t = np.arange(1, i1 + 1) / SAMPLES_PER_STRIDE
    slopes = [np.polyfit(t, c[:i1], 1)[0] if np.isfinite(c[:i1]).all()
              else np.nan for c in per_stride]
    block = unc.optimal_block_length(np.array(slopes))
    idx = unc.block_indices(n, block, N_BOOT, unc.rng_for(tag))
    reps = {"S": [], "L": []}
    for k in idx:
        with np.errstate(invalid="ignore", divide="ignore"):
            f = fit_lambdas(sums[k].sum(0) / counts[k].sum(0))
        reps["S"].append(f["S"])
        reps["L"].append(f["L"])
    return dict(curve=curve, lambda_S=fit["S"], lambda_L=fit["L"],
                R2_S=fit["R2_S"], R2_L=fit["R2_L"], block=block,
                reps_S=np.array(reps["S"]), reps_L=np.array(reps["L"]),
                first_row=hs[0], last_row=hs[-1])


def trial_lds(trial, hs_rows, tag=""):
    """Every state space over up to MAX_WINDOWS windows of N_STRIDES strides.

    hs_rows: consecutive heel strikes (rows) of one limb, steady walking.
    -> (summary rows, per-window rows, curves {state space: mean curve})
    """
    n_win = min(MAX_WINDOWS, (len(hs_rows) - 1) // N_STRIDES)
    summary, windows, curves = [], [], {}
    if n_win < 1:
        print(f"  LDS: only {len(hs_rows) - 1} strides, fewer than the "
              f"{N_STRIDES} one window needs -- skipped")
        return summary, windows, curves
    for name, (sigs, dE, scale) in STATE_SPACES.items():
        cols = [trial.kin_walker(col) for col, _ in sigs]
        if any(c is None for c in cols):
            continue
        signals = np.column_stack([c[:, ax] for c, (_, ax) in zip(cols, sigs)])
        results = []
        for w in range(n_win):
            hs = hs_rows[w * N_STRIDES:(w + 1) * N_STRIDES + 1]
            r = window_lambda(signals, hs, dE, scale, tag=f"{tag}{name}{w}")
            if r is None:
                continue
            results.append(r)
            windows.append(dict(state_space=name, window=w, n_strides=N_STRIDES,
                                first_row=r["first_row"],
                                last_row=r["last_row"], lambda_S=r["lambda_S"],
                                lambda_L=r["lambda_L"], R2_S=r["R2_S"],
                                R2_L=r["R2_L"], block=r["block"]))
        if not results:
            continue
        curves[name] = np.nanmean([r["curve"] for r in results], axis=0)
        for tag_ in ("S", "L"):
            value = float(np.mean([r[f"lambda_{tag_}"] for r in results]))
            reps = np.nanmean([r[f"reps_{tag_}"] for r in results], axis=0)
            lo, hi = unc.percentile_ci(reps)
            summary.append(dict(metric=f"lds_lambda_{tag_}", series=name,
                                value=value, ci_low=lo, ci_high=hi,
                                n=N_STRIDES, windows=len(results),
                                primary=name == PRIMARY,
                                block=float(np.mean([r["block"]
                                                     for r in results]))))
    return summary, windows, curves
