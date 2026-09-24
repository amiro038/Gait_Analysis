"""
Stride-to-stride variability, and the trunk signal the within-stride
measures read.

    DFA alpha         long-range correlation of a stride series
    sample entropy    predictability of a stride series, with an r sweep
    multiscale        refined composite MSE of the continuous trunk
                      acceleration (the curve is the complexity, not scale 1)
    harmonic ratio    smoothness of the trunk within each stride, and iHR
    regularity        unbiased autocorrelation of the trunk at the step and
                      stride peaks, and their ratio (symmetry)
    GEM               stride time and length split along and across the goal
                      line L = v T (Dingwell et al. 2010)
    foot placement    ML foot placement regressed on CoM state at midstance
                      (Wang & Srinivasan 2014)
    symmetry angle    left against right, bounded and reference-free
    walk ratio        step length over cadence

The functions at the top are pure maths on arrays -- the validation script
checks each one against a known answer. `stride_series_metrics` and
`trunk_metrics` at the bottom apply them to a loaded trial.

WHY THE SERIES ARE CUT TO A COMMON LENGTH
alpha, entropy and lambda all drift with the number of strides, so comparing
a 756-stride trial with a 486-stride one mixes N into the result. The runner
cuts every trial's series to one length (the shortest trial's, by default)
and the length is written beside every value.

THE METRONOME paces stride timing. alpha on stride time says how tightly the
walker locks to the beat; the uncued series (step width, clearance, margins,
impulses) are the better primary outcomes. Cued series are marked as such.

References
    Peng et al. (1994) Phys Rev E 49, 1685-1689.
    Damouras et al. (2010) Gait Posture 31(3), 336-340.
    Richman & Moorman (2000) Am J Physiol 278, H2039-H2049.
    Wu et al. (2014) Phys Lett A 378, 1369-1374 (refined composite MSE).
    Yentes & Raffalt (2021) Ann Biomed Eng 49(3), 979-990.
    Menz, Lord & Fitzpatrick (2003) Gait Posture 18(1), 35-46.
    Pasciuto et al. (2017) J Biomech 53, 84-89.
    Moe-Nilssen & Helbostad (2004) J Biomech 37, 121-126.
    Dingwell, John & Cusumano (2010) PLoS Comput Biol 6(7), e1000856.
    Wang & Srinivasan (2014) Biol Lett 10, 20140405.
    Zifchock et al. (2008) Gait Posture 27(4), 622-627.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, savgol_filter
from scipy.spatial import cKDTree

from . import uncertainty as unc

# DFA
DFA_MIN_BOX = 16               # Damouras: 16 <= n <= N/9
DFA_MAX_BOX_FRAC = 1.0 / 9
DFA_N_BOXES = 20
DFA_N_SURROGATE = 20           # shuffles per series; each must give ~0.5
GPH_POWER = 0.5                # GPH bandwidth m = N ** this

# Entropy
ENT_M = 2
ENT_R = 0.2                    # x SD
ENT_R_SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30)
MSE_SCALES = 30                # 0.3 s at 100 Hz, about a quarter stride
MSE_MAX_S = 300.0              # seconds of trunk signal used for the curve

# Trunk signal
TRUNK_SOURCES = [("Trunk_Linear_Acceleration", 0),     # (column, derivatives)
                 ("Low_Back_Linear_Acceleration", 0),
                 ("Trunk_Linear_Velocity", 1),
                 ("Low_Back_Linear_Velocity", 1),
                 ("Low_Back_Position", 2),
                 ("Trunk_Position", 2)]
TRUNK_LOWPASS_HZ = 20.0
TRUNK_DERIV_WINDOW = 5         # Savitzky-Golay: within 5% to 18 Hz
HR_HARMONICS = 20
HR_MIN_SAMPLES = 40
REGULARITY_SEARCH = 0.25       # peak searched within +-25% of the expected lag

CUED = {"stride_time", "stance_time", "swing_time", "step_time", "cadence",
        "stance_pct", "swing_pct", "single_support_pct",
        "double_support_pct"}


# %% ==========================================================================
#  DFA
# =============================================================================

def dfa(series, min_box=None, max_box=None, n_boxes=None):
    """DFA-1: alpha, the R^2 of the log-log fit, and the curve it came from.

    Boxes run forwards AND backwards so a length that is not a multiple of
    the box keeps its tail. The straight-line residual of each box is taken
    in closed form, which makes this fast enough to call thousands of times
    for the confidence interval.
    """
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    n = len(x)
    min_box = DFA_MIN_BOX if min_box is None else min_box
    n_boxes = DFA_N_BOXES if n_boxes is None else n_boxes
    max_box = int(DFA_MAX_BOX_FRAC * n) if max_box is None else int(max_box)
    empty = dict(alpha=np.nan, r2=np.nan, n=n, boxes=np.array([]),
                 fluct=np.array([]))
    if max_box <= min_box or np.std(x) <= 1e-12 * max(abs(x.mean()), 1.0):
        return empty
    y = np.cumsum(x - x.mean())
    boxes = np.unique(np.round(np.logspace(np.log10(min_box),
                                           np.log10(max_box), n_boxes)).astype(int))
    fluct = np.full(len(boxes), np.nan)
    for i, s in enumerate(boxes):
        count = n // s
        if count < 2:
            continue
        seg = np.vstack([y[:count * s].reshape(count, s),
                         y[n - count * s:].reshape(count, s)])
        t = np.arange(s) - (s - 1) / 2
        seg = seg - seg.mean(axis=1, keepdims=True)
        slope = seg @ t / (t @ t)
        resid = np.sum(seg ** 2, axis=1) - slope ** 2 * (t @ t)
        fluct[i] = np.sqrt(max(resid.sum(), 0.0) / (2 * count * s))
    ok = np.isfinite(fluct) & (fluct > 0)
    if ok.sum() < 4:
        return empty
    lx, ly = np.log10(boxes[ok]), np.log10(fluct[ok])
    slope, icept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + icept)
    r2 = 1 - np.sum(resid ** 2) / np.sum((ly - ly.mean()) ** 2)
    return dict(alpha=float(slope), r2=float(r2), n=n, boxes=boxes[ok],
                fluct=fluct[ok])


def dfa_alpha(series):
    return dfa(series)["alpha"]


def gph_alpha(series):
    """Geweke & Porter-Hudak on the periodogram: alpha = d + 0.5. A ballpark
    second opinion reached by a completely different route."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    n = len(x)
    m = int(n ** GPH_POWER)
    per = np.abs(np.fft.rfft(x - x.mean())) ** 2 / (2 * np.pi * n)
    w = 2 * np.pi * np.arange(len(per)) / n
    j = np.arange(1, min(m, len(per) - 1) + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        reg, resp = np.log(4 * np.sin(w[j] / 2) ** 2), np.log(per[j])
    ok = np.isfinite(reg) & np.isfinite(resp)
    if ok.sum() < 4:
        return np.nan
    return float(0.5 - np.polyfit(reg[ok], resp[ok], 1)[0])


# %% ==========================================================================
#  ENTROPY
# =============================================================================

def match_counts(x, m, r):
    """Template pairs matching within r (Chebyshev) at lengths m and m+1,
    as COUNTS, self-matches excluded. A KD-tree does the counting."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < m + 2:
        return 0, 0
    k = n - m                      # the same start indices at both lengths
    starts = np.arange(k)
    tm = cKDTree(x[starts[:, None] + np.arange(m)])
    tm1 = cKDTree(x[starts[:, None] + np.arange(m + 1)])
    b = int((tm.count_neighbors(tm, r, p=np.inf) - k) // 2)
    a = int((tm1.count_neighbors(tm1, r, p=np.inf) - k) // 2)
    return a, b


def sample_entropy(series, m=ENT_M, r=ENT_R):
    """SampEn of the z-scored series, so r is in SD units. NaN, not 0,
    when nothing matches."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    if len(x) < m + 2 or np.std(x) <= 1e-12 * max(abs(x.mean()), 1.0):
        return np.nan
    a, b = match_counts((x - x.mean()) / np.std(x), m, r)
    return float(-np.log(a / b)) if a > 0 and b > 0 else np.nan


def rcmse(series, scales=MSE_SCALES, m=ENT_M, r=ENT_R):
    """Refined composite multiscale entropy: match counts summed over every
    coarse-graining phase before the log. r is fixed on the ORIGINAL
    series' SD and never recomputed per scale."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    x = (x - x.mean()) / np.std(x)
    out = np.full(scales, np.nan)
    for s in range(1, scales + 1):
        a_sum = b_sum = 0
        for p in range(s):
            cut = x[p:]
            k = len(cut) // s
            if k < m + 2:
                continue
            a, b = match_counts(cut[:k * s].reshape(k, s).mean(axis=1), m, r)
            a_sum, b_sum = a_sum + a, b_sum + b
        if a_sum > 0 and b_sum > 0:
            out[s - 1] = -np.log(a_sum / b_sum)
    return out


# %% ==========================================================================
#  HARMONIC RATIO, REGULARITY
# =============================================================================

def harmonics(segment, n_harmonics=HR_HARMONICS, power=0):
    """Amplitudes of harmonics 1..n of one stride (the segment IS one
    stride, so DFT bin k is harmonic k; no window). power = number of
    differentiations to reach acceleration: harmonic k is weighted k**power,
    which is exact where numerical differentiation is not."""
    seg = np.asarray(segment, float)
    amp = np.abs(np.fft.rfft(seg - seg.mean()))
    if len(amp) <= n_harmonics:
        return None
    k = np.arange(1, n_harmonics + 1)
    return amp[1:n_harmonics + 1] * k.astype(float) ** power


def harmonic_ratio(amp, odd_dominant=False):
    """(HR, iHR %). AP and vertical repeat twice a stride (even harmonics
    intrinsic); ML once (odd intrinsic), so its ratio is inverted. iHR is
    the intrinsic share of the total power, bounded 0-100% (Pasciuto)."""
    k = np.arange(1, len(amp) + 1)
    odd, even = amp[k % 2 == 1], amp[k % 2 == 0]
    intrinsic, other = (odd, even) if odd_dominant else (even, odd)
    hr = intrinsic.sum() / other.sum() if other.sum() > 0 else np.nan
    total = np.sum(amp ** 2)
    ihr = 100 * np.sum(intrinsic ** 2) / total if total > 0 else np.nan
    return float(hr), float(ihr)


def unbiased_autocorr(x, max_lag):
    """Autocorrelation divided by the overlap at each lag, not by N, so a
    longer lag is not penalised just for being longer."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    ac = np.full(max_lag + 1, np.nan)
    for k in range(min(max_lag, n - 10) + 1):
        ac[k] = x[:n - k] @ x[k:] / (n - k)
    return ac / ac[0]


def regularity(ac, stride_lag):
    """Step and stride regularity: the autocorrelation PEAKS near half a
    stride and a stride (searched +-REGULARITY_SEARCH), not the values at
    fixed lags, which miss the peak whenever the stride time drifts."""
    def peak(lag):
        lo = int(np.floor(lag * (1 - REGULARITY_SEARCH)))
        hi = int(np.ceil(lag * (1 + REGULARITY_SEARCH)))
        hi = min(hi, len(ac) - 1)
        if hi <= lo:
            return np.nan, np.nan
        i = lo + int(np.nanargmax(ac[lo:hi + 1]))
        return float(ac[i]), i
    d1, l1 = peak(stride_lag / 2)
    d2, l2 = peak(stride_lag)
    return dict(step_regularity=d1, stride_regularity=d2,
                symmetry=d1 / d2 if d2 else np.nan, step_lag=l1,
                stride_lag=l2)


# %% ==========================================================================
#  CONTROL: GEM, FOOT PLACEMENT, SYMMETRY
# =============================================================================

def lag1(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] - np.nanmean(x)
    s = np.sum(x ** 2)
    return float(np.sum(x[:-1] * x[1:]) / s) if s > 0 else np.nan


def gem_decompose(stride_time, stride_length):
    """Goal-equivalent manifold at constant speed.

    Stride time and length are divided by their means, so both axes are
    dimensionless (a fraction of the mean stride) and the goal line
    L / L_mean = T / T_mean is the 45-degree diagonal. Deviations are then
    split along the line (goal-equivalent: speed unchanged) and across it
    (goal-relevant: speed error). Dividing first matters: rotating seconds
    and metres together, as a raw (T, L) plot invites, mixes units and the
    split then depends on the units chosen.
    """
    T = np.asarray(stride_time, float)
    L = np.asarray(stride_length, float)
    ok = np.isfinite(T) & np.isfinite(L)
    T, L = T[ok] / T[ok].mean() - 1, L[ok] / L[ok].mean() - 1
    par = (T + L) / np.sqrt(2)
    perp = (L - T) / np.sqrt(2)
    return dict(v_star=float(np.mean(stride_length[ok])
                             / np.mean(stride_time[ok])),
                n=int(ok.sum()), parallel=par, perpendicular=perp,
                parallel_sd=float(np.std(par, ddof=1)),
                perpendicular_sd=float(np.std(perp, ddof=1)),
                parallel_lag1=lag1(par), perpendicular_lag1=lag1(perp))


def foot_placement_model(com_position, com_velocity, foot):
    """Least squares foot = b0 + b1 * CoM position + b2 * CoM velocity."""
    z, v, y = (np.asarray(a, float) for a in (com_position, com_velocity, foot))
    ok = np.isfinite(z) & np.isfinite(v) & np.isfinite(y)
    if ok.sum() < 10:
        return dict(r2=np.nan, intercept=np.nan, gain_position=np.nan,
                    gain_velocity=np.nan, n=int(ok.sum()))
    X = np.column_stack([np.ones(ok.sum()), z[ok], v[ok]])
    beta = np.linalg.lstsq(X, y[ok], rcond=None)[0]
    ss = np.sum((y[ok] - y[ok].mean()) ** 2)
    r2 = 1 - np.sum((y[ok] - X @ beta) ** 2) / ss if ss > 0 else np.nan
    return dict(r2=float(r2), intercept=float(beta[0]),
                gain_position=float(beta[1]), gain_velocity=float(beta[2]),
                n=int(ok.sum()))


def symmetry_angle(left, right):
    """Zifchock et al.: 0% symmetric, bounded, the sign says which is
    larger (positive: right larger)."""
    sa = (45.0 - np.degrees(np.arctan2(left, right))) / 90.0 * 100.0
    return float(sa - 200.0 if sa > 100.0 else sa)


# %% ==========================================================================
#  THE TRUNK SIGNAL
# =============================================================================

def trunk_signal(trial):
    """The trunk's linear acceleration, derived ONCE for the entropy curve,
    the harmonic ratio and the regularity, in (ML, AP, VT) axes.

    The first LINEAR column in TRUNK_SOURCES is used -- not the *_Joint_Acc
    family, which are angular (deg/s^2). Gaps are filled to filter, then put
    back (widened by the differentiator's reach) so nothing downstream sees
    invented data. The harmonic ratio does not use the time-domain
    acceleration: it takes the source and weights harmonic k by k**power,
    which is exact.
    -> dict(acc, source, power, name) or None
    """
    for name, power in TRUNK_SOURCES:
        raw = trial.kin(name)
        if raw is not None:
            break
    else:
        return None
    gap = ~np.isfinite(raw).all(axis=1)
    n = np.arange(len(raw))
    fill = raw.copy()
    for k in range(3):
        ok = np.isfinite(fill[:, k])
        fill[~ok, k] = np.interp(n[~ok], n[ok], fill[ok, k])
    sos = butter(4, TRUNK_LOWPASS_HZ, fs=trial.fs, output="sos")
    source = sosfiltfilt(sos, fill, axis=0)
    acc = source.copy()
    for _ in range(power):
        acc = savgol_filter(acc, TRUNK_DERIV_WINDOW, 3, deriv=1,
                            delta=1.0 / trial.fs, axis=0)
    reach = power * (TRUNK_DERIV_WINDOW // 2)
    wide = np.convolve(gap, np.ones(2 * reach + 1), "same") > 0
    axes = np.column_stack([np.r_[trial.lateral, 0], np.r_[trial.forward, 0],
                            [0, 0, 1]])                 # -> ML, AP, VT
    acc, source = acc @ axes, source @ axes
    acc[wide], source[gap] = np.nan, np.nan
    return dict(acc=acc, source=source, power=power, name=name)


# %% ==========================================================================
#  APPLIED TO A TRIAL
# =============================================================================

def stride_series_metrics(series, cued=CUED, tag=""):
    """DFA (with a parametric bootstrap CI, a shuffle check and GPH) and
    sample entropy (with an r sweep) for every column of `series`, a
    DataFrame of one limb's consecutive strides. -> list of dict rows."""
    rows = []
    for name in series.columns:
        x = pd.to_numeric(series[name], errors="coerce").to_numpy(float)
        ok = np.isfinite(x)
        if ok.mean() < 0.95 or ok.sum() < 50:
            continue                   # a series with gaps is not a series
        x = x[ok]
        if np.std(x) <= 1e-12 * max(abs(x.mean()), 1.0):
            continue
        # DFA needs boxes from 16 to N/9, so at least 9 * 16 * 2 strides
        # for an octave of box sizes; below that only the entropy is run
        res = dfa(x) if DFA_MAX_BOX_FRAC * len(x) >= 2 * DFA_MIN_BOX \
            else dict(alpha=np.nan)
        lo, hi, se, bias = unc.dfa_interval(res["alpha"], len(x), dfa_alpha,
                                            tag=f"{tag}{name}")
        sur = []
        if np.isfinite(res["alpha"]):
            rng = unc.rng_for(f"shuffle{tag}{name}")
            sur = [dfa_alpha(rng.permutation(x))
                   for _ in range(DFA_N_SURROGATE)]
        base = dict(series=name, n=len(x), cued=name in cued)
        if np.isfinite(res["alpha"]):
            rows.append(dict(base, metric="dfa_alpha", value=res["alpha"],
                             ci_low=lo, ci_high=hi, se=se, bias=bias,
                             r2=res["r2"], gph=gph_alpha(x),
                             shuffled=float(np.nanmean(sur)),
                             boxes=f"{res['boxes'][0]}-{res['boxes'][-1]}"))
        rows.append(dict(base, metric="sample_entropy",
                         value=sample_entropy(x),
                         **{f"r{r:.2f}": sample_entropy(x, r=r)
                            for r in ENT_R_SWEEP}))
    return rows


def trunk_metrics(trial, trunk, strides):
    """Harmonic ratio per stride, regularity, and the MSE curve.

    strides: (start_row, stop_row) of consecutive strides of one limb.
    -> (per-stride DataFrame, dict of trial values, MSE DataFrame)
    """
    acc, src, power = trunk["acc"], trunk["source"], trunk["power"]
    per = []
    for a, b in strides:
        a, b = int(round(a)), int(round(b))
        if b - a < HR_MIN_SAMPLES or b > len(src):
            continue
        seg = src[a:b]
        if not np.isfinite(seg).all():
            continue
        row = dict(start=a)
        for ax, label, odd in ((1, "AP", False), (2, "VT", False),
                               (0, "ML", True)):
            amp = harmonics(seg[:, ax], power=power)
            if amp is None:
                row = None
                break
            row[f"hr_{label}"], row[f"ihr_{label}"] = harmonic_ratio(amp, odd)
        if row:
            per.append(row)
    per = pd.DataFrame(per)

    values = {}
    stride_rows = np.median([b - a for a, b in strides])
    steady = acc[int(strides[0][0]):int(strides[-1][1])]
    for ax, label in ((0, "ML"), (1, "AP"), (2, "VT")):
        ac = unbiased_autocorr(steady[:, ax], int(1.6 * stride_rows))
        reg = regularity(ac, stride_rows)
        for k in ("step_regularity", "stride_regularity", "symmetry"):
            values[f"{k}_{label}"] = reg[k]
        values[f"trunk_acc_rms_{label}"] = float(np.sqrt(np.nanmean(
            steady[:, ax] ** 2)))

    seg = steady[:int(MSE_MAX_S * trial.fs)]
    curves = {label: rcmse(seg[:, ax]) for ax, label in
              ((0, "ML"), (1, "AP"), (2, "VT"))}
    curves["white_noise"] = rcmse(unc.rng_for("mse").normal(size=len(seg)))
    mse = pd.DataFrame(curves, index=pd.RangeIndex(1, MSE_SCALES + 1,
                                                   name="scale"))
    for label in ("ML", "AP", "VT"):
        values[f"mse_area_{label}"] = float(np.nansum(mse[label]))
    return per, values, mse
