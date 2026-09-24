"""
Confidence intervals for metrics computed from ONE trial.

Every number this pipeline reports comes from a few hundred strides of one
walk. Walk again and it changes. A confidence interval says by how much, so a
difference between conditions can be read against it.

Two kinds of bootstrap are used, because the metrics come in two kinds.

CIRCULAR BLOCK BOOTSTRAP -- for anything computed from a stride series (the
mean, SD or CV of step width, a regression over strides, ...). Strides are not
independent: a long stride tends to be followed by another long one. Drawing
strides one at a time would throw that away and make every interval too
narrow. So strides are drawn in runs of consecutive strides -- blocks -- and
the runs are glued together into a new series of the same length. Inside a
block the neighbour-to-neighbour correlation is kept.

    BLOCK LENGTH is how many consecutive strides go into each run. Too short
    and the correlation is lost (intervals too narrow); too long and there are
    only a few distinct blocks to draw from (intervals noisy). It is chosen
    automatically for every series by the Politis & White (2004) rule, with
    the correction of Patton, Politis & White (2009): the more slowly the
    series' autocorrelation dies away, the longer the block. An uncorrelated
    series gets a block of 1 -- the ordinary bootstrap. The length used is
    reported beside each interval.

PARAMETRIC BOOTSTRAP -- for the DFA exponent. Cutting a series into blocks
destroys exactly the long-range correlation DFA measures, so blocks cannot be
used. Instead many series are simulated from a process with the estimated
alpha (fractional Gaussian noise, exact by Davies-Harte), DFA is run on each,
and the spread of those answers is the interval. It also gives the bias of
DFA at this N, which is reported.

The Lyapunov exponent has its own stride-block bootstrap in lds.py: it
resamples the reference strides of the divergence curve.

References
    Kunsch (1989) Ann Stat 17(3), 1217-1241.
    Politis & Romano (1992) circular block bootstrap, in Exploring the
        Limits of Bootstrap, 263-270.
    Politis & White (2004) Econometric Rev 23(1), 53-70.
    Patton, Politis & White (2009) Econometric Rev 28(4), 372-375.
    Davies & Harte (1987) Biometrika 74(1), 95-101.
"""

import numpy as np

N_BOOT = 1000          # resamples for the block bootstrap
N_BOOT_DFA = 200       # simulated series for the DFA interval
CI_LEVEL = 0.95
SEED = 0


def rng_for(tag=""):
    """A generator seeded from SEED and a tag, so every interval is
    reproducible run to run but different series do not share draws."""
    return np.random.default_rng([SEED, sum(map(ord, str(tag)))])


# %% --------------------------------------------------------------------------
#  block length
# -----------------------------------------------------------------------------

def optimal_block_length(x):
    """Politis-White block length for the circular block bootstrap.

    The rule balances the bias of a short block against the variance of a
    long one, using a flat-top lag window estimate of the spectrum at zero.
    Returns a length in samples, at least 1.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8 or np.std(x) == 0:
        return 1
    x = x - x.mean()
    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    m_max = min(int(np.ceil(np.sqrt(n))) + kn, n - 1)
    b_max = int(np.ceil(min(3 * np.sqrt(n), n / 3)))
    acov = np.array([x[:n - k] @ x[k:] / n for k in range(m_max + 1)])
    rho = acov / acov[0]

    # the first lag after which kn autocorrelations in a row are negligible
    small = np.abs(rho[1:]) < 2.0 * np.sqrt(np.log10(n) / n)
    m_hat = None
    for m in range(0, m_max - kn + 1):
        if small[m:m + kn].all():
            m_hat = m
            break
    if m_hat is None:                    # never settles: use the last large one
        big = np.flatnonzero(~small)
        m_hat = int(big[-1] + 1) if len(big) else 1
    M = min(2 * max(m_hat, 1), m_max)

    k = np.arange(-M, M + 1)
    t = np.abs(k) / M
    lam = np.where(t <= 0.5, 1.0, np.where(t <= 1.0, 2.0 * (1.0 - t), 0.0))
    R = acov[np.abs(k)]
    G = np.sum(lam * np.abs(k) * R)
    g0 = np.sum(lam * R)
    if g0 <= 0 or G == 0:
        return 1
    D = 4.0 / 3.0 * g0 ** 2                       # circular block bootstrap
    b = (2.0 * G ** 2 / D) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return int(np.clip(np.round(b), 1, b_max))


# %% --------------------------------------------------------------------------
#  circular block bootstrap
# -----------------------------------------------------------------------------

def block_indices(n, block, n_boot=N_BOOT, rng=None):
    """(n_boot, n) indices: runs of `block` consecutive positions, starting
    anywhere, wrapping round the end (circular), glued to length n."""
    rng = rng_for("indices") if rng is None else rng
    block = int(max(1, min(block, n)))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    return idx.reshape(n_boot, -1)[:, :n]


def percentile_ci(replicates, level=CI_LEVEL):
    r = np.asarray(replicates, float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return np.nan, np.nan
    a = (1 - level) / 2
    return float(np.quantile(r, a)), float(np.quantile(r, 1 - a))


STATISTICS = {
    "mean": lambda s: np.nanmean(s, axis=-1),
    "sd": lambda s: np.nanstd(s, axis=-1, ddof=1),
    "cv": lambda s: 100 * np.nanstd(s, axis=-1, ddof=1)
    / np.abs(np.nanmean(s, axis=-1)),
}


def series_summary(x, statistics=("mean", "sd", "cv"), n_boot=N_BOOT,
                   tag=""):
    """Mean, SD and CV of a stride series, each with a block bootstrap CI.

    -> {statistic: dict(value, ci_low, ci_high, n, block)}
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    out = {}
    if len(x) < 10:
        return {s: dict(value=np.nan, ci_low=np.nan, ci_high=np.nan,
                        n=len(x), block=np.nan) for s in statistics}
    block = optimal_block_length(x)
    samples = x[block_indices(len(x), block, n_boot, rng_for(tag))]
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in statistics:
            lo, hi = percentile_ci(STATISTICS[s](samples))
            out[s] = dict(value=float(STATISTICS[s](x)), ci_low=lo,
                          ci_high=hi, n=len(x), block=block)
    return out


def bootstrap_rows(n, statistic, block, n_boot=N_BOOT, tag=""):
    """CI for any statistic of several aligned per-stride arrays: statistic
    takes an index array (into the strides) and returns a number."""
    idx = block_indices(n, block, n_boot, rng_for(tag))
    reps = np.array([statistic(i) for i in idx])
    return percentile_ci(reps)


# %% --------------------------------------------------------------------------
#  fractional Gaussian noise, for the DFA interval
# -----------------------------------------------------------------------------

def fractional_gaussian_noise(n, hurst, rng):
    """Exact fGn with Hurst exponent in (0, 1), by circulant embedding."""
    k = np.arange(0, n + 1)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * hurst) - 2 * np.abs(k) ** (2 * hurst)
                   + np.abs(k + 1) ** (2 * hurst))
    circ = np.concatenate([gamma, gamma[-2:0:-1]])
    eig = np.maximum(np.fft.fft(circ).real, 0.0)
    m = len(eig)
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    return np.fft.fft(np.sqrt(eig / (2 * m)) * z).real[:n]


def simulate_alpha(n, alpha, rng):
    """A series whose DFA exponent is alpha: fGn below 1, its running sum
    (fractional Brownian motion) above."""
    if alpha < 1.0:
        return fractional_gaussian_noise(n, float(np.clip(alpha, 0.02, 0.98)),
                                         rng)
    return np.cumsum(fractional_gaussian_noise(
        n, float(np.clip(alpha - 1.0, 0.02, 0.98)), rng))


def dfa_interval(alpha_hat, n, dfa, n_boot=N_BOOT_DFA, tag=""):
    """Parametric bootstrap interval for a DFA exponent.

    dfa(series) -> alpha. Returns (ci_low, ci_high, se, bias): the basic
    bootstrap interval 2*alpha_hat - quantiles of the simulated estimates,
    their SD, and how far DFA lands from the truth on average at this N.
    The interval is bias-corrected: where DFA is biased by more than its
    scatter (alpha near 0, or few strides) it is centred on
    alpha_hat - bias and can exclude alpha_hat itself.
    """
    if not np.isfinite(alpha_hat):
        return np.nan, np.nan, np.nan, np.nan
    rng = rng_for(tag)
    reps = np.array([dfa(simulate_alpha(n, alpha_hat, rng))
                     for _ in range(n_boot)])
    reps = reps[np.isfinite(reps)]
    if len(reps) < 20:
        return np.nan, np.nan, np.nan, np.nan
    a = (1 - CI_LEVEL) / 2
    q_lo, q_hi = np.quantile(reps, [a, 1 - a])
    return (float(2 * alpha_hat - q_hi), float(2 * alpha_hat - q_lo),
            float(np.std(reps, ddof=1)), float(np.mean(reps) - alpha_hat))
