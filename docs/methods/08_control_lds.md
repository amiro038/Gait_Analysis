# 17. Control of stepping

## 17.1 The goal-equivalent manifold (GEM)

**What.** Which stride-to-stride fluctuations the walker corrects, and which they leave alone (Dingwell, John & Cusumano 2010).

On a belt at a fixed speed, the task goal is to keep up with the belt: $L/T = v$. Many combinations of stride length $L$ and stride time $T$ achieve it; they lie on the line $L = vT$, the **goal-equivalent manifold**. After dividing each by its mean, the goal line becomes the diagonal, and each stride's deviation splits into two parts:

$$\hat T = \frac{T}{\bar T} - 1,\qquad \hat L = \frac{L}{\bar L} - 1$$

$$\delta_{\parallel} = \frac{\hat T + \hat L}{\sqrt2},\qquad \delta_{\perp} = \frac{\hat L - \hat T}{\sqrt2}.$$

$\delta_{\parallel}$ is the deviation **along** the line: it is goal-equivalent and leaves the speed unchanged. $\delta_{\perp}$ is the deviation **across** it: it is goal-relevant, a speed error.

For each component, per foot, the code reports:

* its SD in % of the mean stride (with a block-bootstrap interval);
* its lag-1 autocorrelation;
* its DFA α (with the parametric interval of §12.5).

These are computed only when there are at least 64 strides (4 × the smallest DFA box).

**Steps** (`gem_decompose()`, `lag1()`, `variability_rows()`): stride time from §6.1, stride length **over the belt** from §6.2, on the series cut to the common length (§12.4).

::: {custom-style="Why Box"}
**Why stride length over the belt** — the goal is $L/T$ equal to the belt speed. That holds only for the distance travelled over the belt (§6.2), not for the sum of two heel-to-heel steps. **Why normalise** — time and length have different units. Dividing by their means makes both dimensionless, so "along" and "across" the goal line are well defined and the goal line is the diagonal. **What to expect** — healthy walkers correct speed errors quickly: $\delta_\perp$ is small and anti-persistent (α < 0.5). They leave the goal-equivalent fluctuations alone: $\delta_\parallel$ is larger and persistent (α ≈ 1). A load, or fatigue, could change that strategy. **The metronome** fixes $T$ on average, which constrains the walker further; the GEM shows how they trade $L$ against $T$ around that.
:::

## 17.2 Foot placement control

**What.** How much of where the foot lands sideways is explained by the state of the centre of mass during the preceding stance (Wang & Srinivasan 2014).

**Equations.** At this foot's **mid-stance**, $t_{mid} = t_{HS} + \tfrac12(t_{TO}-t_{HS})$, relative to the stance ankle:

$$z = (\mathbf{x}_{CoM} - \mathbf{p}_{ankle})\cdot\hat\ell,\qquad v = \mathbf{v}_{CoM}\cdot\hat\ell,\qquad y = \left(\mathbf{p}_{heel,o}(t_{HS,o}) - \mathbf{p}_{ankle}\right)\cdot\hat\ell$$

where $y$ is where the other foot then lands (its heel at its heel strike). An ordinary least-squares fit gives

$$y = \beta_0 + \beta_1 z + \beta_2 v,\qquad R^2 = 1 - \frac{\sum (y - \hat y)^2}{\sum (y - \bar y)^2}.$$

$R^2$ is reported with a block-bootstrap interval (400 resamples), together with the two gains. It is computed per stance foot and needs at least 20 steps.

![Figure 23. A: the GEM. Each dot is one stride's time and length in % from the mean; the orange line is the goal (stride speed = belt speed). Deviations along the line are goal-equivalent, and across it a speed error. The synthetic walker's strides lie close to the line: across-line SD 0.10%, along-line 1.29%. B: foot placement. Where the left foot landed against where the CoM state at the right foot's mid-stance predicts it.](figures/fig23.png){width=6.5in}

::: {custom-style="Why Box"}
**Why foot placement** — lateral balance in walking is controlled mainly by where the next foot is placed. A walker who controls it well places the foot where the CoM's position and velocity call for, so $R^2$ is high. **Why mid-stance** — the CoM state at mid-stance already predicts most of the next foot placement in healthy walkers (Wang & Srinivasan 2014). Using an earlier point asks whether control has started; using a later one mixes in the swing itself. **Why the system CoM and the fused velocity** — the load is part of what has to be balanced (§5.6), and the velocity must be clean (§5.7).
:::

::: {custom-style="Code Box"}
**In the code** — §17: `lag1()`, `gem_decompose()`, `foot_placement_model()`, `foot_placement_rows()`, `series_for()`, `variability_rows()`. Results: `gem_parallel_sd_pct`, `gem_perpendicular_sd_pct`, `gem_*_lag1`, `gem_*_dfa_alpha` (each `_L`/`_R`), `foot_placement_r2_L/_R` (the gains are in the note of the long table).
:::

{{ranges: gem_perpendicular_dfa_alpha, gem_parallel_dfa_alpha, foot_placement_r2}}

: Table 17.1. Healthy ranges for the GEM and foot placement (treadmill walking).

# 18. Local dynamic stability

**What.** How fast two nearly identical states of the walker drift apart: the local divergence exponent, or maximal Lyapunov exponent λ (Rosenstein, Collins & De Luca 1993; implemented as in Bruijn's LocalDynamicStability toolbox; review: Bruijn et al. 2013). A larger λ means faster divergence: the walker is less locally stable.

**Steps** (`lds_state_space()`, `divergence()`, `fit_lambdas()`, `window_lambda()`, `lds_rows()`):

1. **A window of 150 consecutive strides** of the right foot (`LDS_N_STRIDES`). The whole window is resampled by cubic spline to exactly **100 samples per stride** (`LDS_SAMPLES_PER_STRIDE`), 15 000 samples in total.
2. **Delay embedding.** From one signal $x$, build the state
$$\mathbf{X}(i) = \left[x(i),\ x(i+\tau),\ x(i+2\tau),\ x(i+3\tau),\ x(i+4\tau)\right],\qquad \tau = 10\ \text{samples},\ d_E = 5.$$
3. **Nearest neighbours.** For every point $j$, find its nearest neighbour $\hat{j}$ that is more than **half a stride** (50 samples) away in time. This is the Theiler window, which stops the neighbour from being the same moment of the same stride.
4. **Divergence.** Follow each pair for **10 strides** (`LDS_WS`), and average the log distance over all pairs:
$$y(i) = \left\langle \ln\lVert\mathbf{X}(j+i) - \mathbf{X}(\hat{j}+i)\rVert\right\rangle_j.$$
5. **Exponents.** λ is the slope of $y$ against time in strides: over **0–0.5 stride** for the short-term exponent $\lambda_S$, and over **4–10 strides** for the long-term exponent $\lambda_L$ (`LDS_FIT`).
6. **Windows.** A long trial gives up to **4** non-overlapping windows of 150 strides (`LDS_MAX_WINDOWS`); the trial value is their mean. The interval is the stride bootstrap of §12.6.

**State spaces** (`LDS_STATE_SPACES`): Theia's trunk linear velocity in the walker's axes. ML, AP and VT are each delay-embedded with $d_E = 5$, and the 3-D velocity is embedded with $d_E = 2$ (6 dimensions). The **primary** result is the AP trunk velocity (`LDS_PRIMARY`).

![Figure 24. A: two of the five dimensions of the state space (AP trunk velocity against itself 10 samples later) over 15 strides. The trajectory is a loop that repeats every stride. B: the mean log divergence of nearest neighbours, with the short-term (orange, 0–0.5 stride) and long-term (green, 4–10 strides) fits. The saw-tooth in the first half-stride comes from the synthetic signal: its unfiltered white noise repeats in the embedding every τ = 10 samples. Theia's filtered output should not show it.](figures/fig24.png){width=6.5in}

::: {custom-style="Why Box"}
**Why the same number of strides in every window** — λ depends on the length of the series, so every window is exactly 150 strides, in every trial, and the trial value is a mean over windows rather than one estimate from a longer series. **Why normalise to 100 samples per stride** — it removes differences in stride time between trials and conditions, and it expresses λ per stride, the natural unit of walking. **Why fixed τ and $d_E$** — choosing them per trial (by mutual information and false nearest neighbours) would make λ incomparable between trials. τ = 10% of a stride and $d_E$ = 5 are common fixed choices for gait. `lds_validation.py` (part 2) estimates τ and $d_E$ by average mutual information and false nearest neighbours over the whole dataset, so one fixed value can be chosen from the data before the final run. **Why the trunk velocity** — the trunk carries most of the body's mass and the load, Theia estimates its velocity smoothly, and trunk kinematics is the most studied state space for gait stability (Dingwell & Cusumano 2000; Bruijn et al. 2013). **Why two exponents** — $\lambda_S$ (within a stride) is the one most related to fall risk and to perturbations. $\lambda_L$ is small and less consistent, and it is reported for completeness.
:::

::: {custom-style="Note Box"}
**Caution** — λ values depend strongly on the method: the signal, the normalisation, τ, $d_E$, the series length and the fit windows. Compare them between conditions within this study. Published ranges are only a rough check.
:::

::: {custom-style="Code Box"}
**In the code** — §18: `lds_state_space()`, `divergence()`, `fit_lambdas()`, `window_lambda()`, `lds_rows()`. Settings: `LDS_N_STRIDES = 150`, `LDS_SAMPLES_PER_STRIDE = 100`, `LDS_TAU = 10`, `LDS_DE = 5`, `LDS_WS = 10`, `LDS_FIT`, `LDS_MAX_WINDOWS = 4`, `LDS_N_BOOT = 500`, `LDS_STATE_SPACES`, `LDS_PRIMARY`, `RUN_LDS`. Results: `lds_lambda_S_<space>`, `lds_lambda_L_<space>`; each window is in `{trial}_lds_windows.csv`. `lds_validation.py` (part 1) checks that the shipped `divergence()` recovers known exponents.
:::

{{ranges: lds_lambda_S_trunkVel_AP}}

: Table 18.1. Healthy range for the short-term Lyapunov exponent of the trunk (per stride).
