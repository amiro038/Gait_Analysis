# 14. Long-range correlations: detrended fluctuation analysis

**What.** Whether each stride "remembers" strides tens to hundreds of strides earlier. DFA (Peng et al. 1994) measures how the fluctuations of a series grow with the time scale over which they are observed. Its exponent α summarises that growth.

**Equations and steps** (`dfa()`), on one foot's consecutive steady strides $x_1..x_N$:

1. **Integrate** the centred series into its profile:
$$Y(k) = \sum_{i=1}^{k}(x_i - \bar{x}),\qquad k=1..N.$$
2. **Cut** $Y$ into non-overlapping boxes of $n$ strides, counting both from the start **and** from the end, so no stride is left out. This gives $2\operatorname{floor}( N/n)$ boxes.
3. **Detrend**: in each box, fit a straight line $\hat Y_{box}(k)$ by least squares and keep the residual (DFA-1).
4. **Fluctuation** at box size $n$: the RMS of all residuals,
$$F(n) = \sqrt{\frac{1}{2\operatorname{floor}( N/n)\,n}\sum_{\text{boxes}}\sum_{k\in\text{box}}\left(Y(k) - \hat Y_{box}(k)\right)^2}.$$
5. **Exponent**: repeat for 20 box sizes, log-spaced from **16** to **N/9** strides. α is the slope of $\log F(n)$ against $\log n$:
$$F(n) \propto n^{\alpha}.$$

| α | Meaning |
|:--|:--|
| < 0.5 | anti-persistent: a long stride tends to be followed by a short one (actively corrected, e.g. by a metronome or a fixed belt speed) |
| ≈ 0.5 | uncorrelated (white noise) |
| 0.5–1 | persistent long-range correlations (healthy free-walking stride time: 0.75–0.9) |
| > 1 | non-stationary (drifting) |

**Checks reported with every α** (in the note of the long table):

* **Shuffle test:** 20 random shuffles of the series must give α ≈ 0.5 (`DFA_N_SURROGATE`). Shuffling destroys any order, so a shuffled value far from 0.5 would mean a coding error or too short a series.
* **A second estimator:** the Geweke–Porter-Hudak periodogram slope. The log periodogram at the $\sqrt N$ lowest frequencies is regressed on $\log\left(4\sin^2(\omega/2)\right)$; the slope is $-d$, and $\alpha_{GPH} = d + \tfrac12$. It scatters about twice as much as DFA, so it is a ballpark check, not a replacement.
* the R² of the log–log fit and the box range used.

The 95% interval is the parametric bootstrap of §12.5. A series is analysed only if at least 95% of its strides are valid, with at least 50 strides. Gaps are **not** bridged, because a bridged series is not a series of consecutive strides. The box range 16 to $N/9$ needs at least one octave, so $N \ge 288$ strides.

![Figure 20. DFA step by step on the step-width series of the synthetic trial. A: the series. B: its integrated profile, with the straight lines fitted in the smallest (16-stride) and largest (N/9) boxes. C: log F(n) against log n; the slope is α (orange). The dotted line has the slope 0.5 of an uncorrelated series.](figures/fig20.png){width=6.5in}

**Series analysed** (`SERIES`): stride, stance and swing time, double support %, step length and width, stride length and speed, MFC, the margins of stability, propulsive and braking impulses, F1 and trunk lean. DFA is computed for each foot separately (`_L`, `_R`).

::: {custom-style="Why Box"}
**Why DFA** — the SD says how much strides vary; α says how they are organised in time. Persistent correlations in free walking are thought to reflect a healthy, flexible control; their loss is seen with ageing and disease (Hausdorff 2005). On a treadmill, the metrics the task constrains (stride speed, which must match the belt) are anti-persistent, and the unconstrained ones (stride length, step width) persistent. The pattern shows which variables the walker controls tightly (Dingwell et al. 2010). **Why boxes 16 to N/9** — the widely used range of 4 to N/4 biases α upwards for gait-length series: small boxes are distorted by short-term correlations, and large boxes have too few boxes to average (Damouras et al. 2010). **Why both ends** — with $N$ not a multiple of $n$, counting from one end only would drop the last strides.
:::

::: {custom-style="Note Box"}
**Caution: metronome-paced series.** Stride, stance, swing and step time, cadence and the support percentages are paced by the 108 steps/min metronome (`CUED`). Their α measures how the walker follows the beat, typically < 0.5, and must not be compared with the uncued healthy range. The note of every such α says "metronome-paced". The spatial series (lengths, width, clearance, margins) are not paced and are compared with the treadmill ranges.
:::

::: {custom-style="Code Box"}
**In the code** — §14: `CUED`, `SERIES`, `dfa()`, `dfa_alpha()`, `gph_alpha()`. The loop over series is `stride_series_metrics()` (§15), called per foot by `variability_rows()` (§17). Settings: `DFA_MIN_BOX = 16`, `DFA_MAX_BOX_FRAC = 1/9`, `DFA_N_BOXES = 20`, `DFA_N_SURROGATE = 20`, `GPH_POWER = 0.5`. Results: `dfa_alpha_<series>_L/_R`.
:::

{{ranges: dfa_alpha_stride_s, dfa_alpha_stride_length_m, dfa_alpha_stride_speed_ms, dfa_alpha_step_width_m}}

: Table 14.1. Healthy ranges for DFA α. Stride time is uncued overground walking, and does not apply to metronome-paced trials; the other three are treadmill walking at a fixed speed.

# 15. Entropy

## 15.1 Sample entropy of the stride series

**What.** How predictable the series is: if two runs of $m$ strides are alike, how often are the next strides alike too (Richman & Moorman 2000)?

**Equations.** $z$-score the series (so the tolerance $r$ is in SD units). Take the templates $\mathbf{u}_m(i) = (z_i,\dots,z_{i+m-1})$ for $i = 1..N-m$, using the same start points for length $m$ and length $m+1$. Two templates match when their Chebyshev (largest-component) distance is at most $r$. Then

$B$ is the number of pairs $i<j$ that match at length $m$, $\lVert\mathbf{u}_m(i)-\mathbf{u}_m(j)\rVert_\infty \le r$. $A$ is the number of pairs that still match at length $m+1$, $\lVert\mathbf{u}_{m+1}(i)-\mathbf{u}_{m+1}(j)\rVert_\infty \le r$.

$$\text{SampEn}(m, r) = -\ln\frac{A}{B},$$

with $m = 2$ and $r = 0.2$ (`ENT_M`, `ENT_R`). Self-matches are excluded. If nothing matches at length $m+1$, SampEn is **missing**, not zero.

**Steps** (`match_counts()`, `sample_entropy()`): the pairs are counted with a KD-tree, which gives the same counts as the textbook double loop, much faster. Each series is also analysed with $r$ = 0.10, 0.15, 0.20, 0.25 and 0.30 (`ENT_R_SWEEP`). The sweep is written in the note, to show whether a conclusion depends on $r$ (Yentes & Raffalt 2021).

## 15.2 Refined composite multiscale entropy of the trunk

**What.** Entropy across time scales, from the continuous trunk acceleration (§16) at 100 Hz (Costa et al. 2002; refined composite version: Wu et al. 2014).

**Equations and steps** (`rcmse()`):

1. $z$-score the steady trunk acceleration (up to 300 s, `MSE_MAX_S`).
2. For each scale $\tau = 1..30$ (`MSE_SCALES`; 0.01–0.30 s), and **each** of the $\tau$ starting phases $p$, coarse-grain by averaging non-overlapping runs of $\tau$ samples:
$$y^{(\tau,p)}_j = \frac1\tau\sum_{i=0}^{\tau-1} z_{p+(j-1)\tau+i}.$$
3. Count $A^{(\tau,p)}$ and $B^{(\tau,p)}$ on each coarse-grained series, with $r$ fixed at 0.2 SD **of the original** series. Sum over the phases before taking the logarithm:
$$\text{RCMSE}(\tau) = -\ln\frac{\sum_p A^{(\tau,p)}}{\sum_p B^{(\tau,p)}}.$$
4. The **area** under the curve over scales 1–30 is the complexity index (`mse_area_ML/AP/VT`). White noise of the same length goes through the identical calculation as a reference.

![Figure 21. A: sample entropy, on part of the step-width series in SD units. The template (black) is two consecutive strides. Every other pair of strides within ±0.2 SD of it (shaded bands) is a match at m = 2 (blue); those whose third stride also matches are matches at m = 3 (orange). SampEn = −ln(A/B) over all templates. B: the multiscale entropy of the trunk acceleration. White noise (dashed) starts high and falls with scale; a structured signal keeps some of its entropy over scales.](figures/fig21.png){width=6.5in}

::: {custom-style="Why Box"}
**Why both measures** — sample entropy asks whether the step-to-step pattern is regular (low) or unpredictable (high). But white noise is the *most* unpredictable signal and also the *least* complex, so one value of SampEn cannot measure complexity. The multiscale curve separates the two: noise loses its entropy as it is averaged, while a signal with structure on many time scales keeps it (Costa et al. 2002). **Why the refined composite version** — plain coarse-graining shortens the series τ-fold, and at large scales too few templates match, which makes the estimate unstable or undefined. Pooling the counts over every starting phase before the logarithm fixes that (Wu et al. 2014). **Why r fixed on the original series** — re-scaling r at each scale would remove exactly the change in variance that the curve is meant to show.
:::

::: {custom-style="Code Box"}
**In the code** — §15: `match_counts()`, `sample_entropy()`, `rcmse()`, `stride_series_metrics()`. The trunk MSE is computed in `trunk_rows()` (§16). Settings: `ENT_M = 2`, `ENT_R = 0.2`, `ENT_R_SWEEP`, `MSE_SCALES = 30`, `MSE_MAX_S = 300`. Results: `sample_entropy_<series>_L/_R`, `mse_area_ML/AP/VT`. The curves are in `{trial}_waveforms.npz`.
:::

::: {custom-style="Range Box"}
**Healthy range** — none is given. Entropy values depend strongly on $N$, $m$, $r$, the sampling rate and the filtering, so values from other studies are not comparable (Yentes & Raffalt 2021). Compare conditions within this study, with the same settings and the same $N$ (§12.4).
:::

# 16. Trunk dynamics: harmonic ratio and regularity

## 16.1 The trunk signal

The first **linear** trunk signal in the export is used, in the order of `TRUNK_SOURCES`: `Trunk_Linear_Acceleration`, `Low_Back_Linear_Acceleration`, `Trunk_Linear_Velocity` (the one on DICE), `Low_Back_Linear_Velocity`, then the positions. It is low-passed at 20 Hz, rotated into the walker's ML/AP/VT axes (§5.2), and differentiated to an acceleration where needed. The derivative is a short Savitzky–Golay filter (5 samples, cubic), accurate up to about 18 Hz. Tracking gaps stay missing.

::: {custom-style="Note Box"}
**Caution** — do not use `*_Joint_Acc` or `Trunk_Joint_Acceleration`: those are **angular** accelerations in deg/s². Also, the published ranges below were measured with accelerometers strapped to the lower back. Theia's trunk signal comes from a model fitted to video, which is smoother, so compare conditions within the study first and the absolute ranges second.
:::

## 16.2 Harmonic ratio

**What.** How symmetric the trunk's motion is from one step to the next (Menz et al. 2003).

**Equations.** Take each stride of the right foot (heel strike to heel strike, `LIMB_FOR_TRUNK`), and compute the discrete Fourier transform of **exactly one stride**. Bin $k$ is then the $k$-th harmonic of the stride frequency, with amplitude $a_k$. AP and vertical accelerations repeat **twice** per stride (once per step), so their even harmonics are the intrinsic, symmetric part. ML repeats **once**, so its odd harmonics are intrinsic. Over $k = 1..20$:

$$\text{HR}_{AP,VT} = \frac{\sum_{k\ \text{even}} a_k}{\sum_{k\ \text{odd}} a_k},\qquad \text{HR}_{ML} = \frac{\sum_{k\ \text{odd}} a_k}{\sum_{k\ \text{even}} a_k}$$

The **improved harmonic ratio** (Pasciuto et al. 2017) is the intrinsic share of the power:

$$\text{iHR} = 100\,\frac{\sum_{\text{intrinsic}} a_k^2}{\sum_{k=1}^{20} a_k^2}\ \%.$$

When the source is a velocity, the acceleration's $k$-th harmonic is exactly $k\omega$ times the velocity's. The amplitudes are therefore weighted by $k$ instead of differentiating numerically, which would attenuate the high harmonics. The factor $\omega$ cancels in both ratios.

**Steps** (`trunk_signal()`, `harmonics()`, `harmonic_ratio()`, `trunk_rows()`): each stride must be complete, with at least 40 samples (`HR_MIN_SAMPLES`), so it can hold 20 harmonics. HR is unbounded, so the trial value is its **median** over strides. iHR is bounded 0–100%, so its **mean** is reported with a block-bootstrap interval.

## 16.3 Step and stride regularity

**What.** How alike consecutive steps and consecutive strides are (Moe-Nilssen & Helbostad 2004).

**Equations.** The **unbiased** autocorrelation of the trunk acceleration over the steady walk:

$$A(k) = \frac{\frac{1}{N-k}\sum_{i=1}^{N-k} x_i\,x_{i+k}}{\frac1N\sum_{i=1}^{N} x_i^2}.$$

The **step regularity** $d_1$ is the peak of $A$ within ±25% of half the median stride. The **stride regularity** $d_2$ is the peak within ±25% of one stride (`REGULARITY_SEARCH`). Their ratio $d_1/d_2$ is the step symmetry. The RMS of the trunk acceleration per axis is reported too.

![Figure 22. The trunk (synthetic trial). A: the mean trunk acceleration over the stride. AP and VT repeat twice per stride, and ML once. B: AP harmonic amplitudes; the even harmonics (blue) are intrinsic to a symmetric gait. C: the VT autocorrelation, with the step and stride regularity read at its peaks.](figures/fig22.png){width=6.5in}

::: {custom-style="Why Box"}
**Why one stride per transform** — then each bin is exactly a harmonic, with no leakage between them. **Why iHR** — HR is a ratio of sums, and it becomes very large when the odd harmonics happen to be small, so its mean is dominated by a few strides. iHR is bounded and can be averaged. **Why peaks and not fixed lags** — the stride time drifts a little through a trial. At a fixed lag, a 6% drift in stride time makes a perfectly regular signal read 0.94 (validated, §20). **Why the unbiased autocorrelation** — dividing by the overlap $N-k$ instead of $N$ stops a longer lag from being penalised just for being longer.
:::

::: {custom-style="Code Box"}
**In the code** — §16: `trunk_signal()`, `harmonics()`, `harmonic_ratio()`, `unbiased_autocorr()`, `regularity()`, `trunk_rows()`. Settings: `TRUNK_SOURCES`, `TRUNK_LOWPASS_HZ = 20`, `TRUNK_DERIV_WINDOW = 5`, `HR_HARMONICS = 20`, `HR_MIN_SAMPLES = 40`, `REGULARITY_SEARCH = 0.25`, `LIMB_FOR_TRUNK = "R"`.
:::

{{ranges: harmonic_ratio_hr_AP, harmonic_ratio_hr_VT, harmonic_ratio_hr_ML, harmonic_ratio_ihr_AP, harmonic_ratio_ihr_VT, harmonic_ratio_ihr_ML, step_regularity_VT, stride_regularity_VT, step_regularity_AP, stride_regularity_AP}}

: Table 16.1. Healthy ranges for the trunk measures (accelerometer on the lower back).
