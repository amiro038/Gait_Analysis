# 12. Confidence intervals

## 12.1 Why every value needs one

Each value comes from **one** walk of a few hundred strides. Walk again, and the value changes. A 95% confidence interval says by how much, so a difference between conditions can be judged against it. The metrics come in three kinds, and each kind needs its own method:

| Kind | Examples | Method |
|:--|:--|:--|
| a statistic of a series of strides | means, SDs, CVs, R² | circular block bootstrap (§12.2–12.3) |
| a long-range correlation | DFA α | parametric bootstrap (§12.5) |
| a divergence curve fitted over strides | Lyapunov exponents | stride bootstrap (§12.6) |

## 12.2 The circular block bootstrap

**What.** Rebuild the stride series many times from runs of consecutive strides, recompute the statistic each time, and read the interval from the spread.

**Steps** (`series_summary()`, `block_indices()`, `percentile_ci()`):

1. Take the series $x_1,\dots,x_N$ (one value per steady step) and its block length $\ell$ (§12.3).
2. Draw $K=\operatorname{ceil}( N/\ell)$ random start points $s_k$ uniformly from $1..N$. Each block is $x_{s_k}, x_{s_k+1}, \dots, x_{s_k+\ell-1}$, wrapping round the end of the series ($(s_k + j) \bmod N$). Glue the blocks together and cut to length $N$.
3. Compute the statistic $\hat\theta_b$ on the new series (resample $b$).
4. Repeat $B = 1000$ times (`N_BOOT`). The 95% interval runs from the 2.5th to the 97.5th percentile of the $B$ values $\hat\theta_b$.

The random numbers come from a generator seeded by `SEED` and the series' name (`rng_for()`), so the intervals are the same every time the script runs, and different series never share draws.

![Figure 19. The circular block bootstrap. A: a series of strides and five blocks of 6 consecutive strides drawn at random start points. B: the blocks glued into one resampled series. The procedure is repeated 1000 times and the statistic recomputed on each. C: the block length chosen by the Politis–White rule grows with the stride-to-stride correlation: uncorrelated strides give 1 (the ordinary bootstrap), persistent series give longer blocks.](figures/fig19.png){width=6.5in}

::: {custom-style="Why Box"}
**Why blocks** — strides are not independent: a long stride tends to be followed by another long one. Resampling single strides destroys that correlation, and it underestimates the variance of the mean. Blocks of consecutive strides keep it. **Validation** (§20): on a persistent series (AR(1), $\phi$ = 0.6, $N$ = 400), the 95% interval from blocks covered the true mean 93% of the time; resampling single strides covered it only 66% of the time. **Why circular** — wrapping round the end gives every stride the same chance of being drawn. Otherwise the first and last few strides would be under-represented. **Why percentile intervals** — they need no assumption about the shape of the sampling distribution, and SDs and CVs are skewed.
:::

## 12.3 The block length

**What.** How many consecutive strides make one block. If blocks are too short, the correlation is lost; if too long, there are too few distinct blocks. The block length is chosen per series, from the series' own autocorrelation, by the rule of Politis & White (2004), as corrected by Patton, Politis & White (2009).

**Steps** (`optimal_block_length()`):

1. Compute the autocovariances $R(k)$ and autocorrelations $\rho(k)$ of the centred series up to lag $m_{max} = \operatorname{ceil}(\sqrt{N}) + K_N$, with $K_N = \max(5, \operatorname{ceil}(\sqrt{\log_{10}N}))$.
2. Find $\hat m$, the first lag after which $K_N$ autocorrelations in a row are negligible: $\left|\rho(k)\right| < 2\sqrt{\log_{10}N / N}$. Set $M = 2\hat m$.
3. Estimate the spectrum at zero and its "slope" with a flat-top lag window $\lambda(u)$: 1 up to $\left|u\right| = 0.5$, falling linearly to 0 at $\left|u\right| = 1$, and 0 beyond:
$$\hat g = \sum_{k=-M}^{M}\lambda\left(\tfrac{k}{M}\right)R(k),\qquad \hat G = \sum_{k=-M}^{M}\lambda\left(\tfrac{k}{M}\right)\left|k\right|R(k).$$
4. The optimal block length for the circular block bootstrap is
$$\ell = \left(\frac{2\hat G^2}{\tfrac43\hat g^2}\right)^{1/3} N^{1/3},$$
rounded, and kept between 1 and $\min(3\sqrt N,\ N/3)$.

::: {custom-style="Why Box"}
**Why chosen from the data** — the correlation between strides differs between metrics (step width is more persistent than stance time) and between people. A fixed block length would be too short for some series and too long for others. The rule gives 1 for an uncorrelated series, so it reduces to the ordinary bootstrap when that is correct. The block length used is written next to every value in the long results table (`block`).
:::

## 12.4 The same number of strides for every trial

DFA α, sample entropy and the Lyapunov exponents depend on how many strides the series has. Even an SD is slightly biased low in short series. Comparing conditions with different series lengths would mix the effect of the condition with the effect of $N$. With `SERIES_LENGTH = "shortest"` (the default), `main()` counts the steady strides per foot in every trial and cuts **every** trial's series to the shortest one's. The length used is written in the results (`series_length`). An integer fixes $N$ instead, and `None` uses every stride.

::: {custom-style="Code Box"}
**In the code** — §12: `rng_for()`, `optimal_block_length()`, `block_indices()`, `percentile_ci()`, `series_summary()`, `bootstrap_rows()`. The series length is set in `main()` (RUN) through `steady_strides()`, and applied by `series_for()` (§17). Settings: `N_BOOT = 1000`, `CI_LEVEL = 0.95`, `SEED = 0`, `SERIES_LENGTH = "shortest"`.
:::

## 12.5 DFA α: a parametric bootstrap

Blocks would cut exactly the long-range correlation that α measures, so the block bootstrap cannot be used for DFA. Instead, $B=200$ series (`N_BOOT_DFA`) with the estimated $\hat\alpha$ and the same length are **simulated** as exact fractional Gaussian noise (Davies & Harte 1987). For $\hat\alpha\ge1$ the simulated noise is summed. DFA is rerun on each simulated series. The **basic** bootstrap interval is

$$\left[\,2\hat\alpha - q_{97.5},\ \ 2\hat\alpha - q_{2.5}\,\right],$$

where $q_{2.5}$ and $q_{97.5}$ are the percentiles of the α values of the simulated series.

It reflects the simulated spread around $\hat\alpha$, which corrects for DFA's small bias at this length. **Validation** (§20): at $N$ = 486 and α = 0.75, it covered the true α in 93% of trials, with a width of 0.33.

::: {custom-style="Note Box"}
**Caution** — a 95% interval about 0.3 wide means that a difference in α between two conditions smaller than about 0.15 is within one trial's uncertainty. DFA needs long series, and a few hundred strides is at the lower limit.
:::

## 12.6 Lyapunov exponents: a stride bootstrap

The divergence curve of §18 is kept separately for each reference stride. Blocks of strides are resampled (block length from the per-stride short-term slopes, §12.3; 500 resamples, `LDS_N_BOOT`), and the exponents are refitted to each resampled mean curve. The interval shows how much λ depends on which strides happened to be walked. It does not include the method's own bias, which is why λ is always compared between conditions with the same settings and the same $N$ (§18).

::: {custom-style="Code Box"}
**In the code** — `fractional_gaussian_noise()`, `simulate_alpha()`, `dfa_interval()` (§12); the LDS interval is in `window_lambda()` (§18).
:::

# 13. Summary statistics, symmetry and walk ratio

## 13.1 Mean, SD and CV of every per-step metric

For every per-step metric of §6–§11 (listed in `STEP_METRICS`), over the **steady** steps:

$$\bar x = \frac1N\sum x_i,\qquad \operatorname{sd} = \sqrt{\frac{1}{N-1}\sum (x_i-\bar x)^2},\qquad \text{CV} = 100\,\frac{\operatorname{sd}}{\left|\bar x\right|}\ \%.$$

Each is computed for both feet together, with its 95% block-bootstrap interval, and each foot's mean separately. In the results workbook:

| Column | Meaning |
|:--|:--|
| `step_width_m` | mean of all steady steps (both feet) |
| `step_width_m_sd`, `step_width_m_cv` | SD and CV over all steady steps |
| `step_width_m_L`, `step_width_m_R` | mean of the left foot's and of the right foot's steps |

::: {custom-style="Why Box"}
**Why SD and CV** — variability is the classic marker of gait control. Step width variability in particular separates groups that differ in lateral balance (Owings & Grabiner 2004). CV makes variability comparable between metrics with different means. It is not used for signed quantities near zero (such as the AP margin), where it is meaningless, and the SD is reported for those.
:::

## 13.2 Symmetry angle

For the metrics in `SYMMETRY_METRICS` (stance and swing time, step length, MFC, ML margin, propulsive impulse, F1, TRI), from the left and right means (Zifchock et al. 2008):

$$\text{SA} = \frac{45^\circ - \arctan\left(X_L / X_R\right)}{90^\circ}\times100\%,\qquad \text{SA} \leftarrow \text{SA} - 200\%\ \ \text{if SA} > 100\%.$$

0% is perfect symmetry, and a positive value means the right is larger.

::: {custom-style="Why Box"}
**Why the symmetry angle and not the symmetry index** — the usual index $(X_R - X_L)/\tfrac12(X_R+X_L)$ blows up when the mean of the two sides approaches zero (as it can for impulses or margins). It also needs one side to be chosen as the reference. The angle is bounded and needs no reference limb. An asymmetric load, or a limb that is favoured, would show up here.
:::

## 13.3 Walk ratio

$$\text{walk ratio} = \frac{\text{step length}}{\text{cadence}}\quad[\text{m}/(\text{steps/min})]$$

It is computed per step and averaged. Healthy adults keep it remarkably constant across walking speeds, at about 0.0063 m/(steps/min) (Sekiya & Nagasaki 1998). It shows *how* a walker achieves a speed: with longer steps or with faster ones. With the belt speed and the step rate both fixed, it mostly shows whether the participant follows the metronome.

## 13.4 Data-quality descriptors

Every trial row also says what the trial was and how trustworthy its data are. These columns are family "Trial" in the workbook:

* measured belt speed, system mass, load, and the CV of the quiet standing;
* the share of events from the force plates, from Zeni and from interpolation;
* the number of steady steps, and the share of stances trusted for kinetics;
* the share of swings without an MFC, and with an MFC below the belt;
* the share of steps with handrail contact;
* the GRF-vs-marker correlations (§5.7);
* the load offset in the trunk frame (§5.6).

Check these columns first. A trial with, say, 30% of its events interpolated or a GRF-vs-marker correlation below 0.7 needs a closer look before its metrics are used.

::: {custom-style="Code Box"}
**In the code** — §13: `STEP_METRICS`, `SYMMETRY_METRICS`, `row()`, `steady_steps()`, `trial_descriptors()`, `step_summaries()`, `symmetry_angle()`, `symmetry_rows()`. The walk ratio is computed per step in `spatiotemporal()` (§6).
:::

{{ranges: walk_ratio, symmetry_angle_stance_s, symmetry_angle_step_length_m, symmetry_angle_f1_bw}}

: Table 13.1. Healthy ranges for the walk ratio and the symmetry angle.
