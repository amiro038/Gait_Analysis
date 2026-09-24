# 19. Results tables and figures

## 19.1 What is written

Everything goes to `OUTPUT_FOLDER` (`gait_analysis_outputs/`, next to the event outputs).

| File | Shape | Use |
|:--|:--|:--|
| `all_trials_metrics.xlsx` | **one row per trial, one column per metric** | the file to read (sheets below) |
| `all_trials_metrics.csv` | the same values as CSV | for R, SPSS, Python |
| `all_trials_metrics_ci.csv` | the 95% limits of every value that has one (`_lo`, `_hi`) | error bars |
| `metric_dictionary.csv` | every column: family, description, unit, document section, healthy range and source | look up a column |
| `all_trials_summary.csv` | long form: one row per number, with its interval, N, block length and a note | mixed models (participant random, condition fixed) |
| `{trial}_steps.csv` | one row per step, every per-step value | checking, per-step statistics |
| `{trial}_waveforms.npz` | force and angle curves (101 points), LDS and MSE curves | plotting |
| `{trial}_lds_windows.csv` | every LDS window | checking λ |
| `{trial}_qa.png`, `{trial}_variability.png` | per-trial figures (§19.3) | a first look at every trial |

**The workbook's sheets:**

* **Key metrics**: about 35 columns that answer the main questions.
* **One sheet per family**: Trial, Spatiotemporal, Stability, Kinetics, CoM work, Posture, Symmetry, DFA, Entropy, Trunk, Control, LDS.
* **Healthy ranges**: every column that has a range, with each trial's cell **green** inside the range and **orange** outside it; the range is in the header.
* **95% CI**: the interval limits.
* **Dictionary**: the meaning of every column.

In every sheet, row 1 describes the column in words with its unit, row 2 gives the column name, and the first three columns (participant, condition, trial) stay frozen.

**Column names.** A metric's name alone (`step_width_m`) is the mean over all steady steps of both feet. `_sd` and `_cv` are its SD and CV. `_L` and `_R` are each foot's mean or, for DFA, entropy, GEM and foot placement, the value from that foot's own series.

::: {custom-style="Code Box"}
**In the code** — §19: `summarise()`, `CATALOG` (the meaning, unit and section of every column), `FAMILIES`, `KEY_METRICS`, `describe()`, `wide_column()`, `reference_for()`, `wide_tables()`, `build_tables()`, `write_workbook()`. `python gait_analysis.py --tables` rebuilds the tables from the per-trial summaries without re-running any trial.
:::

## 19.2 The healthy reference ranges

The ranges in Table 19.1 are the ones used throughout this document and in the workbook's "Healthy ranges" sheet. They are stored once, in `REFERENCE_RANGES` (§19 of the code). Per-foot columns (`_L`, `_R`) share their metric's range.

![Figure 25. How a trial is compared with the healthy ranges (synthetic trial, 20 kg load). The green band is each metric's healthy range, scaled so that it runs from its low to its high end. The dot is the trial's value: green inside, orange outside. The workbook colours each trial's cell in the same way.](figures/fig25.png){width=6.3in}

::: {custom-style="Note Box"}
**How to read an orange cell** — first ask whether the difference is expected: from the load (Appendix B), from the treadmill, from the metronome (§2.5, §14), or from a different boundary definition (§10). Then check the trial's quality columns (§13.4). A value outside the range is a prompt to look, not a verdict.
:::

{{allranges}}

: Table 19.1. Every healthy reference range (healthy adults without load, about 1.2–1.4 m/s, unless stated). "Checked" = verified against the source's abstract or a review; "verify" = check against the full paper before publication.

## 19.3 The per-trial figures

With `MAKE_FIGURES = True`, each trial gets two figures:

* **`{trial}_qa.png`: is the trial sound?** It shows the quiet standing and the weight, where the events came from along the trial, the clearance over the belt, the margins through the trial, the force of the trusted stances and the velocity fusion.
* **`{trial}_variability.png`**: DFA with its interval, α for every series, the LDS divergence curves and the multiscale entropy curves.

With `MAKE_VIDEO = True`, a bird's-eye video of one typical stride is also saved (`{trial}_mos_stride.mp4`). It shows the boots, the CoM, the xCoM and the margins in the belt frame. The typical stride is the one closest to the median in stride time, step width and both margins.

::: {custom-style="Code Box"}
**In the code** — §19 FIGURES: `qa_figure()`, `variability_figure()`, `typical_stride()`, `mos_video()`. Settings: `MAKE_FIGURES`, `MAKE_VIDEO`.
:::

# 20. Validation

Every method was checked on data whose answer is known, using the code exactly as shipped (not a copy of it). There are three checks, and they should be rerun after any change to the code.

## 20.1 The synthetic trial

`synthetic_trial.py` builds a complete trial in the exact file formats of the real data:

* an 80 kg walker carrying a 20 kg load 150 mm behind the trunk;
* a quiet standing at the start, then walking on a belt at 1.3 m/s on a treadmill pitched 0.84°;
* the participant's real boot meshes, bending at the MTP;
* force plates whose forces are computed from where each sole touches, in the plate's own frame (rotated 90° and shifted from Theia's), with baselines, drift and noise;
* steps that cross onto the other belt, straddle the gap or hang over it;
* a slowly unloading belt, a tracking dropout, and swings with a known clearance.

The figures in this document were all drawn from a 7-minute version of it.

## 20.2 Event detection (`test_detect_gait_events.py`)

| Check | Result |
|:--|:--|
| plate registration (rotation, translation) | 0.09°, 5.0 mm from the truth |
| belt pitch | 0.89° (true 0.84°) |
| deepest sole point in stance | bent boot −3/−4 mm; rigid boot −17/−16 mm (the reason for the toe hinge) |
| trusted force events | 184 of 200; worst 0.20 frames (2 ms) off |
| force events from crossed or straddled steps | none |
| the step hanging over the gap | keeps both force events |
| every true event recovered | 205 of 205, none missed, worst 0.33 frames |
| Zeni offsets | heel strike −22 ms, toe-off +19 ms; after correction 0.7–0.8 ms mean and 1.8 ms 95th-percentile error |

## 20.3 The whole analysis (`test_gait_analysis.py`)

| Check | Result |
|:--|:--|
| weight in quiet standing | 99.9 kg; load 19.9 kg (true 100 / 20) |
| load position | 152 mm behind the trunk (true 150) |
| system CoM | 1.9 mm from the truth (median); Theia's body CoM alone is 82 mm off |
| fused CoM velocity | 4–8 mm/s RMS error; markers alone 25 mm/s |
| belt speed | 1.299 m/s (true 1.3) |
| stance time | −0.1 ± 0.4 ms from the truth |
| step width | 1.0 mm from the truth (median) |
| MFC | found in 334 of 337 swings, 1.0 mm from the built-in clearance |
| ML margin of stability | 0.8 mm from the margin computed with the true CoM and velocity |
| kinetics | each limb's vertical impulse carries 50.2% of the weight over a stride (50%); braking and propulsion cancel (net −0.0001 BW·s); free moment 1.00 × the one built in; CoM work nets to zero (28.2 J in, 28.2 J out) |
| results | 373 metric columns, each in the dictionary with its section; DFA and LDS intervals present |

## 20.4 The metric methods (`gait_metrics_validation.py`, `lds_validation.py`)

| Method | Test | Result |
|:--|:--|:--|
| velocity fusion (§5.7) | 1, 2, 5 mm marker noise | fused 0.5–3, 2–8, 5–19 mm/s vs 25, 49, 124 mm/s for markers alone; no bias below 0.5 Hz |
| margin of stability (§10) | hand-computed case | exact to 0.01 mm |
| MFC, MoI, TRI (§11) | built swing | exact; a swing without a minimum gives missing |
| DFA (§14) | fractional Gaussian noise, H = 0.5, 0.7, 0.9, N = 486 and 756 | unbiased within ±0.02; shuffled → 0.505 |
| DFA interval (§12.5) | 60 trials, N = 486, α = 0.75 | covers the truth 93%, 0.33 wide |
| block bootstrap (§12.2) | AR(1), φ = 0.6, N = 400 | covers the true mean 93% (single strides: 66%) |
| sample entropy (§15.1) | white noise against the closed form | within 0.006 |
| multiscale entropy (§15.2) | white vs 1/f noise | white 2.19 → 1.02 over scales; 1/f stays 1.53 → 1.50 |
| harmonic ratio (§16.2) | two-tone signals | exact (HR 20, 4, 1); ML inverts; k-weighting from velocity exact |
| regularity (§16.3) | drifting stride time | peaks 0.999 where a fixed lag gives 0.937 |
| GEM, foot placement (§17) | built components and gains | recovered exactly |
| Lyapunov exponents (§18) | systems with known exponents (`lds_validation.py`) | recovered by the shipped `divergence()` |

## 20.5 What the synthetic data cannot check

The synthetic trial tests the **calculations**. It does not test:

* how accurate Theia's skeleton is on real people in boots and loads (see Kanko et al. 2021 for unloaded gait);
* the load's true height (it cannot be measured from standing, §5.6);
* whether a real participant's quiet standing was truly still (the 2% walking-weight check, §5.5, is the safeguard).

The per-trial quality columns (§13.4) and the QA figure (§19.3) are there to catch problems with real data.
