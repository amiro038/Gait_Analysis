# 6. Spatiotemporal metrics

**What.** The timing and the geometry of each step. One value is computed per heel strike; §13 turns them into trial means, SDs and CVs.

## 6.1 Timing

With the events of §5.1, all in seconds:

$$T_{stance} = t_{TO} - t_{HS},\qquad T_{swing} = t_{HS}^{next} - t_{TO},\qquad T_{stride} = t_{HS}^{next} - t_{HS},\qquad T_{step} = t_{HS} - t_{HS,o}^{prev}$$

$$\text{cadence} = \frac{60}{T_{step}}\ \text{steps/min}$$

$$T_{DS} = (t_{TO,o} - t_{HS}) + (t_{TO} - t_{HS,o}),\qquad T_{SS} = t_{HS,o} - t_{TO,o}$$

The first bracket of $T_{DS}$ is the initial double support, and the second the terminal double support.

Stance, swing, single and double support are also given as a percentage of the stride (`_pct`), which is the usual convention (the "gait cycle").

**Steps** (`spatiotemporal()`):

1. Compute each interval from the fractional event times. An interval is kept only if its end comes after its start.
2. Compute double and single support only when the five events are in their natural order, $t_{HS} < t_{TO,o} < t_{HS,o} < t_{TO} < t_{HS}^{next}$. A missed or doubled event would otherwise give a negative or doubled time.

::: {custom-style="Why Box"}
**Why double support matters here** — with a load, walkers spend longer with both feet down. That buys stability at the cost of speed, and it is one of the most consistent effects of load carriage (Appendix B). **Why cadence from step time** — cadence counts steps, and each step time is measured directly between consecutive heel strikes of opposite feet.
:::

## 6.2 Distances

The two **heels** are read at the heel strike, at the exact fractional time, and projected on the walker's axes (§5.2):

$$\text{step length} = (\mathbf{p}_{heel,b} - \mathbf{p}_{heel,o})\cdot\hat{f},\qquad \text{step width} = \left|(\mathbf{p}_{heel,b} - \mathbf{p}_{heel,o})\cdot\hat{\ell}\right|.$$

**Stride length on a treadmill** is how far the foot travelled **over the belt** from one landing to the next (Dingwell et al. 2010):

$$L_{stride} = v_b\,T_{stride} + \left(\mathbf{p}_{heel,b}(t_{HS}^{next}) - \mathbf{p}_{heel,b}(t_{HS})\right)\cdot\hat{f},\qquad v_{stride} = \frac{L_{stride}}{T_{stride}}.$$

The second term is small. It is how much further forward (or back) in the lab the foot landed this time than last time.

![Figure 13. A: a heel strike seen from above, with travel to the right. The step length is the distance along the belt from the other heel (open circle) to the landing heel (filled circle); the step width is the distance across it. B: why two step lengths do not add up to a stride on a belt. During one stance the heel is compared with a point riding the belt. It stays with the belt through mid-stance (flat part), but by the other foot's heel strike (dashed line) it has already started to lift, 30–50 mm ahead of where it landed.](figures/fig13.png){width=6.5in}

::: {custom-style="Why Box"}
**Why heels** — the heel is the point that lands, and it is tracked well at heel strike. **Why along and across the belt** — see §1.3: projecting on Theia's X and Y would move part of the step length into the width. **Why stride length is not two step lengths** — overground, a stride is the sum of two steps. On a belt, the rear heel has already begun to lift, and has moved relative to the belt, by the time the other foot lands (Fig. 13B). Two heel-to-heel step lengths therefore overestimate the stride by 30–50 mm each. Measuring the foot's travel over the belt gives the true stride. It also makes stride speed ($L/T$) equal to the belt speed on average, which is the task goal used by the GEM (§17).
:::

## 6.3 Where the events came from

Each step also records `events_from_kinematics` (either event came from Zeni or was interpolated) and `events_interpolated`. The per-step data can therefore be filtered to force-plate events only, to check that a result does not depend on the event source.

::: {custom-style="Code Box"}
**In the code** — §6: `spatiotemporal()`. Per-step columns in `{trial}_steps.csv`: `stance_s`, `swing_s`, `stride_s`, `step_s`, `cadence_spm`, `initial_ds_s`, `terminal_ds_s`, `double_support_s`, `single_support_s` and their `_pct`, `step_length_m`, `step_width_m`, `stride_length_m`, `stride_speed_ms`, `walk_ratio`.
:::

::: {custom-style="Range Box"}
**Healthy range** — for healthy adults at about 1.3 m/s (Table 6.1). With the 108 steps/min metronome, cadence is set at 108 and the stride time at 1.11 s. Their variability is then partly the precision of beat-following.
:::

{{ranges: stride_s, cadence_spm, stance_pct, double_support_pct, single_support_pct, step_length_m, step_width_m, stride_s_cv, stride_length_m_cv, step_width_m_sd}}

: Table 6.1. Healthy ranges for the spatiotemporal metrics (column names as in the results workbook).

# 7. Posture and joint motion

**What.** How the trunk, pelvis and lower-limb joints move over each stride.

For each stride (this foot's heel strike to its next), using the first (sagittal) component of Visual3D's angles:

| Results column | Signal | Statistic over the stride |
|:--|:--|:--|
| `trunk_lean_mean_deg`, `_rom_deg` | `Thorax_Seg_Angle` X (thorax relative to the lab) | mean, range |
| `pelvis_tilt_mean_deg` | `Pelvis_Seg_Angle` X | mean |
| `hip_rom_deg`, `knee_rom_deg`, `ankle_rom_deg` | `<Side>_Hip/Knee/Ankle_Joint_Angle` X | range (max − min) |

$$\text{mean} = \frac{1}{n}\sum_{i\in\text{stride}}\theta_i,\qquad \text{ROM} = \max_{\text{stride}}\theta - \min_{\text{stride}}\theta$$

**Steps** (`posture()`):

1. For each stride with a valid next heel strike, take the frames from the heel strike to the next one.
2. Skip the stride for that angle if any frame is missing, rather than filling it.
3. Compute the mean and the range. Store the curve resampled to 101 points (0–100% of the stride) for the figures and the waveform file.

![Figure 14. Mean ± SD angle curves over the stride (left foot, synthetic trial; the synthetic angles are simple sinusoids, not real joint curves). The arrows show the range of motion, the statistic reported for the joints; for the trunk the mean is also reported.](figures/fig14.png){width=6.5in}

::: {custom-style="Why Box"}
**Why these variables** — a load changes posture before it changes timing. The trunk leans forward to bring the system CoM back over the feet, and the knee flexes more during weight acceptance to absorb the extra load. Trunk lean and knee range of motion are therefore the most load-sensitive angles (Appendix B). **Why the sagittal component only** — it is the plane in which walking and load carriage act, and markerless sagittal angles agree best with marker-based ones (Kanko et al. 2021). **Why ranges rather than peaks at events** — a range does not depend on the exact event time, and it is robust to a constant offset in the angle definition. **Signs** are Visual3D's. They are the same in every trial, so differences between conditions are valid whichever direction counts as positive.
:::

::: {custom-style="Code Box"}
**In the code** — §7: dictionary `ANGLES` (output name → Visual3D signal), function `posture()`. Setting: `WAVE_POINTS = 101`.
:::

{{ranges: hip_rom_deg, knee_rom_deg, ankle_rom_deg}}

: Table 7.1. Healthy sagittal ranges of motion over the stride. Trunk lean and pelvis tilt depend on each system's reference posture, so no absolute range is given: compare them between conditions.

# 8. Kinetics

## 8.1 Which stances

Kinetics are computed only for stances whose heel strike **and** toe-off were both trusted force events of the **same belt contact** (§4.3): one boot on that belt, the other boot off it, and the CoP under the boot. Any other stance may carry two feet's force (a crossover onto a shared belt) or part of one foot's (a straddle). Such stances are left **missing**, not estimated. Because the event record names the belt, a clean crossover step is read from the belt it actually landed on. The share of steady stances used is reported (`stances_kinetics_pct`).

::: {custom-style="Why Box"}
**Why so strict** — a single stance mixing two feet's force would add a peak of about 2 BW and bias the trial's mean F1. Leaving it out loses one stance among hundreds.
:::

## 8.2 Vertical force landmarks

For one trusted stance, with $F_z(t)$ the 50 Hz-filtered vertical force:

$$F_1 = \max_{t < T/2} F_z,\qquad F_2 = \max_{t \ge T/2} F_z,\qquad \text{trough} = \min_{t_{F1}\le t\le t_{F2}} F_z$$

**Loading rate** is the least-squares slope of $F_z$ against time, between the moments it first reaches 20% and 80% of $F_1$ (`LOADING_BAND`):

$$\text{LR} = \frac{\sum (t_i-\bar{t})(F_i-\bar{F})}{\sum (t_i-\bar{t})^2},\qquad t_i \in [t_{20\%},\ t_{80\%}].$$

## 8.3 Impulses along the belt

With $F_{AP} = \mathbf{F}\cdot\hat{f}$ (positive = forward) at 1000 Hz, $\Delta t$ = 1 ms:

$$J_{brake} = \sum \min(F_{AP}, 0)\,\Delta t,\qquad J_{prop} = \sum \max(F_{AP}, 0)\,\Delta t,\qquad J_{vert} = \sum F_z\,\Delta t.$$

Also reported: the peak braking and peak propulsive forces.

## 8.4 CoP excursion and free moment

The CoP (low-passed at 15 Hz, `COP_FILTER_HZ`) gives its range along and across the belt within the stance (`cop_ap_range_mm`, `cop_ml_range_mm`). The free moment (§5.4, 15 Hz) gives its largest absolute value (`free_moment_peak_nm`).

## 8.5 Normalisation: per body weight and per total weight

Every force and impulse is given twice: divided by **body weight** $m_b g$ (`_bw`), and by **total weight** $m g$ weighed in §5.5 (`_tw`).

::: {custom-style="Why Box"}
**Why both** — per body weight is the convention, and it shows the load the body bears: with a load, $F_1$ in BW rises roughly in proportion to the added weight (Birrell et al. 2007). Per total weight asks whether the *pattern* changed beyond simply carrying more. If $F_1$/TW is unchanged, the walker is carrying the extra weight in the same way. **Why a fitted slope for the loading rate** — a two-point slope, or the peak of the derivative, is dominated by the heel-strike transient and by noise. A least-squares fit over 20–80% of F1 is the stable, conventional definition. **Why the impulses** — braking and propulsion must cancel over a stride at steady speed (a check). Propulsive impulse reflects push-off effort, which load increases.
:::

![Figure 15. One trusted stance (synthetic trial, 20 kg load, per body weight). A: the vertical force with F1, the trough and F2; the blue line is the loading-rate fit (20–80% of F1). B: the force along the belt. Braking (blue area, negative) is followed by propulsion (orange area). At steady speed the two impulses cancel over the stride.](figures/fig15.png){width=6.5in}

::: {custom-style="Code Box"}
**In the code** — §8: `trusted_stance()`, `kinetics()`. Settings: `LOADING_BAND = (0.20, 0.80)`, `COP_FILTER_HZ = 15`. The waveforms (vertical and AP force per total weight, 101 points) are saved in `{trial}_waveforms.npz`.
:::

::: {custom-style="Range Box"}
**Healthy range** — for healthy, unloaded adults per body weight (Table 8.1). With a load, the `_bw` values are expected to exceed these ranges roughly in proportion to $m/m_b$; the `_tw` values should stay close to the unloaded pattern.
:::

{{ranges: f1_bw, trough_bw, f2_bw, f1_tw, f2_tw, loading_rate_bw_per_s, peak_braking_bw, peak_propulsive_bw, braking_impulse_bw_s, propulsive_impulse_bw_s, vertical_impulse_bw_s, cop_ap_range_mm, cop_ml_range_mm, free_moment_peak_nm}}

: Table 8.1. Healthy ranges for the kinetic metrics.

# 9. Centre-of-mass work: the individual limbs method

**What.** The mechanical work each leg does on the centre of mass, phase by phase through the stance (Donelan, Kram & Kuo 2002).

**Equations.** Each leg's power into the system CoM is the dot product of that leg's GRF with the CoM velocity in the **belt frame** (§5.7):

$$P_{leg}(t) = \mathbf{F}_{leg}(t)\cdot\mathbf{v}_{belt}(t)$$

It is integrated over the phases of this foot's stance (Fig. 7), counting positive and negative power separately:

| Phase | From → to | Work |
|:--|:--|:--|
| collision | heel strike → other toe-off (initial double support) | negative part |
| rebound | other toe-off → other heel strike (single support) | positive part |
| preload | other toe-off → other heel strike (single support) | negative part |
| push-off | other heel strike → toe-off (terminal double support) | positive part |

$$W_{pos} = \sum \max(P, 0)\,\Delta t,\qquad W_{neg} = \sum \min(P, 0)\,\Delta t$$

Work is given in J and per kg of system mass (`_j_kg`), together with the positive and negative work over the whole stance.

**Steps** (`kinetics()`, `phase_work()`):

1. Only trusted stances (§8.1), and only if the five events are in order.
2. Read the fused belt-frame velocity at every 1000 Hz force sample, interpolating between the 100 Hz frames.
3. Compute the power, and sum its positive and negative parts over each phase.

![Figure 16. Each leg's power into the CoM over one stride from the left heel strike, with the four phases of the left stance labelled (shading alternates by phase). The work in a phase is the area under the power curve, its positive and negative parts summed separately. The synthetic forces come from a simple model of the CoM's motion, so these curve shapes are not those of real walking. In real walking the leading leg does negative work in the collision while the trailing leg does positive work at push-off (Donelan et al. 2002).](figures/fig16.png){width=6.5in}

::: {custom-style="Why Box"}
**Why individual limbs** — the older "combined limbs" method adds both legs' forces before multiplying by the velocity. In double support the trailing leg pushes (positive) while the leading leg collides (negative), and their sum hides both. Separating the legs reveals the step-to-step transition, which is the main mechanical cost of walking and rises with load. **Why the belt frame** — on a treadmill the lab-frame CoM hardly moves forward. The work done on a mass moving at 1.3 m/s over still ground is recovered by adding the belt speed to the velocity. **Why the system velocity and mass** — the legs move the body *and* the load.
:::

::: {custom-style="Code Box"}
**In the code** — §9 is in the same loop as §8: `kinetics()` calls `phase_work()`. Results: `collision_j_kg`, `rebound_j_kg`, `preload_j_kg`, `push_off_j_kg`, `com_work_pos_j_kg`, `com_work_neg_j_kg`.
:::

{{ranges: push_off_j_kg, collision_j_kg}}

: Table 9.1. Healthy ranges for CoM work per step (unloaded, about 1.25 m/s). Values per kg of system mass.
