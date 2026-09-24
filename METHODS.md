# Gait analysis methods — DICE split-belt treadmill, load carriage

This document describes how every number in the results tables is produced:
the equipment and data, how the boots are reconstructed, how gait events are
found, how each trial is prepared, and then, metric by metric, the definition,
the equations, the parameters and why they were chosen, the output columns, and
how to read the value. Section numbers match the `methods_section` column of
`metric_dictionary.csv` and the **Dictionary** sheet of
`all_trials_metrics.xlsx`.

The code is `detect_gait_events.py` (events) and `run_gait_analysis.py` with the
`gait/` package (metrics); the module that implements each section is named in
it. Parameter values are collected in Appendix A.

**Citation confidence.** References marked ✔ were checked against the source;
those marked ~ are from domain knowledge and their volume/page should be
verified before publication. Literature values are approximate and
protocol-dependent: use them as sanity ranges, not norms.

---

## Contents

1. Overview
2. Data and equipment
3. The boots: MOVE4D meshes on the Theia feet
4. Gait events
5. Preparing a trial
6. Spatiotemporal metrics
7. Posture and joint motion
8. Kinetics
9. Centre-of-mass work (individual limbs method)
10. Margin of stability
11. Minimum foot clearance, margin of instability, trip risk
12. Summary statistics, symmetry and walk ratio
13. Long-range correlations (DFA)
14. Entropy
15. Trunk acceleration: harmonic ratio and regularity
16. Control of stepping: goal-equivalent manifold and foot placement
17. Local dynamic stability
18. Uncertainty and comparability
19. The results tables
20. Validation

Appendix A — parameters. Appendix B — references.

---

## 1. Overview

```
build_foot_binding.py   once per participant: MOVE4D boot scan -> Theia foot frames
detect_gait_events.py   per trial: every heel strike and toe-off
run_gait_analysis.py    per trial: prepare (gait/trial.py), then every metric,
                        then one row per trial in the results tables
```

Each trial is analysed in the same order:

1. **Events** (§4): trusted force-plate events first, Zeni's kinematic method for
   the rest.
2. **Preparation** (§5): steps and steady walking, the belt's speed and
   direction, the posed boot soles over the pitched belt, the ground reaction
   forces in the motion-capture frame, the **load** weighed in the opening quiet
   standing, the **system (body + load) centre of mass** and its velocity.
3. **Per-step metrics** (§6–11): one value per step, in `{trial}_steps.csv`.
4. **Per-trial metrics** (§12–17): summaries of the steady steps, and the
   stride-series and continuous-signal measures.
5. **Uncertainty** (§18): a 95% confidence interval for every value that has
   one.
6. **Tables** (§19): one row per trial, one column per metric.

Two protocol facts run through everything:

- **The treadmill.** The belt carries the stance foot backwards at a fixed
  speed, so anything that depends on the body's forward motion must be
  expressed relative to the belt, not the laboratory (§5.2).
- **The metronome.** Speed (belt) and cadence (108 bpm) are both prescribed.
  Timing series are therefore partly set by the protocol; their variability
  structure measures how the walker follows the beat as much as intrinsic
  control. Series the metronome does not constrain — step width, foot
  clearance, margins, impulses — are the better primary outcomes for the
  variability measures (§13–14). The tables mark metronome-paced series.

---

## 2. Data and equipment

| Source | What | Rate |
|---|---|---|
| Theia3D markerless motion capture, exported through Visual3D (`{trial}_metrics.csv`) | segment poses (`*_Global_4x4`), joint centres, `Whole_body_COG`, trunk and pelvis signals, joint and segment angles | 100 Hz |
| DICE split-belt instrumented treadmill (`FP_renamed/{trial}.csv`) | per belt: force, moment and centre of pressure (CoP) in the plate's own frame; handrail forces | 1000 Hz |
| C3D force-plate parameters (`force_plates_DICE_treadmill.txt`) | each plate's measured corners, length and width | — |
| MOVE4D 3D body scan (per participant) | the participant's boots as segmented surface meshes | — |
| `participants.csv` | body mass **without the carried load**, leg length, height | — |

Time alignment: kinematic frame (row) *r* corresponds to force sample 10*r*
(the two recordings start together). Every metric works in kinematic **rows**
(0 = first frame), with sub-frame event times where available.

Axes: Theia's frame is used throughout, but every AP (anteroposterior) and ML
(mediolateral) quantity is projected on the **belt's measured direction of
travel** and the horizontal direction perpendicular to it (§5.2), never on
Theia's X and Y directly.

---

## 3. The boots: MOVE4D meshes on the Theia feet

*`build_foot_binding.py`, `apply_binding.py`; used through `detect_gait_events.py`.*

**Transform.** A static A-pose recorded by both systems gives the rigid
transform from MOVE4D to Theia coordinates: the rotation from eight segment
long axes and four bilateral medio-lateral axes (Kabsch), the translation from
joint centres, the vertical offset fitted (+29.8 mm for D05) and corroborated by
the floor under the scanned soles (README, "build_foot_binding.py").

**Binding.** Each boot mesh is expressed in its Theia foot segment's local frame
using the same static pose. In any trial a vertex's position is then exactly

$$\mathbf{p}(t) = \mathbf{R}_{foot}(t)\,\mathbf{v}_{local} + \mathbf{t}_{foot}(t)$$

with $\mathbf{R}, \mathbf{t}$ from `<Side>_Foot_Global_4x4`.

**Toe hinge.** A rigid boot drives its toe cap 30–40 mm through the belt at
every push-off. The toe cap is therefore rotated about the MTP joint (the medio-
lateral axis through `<Side>_Toes_Position`, fixed in the foot frame) by
Theia's `<Side>_Toes_Joint_Angle` (X component), measured from its value in the
static scan. Vertices in front of the MTP ride the toe; the rest ride the foot.

**The sole.** For contact, clearance and base of support only the underside
matters: the boot footprint is divided into 10 × 10 mm cells and the lowest
vertex of each (in the foot's "up" direction, up to 35 mm above the lowest of
all) forms the sole (~290 points per boot). Every analysis that needs the boot
uses these posed, toe-bent sole points.

**The belt surface.** The DICE plates are pitched ~0.84° (13 mm over one
stance), and Theia's foot height is not the same standing and walking, so the
floor is not assumed to be z = 0. The surface is fitted to the lowest sole
point of each boot in contact frames (robust iterative least squares):

$$z_{belt}(s) = z_0^{L\,or\,R} + k\,s$$

with $s$ the horizontal distance along the walking direction, one slope $k$
for the treadmill and one offset per boot (absorbing a height bias in either
boot's pose). All clearances are heights above this surface.

---

## 4. Gait events

*`detect_gait_events.py`.* Output: `{trial}_merged_events.csv`,
`{trial}_event_qa.csv` (sub-frame times, method, belt), `{trial}_grf_contacts.csv`
(every contact and why its events were or were not trusted),
`{trial}_event_summary.json` (plate fit, registration, belt surface, Zeni error).

### 4.1 Contacts from the force plates

Per belt, the vertical force is low-pass filtered (4th-order Butterworth,
50 Hz, zero lag) and its unloaded baseline removed; the baseline is re-measured
every 10 s from the lowest 20% of samples, and a window that is never unloaded
(the quiet standing) borrows its neighbours'. A contact is a run above 20 N,
merging dips shorter than 30 ms and dropping runs shorter than 80 ms.

**Edge refinement.** A zero-lag filter spreads a steep landing over a few
milliseconds, so the filtered force crosses the threshold 1–2 ms before the
foot lands and after it leaves. Each edge is therefore moved to where the
**raw** force crosses 20 N and stays across it for 5 samples, searched within
±15 ms of the filtered crossing.

### 4.2 Which force-plate events to trust

The plate outlines come from the C3D corners (the nominal 559 × 1778 mm
rectangle fitted rigidly to the four measured corners; the worst corner miss,
8 mm, is the outline's uncertainty). The CoP is converted from the plate frame
to the lab, and the lab is registered to Theia from single-support frames
(rotation from the belt's axis against the stance boot's velocity, translation
from the CoP against the boot's contact patch).

A contact is labelled by the posed boots over it (`clean`, `crossover`,
`straddle`, `shared`), and **each of its two events is judged on its own**
over ±50 ms. It is trusted only if:

| Check | Fails as |
|---|---|
| the contact is complete in the recording | `incomplete` |
| the force reaches 200 N within 60 ms of landing / leaves it within 80 ms of lift-off | `slow_loading` / `slow_unloading` |
| both boots are tracked | `no_mesh` |
| the other boot has no touching sole points on this belt | `other_foot_on_belt` |
| this boot is touching the belt | `foot_not_in_contact` |
| this boot has no touching sole points on the other belt | `foot_on_other_belt` |
| the CoP lies within 30 mm of the boot's contact patch for 90% of the contact | `cop_outside_foot` |

"Touching" is a sole point within 15 mm of the belt surface; "on a belt" is at
least 3 touching points inside the belt outline grown by its 8 mm uncertainty.
Hanging over the gap or the outer edge does not cost an event, since it moves
no force onto the other plate. The limb is the boot's, so a clean crossover
step keeps its events with the right limb.

### 4.3 Zeni for the rest

Every event that is not trusted is taken from Zeni et al. (2008): heel strike
at the maximum of the heel's forward distance ahead of the pelvis, toe-off at
the maximum of the toe's distance behind it (positions low-pass filtered at
10 Hz, sub-frame peak by parabolic interpolation). Each missing event is
searched for only in the gap the sequence L HS → R TO → R HS → L TO leaves for
it, and then shifted by Zeni's median offset from this trial's trusted GRF
events for that limb and event (both limbs pooled if fewer than 10). The
median and 95th-percentile error of Zeni against GRF are written to the event
summary. A Zeni peak must stand at least 50 mm above the lowest point within
±0.6 s: standing still, the heel-to-pelvis distance only wobbles by millimetres,
and without this the quiet standing produced phantom steps.

Across 11 D05 trials Zeni had the smallest worst-trial error of seven
candidate kinematic methods.

### 4.4 Last resorts and flags

Where Zeni finds nothing (a tracking dropout): a clean-looking contact the mesh
could not check (`GRF_unverified`), then interpolation between neighbouring
events (`interpolated`). Each limb's sequence is trimmed to start on a heel
strike and end on a toe-off; stance and stride times outside 0.75–1.25 × the
limb's median are flagged in the QA file.

**Accuracy (synthetic trial with known events):** trusted GRF events within 0.2
frames (2 ms), Zeni-filled events within 0.33 frames, stance time −0.1 ± 0.4 ms.

---

## 5. Preparing a trial

*`gait/trial.py`.*

### 5.1 Steps and steady walking

Events are read from `{trial}_event_qa.csv` (sub-frame rows). One row is made
per heel strike, with:

| Field | Meaning |
|---|---|
| `hs`, `to`, `next_hs` | this foot's heel strike, its toe-off, its next heel strike |
| `contra_to`, `contra_hs` | the other foot's toe-off and heel strike that follow `hs` |
| `prev_contra_hs` | the other foot's heel strike before `hs` |
| `hs_source`, `to_source`, `hs_belt`, … | how each event was found, which belt carried it |

**Steady walking** starts `WARMUP_S` = 60 s after the first event (the belt
ramp and acclimatisation). Every summary value uses steady steps only
(`steps_steady`).

### 5.2 The belt's speed and direction of travel

A stance foot rides the belt, so during foot-flat (25–55% of stance) its heel
moves at the belt's velocity. A straight line is fitted to the heel's
horizontal position over that window in every stance; the belt's direction of
travel is minus the median unit velocity, and the belt speed the median speed
along it (`belt_speed_ms`).

This direction — not the way the boots point — defines AP and ML everywhere:
boots toe out, and 1.3° of that leaked ±17 mm of step length into step width
in testing. **Belt frame:** wherever the body's forward velocity matters (the
extrapolated CoM, CoM work), the belt speed is added along the direction of
travel, which makes the treadmill equivalent to walking over stationary
ground.

### 5.3 Boots over the belt

The posed sole points (§3) are computed for every frame, and kept per frame:
the lowest point's height above the belt surface (and which point it is), and
the sole's extent along and across the direction of travel (for the base of
support). Whole-sole positions are rebuilt on demand for swings and figures.

### 5.4 The load and the system centre of mass

Theia's `Whole_body_COG` is the body the markerless model sees; it does not
include the mass of a pack or a vest. The force plates see everything.

**Quiet standing.** Every trial opens with a few seconds of standing still. The
steadiest 2 s window before the first step is found in which both belts are
loaded (each carries ≥15% of the total), both heels are still (≤20 mm of travel
between the first and last 0.2 s), and the total vertical force varies by less
than 2% (CV, `quiet_standing_cv`). Its mean total vertical GRF is the system
weight $W$:

$$m = \frac{W}{g}, \qquad m_{load} = m - m_{body}$$

with $m_{body}$ from `participants.csv` (`system_mass_kg`, `load_kg`). The mean
vertical GRF during steady walking is compared with $W$; more than 2% apart
signals plate drift or a standing that was not still.

**Where the load is.** Standing still, the CoP lies vertically below the system
CoM, so horizontally

$$\mathbf{x}_{load} = \frac{m\,\mathbf{x}_{CoP} - m_{body}\,\mathbf{x}_{CoM,body}}{m_{load}}.$$

Two parts cannot be measured this way and are set:

- **Height** — standing still gives no vertical information; the load CoM is
  placed at `Trunk_Position` + `LOAD_HEIGHT_M` (0 m).
- **Left–right** — the CoP locates the load only to the plate registration's
  error times $m/m_{load}$ (≈5: 4 mm of registration becomes 20 mm of load),
  and packs and vests sit on the midline, so it is centred (`LOAD_CENTRED`).

The load point is expressed in a trunk frame (z from `Low_Back_Position` to
`Neck_Position`, x to the right, y forward) and carried with the trunk through
the trial (`load_offset_x_m`, `_y_m`, `_z_m`). The **system CoM** is

$$\mathbf{x}_{CoM} = \frac{m_{body}\,\mathbf{x}_{CoM,body} + m_{load}\,\mathbf{x}_{load}}{m}.$$

It is used by every margin of stability, pendulum length, foot-placement model
and CoM work calculation. If no quiet standing is found, $m$ comes from the
walking mean GRF and the CoM stays the body's (both are reported).

*Synthetic check (80 kg + 20 kg carried 150 mm behind the trunk):* 19.9 kg,
152 mm behind, system CoM within 2 mm of the truth; Theia's body CoM was 82 mm
from it.

### 5.5 Ground reaction forces and the CoM velocity

**Forces into Theia's frame.** The export's forces and moments are in each
plate's frame. The plate's rotation from its fitted corners takes them to the
lab and the registration of §4.2 takes them to Theia. The sign is the one that
makes the vertical force hold the walker up (the export may give the force on
the plate). Each belt's baseline is removed (vertical: §4.1; horizontal: the
median over unloaded samples) and the forces used for peaks are low-pass
filtered at 50 Hz. Handrail channels, when present, flag contact above 15 N
(`handrail_steps_pct`).

**Checking the axes.** The total GRF divided by $m$ must equal the CoM's
acceleration. Both are band-passed (0.5–5 Hz) and correlated per axis
(`grf_vs_marker_acc_r_X/_Y/_Z`); a strongly negative horizontal axis is flipped
and reported, and anything below 0.7 means the force axes or the mass are
suspect.

**Fused velocity.** Differentiating a tracked CoM amplifies noise where the
within-stride dynamics live; integrating the GRF is clean there but drifts.
A complementary filter takes each where it is good, with the same 2nd-order
0.5 Hz Butterworth on both branches so the two sum to one:

$$\mathbf{v} = \mathrm{LP}\!\left(\tfrac{d}{dt}\mathbf{x}_{CoM}\right) + \int\!\frac{\mathbf{F}_{tot} - \overline{\mathbf{F}_{tot}}}{m}\,dt \;-\; \mathrm{LP}\!\left(\int\!\frac{\mathbf{F}_{tot} - \overline{\mathbf{F}_{tot}}}{m}\,dt\right)$$

The derivative is a Savitzky–Golay filter (window 11, order 3); the force is
anti-aliased at 40 Hz before being taken to 100 Hz and integrated by the
trapezoid rule. Frames where the CoM was not tracked are returned as missing
(widened by the differentiator's reach), not interpolated. On the synthetic
trial the fused velocity is 4–8 mm/s RMS from the truth against 25 mm/s for
markers alone. Velocity error reaches the extrapolated CoM divided by
$\omega_0$ (≈3.3 s⁻¹), so this matters against margins of a few centimetres.

---

## 6. Spatiotemporal metrics

*`gait/spatiotemporal.py`.* One value per heel strike (row), from sub-frame
event times (not rounded to the 10 ms frame grid).

| Metric | Definition | Column |
|---|---|---|
| Stance time | $t_{TO} - t_{HS}$ | `stance_s` |
| Swing time | $t_{HS,next} - t_{TO}$ | `swing_s` |
| Stride time | $t_{HS,next} - t_{HS}$ | `stride_s` |
| Step time | the step that ends at this heel strike: $t_{HS} - t_{HS,contra,prev}$ | `step_s` |
| Cadence | $60 / \text{step time}$ | `cadence_spm` |
| Double support | initial ($t_{HS} \to t_{TO,contra}$) + terminal ($t_{HS,contra} \to t_{TO}$), % of stride | `double_support_pct` |
| Single support | $t_{TO,contra} \to t_{HS,contra}$ (the other foot's swing), % of stride | `single_support_pct` |
| Stance % | stance / stride × 100 | `stance_pct` |
| Step length | this heel ahead of the other, along the direction of travel, at this heel strike | `step_length_m` |
| Step width | the same two heels, across the direction of travel | `step_width_m` |
| Stride length | distance travelled over the belt from one heel strike to the next | `stride_length_m` |
| Stride speed | stride length / stride time | `stride_speed_ms` |
| Walk ratio | step length / cadence (§12.3) | `walk_ratio` |

Supports and stance percentages are computed only when the events are in their
natural order ($t_{HS} < t_{TO,contra} < t_{HS,contra} < t_{TO} < t_{HS,next}$).
Heel positions are `<Side>_Heel_Position`, linearly interpolated to the
sub-frame event time.

**Stride length on a treadmill** is

$$L = v_{belt}\,T + \left(y_{HS,next} - y_{HS}\right)$$

— belt speed × stride time plus how much further forward the foot landed than
last time [Dingwell et al. 2010 ✔]. The sum of two heel-to-heel step lengths
does **not** equal it on a treadmill: at the other foot's heel strike this heel
has already begun to lift, which adds 30–50 mm to each step (≈100 mm per stride
in testing). Stride speed therefore averages the belt speed, and its stride-to-
stride fluctuation is the walker's speed error relative to the belt.

**Typical values** at 1.3 m/s: stance ≈ 60–62% of stride, double support
≈ 20–24%, step width 0.08–0.15 m. Load carriage typically lengthens stance and
double support and widens steps [Knapik et al. 2004 ~].

Per trial, each is summarised as mean, SD and CV with confidence intervals
(§12.1), for both feet together and each foot's mean separately.

---

## 7. Posture and joint motion

*`gait/kinematics.py`.* Per stride (heel strike to heel strike of the row's
foot), from the first (sagittal, Visual3D X–Y–Z Cardan) component of each
angle signal:

| Metric | Signal | Column |
|---|---|---|
| Trunk lean | mean of `Thorax_Seg_Angle` X (thorax against the lab) | `trunk_lean_mean_deg` |
| Trunk lean range | its range within the stride | `trunk_lean_rom_deg` |
| Pelvis tilt | mean of `Pelvis_Seg_Angle` X | `pelvis_tilt_mean_deg` |
| Hip, knee, ankle range of motion | range of `<Side>_<Joint>_Joint_Angle` X | `hip_rom_deg`, `knee_rom_deg`, `ankle_rom_deg` |

Signs are Visual3D's and constant across trials, so between-condition
comparisons hold whichever way forward lean counts. A load typically leans the
trunk forward and increases knee flexion in weight acceptance [Knapik et al.
2004 ~]. Time-normalised waveforms (101 points per stride) are saved in
`{trial}_waveforms.npz`.

---

## 8. Kinetics

*`gait/kinetics.py`.* Per stance, **only when both its heel strike and toe-off
were trusted GRF events of the same belt contact** (§4.2): one boot on that
belt, the other off it, the CoP under the boot. The event record names the
belt, so a clean crossover step is read from the belt it actually landed on.
Any other stance carries two feet's force, or none, and is left missing
(`stances_kinetics_pct` reports how many were usable).

The force is the GRF on the body in Theia's frame (§5.5), baseline removed and
filtered at 50 Hz, between the heel-strike and toe-off samples. AP is the
component along the direction of travel (positive = propulsive), vertical is
up.

| Metric | Definition | Column |
|---|---|---|
| F1 | maximum vertical force in the first half of stance (weight acceptance) | `f1_bw` |
| Trough | minimum vertical force between F1 and F2 | `trough_bw` |
| F2 | maximum vertical force in the second half (push-off) | `f2_bw` |
| Loading rate | least-squares slope of vertical force between 20% and 80% of F1 | `loading_rate_bw_per_s` |
| Peak braking / propulsive force | most negative / most positive AP force | `peak_braking_bw`, `peak_propulsive_bw` |
| Braking impulse | $\int \min(F_{AP}, 0)\,dt$ (negative) | `braking_impulse_bw_s` |
| Propulsive impulse | $\int \max(F_{AP}, 0)\,dt$ | `propulsive_impulse_bw_s` |
| Vertical impulse | $\int F_{V}\,dt$ | `vertical_impulse_bw_s` |
| CoP excursion | range of the CoP along / across the direction of travel (15 Hz, loaded samples) | `cop_ap_range_mm`, `cop_ml_range_mm` |
| Free moment | peak $\lvert T_z \rvert$, $T_z = M_z - (r_x F_y - r_y F_x)$ with $r$ the CoP relative to the plate's moment origin (15 Hz) | `free_moment_peak_nm` |

**Normalisation.** `_bw` divides by body weight ($m_{body} g$); `_tw` by the
total weight standing (body + load, $W$). A load raises the `_bw` values
roughly in proportion to its mass; the `_tw` values show whether the force
pattern changed beyond carrying more.

**Checks.** At steady speed braking and propulsion cancel over a stride, and
each limb's vertical impulse carries half the weight over a stride. On the
synthetic trial the net AP impulse is −0.0001 BW·s and each limb carries 50.2%
of the weight.

**Typical values** (unloaded, 1.3 m/s): F1 and F2 ≈ 1.1–1.3 BW, trough
≈ 0.6–0.8 BW, propulsive impulse ≈ 0.02–0.03 BW·s [Winter 2009 ~].

---

## 9. Centre-of-mass work (individual limbs method)

*`gait/kinetics.py`.* Each leg does work on the CoM at the rate

$$P_{leg}(t) = \mathbf{F}_{leg}(t) \cdot \mathbf{v}_{CoM}(t)$$

[Donelan, Kram & Kuo 2002 ~], with $\mathbf{v}_{CoM}$ the fused system-CoM
velocity **in the belt frame** (§5.2, §5.5), in which the stance foot is still
as in overground walking. Integrated over the phases of the stance:

| Phase | Interval | Work | Column |
|---|---|---|---|
| Collision | heel strike → other toe-off | negative part | `collision_j_kg` |
| Rebound | single support | positive part | `rebound_j_kg` |
| Preload | single support | negative part | `preload_j_kg` |
| Push-off | other heel strike → toe-off | positive part | `push_off_j_kg` |
| Positive / negative work | whole stance | | `com_work_pos_j_kg`, `com_work_neg_j_kg` |

In J/kg of system mass (J in `{trial}_steps.csv`). Trusted stances only. Over
a stride of steady walking the net work is zero (verified to 0.3%). Push-off
and collision work are the mechanical cost of the step-to-step transition,
which rises with load; heavier collisions for a given push-off indicate a less
efficient transition.

---

## 10. Margin of stability

*`gait/stability.py`.* [Hof, Gazendam & Sinke 2005 ✔]

$$x_{CoM}^{ext} = x_{CoM} + \frac{v_{CoM}}{\omega_0}, \qquad \omega_0 = \sqrt{\frac{g}{\ell}}, \qquad MoS = u_{max} - x_{CoM}^{ext}$$

| Symbol | Here |
|---|---|
| $x_{CoM}$ | the system CoM (§5.4), projected on the direction of interest |
| $v_{CoM}$ | the fused velocity (§5.5), **belt frame** (belt speed added along the direction of travel) |
| $\ell$ | distance from the system CoM to the stance ankle joint centre, per frame |
| $u_{max}$ | the stance boot's sole: its outermost point across the direction of travel (ML), its most anterior point along it (AP) |

"Outward" for ML is away from the other ankle at heel strike, so nothing
depends on which way Theia's axes point. ML is positive when the extrapolated
CoM is inside the lateral border of the boot.

| Metric | When | Column |
|---|---|---|
| ML margin at contact | heel strike, the new stance boot as the boundary | `mos_ml_contact` |
| ML margin, minimum | minimum over single support | `mos_ml_min` |
| AP margin at contact | heel strike | `mos_ap_contact` |
| AP margin, minimum | minimum over single support | `mos_ap_min` |

**Why single support for the minima.** Before the other foot's toe-off and
after its heel strike, the stance boot alone is not the base of support; a
minimum over the whole stance picks up late stance, when the extrapolated CoM
is far beyond a foot the body has already left.

**Why the boot.** The conventional boundary (ankle joint centre or a toe
marker) lies several centimetres inside the real edge of the foot. The
ankle-based ML margin is kept in `{trial}_steps.csv` (`mos_ml_contact_ankle`)
so the difference can be reported. Margins are also given per leg length in
the steps file (`*_per_leg`) for comparison between people.

**Reading it.** ML margin > 0: the extrapolated CoM is inside the base of
support. The AP margin in single support is normally negative — walking is
controlled falling forward — so a negative AP value is not instability. Larger
is not simply better: less stable walkers often adopt larger ML margins as
compensation, so read the margin alongside step width and foot placement
(§16.2). Typical ML margin at contact in healthy adults ≈ 0.03–0.10 m ~
(strongly dependent on the boundary definition; the boot edge gives larger
values than the ankle).

---

## 11. Minimum foot clearance, margin of instability, trip risk

*`gait/stability.py`.* Per swing (toe-off to the next heel strike of the same
foot, 2% trimmed at each end).

### Minimum foot clearance (MFC)

Clearance at each frame is the swing boot's lowest sole point above the belt
surface (§3, §5.3):

$$c(t) = \min_j\; h_j(t)$$

The MFC is a **local** minimum in mid-swing that satisfies the three criteria
of [Schulz 2017 ✔]: (a) lower than the two frames before and after, (b) the
foot's speed is in the fastest quarter of that swing, (c) the rear of the sole
is not lower (the minimum is at the front of the boot). If several qualify the
lowest is taken; if none do, the swing is a non-MTC cycle and its MFC is
missing, not replaced by the global minimum (`swings_without_mfc_pct`). The
vertex and whether it lies on the toe cap are recorded (`mfc_vertex`,
`mfc_on_toes` in the steps file). Column: `mfc_m`.

MFC below zero is pose error, counted as `mfc_below_belt_pct`. Healthy MFC
≈ 1–2 cm with SD ≈ 0.5 cm ~; for tripping the low tail matters more than the
mean, so the SD (`mfc_m_sd`) is reported with it.

### Margin of instability and trip risk integral [Schulz 2017 ✔]

$$MoI(t) = \max\!\left(x_{CoM}^{ext}(t) - u_{ant}(t),\; 0\right), \qquad TRI = \int_{t_1}^{t_2} \frac{MoI(t)}{\max(c(t),\,1\,\mathrm{mm})}\,dt$$

- $u_{ant}$ is the most anterior sole point of **either** boot: where the front
  of the base of support would be if the swing foot came down now.
- $x_{CoM}^{ext}$ is computed with the **stance** leg's pendulum (CoM to the
  other foot's ankle) and the belt-frame velocity.
- MoI and clearance are in mm, so the ratio is dimensionless and TRI is in s.
- $t_1, t_2$: peak acceleration and peak deceleration of the MFC point's
  resultant speed, which excludes lift-off and landing, when the foot is meant
  to be near the ground.
- Time is not normalised (as Schulz), so a longer swing gives a larger TRI.

Columns: `moi_peak_mm`, `tri_s`. Higher TRI = greater trip risk (the opposite
direction to MFC). The fixed belt speed removes gait speed, the main confound
Schulz identified.

---

## 12. Summary statistics, symmetry and walk ratio

*`gait/summary.py`, `gait/uncertainty.py`, `gait/variability.py`.*

### 12.1 Mean, SD and CV of the per-step metrics

Every per-step metric of §6–11 is summarised over the steady steps (§5.1):

| Column | What |
|---|---|
| `<metric>` | mean, both feet |
| `<metric>_sd` | standard deviation (n − 1), both feet |
| `<metric>_cv` | coefficient of variation, SD / \|mean\| × 100 |
| `<metric>_L`, `<metric>_R` | mean of the left / right foot's steps |

Each has a 95% confidence interval from the circular block bootstrap (§18.1)
in `all_trials_metrics_ci.csv`. SD and CV measure the **amount** of variability;
§13–17 measure its **structure**.

### 12.2 Symmetry angle

For a variable with left and right means $X_L, X_R$ [Zifchock et al. 2008 ~]:

$$SA = \frac{45^\circ - \arctan(X_L / X_R)}{90^\circ} \times 100\%$$

(−200% applied when the result exceeds 100%). 0% is perfect symmetry; positive
means the right is larger. Unlike the classic symmetry index it does not depend
on which limb is the reference and does not diverge as one value nears zero.
Reported for stance, swing, step length, MFC, ML margin, propulsive impulse, F1
and TRI (`symmetry_angle_<metric>`).

### 12.3 Walk ratio

Step length / cadence (m per step/min), nearly constant within a person across
speeds; ≈ 0.006 in healthy adults ~. With both speed and cadence prescribed by
this protocol it is largely set by the protocol, so it is reported but should
not be read as a free control variable (`walk_ratio`).

---

## 13. Long-range correlations — detrended fluctuation analysis

*`gait/variability.py`.* [Peng et al. 1994 ~; Hausdorff et al. 1996 ~]

**What it measures.** Whether a stride is statistically related to strides
hundreds of strides earlier: not how much gait varies but whether the
variation has memory. Shuffling a series leaves its SD unchanged and sends α to
0.5.

**Computation.** For a stride series $x_1 \dots x_N$ (one foot's consecutive
steady strides):

1. integrate: $Y(k) = \sum_{i=1}^{k}(x_i - \bar{x})$;
2. cut $Y$ into boxes of $n$ strides, forwards and backwards ($2\lfloor N/n
   \rfloor$ boxes, so no tail is lost);
3. remove a least-squares line from each box (DFA-1);
4. $F(n)$ = RMS of the residuals over all boxes;
5. α = slope of $\log F(n)$ against $\log n$ over 20 log-spaced box sizes.

| α | Meaning |
|---|---|
| < 0.5 | anti-persistent: deviations are corrected stride to stride (metronome, fixed speed) |
| ≈ 0.5 | uncorrelated |
| 0.5–1 | persistent; healthy stride time overground ≈ 0.75–0.9 ~ |
| > 1 | non-stationary; check for drift before interpreting |

**Parameters.** Box sizes $16 \le n \le N/9$ [Damouras et al. 2010 ✔] (the
common 4 to N/4 inflates α). α is computed only when the box range spans at
least an octave (N ≥ 288 strides) and the series is at least 95% complete. All
trials are cut to the same N (§18.2). ≥ 600 strides are recommended; at N = 486
the 95% interval is about ±0.17.

**Series.** Stride, stance and swing time, double support, step length and
width, stride length and speed, MFC, ML and AP margins at contact, ML minimum,
propulsive and braking impulse, F1, trunk lean — each foot separately
(`dfa_alpha_<series>_L/_R`). Timing series are metronome-paced (§1).

**Interval and checks** (in the long table's `note`): a parametric bootstrap
interval with the bias of DFA at this N (§18.1); the mean α of 20 shuffled
copies (must be ≈ 0.5); the Geweke–Porter-Hudak periodogram estimate as a
ballpark second opinion (α ≈ d + 0.5, about twice DFA's scatter); the R² of
the log–log fit.

---

## 14. Entropy

*`gait/variability.py`.*

### 14.1 Sample entropy of the stride series

[Richman & Moorman 2000 ~]. For the z-scored series, with $B$ the number of
pairs of length-$m$ templates within Chebyshev distance $r$ and $A$ the number
still within $r$ at length $m + 1$ (self-matches excluded):

$$\mathrm{SampEn}(m, r, N) = -\ln\frac{A}{B}$$

$m = 2$, $r = 0.2$ SD, on the same series and length as DFA
(`sample_entropy_<series>_L/_R`). Low = regular and predictable; high =
unpredictable. White noise has the highest sample entropy and no complexity, so
single-scale SampEn is not a complexity measure. Values depend on $m$, $r$ and
$N$ and are not comparable with other studies [Yentes et al. 2013 ✔; Yentes &
Raffalt 2021 ✔]; compare conditions analysed identically. The r sweep (0.10–
0.30 SD) is in the long table's note: if the ordering of conditions changes
with $r$, no single-$r$ value is reportable. No confidence interval: resampling
breaks the templates.

### 14.2 Multiscale entropy of the trunk acceleration

Refined composite multiscale entropy [Costa et al. 2002 ~; Wu et al. 2014 ~] of
the continuous trunk acceleration (§15), 300 s of steady walking, per axis.
Coarse-grain at scale $\tau$ by averaging non-overlapping windows of $\tau$
samples, for every one of the $\tau$ starting phases; sum the match counts over
the phases before taking the logarithm; keep $r$ fixed on the original series'
SD. Scales 1–30 (0.01–0.3 s at 100 Hz, up to a quarter stride). The **area**
under the curve is the complexity measure (`mse_area_AP/_VT/_ML`): white noise
starts high and falls away; structured signals hold their entropy across
scales. The full curves, with a white-noise reference, are in the waveform
file and the variability figure.

---

## 15. Trunk acceleration: harmonic ratio and regularity

*`gait/variability.py`.* The trunk signal is `Trunk_Linear_Velocity` (the
first linear trunk signal in the export; the `*_Joint_Acc` signals are angular,
deg/s², and are **not** used), low-pass filtered at 20 Hz, rotated into (ML,
AP, vertical) about the belt's direction of travel. The time-domain
acceleration is its Savitzky–Golay derivative (window 5, order 3, accurate to
18 Hz); gaps stay missing. RMS per axis: `trunk_acc_rms_*`.

### 15.1 Harmonic ratio and improved harmonic ratio

Per stride of the right foot (heel strike to heel strike), the DFT of exactly
one stride, so bin $k$ is the $k$-th harmonic of stride frequency (no window).
From a velocity, the acceleration's harmonic $k$ is exactly $k\omega$ times the
velocity's, so amplitudes are weighted by $k$ — exact, where numerical
differentiation rolls off before the 20th harmonic ($\omega$ cancels):

$$HR_{AP,V} = \frac{\sum_{k\ even} A_k}{\sum_{k\ odd} A_k}, \qquad HR_{ML} = \frac{\sum_{k\ odd} A_k}{\sum_{k\ even} A_k}, \qquad k = 1 \dots 20$$

AP and vertical repeat twice per stride (even harmonics intrinsic); ML once
(odd). The improved harmonic ratio [Pasciuto et al. 2017 ✔] is the intrinsic
harmonics' share of the total power, bounded 0–100%:

$$iHR = 100\,\frac{\sum_{k\ intrinsic} A_k^2}{\sum_{k} A_k^2}$$

HR is unbounded, so its median over strides is reported (`harmonic_ratio_hr_*`);
iHR is bounded, so its mean with a CI (`harmonic_ratio_ihr_*`). Higher = smoother,
more rhythmic trunk motion. HR is **not** a symmetry measure: a symmetric but
jerky gait scores low. Healthy HR ≈ 2–4 ~, strongly speed and site dependent.

### 15.2 Step and stride regularity

The unbiased autocorrelation of the steady trunk acceleration (divided by the
overlap at each lag, so a longer lag is not penalised)
[Moe-Nilssen & Helbostad 2004 ✔]:

- step regularity $A_{d1}$ = the autocorrelation **peak** within ±25% of half
  the median stride (`step_regularity_*`);
- stride regularity $A_{d2}$ = the peak within ±25% of the median stride
  (`stride_regularity_*`);
- symmetry = $A_{d1} / A_{d2}$ (`symmetry_*`).

Peaks rather than fixed lags: 6% of stride-time drift makes a perfectly regular
signal read 0.94 at a fixed lag. Values near 1: each step (stride) repeats the
last; symmetry near 1: the two steps are alike.

---

## 16. Control of stepping

*`gait/variability.py`, `gait/summary.py`.*

### 16.1 Goal-equivalent manifold

[Dingwell, John & Cusumano 2010 ✔]. At a fixed belt speed the task goal is
unambiguous: keep $L/T = v^*$. Stride time and length (§6) are divided by their
means, so both are dimensionless and the goal line is the 45° diagonal (rotating
seconds and metres together would make the split depend on the units). Each
stride's deviation is split into a component **along** the line
(goal-equivalent: speed unchanged) and **across** it (goal-relevant: speed
error):

$$\delta_\parallel = \frac{\hat{T} + \hat{L}}{\sqrt{2}}, \qquad \delta_\perp = \frac{\hat{L} - \hat{T}}{\sqrt{2}}, \qquad \hat{T} = \frac{T}{\bar{T}} - 1,\ \hat{L} = \frac{L}{\bar{L}} - 1$$

Per foot: the SD of each (% of the mean stride, with CI;
`gem_parallel_sd_pct`, `gem_perpendicular_sd_pct`), its lag-1 autocorrelation
(`gem_*_lag1`) and its DFA α with CI (`gem_*_dfa_alpha`). Healthy treadmill
walking corrects the goal-relevant component much more strongly (lag-1 and α
well below those of the goal-equivalent one): the controller exploits the
redundancy instead of fighting all variability.

### 16.2 Mediolateral foot placement

[Wang & Srinivasan 2014 ✔]. How much of where the foot lands is explained by
the body's state:

$$y_{foot} = \beta_0 + \beta_1 y_{CoM} + \beta_2 \dot{y}_{CoM} + \varepsilon$$

at this foot's mid-stance: $y_{CoM}$ the system CoM and $\dot{y}_{CoM}$ its
fused velocity across the direction of travel, both relative to the stance
ankle; $y_{foot}$ where the other heel lands at its next heel strike, relative
to the same ankle (so a slow drift across the belt cannot inflate the fit).
Least squares per foot; the $R^2$ is the measure (`foot_placement_r2_L/_R`),
with a block bootstrap CI; the gains are in the note. Healthy ML placement:
$R^2$ > 0.8 at the next step ~. A lower $R^2$ means placement is less tightly
driven by the body's state.

---

## 17. Local dynamic stability

*`gait/lds.py`* (as Bruijn's LocalDynamicStability toolbox,
`makestatelocal.m` + `lds_calc.m`) [Rosenstein et al. 1993 ~; Bruijn et al.
2013 ~].

1. Take 150 consecutive steady strides of the right foot and time-normalise the
   whole block to 100 samples per stride (cubic spline).
2. Delay-embed: $\tau$ = 10 samples, $d_E$ = 5 (per signal; state spaces
   below).
3. For every point, the nearest neighbour more than half a stride away in time;
   follow both for 10 strides; average $\ln$ distance over all pairs.
4. $\lambda_S$ = slope over 0–0.5 stride, $\lambda_L$ = slope over 4–10
   strides (per stride).

State spaces: trunk linear velocity ML, AP and vertical (each $d_E$ = 5), and
the three together ($d_E$ = 2); AP is the primary (`lds_lambda_S_trunkVel_AP`).
Larger λ = faster divergence of nearby trajectories = less locally stable.

**Windows.** λ depends on the number of strides, so every estimate uses 150. A
long trial holds several windows (up to 4); each gets its own λ and the trial
value is their mean (details in `{trial}_lds_windows.csv`).

**Interval.** The divergence curve is kept per reference stride, so blocks of
strides can be resampled (block length by Politis–White on the per-stride
slopes) and the curve refitted: the interval says how much λ depends on which
strides happened to be walked. It does not include the estimator's own bias:
on systems with a known exponent Rosenstein's method is 3–15% off
(`lds_validation.py`), and a perfectly periodic signal gives $\lambda_S$ =
0.2–0.45 per stride from the neighbour-selection transient alone. Compare
conditions analysed identically; λ is an operational measure of short-term
divergence, not a converged exponent. $\tau$ and $d_E$ must be the same for
every trial compared (`lds_validation.py` Part 2 estimates them for the
dataset).

---

## 18. Uncertainty and comparability

*`gait/uncertainty.py`.*

### 18.1 Confidence intervals

Every value comes from one walk of a few hundred strides; walk again and it
changes. The 95% confidence intervals (`all_trials_metrics_ci.csv`, the 95% CI
sheet) say by how much.

**Circular block bootstrap** — means, SDs, CVs, GEM SDs, foot-placement $R^2$,
iHR [Künsch 1989 ~; Politis & Romano 1992 ~]. Strides are not independent: a
long stride tends to be followed by another. Resampling single strides would
destroy that and make intervals too narrow. So consecutive strides are drawn in
runs — **blocks** — starting anywhere and wrapping round the end, glued into a
new series of the same length, 1000 times; the 2.5th and 97.5th percentiles of
the statistic are the interval.

The **block length** is how many consecutive strides make up each run. Too
short loses the correlation; too long leaves few distinct blocks. It is chosen
for each series by the rule of Politis & White (2004 ~), corrected by Patton,
Politis & White (2009 ~): estimated from the series' own autocorrelation, so an
uncorrelated series gets 1 (the ordinary bootstrap) and a persistent one a
longer block. It is the `block` column of the long table. Validation: on a
persistent AR(1) series (φ = 0.6), blocks give 93% coverage of a 95% interval;
single-stride resampling gives 66%.

**Parametric bootstrap for DFA.** Blocks would cut the long-range correlation
α measures, so instead 200 series of exact fractional Gaussian noise (Davies &
Harte 1987 ~) are simulated with the estimated α (its running sum for α > 1),
and DFA is rerun on each. The **basic** bootstrap interval
$[2\hat\alpha - q_{97.5},\ 2\hat\alpha - q_{2.5}]$ corrects for DFA's bias at
this N; where that bias exceeds its scatter (α near 0, or few strides) the
interval is centred on $\hat\alpha$ − bias and may exclude $\hat\alpha$ itself.
The bias is in the note. Coverage at N = 486: 93%.

**Stride bootstrap for λ** — §17.

Sample entropy and the regularity and MSE values have no interval.

### 18.2 Comparability

- **Warm-up**: the first 60 s after the first event are excluded from every
  summary.
- **Series length**: DFA, entropy, GEM and foot placement are N-dependent, so
  every trial in a run is cut to the same number of strides — by default the
  shortest trial's (`SERIES_LENGTH = "shortest"`), printed at the start and
  recorded in the `series_length` column of the long table. Re-running a subset
  of trials can change N; set an integer to fix it for the whole study.
- **Completeness**: a stride series less than 95% complete is not used for
  DFA or entropy (both would still return a number).
- **LDS**: always 150 strides per window.

---

## 19. The results tables

*`gait/tables.py`.* After every run, the tables are rebuilt from every trial in
the output folder (earlier runs included; `python run_gait_analysis.py --tables`
rebuilds them without re-analysing).

| File | Content |
|---|---|
| `all_trials_metrics.xlsx` | **Key metrics** sheet; one sheet per family (Trial, Spatiotemporal, Stability, Kinetics, CoM work, Posture, Symmetry, DFA, Entropy, Trunk, Control, LDS); the **95% CI** sheet; the **Dictionary**. Row 1 of every sheet is a readable description with the unit, row 2 the column name, one row per trial below, participant/condition/trial frozen on the left. |
| `all_trials_metrics.csv` | the same values, one row per trial |
| `all_trials_metrics_ci.csv` | `<column>_lo` and `<column>_hi` for every column with an interval |
| `metric_dictionary.csv` | every column: family, description, unit, limb, statistic, section of this document |
| `all_trials_summary.csv` | the long form: one row per number with its CI, N, block length and note — for mixed models |

**Column names.** `step_width_m` is the mean of all steady steps; `_sd`, `_cv`
the SD and CV; `_L`, `_R` each foot's mean or, for DFA, entropy, GEM and foot
placement, the value from that foot's own series. The unit ends the name:
`_s`, `_m`, `_mm`, `_pct`, `_deg`, `_bw` (body weight), `_tw` (total weight),
`_j_kg`, `_ms` (m/s), `_spm` (steps/min).

**Per-trial files** (for checking and re-analysis): `{trial}_steps.csv` (every
per-step value), `{trial}_summary.csv`, `{trial}_waveforms.npz`,
`{trial}_lds_windows.csv`, and two figures: `{trial}_qa.png` (quiet standing
and load, event sources, clearance, margins, GRF, velocity fusion) and
`{trial}_variability.png` (DFA, α with CIs, LDS curves, MSE curves).

**Group level.** Trials are nested in participants, so compare conditions with
a mixed model, participant as a random effect and condition (load) as fixed,
on the long table; the confidence intervals above describe within-trial
uncertainty and can be used as precision weights. Example (README): statsmodels
`mixedlm("value ~ condition", groups=participant)`.

---

## 20. Validation

Before a number is used to compare conditions, what it does on signals with a
known answer is established.

| Script | What | Result |
|---|---|---|
| `test_detect_gait_events.py` | synthetic trial: crossovers, straddles, an overhanging step, a slow-unloading tail, a tracking dropout, a quiet-standing start, pitched plates, a rotated lab frame | every true event found once; GRF events within 0.2 frames; Zeni within 0.33 frames; no event from a crossed or straddled step |
| `test_gait_analysis.py` | synthetic trial with a known load (20 kg, 150 mm behind the trunk), CoM, belt, clearance and forces, through the whole pipeline | load 19.9 kg at 152 mm; CoM 2 mm; velocity 4–8 mm/s; stance −0.1 ± 0.4 ms; step width 1 mm; MFC −1 mm; ML margin 0.8 mm from the true-CoM margin; impulses and CoM work balance; tables complete |
| `gait_metrics_validation.py` | each estimator on constructed signals | fusion unbiased and better than markers; MoS, MoI, TRI exact; DFA bias within 0.02, shuffle → 0.5, interval coverage 93%; block bootstrap coverage 93% (single strides 66%); SampEn matches its closed form for white noise; MSE separates white from 1/f noise; HR exact and k-weighting exact; regularity peaks, GEM, foot placement, symmetry angle exact |
| `lds_validation.py` | Rosenstein on Lorenz and Rössler; a periodic signal; AMI/FNN for τ and $d_E$ | −3% and −15% of the known exponents; λ_L ≈ 0 for a periodic signal; the λ_S noise floor |

---

## Appendix A — parameters

| Setting | Value | Where | Section |
|---|---|---|---|
| Kinematic / force rate | 100 / 1000 Hz | `detect_gait_events.py` | 2 |
| Force filter, contact threshold | 50 Hz, 20 N | `detect_gait_events.py` | 4.1 |
| Edge search / hold | ±15 ms / 5 samples | `detect_gait_events.py` | 4.1 |
| Merge gap, minimum contact | 30 ms, 80 ms | `detect_gait_events.py` | 4.1 |
| Loading / unloading shape | 200 N within 60 / 80 ms | `detect_gait_events.py` | 4.2 |
| Trust window | ±50 ms | `detect_gait_events.py` | 4.2 |
| Touching / on a belt | 15 mm; 3 sole points; outline + 8 mm | `detect_gait_events.py` | 4.2 |
| CoP under the boot | within 30 mm for 90% | `detect_gait_events.py` | 4.2 |
| Zeni filter, prominence | 10 Hz; 50 mm within ±0.6 s | `detect_gait_events.py` | 4.3 |
| Toe hinge | on, sign +1, reference 0° | `detect_gait_events.py` | 3 |
| Sole cells | 10 mm, up to 35 mm rise | `detect_gait_events.py` | 3 |
| Warm-up | 60 s | `gait/trial.py` `WARMUP_S` | 5.1 |
| Foot-flat window for the belt | 25–55% of stance | `gait/trial.py` | 5.2 |
| Quiet standing | 2 s window in the first 30 s, CV < 2%, each belt ≥ 15%, heels ≤ 20 mm | `gait/trial.py` | 5.4 |
| Load | height +0 m on Trunk_Position, centred, ≤ 0.40 m from the trunk, ≥ 1 kg | `gait/trial.py` | 5.4 |
| Fusion | 0.5 Hz crossover; Savitzky–Golay 11/3; anti-alias 40 Hz | `gait/trial.py` | 5.5 |
| Handrail contact | 15 N | `gait/trial.py` | 5.5 |
| Loading rate band | 20–80% of F1 | `gait/kinetics.py` | 8 |
| CoP / free moment filter | 15 Hz | `gait/kinetics.py` | 8 |
| MFC | local minimum ±2 frames, fastest 25% of swing, 2% trim, 1 mm floor for TRI | `gait/stability.py` | 11 |
| DFA | DFA-1, boxes 16 to N/9, 20 sizes, 20 shuffles, GPH m = N^0.5 | `gait/variability.py` | 13 |
| Sample entropy | m = 2, r = 0.2 SD, sweep 0.10–0.30 | `gait/variability.py` | 14.1 |
| MSE | scales 1–30, 300 s | `gait/variability.py` | 14.2 |
| Trunk signal | 20 Hz low-pass, Savitzky–Golay 5/3 | `gait/variability.py` | 15 |
| Harmonic ratio | 20 harmonics, ≥ 40 samples per stride | `gait/variability.py` | 15.1 |
| Regularity peak search | ±25% of the expected lag | `gait/variability.py` | 15.2 |
| LDS | 150 strides, 100 samples/stride, τ 10, $d_E$ 5, 10 strides followed, fits 0–0.5 and 4–10, ≤ 4 windows, 500 resamples | `gait/lds.py` | 17 |
| Block bootstrap | 1000 resamples, 95%, Politis–White block | `gait/uncertainty.py` | 18.1 |
| DFA interval | 200 simulated series | `gait/uncertainty.py` | 18.1 |
| Series length | shortest trial in the run | `run_gait_analysis.py` `SERIES_LENGTH` | 18.2 |
| Limb for trunk measures and LDS | right | `run_gait_analysis.py` `LIMB_FOR_TRUNK` | 15, 17 |

---

## Appendix B — references

- ~ Bruijn SM, Meijer OG, Beek PJ, van Dieën JH (2013). Assessing the stability of human locomotion: a review of current measures. *J R Soc Interface* 10:20120999.
- ~ Costa M, Goldberger AL, Peng CK (2002). Multiscale entropy analysis of complex physiologic time series. *Phys Rev Lett* 89:068102.
- ✔ Damouras S, Chang MD, Sejdić E, Chau T (2010). An empirical examination of detrended fluctuation analysis for gait data. *Gait Posture* 31(3):336–340.
- ~ Davies RB, Harte DS (1987). Tests for Hurst effect. *Biometrika* 74(1):95–101.
- ✔ Dingwell JB, John J, Cusumano JP (2010). Do humans optimally exploit redundancy to control step variability in walking? *PLoS Comput Biol* 6(7):e1000856.
- ~ Donelan JM, Kram R, Kuo AD (2002). Simultaneous positive and negative external mechanical work in human walking. *J Biomech* 35(1):117–124.
- ~ Geweke J, Porter-Hudak S (1983). The estimation and application of long memory time series models. *J Time Ser Anal* 4(4):221–238.
- ~ Hausdorff JM, Purdon PL, Peng CK, Ladin Z, Wei JY, Goldberger AL (1996). Fractal dynamics of human gait: stability of long-range correlations in stride interval fluctuations. *J Appl Physiol* 80(5):1448–1457.
- ✔ Hof AL, Gazendam MGJ, Sinke WE (2005). The condition for dynamic stability. *J Biomech* 38:1–8.
- ~ Knapik JJ, Reynolds KL, Harman E (2004). Soldier load carriage: historical, physiological, biomechanical, and medical aspects. *Mil Med* 169(1):45–56.
- ~ Künsch HR (1989). The jackknife and the bootstrap for general stationary observations. *Ann Stat* 17(3):1217–1241.
- ~ Menz HB, Lord SR, Fitzpatrick RC (2003). Acceleration patterns of the head and pelvis when walking on level and irregular surfaces. *Gait Posture* 18(1):35–46.
- ✔ Moe-Nilssen R, Helbostad JL (2004). Estimation of gait cycle characteristics by trunk accelerometry. *J Biomech* 37:121–126.
- ✔ Pasciuto I, Bergamini E, Iosa M, Vannozzi G, Cappozzo A (2017). Overcoming the limitations of the Harmonic Ratio for the reliable assessment of gait symmetry. *J Biomech* 53:84–89.
- ~ Patton A, Politis DN, White H (2009). Correction to "Automatic block-length selection for the dependent bootstrap". *Econometric Rev* 28(4):372–375.
- ~ Peng CK, Buldyrev SV, Havlin S, Simons M, Stanley HE, Goldberger AL (1994). Mosaic organization of DNA nucleotides. *Phys Rev E* 49:1685–1689.
- ~ Politis DN, Romano JP (1992). A circular block-resampling procedure for stationary data. In *Exploring the Limits of Bootstrap*, Wiley, 263–270.
- ~ Politis DN, White H (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Rev* 23(1):53–70.
- ✔ Ravi DK, Marmelat V, Taylor WR, Newell KM, Stergiou N, Singh NB (2020). Assessing the temporal organization of walking variability: a systematic review and consensus guidelines on detrended fluctuation analysis. *Front Physiol* 11:562.
- ~ Richman JS, Moorman JR (2000). Physiological time-series analysis using approximate entropy and sample entropy. *Am J Physiol Heart Circ Physiol* 278(6):H2039–H2049.
- ~ Rosenstein MT, Collins JJ, De Luca CJ (1993). A practical method for calculating largest Lyapunov exponents from small data sets. *Physica D* 65:117–134.
- ✔ Schulz BW (2017). A new measure of trip risk integrating minimum foot clearance and dynamic stability across the swing phase of gait. *J Biomech* 55:107–112.
- ✔ Wang Y, Srinivasan M (2014). Stepping in the direction of the fall: the next foot placement can be predicted from current upper body state in steady-state walking. *Biol Lett* 10(9):20140405.
- ~ Winter DA (2009). *Biomechanics and Motor Control of Human Movement*, 4th ed. Wiley.
- ~ Wu SD, Wu CW, Lin SG, Lee KY, Peng CK (2014). Analysis of complex time series using refined composite multiscale entropy. *Phys Lett A* 378(20):1369–1374.
- ✔ Yentes JM, Hunt N, Schmid KK, Kaipust JP, McGrath D, Stergiou N (2013). The appropriate use of approximate entropy and sample entropy with short data sets. *Ann Biomed Eng* 41(2):349–365.
- ✔ Yentes JM, Raffalt PC (2021). Entropy analysis in gait research: methodological considerations and recommendations. *Ann Biomed Eng* 49(3):979–990.
- ~ Zeni JA, Richards JG, Higginson JS (2008). Two simple methods for determining gait events during treadmill and overground walking using kinematic data. *Gait Posture* 27(4):710–714.
- ~ Zifchock RA, Davis I, Higginson J, Royer T (2008). The symmetry angle: a novel, robust method of quantifying asymmetry. *Gait Posture* 27(4):622–627.
