# 5. Preparing a trial

Before any metric is computed, `load_trial()` assembles everything one trial needs, in this order: the steps (§5.1), the belt's speed and direction (§5.2), the boots (§5.3), the forces (§5.4), the weight of body + load (§5.5), the system centre of mass (§5.6) and its velocity (§5.7). Each step prints a line, so a problem shows up where it starts.

## 5.1 Events → steps

**What.** One row per heel strike, holding every event that step needs.

For a heel strike of foot $b$ at $t_{HS}$, with $o$ the other foot, the row holds (as fractional frames, 0 = first frame):

| Column | Event | Used for |
|:--|:--|:--|
| `hs` | this heel strike | start of stance and stride |
| `to` | this foot's next toe-off | end of stance |
| `next_hs` | this foot's next heel strike | end of the stride |
| `contra_to` | the other foot's first toe-off after `hs` | end of initial double support |
| `contra_hs` | the other foot's first heel strike after `hs` | start of terminal double support |
| `prev_contra_hs` | the other foot's last heel strike before `hs` | start of the step that ends at `hs` |

Each row also records where each event came from (force plate, Zeni or interpolated) and, for force events, which belt carried it. A toe-off that comes after the next heel strike means a missed event, and it is set to missing rather than used.

![Figure 7. The timing of one stride of the left foot. A stride runs from one left heel strike to the next. Stance (left foot down) splits into initial double support (both feet down, until the right toe-off), single support (the right foot swinging), and terminal double support (from the right heel strike to the left toe-off). Swing follows.](figures/fig07.png){width=6.2in}

**Steps** (`read_events()`, `build_steps()`):

1. Read `{trial}_event_qa.csv`. Its `frame_float` column gives each event to a fraction of a frame. If only `merged_events.csv` exists, whole frames are used, the kinetics cannot know which belt carried a step, and the script warns about both.
2. For each foot, look up each of the five other events with a sorted search: the first event of the right kind after (or before) this heel strike.
3. Mark as **steady** every step whose heel strike is at least **60 s** after the first event (`WARMUP_S`).

::: {custom-style="Why Box"}
**Why fractional frames** — at 100 Hz a frame is 10 ms, which is 1.5% of a stance. Rounding each event to the nearest frame adds up to ±5 ms to each, which is larger than many effects of load. Force events are known to 1 ms and Zeni events to a fraction of a frame, and positions are read at those exact times by linear interpolation (`at()`). **Why build the row once** — every per-step metric is a difference between these events. Building and checking the row once means every metric uses the same, validated set of events. **Why 60 s of warm-up** — the belt ramps up, and people take tens of seconds to settle into a steady treadmill gait, especially with a metronome and a load.
:::

::: {custom-style="Code Box"}
**In the code** — §5.1: `read_events()`, `build_steps()`. The warm-up is applied in `load_trial()`. Settings: `WARMUP_S`.
:::

## 5.2 The belt's speed and direction of travel

**What.** The velocity of the belt, from the feet standing on it.

**Equations.** For each stance, take the frames from 25% to 55% of stance, $t\in[t_{HS}+0.25\,T_{st},\ t_{HS}+0.55\,T_{st}]$, and fit a straight line to the heel's horizontal position against time. Its slope is the heel's velocity $\mathbf{u}_i$. Then

$$\hat{f} = -\frac{\operatorname{median}_i\left(\mathbf{u}_i/\lVert\mathbf{u}_i\rVert\right)}{\left\lVert\operatorname{median}_i\left(\mathbf{u}_i/\lVert\mathbf{u}_i\rVert\right)\right\rVert},\qquad v_b = \operatorname{median}_i\left(-\mathbf{u}_i\cdot\hat{f}\right),\qquad \hat{\ell} = (-f_y,\ f_x).$$

The minus sign is there because a foot on the belt moves **backwards**. $\hat{\ell}$ points to the walker's left.

![Figure 8. The belt speed from the heel. Grey: the left heel's position along the direction of travel. In each stance, a straight line is fitted over 25–55% of stance (shaded; thick blue). There the heel is flat on the belt and moves with it, and the dashed line extends the fit over the whole stance. The belt speed is the median slope over every stance of the trial.](figures/fig08.png){width=6.5in}

::: {custom-style="Why Box"}
**Why measure the belt at all** — the nominal speed is a setting on the treadmill's controller; the belt's real speed under load can differ from it. Three things need the real speed: the belt frame (§1.3), the stride length over the belt (§6), and the GEM goal line (§17). **Why 25–55% of stance** — before 25% the heel is still rolling onto the belt, and after 55% it starts to lift. In between, the heel is fixed on the belt and moves exactly with it. **Why the median** — a few stances are disturbed (a stumble, a tracking error). The median ignores them; a mean would not. **Why the direction from the feet** — see §1.3.
:::

::: {custom-style="Code Box"}
**In the code** — §5.2: `belt_motion()`. Settings: `BELT_FLAT = (0.25, 0.55)`. The result is `belt_speed_ms` in the results.
:::

## 5.3 The boots over the belt

The boots of §3 are posed on Theia's feet in every frame. For every frame and boot, `Boots` keeps the height and index of the lowest sole point, and how far the boot reaches forward, backward, left and right along the walker's axes. The margin of stability (§10) uses those extents, and foot clearance (§11) uses the heights. The belt surface is the one fitted by the event detection and saved in its summary, so both steps measure against the same surface.

::: {custom-style="Code Box"}
**In the code** — §5.3: class `Boots` (`height()`, `world()`), which uses `dge.boot_sole()`, `dge.toe_hinge()`, `dge.fit_belt_surface()` and `dge.above_belt()`.
:::

## 5.4 Forces into the motion-capture frame

**What.** Each belt's force, CoP and free moment, with the unloaded baseline removed and expressed in Theia's frame. The forces are combined with Theia's CoM (§5.7, §9) and with the boots (§8), so they must be in the same frame and point the same way.

**Steps** (`read_forces()`):

1. **Baseline.** Vertical: the drift-tracking baseline of §4.1. Horizontal: the median over the unloaded samples.
2. **Plate → lab → Theia.** Rotate by the plate's fitted rotation $\mathbf{R}_p$ (§4.2), then rotate the horizontal part of $\mathbf{R}_p\mathbf{F}$ by the lab → Theia rotation $\mathbf{R}(\varphi)$ (§4.2). The vertical part is unchanged, because both frames share the vertical.
3. **Sign.** The export does not document whether it gives the force on the plate or on the body. The sign is chosen so that the vertical force **holds the walker up** (positive when loaded). §5.7 then checks the horizontal axes against the markers.
4. **Two copies.** One is low-passed at 50 Hz (`FORCE_FILTER_HZ`), for peaks (§8). The other is kept raw, for integration (§5.7, §9), where filtering is unnecessary and would only blur the timing.
5. **CoP into Theia.** The DICE export gives the CoP in each plate's own frame. It is moved into the lab with $\mathbf{p} = \mathbf{R}_p\,(\text{CoP}_x,\ \text{CoP}_y,\ 0) + \mathbf{t}_p$, then into Theia. It is kept only where the belt carries more than 150 N, because at low force the CoP is mostly noise.
6. **Free moment.** This is the vertical torque between the foot and the belt that is *not* explained by the horizontal forces acting at the CoP:
$$T_z = M_z - \left(r_x F_y - r_y F_x\right),\qquad \mathbf{r} = \text{CoP} - \mathbf{o},$$
where $\mathbf{o}$ is the plate's moment origin. Because the export does not give $\mathbf{o}$, it is recovered from the data: $o_x = \operatorname{median}(\text{CoP}_x + M_y/F_z)$ and $o_y = \operatorname{median}(\text{CoP}_y - M_x/F_z)$.
7. **Handrails.** Any handrail force more than **15 N** from its median counts as a hand on the rail (`HANDRAIL_N`). The GRF then misses an external force. Steps with handrail contact are flagged, and their share is reported (`handrail_steps_pct`).

::: {custom-style="Why Box"}
**Why recover the moment origin** — the free moment is small (a few N·m) compared with the moments of the vertical force about the plate centre (hundreds of N·m). An origin that is wrong by 1 cm gives an error comparable to the free moment itself. The origin is exactly the point about which $\text{CoP} = (-M_y/F_z, M_x/F_z)$ holds, so it can be read back from the exported CoP and moments.
:::

::: {custom-style="Code Box"}
**In the code** — §5.4: `read_forces()`. Settings: `FORCE_FILTER_HZ`, `HANDRAIL_N`. The plate geometry and registration come from `detect_gait_events.py` (`fit_plate`, the saved `lab_to_theia`).
:::

## 5.5 Quiet standing: weighing body + load

**What.** The weight of everything the walker carries, measured by the force plates while the walker stands still at the start of the trial.

**Equations.** Standing still, nothing accelerates, so the plates carry exactly the weight of body + load (Newton's second law with $\mathbf{a}=0$):

$$W = \overline{F_{z,L} + F_{z,R}},\qquad m = \frac{W}{g},\qquad m_\ell = m - m_b.$$

**Steps** (`quiet_standing()`):

1. Search from the start of the recording to the first gait event (at most the first **30 s**, `QS_SEARCH_S`).
2. Slide a **2 s** window (`QS_WINDOW_S`) in steps of 0.1 s. A window qualifies only if:
    * the total vertical force is steady: $\operatorname{sd}/\text{mean} < 2\%$ (`QS_MAX_CV`);
    * both feet are loaded: each belt carries 15–85% of the total (`QS_MIN_SHARE`);
    * both heels are still: each moved less than **20 mm** between the medians of the first and last 0.2 s of the window (`QS_MAX_FOOT_TRAVEL`).
3. Keep the qualifying window with the smallest CV. $W$ is its mean total vertical force.
4. **Check:** over the steady walk the body does not accelerate on average, so the mean vertical GRF while walking must also equal $W$. More than **2%** apart means the plates drifted or the standing was not still, and the script warns.
5. $m_\ell = m - m_b$, with $m_b$ from `participants.csv`. A load below −1 kg means the entered body mass or the plate zero is wrong, and the script warns.
6. Keep the force-weighted CoP of both belts during the window, for §5.6.

If no window qualifies, the system mass falls back to the mean vertical GRF while walking divided by $g$. The load then cannot be placed (§5.6), and the note in the results says so.

![Figure 10. Weighing in the quiet standing. Blue and orange: the vertical force on each belt; black: their sum. The shaded 2 s is the steadiest window with both feet loaded and the heels still; its mean is the weight of body + load. In this synthetic trial, 981 N gives 100.0 kg, and with an 80 kg body mass the load is 20.0 kg.](figures/fig10.png){width=6.5in}

::: {custom-style="Why Box"}
**Why weigh at all** — the load differs between trials and conditions, and Theia cannot see it: its centre of mass is the body's alone. Mass, total weight and the load's position all enter the kinetics, the CoM and the margin of stability. **Why standing** — only when nothing accelerates does the force equal the weight exactly. While walking, the vertical force swings between about 0.7 and 1.3 body weights within each step. **Why the three conditions** — a window with one foot lifting, a shuffle or a lean would weigh correctly but place the load wrongly (§5.6). Asking for both feet down, still heels and a steady force picks true quiet standing.
:::

::: {custom-style="Code Box"}
**In the code** — §5.5: `quiet_standing()`. Settings: `QS_SEARCH_S`, `QS_WINDOW_S`, `QS_MAX_CV`, `QS_MIN_SHARE`, `QS_MAX_FOOT_TRAVEL`. Results: `system_mass_kg`, `load_kg`, `quiet_standing_cv`.
:::

## 5.6 The system (body + load) centre of mass

**What.** The centre of mass of everything the walker moves: body and load.

**Equations.** Standing still, the CoP lies vertically below the system centre of mass. Horizontally, therefore,

$$m\,\mathbf{x}_{CoP} = m_b\,\mathbf{x}_{b} + m_\ell\,\mathbf{x}_{\ell}\quad\Rightarrow\quad \mathbf{x}_{\ell,xy} = \frac{m\,\mathbf{x}_{CoP} - m_b\,\mathbf{x}_{b,xy}}{m_\ell},$$

with $\mathbf{x}_b$ Theia's body CoM averaged over the quiet standing. Standing still gives no information about the load's **height**. It is placed at the height of `Trunk_Position` (plus `LOAD_HEIGHT_M`, 0 by default) and on the body's mid-line left-right (`LOAD_CENTRED`).

The position is then stored in a **trunk frame**:

* $\hat{z}$ points from `Low_Back_Position` to `Neck_Position`;
* $\hat{x}$ points to the right, made perpendicular to $\hat{z}$;
* $\hat{y} = \hat{z}\times\hat{x}$ points forward.

The stored offset is $\mathbf{d} = \mathbf{R}_{trunk}^T(\mathbf{x}_\ell - \mathbf{x}_{trunk})$, taken as the median over the standing. It moves with the trunk through the whole trial:

$$\mathbf{x}_\ell(t) = \mathbf{x}_{trunk}(t) + \mathbf{R}_{trunk}(t)\,\mathbf{d},\qquad \mathbf{x}_{CoM}(t) = \frac{m_b\,\mathbf{x}_b(t) + m_\ell\,\mathbf{x}_\ell(t)}{m}.$$

![Figure 11. Placing the load (synthetic trial: 20 kg, 150 mm behind the trunk). A: side view in the quiet standing. The standing CoP (green dotted line) is below the system CoM (black diamond), which lies on the line between the body CoM (blue) and the load (orange). Knowing the two masses, the load is placed on that line. B: the equations. The load was recovered 155 mm behind the trunk, and the system CoM is 83 mm from the body's.](figures/fig11.png){width=6.5in}

::: {custom-style="Why Box"}
**Why the system CoM** — a 20–40 kg pack moves the centre of mass by several centimetres, backwards and upwards. The margin of stability (§10) and CoM work (§9) concern the mass that has to be kept over the feet and moved: body **and** load. **Why centre it left-right** — the equation divides by $m_\ell$, so any CoP error is multiplied by $m/m_\ell$. That factor is about 5 for a 20 kg load on an 80 kg body, so a few millimetres of registration error would place the load centimetres to one side. Packs and vests are symmetric, so the mid-line is the better estimate. **Why carry it with the trunk** — a pack moves with the trunk, and the trunk leans and sways in every stride. **Why the 0.40 m check** — a load placed more than 40 cm from the trunk is physically implausible and means the standing or the body mass is wrong. It is then placed on the trunk instead, with a warning (`LOAD_MAX_OFFSET_M`).
:::

::: {custom-style="Code Box"}
**In the code** — §5.6: `trunk_frame()`, `system_com()`. Settings: `LOAD_MIN_KG`, `LOAD_HEIGHT_M`, `LOAD_CENTRED`, `LOAD_MAX_OFFSET_M`. Results: `load_offset_x_m`, `load_offset_y_m`, `load_offset_z_m` (trunk frame: x right, y forward, z up).
:::

## 5.7 CoM velocity: markers and force plates together

**What.** The velocity of the system centre of mass, taken from the markers at low frequencies and from the integrated force at high frequencies.

**Equations.**

$$\mathbf{v}_{mk} = \frac{d}{dt}\mathbf{x}_{CoM},\qquad \mathbf{a} = \frac{\mathbf{F}_L + \mathbf{F}_R - \overline{\mathbf{F}_L + \mathbf{F}_R}}{m},\qquad \mathbf{v}_F(t) = \int_0^t\mathbf{a}\,d\tau$$

$$\mathbf{v} = H_{LP}\{\mathbf{v}_{mk}\} + \left(\mathbf{v}_F - H_{LP}\{\mathbf{v}_F\}\right)$$

The marker derivative $d/dt$ is a Savitzky–Golay filter (11 samples, cubic). $H_{LP}$ is a 2nd-order Butterworth low-pass at $f_c=0.5$ Hz, run forwards and backwards. Its gain is $\left|H(f)\right|^2$, so the force branch has gain $1-\left|H(f)\right|^2$, and **the two gains add up to exactly 1 at every frequency** (Fig. 12A). Nothing is counted twice and nothing is lost. In the belt frame,

$$\mathbf{v}_{belt} = \mathbf{v} + v_b\,\hat{f}.$$

**Steps** (`complementary()`, `fuse_velocity()`):

1. Fill tracking gaps in the CoM by straight lines, only so the filters can run. The filled frames are set back to missing at the end, widened by the filter's reach.
2. **Marker branch:** one Savitzky–Golay step smooths the CoM and differentiates it.
3. **Force branch:** add both belts' raw GRF and subtract the mean over the trial. That removes gravity, because the mean vertical force over a steady walk is $mg$, and it removes the plates' zero offsets. Divide by the system mass. Anti-alias at 40 Hz and take every 10th sample (1000 → 100 Hz). Integrate by the trapezoid rule and remove the mean.
4. **Axes check** (Fig. 9). Band-pass (0.5–5 Hz) both the force-based acceleration and the markers' second derivative, and correlate them per axis. If a horizontal axis correlates below −0.5, the force axis is reversed: it is flipped with a warning. If any axis falls below 0.7, the script warns that the axes or the mass are off.
5. Combine the two branches with the complementary filter, and add the belt velocity.

![Figure 9. The axes check. Band-passed (0.5–5 Hz) acceleration of the tracked CoM (grey) against the GRF divided by the system mass (blue), across and along the belt. The two must agree: this tests the plate-to-Theia rotation, the sign of each horizontal axis and the mass. A reversed axis would give r ≈ −1. The correlations are reported per trial (`grf_vs_marker_acc_r_X/Y/Z`).](figures/fig09.png){width=6.5in}

![Figure 12. The complementary filter. A: the marker branch (blue) keeps frequencies below 0.5 Hz, and the force branch (orange) keeps those above; their gains add up to 1 (dotted). The stride frequency (grey band) lies well inside the force branch. B: the sideways CoM velocity in the synthetic trial: markers alone (grey), fused (black) and the truth (green). The RMS error falls from 25 to 2 mm/s.](figures/fig12.png){width=6.5in}

::: {custom-style="Why Box"}
**Why fuse** — the margin of stability (§10) divides the velocity by $\omega_0\approx3.3$ s⁻¹, so an error of 30 mm/s in velocity becomes 9 mm in the margin, against margins of a few centimetres. Differentiating a tracked position amplifies its noise exactly at the within-stride frequencies. Integrating the force is exact there (Newton's law, sampled at 1000 Hz, no differentiation), but it drifts slowly with any error in the mass or the plate zero. Each source is used where it is good. **Why 0.5 Hz** — the stride frequency is about 0.9 Hz, so everything within a stride comes from the force, and the slow drift of the integral is replaced by the markers. **Why the same filter on both branches** — so the gains sum to exactly one and the fused signal has no gap or overlap in frequency. **Validation** (§20): with 1 mm of marker noise, the fused error is 0.5–3 mm/s against 25 mm/s for the markers alone, with no bias below 0.5 Hz.
:::

::: {custom-style="Code Box"}
**In the code** — §5.7: `complementary()`, `fuse_velocity()`. Settings: `CROSSOVER_HZ = 0.5`, `MARKER_SAVGOL = (11, 3)`, `ANTIALIAS_HZ = 40`. The trial holds `com_vel` (lab frame), `com_vel_belt` (belt frame) and `com_vel_markers` (markers alone, for comparison).
:::
