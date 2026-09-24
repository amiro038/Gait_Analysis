# 3. The boots on Theia's feet

## 3.1 Binding the scanned boots to Theia's feet

**What.** Each boot mesh is attached rigidly to Theia's foot segment, so it can be posed on the foot in every frame of every trial.

**Steps** (`build_foot_binding.py`, run once per participant):

1. **MOVE4D → Theia transform.** The participant stands in an A-pose that both systems record. The rigid transform $\mathbf{T}_{M\to T}$ between the two lab frames is solved from matching landmarks, with the rotation about the vertical only, because both systems take their vertical from gravity. The solve checks for mirroring, units and consistency between body segments, and estimates how uncertain the heading is.
2. **The boot meshes in Theia's frame.** The segmented `.obj` boot meshes are moved into Theia coordinates. The script checks that the soles land on Theia's floor.
3. **Binding.** With Theia's foot pose $\mathbf{T}_{foot}(t_0)$ in the same static pose, every vertex is expressed in the foot's own frame:

$$\mathbf{v}^{foot} = \mathbf{T}_{foot}(t_0)^{-1}\;\mathbf{T}_{M\to T}\;\mathbf{v}^{M4D}$$

In any later frame $t$, the boot is then posed by

$$\mathbf{v}(t) = \mathbf{T}_{foot}(t)\;\mathbf{v}^{foot}.$$

::: {custom-style="Why Box"}
**Why a scan and not the joint centres** — the ankle joint centre sits about 3–5 cm inside the lateral edge of a boot, and the "toe" landmark is not the lowest point of the toe cap. The margin of stability measured to the ankle understates the base of support by that much (§10). Foot clearance measured at a landmark is not the clearance of the sole (§11). The scan gives the boot's real edge and its real underside.
:::

## 3.2 The sole

**What.** The small set of vertices that can touch the ground.

**Steps** (`dge.boot_sole()`):

1. Take "up" as the world vertical seen from the foot (the median over the trial), and the boot's long axis as the first principal direction of the mesh in the horizontal plane.
2. Divide the footprint into **10 mm × 10 mm cells** (`SOLE_CELL_MM`) and keep the **lowest vertex in each cell**.
3. Keep only cells whose lowest vertex is within **35 mm of the lowest point of the boot** (`SOLE_MAX_RISE_MM`). This keeps the curved toe cap and heel, and drops the boot's sides.

![Figure 2. A: the scanned boot (grey), its sole points under the foot (orange) and under the toe cap (green). B: at push-off the heel is raised 30°. If the boot were rigid, the toe cap would pass 17 mm through the belt (grey points). Bending the toe cap about the MTP by Theia's toe angle keeps it on the belt (orange and green).](figures/fig02.png){width=6.5in}

::: {custom-style="Why Box"}
**Why only the sole** — a boot mesh has tens of thousands of vertices. Only the underside can touch the belt, and clearance is the height of the lowest point of that underside. Keeping one vertex per 10 mm cell leaves a few hundred points: enough to resolve the sole's shape, few enough to pose them in every frame of a 7-minute trial.
:::

## 3.3 The toe hinge

**What.** The toe cap bends about the metatarsophalangeal (MTP) joint, following Theia's toe angle.

With $\mathbf{m}$ the MTP in the foot frame (from `<Side>_Toes_Position`), $\theta_{toe}(t)$ Theia's toe angle, and $\mathbf{R}_x$ a rotation about the foot's medio-lateral axis, each sole point in front of the MTP is posed by

$$\mathbf{v}(t) = \mathbf{T}_{foot}(t)\,\left[\mathbf{m} + \mathbf{R}_x\left(\theta_{toe}(t) - \theta_{scan}\right)\,(\mathbf{v}^{foot} - \mathbf{m})\right],$$

where $\theta_{scan}$ is the toe angle during the scan (`TOE_REFERENCE_DEG`). If the export has the toes' own 4 × 4 pose, that pose is used directly.

::: {custom-style="Why Box"}
**Why** — a rigid boot pitched heel-up at push-off drives its toe cap 30–40 mm through the belt (Fig. 2B). That would wreck the contact test in §4.3 (the boot would look as if it touched the wrong belt). It would also wreck the clearance just after toe-off, because the lowest point would be below the belt.
:::

## 3.4 The belt surface

**What.** The belt's height under any point, as a plane pitched along the direction of travel:

$$z_{belt,b}(s) = z_{0,b} + k\,s,\qquad s = \mathbf{p}\cdot\hat{f},$$

with one slope $k$ shared by both belts and one offset $z_{0,b}$ per boot. The height of any sole point above the belt is then

$$h = z - z_{0,b} - k\,s.$$

**Steps** (`dge.fit_belt_surface()`):

1. In every frame, find each boot's lowest sole point. Keep its height and its position $s$ along the direction of travel.
2. Start from $z_{0,b}$ = the 25th percentile of those heights and $k=0$.
3. Fit $z_{0,L}$, $z_{0,R}$ and $k$ by least squares, using only frames whose lowest point is within **20 mm** of the current surface. These are the frames with the boot on the belt; frames in swing are excluded.
4. Refit three more times with a **8 mm** band.

![Figure 3. The belt surface. Each dot is the lowest sole point of a boot in mid-stance (every third stance shown). The lines are the fitted surface: one slope (0.88° here), one offset per boot. The 1–2 mm offset between the two boots absorbs a small bias in either foot's pose.](figures/fig03.png){width=6.5in}

::: {custom-style="Why Box"}
**Why one slope and two offsets** — the belts are one rigid, slightly pitched structure, so they share a slope. Each foot's pose can carry its own small vertical bias, which the separate offsets absorb. Otherwise that bias would show up as a clearance difference between the feet. **Why the iteration** — the lowest point of a swinging foot is not on the belt. The narrowing band removes those frames without having to know the gait events first.
:::

## 3.5 Is the boot touching the belt?

A sole vertex **touches** the belt when $h < 15$ mm (`CONTACT_HEIGHT_MM`). The boot is **down** when at least **3** vertices touch (`MIN_CONTACT_VERTICES`). A tolerance of 15 mm covers the sole's own compression and the pose error, and three vertices cannot be produced by a single noisy point.

::: {custom-style="Code Box"}
**In the code** — `Boots` (gait_analysis.py §5.3) poses the soles in every frame and keeps, per frame, the lowest sole point's height and index and the boot's extent along and across the belt. `Boots.height()` gives the height above the belt, and `Boots.world()` rebuilds all sole points for the frames a swing or a figure needs. The geometry itself comes from `detect_gait_events.py` (`boot_sole`, `toe_hinge`, `fit_belt_surface`, `above_belt`), so both steps use the same boot.
:::

# 4. Gait events (detect_gait_events.py)

Every heel strike (HS) and toe-off (TO) is needed, for both feet, through the whole trial. Force plates time these events to the millisecond, but only when **one** foot is on **one** belt. On a split-belt treadmill a foot sometimes lands across the gap or on the other belt, and then a plate carries two feet, or part of one. The strategy is therefore:

1. **Force first.** Find every belt contact from the force. Keep each of its two events only if the boots show that it belongs to one foot, alone on that belt (§4.1–4.3).
2. **Kinematics for the rest.** Take every other event from the Zeni method, calibrated against the trusted force events of the same trial (§4.4).
3. **Last resort.** Where Zeni finds nothing (a tracking dropout), use a clean-looking contact that the mesh could not check, then interpolation (§4.5).

## 4.1 Belt contacts from the vertical force

**Steps** (`dge.belt_baseline()`, `dge.find_contacts()`, `dge.raw_edge()`):

1. **Baseline.** An unloaded plate does not read zero, and the reading drifts. Every **10 s** (`BASELINE_WINDOW_S`), the baseline is the median of that window's lowest 20% of samples. A window whose level is more than 20 N from the median of all windows is one where the belt was never unloaded (standing before the belt starts). It borrows its neighbours' baseline. The baseline is interpolated between windows.
2. **Filter.** Apply a 4th-order Butterworth low-pass at **50 Hz**, run forwards and backwards so there is no time lag (`FORCE_FILTER_HZ`).
3. **Threshold.** The belt is loaded where the filtered force above baseline exceeds **20 N** (`FORCE_THRESHOLD_N`).
4. **Merge and drop.** Dips shorter than **30 ms** do not split a contact (`MERGE_GAP_S`). Contacts shorter than **80 ms** are dropped (`MIN_CONTACT_S`).
5. **Refine each edge on the raw force.** Within **±15 ms** of the filtered crossing (`EDGE_SEARCH_S`), the landing is the first sample from which the **raw** force stays above 20 N for **5** samples (`EDGE_HOLD`). The lift-off is the first sample from which it stays at or below 20 N for 5 samples.
6. **Shape.** A real landing rises from 20 N to **200 N** within **60 ms** (`MAX_LOADING_S`). A real lift-off falls from 200 N within **80 ms** (`MAX_UNLOADING_S`). Slower edges usually mean a foot sliding on or off the belt, or a second foot, and those events are not trusted.

$$\text{contact} = \left\{t : F_z^{filt}(t) - b(t) > 20\ \text{N}\right\}$$

![Figure 4. Why each edge is placed on the raw force. The zero-lag filter spreads a steep landing over a few milliseconds, so the filtered force (blue) crosses 20 N a few milliseconds before the raw force does (dashed blue line), and crosses it after the foot has left. The edge is placed where the raw force crosses and stays crossed (orange line).](figures/fig04.png){width=6.5in}

::: {custom-style="Why Box"}
**Why filter, then refine** — the raw force is noisy enough to cross 20 N on noise alone, so the filtered force decides *whether* the belt is loaded. But a zero-lag filter spreads a steep edge symmetrically, so the filtered crossing is early at a landing and late at a lift-off. That makes every stance a few milliseconds too long (up to about 1% of a stance). Searching the raw force near the filtered crossing gives the timing back. The 5-sample hold stops a single noise spike from triggering the edge. **Why 20 N** — well above the plate noise (a few N), and well below any real loading (hundreds of N within 20 ms).
:::

## 4.2 The plates in Theia's frame

The plates' corners are in the motion-capture lab frame, which is not Theia's. To check which boot stands on which belt, the plate outlines and the CoP must be in Theia's frame.

**Steps** (`dge.fit_plate()`, `dge.register_lab_to_theia()`):

1. **Plate outlines.** The nominal plate rectangle (length × width) is fitted rigidly onto the four measured corners (Kabsch algorithm), giving the plate's rotation $\mathbf{R}_p$ and position $\mathbf{t}_p$. How far the measured corners miss the fitted ones (8 mm at most on DICE) is how uncertain the outline is.
2. **The rotation, lab → Theia.** This is the angle between the belts' direction of travel as seen by the plates (the plates' long axis, pointed the way the CoP moves) and as seen by Theia (the stance boot's velocity). Only single-support frames are used, with one boot down, the other boot well clear and the other belt unloaded.
3. **The translation.** In those frames, the CoP must lie under the touching part of the boot. The translation is the median offset between the rotated CoP and the centre of the touching patch, computed per boot and **averaged over the two boots**, so a medial/lateral bias that mirrors between the feet cancels.
4. **Check.** If the across-belt CoP runs against the across-belt boot position, the frames are mirrored, and the script stops rather than guess.

$$\mathbf{p}_{Theia} = \mathbf{R}(\varphi)\,\mathbf{p}_{lab} + \mathbf{t}$$

## 4.3 Which force events to trust

**Contact labels.** Around each belt contact, the posed boots show which boot is on that belt, and whether it is alone:

| Label | Meaning | Trusted? |
|:--|:--|:--|
| clean | one boot on its own belt | yes, if the event checks pass |
| crossover | one boot, alone, on the *other* foot's belt | yes; the limb comes from the mesh, not from the belt's name |
| straddle | the boot also stands on the other belt | no: the force is split |
| shared | the other boot also stands on this belt | no: two feet's force |

**Per-event checks.** Each event is judged on its own, over **±50 ms** around it (`TRUST_WINDOW_S`). It is trusted only if **all** of these pass, and the first one that fails is written as the reason in `{trial}_grf_contacts.csv`:

1. the contact is complete (it does not run off the start or end of the recording);
2. the edge has the shape of a real landing or lift-off (§4.1 step 6);
3. both boots are tracked (the mesh can be posed);
4. the other boot is not on this belt;
5. this boot is down near the event;
6. this boot is not also on the other belt;
7. while this foot alone loads the belt, the CoP lies within **30 mm** of the boot's touching patch at least **90%** of the time (`COP_TOLERANCE_MM`, `COP_MIN_INSIDE`).

A boot is "on" a belt when at least 3 of its touching sole vertices lie inside the plate outline, grown by the outline's own uncertainty. Hanging over the gap or the outer edge is fine: it moves no force anywhere.

![Figure 5. Two contacts seen from above, half-way through stance. Grey: the belts. Coloured outlines: the boots; dots: sole points touching the belt; ×: the CoP. Left, a clean contact: the right boot alone on its belt, the CoP under it; both events are trusted. Right, a straddle: the left boot stands across the gap, so each plate carries part of its force and the heel strike is not trusted (reason "foot_on_other_belt"). A crossover, where one boot lands wholly on the other belt, is trusted with the right limb.](figures/fig05.png){width=6.0in}

::: {custom-style="Why Box"}
**Why check each event separately** — a foot can land cleanly and then drift onto the gap before toe-off, or the reverse. Checking only the whole contact would throw away a good heel strike because of a bad toe-off, or keep a bad one. **Why the CoP test** — it catches what the geometry cannot: a trailing toe still loading this plate, or a hand on the rail.
:::

## 4.4 Zeni events for the rest

**What.** Zeni et al. (2008): in walking, the heel strike is where the heel is furthest **in front of** the pelvis, and the toe-off is where the toe is furthest **behind** it. Heel strike is therefore a maximum of $x_{HS}$, and toe-off a maximum of $x_{TO}$:

$$x_{HS}(t) = (\mathbf{p}_{heel} - \mathbf{p}_{pelvis})\cdot\hat{f},\qquad x_{TO}(t) = -(\mathbf{p}_{toe} - \mathbf{p}_{pelvis})\cdot\hat{f}.$$

**Steps** (`dge.zeni_signals()`, `dge.peaks_between()`, `dge.zeni_offsets()`, `dge.build_sequence()`):

1. Bridge tracking gaps of up to 10 frames. Low-pass the heel, toe and pelvis at **10 Hz** (`KINEMATIC_FILTER_HZ`).
2. Find the maxima to a fraction of a frame by fitting a parabola through the peak sample $i$ and its neighbours:
$$\delta = \tfrac{1}{2}\,\frac{x_{i-1} - x_{i+1}}{x_{i-1} - 2x_i + x_{i+1}},\qquad t_{peak} = i + \delta.$$
3. Keep a maximum only if it stands at least **50 mm** above the lowest point within **±0.6 s** on each side (`ZENI_MIN_PROMINENCE_M`, `ZENI_WINDOW_S`). In walking a real swing gives about 0.3 m; a foot standing still during the opening quiet standing gives a few millimetres of tracking noise.
4. **Calibrate on this trial.** Zeni's events sit a consistent few milliseconds from the true ones, and the offset differs per person and per event. For every trusted force event, Zeni's event is looked for exactly as a missing one would be, and the **median difference** is that limb's and that event's offset. If one limb has fewer than 10 such pairs, the two limbs are pooled. The spread of the differences is reported (in `{trial}_event_summary.json`) as the likely error of a Zeni event.
5. **Search only where the gait sequence leaves room.** The events must come in the order L HS → R TO → R HS → L TO. Between two trusted events, the sequence says which events are missing and, from this trial's median intervals, roughly when. Each missing event is searched for only in its own gap.

![Figure 6. The Zeni signals over three strides: heel ahead of the pelvis (solid) and toe behind it (dashed), left (blue) and right (orange). Downward triangles are trusted force-plate heel strikes and upward triangles trusted toe-offs; each falls at the corresponding maximum.](figures/fig06.png){width=6.5in}

::: {custom-style="Why Box"}
**Why Zeni rather than a velocity or height threshold** — it needs no threshold that would depend on the participant, the footwear or the load. It works at any belt speed, and it is among the most accurate kinematic methods for treadmill walking. **Why calibrate it** — the median offset from the same trial's force events removes Zeni's systematic error. What remains is its random error, which the event summary reports.
:::

## 4.5 The full sequence, and what is written

Where Zeni finds nothing in a gap, the gap is filled by a clean-looking force event that fell in a tracking dropout (source `GRF_unverified`). Only if there is none, and only between two trusted events, is the event interpolated at the time the sequence predicts (source `interpolated`). Each limb's list is trimmed to start on a heel strike and end on a toe-off. Every stance and stride more than 25% away from the median is flagged (`FLAG_RATIO`).

Outputs per trial: `{trial}_merged_events.csv` (event, limb, frame, source), `{trial}_event_qa.csv` (the same events to a fraction of a frame, with the belt, the contact label and how each was found), `{trial}_grf_contacts.csv` (every contact and why each event was or was not trusted), and `{trial}_event_summary.json` (plate fit, registration, belt surface, Zeni offsets and errors).

The results workbook reports the share of events from each source: `events_trusted_grf_pct`, `events_zeni_pct` and `events_interpolated_pct`. A trial with many Zeni or interpolated events has less precise timing, and fewer stances can be used for kinetics (§8).
