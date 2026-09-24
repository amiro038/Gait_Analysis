# 1. Overview

## 1.1 What the analysis does

Participants walk on an instrumented split-belt treadmill while a markerless system (Theia3D) films them. The belt carries them at a fixed speed and a metronome sets their step rate. Each trial starts with a few seconds of standing still. The analysis turns three raw data streams into one row of results per trial:

* the **Theia3D** skeleton (100 Hz), exported from Visual3D;
* the **two force plates** under the belts (1000 Hz);
* a **MOVE4D** 3-D scan of the participant's boots.

The row has about 370 columns, grouped in families: spatiotemporal, posture, kinetics, centre-of-mass work, stability, foot clearance, variability, entropy, trunk dynamics, control of stepping and local dynamic stability. Wherever it can be computed, each value comes with a 95% confidence interval.

![Figure 1. The pipeline. Step 1 (detect_gait_events.py) finds every heel strike and toe-off, trusting the force plates only where the boot meshes show that the event belongs to one foot on one belt. Step 2 (gait_analysis.py) prepares the trial (§5), computes the per-step metrics (§6–11) and then the per-trial metrics with their confidence intervals (§12–19).](figures/fig01.png){width=6.3in}

## 1.2 Why two scripts

Event detection is the foundation: nearly every metric is a time or a position read at a heel strike or a toe-off. It is therefore done once, carefully, and saved (`{trial}_event_qa.csv`), so it can be checked before any metric is computed. `gait_analysis.py` then reads those events and never re-detects them. Both scripts use the **same** boot geometry: `gait_analysis.py` imports it from `detect_gait_events.py`. That way the step that decides which force-plate events to trust and the step that measures foot clearance agree on where the boot is.

::: {custom-style="Code Box"}
**In the code** — `detect_gait_events.py` (step 1), then `gait_analysis.py` (step 2). In `gait_analysis.py`, `analyse_trial()` (end of the file, section RUN) carries out §5 → §19 for one trial, in the order of this document. `main()` repeats it for every trial and rebuilds the results workbook.
:::

## 1.3 The three frames of reference

**Theia's lab frame.** Every position is in Theia's global frame: X and Y horizontal, Z vertical, in metres.

**The belt frame.** On a treadmill the walker hardly moves in the lab; the belt moves under them. A metric about forward motion, such as the extrapolated centre of mass (§10) or centre-of-mass work (§9), must be computed as if the walker were moving over still ground. To do that, the belt's velocity ($v_b\hat{f}$, §5.2) is added to every velocity. This is the **belt frame**.

**The walker's axes.** AP and ML are along and across the **belt's direction of travel**, measured from the feet (§5.2), and not along Theia's X and Y. The two need not line up. Even a 1–2° misalignment moves part of the step length (≈ 0.75 m) into the step width (≈ 0.1 m): 1.3° is 17 mm, which is the size of the effects of interest.

::: {custom-style="Why Box"}
**Why the direction of travel comes from the feet and not from the boots' long axes** — people toe out, and a boot's long axis points a few degrees away from the direction the foot moves. A foot standing on the belt moves exactly with the belt, so its velocity gives the true direction of travel.
:::

## 1.4 Steady walking

Each trial starts with a quiet standing, the belt ramps up, and the walker settles into the rhythm. Every per-trial number is computed only from the steps after a **60 s warm-up** from the first event (`WARMUP_S`, §5.1). All steps are kept in `{trial}_steps.csv`, with a `steady` flag, so the warm-up can be inspected.

# 2. Equipment and data

## 2.1 Motion capture: Theia3D, exported from Visual3D

Theia3D estimates a skeleton from synchronised video without markers. Visual3D then exports, per frame at 100 Hz (`{trial}_metrics.csv`):

* each segment's 4 × 4 pose (`*_Global_4x4`), including the feet and toes;
* joint centres and landmarks (`Left_Heel_Position`, `Right_Toes_Position`, `Pelvis_Position`, `Low_Back_Position`, `Neck_Position`, `Trunk_Position`, …);
* the whole-body centre of mass (`Whole_body_COG`);
* joint and segment angles (`*_Joint_Angle`, `Thorax_Seg_Angle`, `Pelvis_Seg_Angle`);
* the trunk's linear velocity (`Trunk_Linear_Velocity`).

Markerless joint-angle estimates agree with marker-based ones to within a few degrees in the sagittal plane for healthy gait (Kanko et al. 2021). The **body** centre of mass comes from Theia's segment model, which does not include the carried load. §5.6 adds the load.

::: {custom-style="Code Box"}
**In the code** — `read_export()` reads the file (5 header rows, then one row per frame) into a dictionary of (frames × components) arrays. `foot_pose()` rebuilds each 4 × 4 pose and makes its rotation exactly orthonormal, because the export rounds it. The `Trial` class holds everything about one trial: `kin()` returns a signal in Theia's frame, and `kin_walker()` returns it in the walker's ML/AP/VT axes.
:::

## 2.2 The split-belt instrumented treadmill

Each belt sits on its own force plate, sampled at 1000 Hz, which gives force (X, Y, Z), moment (X, Y, Z) and centre of pressure (CoP) per belt. The four measured corners of each plate come from the C3D force-plate parameters (`force_plates_DICE_treadmill.txt`). The handrails carry their own force channels. The DICE plates are pitched by about 0.8° (≈ 13 mm of height change over one stance). This is small, but it matters for foot clearance, which is measured in millimetres, so §3.4 fits the belt surface.

## 2.3 The boots: MOVE4D scans

Joint centres sit centimetres inside the real edge of a boot, and the underside of a boot is not the "toe" landmark. Two metrics are about the real boot: the base of support (§10) and foot clearance (§11). So each participant's boots are scanned with MOVE4D and fixed to Theia's feet (§3).

## 2.4 The participants file

`participants.csv` holds one row per participant: `participant`, `body_mass_kg` (**without** the load), `leg_length_m`, `height_m`, `foot_length_m`. Body mass is needed to know the load (§5.5). Without it, the load is unknown and the centre of mass is the body's alone; the script says so in its output.

## 2.5 The trials

Trial names carry the condition, for example `D05_C1_Treadmill_1.3mpers 108bpm`: participant D05, condition C1, belt at 1.3 m/s, metronome at 108 beats (steps) per minute. The metronome fixes the step time at 60/108 = 0.556 s and the stride time at 1.111 s.

::: {custom-style="Note Box"}
**Caution: the metronome.** With a fixed belt speed and a fixed step rate, the step length is also nearly fixed on average. The metronome also changes the **variability structure** of the timing series. Healthy adults walking freely show persistent long-range correlations in stride time (DFA α ≈ 0.75–0.9). When they follow a metronome, α falls below 0.5, because each step is corrected towards the beat (Hausdorff et al. 1996; Terrier et al. 2005). The code flags every timing series as "metronome-paced" (§14), and its α must not be compared with the uncued healthy range.
:::
