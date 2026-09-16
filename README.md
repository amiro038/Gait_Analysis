# MOVE4D → Theia3D rigid alignment

Finds the 4×4 transform `T` such that `p_theia = T @ [p_move4d; 1]`, from a
static standing (A-pose) trial recorded simultaneously by both systems.

**Start here: `m4d_to_theia_transform.py`.** It is a single self-contained
script — open it in Spyder, set the two file paths in the CONFIG cell, press
F5. Only `numpy` is required (`matplotlib` for the optional figure). No
Autodesk FBX SDK, no Blender.

## Files

| file | purpose |
|---|---|
| **`m4d_to_theia_transform.py`** | **everything: FBX reader, scene evaluator, solver, diagnostics, figure** |
| `fbx_raw.py`, `fbx_scene.py`, `align_m4d_to_theia.py`, `plot_alignment.py` | the earlier multi-module version, kept for reference. Superseded. |

## Usage

```python
# in Spyder: edit the CONFIG cell, then F5. Afterwards:
RESULT["yaw_deg"]
pts_theia = apply_transform(T, pts_m4d)          # (..., 3)
R_theia   = transform_orientation(T, R_m4d)      # (..., 3, 3)
```

```bash
python m4d_to_theia_transform.py THEIA.fbx M4D.fbx        # yaw-only (default)
python m4d_to_theia_transform.py THEIA.fbx M4D.fbx --full # unconstrained 3-DOF
```

Writes `T_m4d_to_theia.txt` / `.npy`, `alignment_report.json` and
`alignment_check.png` into `OUTDIR`.

`T` maps **raw MOVE4D file coordinates** (Y-up, as exported) directly into
**raw Theia file coordinates** (Z-up). The up-axis conversion and the unit
scale factors are baked in — do not pre-convert.

## Method

The objective is that **segment orientations agree once both systems are in
the same coordinate frame**. Because the two skeletons are defined
differently, the fit uses only the **arm and leg segments** — thigh, shank,
upper arm, forearm. Feet, hands, pelvis, trunk and head are excluded from the
fit and reported as *validation* landmarks instead.

Two families of observation, both built from joint-centre positions:

* **segment long axes** — hip→knee, knee→ankle, shoulder→elbow, elbow→wrist.
  These constrain the out-of-plane part of the rotation.
* **bilateral medio-lateral axes** — right→left at hip, knee, ankle, shoulder,
  elbow and wrist. These are horizontal, so they are what actually pins the
  heading, and they are insensitive to a joint-centre offset that is
  *symmetric* between sides, which is the usual case.

Everything is a unit direction between two landmarks, so the rotation is
translation-invariant and decouples from the translation. The translation is
then fitted by robust point matching on the same joint centres.

Pipeline: parse both FBX files → evaluate the full Autodesk transform chain
to global joint transforms per frame → canonicalise to Z-up/metres → reject
bind poses and filter-edge frames → check handedness and units → solve the
rotation (Wahba/Kabsch with Huber IRLS, or the closed-form constrained yaw)
→ solve the translation → diagnose.

Static trials collapse to their mean pose (the frame grids need not overlap);
dynamic trials are resampled onto a common time base, which is what handles
the differing frame rates.

### Why not fit on the full segment orientation matrices?

Because the two rigs use unrelated local bone-axis conventions, so every
segment carries its own constant offset:

```
R_theia,s = G · R_m4d,s · O_s
```

and from a single posture `G` and the `O_s` are perfectly confounded. The
script measures the `O_s` and prints them. On `D05_C1` they are **87–118°**
and differ *between segments* by up to 30°, so even the "shared offset" model
(all `O_s` equal — which *would* be identifiable) fails: it leaves 8–9° of
residual and its 3-DOF solution is unstable (tilt wanders to 16–81°).

The **long axis** is the part of a segment frame that both systems define
anatomically rather than by convention. That is what the fit uses; the axial
roll is ignored.

> **Caveat.** `T[:3,:3] @ R_m4d` re-expresses a MOVE4D segment frame in
> Theia's *coordinate system*. It does **not** convert MOVE4D's local bone
> convention into Theia's anatomical convention — that is a separate,
> per-segment problem.

### Rotation model

`ROTATION_MODEL = "yaw"` (default) constrains the rotation to the vertical.
Both systems establish their vertical from the same physical gravity/lab
calibration, so in principle only the heading and the origin differ. The yaw
is solved **in closed form** under that constraint:

```
θ = atan2( Σ w (u_y v_x − u_x v_y),  Σ w (u_x v_x + u_y v_y) )
```

This is a genuinely different estimator from fitting 3 DOF and reading the
yaw off the result — in the unconstrained fit the tilt is free to absorb
skeleton mismatch and it drags the yaw with it. On this trial that difference
is what makes the anatomical subsets agree to 0.8° instead of 2.7°.

### Observation weighting

Weights are derived, not hand-tuned. Each landmark carries an expected
between-system definition uncertainty σ (mm, `LANDMARK_SIGMA_MM`); a
direction between landmarks *a* and *b* then has angular uncertainty
≈ `sqrt(σa² + σb²) / L`, plus a systematic floor, and is weighted by `1/σ²`.
So the same 12 mm of landmark ambiguity is 5.6° across a 0.12 m hip width but
0.7° across a 0.95 m wrist-to-wrist span, and the fit weights them
accordingly.

Bilateral ML axes get an extra discount (`ML_ASYMMETRY_FRACTION`): their two
endpoint errors are strongly *correlated*, so only the left–right asymmetric
part tilts the vector.

## What the diagnostics mean

**Length ratios.** Per-segment Theia/MOVE4D length ratio. The median catches
a unit error instantly (it would be 100, 2.54, 1000). The *scatter* around it
measures how much the two skeletons disagree, and sets a floor on achievable
residuals.

**End-to-end check.** `T` is assembled in raw file coordinates from a fit done
in a canonicalised frame, which involves both files' up-axis matrices and unit
scale factors. The script pushes the raw MOVE4D landmarks through `T` and
compares against the raw Theia landmarks. In static mode this **must**
reproduce the fitted residual exactly; if it does not, the bookkeeping is
wrong and the matrix is unusable.

**Validation landmarks.** Segment origins that are a rigging convention
rather than an anatomical point are reported but never fitted. Their offsets
are skeleton definition gaps, **not** alignment error — do not quote them as
accuracy.

**Yaw information** = Σ w·|horizontal component|². The Fisher information for
rotation about the vertical. Parallel horizontal vectors each contribute;
they are redundant, not degenerate.

**Azimuthal diversity** = eigenvalue ratio of the horizontal scatter. This
does *not* affect whether yaw is identifiable. It measures whether
independent anatomical directions exist to cross-check each other, which is
what protects against a bias shared by every vector pointing the same way. An
A-pose is nearly planar, so this is low by construction.

**Subset agreement.** Yaw estimated separately from leg ML axes, arm ML axes,
leg long axes and arm long axes. If these disagree, the disagreement *is* the
uncertainty, whatever the residuals say. Subsets below `MIN_YAW_LEVERAGE`
(mean horizontal component) are shown but excluded from the quoted spread: in
a standing pose the leg long axes are ~9° off vertical, so a 0.5° error in
them becomes several degrees of yaw.

**Bootstrap** resamples direction *labels*, not individual observations,
because the dominant error is a systematic per-segment definition mismatch
rather than independent per-frame noise. The segment is the unit of
uncertainty.

**Rotation model comparison.** Both the 3-DOF and yaw-only solutions are
always evaluated. If constraining to yaw does not hurt the point fit, the
out-of-plane rotation was absorbing skeleton mismatch rather than a real
calibration tilt.

## Findings for D05_C1_apose

| | Theia | MOVE4D |
|---|---|---|
| frames | 8 @ 40 fps | 4 @ 60 fps |
| up axis | Z | Y |
| units | metres | metres |

- **Theia frame 0 is a rest/bind pose, not data** (514 mm from the rest of the
  trial); frame 7 is a filter edge artifact. Both are auto-rejected, leaving
  6 frames stable to ~2 mm. All 4 MOVE4D frames are good (1.3 mm of motion).
- Handedness matches; no mirroring. Median length ratio 0.988 → same units.
- Skeleton disagreement is real: thigh 0.94–0.95, shank 1.01–1.02, upper arm
  1.02–1.03, forearm 0.94–0.98, **foot 0.86–0.88, trunk 1.16**. The last two
  are why feet and trunk are excluded — Theia `thorax` sits at shoulder level
  while MOVE4D `Chest4` is 152 mm below it.
- Fit uses 14 directions (8 long axes + 6 ML axes) and 12 joint centres.

**Result (yaw-only, recommended):**

```
yaw −1.93°,  translation (−0.386, +0.225, +0.034) m
```

```
T = [ 0.999434   0.000000  -0.033643  -0.385874 ]
    [-0.033643   0.000000  -0.999434   0.224698 ]
    [ 0.000000   1.000000   0.000000   0.033550 ]
    [ 0.000000   0.000000   0.000000   1.000000 ]
```

| | yaw-only | full 3-DOF |
|---|---|---|
| segment-orientation residual | **1.71° RMS** | 1.67° |
| joint-centre residual | **12.6 mm RMS** | 13.3 mm |

The extra 2 DOF buy 0.04° of angular fit and cost 0.7 mm of position fit —
the 0.60° tilt they find is absorbing skeleton mismatch, not a real
calibration difference. Keep `"yaw"`.

**Uncertainty.** Independent subsets:

| subset | yaw | leverage |
|---|---|---|
| leg ML (hip/knee/ankle width) | −2.12° | 1.00 |
| arm ML (shoulder/elbow/wrist) | −1.71° | 1.00 |
| arm long axes | −2.54° | 0.60 |
| leg long axes | −6.36° | 0.15 — ignore, near-vertical |

Spread over the informative subsets: **0.84°**. Bootstrap over segment
labels: −1.93° ± 0.32°, 95% CI [−2.58, −1.39]. Propagated onto points on the
subject — the number that matters downstream — **1.5 mm RMS, 3.3 mm at the
95th percentile**.

So the error budget is ~13 mm of irreducible skeleton-definition mismatch
plus ~2 mm of alignment uncertainty. **The skeletons, not the alignment, are
the limiting factor.**

## Limitations and how to improve

1. **Pool a dynamic trial.** Azimuthal diversity is 0.007 — every horizontal
   direction in an A-pose is medio-lateral, so a systematic bias shared by all
   of them cannot be detected from within this trial. A walk or a turn from
   the same session, where the pelvis and limbs point in varied horizontal
   directions, gives real diversity. Add it to `TRIAL_PAIRS`. This is the
   single highest-value change.
2. **Translation carries a residual bias** from joint-centre definition
   differences that a single posture cannot separate from misalignment.
   Bilateral joints are used in pairs so the medio-lateral component largely
   cancels; the antero-posterior and vertical components do not. Multiple
   varied postures let the per-joint offsets be solved jointly.
3. **Validate on held-out trials.** Fit on one set, evaluate on another, and
   plot residual against frame. Flat noise means one constant matrix is valid.
   Drift or oscillation with movement means it is not — which points at sync
   error or retargeting artifacts rather than alignment.
4. `LANDMARKS`, `SEGMENT_AXES`, `ML_AXES`, `TRANSLATION_LANDMARKS` and
   `LANDMARK_SIGMA_MM` are explicit tables in the CONFIG cells. If your
   MOVE4D rig version names joints differently, edit those and nothing else.
   MOVE4D has no single wrist node, so the wrist is taken as the centroid of
   the carpometacarpal roots hanging off `*Forearm` — the
   `("children_of", ...)` landmark spec.
