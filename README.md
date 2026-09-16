# MOVE4D → Theia3D rigid alignment

Finds the 4×4 transform `T` such that `p_theia = T @ [p_move4d; 1]`, using
corresponding skeleton joints from FBX exports of the same trial.

## Files

| file | purpose |
|---|---|
| `fbx_raw.py` | binary FBX reader (v7100–7700). Node-record tree, zlib arrays. No SDK needed. |
| `fbx_scene.py` | scene-graph evaluator: hierarchy, animation curves, full Autodesk transform chain, per-frame global joint transforms. |
| `align_m4d_to_theia.py` | the solver and its diagnostics. |
| `plot_alignment.py` | visual validation figure. |

Only `numpy` is required (`scipy` unused, `matplotlib` only for `--plot`).
There is no dependency on the Autodesk FBX SDK or on Blender.

## Usage

```bash
# single trial
python align_m4d_to_theia.py THEIA.fbx M4D.fbx -o alignment_out --plot

# constrain the rotation to the vertical axis (recommended, see below)
python align_m4d_to_theia.py THEIA.fbx M4D.fbx -o alignment_out --yaw-only

# pool several synchronised trials into one fit -- strongly preferred
printf 'D05_C1_apose_Theia.fbx D05_C1_apose_M4D.fbx\n' >  pairs.txt
printf 'D05_C2_walk_Theia.fbx  D05_C2_walk_M4D.fbx\n'  >> pairs.txt
python align_m4d_to_theia.py --pairs pairs.txt -o alignment_out
```

Outputs `T_m4d_to_theia.txt`, `T_m4d_to_theia.npy`, `alignment_report.json`
and optionally `alignment_check.png`.

Applying it:

```python
import numpy as np
from align_m4d_to_theia import apply_transform

T = np.load("alignment_out/T_m4d_to_theia.npy")
pts_theia = apply_transform(T, pts_m4d)      # (..., 3) array, raw M4D coords
```

`T` maps **raw MOVE4D file coordinates** (Y-up, as exported) directly into
**raw Theia file coordinates** (Z-up). The up-axis conversion is baked in —
do not pre-convert.

To rotate an orientation (segment frame, not a point) use `T[:3,:3] @ R_m4d`.

## Method

1. **Parse** both FBX files and evaluate global joint transforms per frame.
2. **Reject bad frames** in two stages: gross outliers (a bind/rest pose sits
   tens of cm from the data) and then relative outliers within the survivors
   (filter edge artifacts sit a few mm away and survive any threshold loose
   enough to be safe against real motion).
3. **Check** units via segment-length ratios, and handedness via the signed
   volume of (medio-lateral, vertical, antero-posterior). A mirrored file
   cannot be fixed by a rotation, so this must be caught, not fitted.
4. **Rotation** from *unit direction vectors* between corresponding joint
   pairs, solved by weighted SVD (Wahba/Kabsch) with Huber IRLS. Direction
   vectors are translation-invariant, so `R` decouples from `t` and is immune
   to joint-centre definition offsets to first order. Full segment *frames*
   are deliberately not used: their axial rotation is pure convention and
   would reintroduce an unknown constant offset per segment.
5. **Translation** by robust point matching given `R`, using only true
   anatomical joint centres.
6. **Diagnose** — see below.

Static trials are collapsed to their mean pose (frame grids need not
overlap). Dynamic trials are resampled onto the overlapping part of a common
time base, which is what handles the differing frame rates.

## What the diagnostics mean

**Scale ratios.** Per-segment Theia/MOVE4D length ratio. The median catches a
unit error instantly (would be 100, 2.54, 1000). The *scatter* around it is a
free measure of how much the two skeletons disagree, and sets a floor on
achievable residuals.

**Excluded landmarks.** Segment origins that are a rigging convention rather
than an anatomical point are reported but not fitted. Their offsets are
skeleton definition gaps, **not** alignment error — do not quote them as
accuracy.

**Yaw information** = Σ w·|horizontal component|². The Fisher information for
rotation about the vertical. Parallel horizontal vectors each contribute;
they are redundant, not degenerate.

**Azimuthal diversity** = eigenvalue ratio of the horizontal scatter. This
does *not* affect whether yaw is identifiable. It measures whether
independent anatomical directions exist to cross-check each other, which is
what protects against a systematic bias shared by every vector pointing the
same way. An A-pose is nearly planar, so this is low.

**Subset agreement.** Yaw estimated separately from medio-lateral vectors,
antero-posterior vectors, and limb long axes. If these disagree, the
disagreement *is* the uncertainty, whatever the residuals say. This is the
most interpretable number in the report.

**Bootstrap** resamples segment *labels*, not individual observations,
because the dominant error is a systematic per-segment definition mismatch
rather than independent per-frame noise. The segment is the unit of
uncertainty.

**Rotation model comparison.** Both the full 3-DOF and yaw-only solutions are
always evaluated. If constraining to yaw does not hurt the point fit, the
out-of-plane rotation was absorbing skeleton mismatch rather than a real
calibration tilt.

## Findings for D05_C1_apose

| | Theia | MOVE4D |
|---|---|---|
| frames | 8 @ 40 fps (0.175 s) | 4 @ 60 fps (0.05 s) |
| up axis | Z | Y |
| units | metres | metres |

- **Theia frame 0 is a rest/bind T-pose, not data** (pelvis at exactly
  (0,0,0.932), hands symmetric at ±0.69). Frame 7 is a filter edge artifact
  ~44 mm off. Frames 1–6 are stable to <3 mm. Both are auto-rejected.
- Handedness matches; no mirroring. Scale ratio 1.013, so no unit conversion.
- Skeleton disagreement is real: thigh 0.94–0.95, shank 1.01–1.02, upper arm
  1.02–1.03, foot 0.86–0.88, trunk 1.16. Theia `thorax` sits at shoulder
  level while MOVE4D `Chest4` is well below — different landmarks, excluded.
- Fit uses 10 joint centres: hips, knees, ankles, shoulders, elbows.

**Result (yaw-only, recommended):** yaw −2.00°, translation
(−0.386, +0.225, +0.032) m, **12.8 mm position RMS**. The unconstrained
3-DOF fit gives 14.2 mm, i.e. the extra freedom makes the fit *worse* —
the 0.87° tilt it finds is not a real calibration difference.

**Yaw uncertainty is the weak point.** Independent subsets give:

| subset | yaw | leverage |
|---|---|---|
| medio-lateral (pelvis + shoulder width) | −2.76° | 1.00 |
| antero-posterior (feet) | −0.93° | 0.95 |
| limb long axes | −0.08° | 0.30 |

Spread 2.68°. Bootstrap: −2.00° ± 0.97°, 95% CI [−3.39, +0.46] — crosses
zero.

Propagating that to actual points on the subject — the number that matters
downstream — gives a **3.4 mm RMS spread (7.2 mm at the 95th percentile)**
for the yaw-only model, versus 10.6 mm (20.4 mm) for the unconstrained fit.
Yaw-only is both the better fit and three times the more stable, which is why
it is recommended here.

So the overall error budget: ~13 mm of irreducible skeleton-definition
mismatch, plus ~3 mm of alignment uncertainty. The skeletons, not the
alignment, are the limiting factor.

This is inherent to a single A-pose, and it is an *azimuthal diversity*
problem, not a rank problem. The pose is nearly planar: the only strong
antero-posterior information comes from the feet, whose definitions
demonstrably differ most (length ratio 0.86). Yaw is identifiable but not
cross-checkable from within this trial.

## Limitations and how to improve

1. **Pool a dynamic trial.** A walk or a turn from the same session, where
   the feet and pelvis point in varied horizontal directions, gives real
   azimuthal diversity and should cut the yaw uncertainty severalfold. Use
   `--pairs`. This is the single highest-value change.
2. **Translation carries a residual bias** from joint-centre definition
   differences that a single posture cannot separate from misalignment.
   Bilateral joints are used in pairs so the medio-lateral component largely
   cancels; the antero-posterior and vertical components do not.
   Multiple varied postures let the per-joint offsets be solved jointly.
3. **Validate on held-out trials.** Fit on one set, evaluate on another, and
   plot residual against frame. Flat noise means one constant matrix is
   valid. Drift or oscillation with movement means it is not — which would
   point at sync error or retargeting artifacts rather than alignment.
4. The joint correspondence in `JOINT_MAP`, the direction vectors in
   `DIRECTION_VECTORS` and the fitted joints in `TRANSLATION_JOINTS` are
   explicit tables at the top of `align_m4d_to_theia.py`. If your MOVE4D
   rig version names joints differently, edit those and nothing else.
