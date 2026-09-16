# MOVE4D → Theia3D rigid alignment

Finds the 4×4 transform `T` such that `p_theia = T @ [p_move4d; 1]`, by aligning
the **global orientations of the thigh, shank, upper-arm and forearm segments**
(the hip, knee, shoulder and elbow segments) from a static standing trial
recorded simultaneously by both systems.

Everything is in **`m4d_to_theia_transform.py`** — one self-contained script.
Open it in Spyder, set the two paths in CONFIG, press F5. Only `numpy` is
required (`matplotlib` for the check figure).

## The six steps

The script is laid out as one section per step:

1. **load** both FBX files
2. **read** the kinematic data for every segment
3. **local → global** via the full Autodesk transform chain
4. **common units and coordinate system** — metres, up-axis rotated onto +Z
5. **rotation** — global segment orientations as close as possible
6. **translation** — joint positions as close as possible

```python
# after F5, the namespace holds T, RESULT, THEIA, M4D
pts_theia = apply_transform(T, pts_m4d)          # (..., 3)
R_theia   = transform_orientation(T, R_m4d)      # (..., 3, 3)
```

```bash
python m4d_to_theia_transform.py THEIA.fbx M4D.fbx          # vertical-axis only
python m4d_to_theia_transform.py THEIA.fbx M4D.fbx --full   # all 3 DOF
```

`T` maps **raw MOVE4D file coordinates** (Y-up) directly into **raw Theia file
coordinates** (Z-up). The up-axis conversion and both unit scale factors are
baked in — do not pre-convert.

## Why not the Autodesk FBX Python SDK?

It is not installable. The SDK is not on PyPI (`pip install fbx` fails);
Autodesk ships it as a manual download built against specific Python versions,
so it breaks whenever you move Python. Of the pip-installable readers, `bpy`
(Blender as a module) is the only realistic one, but it is a ~1 GB dependency
and its FBX importer applies its own axis and unit conversion on the way in —
precisely the quantity this script is trying to measure. Reading the node
records directly is ~250 lines of plumbing (sections A and B), and removes both
problems.

## Why not the full 3×3 segment orientation?

Because two thirds of it is naming convention, not anatomy. The script measures
this every run and prints it:

* The **long axis** is consistent and meaningful in both rigs. Theia puts it on
  local −Z for every segment, MOVE4D on local +Y, both to four decimal places.
  The script *detects* this rather than assuming it, which is also how it gets a
  forearm axis without needing a wrist landmark (neither rig exposes a clean
  one). After alignment the long axes agree to **0.8–3.7°**. That is real
  anatomy, and it is what the fit uses.

* The **axial roll** about that axis is convention, and the two rigs do not
  share it. The per-segment residual offset `O_s = (R·R_m4d,s)ᵀ·R_theia,s` runs
  **87–117°**, and the offsets differ *between* segments by 5–27° within the
  legs and arms and by **161° between the left and right forearm** — MOVE4D
  rolls the forearm bone with pronation, Theia does not.

So `R_theia,s = G · R_m4d,s · O_s`, and from a single posture `G` and the `O_s`
are confounded. Fitting the full orientations means letting a quantity that
disagrees by 90° drive the estimate. Tested on this trial: the vertical-constrained
version lands 3° from the long-axis answer, and the unconstrained one diverges
completely (yaw +106°, tilt +91°).

**If you want the full orientations honestly, you need two or more distinct
postures.** Then the offsets cancel in the relative rotations,

```
R_theia,s(t₂) · R_theia,s(t₁)ᵀ  =  G · R_m4d,s(t₂) · R_m4d,s(t₁)ᵀ · Gᵀ
```

and `G` is identifiable without knowing any `O_s`. A single A-pose cannot do it.

> **Caveat.** `T[:3,:3] @ R_m4d` re-expresses a MOVE4D segment frame in Theia's
> *coordinate system*. It does not convert MOVE4D's bone convention into Theia's
> anatomical convention — that is the `O_s` problem above.

## Rotation model

`VERTICAL_ONLY = True` (default) constrains the rotation to the vertical. Both
systems get their vertical from the same physical gravity/lab calibration, so in
principle only the heading differs. The constrained fit has a closed form,

```
θ = atan2( Σ (u_y v_x − u_x v_y),  Σ (u_x v_x + u_y v_y) )
```

which is the right way to impose the constraint — fitting 3 DOF and then reading
the heading off the result is a different, worse estimator, because the free tilt
absorbs skeleton mismatch and drags the heading with it. Both models are
evaluated every run so you can see what the extra freedom buys.

## Result for D05_C1_apose

| | Theia | MOVE4D |
|---|---|---|
| nodes / segments | 53 / 35 | 78 / 76 |
| frames | 8 @ 40 fps, 6 kept | 4 @ 60 fps, 4 kept |
| up axis / units | Z / metres | Y / metres |

Theia frame 0 is a bind pose (475 mm from the median pose) and frame 7 a filter
edge transient (16 mm); both are dropped automatically.

```
heading (yaw)  −2.354°     translation  (−0.386, +0.227, +0.030) m

T = [ 0.999156   0.000000  -0.041074  -0.386519 ]
    [-0.041074   0.000000  -0.999156   0.227215 ]
    [ 0.000000   1.000000   0.000000   0.029752 ]
    [ 0.000000   0.000000   0.000000   1.000000 ]
```

| | segment orientation | joint position |
|---|---|---|
| vertical-axis only | **2.55° RMS** | **13.2 mm RMS** |
| full 3-DOF | 2.53° | 13.1 mm |

The 0.30° tilt the 3-DOF fit wants buys 0.1 mm, so it is absorbing skeleton
mismatch rather than a real calibration difference.

Segment length ratios (Theia/MOVE4D): thigh 0.94–0.95, shank 1.01–1.02, upper
arm 1.02–1.03. **That 5% disagreement is the floor on the residuals** — the
skeletons, not the alignment, are the limiting factor. Feet and trunk are worse
still (0.86 and 1.16), which is why nothing below the ankle or above the shoulder
is used.

## Checks the script runs every time

**Mirroring.** Kabsch forces a proper rotation by flipping the smallest singular
direction. The honest test is whether it had to: if the raw SVD product has a
negative determinant, the data are better explained by a mirror than by any
rotation, so one file is mirrored and no rigid transform can fix it.

**Units.** Median segment length ratio. A unit error would be a round number
(2.54, 10, 100, 1000); 1.015 is two skeletons of slightly different size.

**Bone-axis consistency.** Reported as the worst |dot| over the segments whose
two endpoints are both known. Anything below ~0.99 means that rig does *not* have
a consistent bone axis and step 5 needs rethinking for your rig version.

**End-to-end.** `T` is fitted in a canonicalised frame and rebuilt to act on raw
FBX coordinates, which drags in both files' up-axis matrices and unit scale
factors. The script pushes the raw MOVE4D joints through `T` and compares against
the raw Theia joints. This must reproduce the fitted residual; if it does not,
the bookkeeping is wrong and the matrix is unusable.

## Adapting it

`SEGMENTS`, `JOINTS` and `SEGMENT_ENDS` / `M4D_ENDS` are explicit tables in
CONFIG. If your MOVE4D rig version names joints differently, edit those and
nothing else.

`fbx_raw.py`, `fbx_scene.py`, `align_m4d_to_theia.py` and `plot_alignment.py`
are the original multi-module version, kept for reference. Nothing depends on
them.
