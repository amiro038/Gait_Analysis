# MOVE4D → Theia3D rigid alignment

Finds the 4×4 transform `T` such that `p_theia = T @ [p_move4d; 1]`, by aligning
the **global orientations of the thigh, shank, upper-arm and forearm segments**
(the hip, knee, shoulder and elbow segments) from a static standing trial
recorded simultaneously by both systems.

Everything is in **`m4d_to_theia_transform.py`** — one self-contained script.
Open it in Spyder, set the two paths in CONFIG, press F5. Only `numpy` is
required (`matplotlib` for the check figure).

## The six steps

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

## What the rotation is fitted to

**Eight segment long axes.** The long axis is the part of a segment frame both
rigs define anatomically rather than by convention. The script *detects* each
rig's bone axis rather than assuming it, by expressing the proximal→distal
direction in each segment's own frame and voting: Theia uses local **−Z**,
MOVE4D local **+Y**, both to four decimal places. That also means the long axis
can be read off any segment's orientation matrix, so the forearm needs no wrist
landmark (neither rig exposes a clean one).

**Four bilateral medio-lateral axes** — right→left at hip, knee, shoulder and
elbow. Not segment orientations, but the best *heading* information a standing
pose contains. Only the **horizontal** component of a direction constrains the
heading, and in a standing pose the limb long axes are nearly vertical:

```
l_thigh 0.154   r_thigh 0.163   l_shank 0.166   r_shank 0.123
l_uarm  0.578   r_uarm  0.608   l_larm  0.616   r_larm  0.652
hip_width 1.000  knee_width 1.000  shoulder_width 0.999  elbow_width 1.000
```

A 1° error in a thigh axis becomes ~6.7° of heading error. A right→left vector
is horizontal by construction, and a joint-centre offset that is *symmetric*
between sides — the usual case — does not rotate it at all. Adding these four
cuts the leave-one-out heading spread from **2.34° to 0.56°** while moving the
answer itself by 0.02°.

No weighting is needed: a plain Kabsch already weights each observation's
heading contribution by its squared horizontal component, so the ML axes outvote
the long axes ~7:1 for yaw and not at all for tilt, which is exactly right.

## The vertical offset

The two coordinate systems **are** offset vertically, so `t_z` is fitted like any
other component: **+29.8 mm**. Set `ASSUME_SHARED_FLOOR = True` only if you know
both systems put z = 0 on the same floor plane; it then locks `t_z = 0` and
reports what the joint centres implied instead of burying it in the matrix.

Know what a fitted `t_z` contains: the true offset between the two calibrations
**plus** any systematic vertical bias between the two skeletons' joint-centre
definitions. One static pose cannot separate them.

The script gives you two independent reads on whether the offset is real.

**The floor, from the body surface** — because a joint centre is an inference and
the sole of a foot is not. MOVE4D ships the real scan (49,530 vertices, skinned
to the rig; the script does the linear blend skinning). Theia ships its body
model as rigid meshes parented to the segments.

```
Theia model   34054 verts | lowest  +43.7 mm | top 1750.0 mm |   0 within 5 mm of z=0
MOVE4D scan   49530 verts | lowest   -5.4 mm | top 1769.6 mm | 757 within 5 mm of z=0
```

The scanned subject is standing **on** the floor, not floating, so MOVE4D's
origin is on the floor. A +29.8 mm offset then puts Theia's floor at z = +29.8 mm
in Theia coordinates — its origin that far *below* the floor — and Theia's lowest
model vertex clears that floor by **13.9 mm**, against 43.7 mm if the floors were
assumed shared. A ~270-vertex foot primitive clearing by 14 mm is far more
plausible than one clearing by 44 mm, so the mesh independently supports a real
vertical offset. It is corroboration, not proof: Theia's meshes are a model.

**The residual signs.** Suppress a real offset and every vertical residual comes
out the same sign; absorb it correctly and they scatter about zero with only
genuine definition differences left. The report checks this automatically —
forcing `ASSUME_SHARED_FLOOR = True` on this dataset makes all eight residuals
positive and the script says so.

With `t_z` fitted, the vertical residuals are −7 to −2 mm at the hips, shoulders
and elbows and **+13 to +15 mm at the knees** — a real knee joint-centre
definition difference, no longer masked by an offset.

## Why not the full 3×3 segment orientation?

Because two thirds of it is naming convention. The script measures it every run:
the per-segment residual offset `O_s = (R·R_m4d,s)ᵀ·R_theia,s` runs **87–117°**,
and the offsets differ *between* segments by 47° on average and **161° between
the left and right forearm** — MOVE4D rolls the forearm bone with pronation,
Theia does not.

So `R_theia,s = G · R_m4d,s · O_s`, and from a single posture `G` and the `O_s`
are confounded. Tested: a shared-offset fit lands 3° from the long-axis answer,
and unconstrained it diverges completely (yaw +106°, tilt +91°).

**With two or more distinct postures the offsets cancel**, in

```
R_theia,s(t₂) · R_theia,s(t₁)ᵀ  =  G · R_m4d,s(t₂) · R_m4d,s(t₁)ᵀ · Gᵀ
```

and `G` becomes identifiable without knowing any `O_s`. A single A-pose cannot.

> **Caveat.** `T[:3,:3] @ R_m4d` re-expresses a MOVE4D segment frame in Theia's
> *coordinate system*. It does not convert MOVE4D's bone convention into Theia's
> anatomical convention — that is the `O_s` problem above.

## Why not surface ICP?

Tested and rejected. Nearest-neighbour distance from Theia's model surface to
the MOVE4D scan, after the current transform:

| region | median | region | median |
|---|---|---|---|
| pelvis | 59 mm | head | 15 mm |
| thigh | 50–55 mm | hands | 12–18 mm |
| humerus | 37 mm | toes | 19–20 mm |
| **all** | **22 mm** | | |

Theia's limb meshes are 270-vertex primitives — tapered cylinders, not a real
thigh. ICP would fit that shape-model difference, not the coordinate transform.
Worth revisiting only if Theia can export a measured surface.

## How well is the heading determined?

The residual cannot tell you. Refit on independent subsets and see how far apart
they land — that spread *is* the uncertainty.

| subset | yaw | leverage | n |
|---|---|---|---|
| segment long axes | −2.35° | 0.383 | 8 |
| bilateral ML axes | −2.33° | 1.000 | 4 |
| legs | −3.02° | 0.434 | 6 |
| arms | −1.93° | 0.742 | 6 |
| left side | −1.23° | 0.378 | 4 |
| right side | −3.39° | 0.387 | 4 |

Subset spread **2.17°**, leave-one-out spread **0.56°**. The two say different
things: leave-one-out being small means no single observation drives the answer;
the subset spread being larger means independent *parts of the body* disagree,
which averaging cannot remove. Quote the larger. At 2.17°, a point 1 m from the
origin moves 38 mm, and 3 m from the origin 113 mm.

Caveat in both directions: a subset with low leverage amplifies its own residual
by 1/leverage, so some of that spread is noise rather than bias — which is
exactly why **more than one standing pose, at different headings, is worth
capturing**. That single change would both shrink this number and make the full
orientations usable.

## Result for D05_C1_apose

```
heading (yaw)  −2.337°     translation  (−0.386, +0.227, +0.030) m

T = [ 0.999169   0.000000  -0.040771  -0.386516 ]
    [-0.040771   0.000000  -0.999169   0.227162 ]
    [ 0.000000   1.000000   0.000000   0.029752 ]
    [ 0.000000   0.000000   0.000000   1.000000 ]
```

| | direction residual | horizontal position | vertical | total |
|---|---|---|---|---|
| vertical-axis only | **2.16° RMS** | **10.2 mm RMS** | 8.5 mm | 13.2 mm |
| full 3-DOF | 2.14° | 10.1 mm | | |

The 0.31° tilt the 3-DOF fit wants buys 0.1 mm, so it is absorbing skeleton
mismatch rather than a real calibration difference.

Theia frame 0 is a bind pose (475 mm from the median pose) and frame 7 a filter
edge transient (16 mm); both are dropped automatically.

Segment length ratios (Theia/MOVE4D): thigh 0.94–0.95, shank 1.01–1.02, upper
arm 1.02–1.03 — in *opposite* directions, so it is joint-centre placement, not
scale. **That disagreement is the floor on the residuals**; the skeletons, not
the alignment, are the limiting factor. Feet and trunk are worse still (0.86 and
1.16), which is why nothing below the ankle or above the shoulder is used.

## Checks the script runs every time

**Mirroring.** Kabsch forces a proper rotation by flipping the smallest singular
direction. The honest test is whether it had to: a negative determinant on the
raw SVD product means the data are better explained by a mirror than by any
rotation, so one file is mirrored and no rigid transform can fix it.

**Units.** Median segment length ratio. A unit error would be a round number
(2.54, 10, 100, 1000); 1.015 is two skeletons of slightly different size.

**Bone-axis consistency.** The worst |dot| over the segments whose two endpoints
are both known. Below ~0.99 means that rig does *not* have a consistent bone
axis and step 5 needs rethinking for your rig version.

**Floor.** As above, from the surface, plus the residual-sign test.

**End-to-end.** `T` is fitted in a canonicalised frame and rebuilt to act on raw
FBX coordinates, which drags in both files' up-axis matrices and unit scale
factors. The script pushes the raw MOVE4D joints through `T` and compares
against the raw Theia joints. This must reproduce the fitted residual; if it
does not, the bookkeeping is wrong and the matrix is unusable.

## What would improve this further

Ordered by value, and none of it is an algorithm change:

1. **A shared physical reference** — a rigid object visible to both systems
   turns this from a biomechanics problem into a geometry problem, and the
   answer stops depending on skeleton definitions at all.
2. **Two or three static poses at different headings.** Azimuthal diversity is
   currently ~0.007: every horizontal direction in an A-pose is medio-lateral,
   so a bias shared by all of them is invisible. This also makes the full
   segment orientations usable (see above).
3. **Pool a dynamic trial — after verifying sync.** Limb swing supplies the
   antero-posterior information an A-pose lacks, but at 1.3 m/s a segment hits
   ~300°/s, so 10 ms of sync error is 3° of orientation error, larger than the
   entire current residual. `..._merged_events.csv` has heel strikes from both
   `kinematic` and `GRF` sources at 100 Hz — align those first.
4. **Confirm how MOVE4D's skeleton is produced.** `Chest`/`Chest2`/`Chest3`/
   `Chest4`, `LeftCollar`, a `LeftForearm` twist joint, 50 finger joints,
   `End_*` tips — that naming is a character-animation rig, and the 161° roll
   disagreement between left and right forearm is what an auto-rigger produces.
   If `LeftHip` is a rig joint fitted to the mesh rather than an estimated hip
   joint centre, that is the dominant error source.
5. **Validate on held-out data.** Fit on one capture, evaluate on another, plot
   residual against frame. Flat noise → one constant matrix is valid. Drift →
   sync error or a position-dependent calibration difference.

## Adapting it

`SEGMENTS`, `JOINTS`, `ML_AXES` and `SEGMENT_ENDS` / `M4D_ENDS` are explicit
tables in CONFIG. If your MOVE4D rig version names joints differently, edit those
and nothing else. `USE_ML_AXES`, `ASSUME_SHARED_FLOOR`, `CHECK_FLOOR_FROM_MESH`
and `VERTICAL_ONLY` are the switches.

`VERTICAL_ONLY` and `ASSUME_SHARED_FLOOR` are independent and mean different
things: the first constrains the *rotation* to be about the vertical, the second
locks the *translation's* vertical component to zero.

`fbx_raw.py`, `fbx_scene.py`, `align_m4d_to_theia.py` and `plot_alignment.py`
are the original multi-module version, kept for reference. Nothing depends on
them.
