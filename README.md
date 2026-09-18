# MOVE4D scan meshes on the Theia3D skeleton

Two scripts. The first turns segmented MOVE4D foot meshes into a binding
against the Theia foot segments; the second replays that binding over any
trial to get vertex positions.

```bash
python build_foot_binding.py                       # -> foot_mesh_binding.npz
python apply_binding.py TRIAL_metrics.csv         # -> TRIAL_posed.npz + PNG + movie
```

| file | what it does |
|---|---|
| **`build_foot_binding.py`** | steps 1–3: solve the MOVE4D→Theia transform, put the meshes in Theia coordinates, bind them to the foot segments |
| **`apply_binding.py`** | replays a binding over any trial; shares no imports with the builder, so the two can live apart |
| `export_posed_fbx.py` | optional: animated FBX for Maya/Blender. Needs `pip install bpy` |
| `archive/` | the earlier multi-module versions, superseded and folded into the two above |

Only `numpy` is required. `matplotlib` for the check figures. No FBX SDK.

## `build_foot_binding.py` — the three steps

1. **Solve the MOVE4D → Theia3D transform** from a static A-pose recorded by
   both systems, by aligning the global orientations of the thigh, shank,
   upper-arm and forearm segments.
2. **Put the segmented `.obj` meshes into Theia coordinates.**
3. **Bind them to the Theia foot segments** using a Visual3D metrics export of
   that same static pose.

**Step 1 is a per-session calibration.** The two lab frames do not move between
trials, so it is solved once, cached in `T_m4d_to_theia.npy` and reused. Pass
`--resolve` to redo it, or drop a known matrix in `T_MATRIX` to skip it
entirely.

Everything editable — paths, flags and every landmark table — is in the one
CONFIG cell at the top.

## Step 1, in detail

Internally it runs six stages:

1. **load** both FBX files
2. **read** the kinematic data for every segment
3. **local → global** via the full Autodesk transform chain
4. **common units and coordinate system** — metres, up-axis rotated onto +Z
5. **rotation** — global segment orientations as close as possible
6. **translation** — joint positions as close as possible

```python
# the helpers are there if you want T on its own
pts_theia = apply_transform(T, pts_m4d)          # (..., 3)
R_theia   = transform_orientation(T, R_m4d)      # (..., 3, 3)
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

## Applying the transform to meshes

Step 2 takes Wavefront `.obj` files in raw MOVE4D coordinates — parts segmented out of the scan, for instance — and writes them
in Theia coordinates.

Vertices get `T`; vertex normals get the inverse-transpose of its linear block,
renormalised. Faces, groups, comments and material references pass through byte
for byte, so topology survives.

**The output is Z-up.** MOVE4D files are Y-up and Theia files are Z-up, and that
conversion is part of `T`. Open the result in a Y-up viewer and the mesh will
appear to lie on its side — that is what "in Theia's coordinate system" means.

Two checks run every time:

- **Input frame.** The mesh must sit inside the MOVE4D scan's own bounding box.
  Re-centring, a switch to millimetres or an axis flip are all caught here rather
  than producing a plausible-looking wrong answer.
- **Floor.** MOVE4D's floor is z = 0 in its own file, so it lands at `T[2,3]` in
  Theia coordinates, and a foot's sole should be right there. On the supplied
  feet the soles land at **+28.2 mm against an expected +29.8 mm**.

It also writes `obj_transform_check.png`: the transformed mesh against Theia's
own body model and joint centres, clipped to the region of interest. On the
supplied feet, Theia's ankle joint centres fall **inside** the transformed scan
surface, ~41 mm from the nearest skin vertex, and the toe joints ~26 mm inside —
which is where joint centres belong.

## Driving the meshes from a trial

Two scripts, run in order. The first builds the mesh-to-bone relationship
once; the second replays it over any trial.

```bash
python build_foot_binding.py                       # -> foot_mesh_binding.npz
python apply_binding.py TRIAL_metrics.csv          # -> TRIAL_posed.npz
python apply_binding.py TRIAL_metrics.csv --vertices --obj
```

### What drives it

**`<Side>_Foot_Global_4x4`**, when the export carries it. That is the segment
pose itself: a row-major 4×4 whose translation is the ankle joint centre and
whose rotation block is the foot segment's global orientation. It is used
directly — nothing fitted, nothing reconstructed. Verified on D05_C01:

- its translation equals `<Side>_Ankle_Position` to **0.000 mm**;
- its rotation block matches the segment orientation in Theia's own FBX to
  **0.01° mean, 0.03° max**;
- every other foot landmark sits at a **fixed** position in its frame to
  **±0.01 mm** across all 484 frames of a movement trial.

Without it, the script falls back to fitting a pose from three landmarks
(ankle, foot COM, heel) by Kabsch. That works — three non-collinear points
determine a rigid pose — but it is an estimate, and 1 mm of landmark error
becomes about 2.3° of inversion/eversion. `Foot_Position` sits 25.15 mm off the
ankle→toes line, and that off-axis lever arm is what makes the fallback
possible at all; the script refuses to proceed silently if the landmarks you
give it turn out collinear.

**Do not put the toes in the fallback set.** `<Side>_Toes_Position` is the toes
segment's origin and is *not* rigid to the foot: between the A-pose and the SKS
trial it moves 29 mm in the foot's own frame on the left side. It would drag
the fitted foot pose with it. Use the heel instead — it holds to 0.1 mm across
both trials.

When both a 4×4 and landmarks are present, `apply_binding.py` fits the
landmarks anyway and reports the disagreement. On the SKS trial the two
independent routes agree to **0.006° and 0.003 mm**.

### The relationship that gets stored

`foot_mesh_binding.npz` holds every vertex, and each segment's landmarks, in
that segment's local frame. Replaying is a Kabsch fit of the stored landmarks
onto the trial's landmarks, once per frame — least squares over all of them, so
extra landmarks improve the result rather than being ignored.

Segment assignment is **per vertex**, not per file, by proximity to each
segment's landmarks. That is what makes `both_feet.obj` work: its two halves
bind to different segments and move independently. Verified at 10208/10208
vertices correct, including the 4 vertices where `both_feet`'s internal
ordering crosses sides.

`apply_binding.py` always writes `poses` (frames, segments, 4, 4), which is tiny
and is the whole result — vertices are one matrix multiply away. `--vertices`
stores the (frames, vertices, 3) array and `--obj` writes one .obj per frame;
both get large fast.

#### Getting vertex positions for analysis

The vertices are **not** in the `_posed.npz` by default. Reconstruct the ones
you need with `vertex_tracks()` rather than storing all of them:

```python
import apply_binding as ab

V, idx = ab.vertex_tracks("TRIAL_posed.npz")                 # all 10208
V, idx = ab.vertex_tracks("TRIAL_posed.npz", mesh="left")    # one foot
V, idx = ab.vertex_tracks(res, vertices=[0, 1500, 9000])     # three vertices
```

`V` is `(frames, len(idx), 3)` in **Theia world metres** — the same frame and
units as the joint positions in the export, so it drops straight into an
analysis that already works on those. `idx` carries the global vertex index of
each column, so a selection means the same thing across trials. Frames whose
pose was missing come back as `NaN` rather than being dropped, so the frame
axis still lines up with the export row for row.

`source` is either a `_posed.npz` path or the dict `apply_binding()` returned.
Sizing, for a 484-frame trial: all 10208 vertices is 50 MB, one foot 12 MB, a
single vertex 12 kB. The `poses`-only npz that produces all of them is 0.7 MB.

### Checking it worked

`apply_binding.py --plot` draws the whole thing in 3D: every joint centre and
segment origin from the export, the skeleton linking them, the posed mesh on
top, and each driven segment's own axes as arrows. If the mesh sits on the
skeleton's feet and its axes follow the ankle, the binding is right.

Pressing **Run** in Spyder passes no command-line arguments, so the flags
below never fire that way. The CONFIG block is what decides in that case, and
out of the box both switches are on — run the file and you get a figure:

```python
SHOW_3D_VIEWER = True        # open the interactive frame-slider window
SAVE_3D_FIGURE = True        # also write a PNG grid next to the output .npz
PLOT_FRAMES = None           # frames for the PNG; None = 6 across the trial
PLOT_ZOOM = "body"           # or "feet"
```

From a terminal the flags override CONFIG:

```bash
python apply_binding.py TRIAL_metrics.csv --plot                  # slider viewer
python apply_binding.py TRIAL_metrics.csv --plot-frames 0,120,240 # static PNG
python apply_binding.py TRIAL_metrics.csv --plot-frames 0,120 --feet
python apply_binding.py TRIAL_metrics.csv --no-plot               # neither
```

`--plot` needs an interactive backend (in Spyder: *Preferences → IPython
console → Graphics → Backend: Automatic*); on the inline or Agg backend the
script says so instead of opening a slider that cannot move. `--plot-frames`
writes a PNG and works headless. `--feet` crops to the bound meshes, which is
where you can actually judge the fit — at that zoom the ankle, toe and heel
centres should sit *inside* the mesh surface.

#### The movie

A still only says the binding was right *in that pose*. A mesh that swims
against the foot, lags the skeleton, or flips at midstance looks fine frame by
frame and obvious in motion, so the run also writes a video of the whole trial
by default:

```python
SAVE_3D_VIDEO = True
VIDEO_STRIDE  = 2        # render every Nth frame
VIDEO_FPS     = 25
VIDEO_FORMAT  = "auto"   # mp4 if ffmpeg is there, else gif
VIDEO_ZOOM    = None     # None follows PLOT_ZOOM; "feet" crops to the mesh
VIDEO_FOLLOW  = False    # re-centre on the mesh each frame (overground trials)
VIDEO_SPIN_DEG = 0.0     # total camera rotation across the clip
```

```bash
python apply_binding.py TRIAL_metrics.csv --video          # movie only
python apply_binding.py TRIAL_metrics.csv --video --feet
python apply_binding.py TRIAL_metrics.csv --no-video
```

**mp4 needs ffmpeg**, which most Windows Python installs do not have. `pip
install imageio-ffmpeg` is the easiest fix — the script finds that binary on
its own, no system install and no PATH editing. Without it the writer falls
back to **gif** via Pillow, which always works but makes much larger files for
the same clip. `VIDEO_FORMAT = "mp4"` turns the fallback off and raises
instead, if you would rather know.

A 484-frame trial at `VIDEO_STRIDE = 2` renders in about 25 s and gives a 10 s
clip. Raise the stride for a quicker look; `VIDEO_FOLLOW = True` keeps the feet
in shot on a trial where the subject travels, which a fixed box cannot do.

**This is the check to trust.** It reads the same arrays the rest of the
script works with, so nothing can be lost in a file format on the way out —
unlike the FBX route, where two separate convention bugs produced
plausible-looking wrong output before being caught.

### Watching it move

`export_posed_fbx.py` writes an animated FBX: each foot mesh as a rigid object
driven by its segment pose, plus small cubes at the lower-limb joint centres
for context. It needs `pip install bpy` (~1 GB); the binding and replay scripts
do not.

```bash
python export_posed_fbx.py TRIAL_metrics.csv     # -> TRIAL_posed.fbx
```

Two FBX traps, both of which produce plausible-looking wrong output:

**Match Theia's axis system, not just its up axis.** An FBX header declares the
front and right axes as well as up, and a reader that honours them (Maya does)
rotates each file's contents to reconcile its declared system with the scene.
Theia writes `Up=+Z, Front=+Y, Coord=−X`. Blender's native system is
`Up=+Z, Front=−Y, Coord=+X` — same up axis, **opposite horizontal orientation,
which is a 180° rotation about the vertical**. Export naively and the numbers
are right but the header says they mean something spun around, so Maya lands
the meshes 180° from the skeleton. The fix is to rotate the scene 180° about Z
on the way in and ask for `axis_forward='-Y'`, so Blender's export-time
conversion cancels the pre-rotation: the file then declares Theia's system
*and* stores Theia's coordinates unchanged. `FBX_SCALE_UNITS` also makes it
write metres (`UnitScaleFactor=100`) like Theia, rather than Blender's default
centimetres.

**Key from frame 0, not frame 1.** Blender's FBX exporter writes keyframe times
relative to time zero and its importer maps FBX time 0 back onto frame 1, so
keying from frame 1 lands every key one frame late. On this trial that was a
silent **12.8 mm** error. Keying from frame 0 round-trips to **0.00005 mm**.

A Blender round trip cannot catch the axis problem, because Blender's importer
applies the inverse of its own convention and the error cancels. Verify by
reading the raw FBX with a reader that does not convert, and by loading your
file and Theia's into one scene and comparing.

A mesh spanning more than one segment (`both_feet.obj`) is skipped — it cannot
be one rigid object, and `left_feet` + `right_feet` already cover it.

### Verified

- Round trip is **exactly 0.00 mm** when replayed at the reference frame.
- The exported FBX, read back with our own non-converting FBX reader and
  evaluated vertex by vertex against the computed positions, agrees to
  **0.000 mm** across the trial (a 180°-rotated version would be ~300 mm off).
- Its header matches Theia's exactly: `Up=+Z Front=+Y Coord=−X Unit=100`.
- Loaded into the same scene as Theia's own FBX, the exported joint cubes sit
  **0.6–2.6 mm** from the matching Theia bones — that residual being the known
  2–3 mm difference between Visual3D's exported positions and Theia's FBX
  joints. A 180° error would show as 578 mm.
- Replaying at neighbouring static frames gives 2–7 mm of genuine
  frame-to-frame segment motion, and 20/40 mm at the two filter edge
  transients the FBX analysis flagged independently. The binding script drops
  those from the reference pose automatically.

### What it cannot do

Each region is rigid. The arch does not flatten and the toes do not flex,
because Theia exports no toe orientation to drive them with. **27% of the left
mesh and 22% of the right sit distal to the MTP joint** and are held rigid
relative to the rest. Fine for skin position in the lab frame; not fine for toe
kinematics. If you need that, export more landmarks from Visual3D and add a
toes segment — the landmark lists are one line of config.

## Adapting it

`ALIGN_SEGMENTS`, `ALIGN_JOINTS`, `ALIGN_ML_AXES` and `ALIGN_SEGMENT_ENDS` /
`ALIGN_M4D_ENDS` are explicit tables in `build_foot_binding.py`'s CONFIG. If your MOVE4D rig version names joints differently, edit those
and nothing else. `USE_ML_AXES`, `ASSUME_SHARED_FLOOR`, `CHECK_FLOOR_FROM_MESH`
and `VERTICAL_ONLY` are the switches.

`VERTICAL_ONLY` and `ASSUME_SHARED_FLOOR` are independent and mean different
things: the first constrains the *rotation* to be about the vertical, the second
locks the *translation's* vertical component to zero.

`archive/` holds the earlier multi-module versions — `m4d_to_theia_transform.py`,
`transform_obj_to_theia.py`, `bind_mesh_to_bones.py` and the original four-file
pipeline before them. They are superseded, nothing depends on them, and
`build_foot_binding.py` reproduces their output bit for bit. Kept only so the
history is inspectable without digging through git.
