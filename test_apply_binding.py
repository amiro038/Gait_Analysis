"""Regression test for apply_binding.py.

Builds a synthetic binding and a synthetic Visual3D metrics export whose
poses are known exactly, then checks that

  * the poses come back out of the export to machine precision,
  * a frame with a gap survives as NaN instead of killing the run,
  * the fast batched rebuild in vertex_tracks() agrees BIT FOR BIT with the
    per-frame loop it replaced, on every selection path,
  * the HDF5 round trip is lossless at float32,
  * n_good counts a dropped frame as dropped.

Run it with `python test_apply_binding.py` from anywhere.
"""
import os, sys, json, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WORK = tempfile.mkdtemp(prefix="apply_binding_test_")
os.chdir(WORK)

NV, F = 10208, 600
SEGS = ["left_foot", "right_foot"]
LM = {"left_foot": ["Left_Ankle_Position", "Left_Toes_Position", "Left_Heel_Position"],
      "right_foot": ["Right_Ankle_Position", "Right_Toes_Position", "Right_Heel_Position"]}
rng = np.random.default_rng(1)

v_local = np.column_stack([rng.uniform(-.06,.20,NV), rng.uniform(-.05,.05,NV),
                           rng.uniform(-.03,.09,NV)])
seg_of = np.zeros(NV, dtype=np.int64); seg_of[NV//2:] = 1
lm_local = np.array([[[0,0,0],[.18,0,0],[-.05,0,.01]],
                     [[0,0,0],[.18,0,0],[-.05,0,.01]]], dtype=float)
meta = dict(segments=SEGS, static_metrics_csv="static.csv",
            coordinate_system="theia",
            pose_signal={"left_foot":"Left_Foot_Global_4x4",
                         "right_foot":"Right_Foot_Global_4x4"},
            landmark_names=LM,
            meshes=[dict(name="left_foot.obj", start=0, count=NV//2,
                         has_normals=False, lines=[], v_at=[], n_at=[]),
                    dict(name="right_foot.obj", start=NV//2, count=NV//2,
                         has_normals=False, lines=[], v_at=[], n_at=[])])
np.savez_compressed("binding.npz", vertices_local=v_local,
                    normals_local=np.zeros_like(v_local),
                    vertex_segment=seg_of, landmarks_local=lm_local,
                    meta=np.array(json.dumps(meta)))

# ground-truth poses, with one dropped frame
t = np.arange(F)/100.
TRUE = np.zeros((F,2,4,4)); TRUE[:,:,3,3]=1.
for k in range(2):
    ph=k*np.pi; a=np.radians(28.)*np.sin(2*np.pi*.9*t+ph)
    ca,sa=np.cos(a),np.sin(a)
    R=np.zeros((F,3,3)); R[:,0,0]=ca;R[:,0,2]=sa;R[:,1,1]=1;R[:,2,0]=-sa;R[:,2,2]=ca
    TRUE[:,k,:3,:3]=R
    TRUE[:,k,:3,3]=np.column_stack([.35*np.sin(2*np.pi*.9*t+ph),
                                    np.full(F,(-1)**k*.1), np.full(F,.06)])
DROP = 17
TRUE[DROP,0] = np.nan

# --- write a Visual3D-style metrics export ---------------------------------
sig_cols, vec_cols = [], []
for k,s in enumerate(SEGS):
    sig_cols.append((meta["pose_signal"][s], k))
    for j,n in enumerate(LM[s]):
        vec_cols.append((n, k, j))
names, comps, data = [], [], []
for nm,k in sig_cols:
    for c in range(16):
        names.append(nm); comps.append(str(c))
        data.append(TRUE[:,k].reshape(F,16)[:,c])
for nm,k,j in vec_cols:
    world = np.einsum('fab,b->fa', TRUE[:,k,:3,:3], lm_local[k,j]) + TRUE[:,k,:3,3]
    for c,ax in enumerate("XYZ"):
        names.append(nm); comps.append(ax); data.append(world[:,c])
D = np.column_stack(data)
with open("trial_metrics.csv","w",encoding="utf-8") as fh:
    for r in (["ITEM"]+names, ["x"]*(len(names)+1), ["x"]*(len(names)+1)):
        fh.write("\t".join(r)+"\n")
    fh.write("\t".join(["ITEM"]+names)+"\n")      # row index 3 unused
    fh.write("\t".join(["ITEM"]+comps)+"\n")      # row index 4 = components
    for i in range(F):
        fh.write("\t".join([str(i)]+["" if not np.isfinite(x) else "%.17g" % x
                                     for x in D[i]])+"\n")
# fix: read_metrics uses rows[1] for names, rows[4] for comps
lines = open("trial_metrics.csv",encoding="utf-8").read().split("\n")
lines[1] = "\t".join(["ITEM"]+names)
open("trial_metrics.csv","w",encoding="utf-8").write("\n".join(lines))

import apply_binding as ab
ab.BINDING_FILE = os.path.join(WORK, "binding.npz")
ab.TRIAL_METRICS_CSV = os.path.join(WORK,"trial_metrics.csv")
ab.SHOW_3D_VIEWER = ab.SAVE_3D_FIGURE = ab.SAVE_3D_VIDEO = False

res = ab.apply_binding("trial_metrics.csv", verbose=False)
poses = res["poses"]

ok = np.isfinite(TRUE).all(axis=(2,3))
print(f"poses recovered: max err {np.nanmax(np.abs(poses[ok]-TRUE[ok])):.2e}")
assert np.nanmax(np.abs(poses[ok]-TRUE[ok])) < 1e-9
assert not np.isfinite(poses[DROP,0]).all(), "dropped frame should stay NaN"
ng = [x["n_good"] for x in res["stats"]]
print(f"n_good (one frame dropped from left_foot): {ng}  expected [{F-1}, {F}]")
assert ng == [F-1, F], ng

# --- reference: the ORIGINAL vertex_tracks loop ----------------------------
def original(poses, v_local, seg_of, idx):
    Vl, seg = v_local[idx], seg_of[idx]
    out = np.full((len(poses), len(idx), 3), np.nan)
    for k in range(poses.shape[1]):
        m = seg == k
        if not m.any(): continue
        for i,P in enumerate(poses[:,k]):
            if np.isfinite(P).all():
                out[i,m] = Vl[m] @ P[:3,:3].T + P[:3,3]
    return out

for label, kw in (("all vertices", {}),
                  ("mesh='left'",  dict(mesh="left")),
                  ("vertices=[0,1500,9000,10207]", dict(vertices=[0,1500,9000,10207]))):
    V, idx = ab.vertex_tracks(res, **kw)
    ref = original(poses, v_local, seg_of, idx)
    bad = np.nanmax(np.abs(V-ref)) if np.isfinite(ref).any() else 0.0
    same_nan = np.array_equal(np.isnan(V), np.isnan(ref))
    print(f"  {label:<32} shape {str(V.shape):<18} max|new-orig| {bad:.2e}  NaN pattern {'match' if same_nan else 'DIFFER'}")
    assert bad < 1e-12 and same_nan

V32, _ = ab.vertex_tracks(res, mesh="left", dtype=np.float32)
print(f"  float32 path                     dtype {V32.dtype}, "
      f"max err vs float64 {np.nanmax(np.abs(V32-ab.vertex_tracks(res, mesh='left')[0]))*1e3:.2e} mm")

Vf = ab.vertices_at_frame(np.load("binding.npz"), poses, 100)
assert np.nanmax(np.abs(Vf - original(poses, v_local, seg_of, np.arange(NV))[100])) < 1e-12
print("  vertices_at_frame                matches original")

# --- HDF5 round trip --------------------------------------------------------
p = ab.save_mesh_h5(res, trial_csv=os.path.join(WORK,"trial_metrics.csv"), verbose=True)
full = ab.load_mesh_h5(os.path.join(WORK,"trial_metrics.csv"))
ref_all = original(poses, v_local, seg_of, np.arange(NV)).astype(np.float32)
print(f"  HDF5 round trip                  max err {np.nanmax(np.abs(full-ref_all)):.2e} m, "
      f"NaN {'match' if np.array_equal(np.isnan(full),np.isnan(ref_all)) else 'DIFFER'}")
assert np.nanmax(np.abs(full-ref_all)) == 0.0
one = ab.load_mesh_h5(os.path.join(WORK,"trial_metrics.csv"), vertices=[9000])
assert np.nanmax(np.abs(one[:,0]-ref_all[:,9000])) == 0.0
print("  HDF5 single-vertex partial read  matches")

df = ab.mesh_dataframe(res)
print(f"  mesh_dataframe                   {df.shape}, dtype {df.dtypes.iloc[0]}, "
      f"attrs units={df.attrs['units']}")

print("\n  sizes:")
for f in ("trial_posed.npz","trial_metrics_mesh.h5","trial_metrics.csv"):
    if os.path.exists(f): print(f"    {f:<26}{os.path.getsize(f)/1e6:>9.3f} MB")

r2 = ab.main(["trial_metrics.csv","--no-plot","--h5"])
print("\n  main() with --no-plot --h5: OK")
print(f"\n  workdir: {WORK}")
print("\nALL CHECKS PASSED")
