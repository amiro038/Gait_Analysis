#!/usr/bin/env python3
"""
Rigid alignment of MOVE4D motion-capture data into the Theia3D global frame.

Strategy
--------
1.  Read both FBX files, evaluate the full scene graph to global joint
    transforms per frame.
2.  Verify units and handedness empirically; canonicalise the declared
    up-axis of each file to Z-up.
3.  Estimate ROTATION from unit direction vectors between corresponding
    joint pairs (Wahba / Kabsch with robust IRLS re-weighting).
    Direction vectors are translation-invariant, so R is decoupled from t
    and is immune to joint-centre definition offsets to first order.
4.  Estimate TRANSLATION by robust point matching given R.
5.  Diagnose: scale, yaw/tilt decomposition, yaw observability,
    leave-one-segment-out stability, per-joint residuals.

Output is a 4x4 matrix T such that   p_theia = T @ [p_move4d; 1]
with p_move4d taken directly from the MOVE4D FBX (no pre-conversion).

Usage
-----
    python align_m4d_to_theia.py THEIA.fbx M4D.fbx [-o outdir] [--yaw-only]
    python align_m4d_to_theia.py --pairs trials.txt        # pool trials

Pooling several trials of the SAME session (especially dynamic ones)
is strongly recommended over a single static pose -- see README.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import fbx_scene

# ----------------------------------------------------------------------
# Joint correspondence
# ----------------------------------------------------------------------
# Theia3D name (without the 'person_N:' namespace) -> MOVE4D name.
# Only joints whose anatomical definition is comparable between the two
# skeletons are listed.
JOINT_MAP = {
    "pelvis":  "Hips",
    "l_thigh": "LeftHip",    "r_thigh": "RightHip",
    "l_shank": "LeftKnee",   "r_shank": "RightKnee",
    "l_foot":  "LeftAnkle",  "r_foot":  "RightAnkle",
    "l_toes":  "LeftToe",    "r_toes":  "RightToe",
    "thorax":  "Chest4",
    "l_uarm":  "LeftShoulder",  "r_uarm":  "RightShoulder",
    "l_larm":  "LeftElbow",     "r_larm":  "RightElbow",
    "head":    "Head",
}

# Direction vectors used for the rotation fit:
#   (label, from_joint, to_joint, weight)
# Weights encode how consistently the two systems define that direction.
# Long limb segments and inter-joint width vectors are reliable; the foot
# and anything involving the hand or head is convention-heavy.
DIRECTION_VECTORS = [
    ("l_thigh",     "l_thigh", "l_shank", 1.0),
    ("r_thigh",     "r_thigh", "r_shank", 1.0),
    ("l_shank",     "l_shank", "l_foot",  1.0),
    ("r_shank",     "r_shank", "r_foot",  1.0),
    ("l_uarm",      "l_uarm",  "l_larm",  1.0),
    ("r_uarm",      "r_uarm",  "r_larm",  1.0),
    ("pelvis_ML",   "r_thigh", "l_thigh", 1.0),   # horizontal, pins yaw
    ("shoulder_ML", "r_uarm",  "l_uarm",  1.0),   # horizontal, pins yaw
    ("l_foot_AP",   "l_foot",  "l_toes",  0.5),   # anterior: the only
    ("r_foot_AP",   "r_foot",  "r_toes",  0.5),   # antero-posterior information
]
# Deliberately NOT used: pelvis->thorax.  Theia's 'thorax' sits at shoulder
# level while MOVE4D's 'Chest4' is well below it; the measured length ratio
# (~1.15) confirms they are different landmarks, so the direction between
# them is not a like-for-like comparison.

# Joints used for the translation fit.  Restricted to true anatomical joint
# CENTRES, which both pipelines estimate with comparable meaning.  Segment
# origins that are a rigging convention (Theia 'pelvis', MOVE4D 'Hips',
# 'Chest4') are excluded from the fit and reported as diagnostics instead.
TRANSLATION_JOINTS = {
    "l_thigh": 1.0, "r_thigh": 1.0,   # hip centres
    "l_shank": 1.0, "r_shank": 1.0,   # knee centres
    "l_foot":  1.0, "r_foot":  1.0,   # ankle centres
    "l_uarm":  1.0, "r_uarm":  1.0,   # shoulder centres
    "l_larm":  0.8, "r_larm":  0.8,   # elbow centres
}
DIAGNOSTIC_JOINTS = ["pelvis", "thorax", "head", "l_toes", "r_toes"]

# Segments whose length ratio is used for the scale / unit check.
SCALE_SEGMENTS = [
    ("l_thigh", "l_thigh", "l_shank"), ("r_thigh", "r_thigh", "r_shank"),
    ("l_shank", "l_shank", "l_foot"),  ("r_shank", "r_shank", "r_foot"),
    ("l_uarm",  "l_uarm",  "l_larm"),  ("r_uarm",  "r_uarm",  "r_larm"),
    ("l_foot",  "l_foot",  "l_toes"),  ("r_foot",  "r_foot",  "r_toes"),
    ("pelvis_w", "r_thigh", "l_thigh"),
    ("shoulder_w", "r_uarm", "l_uarm"),
    ("trunk",   "pelvis",  "thorax"),
]

STATIC_MOTION_THRESHOLD = 0.02   # m; below this a trial is treated as static
N_BOOTSTRAP = 400


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _normalise(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-12, v / n, 0.0)


def up_axis_to_z_matrix(up_axis: int, up_sign: float) -> np.ndarray:
    """Right-handed 3x3 rotation taking the file's up-axis onto +Z."""
    s = 1.0 if up_sign >= 0 else -1.0
    if up_axis == 2 and s > 0:                       # already Z-up
        return np.eye(3)
    if up_axis == 2:                                 # -Z up
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    if up_axis == 1 and s > 0:                       # Y-up  -> Rx(+90)
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    if up_axis == 1:                                 # -Y up
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    if up_axis == 0 and s > 0:                       # X-up  -> Ry(-90)
        return np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)
    return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1, 1))))


def decompose_yaw_tilt(R: np.ndarray):
    """Split R into a yaw about +Z and the residual tilt of the vertical."""
    z_rot = R @ np.array([0.0, 0.0, 1.0])
    tilt = float(np.degrees(np.arccos(np.clip(z_rot[2], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return yaw, tilt


def yaw_matrix(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# ----------------------------------------------------------------------
# Trial loading
# ----------------------------------------------------------------------
class Trial:
    """One FBX file reduced to a dict of canonicalised joint positions."""

    def __init__(self, path, name_map=None, label=""):
        self.path = path
        self.label = label or os.path.basename(path)
        self.scene = fbx_scene.Scene(path)
        self.C = up_axis_to_z_matrix(int(self.scene.globals["UpAxis"]),
                                     self.scene.globals["UpAxisSign"])
        self._resolve_names(name_map)
        self._evaluate()

    # -- name resolution (handles Theia's 'person_N:' namespace) --------
    def _resolve_names(self, wanted):
        available = self.scene.limb_joints()
        self.resolved = {}
        self.missing = []
        for key in (wanted or {}):
            hit = None
            if key in available:
                hit = key
            else:
                cands = [n for n in available if n.split(":")[-1] == key]
                if len(cands) == 1:
                    hit = cands[0]
                elif len(cands) > 1:
                    cands.sort()
                    hit = cands[0]
            if hit is None:
                self.missing.append(key)
            else:
                self.resolved[key] = hit

    # -- evaluate global transforms and drop bad frames -----------------
    def _evaluate(self):
        sc = self.scene
        self.dropped_gross = []
        self.dropped_edge = []
        times = sc.key_times
        G = sc.global_transforms(times)
        names = list(self.resolved.values())
        # (F, J, 3) in canonical Z-up frame
        P = np.stack([np.einsum("ij,fj->fi", self.C, G[n][:, :3, 3])
                      for n in names], axis=1)

        self.all_times = times
        self.valid = self._select_frames(P)
        self.times = times[self.valid]
        self.pos = {k: P[self.valid, i] for i, k in enumerate(self.resolved)}

        # motion amplitude within the retained frames
        if P[self.valid].shape[0] > 1:
            self.motion = float(np.ptp(P[self.valid], axis=0).max())
        else:
            self.motion = 0.0
        self.is_static = self.motion < STATIC_MOTION_THRESHOLD

        up = P[self.valid][:, :, 2]
        self.height = float(np.median(up.max(1) - up.min(1)))

    def _select_frames(self, P):
        """Reject rest/bind-pose frames and filter edge artifacts.

        Two stages, because they are different failure modes with very
        different magnitudes.  A bind pose sits tens of centimetres from the
        data; a filter edge artifact sits a few millimetres away and would
        survive any threshold loose enough to be safe against real motion.
        """
        F = P.shape[0]
        if F < 3:
            return np.arange(F)
        centred = P - P.mean(axis=1, keepdims=True)   # shape only, no position

        def deviation(idx):
            med = np.median(centred[idx], axis=0)
            return np.sqrt(((centred - med) ** 2).sum(-1)).mean(-1)

        # stage 1: gross outliers (bind / rest pose), absolute floor 50 mm
        dev = deviation(np.arange(F))
        mad = np.median(np.abs(dev - np.median(dev)))
        thr1 = np.median(dev) + 6.0 * max(mad, 1e-4) + 0.05
        keep = np.where(dev <= thr1)[0]
        if len(keep) < 3:
            return keep if len(keep) >= 2 else np.arange(F)
        self.dropped_gross = sorted(set(range(F)) - set(keep.tolist()))

        # stage 2: relative outliers within the surviving frames, no floor
        dev = deviation(keep)
        sub = dev[keep]
        mad = np.median(np.abs(sub - np.median(sub)))
        thr2 = np.median(sub) + 5.0 * max(mad, 1e-6)
        keep2 = keep[sub <= thr2]
        if len(keep2) >= 3:
            self.dropped_edge = sorted(set(keep.tolist()) - set(keep2.tolist()))
            return keep2
        return keep

    def mean_pose(self):
        return {k: v.mean(axis=0) for k, v in self.pos.items()}

    def sample_at(self, t):
        out = {}
        for k, v in self.pos.items():
            out[k] = np.stack([np.interp(t, self.times, v[:, d])
                               for d in range(3)], axis=-1)
        return out


# ----------------------------------------------------------------------
# Building correspondence observations
# ----------------------------------------------------------------------
def build_observations(theia: Trial, m4d: Trial):
    """Return matched direction vectors and point pairs across frames.

    Static trials are collapsed to their mean pose (the frame grids need
    not overlap).  Dynamic trials are resampled onto the overlapping part
    of the common time base.
    """
    static = theia.is_static and m4d.is_static
    if static:
        poses = [(theia.mean_pose(), m4d.mean_pose())]
        info = dict(mode="static", n_samples=1,
                    theia_motion=theia.motion, m4d_motion=m4d.motion)
    else:
        t0 = max(theia.times[0], m4d.times[0])
        t1 = min(theia.times[-1], m4d.times[-1])
        if t1 <= t0:
            raise ValueError(
                f"{theia.label}/{m4d.label}: trials are dynamic but their time "
                f"ranges do not overlap (Theia {theia.times[0]:.3f}-{theia.times[-1]:.3f}s, "
                f"M4D {m4d.times[0]:.3f}-{m4d.times[-1]:.3f}s). Check the sync.")
        fps = min(theia.scene.fps or 60.0, m4d.scene.fps or 60.0)
        n = max(2, int(round((t1 - t0) * fps)) + 1)
        grid = np.linspace(t0, t1, n)
        pt = theia.sample_at(grid)
        pm = m4d.sample_at(grid)
        poses = [({k: pt[k][i] for k in pt}, {k: pm[k][i] for k in pm})
                 for i in range(n)]
        info = dict(mode="dynamic", n_samples=n, t_start=t0, t_end=t1,
                    theia_motion=theia.motion, m4d_motion=m4d.motion)

    dirs = []    # (label, u_theia, v_m4d, weight)   -- fitted
    pts = []     # (label, p_theia, p_m4d, weight)   -- fitted
    diag = []    # (label, p_theia, p_m4d)           -- reported only
    for pt, pm in poses:
        for label, a, b, w in DIRECTION_VECTORS:
            if not all(j in pt and j in pm for j in (a, b)):
                continue
            u = pt[b] - pt[a]
            v = pm[b] - pm[a]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < 1e-6 or nv < 1e-6:
                continue
            dirs.append((label, u / nu, v / nv, w))
        for j, w in TRANSLATION_JOINTS.items():
            if j in pt and j in pm:
                pts.append((j, pt[j], pm[j], w))
        for j in DIAGNOSTIC_JOINTS:
            if j in pt and j in pm:
                diag.append((j, pt[j], pm[j]))
    return dirs, pts, diag, info


# ----------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------
def solve_rotation(dirs, huber_deg=3.0, n_iter=12):
    """Weighted Wahba/Kabsch on unit vectors with Huber IRLS."""
    U = np.array([d[1] for d in dirs])
    V = np.array([d[2] for d in dirs])
    w0 = np.array([d[3] for d in dirs])
    w = w0.copy()
    R = np.eye(3)
    for _ in range(n_iter):
        M = (U * w[:, None]).T @ V
        Uu, _, Vt = np.linalg.svd(M)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Uu @ Vt))])
        R = Uu @ D @ Vt
        res = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", U, (R @ V.T).T),
                                           -1.0, 1.0)))
        scale = max(huber_deg, 1.4826 * np.median(np.abs(res - np.median(res))))
        w_new = w0 * np.minimum(1.0, scale / np.maximum(res, 1e-6))
        if np.allclose(w_new, w, atol=1e-8):
            w = w_new
            break
        w = w_new
    res = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", U, (R @ V.T).T),
                                       -1.0, 1.0)))
    return R, res, w


def solve_translation(pts, R, huber_m=0.02, n_iter=12):
    """Robust t minimising || p_theia - (R p_m4d + t) ||."""
    A = np.array([p[1] for p in pts])
    B = np.array([p[2] for p in pts])
    w0 = np.array([p[3] for p in pts])
    Brot = (R @ B.T).T
    d = A - Brot
    w = w0.copy()
    t = np.average(d, axis=0, weights=w)
    for _ in range(n_iter):
        r = np.linalg.norm(d - t, axis=1)
        scale = max(huber_m, 1.4826 * np.median(np.abs(r - np.median(r))))
        w = w0 * np.minimum(1.0, scale / np.maximum(r, 1e-9))
        t_new = np.average(d, axis=0, weights=w)
        if np.allclose(t_new, t, atol=1e-9):
            t = t_new
            break
        t = t_new
    resid = d - t
    return t, resid, w


def scale_report(theia: Trial, m4d: Trial):
    """Per-segment length ratio Theia/M4D -- the unit and skeleton check."""
    pt, pm = theia.mean_pose(), m4d.mean_pose()
    rows = []
    for label, a, b in SCALE_SEGMENTS:
        if not all(j in pt and j in pm for j in (a, b)):
            continue
        lt = float(np.linalg.norm(pt[b] - pt[a]))
        lm = float(np.linalg.norm(pm[b] - pm[a]))
        if lt < 1e-6 or lm < 1e-6:
            continue
        rows.append(dict(segment=label, theia_m=lt, m4d_m=lm, ratio=lt / lm))
    return rows


def handedness(trial: Trial):
    """Signed volume of (medio-lateral, vertical, antero-posterior)."""
    p = trial.mean_pose()
    need = ["l_thigh", "r_thigh", "head", "pelvis", "l_foot", "l_toes",
            "r_foot", "r_toes"]
    if not all(k in p for k in need):
        return None
    ml = p["l_thigh"] - p["r_thigh"]
    up = p["head"] - p["pelvis"]
    ap = (p["l_toes"] - p["l_foot"]) + (p["r_toes"] - p["r_foot"])
    return float(np.linalg.det(np.stack([ml, up, ap], axis=1)))


def yaw_observability(dirs):
    """Two distinct things, which are easy to conflate.

    yaw_information = sum w |horizontal component|^2.  This is the Fisher
    information for a rotation about the vertical.  A purely vertical
    direction contributes nothing; a horizontal one contributes fully.
    Parallel horizontal vectors still each contribute -- they are redundant,
    not degenerate.

    azimuthal_diversity = ratio of the eigenvalues of the horizontal scatter.
    This does NOT affect whether yaw is identifiable.  It measures whether
    independent anatomical directions are available to cross-check each
    other, which is what protects against a SYSTEMATIC skeleton-definition
    bias shared by every vector pointing the same way.
    """
    H = np.array([d[2][:2] * np.sqrt(d[3]) for d in dirs])
    if len(H) < 2:
        return None
    info = float((H ** 2).sum())
    ev = np.linalg.eigvalsh(H.T @ H)
    n_eff = float(sum(d[3] for d in dirs))
    return dict(yaw_information=info,
                yaw_information_per_obs=info / max(n_eff, 1e-9),
                azimuthal_diversity=float(ev[0] / ev[1]) if ev[1] > 1e-12 else 0.0)


def subset_agreement(dirs, yaw_only):
    """Yaw estimated from anatomically independent subsets.

    The most interpretable trustworthiness check available: if the
    medio-lateral vectors and the antero-posterior vectors disagree, the
    disagreement is the real uncertainty, whatever the residuals say.
    """
    groups = {
        "medio-lateral (pelvis + shoulder width)": ["pelvis_ML", "shoulder_ML"],
        "antero-posterior (feet)": ["l_foot_AP", "r_foot_AP"],
        "limb long axes": ["l_thigh", "r_thigh", "l_shank", "r_shank",
                           "l_uarm", "r_uarm"],
    }
    out = []
    for name, labs in groups.items():
        sub = [d for d in dirs if d[0] in labs]
        if len(sub) < 2:
            continue
        R, _, _ = solve_rotation(sub)
        if yaw_only:
            R = yaw_matrix(decompose_yaw_tilt(R)[0])
        yaw, tilt = decompose_yaw_tilt(R)
        lever = float(np.mean([np.linalg.norm(d[2][:2]) for d in sub]))
        out.append(dict(subset=name, yaw_deg=yaw, tilt_deg=tilt,
                        mean_yaw_leverage=lever, n=len(sub)))
    if out:
        ys = [o["yaw_deg"] for o in out]
        spread = float(max(ys) - min(ys))
    else:
        spread = 0.0
    return out, spread


def bootstrap_uncertainty(dirs, pts, yaw_only, n=N_BOOTSTRAP, seed=0):
    """Resample direction-vector LABELS with replacement.

    Resampling labels rather than individual observations is deliberate: the
    dominant error here is a systematic per-segment definition mismatch, not
    independent per-frame noise, so the segment is the unit of uncertainty.
    """
    rng = np.random.default_rng(seed)
    labels = sorted({d[0] for d in dirs})
    by_label = {l: [d for d in dirs if d[0] == l] for l in labels}

    # representative points ON THE SUBJECT, in M4D canonical coordinates.
    # The spread of these under resampling is what the user actually cares
    # about: the translation vector alone understates it whenever the
    # subject sits away from the source origin, and overstates it when near.
    probes = np.array([p[2] for p in pts])

    yaws, tilts, ts, probe_sets = [], [], [], []
    for _ in range(n):
        pick = rng.choice(len(labels), size=len(labels), replace=True)
        sub = [d for i in pick for d in by_label[labels[i]]]
        if len({d[0] for d in sub}) < 3:
            continue
        try:
            R, _, _ = solve_rotation(sub, n_iter=6)
            if yaw_only:
                R = yaw_matrix(decompose_yaw_tilt(R)[0])
            t, _, _ = solve_translation(pts, R, n_iter=6)
        except Exception:
            continue
        y, ti = decompose_yaw_tilt(R)
        yaws.append(y)
        tilts.append(ti)
        ts.append(t)
        probe_sets.append((R @ probes.T).T + t)
    if len(yaws) < 10:
        return None
    ts = np.array(ts)
    probe_sets = np.array(probe_sets)              # (B, J, 3)
    # per-probe SD across bootstrap replicates, then summarised over probes
    probe_sd = probe_sets.std(axis=0)              # (J, 3)
    probe_dist = np.linalg.norm(probe_sets - probe_sets.mean(axis=0), axis=2)
    return dict(
        n=len(yaws),
        yaw_sd_deg=float(np.std(yaws)),
        yaw_ci95_deg=[float(np.percentile(yaws, 2.5)),
                      float(np.percentile(yaws, 97.5))],
        tilt_sd_deg=float(np.std(tilts)),
        t_sd_mm=(np.std(ts, axis=0) * 1000).tolist(),
        t_ci95_mm=[(np.percentile(ts, 2.5, axis=0) * 1000).tolist(),
                   (np.percentile(ts, 97.5, axis=0) * 1000).tolist()],
        transformed_point_sd_mm=(probe_sd.mean(axis=0) * 1000).tolist(),
        transformed_point_rms_mm=float(np.sqrt((probe_dist ** 2).mean()) * 1000),
        transformed_point_p95_mm=float(np.percentile(probe_dist, 95) * 1000),
    )


def leave_one_out(dirs, pts, yaw_only):
    """Refit with each direction-vector label removed in turn."""
    labels = sorted({d[0] for d in dirs})
    out = []
    for lab in labels:
        sub = [d for d in dirs if d[0] != lab]
        if len(sub) < 3:
            continue
        R, _, _ = solve_rotation(sub)
        if yaw_only:
            R = yaw_matrix(decompose_yaw_tilt(R)[0])
        t, _, _ = solve_translation(pts, R)
        yaw, tilt = decompose_yaw_tilt(R)
        out.append(dict(excluded=lab, yaw_deg=yaw, tilt_deg=tilt,
                        t=t.tolist()))
    return out


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def solve_alignment(pairs, yaw_only=False, verbose=True):
    """pairs: list of (theia_fbx_path, m4d_fbx_path)."""
    trials, all_dirs, all_pts, all_diag, infos, scales = [], [], [], [], [], []

    for tp, mp in pairs:
        th = Trial(tp, JOINT_MAP.keys(), label=os.path.basename(tp))
        m4 = Trial(mp, JOINT_MAP.values(), label=os.path.basename(mp))
        # remap M4D keys to Theia keys so both dicts share a vocabulary
        m4.pos = {tk: m4.pos[mk] for tk, mk in JOINT_MAP.items()
                  if mk in m4.pos}
        m4.resolved = {tk: m4.resolved[mk] for tk, mk in JOINT_MAP.items()
                       if mk in m4.resolved}
        d, p, dg, info = build_observations(th, m4)
        info["theia_file"] = th.label
        info["m4d_file"] = m4.label
        all_dirs += d
        all_pts += p
        all_diag += dg
        infos.append(info)
        scales.append(scale_report(th, m4))
        trials.append((th, m4))

    # ---- scale / unit check ------------------------------------------
    flat = [r for s in scales for r in s]
    ratios = np.array([r["ratio"] for r in flat])
    median_ratio = float(np.median(ratios))

    # ---- rotation ----------------------------------------------------
    R_full, ang_res, wrot = solve_rotation(all_dirs)
    yaw, tilt = decompose_yaw_tilt(R_full)
    R_yaw = yaw_matrix(yaw)
    R = R_yaw if yaw_only else R_full

    # ---- translation -------------------------------------------------
    t, pos_res, wtr = solve_translation(all_pts, R)

    # Evaluate the alternative too.  If dropping the out-of-plane part of
    # the rotation does not hurt -- or helps -- the point fit, that tilt was
    # fitting the skeleton mismatch, not a real calibration difference.
    R_alt = R_full if yaw_only else R_yaw
    t_alt, res_alt, _ = solve_translation(all_pts, R_alt)
    rms_alt = float(np.sqrt((np.linalg.norm(res_alt, axis=1) ** 2).mean()) * 1000)

    # ---- diagnostics -------------------------------------------------
    per_dir = {}
    for (lab, _, _, _), r, w in zip(all_dirs, ang_res, wrot):
        per_dir.setdefault(lab, []).append((r, w))
    per_dir = {k: dict(mean_deg=float(np.mean([a for a, _ in v])),
                       final_weight=float(np.mean([b for _, b in v])),
                       n=len(v))
               for k, v in per_dir.items()}

    per_joint = {}
    for (lab, _, _, _), r, w in zip(all_pts, pos_res, wtr):
        per_joint.setdefault(lab, []).append((np.linalg.norm(r), r, w))
    per_joint = {
        k: dict(mean_mm=float(np.mean([a for a, _, _ in v]) * 1000),
                vector_mm=(np.mean([b for _, b, _ in v], axis=0) * 1000).tolist(),
                final_weight=float(np.mean([c for _, _, c in v])),
                n=len(v))
        for k, v in per_joint.items()}

    # landmarks excluded from the fit, reported so the user can see how far
    # apart the two skeletons' segment-origin conventions actually are
    diag_joint = {}
    for lab, pa, pb in all_diag:
        d = pa - (R @ pb + t)
        diag_joint.setdefault(lab, []).append(d)
    diag_joint = {k: dict(mean_mm=float(np.linalg.norm(np.mean(v, axis=0)) * 1000),
                          vector_mm=(np.mean(v, axis=0) * 1000).tolist())
                  for k, v in diag_joint.items()}

    rms_mm = float(np.sqrt((np.linalg.norm(pos_res, axis=1) ** 2).mean()) * 1000)
    # observability computed with the FINAL robust weights, so that a
    # direction the fit has effectively discarded does not flatter it
    obs = yaw_observability([(d[0], d[1], d[2], float(w))
                             for d, w in zip(all_dirs, wrot)])
    boot = bootstrap_uncertainty(all_dirs, all_pts, yaw_only)
    subsets, subset_spread = subset_agreement(all_dirs, yaw_only)
    loo = leave_one_out(all_dirs, all_pts, yaw_only)
    yaw_spread = float(np.ptp([l["yaw_deg"] for l in loo])) if loo else 0.0
    t_spread_mm = (float(np.ptp(np.array([l["t"] for l in loo]), axis=0).max()) * 1000
                   if loo else 0.0)

    # ---- assemble the final matrix in ORIGINAL M4D file coordinates ---
    C4 = np.eye(4)
    C4[:3, :3] = trials[0][1].C          # M4D up-axis -> Z-up
    T_res = np.eye(4)
    T_res[:3, :3] = R
    T_res[:3, 3] = t
    T_total = T_res @ C4

    # Theia file frame is already Z-up in these data; if it were not, undo it
    C_theia = trials[0][0].C
    if not np.allclose(C_theia, np.eye(3)):
        Ct = np.eye(4)
        Ct[:3, :3] = C_theia.T
        T_total = Ct @ T_total

    result = dict(
        T_m4d_to_theia=T_total.tolist(),
        rotation=R.tolist(),
        translation_m=t.tolist(),
        yaw_deg=yaw,
        residual_tilt_deg=tilt,
        total_rotation_deg=rotation_angle_deg(R_full),
        yaw_only=yaw_only,
        scale_median_ratio=median_ratio,
        scale_iqr=float(np.percentile(ratios, 75) - np.percentile(ratios, 25)),
        scale_segments=flat,
        handedness=dict(theia=handedness(trials[0][0]),
                        m4d=handedness(trials[0][1])),
        heights_m=dict(theia=trials[0][0].height, m4d=trials[0][1].height),
        angular_residual_deg=dict(
            mean=float(np.mean(ang_res)), median=float(np.median(ang_res)),
            max=float(np.max(ang_res)), per_vector=per_dir),
        position_residual=dict(rms_mm=rms_mm, per_joint=per_joint),
        model_comparison=dict(
            chosen="yaw-only" if yaw_only else "full 3-DOF",
            chosen_rms_mm=rms_mm,
            alternative="full 3-DOF" if yaw_only else "yaw-only",
            alternative_rms_mm=rms_alt),
        excluded_landmark_offsets=diag_joint,
        yaw_observability=obs,
        subset_agreement=dict(subsets=subsets, yaw_spread_deg=subset_spread),
        uncertainty=boot,
        stability=dict(leave_one_out=loo, yaw_spread_deg=yaw_spread,
                       t_spread_mm=t_spread_mm),
        trials=infos,
        n_direction_obs=len(all_dirs),
        n_point_obs=len(all_pts),
    )

    if verbose:
        _print_report(result, trials)
    return result, T_total


def _print_report(r, trials):
    W = 74
    print("=" * W)
    print("  MOVE4D  ->  THEIA3D   rigid alignment")
    print("=" * W)

    for (th, m4), info in zip(trials, r["trials"]):
        print(f"\n  {info['theia_file']}  /  {info['m4d_file']}")
        for tr, lab in ((th, "Theia"), (m4, "M4D  ")):
            sc = tr.scene
            print(f"    {lab}: {len(tr.all_times):>3d} keys @ "
                  f"{sc.fps:g} fps, kept {len(tr.times)}, "
                  f"unit={sc.unit_cm:g} cm, up-axis={int(sc.globals['UpAxis'])}, "
                  f"residual motion={tr.motion*1000:.1f} mm")
            if tr.dropped_gross:
                print(f"           dropped frame(s) {tr.dropped_gross} "
                      f"- rest/bind pose, not data")
            if tr.dropped_edge:
                print(f"           dropped frame(s) {tr.dropped_edge} "
                      f"- filter edge artifact")
        print(f"    -> treated as {info['mode'].upper()} "
              f"({info['n_samples']} sample(s) used)")

    print("\n" + "-" * W)
    print("  CHECKS")
    print("-" * W)
    hs = r["handedness"]
    ok_hand = (hs["theia"] is None or hs["m4d"] is None or
               np.sign(hs["theia"]) == np.sign(hs["m4d"]))
    print(f"    handedness      Theia {hs['theia']:+.4f} | M4D {hs['m4d']:+.4f}"
          f"   {'OK (no mirroring)' if ok_hand else '*** MIRRORED ***'}")
    print(f"    vertical span   Theia {r['heights_m']['theia']:.3f} m | "
          f"M4D {r['heights_m']['m4d']:.3f} m   (mapped joints, not stature)")
    print(f"    scale ratio     median {r['scale_median_ratio']:.4f} "
          f"(IQR {r['scale_iqr']:.4f})")
    print("      segment          Theia      M4D      ratio")
    for s in r["scale_segments"]:
        print(f"      {s['segment']:14s} {s['theia_m']:7.4f}  {s['m4d_m']:7.4f}   "
              f"{s['ratio']:6.4f}")

    print("\n" + "-" * W)
    print("  SOLUTION")
    print("-" * W)
    T = np.array(r["T_m4d_to_theia"])
    print("    T (M4D file coords -> Theia file coords):")
    for row in T:
        print("      [" + "  ".join(f"{v:9.6f}" for v in row) + "]")
    print(f"\n    yaw about vertical      {r['yaw_deg']:+8.3f} deg")
    print(f"    residual tilt           {r['residual_tilt_deg']:8.3f} deg"
          f"   {'(constrained away)' if r['yaw_only'] else ''}")
    print(f"    total rotation          {r['total_rotation_deg']:8.3f} deg")
    print(f"    translation (m)         "
          f"[{r['translation_m'][0]:+.4f} {r['translation_m'][1]:+.4f} "
          f"{r['translation_m'][2]:+.4f}]")

    print("\n" + "-" * W)
    print("  FIT QUALITY")
    print("-" * W)
    a = r["angular_residual_deg"]
    print(f"    angular residual  mean {a['mean']:.2f} deg | "
          f"median {a['median']:.2f} | max {a['max']:.2f}")
    print("      direction         resid(deg)  weight")
    for k, v in sorted(a["per_vector"].items(), key=lambda x: -x[1]["mean_deg"]):
        flag = "  <-- downweighted" if v["final_weight"] < 0.5 else ""
        print(f"      {k:16s} {v['mean_deg']:7.2f}   {v['final_weight']:5.2f}{flag}")

    p = r["position_residual"]
    print(f"\n    position residual RMS   {p['rms_mm']:.1f} mm   "
          f"(fitted joint centres)")
    print("      joint            resid(mm)   residual vector (mm)")
    for k, v in sorted(p["per_joint"].items(), key=lambda x: -x[1]["mean_mm"]):
        vec = "  ".join(f"{c:+6.1f}" for c in v["vector_mm"])
        print(f"      {k:14s} {v['mean_mm']:8.1f}    [{vec}]")

    mc = r.get("model_comparison")
    if mc:
        print(f"\n    rotation model comparison (position RMS):")
        print(f"      {mc['chosen']:12s} (chosen)  {mc['chosen_rms_mm']:6.1f} mm")
        print(f"      {mc['alternative']:12s}           "
              f"{mc['alternative_rms_mm']:6.1f} mm")
        delta = mc["alternative_rms_mm"] - mc["chosen_rms_mm"]
        if delta < -0.5 and not r["yaw_only"]:
            print("      -> the yaw-only model fits BETTER. The out-of-plane")
            print("         part of the rotation is absorbing skeleton mismatch,")
            print("         not a real calibration tilt. Re-run with --yaw-only.")
        elif delta < 0.5 and not r["yaw_only"]:
            print("      -> no meaningful gain from the extra 2 DOF; prefer")
            print("         --yaw-only, which cannot invent a spurious tilt.")
        elif r["yaw_only"] and delta < -0.5:
            print("      -> the unconstrained fit is better by "
                  f"{-delta:.1f} mm; the tilt may be real. Check both floors.")

    ex = r["excluded_landmark_offsets"]
    if ex:
        print("\n    landmarks EXCLUDED from the fit (skeleton definition gap,")
        print("    not alignment error -- do not read these as accuracy):")
        for k, v in sorted(ex.items(), key=lambda x: -x[1]["mean_mm"]):
            vec = "  ".join(f"{c:+6.1f}" for c in v["vector_mm"])
            print(f"      {k:14s} {v['mean_mm']:8.1f}    [{vec}]")

    print("\n" + "-" * W)
    print("  OBSERVABILITY & UNCERTAINTY")
    print("-" * W)
    o = r["yaw_observability"]
    if o:
        print(f"    yaw information        {o['yaw_information']:.2f} "
              f"({o['yaw_information_per_obs']:.2f} per weighted obs)")
        if o["yaw_information"] < 0.5:
            print("      *** yaw is barely identifiable: almost every direction")
            print("          is vertical. Do not trust the yaw. ***")
        elif o["yaw_information_per_obs"] < 0.25:
            print("      LOW: most directions are near-vertical and contribute")
            print("           little; yaw rests on a few horizontal vectors.")
        else:
            print("      adequate.")
        print(f"    azimuthal diversity    {o['azimuthal_diversity']:.3f}"
              f"   (cross-check capacity, not identifiability)")
        if o["azimuthal_diversity"] < 0.1:
            print("      LOW: the horizontal directions the fit trusts are")
            print("           nearly parallel, so a systematic bias shared by")
            print("           them cannot be detected from within this trial.")

    sa = r["subset_agreement"]
    if sa["subsets"]:
        print("\n    yaw from anatomically independent subsets:")
        print("      subset                                  yaw     leverage")
        for s in sa["subsets"]:
            print(f"      {s['subset']:36s} {s['yaw_deg']:+7.2f}  "
                  f"{s['mean_yaw_leverage']:8.2f}")
        print(f"      -> spread {sa['yaw_spread_deg']:.2f} deg"
              f"   (a direct read on systematic uncertainty)")

    b = r["uncertainty"]
    if b:
        print(f"\n    bootstrap over segments (n={b['n']}):")
        print(f"      yaw   {r['yaw_deg']:+.2f} deg  SD {b['yaw_sd_deg']:.2f}  "
              f"95% CI [{b['yaw_ci95_deg'][0]:+.2f}, {b['yaw_ci95_deg'][1]:+.2f}]")
        sd = b["t_sd_mm"]
        print(f"      t     SD [{sd[0]:.1f}, {sd[1]:.1f}, {sd[2]:.1f}] mm "
              f"(translation vector alone)")
        print(f"\n    propagated to points on the subject -- the number that")
        print(f"    actually matters for downstream use:")
        print(f"      transformed-point RMS spread  "
              f"{b['transformed_point_rms_mm']:.1f} mm")
        print(f"      95th percentile               "
              f"{b['transformed_point_p95_mm']:.1f} mm")
    s = r["stability"]
    print(f"\n    leave-one-segment-out  yaw spread {s['yaw_spread_deg']:.3f} deg, "
          f"t spread {s['t_spread_mm']:.1f} mm")
    print("=" * W)


def _write_outputs(result, T, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "alignment_report.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savetxt(os.path.join(outdir, "T_m4d_to_theia.txt"), T,
               fmt="%.9f", header="4x4 homogeneous: p_theia = T @ [p_m4d; 1]")
    np.save(os.path.join(outdir, "T_m4d_to_theia.npy"), T)
    return outdir


def apply_transform(T, points):
    """Apply a 4x4 transform to an (...,3) array of MOVE4D points."""
    pts = np.asarray(points, dtype=float)
    shape = pts.shape
    flat = pts.reshape(-1, 3)
    out = (T[:3, :3] @ flat.T).T + T[:3, 3]
    return out.reshape(shape)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("theia", nargs="?", help="Theia3D .fbx")
    ap.add_argument("m4d", nargs="?", help="MOVE4D .fbx")
    ap.add_argument("--pairs", help="text file, one 'theia.fbx m4d.fbx' per line")
    ap.add_argument("-o", "--outdir", default="alignment_out")
    ap.add_argument("--yaw-only", action="store_true",
                    help="constrain the rotation to the vertical axis")
    ap.add_argument("--plot", action="store_true", help="write a diagnostic PNG")
    args = ap.parse_args(argv)

    if args.pairs:
        pairs = []
        with open(args.pairs) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    a, b = line.split()
                    pairs.append((a, b))
    elif args.theia and args.m4d:
        pairs = [(args.theia, args.m4d)]
    else:
        ap.error("give THEIA.fbx M4D.fbx, or --pairs FILE")

    result, T = solve_alignment(pairs, yaw_only=args.yaw_only)
    _write_outputs(result, T, args.outdir)
    print(f"\n  wrote {args.outdir}/T_m4d_to_theia.{{txt,npy}} "
          f"and alignment_report.json")

    if args.plot:
        import plot_alignment
        plot_alignment.make_plot(pairs, T, os.path.join(args.outdir,
                                                        "alignment_check.png"))
        print(f"  wrote {args.outdir}/alignment_check.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
