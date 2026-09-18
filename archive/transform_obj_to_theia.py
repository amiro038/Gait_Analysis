# -*- coding: utf-8 -*-
"""
=============================================================================
 Put MOVE4D .obj meshes into the Theia3D coordinate system
=============================================================================

Applies the 4x4 transform produced by m4d_to_theia_transform.py to one or
more Wavefront .obj files, writing new .obj files in Theia coordinates.

    p_theia = T @ [p_m4d, 1]

Run it in Spyder: set the paths in CONFIG and press F5. Command line works
too:  python transform_obj_to_theia.py a.obj b.obj

WHAT THE INPUT HAS TO BE
------------------------
The .obj vertices must be in RAW MOVE4D FBX file coordinates -- the same
frame the scan mesh lives in: Y-up, metres, origin on the floor. Meshes
segmented out of the MOVE4D scan already are. The script checks this for you
against the scan itself (see CHECK_AGAINST_SCAN) rather than assuming it; a
mesh that has been re-centred, scaled to millimetres or Z-flipped by some
intermediate tool will be caught.

THE UP AXIS CHANGES
-------------------
MOVE4D files are Y-up, Theia files are Z-up, and that conversion is part of
T. So the OUTPUT is Z-up. Open it in a Y-up viewer (Blender's default
import, MeshLab) and the foot will appear to lie on its side. That is
correct, not a bug -- it is what "in Theia's coordinate system" means.

NORMALS
-------
Vertex normals (vn) are transformed by the inverse-transpose of the linear
block and renormalised, which is right for a rotation and stays right if you
ever fit a scale. Faces, groups, comments and every other line are passed
through byte for byte, so topology and any material references survive.
"""

from __future__ import annotations

import os
import sys

import numpy as np


# %%==========================================================================
#  CONFIG
# ============================================================================

# The .obj files to convert. Output goes next to each input with OUT_SUFFIX
# inserted before the extension, unless OUT_DIR is set.
OBJ_FILES = ["left_feet.obj", "right_feet.obj", "both_feet.obj"]
OUT_SUFFIX = "_theia"
OUT_DIR = None                      # None = alongside the input

# The transform. Either the .npy / .txt written by m4d_to_theia_transform.py,
# or set T_MATRIX directly to a 4x4 array and leave T_FILE as None.
T_FILE = "T_m4d_to_theia.npy"
T_MATRIX = None

# Sanity checks. Both are cheap and both have caught real mistakes.
CHECK_AGAINST_SCAN = "D05_C1_apose_M4D.fbx"   # None to skip
FLOOR_TOLERANCE_MM = 25.0           # how far the sole may sit from the floor

# Draw the result next to Theia's own body model and joint centres. The most
# convincing check there is: if the transform is right, a segmented MOVE4D
# part lands around the matching Theia anatomy.
MAKE_PLOT = True
COMPARE_WITH_THEIA = "D05_C1_apose_Theia.fbx"     # None to skip the overlay
PLOT_FILE = "obj_transform_check.png"


# %%==========================================================================
#  OBJ I/O  -- geometry is rewritten, everything else passes through
# ============================================================================

def read_obj(path):
    """Parse an .obj into (lines, vertices, normals, index maps).

    `lines` is the file verbatim. `v_at[i]` is the line number of the i-th
    vertex, so the file can be written back out with only those lines
    changed and nothing else touched.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    verts, norms, v_at, n_at = [], [], [], []
    for i, line in enumerate(lines):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
            v_at.append(i)
        elif line.startswith("vn "):
            norms.append([float(x) for x in line.split()[1:4]])
            n_at.append(i)
    if not verts:
        raise ValueError(f"{path}: no vertices found")
    return (lines, np.array(verts, float),
            np.array(norms, float) if norms else np.zeros((0, 3)),
            v_at, n_at)


def write_obj(path, lines, verts, norms, v_at, n_at, header=None):
    out = list(lines)
    for k, i in enumerate(v_at):
        out[i] = "v %.6f %.6f %.6f" % tuple(verts[k])
    for k, i in enumerate(n_at):
        out[i] = "vn %.6f %.6f %.6f" % tuple(norms[k])
    if header:
        out = ["# " + h for h in header] + out
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


# %%==========================================================================
#  THE TRANSFORM
# ============================================================================

def load_transform():
    """The 4x4, from T_MATRIX or from T_FILE (.npy or whitespace .txt)."""
    if T_MATRIX is not None:
        T = np.asarray(T_MATRIX, float)
    elif T_FILE and os.path.exists(T_FILE):
        T = (np.load(T_FILE) if T_FILE.endswith(".npy")
             else np.loadtxt(T_FILE))
    else:
        raise FileNotFoundError(
            f"no transform: set T_MATRIX, or run m4d_to_theia_transform.py "
            f"to produce {T_FILE!r}")
    if T.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {T.shape}")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError(f"bottom row of T is {T[3]}, expected [0 0 0 1]")
    return T


def transform_points(T, P):
    """(N, 3) positions -> (N, 3), i.e. p_theia = T @ [p_m4d, 1]."""
    return P @ T[:3, :3].T + T[:3, 3]


def transform_normals(T, N):
    """(N, 3) unit normals -> (N, 3).

    Normals follow the inverse-transpose of the linear block, not the block
    itself. For a pure rotation those are the same thing, so this is belt
    and braces -- but it stays correct if the transform ever carries a scale.
    """
    if len(N) == 0:
        return N
    out = N @ np.linalg.inv(T[:3, :3])
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, n, out=np.zeros_like(out), where=n > 1e-12)


# %%==========================================================================
#  CHECKS
# ============================================================================

def scan_bounds(fbx_path):
    """Bounding box of the MOVE4D scan surface, in raw file coordinates.

    Reuses the skinning already implemented in m4d_to_theia_transform, so
    there is one implementation of it, not two.
    """
    try:
        import m4d_to_theia_transform as m4t
    except ImportError:
        return None
    if not os.path.exists(fbx_path):
        return None
    surf = m4t.body_surface(m4t.Scene(fbx_path))
    return None if surf is None else (surf.min(0), surf.max(0))


def check_input_frame(name, V, bounds):
    """Is this mesh really in raw MOVE4D coordinates?

    A mesh segmented out of the scan must sit inside the scan's own bounding
    box. Re-centring, a unit change or an axis flip all show up here, and all
    of them would silently produce a wrong-looking result otherwise.
    """
    if bounds is None:
        return ["scan not available, input frame NOT verified"]
    lo, hi = bounds
    pad = 0.02                                   # 20 mm of slack
    inside = np.all(V.min(0) >= lo - pad) and np.all(V.max(0) <= hi + pad)
    if inside:
        return []
    msgs = [f"{name}: vertices fall OUTSIDE the MOVE4D scan's bounding box, "
            f"so this mesh is probably not in raw MOVE4D coordinates."]
    span_v, span_s = V.max(0) - V.min(0), hi - lo
    ratio = np.median(span_v / np.maximum(span_s, 1e-9))
    if ratio > 10:
        msgs.append(f"  its extent is ~{ratio:.0f}x the scan's -- millimetres "
                    f"rather than metres?")
    elif ratio < 0.1:
        msgs.append(f"  its extent is ~{ratio:.3f}x the scan's -- check the "
                    f"units.")
    else:
        msgs.append("  extent looks right, so it is more likely re-centred "
                    "or axis-flipped than rescaled.")
    return msgs


def check_floor(name, V_theia, T):
    """After transforming, does the sole sit on Theia's floor?

    MOVE4D's floor is z = 0 in its own file, so it lands at z = T[2, 3] in
    Theia coordinates. A foot mesh's lowest vertices should be right there.
    This is the end-to-end test: it fails if the transform was applied to
    the wrong axis convention, or not at all.
    """
    expected = T[2, 3] * 1000.0
    sole = np.percentile(V_theia[:, 2], 0.5) * 1000.0
    off = sole - expected
    ok = abs(off) <= FLOOR_TOLERANCE_MM
    return dict(ok=ok, expected_mm=expected, sole_mm=sole, offset_mm=off)


# %%==========================================================================
#  FIGURE
# ============================================================================

def theia_reference(fbx_path, bbox, pad=0.05):
    """Theia's own surface and joint centres, clipped to `bbox`. -> (pts, joints)

    Clipping to the transformed mesh's own bounding box keeps the comparison
    readable whatever part of the body was segmented out -- a foot, a hand,
    a whole leg.
    """
    try:
        import m4d_to_theia_transform as m4t
    except ImportError:
        return None, {}
    if not fbx_path or not os.path.exists(fbx_path):
        return None, {}
    sc = m4t.Scene(fbx_path)
    surf = m4t.body_surface(sc)
    lo, hi = bbox[0] - pad, bbox[1] + pad
    if surf is not None:
        inside = np.all((surf >= lo) & (surf <= hi), axis=1)
        surf = surf[inside] if inside.any() else None
    G = sc.global_transforms()
    frame = len(sc.times) // 2
    joints = {}
    for name, M4 in G.items():
        p = M4[frame, :3, 3]
        if np.all((p >= lo) & (p <= hi)):
            joints[name.split(":")[-1]] = p
    return surf, joints


def make_figure(results, T, path=None):
    """Three orthographic views of every transformed mesh, in Theia coords."""
    import matplotlib.pyplot as plt

    pts = []
    for r in results:
        V = read_obj(r["output"])[1]
        pts.append(V)
    if not pts:
        return None
    allpts = np.vstack(pts)
    bbox = (allpts.min(0), allpts.max(0))
    ref, joints = theia_reference(COMPARE_WITH_THEIA, bbox)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    views = [(0, 1, "top  (X-Y)"), (1, 2, "side  (Y-Z)"), (0, 2, "front  (X-Z)")]
    step = max(1, len(allpts) // 3000)
    for ax, (i, j, name) in zip(axes, views):
        ax.scatter(allpts[::step, i], allpts[::step, j], s=1.2, c="#c0392b",
                   alpha=0.45, label="MOVE4D mesh, transformed", rasterized=True)
        if ref is not None and len(ref):
            s2 = max(1, len(ref) // 3000)
            ax.scatter(ref[::s2, i], ref[::s2, j], s=1.2, c="#1f4e79",
                       alpha=0.45, label="Theia body model", rasterized=True)
        for nm, p in joints.items():
            ax.plot(p[i], p[j], "k+", ms=11, mew=2)
            ax.annotate(nm, (p[i], p[j]), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        if j == 2:                                  # a vertical view
            ax.axhline(T[2, 3], color="k", ls=":", lw=1)
            ax.annotate(f"Theia floor (z = {T[2, 3] * 1000:+.1f} mm)",
                        (ax.get_xlim()[0], T[2, 3]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, markerscale=6, loc="upper left")
    fig.suptitle("MOVE4D mesh placed in Theia3D coordinates   "
                 "(black + = Theia joint centres)", fontsize=11, y=1.0)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


# %%==========================================================================
#  MAIN
# ============================================================================

def convert(paths=None, verbose=True):
    """Transform every .obj in `paths`. Returns a list of result dicts."""
    paths = OBJ_FILES if paths is None else paths
    T = load_transform()
    bounds = scan_bounds(CHECK_AGAINST_SCAN) if CHECK_AGAINST_SCAN else None

    if verbose:
        W = 74
        print("=" * W)
        print("  MOVE4D .obj  ->  THEIA3D coordinates")
        print("=" * W)
        print("\n  T (raw MOVE4D -> raw Theia):")
        for row in T:
            print("    [" + "  ".join(f"{v:10.6f}" for v in row) + "]")
        print("\n  MOVE4D is Y-up, Theia is Z-up, and that conversion is part")
        print("  of T -- so the OUTPUT files are Z-UP. A Y-up viewer will show")
        print("  them lying on their side. That is correct.")
        if bounds is None and CHECK_AGAINST_SCAN:
            print(f"\n  ! {CHECK_AGAINST_SCAN} not readable -- the input frame "
                  f"cannot be verified.")

    results = []
    for path in paths:
        if not os.path.exists(path):
            print(f"\n  !! {path}: not found, skipped")
            continue
        lines, V, N, v_at, n_at = read_obj(path)
        warnings = check_input_frame(os.path.basename(path), V, bounds)

        Vt = transform_points(T, V)
        Nt = transform_normals(T, N)

        stem, ext = os.path.splitext(os.path.basename(path))
        out_dir = OUT_DIR or (os.path.dirname(path) or ".")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, stem + OUT_SUFFIX + ext)
        write_obj(out_path, lines, Vt, Nt, v_at, n_at, header=[
            "Transformed from MOVE4D into Theia3D coordinates (Z-up, metres)",
            "by transform_obj_to_theia.py. p_theia = T @ [p_m4d, 1]",
            "T rows: " + " | ".join(
                " ".join(f"{v:.9f}" for v in row) for row in T)])

        floor = check_floor(os.path.basename(path), Vt, T)
        res = dict(input=path, output=out_path, n_vertices=len(V),
                   n_normals=len(N), warnings=warnings, floor=floor,
                   bbox_in=(V.min(0), V.max(0)),
                   bbox_out=(Vt.min(0), Vt.max(0)))
        results.append(res)

        if verbose:
            print(f"\n  {os.path.basename(path)} -> {os.path.basename(out_path)}")
            print(f"    {len(V)} vertices, {len(N)} normals")
            print(f"    in  (MOVE4D, Y-up)  min {np.round(V.min(0), 4)}  "
                  f"max {np.round(V.max(0), 4)}")
            print(f"    out (Theia,  Z-up)  min {np.round(Vt.min(0), 4)}  "
                  f"max {np.round(Vt.max(0), 4)}")
            for w in warnings:
                print(f"    ! {w}")
            f = floor
            verdict = "OK" if f["ok"] else "*** CHECK THIS ***"
            print(f"    floor check: sole at z = {f['sole_mm']:+.1f} mm, "
                  f"Theia's floor is at {f['expected_mm']:+.1f} mm "
                  f"({f['offset_mm']:+.1f} mm)  {verdict}")
            if not f["ok"]:
                print("      A foot should sit ON the floor. If this is far "
                      "out, either the")
                print("      input was not in raw MOVE4D coordinates or the "
                      "wrong T was used.")

    if results and MAKE_PLOT:
        try:
            make_figure(results, T, PLOT_FILE)
            if verbose:
                print(f"\n  wrote {PLOT_FILE}")
        except ImportError:
            if verbose:
                print("\n  (no matplotlib -- skipping the figure)")

    if verbose and results:
        bad = [r for r in results if not r["floor"]["ok"] or r["warnings"]]
        print("\n" + "-" * 74)
        print(f"  {len(results)} file(s) written"
              + (f", {len(bad)} with warnings" if bad else ", all checks passed"))
        print("-" * 74)
    return results


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    return convert(argv if argv else None)


if __name__ == "__main__":
    RESULTS = main()
