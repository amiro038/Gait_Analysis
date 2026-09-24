# -*- coding: utf-8 -*-
"""
Figures for the Word methods document (docs/Gait_Analysis_Methods.docx).

Every figure is drawn from the REAL functions of gait_analysis.py and
detect_gait_events.py, run on a synthetic trial from synthetic_trial.py
(80 kg walker, 20 kg load 150 mm behind the trunk, 1.3 m/s belt, a quiet
standing to start), and from the participant's real boot mesh. So each
figure shows exactly what the code does.

    python docs/make_methods_figures.py          # -> docs/figures/fig01.png ...

Takes a few minutes (the synthetic trial is 7 minutes long, so the stride
series are long enough for DFA and LDS). The analysed trial is cached in
docs/figures/_trial.pkl; delete it to recompute.
"""

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon            # noqa: E402
from scipy.signal import butter, sosfiltfilt                    # noqa: E402
from scipy.spatial import ConvexHull                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import detect_gait_events as dge                               # noqa: E402
import gait_analysis as ga                                     # noqa: E402
import synthetic_trial as syn                                  # noqa: E402

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# the validated reference palette (dataviz skill): identity by fixed slot
LEFT, RIGHT, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, YELLOW = "#4a3aa7", "#eda100"
INK, MUTED, GRID, SHADE = "#0b0b0b", "#6b6a66", "#e4e3df", "#efeee9"
COLOUR = {"L": LEFT, "R": RIGHT}
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.4,
    "figure.dpi": 100, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
W = 6.5                                     # inches: the page text width


def save(fig, n):
    fig.savefig(OUT / f"fig{n:02d}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  fig{n:02d}.png")


def letter(ax, s):
    """Panel letter in bold, then the panel's title, both left-aligned."""
    title = ax.get_title() or ax.get_title(loc="left")
    ax.set_title("")
    ax.set_title(rf"$\bf{{{s}}}$   {title}", loc="left")


# %% --------------------------------------------------------------------------
#  the synthetic trial, analysed by gait_analysis.py
# -----------------------------------------------------------------------------

def analysed_trial():
    cache = OUT / "_trial.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    work = Path(tempfile.mkdtemp(prefix="methods_figures_"))
    truth = syn.make_trial(work, duration=420.0, special=True)
    dge.FORCE_FOLDER, dge.KINEMATIC_FOLDER = truth["force_dir"], truth["kin_dir"]
    dge.OUTPUT_FOLDER, dge.PLATE_FILE = work / "events", truth["plates"]
    binding = dge.HERE / "foot_mesh_binding.npz"
    plates = {b: dge.fit_plate(p) for b, p in dge.read_plates(truth["plates"]).items()}
    boots, pose_names = dge.read_binding(binding)
    res = dge.process_trial(truth["force_dir"] / f"{truth['name']}.csv", boots,
                            pose_names, plates, verbose=False)
    ga.WARMUP_S = 30.0
    ga.MAKE_FIGURES = False
    ga.FOLDERS = dict(force=truth["force_dir"], kinematic=truth["kin_dir"],
                      events=work / "events", binding=binding,
                      plates=truth["plates"])
    ga.OUTPUT_FOLDER = work / "analysis"
    people = ga.read_participants(work / "participants.csv")
    t, table = ga.analyse_trial(truth["name"], people["S01"], None)
    _, extras = ga.summarise(t, None)
    ga.build_tables(ga.OUTPUT_FOLDER)
    data = dict(t=t, table=table, extras=extras, truth=truth, res=res,
                plates=plates, work=work, boots=boots)
    cache.write_bytes(pickle.dumps(data))
    return data


# %% --------------------------------------------------------------------------
#  the figures
# -----------------------------------------------------------------------------

def fig_pipeline():
    """Fig 1: the pipeline."""
    fig, ax = plt.subplots(figsize=(W, 4.3))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    def box(x, y, w, h, title, body, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc="white", ec=colour, lw=1.6))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top",
                fontweight="bold", fontsize=8, color=INK)
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top",
                fontsize=6.8, color=MUTED, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))

    box(0.1, 5.3, 2.9, 1.5, "Theia3D (100 Hz)", "segment poses, joint centres,\nbody CoM, angles", LEFT)
    box(3.55, 5.3, 2.9, 1.5, "Split-belt plates (1000 Hz)", "force, moment and CoP per\nbelt; plate corners", RIGHT)
    box(7.0, 5.3, 2.9, 1.5, "MOVE4D scan", "the participant's boots;\nbuild_foot_binding.py (§3)", THIRD)
    box(1.2, 3.25, 7.6, 1.45, "detect_gait_events.py (§4)",
        "trusted force-plate events first (one boot on the belt, CoP under it)\n"
        "Zeni kinematic events for the rest; every heel strike and toe-off", VIOLET)
    box(0.1, 0.2, 3.1, 2.45, "Preparing a trial (§5)",
        "steps and steady walking\nbelt speed and direction\nboots over the pitched belt\n"
        "forces in Theia's frame\nload from quiet standing\nsystem CoM and velocity", INK)
    box(3.45, 0.2, 3.1, 2.45, "Per-step metrics (§6-11)",
        "spatiotemporal, posture\nkinetics, CoM work\nmargin of stability\n"
        "foot clearance, trip risk", INK)
    box(6.8, 0.2, 3.1, 2.45, "Per-trial metrics (§12-19)",
        "means, SD, CV with 95% CI\nsymmetry, DFA, entropy\nharmonic ratio, regularity\n"
        "GEM, foot placement, LDS\nresults workbook", INK)
    for x in (1.55, 5.0, 8.45):
        arrow(x, 5.3, x if x != 8.45 else 7.5, 4.7)
    arrow(5.0, 3.25, 1.65, 2.65)
    arrow(3.2, 1.4, 3.45, 1.4)
    arrow(6.55, 1.4, 6.8, 1.4)
    ax.text(5.0, 6.93, "gait_analysis.py runs everything below the event detection",
            ha="center", fontsize=7, color=MUTED, style="italic")
    save(fig, 1)


def fig_boot(d):
    """Fig 2: the boot sole and the toe hinge."""
    boots = d["boots"]
    t = d["t"]
    b = "R"
    v = boots[b]
    sole = t.boots.sole[b]
    sv, toe = sole["v"], sole["toe"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
    a = ax[0]
    a.scatter(v[::6, 1] * 1000, v[::6, 2] * 1000, s=1, c=GRID, label="boot mesh (every 6th vertex)")
    a.scatter(sv[~toe, 1] * 1000, sv[~toe, 2] * 1000, s=6, c=RIGHT, label="sole points: foot")
    a.scatter(sv[toe, 1] * 1000, sv[toe, 2] * 1000, s=6, c=THIRD, label="sole points: toe cap")
    a.set_aspect("equal")
    a.set_xlabel("foot frame, forward (mm)")
    a.set_ylabel("up (mm)")
    a.set_title("Sole: lowest vertex per cell")
    a.legend(loc="upper left", fontsize=6.5, markerscale=2)
    letter(a, "A")
    # toe hinge: the sole at push-off, rigid vs bent
    a = ax[1]
    mtp = np.median(sv[toe][:, 1]) - 0.03
    pitch = np.radians(-30)                  # heel up
    c, s = np.cos(pitch), np.sin(pitch)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    rigid = sv @ R.T
    bent = rigid.copy()
    th = np.radians(30)                      # toes extended back to flat
    ct, st = np.cos(th), np.sin(th)
    Rt = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    m = np.array([0, mtp, np.median(sv[:, 2])]) @ R.T
    bent[toe] = (rigid[toe] - m) @ Rt.T + m
    z0 = bent[:, 2].min()
    a.axhline(0, color=INK, lw=1)
    a.scatter(rigid[:, 1] * 1000, (rigid[:, 2] - z0) * 1000, s=4, c=MUTED, alpha=0.35, label="rigid boot")
    a.scatter(bent[~toe, 1] * 1000, (bent[~toe, 2] - z0) * 1000, s=5, c=RIGHT, label="bent at the MTP")
    a.scatter(bent[toe, 1] * 1000, (bent[toe, 2] - z0) * 1000, s=5, c=THIRD)
    a.set_aspect("equal")
    a.set_xlabel("forward (mm)")
    a.set_ylabel("height above belt (mm)")
    a.set_title("Push-off: toe cap follows the toes")
    a.legend(loc="upper left", fontsize=6.5, markerscale=2)
    letter(a, "B")
    fig.tight_layout(w_pad=2.5)
    save(fig, 2)


def fig_surface(d):
    """Fig 3: the belt surface fitted to the boots in stance."""
    t = d["t"]
    s = t.steps[t.steps["steady"]]
    surf = t.boots.surface
    fwd = np.asarray(surf["forward"])[:2]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for b in ga.LIMBS:
        xs, zs = [], []
        for st in s[s["limb"] == b].iloc[::3].itertuples():
            if not np.isfinite(st.to):
                continue
            rows = np.arange(int(st.hs + 0.2 * (st.to - st.hs)), int(st.hs + 0.5 * (st.to - st.hs)))
            W_ = t.boots.world(b, rows)
            k = W_[..., 2].argmin(1)
            p = W_[np.arange(len(rows)), k]
            xs.append(p[:, :2] @ fwd)
            zs.append(p[:, 2])
        xs, zs = np.concatenate(xs), np.concatenate(zs)
        ax.scatter(xs, zs * 1000, s=3, c=COLOUR[b], alpha=0.5, label=f"{ga.SIDE[b]} boot, lowest point in mid-stance")
        line = np.linspace(xs.min(), xs.max(), 10)
        ax.plot(line, (surf["floor"][b] + surf["slope"] * line) * 1000, color=COLOUR[b], lw=2)
    ax.set_xlabel("position along the walking direction (m)")
    ax.set_ylabel("Theia height (mm)")
    ax.set_title(f"Belt surface: one slope ({np.degrees(np.arctan(surf['slope'])):.2f}°), "
                 "one offset per boot")
    ax.legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    save(fig, 3)


def fig_contact_edges(d):
    """Fig 4: filtered vs raw force edges."""
    truth = d["truth"]
    import pandas as pd
    f = pd.read_csv(truth["force_dir"] / f"{truth['name']}.csv",
                    usecols=["Left belt_Force_Z"])
    fz = f["Left belt_Force_Z"].to_numpy()
    contacts, force, _ = dge.find_contacts(fz, "L")
    base, _ = dge.belt_baseline(fz)
    raw = fz - base
    c = [c for c in contacts if c["start"] > 60000][0]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    for a, centre, title, key in ((ax[0], c["start"], "Landing", "start"),
                                  (ax[1], c["stop"], "Lift-off", "stop")):
        sl = slice(centre - 25, centre + 25)
        tt = np.arange(sl.start, sl.stop) - centre
        a.plot(tt, raw[sl], color=GRID, lw=1.2, label="raw force")
        a.plot(tt, force[sl], color=LEFT, lw=1.8, label="filtered 50 Hz (zero lag)")
        a.axhline(dge.FORCE_THRESHOLD_N, color=MUTED, ls=":", lw=1)
        filt_cross = np.flatnonzero(np.diff((force[sl] > dge.FORCE_THRESHOLD_N).astype(int)))
        if len(filt_cross):
            a.axvline(tt[filt_cross[0]], color=LEFT, ls="--", lw=1)
        a.axvline(0, color=RIGHT, lw=1.6, label="edge on the raw force")
        a.set_title(title)
        a.set_xlabel("ms from the refined edge")
        a.set_ylim(-20, 250)
    ax[0].set_ylabel("vertical force above baseline (N)")
    ax[0].legend(fontsize=6.5, loc="upper left")
    ax[0].text(-24, dge.FORCE_THRESHOLD_N + 5, "20 N", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, 4)


def fig_trust(d):
    """Fig 5: a clean contact and a crossover, seen from above."""
    t, res, plates = d["t"], d["res"], d["plates"]
    A = res["registration"][0]
    belts = {b: dge.apply_2d(A, dge.polygon_ccw(plates[b]["outline"] / 1000))
             for b in ga.LIMBS}
    want = {"clean": None, "straddle": None}
    for c in res["contacts"]:
        if c["label"] in want and want[c["label"]] is None and c["start"] > 5000:
            want[c["label"]] = c
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.2))
    for a, (label, c) in zip(ax, want.items()):
        for b, poly in belts.items():
            a.add_patch(Polygon(poly, closed=True, fc=SHADE, ec=MUTED, lw=0.8))
        if c is None:
            a.set_title(f"{label}: none in this trial")
            continue
        row = int((c["start"] + c["stop"]) / 2 / 10)
        for b in ga.LIMBS:
            Wp = t.boots.world(b, [row])[0]
            h = t.boots.height(b, Wp[None])[0]
            touch = h < dge.CONTACT_HEIGHT_MM / 1000
            p = Wp[:, :2]
            hull = p[ConvexHull(p).vertices]
            a.add_patch(Polygon(hull, closed=True, fill=False, ec=COLOUR[b], lw=1.2))
            a.scatter(p[touch, 0], p[touch, 1], s=2, c=COLOUR[b])
        cop = t.cop[c["belt"]][int((c["start"] + c["stop"]) / 2)]
        a.plot(*cop, "X", color=INK, ms=7)
        a.set_aspect("equal")
        centre = t.boots.world(c["foot"] or "L", [row])[0][:, :2].mean(0)
        a.set_xlim(centre[0] - 0.45, centre[0] + 0.45)
        a.set_ylim(centre[1] - 0.45, centre[1] + 0.45)
        a.set_title(f"{label}: HS {'trusted' if c['hs_trusted'] else c['hs_reason']}",
                    fontsize=8)
        a.set_xticks([])
        a.set_yticks([])
    ax[0].text(0.02, 0.02, "grey: belts\ncoloured: boot outlines\ndots: touching sole points\n× CoP",
               transform=ax[0].transAxes, fontsize=6, color=MUTED)
    fig.tight_layout()
    save(fig, 5)


def fig_zeni(d):
    """Fig 6: Zeni signals with trusted GRF events."""
    t, res = d["t"], d["res"]
    fwd3 = np.r_[t.forward, 0]
    pel = t.kin("Pelvis_Position")
    r0 = int(t.steps.loc[t.steps["steady"], "hs"].iloc[20])
    rows = np.arange(r0, r0 + 330)
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for b in ga.LIMBS:
        heel = (t.kin(f"{ga.SIDE[b]}_Heel_Position") - pel) @ fwd3
        toe = -(t.kin(f"{ga.SIDE[b]}_Toes_Position") - pel) @ fwd3
        ax.plot(rows / 100, heel[rows], color=COLOUR[b], label=f"{ga.SIDE[b]} heel ahead of pelvis")
        ax.plot(rows / 100, toe[rows], color=COLOUR[b], ls="--", lw=1, label=f"{ga.SIDE[b]} toe behind pelvis")
        for e in res["events"]:
            if e["limb"] == b and rows[0] <= e["t"] <= rows[-1]:
                y = heel if e["event"] == ga.HS else toe
                ax.plot(e["t"] / 100, np.interp(e["t"], np.arange(len(y)), y),
                        "v" if e["event"] == ga.HS else "^", color=COLOUR[b], ms=6,
                        mec="white", mew=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance along travel (m)")
    ax.set_title("Zeni: heel strike at the heel's maximum, toe-off at the toe's; "
                 "markers = trusted GRF events")
    ax.legend(ncol=4, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    save(fig, 6)


def fig_timing(d):
    """Fig 7: the events of one stride and what is measured between them."""
    t = d["t"]
    s = t.steps[t.steps["steady"] & (t.steps["limb"] == "L")].iloc[10]
    nxt = t.steps[(t.steps["limb"] == "R") & (t.steps["hs"] < s.hs)].iloc[-1]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.grid(False)
    y = {"L": 1.0, "R": 0.0}
    ax.broken_barh([(s.hs, s.to - s.hs)], (y["L"] - 0.3, 0.6), fc=LEFT, alpha=0.85)
    ax.broken_barh([(nxt.hs, nxt.to - nxt.hs)], (y["R"] - 0.3, 0.6), fc=RIGHT, alpha=0.85)
    ax.broken_barh([(s.contra_hs, s.next_hs + 40 - s.contra_hs)], (y["R"] - 0.3, 0.6), fc=RIGHT, alpha=0.85)
    ax.broken_barh([(s.next_hs, 30)], (y["L"] - 0.3, 0.6), fc=LEFT, alpha=0.85)
    for x, lab in ((s.hs, "L HS"), (s.contra_to, "R TO"), (s.contra_hs, "R HS"),
                   (s.to, "L TO"), (s.next_hs, "L HS")):
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax.text(x, 1.55, lab, ha="center", fontsize=7, color=INK)

    def span(x0, x1, yy, text):
        ax.annotate("", xy=(x1, yy), xytext=(x0, yy), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
        ax.text((x0 + x1) / 2, yy - 0.16, text, ha="center", va="top", fontsize=6.8)
    span(s.hs, s.contra_to, -0.75, "initial\ndouble support")
    span(s.contra_to, s.contra_hs, -0.75, "single support")
    span(s.contra_hs, s.to, -0.75, "terminal\ndouble support")
    span(s.to, s.next_hs, -0.75, "swing (left)")
    span(s.hs, s.to, 1.95, "stance (left)")
    span(s.hs, s.next_hs, 2.4, "stride (left)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["right foot\non the belt", "left foot\non the belt"])
    ax.set_ylim(-1.35, 2.6)
    ax.set_xlim(s.hs - 15, s.next_hs + 20)
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    save(fig, 7)


def fig_belt(d):
    """Fig 8: belt speed from the stance heels."""
    t = d["t"]
    fwd3 = np.r_[t.forward, 0]
    s = t.steps[t.steps["steady"] & (t.steps["limb"] == "L")].iloc[5:9]
    fig, ax = plt.subplots(figsize=(W, 2.4))
    heel = t.heel["L"] @ fwd3
    a0, a1 = int(s["hs"].iloc[0]) - 20, int(s["next_hs"].iloc[-1])
    rows = np.arange(a0, a1)
    ax.plot(rows / 100, heel[rows], color=GRID, lw=1.5, label="left heel, along travel")
    for st in s.itertuples():
        a = int(st.hs + ga.BELT_FLAT[0] * (st.to - st.hs))
        b = int(st.hs + ga.BELT_FLAT[1] * (st.to - st.hs))
        rr = np.arange(a, b)
        p = np.polyfit(rr / 100, heel[rr], 1)
        ax.axvspan(a / 100, b / 100, color=SHADE, lw=0)
        ext = np.arange(int(st.hs), int(st.to))
        ax.plot(ext / 100, np.polyval(p, ext / 100), color=LEFT, lw=1, ls="--")
        ax.plot(rr / 100, np.polyval(p, rr / 100), color=LEFT, lw=2.4)
        ax.text(b / 100 + 0.04, np.polyval(p, a / 100) + 0.02, f"{-p[0]:.3f} m/s",
                fontsize=6.5, color=INK)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("heel position (m)")
    ax.set_title(f"Heel speed over 25-55% of each stance (shaded); "
                 f"median of all stances {t.belt_speed:.3f} m/s")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, 8)


def fig_axes_check(d):
    """Fig 9: GRF/m against the tracked CoM's acceleration."""
    from scipy.signal import savgol_filter
    t = d["t"]
    com, _ = ga.fill_gaps(t.com)
    a_mk = savgol_filter(com, 11, 3, deriv=2, delta=0.01, axis=0)
    total = t.grf_raw["L"] + t.grf_raw["R"]
    acc = (total - total.mean(0)) / t.mass
    a100 = acc[np.arange(t.n) * 10]
    band = butter(2, [0.5, 5.0], btype="band", fs=100, output="sos")
    r0 = int(t.steps.loc[t.steps["steady"], "hs"].iloc[30])
    rows = np.arange(r0, r0 + 300)
    fwd3, lat3 = np.r_[t.forward, 0], np.r_[t.lateral, 0]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), sharex=True)
    for a, axis, name in ((ax[0], lat3, "across the belt"), (ax[1], fwd3, "along the belt")):
        mk = sosfiltfilt(band, a_mk @ axis)[rows]
        fp = sosfiltfilt(band, a100 @ axis)[rows]
        a.plot(rows / 100, mk, color=GRID, lw=2.2, label="tracked CoM, differentiated twice")
        a.plot(rows / 100, fp, color=LEFT, lw=1.1, label="GRF / system mass")
        a.set_title(f"{name}: r = {np.corrcoef(mk, fp)[0, 1]:.2f}")
        a.set_xlabel("time (s)")
    ax[0].set_ylabel("acceleration, 0.5-5 Hz (m/s²)")
    h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l, ncol=2, fontsize=7, loc="lower center")
    save(fig, 9)


def fig_quiet(d):
    """Fig 10: weighing in the quiet standing."""
    t = d["t"]
    L = t.load
    n = 9000
    tt = np.arange(n) / 1000
    fig, ax = plt.subplots(figsize=(W, 2.4))
    tot = t.grf["L"][:n, 2] + t.grf["R"][:n, 2]
    ax.axvspan(L["start_s"], L["stop_s"], color=SHADE, lw=0)
    for b in ga.LIMBS:
        ax.plot(tt, t.grf[b][:n, 2], color=COLOUR[b], lw=1, label=f"{ga.SIDE[b]} belt")
    ax.plot(tt, tot, color=INK, lw=1.2, label="total")
    ax.axhline(L["weight_n"], color=MUTED, ls=":", lw=1)
    ax.text(0.1, 1330, f"W = {L['weight_n']:.0f} N,  m = W/g = {t.mass:.1f} kg\n"
            f"load = m - body mass = {L['load_kg']:.1f} kg", fontsize=7, color=INK, va="top")
    ax.text((L["start_s"] + L["stop_s"]) / 2, 150, "steadiest 2 s\nstanding still",
            ha="center", fontsize=6.8, color=MUTED)
    first = t.events["row"].min() / 100
    ax.axvline(first, color=MUTED, lw=0.8)
    ax.text(first - 0.05, 60, "first step", fontsize=6.5, color=MUTED, ha="right")
    ax.set_xlabel("time from the start of the trial (s)")
    ax.set_ylabel("vertical force (N)")
    ax.set_ylim(0, 1450)
    ax.legend(fontsize=6.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()
    save(fig, 10)


def fig_load(d):
    """Fig 11: where the load is, and the system CoM."""
    t = d["t"]
    L = t.load
    a, b = L["rows"]
    body = np.nanmean(t.com_body[a:b], 0)
    system = np.nanmean(t.com[a:b], 0)
    load = np.nanmean(t.load_position[a:b], 0)
    lo = np.nanmean(t.kin("Low_Back_Position")[a:b], 0)
    hi = np.nanmean(t.kin("Neck_Position")[a:b], 0)
    fwd3 = np.r_[t.forward, 0]
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.1))
    a_ = ax[0]
    def f(p):
        return p @ fwd3
    a_.plot([f(lo), f(hi)], [lo[2], hi[2]], color=MUTED, lw=4, solid_capstyle="round", label="trunk (Low_Back -> Neck)")
    a_.plot(f(body), body[2], "o", color=LEFT, ms=8, label="Theia body CoM")
    a_.plot(f(load), load[2], "s", color=RIGHT, ms=8, label=f"load ({L['load_kg']:.1f} kg)")
    a_.plot(f(system), system[2], "D", color=INK, ms=7, label="system CoM")
    a_.plot([f(body), f(load)], [body[2], load[2]], color=GRID, lw=1, ls="--")
    cop = np.r_[L["cop_xy"], 0] @ fwd3
    a_.axvline(cop, color=THIRD, lw=1.5, ls=":", label="standing CoP (vertical line)")
    a_.set_xlabel("along travel (m)")
    a_.set_ylabel("height (m)")
    a_.set_title("Side view, quiet standing")
    a_.legend(fontsize=6, loc="lower left")
    letter(a_, "A")
    a_ = ax[1]
    a_.axis("off")
    a_.text(0.0, 0.95,
            "Standing still, the CoP lies directly below\nthe system CoM. So, horizontally:\n\n"
            "   x_load = (m x_CoP - m_body x_CoM,body) / m_load\n\n"
            "height: at Trunk_Position (not observable)\n"
            "left-right: on the mid-line\n\n"
            "The load is then carried in the trunk's\nframe through the whole trial, and\n\n"
            "   x_CoM = (m_body x_CoM,body + m_load x_load) / m\n\n"
            f"Here: the load is {abs(1000 * L['load_offset_local'][1]):.0f} mm behind the trunk\n"
            f"(true 150 mm); the system CoM is\n"
            f"{1000 * np.linalg.norm(system - body):.0f} mm from the body's.",
            va="top", fontsize=7.2, family="DejaVu Sans Mono", color=INK)
    fig.tight_layout()
    save(fig, 11)


def fig_fusion(d):
    """Fig 12: the complementary filter."""
    t, truth = d["t"], d["truth"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw=dict(width_ratios=[1, 1.5]))
    b, a = butter(2, ga.CROSSOVER_HZ, fs=100)
    from scipy.signal import freqz
    w, h = freqz(b, a, worN=4096, fs=100)
    lp = np.abs(h) ** 2                     # filtfilt: squared magnitude
    ax[0].semilogx(w[1:], lp[1:], color=LEFT, label="markers: low-pass")
    ax[0].semilogx(w[1:], 1 - lp[1:], color=RIGHT, label="force: high-pass")
    ax[0].semilogx(w[1:], np.ones(len(w) - 1), color=INK, ls=":", lw=1, label="sum = 1")
    ax[0].axvline(0.9, color=GRID, lw=6, alpha=0.8)
    ax[0].text(1.15, 0.45, "stride\nfrequency", fontsize=6.5, color=MUTED)
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("gain")
    ax[0].set_title("The two branches")

    letter(ax[0], "A")
    r0 = int(t.steps.loc[t.steps["steady"], "hs"].iloc[40])
    rows = np.arange(r0, r0 + 250)
    lat3 = np.r_[t.lateral, 0]
    true = truth["com_velocity"][rows] @ lat3
    ax[1].plot(rows / 100, t.com_vel_markers[rows] @ lat3, color=GRID, lw=2, label="markers alone")
    ax[1].plot(rows / 100, true, color=THIRD, lw=3, alpha=0.6, label="truth")
    ax[1].plot(rows / 100, t.com_vel[rows] @ lat3, color=INK, lw=1, label="fused")
    ok = np.isfinite(t.com_vel).all(1)
    e_m = 1000 * np.sqrt(np.nanmean(((t.com_vel_markers - truth["com_velocity"]) @ lat3)[ok] ** 2))
    e_f = 1000 * np.sqrt(np.nanmean(((t.com_vel - truth["com_velocity"]) @ lat3)[ok] ** 2))
    ax[1].set_title(f"Sideways CoM velocity: error {e_m:.0f} -> {e_f:.0f} mm/s")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("m/s")
    letter(ax[1], "B")
    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(h0 + h1, l0 + l1, ncol=6, fontsize=6.5, loc="lower center")
    save(fig, 12)


def fig_spatial(d):
    """Fig 13: step length, step width and stride over the belt."""
    t = d["t"]
    s = t.steps[t.steps["steady"]].iloc[20:25]
    fwd3, lat3 = np.r_[t.forward, 0], np.r_[t.lateral, 0]
    fig, ax = plt.subplots(2, 1, figsize=(W, 4.6), gridspec_kw=dict(height_ratios=[1, 1.15]))
    a = ax[0]
    st = s.iloc[2]
    r = int(round(st.hs))
    for b in ga.LIMBS:                        # the boots at that instant
        p = t.boots.world(b, [r])[0][:, :2]
        pp = np.column_stack([p @ t.forward, p @ t.lateral])
        a.add_patch(Polygon(pp[ConvexHull(pp).vertices], closed=True, fc=COLOUR[b],
                            alpha=0.15, ec=COLOUR[b], lw=1))
    me = ga.at(t.heel[st.limb], [st.hs])[0]
    other = ga.at(t.heel[ga.OTHER[st.limb]], [st.hs])[0]
    x0, x1 = other @ fwd3, me @ fwd3          # along travel
    y0, y1 = other @ lat3, me @ lat3          # to the left
    a.annotate("", xy=(x1, y0), xytext=(x0, y0),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
    a.annotate("", xy=(x1, y1), xytext=(x1, y0),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
    a.plot([x1, x1], [y0, y1], color=INK, lw=0)
    a.text((x0 + x1) / 2, y0 + 0.02 * np.sign(y1 - y0), f"step length {st.step_length_m:.3f} m",
           ha="center", va="bottom" if y1 > y0 else "top", fontsize=7)
    a.text(x1 + 0.03, (y0 + y1) / 2, f"step width\n{st.step_width_m:.3f} m", va="center", fontsize=7)
    a.plot(x1, y1, "o", color=COLOUR[st.limb], ms=8,
           label=f"heel of the landing foot ({ga.SIDE[st.limb].lower()})")
    a.plot(x0, y0, "o", mfc="white", mec=COLOUR[ga.OTHER[st.limb]], ms=8, mew=2,
           label="the other heel, on the belt")
    a.set_aspect("equal")
    a.set_xlim(min(x0, x1) - 0.35, max(x0, x1) + 0.45)
    a.set_xlabel("along travel (m)  ->  walking direction")
    a.set_ylabel("to the left (m)")
    a.set_title("At a heel strike, from above")
    a.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    letter(a, "A")
    a = ax[1]
    st = t.steps[t.steps["steady"] & (t.steps["limb"] == "L")].iloc[15]
    rows = np.arange(int(st.hs), int(st.to))
    y = t.heel["L"][rows] @ fwd3
    ride = y[0] - t.belt_speed * (rows - rows[0]) / 100
    a.plot((rows - rows[0]) / 100, 1000 * (y - ride), color=LEFT, lw=1.8, label="heel ahead of a point riding the belt")
    a.axvline((st.contra_hs - st.hs) / 100, color=RIGHT, ls="--", lw=1.2, label="other foot's heel strike")
    a.set_xlabel("time since this heel strike (s)")
    a.set_ylabel("mm")
    a.set_title("One stance: the heel against a point that rides the belt")
    a.legend(fontsize=6.5, loc="lower left")
    letter(a, "B")
    fig.tight_layout()
    save(fig, 13)


def fig_posture(d):
    """Fig 14: stride-normalised angles."""
    t = d["t"]
    aw = t.angle_waves
    limb = np.array(aw["limb"])
    pct = np.linspace(0, 100, ga.WAVE_POINTS)
    fig, ax = plt.subplots(1, 4, figsize=(W, 2.0))
    for a, name, label in zip(ax, ("trunk_lean", "hip", "knee", "ankle"),
                              ("trunk lean", "hip", "knee", "ankle")):
        w = np.array(aw[name])[limb == "L"]
        w = w[np.isfinite(w).all(1)]
        m, sd = w.mean(0), w.std(0)
        a.fill_between(pct, m - sd, m + sd, color=LEFT, alpha=0.2, lw=0)
        a.plot(pct, m, color=LEFT)
        a.set_title(label)
        a.set_xlabel("% stride")
        a.annotate("", xy=(np.argmax(m), m.max()), xytext=(np.argmax(m), m.min()),
                   arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
    ax[0].set_ylabel("degrees")
    fig.tight_layout()
    save(fig, 14)


def fig_kinetics(d):
    """Fig 15: GRF landmarks and impulses of one trusted stance."""
    t = d["t"]
    s = t.steps[t.steps["steady"] & t.steps["kinetics_trusted"]].iloc[30]
    a0, a1 = int(s.hs_sample), int(s.to_sample)
    F = t.grf[s.hs_belt][a0:a1]
    bw = t.body_mass * ga.G
    vt, ap = F[:, 2] / bw, F @ np.r_[t.forward, 0] / bw
    tt = np.arange(len(vt)) / 1000
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5))
    a = ax[0]
    a.plot(tt, vt, color=INK)
    mid = len(vt) // 2
    i1, i2 = int(np.argmax(vt[:mid])), mid + int(np.argmax(vt[mid:]))
    it = i1 + int(np.argmin(vt[i1:i2]))
    for i, lab in ((i1, "F1"), (it, "trough"), (i2, "F2")):
        a.plot(tt[i], vt[i], "o", color=RIGHT, ms=6)
        a.text(tt[i], vt[i] + 0.05, lab, ha="center", fontsize=7)
    lo = int(np.argmax(vt[:i1 + 1] >= 0.2 * vt[i1]))
    hi = int(np.argmax(vt[:i1 + 1] >= 0.8 * vt[i1]))
    p = np.polyfit(tt[lo:hi + 1], vt[lo:hi + 1], 1)
    a.plot(tt[lo:hi + 1], np.polyval(p, tt[lo:hi + 1]), color=LEFT, lw=3, alpha=0.7)
    a.text(tt[hi] + 0.03, 0.45, f"loading rate\n{p[0]:.1f} BW/s", fontsize=6.8, color=LEFT)
    a.set_xlabel("time in stance (s)")
    a.set_ylabel("vertical GRF (body weights)")
    a.set_title("Vertical force (carrying 20 kg)")
    letter(a, "A")
    a = ax[1]
    a.plot(tt, ap, color=INK)
    a.fill_between(tt, ap, 0, where=ap < 0, color=LEFT, alpha=0.35, lw=0, label=f"braking impulse {np.sum(np.clip(ap, None, 0)) / 1000:.3f} BW·s")
    a.fill_between(tt, ap, 0, where=ap > 0, color=RIGHT, alpha=0.35, lw=0, label=f"propulsive impulse {np.sum(np.clip(ap, 0, None)) / 1000:.3f} BW·s")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xlabel("time in stance (s)")
    a.set_ylabel("AP GRF (body weights)")
    a.set_title("Along the belt (positive = forward)")
    a.legend(fontsize=6.5, loc="lower right")
    letter(a, "B")
    fig.tight_layout()
    save(fig, 15)


def fig_work(d):
    """Fig 16: each leg's power into the CoM over a stride."""
    t = d["t"]
    s = t.steps[t.steps["steady"] & t.steps["kinetics_trusted"] & (t.steps["limb"] == "L")].iloc[20]
    a0, a1 = int(s.hs * 10), int(s.next_hs * 10)
    v = ga.at(t.com_vel_belt, np.arange(a0, a1) / 10)
    tt = (np.arange(a0, a1) - a0) / 1000
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for b in ga.LIMBS:
        P = np.sum(t.grf[b][a0:a1] * v, axis=1)
        ax.plot(tt, P, color=COLOUR[b], label=f"{ga.SIDE[b]} leg: P = F · v_CoM")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    for k, (x0, x1, lab) in enumerate(((s.hs, s.contra_to, "collision\n(L leading)"),
                                       (s.contra_to, s.contra_hs, "rebound + preload\n(L single support)"),
                                       (s.contra_hs, s.to, "push-off\n(L trailing)"),
                                       (s.to, s.next_hs, "L swing\n(R stance)"))):
        ax.axvspan((x0 - s.hs) / 100, (x1 - s.hs) / 100, color=SHADE if k % 2 == 0 else "white",
                   lw=0, zorder=0)
        ax.text(((x0 + x1) / 2 - s.hs) / 100, hi + 0.3 * (hi - lo), lab, ha="center",
                va="center", fontsize=6.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("time from left heel strike (s)")
    ax.set_ylabel("power into the CoM (W)")
    ax.set_title("Individual limbs method: work = area under each leg's power in each phase")
    ax.legend(fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()
    save(fig, 16)


def fig_mos(d):
    """Fig 17: the margin of stability at heel strike."""
    t = d["t"]
    st = t.steps[t.steps["steady"] & (t.steps["limb"] == "R")].iloc[25]
    r = int(np.ceil(st.hs))
    fwd, lat = t.forward, t.lateral
    fig, ax = plt.subplots(2, 1, figsize=(W, 4.8), gridspec_kw=dict(height_ratios=[1, 1.1]))
    a = ax[0]

    def plane(p):
        return np.column_stack([p @ fwd, p @ lat])        # (forward, left)
    hulls = {}
    for b in ga.LIMBS:
        Wp = t.boots.world(b, [r])[0][:, :2]
        pp = plane(Wp)
        hulls[b] = pp[ConvexHull(pp).vertices]
        a.add_patch(Polygon(hulls[b], closed=True, fc=COLOUR[b], alpha=0.25, ec=COLOUR[b], lw=1.3))
    com = plane(t.com[r, :2][None])[0]
    L_ = np.linalg.norm(t.com[r] - t.ankle["R"][r])
    xcom = plane((t.com[r, :2] + t.com_vel_belt[r, :2] / np.sqrt(ga.G / L_))[None])[0]
    a.annotate("", xy=xcom, xytext=com, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    a.plot(*com, "o", color=INK, ms=7, label="system CoM")
    a.plot(*xcom, "o", mfc="white", mec=INK, ms=7, mew=1.5, label="xCoM = CoM + v/ω₀")
    edge = hulls["R"][:, 1].min()             # right boot's outer (right) edge
    front = hulls["R"][:, 0].max()            # right boot's front
    a.plot([xcom[0], xcom[0]], [xcom[1], edge], color=THIRD, lw=2.5,
           label=f"ML margin {1000 * st.mos_ml_contact:.0f} mm")
    a.plot([xcom[0], front], [xcom[1], xcom[1]], color=VIOLET, lw=2.5,
           label=f"AP margin {1000 * st.mos_ap_contact:.0f} mm")
    a.set_aspect("equal")
    allp = np.vstack(list(hulls.values()))
    a.set_xlim(allp[:, 0].min() - 0.1, allp[:, 0].max() + 0.1)
    a.set_ylim(allp[:, 1].min() - 0.08, allp[:, 1].max() + 0.08)
    a.set_xlabel("along travel (m)  ->  walking direction")
    a.set_ylabel("to the left (m)")
    a.set_title("Right heel strike, from above")
    a.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    letter(a, "A")
    a = ax[1]
    rows = np.arange(int(np.ceil(st.hs)), int(np.floor(st.to)) + 1)
    side = np.sign((t.ankle["R"][rows[0], :2] - t.ankle["L"][rows[0], :2]) @ lat)
    edge = t.boots.ext["R"]["lat_max"][rows] if side > 0 else -t.boots.ext["R"]["lat_min"][rows]
    length = np.linalg.norm(t.com[rows] - t.ankle["R"][rows], axis=1)
    xc = t.com[rows, :2] + t.com_vel_belt[rows, :2] / np.sqrt(ga.G / length)[:, None]
    ml = edge - side * (xc @ lat)
    ap = t.boots.ext["R"]["fwd_max"][rows] - xc @ fwd
    tt = (rows - rows[0]) / 100
    a.axvspan((st.contra_to - st.hs) / 100, (st.contra_hs - st.hs) / 100, color=SHADE, lw=0)
    a.plot(tt, 1000 * ml, color=THIRD, label="ML margin")
    a.plot(tt, 1000 * ap, color=VIOLET, label="AP margin")
    a.axhline(0, color=MUTED, lw=0.8)
    a.text(((st.contra_to + st.contra_hs) / 2 - st.hs) / 100, 1000 * np.nanmin(ap) * 0.85,
           "single support:\nminima read here", ha="center", fontsize=6.5, color=INK)
    a.set_xlabel("time from heel strike (s)")
    a.set_ylabel("margin (mm)")
    a.set_title("Through the stance")
    a.legend(fontsize=6.5, loc="lower left")
    letter(a, "B")
    fig.tight_layout()
    save(fig, 17)


def fig_mfc(d):
    """Fig 18: MFC and the trip risk integral in one swing."""
    t = d["t"]
    st = t.steps[t.steps["steady"] & (t.steps["limb"] == "L") & np.isfinite(t.steps["tri_s"])].iloc[20]
    pad = ga.SWING_TRIM * (st.next_hs - st.to)
    rows = np.arange(int(np.ceil(st.to + pad)), int(np.floor(st.next_hs - pad)) + 1)
    Wp = t.boots.world("L", rows)
    H = t.boots.height("L", Wp)
    clear = H.min(1)
    speed = np.linalg.norm(np.gradient(Wp.mean(1), 0.01, axis=0), axis=1)
    fast = speed >= np.quantile(speed, ga.MFC_SPEED_QUANTILE)
    anterior = np.maximum(t.boots.ext["L"]["fwd_max"][rows], t.boots.ext["R"]["fwd_max"][rows])
    length = np.linalg.norm(t.com[rows] - t.ankle["R"][rows], axis=1)
    xcom = t.com[rows, :2] @ t.forward + (t.com_vel_belt[rows, :2] @ t.forward) / np.sqrt(ga.G / length)
    moi = 1000 * np.maximum(xcom - anterior, 0)
    risk = moi / np.maximum(1000 * clear, 1)
    v = int(st.mfc_vertex)
    sp = np.linalg.norm(np.gradient(Wp[:, v], 0.01, axis=0), axis=1)
    acc = np.gradient(sp, 0.01)
    i0, i1 = sorted((int(np.argmax(acc)), int(np.argmin(acc))))
    tt = (rows - rows[0]) / 100
    fig, ax = plt.subplots(3, 1, figsize=(W, 4.2), sharex=True)
    a = ax[0]
    a.fill_between(tt, 0, 1000 * clear.max(), where=fast, color=SHADE, lw=0, label="fastest 25% of the swing")
    a.plot(tt, 1000 * clear, color=LEFT, label="lowest sole point above the belt")
    k = int(st.mfc_row - rows[0])
    a.plot(tt[k], 1000 * clear[k], "v", color=RIGHT, ms=8, label=f"MFC {1000 * st.mfc_m:.1f} mm")
    a.set_ylabel("clearance (mm)")
    a.legend(fontsize=6.5, loc="upper right")
    letter(a, "A")
    a = ax[1]
    a.plot(tt, moi, color=VIOLET)
    a.set_ylabel("MoI (mm)")
    a.text(0.99, 0.8, "how far the xCoM is ahead of\nthe front of either boot", transform=a.transAxes,
           fontsize=6.5, color=MUTED, ha="right", va="top")
    letter(a, "B")
    a = ax[2]
    a.plot(tt, risk, color=INK)
    a.fill_between(tt[i0:i1 + 1], 0, risk[i0:i1 + 1], color=RIGHT, alpha=0.3, lw=0,
                   label=f"TRI = {st.tri_s:.2f} s\n(area from peak acceleration to\npeak deceleration of the MFC point)")
    a.set_ylabel("MoI / clearance")
    a.set_xlabel("time from toe-off (s)")
    a.legend(fontsize=6.5, loc="upper right")
    letter(a, "C")
    fig.tight_layout()
    save(fig, 18)


def fig_bootstrap():
    """Fig 19: the circular block bootstrap."""
    rng = np.random.default_rng(3)
    n = 60
    e = rng.normal(size=n + 50)
    x = np.zeros(n + 50)
    for i in range(1, n + 50):
        x[i] = 0.7 * x[i - 1] + e[i]
    x = x[50:]
    blk = 6
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    a = ax[0]
    starts = [3, 40, 17, 52, 28]
    cols = [LEFT, RIGHT, THIRD, VIOLET, YELLOW]
    a.plot(x, color=GRID, lw=1.2)
    for s0, c in zip(starts, cols):
        idx = (np.arange(blk) + s0) % n
        a.plot(idx, x[idx], "o-", color=c, ms=3, lw=1.5)
    a.set_title("Five blocks of 6 strides")
    a.set_xlabel("stride")
    letter(a, "A")
    a = ax[1]
    pos = 0
    for s0, c in zip(starts, cols):
        idx = (np.arange(blk) + s0) % n
        a.plot(np.arange(pos, pos + blk), x[idx], "o-", color=c, ms=3, lw=1.5)
        pos += blk
    a.set_title("Glued: one resample")
    a.set_xlabel("stride of the resample")
    letter(a, "B")
    a = ax[2]
    phis = np.linspace(0, 0.9, 10)
    bl = []
    for p in phis:
        vals = []
        for k in range(20):
            e = rng.normal(size=450)
            y = np.zeros(450)
            for i in range(1, 450):
                y[i] = p * y[i - 1] + e[i]
            vals.append(ga.optimal_block_length(y[50:]))
        bl.append(np.median(vals))
    a.plot(phis, bl, "o-", color=INK, ms=4)
    a.set_xlabel("stride-to-stride correlation")
    a.set_ylabel("block length (strides)")
    a.set_title("Block length")
    letter(a, "C")
    fig.tight_layout()
    save(fig, 19)


def fig_dfa(d):
    """Fig 20: DFA step by step."""
    t = d["t"]
    s = ga.steady_steps(t, "R")
    x = s["step_width_m"].to_numpy()
    x = x[np.isfinite(x)]
    res = ga.dfa(x)
    y = np.cumsum(x - x.mean())
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.4))
    a = ax[0]
    a.plot(1000 * x, color=LEFT, lw=0.6)
    a.axhline(1000 * x.mean(), color=INK, lw=0.8, ls="--")
    a.set_title("Series (step width)")
    a.set_xlabel("stride")
    a.set_ylabel("mm")
    letter(a, "A")
    a = ax[1]
    a.plot(y, color=INK, lw=0.9)
    for nbox, c in ((res["boxes"][0], LEFT), (res["boxes"][-1], RIGHT)):
        for k in range(len(y) // nbox):
            seg = y[k * nbox:(k + 1) * nbox]
            tt = np.arange(nbox)
            a.plot(k * nbox + tt, np.polyval(np.polyfit(tt, seg, 1), tt), color=c, lw=1.4)
    a.set_title(f"Profile; boxes of {res['boxes'][0]}, {res['boxes'][-1]}")
    a.set_xlabel("stride")
    letter(a, "B")
    a = ax[2]
    ln, lf = np.log10(res["boxes"]), np.log10(res["fluct"])
    a.plot(ln, lf, "o", color=INK, ms=4)
    p = np.polyfit(ln, lf, 1)
    a.plot(ln, np.polyval(p, ln), color=RIGHT, label=f"slope α = {res['alpha']:.2f}")
    a.plot(ln, lf[0] + 0.5 * (ln - ln[0]), color=MUTED, ls=":", lw=1, label="α = 0.5 (no memory)")
    a.set_title("log F(n) vs log n")
    a.set_xlabel("log10 box size n")
    a.set_ylabel("log10 F(n)")
    a.legend(fontsize=6.5)
    letter(a, "C")
    fig.tight_layout()
    save(fig, 20)


def fig_entropy(d):
    """Fig 21: sample entropy templates and the multiscale curve."""
    t, extras = d["t"], d["extras"]
    s = ga.steady_steps(t, "R")
    x = s["step_width_m"].to_numpy()
    x = x[np.isfinite(x)][20:120]
    z = (x - x.mean()) / x.std()
    m, r = 2, 0.2

    def matches(i):
        tpl = z[i:i + m + 1]
        mm, mm1 = [], []
        for j in range(len(z) - m):
            if j != i and np.max(np.abs(z[j:j + m] - tpl[:m])) <= r:
                mm.append(j)
                if abs(z[j + m] - tpl[m]) <= r:
                    mm1.append(j)
        return mm, mm1
    # show a template that has a few matches of each kind
    i = max(range(len(z) - m), key=lambda k: (min(len(matches(k)[1]), 3), -abs(len(matches(k)[0]) - 8)))
    matched_m, matched_m1 = matches(i)
    tpl = z[i:i + m + 1]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
    a = ax[0]
    a.plot(z, color=GRID, lw=1)
    for k in range(m + 1):                   # the tolerance band around each template point
        a.axhspan(tpl[k] - r, tpl[k] + r, color=[LEFT, LEFT, RIGHT][k], alpha=0.08, lw=0)
    for j in matched_m:
        c = RIGHT if j in matched_m1 else LEFT
        a.plot(np.arange(j, j + m + 1), z[j:j + m + 1], "o-", color=c, ms=3, lw=1)
    a.plot(np.arange(i, i + m + 1), tpl, "o-", color=INK, ms=4, lw=2)
    a.set_title(f"Matches: {len(matched_m)} (m = 2), {len(matched_m1)} (m = 3)")
    a.set_xlabel("stride")
    a.set_ylabel("step width (SD units)")
    letter(a, "A")
    a = ax[1]
    mse = extras["mse"]
    for col, c in (("AP", LEFT), ("VT", THIRD), ("ML", RIGHT)):
        a.plot(mse.index, mse[col], color=c, label=f"trunk {col}")
    a.plot(mse.index, mse["white_noise"], color=MUTED, ls="--", lw=1, label="white noise")
    a.set_xlabel("coarse-graining scale (samples)")
    a.set_ylabel("sample entropy")
    a.set_title("Multiscale entropy (trunk)")
    a.legend(fontsize=6.5)
    letter(a, "B")
    fig.tight_layout()
    save(fig, 21)


def fig_trunk(d):
    """Fig 22: harmonic ratio and regularity."""
    t = d["t"]
    sig = ga.trunk_signal(t)
    s = ga.steady_steps(t, "R")
    s = s[np.isfinite(s["next_hs"])]
    amps = {"AP": [], "ML": []}
    waves = {"AP": [], "VT": [], "ML": []}
    for a0, b0 in zip(s["hs"], s["next_hs"]):
        a0, b0 = int(round(a0)), int(round(b0))
        seg = sig["source"][a0:b0]
        if not np.isfinite(seg).all():
            continue
        for ax_, lab in ((1, "AP"), (0, "ML")):
            amps[lab].append(ga.harmonics(seg[:, ax_], power=sig["power"]))
        for ax_, lab in ((1, "AP"), (2, "VT"), (0, "ML")):
            w = sig["acc"][a0:b0, ax_]
            waves[lab].append(np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, len(w)), w))
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.4))
    a = ax[0]
    for lab, c in (("AP", LEFT), ("VT", THIRD), ("ML", RIGHT)):
        w = np.nanmean(waves[lab], axis=0)
        a.plot(np.linspace(0, 100, 101), w, color=c, label=lab)
    a.set_xlabel("% stride")
    a.set_ylabel("trunk acceleration (m/s²)")
    a.set_title("Mean stride")
    a.legend(fontsize=6.5)
    letter(a, "A")
    a = ax[1]
    amp = np.nanmean(amps["AP"], axis=0)
    k = np.arange(1, 21)
    a.bar(k[k % 2 == 0], amp[k % 2 == 0], color=LEFT, width=0.8, label="even (intrinsic)")
    a.bar(k[k % 2 == 1], amp[k % 2 == 1], color=RIGHT, width=0.8, label="odd")
    hr, ihr = ga.harmonic_ratio(amp)
    a.set_title(f"AP: HR {hr:.1f}, iHR {ihr:.0f}%")
    a.set_ylabel("amplitude")
    a.set_xlabel("harmonic of stride frequency")
    a.legend(fontsize=6.5)
    letter(a, "B")
    a = ax[2]
    stride_rows = np.median(s["next_hs"] - s["hs"])
    steady = sig["acc"][int(s["hs"].iloc[0]):int(s["next_hs"].iloc[-1])]
    ac = ga.unbiased_autocorr(steady[:, 2], int(1.6 * stride_rows))
    reg = ga.regularity(ac, stride_rows)
    a.plot(np.arange(len(ac)) / 100, ac, color=THIRD)
    a.plot(reg["step_lag"] / 100, reg["step_regularity"], "o", color=INK, ms=6)
    a.plot(reg["stride_lag"] / 100, reg["stride_regularity"], "s", color=INK, ms=6)
    a.text(reg["step_lag"] / 100, reg["step_regularity"] + 0.12, "step", ha="center", fontsize=6.5)
    a.text(reg["stride_lag"] / 100, reg["stride_regularity"] + 0.12, "stride", ha="center", fontsize=6.5)
    a.set_xlabel("lag (s)")
    a.set_title("VT autocorrelation")
    a.set_ylim(-1.1, 1.3)
    letter(a, "C")
    fig.tight_layout()
    save(fig, 22)


def fig_control(d):
    """Fig 23: GEM and foot placement."""
    t = d["t"]
    s = ga.steady_steps(t, "R")
    g = ga.gem_decompose(s["stride_s"].to_numpy(float), s["stride_length_m"].to_numpy(float))
    Th = (g["parallel"] - g["perpendicular"]) / np.sqrt(2)
    Lh = (g["parallel"] + g["perpendicular"]) / np.sqrt(2)
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9))
    a = ax[0]
    a.scatter(100 * Th, 100 * Lh, s=5, color=LEFT, alpha=0.5)
    lim = 100 * max(np.abs(Th).max(), np.abs(Lh).max())
    a.plot([-lim, lim], [-lim, lim], color=RIGHT, lw=1.5, label="goal: L/T = belt speed")
    a.annotate("", xy=(lim * 0.6, lim * 0.6), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK))
    a.annotate("", xy=(-lim * 0.35, lim * 0.35), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK))
    a.text(lim * 0.62, lim * 0.5, "goal-\nequivalent", fontsize=6.5)
    a.text(-lim * 0.9, lim * 0.42, "goal-relevant\n(speed error)", fontsize=6.5)
    a.set_aspect("equal")
    a.set_xlabel("stride time, % from mean")
    a.set_ylabel("stride length, % from mean")
    a.set_title(f"GEM: SD along {100 * g['parallel_sd']:.2f}%,\nacross {100 * g['perpendicular_sd']:.2f}%")
    a.legend(fontsize=6.5, loc="lower right")
    letter(a, "A")
    a = ax[1]
    s = s[np.isfinite(s["to"]) & np.isfinite(s["contra_hs"])]
    lat = np.r_[t.lateral, 0]
    mid = (s["hs"] + 0.5 * (s["to"] - s["hs"])).to_numpy()
    ank = ga.at(t.ankle["R"], mid)
    z = (ga.at(t.com, mid) - ank) @ lat
    v = ga.at(t.com_vel, mid) @ lat
    y = (ga.at(t.heel["L"], s["contra_hs"].to_numpy()) - ank) @ lat
    fit = ga.foot_placement_model(z, v, y)
    pred = fit["intercept"] + fit["gain_position"] * z + fit["gain_velocity"] * v
    a.scatter(1000 * pred, 1000 * y, s=5, color=LEFT, alpha=0.5)
    lo_, hi_ = 1000 * np.nanmin(y), 1000 * np.nanmax(y)
    a.plot([lo_, hi_], [lo_, hi_], color=MUTED, ls=":", lw=1)
    a.set_xlabel("predicted from CoM state at mid-stance (mm)")
    a.set_ylabel("where the left foot landed (mm)")
    a.set_title(f"Foot placement: R² = {fit['r2']:.2f}\n")
    letter(a, "B")
    fig.tight_layout()
    save(fig, 23)


def fig_lds(d):
    """Fig 24: state space and divergence curve."""
    t, extras = d["t"], d["extras"]
    curves = extras["lds_curves"]
    sig = t.kin_walker("Trunk_Linear_Velocity")[:, 1]
    hs = ga.steady_steps(t, "R")["hs"].to_numpy()[:ga.LDS_N_STRIDES + 1]
    X, _ = ga.lds_state_space(sig[:, None], hs, ga.LDS_DE, "none")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7))
    a = ax[0]
    a.plot(X[:1500, 0], X[:1500, 1], color=LEFT, lw=0.4, alpha=0.8)
    a.set_xlabel("v_AP(i)")
    a.set_ylabel("v_AP(i + τ)")
    a.set_title("State space: 2 of 5 dims")
    letter(a, "A")
    a = ax[1]
    c = curves.get(ga.LDS_PRIMARY)
    tt = np.arange(1, len(c) + 1) / ga.LDS_SAMPLES_PER_STRIDE
    a.plot(tt, c, color=INK, lw=1)
    for tag, col in (("S", RIGHT), ("L", THIRD)):
        w0, w1 = ga.LDS_FIT[tag]
        m = (tt >= max(w0, 0.01)) & (tt <= w1)
        p = np.polyfit(tt[m], c[m], 1)
        a.plot(tt[m], np.polyval(p, tt[m]), color=col, lw=2.5, alpha=0.8,
               label=f"λ_{tag} = {p[0]:.3f} per stride")
    a.set_xlabel("time after the pair were nearest (strides)")
    a.set_ylabel("mean ln(distance)")
    a.set_title("Divergence of nearby trajectories")
    a.legend(fontsize=6.5, loc="lower right")
    letter(a, "B")
    fig.tight_layout()
    save(fig, 24)


def fig_table(d):
    """Fig 25: the Healthy ranges sheet, as it looks."""
    import pandas as pd
    wide = pd.read_csv(d["work"] / "analysis" / "all_trials_metrics.csv")
    cols = ["stance_pct", "double_support_pct", "step_width_m", "stride_s_cv", "mfc_m", "f1_bw",
            "mos_ml_contact", "harmonic_ratio_hr_AP", "dfa_alpha_stride_s_R", "lds_lambda_S_trunkVel_AP"]
    cols = [c for c in cols if c in wide]
    fig, ax = plt.subplots(figsize=(W, 0.32 * len(cols) + 0.7))
    for k, c in enumerate(cols):
        lo, hi = ga.reference_for(c)[:2]
        v = wide.loc[0, c]
        pos = (v - lo) / (hi - lo)            # 0 = bottom of the range, 1 = top
        y = len(cols) - 1 - k
        ax.barh(y, 1, left=0, height=0.5, color="#D9EAD3", edgecolor=THIRD, lw=0.8)
        inside = 0 <= pos <= 1
        ax.plot(np.clip(pos, -0.95, 1.95), y, "o", color=THIRD if inside else RIGHT, ms=7,
                mec="white", mew=1)
        ax.text(2.05, y, f"{v:.4g}  (range {lo:g} to {hi:g})", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols[::-1], fontsize=7)
    ax.set_xlim(-1, 3.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["low end", "high end"])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Where one trial falls against the healthy ranges (green band)", loc="left")
    fig.tight_layout()
    save(fig, 25)


if __name__ == "__main__":
    d = analysed_trial()
    fig_pipeline()
    for f in (fig_boot, fig_surface, fig_contact_edges, fig_trust, fig_zeni,
              fig_timing, fig_belt, fig_axes_check, fig_quiet, fig_load,
              fig_fusion, fig_spatial, fig_posture, fig_kinetics, fig_work,
              fig_mos, fig_mfc):
        f(d)
    fig_bootstrap()
    for f in (fig_dfa, fig_entropy, fig_trunk, fig_control, fig_lds, fig_table):
        f(d)
