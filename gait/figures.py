"""
Quality-check figures, one set per trial, saved next to the results.

    {trial}_qa.png            is the trial sound? the quiet standing and the
                              load, where the events came from, the boot's
                              clearance over the belt, the margins over time,
                              the GRF, and the CoM velocity fusion
    {trial}_variability.png   the stride-series side: DFA with its interval,
                              alpha for every series, the LDS divergence
                              curve, and the multiscale entropy curve
    {trial}_mos_stride.mp4    (optional) birds-eye view of one typical stride:
                              the boots, the base of support, the CoM and
                              xCoM, and the margins as they are measured
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from . import lds, variability as var
from .trial import FS, FS_FORCE, LIMBS, OTHER, SIDE

COLOUR = {"L": "#2a5d9f", "R": "#a8442a"}
SOURCE_COLOUR = {"GRF": "#1b7a3d", "kinematic": "#d98b04",
                 "GRF_unverified": "#7a4fb5", "interpolated": "#c0392b"}


def qa_figure(t, path):
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    s = t.steps

    # 1. quiet standing and the weight
    n = int(min(30 * FS_FORCE, t.n_force))
    tt = np.arange(n) / FS_FORCE
    total = t.grf["L"][:n, 2] + t.grf["R"][:n, 2]
    a = ax[0, 0]
    for b in LIMBS:
        a.plot(tt, t.grf[b][:n, 2], color=COLOUR[b], lw=0.6,
               label=f"{SIDE[b]} belt")
    a.plot(tt, total, color="k", lw=0.8, label="total")
    L = t.load
    if L["found"]:
        a.axvspan(L["start_s"], L["stop_s"], color="#f2d24b", alpha=0.4)
        a.axhline(L["weight_n"], color="k", ls=":", lw=1)
    a.set_title(f"Quiet standing: {t.mass:.1f} kg total, load "
                f"{L['load_kg']:+.1f} kg", fontsize=10)
    a.set_xlabel("s")
    a.set_ylabel("vertical GRF (N)")
    a.legend(fontsize=7, frameon=False)

    # 2. stance time over the trial, coloured by where the events came from
    a = ax[0, 1]
    src = np.where(s["hs_source"] == "GRF", s["to_source"], s["hs_source"])
    for name, c in SOURCE_COLOUR.items():
        m = src == name
        a.plot(s["hs"][m] / FS, s["stance_s"][m], ".", ms=3, color=c,
               label=f"{name} ({m.sum()})")
    a.axvline(s.loc[s["steady"], "hs"].min() / FS, color="k", lw=0.8,
              ls="--")
    a.set_title("Stance time, by event source (dashed: end of warm-up)",
                fontsize=10)
    a.set_xlabel("s")
    a.set_ylabel("stance (s)")
    a.legend(fontsize=7, frameon=False, markerscale=3)

    # 3. clearance over a few strides, with MFC
    a = ax[1, 0]
    mid = s[s["steady"] & (s["limb"] == "R")]
    if len(mid) > 6:
        r0 = int(mid["hs"].iloc[len(mid) // 2])
        r1 = int(mid["hs"].iloc[len(mid) // 2 + 4])
        rows = np.arange(r0, r1)
        for b in LIMBS:
            a.plot(rows / FS, 1000 * t.boots.low_z[b][rows], color=COLOUR[b],
                   lw=1, label=f"{SIDE[b]} lowest sole point")
        m = s[(s["mfc_row"] >= r0) & (s["mfc_row"] < r1)]
        a.plot(m["mfc_row"] / FS, 1000 * m["mfc_m"], "kv", ms=6, label="MFC")
    a.axhline(0, color="k", lw=0.8)
    a.set_ylim(-10, 120)
    a.set_title(f"Height above the belt (MFC median "
                f"{1000 * s['mfc_m'].median():.1f} mm, "
                f"{(s['mfc_m'] < 0).sum()} below the belt)", fontsize=10)
    a.set_xlabel("s")
    a.set_ylabel("mm")
    a.legend(fontsize=7, frameon=False)

    # 4. margins over the trial
    a = ax[1, 1]
    for b in LIMBS:
        m = s["limb"] == b
        a.plot(s["hs"][m] / FS, 1000 * s["mos_ml_contact"][m], ".", ms=2,
               color=COLOUR[b], label=f"ML at contact, {SIDE[b]}")
        a.plot(s["hs"][m] / FS, 1000 * s["mos_ap_min"][m], "x", ms=2,
               color=COLOUR[b], alpha=0.5, label=f"AP stance minimum, {SIDE[b]}")
    a.axhline(0, color="k", lw=0.8)
    a.set_title("Margins of stability (system CoM, boot soles)", fontsize=10)
    a.set_xlabel("s")
    a.set_ylabel("mm")
    a.legend(fontsize=7, frameon=False, markerscale=3, ncol=2)

    # 5. vertical GRF ensemble, trusted stances
    a = ax[2, 0]
    if t.waves["vgrf"]:
        W = np.array(t.waves["vgrf"])
        limb = np.array(t.waves["limb"])
        pct = np.linspace(0, 100, W.shape[1])
        for b in LIMBS:
            if (limb == b).any():
                w = W[limb == b]
                a.fill_between(pct, *np.percentile(w, [5, 95], axis=0),
                               color=COLOUR[b], alpha=0.2, lw=0)
                a.plot(pct, w.mean(0), color=COLOUR[b],
                       label=f"{SIDE[b]} (n={len(w)})")
    a.axhline(1, color="k", ls=":", lw=0.8)
    a.set_title("Vertical GRF, trusted stances (per total weight)",
                fontsize=10)
    a.set_xlabel("% stance")
    a.legend(fontsize=7, frameon=False)

    # 6. the velocity fusion, ML, a few seconds of steady walking
    a = ax[2, 1]
    r0 = int(s.loc[s["steady"], "hs"].min()) if s["steady"].any() else 0
    rows = np.arange(r0, min(r0 + 4 * FS, t.n))
    k = np.r_[t.lateral, 0]
    a.plot(rows / FS, t.com_vel_markers[rows] @ k, color="#bbbbbb", lw=1,
           label="markers, differentiated")
    a.plot(rows / FS, t.com_vel[rows] @ k, color="k", lw=1.2,
           label="fused with integrated GRF")
    r = t.fusion_check
    a.set_title("CoM velocity across the walking direction (GRF vs marker "
                f"acceleration r = {r['X']:.2f}/{r['Y']:.2f}/{r['Z']:.2f})",
                fontsize=10)
    a.set_xlabel("s")
    a.set_ylabel("m/s")
    a.legend(fontsize=7, frameon=False)

    for a in ax.flat:
        a.grid(alpha=0.15)
    fig.suptitle(t.stem, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def variability_figure(t, summary, extras, series_length, path, limb="R"):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # 1. DFA of the first uncued series, with its interval
    s = t.steps[t.steps["steady"] & (t.steps["limb"] == limb)].sort_values("hs")
    s = s.iloc[:series_length] if series_length else s
    a = ax[0, 0]
    for name in ("step_width_m", "mos_ml_contact", "stride_s"):
        x = s[name].to_numpy(float) if name in s else np.array([])
        if np.isfinite(x).mean() > 0.95:
            d = var.dfa(x[np.isfinite(x)])
            r = summary[(summary["metric"] == f"dfa_alpha_{name}")
                        & (summary["limb"] == limb)]
            if np.isfinite(d["alpha"]) and len(r):
                r = r.iloc[0]
                a.loglog(d["boxes"], d["fluct"], "o", color="#2a5d9f")
                lx = np.log10(d["boxes"])
                p = np.polyfit(lx, np.log10(d["fluct"]), 1)
                a.loglog(d["boxes"], 10 ** np.polyval(p, lx), color="#cc6633",
                         label=f"alpha {r['value']:.3f} "
                               f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
                a.set_title(f"DFA, {name}, {d['n']} strides ({limb})",
                            fontsize=10)
                a.legend(fontsize=8, frameon=False)
                break
    a.set_xlabel("box (strides)")
    a.set_ylabel("F(n)")

    # 2. alpha for every series
    a = ax[0, 1]
    d = summary[summary["metric"].str.startswith("dfa_alpha_")
                & (summary["limb"] == limb)].sort_values("value")
    if len(d):
        y = np.arange(len(d))
        err = np.vstack([d["value"] - d["ci_low"], d["ci_high"] - d["value"]])
        a.barh(y, d["value"], xerr=np.abs(err), color=[
            "#bbbbbb" if "metronome" in n else "#2a5d9f" for n in d["note"]],
            error_kw=dict(lw=0.8))
        a.set_yticks(y)
        a.set_yticklabels([m.replace("dfa_alpha_", "") for m in d["metric"]],
                          fontsize=7)
        a.axvline(0.5, color="#888", ls=":")
        a.axvline(1.0, color="#888", ls="--")
    a.set_title("alpha with 95% CI (grey: paced by the metronome)",
                fontsize=10)

    # 3. LDS divergence
    a = ax[1, 0]
    curves = extras.get("lds_curves") or {}
    if curves:
        tt = np.arange(1, lds.WS * lds.SAMPLES_PER_STRIDE + 1) \
            / lds.SAMPLES_PER_STRIDE
        for name, c in curves.items():
            a.plot(tt, c, lw=1.5 if name == lds.PRIMARY else 0.8,
                   label=name)
        a.set_xlabel("strides")
        a.set_ylabel("ln divergence")
        a.legend(fontsize=7, frameon=False)
    a.set_title("Local dynamic stability (mean over windows)", fontsize=10)

    # 4. multiscale entropy
    a = ax[1, 1]
    mse = extras.get("mse")
    if mse is not None:
        for c in mse.columns:
            a.plot(mse.index, mse[c], "o-" if c != "white_noise" else "s--",
                   ms=3, label=c)
        a.set_xlabel("scale (samples)")
        a.set_ylabel("sample entropy")
        a.legend(fontsize=7, frameon=False)
    a.set_title("Refined composite MSE, trunk acceleration", fontsize=10)
    for a in ax.flat:
        a.grid(alpha=0.15)
    fig.suptitle(t.stem, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# %% --------------------------------------------------------------------------
#  birds-eye video of a typical stride
# -----------------------------------------------------------------------------

def typical_stride(t, limb="R"):
    """The steady stride closest to the median on stride time, step width
    and the two margins at contact, in IQR units."""
    s = t.steps[t.steps["steady"] & (t.steps["limb"] == limb)]
    cols = ["stride_s", "step_width_m", "mos_ml_contact", "mos_ap_contact"]
    s = s.dropna(subset=cols + ["next_hs", "to"])
    score = sum(np.abs(s[c] - s[c].median())
                / max(s[c].quantile(0.75) - s[c].quantile(0.25), 1e-9)
                for c in cols)
    return s.loc[score.idxmin()]


def mos_video(t, path, limb="R", fps=20, dpi=100):
    """Belt-frame birds-eye view: positions shifted forward by belt speed x
    time so the stance boot stays planted. Margins are differences at one
    instant, so the shift does not change them."""
    st = typical_stride(t, limb)
    rows = np.arange(int(np.ceil(st.hs)), int(np.floor(st.next_hs)) + 1)
    shift = t.belt_speed * (rows - rows[0]) / FS
    f2, l2 = t.forward, t.lateral

    def plane(xy, i):          # -> (along, across) in the belt frame
        return np.column_stack([xy @ f2 + shift[i], xy @ l2])

    hulls = {b: [] for b in LIMBS}
    for b in LIMBS:
        W = t.boots.world(b, rows)
        for i in range(len(rows)):
            p = plane(W[i, :, :2], i)
            hulls[b].append(p[ConvexHull(p).vertices])
    loaded = {b: np.zeros(len(rows), bool) for b in LIMBS}
    for x in t.steps.itertuples():
        if np.isfinite(x.to):
            loaded[x.limb] |= (rows >= x.hs) & (rows <= x.to)
    o = OTHER[limb]
    side = np.sign((t.ankle[limb][rows[0], :2] - t.ankle[o][rows[0], :2]) @ l2)
    length = np.linalg.norm(t.com[rows] - t.ankle[limb][rows], axis=1)
    w0 = np.sqrt(9.81 / length)
    com = np.array([plane(t.com[r:r + 1, :2], i)[0]
                    for i, r in enumerate(rows)])
    vel = np.column_stack([t.com_vel_belt[rows, :2] @ f2,
                           t.com_vel_belt[rows, :2] @ l2])
    xcom = com + vel / w0[:, None]
    ml = np.array([side * h[:, 1].max() if side > 0 else -h[:, 1].min()
                   for h in hulls[limb]]) - side * xcom[:, 1]
    ap = np.array([h[:, 0].max() for h in hulls[limb]]) - xcom[:, 0]
    stance = rows <= st.to

    allp = np.vstack(hulls["L"] + hulls["R"] + [com])
    lo, hi = allp.min(0) - 0.25, allp.max(0) + 0.25
    fig = plt.figure(figsize=(12, 7))
    top = fig.add_axes((0.06, 0.38, 0.9, 0.55))
    bot = fig.add_axes((0.06, 0.08, 0.9, 0.22))
    tt = (rows - rows[0]) / FS

    def draw(i):
        top.clear()
        bos = [hulls[b][i] for b in LIMBS if loaded[b][i]]
        if bos:
            p = np.vstack(bos)
            top.add_patch(Polygon(p[ConvexHull(p).vertices], closed=True,
                                  fc="#f2e5b8", ec="#8a6d1f", ls="--",
                                  alpha=0.6))
        for b in LIMBS:
            top.add_patch(Polygon(hulls[b][i], closed=True, lw=2,
                                  ec=COLOUR[b], alpha=0.8,
                                  fc=COLOUR[b] if loaded[b][i] else "none"))
        top.annotate("", xy=xcom[i], xytext=com[i],
                     arrowprops=dict(arrowstyle="-|>", color="k"))
        top.plot(*com[i], "ko", ms=9)
        top.plot(*xcom[i], "o", ms=9, mfc="none", mec="k", mew=2)
        top.set_xlim(lo[0], hi[0])
        top.set_ylim(lo[1], hi[1])
        top.set_aspect("equal")
        top.set_title(f"{SIDE[limb]} stride, {'stance' if stance[i] else 'swing'}"
                      f"  MoS ML {1000 * ml[i]:+.0f} mm  AP "
                      f"{1000 * ap[i]:+.0f} mm", fontsize=11)
        top.set_xlabel("walking direction, belt frame (m)")
        top.set_ylabel("across (m)")
        bot.clear()
        bot.plot(tt, 1000 * ml, color="#1b7a3d", label="MoS ML")
        bot.plot(tt, 1000 * ap, color="#7a4fb5", label="MoS AP")
        bot.axvline(tt[i], color="#888")
        bot.axhline(0, color="k", lw=0.8)
        bot.set_xlabel("s")
        bot.set_ylabel("mm")
        bot.legend(fontsize=8, frameon=False)

    from matplotlib.animation import FFMpegWriter, PillowWriter
    try:
        writer, ext = FFMpegWriter(fps=fps), ".mp4"
        if not matplotlib.animation.writers.is_available("ffmpeg"):
            raise RuntimeError
    except Exception:
        writer, ext = PillowWriter(fps=fps), ".gif"
    out = str(path) + ext
    with writer.saving(fig, out, dpi):
        for i in range(len(rows)):
            draw(i)
            writer.grab_frame()
    plt.close(fig)
    return out
