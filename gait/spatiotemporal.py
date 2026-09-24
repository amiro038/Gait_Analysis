"""
Temporal and spatial step metrics, one row per heel strike.

TIMES come from the sub-frame event rows, so a stance is not rounded to the
10 ms frame grid.

    stance        this foot, heel strike to toe-off
    swing         toe-off to its next heel strike
    stride        heel strike to next heel strike, same foot
    step time     the step that ENDS at this heel strike: from the other
                  foot's previous heel strike to this one
    double support  initial (hs -> other foot's toe-off) + terminal (other
                  foot's heel strike -> this toe-off)
    single support  other foot's toe-off -> its next heel strike (the other
                  foot's swing, while this one carries the body)
    percentages   of the stride (the gait cycle)

DISTANCES are between the two heels at this heel strike, along the walking
direction and across it (not along Theia's X and Y, which need not line up
with the belt):

    step length   this heel ahead of the other, along the walking direction
    step width    across it
    stride length how far the foot travelled over the BELT from one heel
                  strike to the next: belt speed x stride time + how much
                  further forward it landed than last time (Dingwell et al.
                  2010). Two step lengths do NOT add up to it on a
                  treadmill: at the other foot's heel strike this heel has
                  already lifted, which adds 30-50 mm to each step
    stride speed  stride length / stride time: the walker's speed over the
                  belt, stride by stride
"""

import numpy as np

from .trial import FS, OTHER, at


def spatiotemporal(t):
    s = t.steps
    hs, to, nxt = s["hs"], s["to"], s["next_hs"]
    cto, chs, pchs = s["contra_to"], s["contra_hs"], s["prev_contra_hs"]

    def span(a, b, ok=True):
        d = (b - a) / FS
        return d.where((b > a) & ok)

    s["stance_s"] = span(hs, to)
    s["swing_s"] = span(to, nxt)
    s["stride_s"] = span(hs, nxt)
    s["step_s"] = span(pchs, hs)
    s["cadence_spm"] = 60 / s["step_s"]
    ordered = (hs < cto) & (cto < chs) & (chs < to) & (to < nxt)
    s["initial_ds_s"] = span(hs, cto, ordered)
    s["terminal_ds_s"] = span(chs, to, ordered)
    s["single_support_s"] = span(cto, chs, ordered)
    s["double_support_s"] = s["initial_ds_s"] + s["terminal_ds_s"]
    for k in ("stance", "swing", "single_support", "double_support"):
        s[f"{k}_pct"] = 100 * s[f"{k}_s"] / s["stride_s"]

    fwd, lat = np.r_[t.forward, 0], np.r_[t.lateral, 0]
    length = np.full(len(s), np.nan)
    width = np.full(len(s), np.nan)
    for b in ("L", "R"):
        m = (s["limb"] == b).to_numpy()
        rows = s.loc[m, "hs"].to_numpy()
        d = at(t.heel[b], rows) - at(t.heel[OTHER[b]], rows)
        length[m] = d @ fwd
        width[m] = np.abs(d @ lat)
    s["step_length_m"] = length
    s["step_width_m"] = width

    moved = np.full(len(s), np.nan)
    for b in ("L", "R"):
        m = (s["limb"] == b).to_numpy()
        a, e = s.loc[m, "hs"].to_numpy(), s.loc[m, "next_hs"].to_numpy()
        ok = np.isfinite(e)
        d = np.full(m.sum(), np.nan)
        d[ok] = (at(t.heel[b], e[ok]) - at(t.heel[b], a[ok])) @ fwd
        moved[m] = d
    s["stride_length_m"] = t.belt_speed * s["stride_s"] + moved
    s["stride_speed_ms"] = s["stride_length_m"] / s["stride_s"]
    s["walk_ratio"] = s["step_length_m"] / s["cadence_spm"]

    kin = s["hs_source"].isin(["kinematic", "interpolated"]) | \
        s["to_source"].isin(["kinematic", "interpolated"])
    s["events_from_kinematics"] = kin
    s["events_interpolated"] = (s["hs_source"] == "interpolated") | \
        (s["to_source"] == "interpolated")
    return s
