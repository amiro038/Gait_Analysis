# -*- coding: utf-8 -*-
"""
Candidate fallback variables for gait_events.py.

Each entry turns one limb's kinematics into two signals, one per event, and
says how the event shows up in it:

    ("level", "down", x)   x falls through a level
    ("level", "up",   x)   x rises through a level
    ("peak",  "max",  x)   a local maximum of x
    ("peak",  "min",  x)   a local minimum of x

The level is never typed in. It is the median of x at the trusted GRF events
of the same limb in the same trial -- what 02/03 did for heel and toe Y
velocity, generalised to any signal. Whatever timing offset is left after
that is measured on the same trusted events and removed, so a peak that
fires 30 ms early still lands on time.

gait_events.py scores every entry here against trusted GRF events that were
held out of its calibration, and writes the table. FALLBACK_ORDER in
gait_events.py decides which ones are actually used, and in what order. To
try a new variable, write a function and add it to DETECTORS.

`k` carries, per limb "L" / "R", arrays over the kinematic frames, low-pass
filtered, NaN where tracking dropped out:

    k.heel[limb], k.toe[limb]   (frames, 3)  landmark positions, Theia metres
    k.pelvis                    (frames, 3)
    k.sole_height[limb]         (frames,)    lowest boot-sole vertex above the
                                             belt surface, metres
    k.rear_height[limb]         (frames,)    ... of the rear 40% of the sole
    k.fore_height[limb]         (frames,)    ... of the front 60%

and three helpers: k.ap(x) is the forward (walking direction) component of a
(frames, 3) array, k.vertical(x) the upward one, k.velocity(x) the time
derivative in units per second.
"""

import numpy as np


def heel_toe_ap_velocity(k, limb):
    """What 02/03 do now: forward velocity of heel (HS) and toe (TO), lab frame."""
    return {"heel_strike": ("level", "down", k.ap(k.velocity(k.heel[limb]))),
            "toe_off": ("level", "up", k.ap(k.velocity(k.toe[limb])))}


def heel_toe_ap_velocity_rel_pelvis(k, limb):
    """Same, relative to the pelvis, so drifting on the belt does not shift it."""
    v_pelvis = k.ap(k.velocity(k.pelvis))
    return {"heel_strike": ("level", "down",
                            k.ap(k.velocity(k.heel[limb])) - v_pelvis),
            "toe_off": ("level", "up",
                        k.ap(k.velocity(k.toe[limb])) - v_pelvis)}


def zeni_position(k, limb):
    """Zeni et al. 2008: heel most anterior / toe most posterior to the pelvis."""
    return {"heel_strike": ("peak", "max", k.ap(k.heel[limb] - k.pelvis)),
            "toe_off": ("peak", "min", k.ap(k.toe[limb] - k.pelvis))}


def heel_toe_vertical_velocity(k, limb):
    """Heel stops descending (HS), toe starts rising (TO)."""
    return {"heel_strike": ("level", "up",
                            k.vertical(k.velocity(k.heel[limb]))),
            "toe_off": ("level", "up", k.vertical(k.velocity(k.toe[limb])))}


def heel_toe_height(k, limb):
    """Heel landmark height (HS), toe landmark height (TO)."""
    return {"heel_strike": ("level", "down", k.vertical(k.heel[limb])),
            "toe_off": ("level", "up", k.vertical(k.toe[limb]))}


def mesh_sole_height(k, limb):
    """Lowest point of the whole boot sole, so no foot-strike pattern is assumed."""
    return {"heel_strike": ("level", "down", k.sole_height[limb]),
            "toe_off": ("level", "up", k.sole_height[limb])}


def mesh_rear_fore_height(k, limb):
    """Rear of the sole down (HS), front of the sole up (TO)."""
    return {"heel_strike": ("level", "down", k.rear_height[limb]),
            "toe_off": ("level", "up", k.fore_height[limb])}


DETECTORS = {
    "heel_toe_ap_velocity": heel_toe_ap_velocity,
    "heel_toe_ap_velocity_rel_pelvis": heel_toe_ap_velocity_rel_pelvis,
    "zeni_position": zeni_position,
    "heel_toe_vertical_velocity": heel_toe_vertical_velocity,
    "heel_toe_height": heel_toe_height,
    "mesh_sole_height": mesh_sole_height,
    "mesh_rear_fore_height": mesh_rear_fore_height,
}


def find_events(spec, level, lo, hi):
    """Sub-frame times of the event inside the open window (lo, hi), in frames.

    `level` is ignored for peaks. NaN stretches produce nothing rather than a
    spurious crossing.
    """
    kind, direction, x = spec
    i0 = max(int(np.floor(lo)), 1 if kind == "peak" else 0)
    i1 = min(int(np.ceil(hi)), len(x) - (2 if kind == "peak" else 1))
    if i1 <= i0:
        return np.empty(0)
    if kind == "level":
        a = x[i0:i1] - level
        b = x[i0 + 1:i1 + 1] - level
        if direction == "down":
            hit = (a >= 0) & (b < 0)
        else:
            hit = (a <= 0) & (b > 0)
        idx = np.flatnonzero(hit)
        t = i0 + idx + a[idx] / (a[idx] - b[idx])
    else:
        y = x if direction == "max" else -x
        mid = y[i0:i1 + 1]
        left, right = y[i0 - 1:i1], y[i0 + 1:i1 + 2]
        idx = np.flatnonzero((mid >= left) & (mid > right))
        curve = left[idx] - 2 * mid[idx] + right[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(curve < 0,
                             0.5 * (left[idx] - right[idx]) / curve, 0.0)
        t = i0 + idx + np.clip(shift, -0.5, 0.5)
    t = t[np.isfinite(t)]
    return t[(t > lo) & (t < hi)]
