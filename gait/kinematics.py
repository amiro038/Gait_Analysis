"""
Posture and joint motion per stride -- the part of load carriage the
spatiotemporal and stability numbers do not show: a load leans the trunk
forward and changes how far the hip, knee and ankle move.

Per stride (heel strike to heel strike of the row's limb), from the angle
signals in the export, first component (sagittal in Visual3D's X-Y-Z
Cardan sequence):

    trunk_lean_deg     mean Thorax_Seg_Angle X (thorax against the lab)
    trunk_lean_rom_deg its range within the stride
    pelvis_tilt_deg    mean Pelvis_Seg_Angle X
    hip/knee/ankle_rom_deg   range of <Side>_<Joint>_Joint_Angle X

The signs are Visual3D's; they are constant across trials, so comparisons
between conditions are valid whichever way forward lean happens to count.
Time-normalised waveforms (101 points) go into the waveform file.
"""

import numpy as np

from .trial import SIDE

WAVE_POINTS = 101
ANGLES = {                     # output name: (signal, per-limb?)
    "trunk_lean": ("Thorax_Seg_Angle", False),
    "pelvis_tilt": ("Pelvis_Seg_Angle", False),
    "hip": ("{side}_Hip_Joint_Angle", True),
    "knee": ("{side}_Knee_Joint_Angle", True),
    "ankle": ("{side}_Ankle_Joint_Angle", True),
}


def stride_kinematics(t):
    s = t.steps
    t.angle_waves = {name: [] for name in ANGLES}
    t.angle_waves["limb"], t.angle_waves["hs"] = [], []
    out = {f"{name}_{k}": np.full(len(s), np.nan) for name in ANGLES
           for k in ("mean_deg", "rom_deg")}
    pct = np.linspace(0, 1, WAVE_POINTS)
    for i, st in enumerate(s.itertuples()):
        if not (np.isfinite(st.next_hs) and st.next_hs > st.hs):
            continue
        a, b = int(np.ceil(st.hs)), int(np.floor(st.next_hs))
        waves = {}
        for name, (signal, per_limb) in ANGLES.items():
            x = t.angle(signal.format(side=SIDE[st.limb]))
            if x is None or b >= len(x) or not np.isfinite(x[a:b + 1]).all():
                waves[name] = np.full(WAVE_POINTS, np.nan)
                continue
            seg = x[a:b + 1]
            out[f"{name}_mean_deg"][i] = seg.mean()
            out[f"{name}_rom_deg"][i] = np.ptp(seg)
            waves[name] = np.interp(pct, np.linspace(0, 1, len(seg)), seg)
        for name in ANGLES:
            t.angle_waves[name].append(waves[name])
        t.angle_waves["limb"].append(st.limb)
        t.angle_waves["hs"].append(st.hs)
    for k, v in out.items():
        s[k] = v
    return s
