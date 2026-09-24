"""
Margin of stability, minimum foot clearance and the trip risk integral, all
on the posed MOVE4D boots and the system (body + load) CoM.

MARGIN OF STABILITY (Hof, Gazendam & Sinke 2005)

    xCoM = CoM + v / w0,  w0 = sqrt(g / l),  MoS = boundary - xCoM

    v         the fused CoM velocity in the BELT frame: the lab-frame CoM
              barely moves on a treadmill while the feet sweep backwards, so
              the belt speed is added along the walking direction
    l         system CoM to stance ankle, per frame
    boundary  the stance boot's sole: its outermost point across the walking
              direction (ML), its most anterior point along it (AP)

  Read at heel strike (contact) and as the minimum over single support
  (the other foot's toe-off to its heel strike), when this boot alone is
  the base of support. ML is positive when the xCoM is inside the lateral
  border. At heel strike the new foot is out ahead, so AP at contact is
  positive; the single-support minimum goes negative as the xCoM passes
  the toe.

MINIMUM FOOT CLEARANCE (Schulz 2017)

  Clearance is the lowest sole point's height above the pitched belt
  surface (detect_gait_events fits it: one slope, one offset per boot). MFC
  is a LOCAL minimum in mid-swing that (1) beats its neighbours, (2) falls
  in the fastest quarter of the foot's swing and (3) is at the front of the
  boot -- the rear of the sole is higher. A swing without one is a
  non-MTC cycle: NaN, counted, not replaced by the global minimum.

TRIP RISK (Schulz 2017)

  MoI(t) = max(xCoM_AP - most anterior point of EITHER boot, 0)   mm
  TRI    = integral of MoI / MFC-point clearance from the MFC point's peak
           acceleration to its peak deceleration, in seconds
  The pendulum for the xCoM during a swing is the STANCE leg's -- the other
  foot's ankle.
"""

import numpy as np

from .trial import FS, G, OTHER

STANCE_WINDOW = (0.0, 1.0)     # fraction of stance searched for the minimum
MFC_LOCAL_WINDOW = 2           # frames either side a minimum must beat
MFC_SPEED_QUANTILE = 0.75
MIN_CLEARANCE_MM = 1.0         # floor for MoI / clearance
SWING_TRIM = 0.02              # fraction of swing trimmed at each end


def stance_rows(a, b, window=(0.0, 1.0)):
    a, b = a + window[0] * (b - a), a + window[1] * (b - a)
    return np.arange(int(np.ceil(a)), int(np.floor(b)) + 1)


def margin_of_stability(t):
    s = t.steps
    cols = ("mos_ml_contact", "mos_ml_min", "mos_ap_contact", "mos_ap_min",
            "mos_ml_min_at_pct", "mos_ml_contact_ankle", "pendulum_m")
    for c in cols:
        s[c] = np.nan
    s["handrail"] = False
    fwd, lat = t.forward, t.lateral
    ext = t.boots.ext
    for i, st in s.iterrows():
        if not (np.isfinite(st.to) and st.to > st.hs):
            continue
        b, o = st.limb, OTHER[st.limb]
        r = stance_rows(st.hs, st.to, STANCE_WINDOW)
        r = r[r < t.n]
        if len(r) < 3:
            continue
        # which way is "out" for this foot: away from the other ankle
        side = np.sign((t.ankle[b][r[0], :2] - t.ankle[o][r[0], :2]) @ lat)
        if not np.isfinite(side) or side == 0:
            continue
        edge = ext[b]["lat_max"][r] if side > 0 else -ext[b]["lat_min"][r]
        length = np.linalg.norm(t.com[r] - t.ankle[b][r], axis=1)
        w0 = np.sqrt(G / length)
        xcom = t.com[r, :2] + t.com_vel_belt[r, :2] / w0[:, None]
        ml = edge - side * (xcom @ lat)
        ap = ext[b]["fwd_max"][r] - xcom @ fwd
        if not np.isfinite(ml).any():
            continue
        # minima over single support: before it the other foot is still
        # down behind, after it the other foot is down in front, and in
        # both the stance boot alone is not the base of support
        single = (r >= st.contra_to) & (r <= st.contra_hs)
        if single.sum() < 3:
            single = np.ones(len(r), bool)
        s.at[i, "mos_ml_contact"] = ml[0]
        s.at[i, "mos_ml_min"] = np.nanmin(ml[single])
        k = np.flatnonzero(single)[np.nanargmin(ml[single])]
        s.at[i, "mos_ml_min_at_pct"] = 100 * k / (len(r) - 1)
        s.at[i, "mos_ap_contact"] = ap[0]
        s.at[i, "mos_ap_min"] = np.nanmin(ap[single])
        s.at[i, "mos_ml_contact_ankle"] = (side * (t.ankle[b][r[0], :2] @ lat)
                                           - side * (xcom[0] @ lat))
        s.at[i, "pendulum_m"] = length[0]
        s.at[i, "handrail"] = bool(t.handrail_rows[r].any())
    norm = t.participant.get("leg_length_m", np.nan)
    if np.isfinite(norm):
        for c in cols[:4]:
            s[c + "_per_leg"] = s[c] / norm
    return s


def swing_clearance(t, b, o, rows):
    """MFC, MoI and TRI for one swing (integer rows, trimmed). -> dict."""
    out = dict(mfc_m=np.nan, mfc_row=np.nan, mfc_vertex=-1, mfc_on_toes=False,
               mfc_minima=0, moi_peak_mm=np.nan, moi_mean_mm=np.nan,
               tri_s=np.nan, tri_window_s=np.nan, tri_peak=np.nan)
    W = t.boots.world(b, rows)
    H = t.boots.height(b, W)
    if not np.isfinite(H).all():
        return out
    clear = H.min(1)
    low = H.argmin(1)
    centroid = W.mean(1)
    speed = np.linalg.norm(np.gradient(centroid, 1 / FS, axis=0), axis=1)
    fast = speed >= np.quantile(speed, MFC_SPEED_QUANTILE)
    along = W[..., :2] @ t.forward
    rear = along < along.mean(1, keepdims=True)
    rear_z = np.where(rear, H, np.inf).min(1)

    w = MFC_LOCAL_WINDOW
    cand = []
    for i in range(w, len(clear) - w):
        before, after = clear[i - w:i], clear[i + 1:i + w + 1]
        if (np.all(clear[i] <= before) and np.all(clear[i] <= after)
                and np.any(clear[i] < before) and np.any(clear[i] < after)
                and fast[i] and rear_z[i] >= clear[i]):
            if not cand or i != cand[-1] + 1:
                cand.append(i)
    out["mfc_minima"] = len(cand)

    # margin of instability: the stance leg carries the pendulum
    anterior = np.maximum(t.boots.ext[b]["fwd_max"][rows],
                          t.boots.ext[o]["fwd_max"][rows])
    length = np.linalg.norm(t.com[rows] - t.ankle[o][rows], axis=1)
    xcom = (t.com[rows, :2] @ t.forward
            + (t.com_vel_belt[rows, :2] @ t.forward) / np.sqrt(G / length))
    moi = 1000 * np.maximum(xcom - anterior, 0)
    out["moi_peak_mm"] = float(np.nanmax(moi))
    out["moi_mean_mm"] = float(np.nanmean(moi))
    if not cand:
        return out

    k = min(cand, key=lambda i: clear[i])
    v = int(low[k])
    out.update(mfc_m=float(clear[k]), mfc_row=int(rows[k]), mfc_vertex=v,
               mfc_on_toes=bool(t.boots.sole[b]["toe"][v]))
    # the MFC point itself, followed through the swing
    risk = moi / np.maximum(1000 * clear, MIN_CLEARANCE_MM)
    p = W[:, v]
    sp = np.linalg.norm(np.gradient(p, 1 / FS, axis=0), axis=1)
    acc = np.gradient(sp, 1 / FS)
    i0, i1 = sorted((int(np.argmax(acc)), int(np.argmin(acc))))
    if i1 > i0:
        out["tri_s"] = float(np.sum(risk[i0:i1 + 1]) / FS)
        out["tri_window_s"] = (i1 - i0) / FS
    out["tri_peak"] = float(np.nanmax(risk))
    return out


def foot_clearance(t):
    s = t.steps
    rows_out = []
    for st in s.itertuples():
        if not (np.isfinite(st.to) and np.isfinite(st.next_hs)
                and st.next_hs > st.to):
            rows_out.append({})
            continue
        pad = SWING_TRIM * (st.next_hs - st.to)
        rows = np.arange(int(np.ceil(st.to + pad)),
                         int(np.floor(st.next_hs - pad)) + 1)
        rows = rows[rows < t.n]
        if len(rows) < 10:
            rows_out.append({})
            continue
        rows_out.append(swing_clearance(t, st.limb, OTHER[st.limb], rows))
    keys = ("mfc_m", "mfc_row", "mfc_vertex", "mfc_on_toes", "mfc_minima",
            "moi_peak_mm", "moi_mean_mm", "tri_s", "tri_window_s", "tri_peak")
    for k in keys:
        s[k] = [r.get(k, np.nan) for r in rows_out]
    return s
