# 10. Margin of stability

**What.** How much base of support is left beyond the point to which the body's momentum would carry the centre of mass (Hof, Gazendam & Sinke 2005).

**Equations.** Treat the body as an inverted pendulum of length $l$. If control stopped now, its dynamics would carry the CoM to the **extrapolated centre of mass**

$$\mathbf{x}\text{CoM} = \mathbf{x}_{CoM} + \frac{\mathbf{v}_{belt}}{\omega_0},\qquad \omega_0 = \sqrt{\frac{g}{l}},\qquad l = \lVert\mathbf{x}_{CoM} - \mathbf{p}_{ankle,stance}\rVert.$$

The margin is the distance from the xCoM to the edge of the base of support, positive when the xCoM is inside it:

$$\text{MoS}_{ML} = e_{ML} - s\,(\mathbf{x}\text{CoM}\cdot\hat{\ell}),\qquad \text{MoS}_{AP} = e_{AP} - \mathbf{x}\text{CoM}\cdot\hat{f}.$$

Here $e_{ML}$ is the stance boot's outermost sole point across the belt, $e_{AP}$ is its most anterior sole point along the belt, and $s = \pm1$ says which way is "outward": away from the other ankle at heel strike.

**Steps** (`margin_of_stability()`):

1. For each stance (heel strike → toe-off), for every frame: compute $l$ and $\omega_0$ frame by frame, and the xCoM from the **system** CoM (§5.6) and its **fused belt-frame** velocity (§5.7).
2. Take the boundary from the stance **boot**'s extents (§5.3).
3. Report the margins **at heel strike** (`_contact`), and their **minima over single support** (`_min`), from the other foot's toe-off to its heel strike. Also report where in stance the ML minimum falls (`mos_ml_min_at_pct`).
4. Also keep the ML margin to the **ankle joint centre** (`mos_ml_contact_ankle`), the boundary most studies use, for comparison with the literature.
5. If the participant's leg length is known, also give each margin divided by it (`_per_leg`), for comparisons between people.

![Figure 17. The margin of stability at a right heel strike, with travel to the right. A: from above. The system CoM (black) and its extrapolation along the belt-frame velocity (open circle, xCoM). The ML margin (green) runs from the xCoM to the outer edge of the right boot, and the AP margin (violet) to its front. B: both margins through the stance. The ML margin stays positive. The AP margin becomes strongly negative: the xCoM moves ahead of the stance foot, and walking "falls" onto the next step. The single-support minima (shaded) are the values reported.](figures/fig17.png){width=6.5in}

::: {custom-style="Why Box"}
**Why the system CoM and the fused velocity** — the load moves the CoM (§5.6). The velocity term is divided by $\omega_0\approx3.3$ s⁻¹, so velocity noise enters the margin directly, which is why §5.7 fuses the force plates in. **Why the belt frame** — in the lab the CoM hardly moves forward on a treadmill, so without the belt speed the AP margin would be meaningless. **Why the boot edge** — the ankle joint centre is 3–5 cm inside the boot's real lateral edge. The boot is the physical base of support, and its edge also differs between footwear (boots vs shoes). The ankle-based value is kept only for comparison with the literature. **Why single-support minima** — only in single support is the stance boot alone the base of support; in double support the base spans both feet. **Why "outward" from the ankles** — it makes the sign independent of which way Theia's axes point and of which foot is stepping.
:::

::: {custom-style="Code Box"}
**In the code** — §10: `margin_of_stability()`. Results: `mos_ml_contact`, `mos_ml_min`, `mos_ap_contact`, `mos_ap_min`, `mos_ml_min_at_pct`, `mos_ml_contact_ankle`, `pendulum_m` (m), and `_per_leg` versions. Steps with a hand on the rail are flagged (`handrail`).
:::

::: {custom-style="Range Box"}
**Healthy range** — most published values use the ankle or the lateral malleolus as the boundary, so the boot-edge margin (`mos_ml_contact`) is expected to be about 3–5 cm larger than the published range; compare `mos_ml_contact_ankle` with it directly. Treadmill margins are larger than overground ones. A negative AP margin in single support is normal.
:::

{{ranges: mos_ml_contact, mos_ml_min}}

: Table 10.1. Healthy ranges for the ML margin of stability (published with an ankle or malleolus boundary).

# 11. Foot clearance and trip risk

## 11.1 Minimum foot clearance (MFC)

**What.** How close the swinging boot comes to the belt in mid-swing, when a trip is most likely: the foot is low and moving fast.

**Equations.** In every frame of the swing,

$$c(t) = \min_{\text{sole points } j} h_j(t),$$

the height of the lowest sole point above the pitched belt (§3.4). The MFC is a **local minimum** of $c(t)$ in mid-swing that meets all three criteria of Schulz (2017):

(a) it is lower than the 2 frames on either side (`MFC_LOCAL_WINDOW`);

(b) the foot is in the **fastest 25%** of its swing (`MFC_SPEED_QUANTILE`), which rules out the dip just after toe-off;

(c) the rear half of the sole is not lower than it (the minimum is at the front of the boot).

If several minima qualify, the lowest one is taken. If none qualifies, the swing is a "non-MTC" cycle: its MFC is **missing**, and it is not replaced by the global minimum, which would be at toe-off or heel strike. The share of such swings is reported (`swings_without_mfc_pct`). The script also records whether the MFC point lies on the toe cap (`mfc_on_toes`) and the share of swings whose MFC is below the belt (`mfc_below_belt_pct`), which can only be a pose error.

**Steps** (`swing_clearance()`, `foot_clearance()`):

1. Take the swing from toe-off to the next heel strike, trimming **2%** at each end (`SWING_TRIM`), where the boot is still on the belt.
2. Pose every sole point in every frame (§3), with the toe hinge, and compute each point's height above the belt.
3. Find the minima that meet (a)–(c), and keep the lowest.

## 11.2 Margin of instability and the trip-risk integral

**What.** If the foot caught on something now, how badly would the body be thrown forward, relative to how close the foot is to the ground (Schulz 2017)?

**Equations.** The **margin of instability** is how far the xCoM (along the belt, using the **stance** leg's pendulum) is ahead of the front of either boot:

$$\text{MoI}(t) = \max\left(\mathbf{x}\text{CoM}_{AP}(t) - \max(e_{AP,b}(t),\ e_{AP,o}(t)),\ 0\right)\quad[\text{mm}]$$

The **trip-risk integral** adds up the ratio of MoI to clearance over the risky part of the swing:

$$\text{TRI} = \int_{t_0}^{t_1}\frac{\text{MoI}(t)}{\max\left(c(t),\ 1\ \text{mm}\right)}\,dt\quad[\text{s}]$$

Here $t_0$ and $t_1$ are the moments of peak acceleration and peak deceleration of the MFC point's speed. The 1 mm floor (`MIN_CLEARANCE_MM`) keeps the ratio finite.

![Figure 18. One swing. A: the clearance of the lowest sole point (blue). The MFC (orange triangle) is the local minimum inside the fastest 25% of the swing (shaded). B: the margin of instability, the distance by which the xCoM is ahead of the front of either boot. C: the ratio MoI/clearance. The trip-risk integral is the shaded area between the MFC point's peak acceleration and peak deceleration.](figures/fig18.png){width=6.5in}

::: {custom-style="Why Box"}
**Why the sole and not a toe marker** — the lowest point of a boot moves around the sole during the swing. The "toe" landmark is neither the lowest point nor the front of the boot, so its height can be centimetres off the real clearance. **Why the pitched belt** — the 0.8° pitch changes the height by up to 13 mm over a stride, the same size as the MFC itself. **Why Schulz's criteria** — the global minimum of a swing is at toe-off or heel strike, when the foot is meant to be on the ground. The criteria pick the mid-swing minimum that matters for tripping, and they say when a swing has none rather than inventing one. **Why TRI** — a low foot is only dangerous when a trip would destabilise the body. TRI weights clearance by the body's forward momentum, and a load increases that momentum.
:::

::: {custom-style="Code Box"}
**In the code** — §11: `swing_clearance()`, `foot_clearance()`. Settings: `MFC_LOCAL_WINDOW`, `MFC_SPEED_QUANTILE`, `SWING_TRIM`, `MIN_CLEARANCE_MM`. Results: `mfc_m`, `moi_peak_mm`, `tri_s` (plus `mfc_on_toes`, `mfc_minima`, `moi_mean_mm`, `tri_window_s`, `tri_peak` per step).
:::

{{ranges: mfc_m, mfc_m_sd}}

: Table 11.1. Healthy ranges for minimum foot clearance. MoI and TRI are recent measures without established healthy ranges: compare them between conditions.
