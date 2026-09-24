# Gait metrics: theory, mathematics and parameter justification

Companion to `run_gait_analysis.py` and the `gait/` package (one module per
metric family, named in each section). Every metric is defined here: what it
measures, the equations, what each symbol means, what a change in the value
implies, why each parameter was set the way it was, and typical healthy-adult
values. Section 0 covers what is shared: the treadmill, the metronome, the
load, and the confidence intervals.

**Citation confidence.** References marked ✔ were verified against the source
during writing. References marked ~ are from domain knowledge and the exact
volume/page should be checked against your reference manager before publication.
Reported literature values are approximate and protocol-dependent — treat them as
sanity-check ranges, not norms.

---

## 0. Two protocol facts that change the maths

### 0.1 The treadmill requires a belt-speed correction for anything using CoM velocity

On a treadmill the CoM has near-zero *mean* anterior–posterior velocity in the lab
frame, while the feet cycle backwards. Feeding lab-frame velocity into the
extrapolated centre of mass produces a meaningless AP margin. Every AP quantity
below uses

$$v_{AP}^{\text{corrected}}(t) = v_{AP}^{\text{lab}}(t) + v_{\text{belt}}$$

where `v_belt` is **measured** per trial from the stance heels during foot-flat
(25–55% of stance): a foot on the belt moves with it, so its velocity is the
belt's (`gait/trial.py`, `belt_motion`). The same velocities give the belt's
**direction of travel**, which is the walking direction every AP and ML
quantity is projected on. The boots' long axes are not used for this: boots toe
out, and 1.3° of that on D05's meshes leaked ±17 mm of step length into step
width. Mediolateral quantities need no belt correction — the belt does not move
sideways. This correction is the single most common omission in treadmill
margin-of-stability work.

### 0.2 The metronome changes what long-range correlations mean

The protocol prescribes **both** speed (1.3 m/s) and cadence (108 bpm). External
rhythmic cueing drives stride-interval fluctuations from persistent toward
**anti-persistent**, because the walker corrects each deviation against the beat.
Ageing and disease also move α — but *toward* 0.5, from the opposite side. The two
effects are not on the same axis, so α on stride time in this protocol measures
**compliance with the metronome**, not intrinsic control.

Consequence for interpretation: the primary long-range-correlation outcomes should
be series the metronome does not directly constrain — **step width**, **MoS_ML**,
**minimum foot clearance**. Stride time and stride length are reported but read as
cueing-compliance measures.

A second constraint compounds this: at fixed belt speed, stride time and stride
length are mechanically forced to trade off (speed is pinned), which *imposes*
anti-persistence on both regardless of neural control [Dingwell et al. 2010 ✔].

### 0.3 The load, weighed in the opening quiet standing

The load differs between trials, and Theia's `Whole_body_COG` knows nothing
about it: the markerless model sees a body, not the mass of a pack or a vest.
The force plates see everything. Every trial opens with a few seconds of quiet
standing, and there (`gait/trial.py`, `quiet_standing`, `system_com`):

- **System weight** $W$ = total vertical GRF over the steadiest 2 s window
  before the first step with both belts loaded and both heels still (CV of the
  total below 2%). System mass $m = W/g$.
- **Load mass** $m_{load} = m - m_{body}$, with $m_{body}$ from
  `participants.csv` (the participant weighed without the carried load).
- **Load position.** Standing still, the CoP is directly below the system CoM,
  so horizontally
  $$x_{load} = \frac{m\,x_{CoP} - m_{body}\,x_{CoM,body}}{m_{load}}.$$
  Its height cannot be seen standing still and is set at `Trunk_Position` +
  `LOAD_HEIGHT_M` (0 by default). Left–right the load is centred on the trunk
  (`LOAD_CENTRED`): the CoP locates it only to the plate registration's error
  times $m/m_{load}$ (≈5: 4 mm of registration is 20 mm of load), and packs and
  vests sit on the midline.
- The load point is then carried in a trunk frame (Low_Back → Neck, walking
  direction), and the **system CoM** is
  $(m_{body}\,x_{CoM,body} + m_{load}\,x_{load})/m$.

On the synthetic trial (80 kg + 20 kg, 150 mm behind the trunk) this recovers
19.9 kg, 152 mm, and a system CoM within 2 mm of the truth; Theia's body CoM is
82 mm from it. The system CoM feeds every margin of stability, pendulum length
and CoM work; the system mass feeds the GRF-integrated velocity. Forces are
reported per body weight (`_bw`) and per total weight (`_tw`).

The walking mean vertical GRF is compared with the standing weight on every
trial; more than 2% apart points at plate drift or a standing that was not
still. If no quiet standing is found, the mass falls back to the walking mean
and the load cannot be placed (printed, and noted in the summary).

### 0.4 Confidence intervals, and what "block length" means

Every value comes from one walk of a few hundred strides; walk again and it
changes. Each value in `{trial}_summary.csv` carries a 95% interval saying by
how much (`gait/uncertainty.py`):

- **Circular block bootstrap** for everything computed from a stride series
  (mean, SD, CV, regression R²). Strides are not independent — a long stride
  tends to be followed by another — so resampling them one at a time would
  destroy that correlation and make the intervals too narrow. Instead strides
  are drawn in runs of consecutive strides (**blocks**) and the runs are glued
  into a new series. The **block length** is how many strides go in a run. It
  is chosen per series by the Politis & White (2004) rule, corrected by Patton,
  Politis & White (2009): the slower the autocorrelation dies away, the longer
  the block; an uncorrelated series gets 1. It is written in the `block`
  column. On a persistent AR(1) series (φ = 0.6) blocks give 93% coverage of a
  95% interval; resampling single strides gives 66%.
- **Parametric bootstrap for DFA α.** Blocks would cut exactly the long-range
  correlation α measures, so instead 200 series are simulated with the
  estimated α (exact fractional Gaussian noise, Davies–Harte) and DFA is run on
  each; the basic bootstrap interval and DFA's bias at this N come out of that.
  At N = 486 the interval covers the truth 93% of the time and is ±0.17 wide.
- **Stride bootstrap for λ** (§7): the divergence curve is kept per reference
  stride, and blocks of strides are resampled.

Sample entropy has no interval: blocks break its templates at every join.

---

## 1. Long-range correlations — Detrended Fluctuation Analysis

### What it measures

Whether a stride is statistically related to strides that occurred *hundreds of
strides earlier*. Not how much gait varies, but whether the variation has memory.
DFA is completely blind to magnitude: shuffle a stride series and its SD is
unchanged while α collapses to 0.5.

### Mathematics

Given a stride-indexed series $x_1 \dots x_N$ (index = stride number, not time):

**1. Integrate** to a cumulative deviation profile:

$$Y(k) = \sum_{i=1}^{k}\left(x_i - \bar{x}\right), \qquad k = 1 \dots N$$

**2. Segment** $Y$ into non-overlapping boxes of length $n$, forwards and backwards
(giving $2\lfloor N/n \rfloor$ boxes so no data is discarded).

**3. Detrend** each box with a least-squares polynomial fit $Y_n(k)$ of order 1
(DFA-1, linear).

**4. Fluctuation** as the RMS residual across all boxes:

$$F(n) = \sqrt{\frac{1}{N}\sum_{k=1}^{N}\left[Y(k) - Y_n(k)\right]^2}$$

**5. Scaling exponent** α is the slope of $\log F(n)$ against $\log n$:

$$F(n) \propto n^{\alpha}$$

### Variables

| Symbol | Meaning | Units |
|---|---|---|
| $x_i$ | value of the *i*-th stride (stride time, step width, …) | s, m |
| $N$ | number of strides in the series | strides |
| $Y(k)$ | integrated profile, cumulative deviation from the mean | s·strides, m·strides |
| $n$ | box size — the timescale being probed | strides |
| $F(n)$ | RMS fluctuation remaining after detrending at scale $n$ | same as $x$ |
| $\alpha$ | scaling exponent, the slope in log–log space | dimensionless |

### What the value means

| α | Interpretation |
|---|---|
| **< 0.5** | Anti-persistent. A long stride is actively corrected by a short one. Tight, error-correcting control — what metronome cueing and fixed-speed constraints produce. |
| **≈ 0.5** | Uncorrelated white noise. Each stride independent of every other. |
| **0.5 – 1.0** | Persistent. A deviation tends to be followed by deviations in the same direction, at every timescale. Healthy stride-interval territory. |
| **≈ 1.0** | 1/f "pink" noise. Scale-free, the classic signature of healthy physiological complexity. |
| **> 1.0** | Non-stationary, random-walk-like. Often a sign of drift in the series rather than a property of control — check for trend before interpreting. |

### Parameter justification

| Parameter | Value | Why |
|---|---|---|
| Detrending order | 1 (linear) | Standard for stride series; higher orders remove the low-frequency structure you are trying to measure. |
| Box size range | $16 \le n \le N/9$ | Directly from the empirical examination in [Damouras et al. 2010 ✔], which showed the commonly used 4 to N/4 range inflates α. Small boxes are dominated by the detrending fit itself; boxes beyond N/9 contain too few samples to average. |
| Series length | ≥ 600 strides recommended | [Damouras et al. 2010 ✔]. Every trial in a run is cut to the SAME N (`SERIES_LENGTH = "shortest"`), since α drifts with N. Below 9 × 16 × 2 strides the box range spans less than an octave and α is not computed. |
| Interval | parametric bootstrap, §0.4 | ±0.17 at N = 486 (validated coverage 93%). |
| Box spacing | logarithmic | Equal weighting per decade in the log–log fit. |
| Series indexing | stride number | Not elapsed time. The series is a sequence of discrete strides. |

### Validation built into the pipeline

1. **Shuffled surrogate** — randomly permute the series. α must return 0.5. If it
   doesn't, the implementation is wrong.
   20 shuffles per series; their mean is in the summary's note.
2. **Synthetic fractional Gaussian noise** of known α (exact, Davies–Harte), in
   `gait_metrics_validation.py`, quantifies the estimator's bias at this N; the
   same simulation gives each trial's interval and bias.
3. **Confirmatory estimator** — Geweke–Porter-Hudak on the periodogram
   (α ≈ d + 0.5 for fGn), reported in the note. Its own scatter is about twice
   DFA's, so only a large disagreement is informative.

### Healthy values

| Series | Population | α |
|---|---|---|
| Stride interval, overground | Healthy young adults | ≈ 0.75 – 0.90 ~ |
| Stride interval | Older adults, neurodegenerative disease | shifts toward 0.5 ~ |
| Stride interval, treadmill | Healthy adults | lower than overground ~ |
| Stride interval, metronome-paced | Healthy adults | often < 0.5 ~ |

Consensus guidance on reporting and on functional thresholds of α:
[Ravi et al. 2020 ✔].

### References

- ✔ Damouras S, Chang MD, Sejdić E, Chau T (2010). An empirical examination of detrended fluctuation analysis for gait data. *Gait Posture* 31(3):336–340.
- ✔ Ravi DK, Marmelat V, Taylor WR, Newell KM, Stergiou N, Singh NB (2020). Assessing the temporal organization of walking variability: a systematic review and consensus guidelines on detrended fluctuation analysis. *Front Physiol* 11:562. doi:10.3389/fphys.2020.00562
- ✔ Dingwell JB, John J, Cusumano JP (2010). Do humans optimally exploit redundancy to control step variability in walking? *PLoS Comput Biol* 6(7):e1000856.
- ~ Peng CK, Buldyrev SV, Havlin S, Simons M, Stanley HE, Goldberger AL (1994). Mosaic organization of DNA nucleotides. *Phys Rev E* 49:1685–1689. — the original DFA method.
- ~ Hausdorff JM, Purdon PL, Peng CK, Ladin Z, Wei JY, Goldberger AL (1996). Fractal dynamics of human gait: stability of long-range correlations in stride interval fluctuations. *J Appl Physiol* 80(5):1448–1457.

---

## 2. Harmonic ratio and improved harmonic ratio

### What it measures

Smoothness and rhythmic regularity of trunk acceleration *within* a stride. This is
a within-stride waveform-shape measure and says nothing about stride-to-stride
variation, which makes it independent of everything in §1 and §3.

### Why odd and even harmonics

One stride contains two steps. If both steps produce identical trunk acceleration,
the signal repeats twice per stride, so energy concentrates on the **even**
harmonics of stride frequency. Anything breaking that — inter-limb asymmetry,
jerkiness, an irregular event — leaks energy into the **odd** harmonics.

Mediolateral is the exception: you sway left over one step and right over the
other, so ML completes **one** cycle per stride and is odd-dominant. The ratio
flips.

### Mathematics

For each stride, take the discrete Fourier transform of the trunk acceleration over
exactly one stride. With $A_k$ the amplitude of the *k*-th harmonic of stride
frequency:

**Anterior–posterior and vertical** (even-dominant):

$$HR_{AP,V} = \frac{\sum_{k=2,4,6,\dots}^{20} A_k}{\sum_{k=1,3,5,\dots}^{19} A_k}$$

**Mediolateral** (odd-dominant):

$$HR_{ML} = \frac{\sum_{k=1,3,5,\dots}^{19} A_k}{\sum_{k=2,4,6,\dots}^{20} A_k}$$

**Improved harmonic ratio** [Pasciuto et al. 2017 ✔] — the power of the *intrinsic*
harmonics (those carrying the symmetric component) relative to the total power,
giving a bounded index:

$$iHR = \frac{\sum_{k \in \text{intrinsic}} A_k^2}{\sum_{k=1}^{20} A_k^2} \times 100\%$$

### Variables

| Symbol | Meaning | Units |
|---|---|---|
| $A_k$ | amplitude of the *k*-th harmonic of stride frequency | m/s² |
| $k$ | harmonic number; $k=1$ is stride frequency, $k=2$ step frequency | – |
| $HR$ | harmonic ratio | dimensionless, unbounded ≥ 0 |
| $iHR$ | improved harmonic ratio | % , bounded 0–100 |

### What the value means

- **Higher HR** = smoother, more rhythmic trunk motion. Falls, ageing and
  neurological disease reduce it.
- **HR is unbounded**, which makes group means unstable when a denominator is small
  — a known reliability problem and the motivation for iHR.
- **HR is not a symmetry measure**, despite being widely described as one. A
  perfectly symmetric but jerky gait scores low [Pasciuto et al. 2017 ✔]. For
  symmetry specifically use the symmetry angle (§8.4) or the autocorrelation
  measure (§8.3).

### Parameter justification

| Parameter | Value | Why |
|---|---|---|
| Signal | `Trunk_Linear_Velocity`, weighted $k^1$ | **Not** `Low_Back_Joint_Acc` or `Trunk_Joint_Acceleration`: those are ANGULAR (deg/s²). The first linear trunk column is used; from a velocity, acceleration harmonic $k$ is exactly $k\,\omega_0$ times velocity harmonic $k$, so weighting by $k$ is exact where numerical differentiation rolls off before the 20th harmonic. $\omega_0$ cancels in the ratio. |
| Harmonics | first 20 | The established convention; covers the spectral content that carries stride structure. |
| Frame | the belt's direction of travel | One fixed heading on a treadmill; a per-frame rotation would mix pelvis rotation into the signal. |
| Segmentation | per stride, ipsilateral heel strike to heel strike | Stride segmentation method measurably affects HR, so it is fixed and stated. |
| Aggregation | median, and mean with a block bootstrap CI | HR is a ratio with occasional large outliers; iHR is bounded and is the better mean. |

### Healthy values

Approximate, and strongly dependent on walking speed and sensor site:

| Axis | Healthy young adults | Older adults / fallers |
|---|---|---|
| AP | ≈ 3 – 4 ~ | lower ~ |
| Vertical | ≈ 3 – 4 ~ | lower ~ |
| ML | ≈ 2 – 3 ~ | lower ~ |

iHR is bounded 0–100% and healthy young adults sit high in that range ~.

### References

- ✔ Pasciuto I, Bergamini E, Iosa M, Vannozzi G, Cappozzo A (2017). Overcoming the limitations of the Harmonic Ratio for the reliable assessment of gait symmetry. *J Biomech* 53:84–89.
- ~ Menz HB, Lord SR, Fitzpatrick RC (2003). Acceleration patterns of the head and pelvis when walking on level and irregular surfaces. *Gait Posture* 18(1):35–46.
- ~ Bellanca JL, Lowry KA, VanSwearingen JM, Brach JS, Redfern MS (2013). Harmonic ratios: a quantification of step to step symmetry. *J Biomech* 46(4):828–831.

---

## 3. Entropy

### What it measures

Predictability. Sample entropy asks: given a short pattern of $m$ consecutive
points somewhere in the series, how often does a similar pattern recur within
tolerance $r$? Frequent recurrence → low entropy → regular and repeatable.

### The interpretation trap

High entropy is casually called "more complex", but **pure random noise has maximal
sample entropy and zero complexity**. This is why single-scale entropy on a
continuous signal is hard to interpret — you may just be measuring your filter
cutoff. Multiscale entropy exists to fix exactly this.

### Mathematics

**Sample entropy** [Richman & Moorman 2000 ~]. For a series $u_1 \dots u_N$, let
$B^m$ be the number of pairs of length-$m$ template vectors within Chebyshev
distance $r$, and $A^m$ the number of those pairs that remain within $r$ when
extended to length $m+1$:

$$\text{SampEn}(m, r, N) = -\ln\frac{A^m}{B^m}$$

Self-matches are excluded, which is what distinguishes SampEn from approximate
entropy and removes ApEn's bias toward regularity.

**Multiscale entropy** [Costa et al. 2002 ~]. Coarse-grain the series at scale
$\tau$ by non-overlapping averaging:

$$y_j^{(\tau)} = \frac{1}{\tau}\sum_{i=(j-1)\tau+1}^{j\tau} u_i$$

then compute SampEn on each $y^{(\tau)}$. The **MSE curve** — or the area under it —
is the complexity measure, *not* the scale-1 value. White noise starts high and
falls off steeply with coarse-graining; structured signals retain entropy across
scales.

**Refined composite MSE** [Wu et al. 2014 ~] averages the match counts across all
$\tau$ possible coarse-graining phases before taking the logarithm, which greatly
reduces variance at long scales.

### Variables

| Symbol | Meaning | Units |
|---|---|---|
| $m$ | template length — how many consecutive points define a "pattern" | samples |
| $r$ | similarity tolerance, as a fraction of the series SD | same as $u$, or ×SD |
| $N$ | series length | samples |
| $\tau$ | coarse-graining scale factor | – |
| $A^m, B^m$ | counts of matching template pairs at length $m+1$ and $m$ | – |

### What the value means

- **Low SampEn** = regular, repeatable, machine-like stepping. Seen in ageing and
  Parkinson's disease.
- **High SampEn** = unpredictable. But see the trap above — check the MSE curve
  before calling it complexity.
- Counter-intuitive pairing worth expecting: impaired populations often show
  **higher SD and lower entropy simultaneously** — magnitude up, unpredictability
  down. Variability and regularity are not the same axis.

### Parameter justification

| Parameter | Value | Why |
|---|---|---|
| Algorithm | SampEn, not ApEn | ApEn counts self-matches, biasing toward regularity, and is more N-dependent. |
| $m$ | 2 | Convention for gait series; $m=3$ reported as a sensitivity check. |
| $r$ | 0.2 × SD, **and swept** | [Yentes & Raffalt 2021 ✔] recommends against assuming a single $r$ transfers between conditions. The sweep is reported, not just the point estimate. |
| $N$ | ≥ 200 | [Yentes et al. 2013 ✔] showed both algorithms become extremely parameter-sensitive at N ≤ 200. This dataset has 798 strides, comfortably above. |
| SD for $r$ | stated explicitly | Whether $r$ is scaled by each condition's own SD or by a pooled SD changes the result; the choice is recorded in the output. |
| MSE signal | trunk acceleration, 300 s after the warm-up | A stride series runs out of points after 2–3 scales; the continuous signal does not. |
| MSE scales | 1 – 30 (0.3 s at 100 Hz) | At 100 Hz, scale 10 is only 0.1 s — well inside one step. 30 reaches a quarter stride with 1000 points still left per phase. |
| Reporting | $m$, $r$, $N$ always | Required by [Yentes & Raffalt 2021 ✔]; entropy values are meaningless without them. |

### Healthy values

**There are no transferable normative values for SampEn.** The number is defined
only relative to its $m$, $r$, $N$ and preprocessing, and those vary across the
literature. Comparisons are valid only within a dataset analysed identically —
exactly the stance `lds_validation.py` takes for λ. Report the parameters and
compare conditions, never absolute values against other papers.

### References

- ✔ Yentes JM, Hunt N, Schmid KK, Kaipust JP, McGrath D, Stergiou N (2013). The appropriate use of approximate entropy and sample entropy with short data sets. *Ann Biomed Eng* 41(2):349–365.
- ✔ Yentes JM, Raffalt PC (2021). Entropy analysis in gait research: methodological considerations and recommendations. *Ann Biomed Eng* 49(3):979–990. doi:10.1007/s10439-020-02616-8
- ~ Richman JS, Moorman JR (2000). Physiological time-series analysis using approximate entropy and sample entropy. *Am J Physiol Heart Circ Physiol* 278(6):H2039–H2049.
- ~ Costa M, Goldberger AL, Peng CK (2002). Multiscale entropy analysis of complex physiologic time series. *Phys Rev Lett* 89:068102.
- ~ Wu SD, Wu CW, Lin SG, Lee KY, Peng CK (2014). Analysis of complex time series using refined composite multiscale entropy. *Phys Lett A* 378(20):1369–1374.

---

## 4. Margin of stability

### What it measures

Pure mechanics, and the only metric here with units of metres and an unambiguous
physical meaning. The CoM is not just *somewhere*, it is *going* somewhere. If all
active control stopped now, inverted-pendulum dynamics would carry it to the
extrapolated CoM. MoS is the distance from there to the edge of the base of
support.

### Mathematics

[Hof, Gazendam & Sinke 2005 ✔]:

$$x_{CoM} = x + \frac{v}{\omega_0}, \qquad \omega_0 = \sqrt{\frac{g}{\ell}}$$

$$MoS = u_{max} - x_{CoM}$$

### Variables

| Symbol | Meaning | Units |
|---|---|---|
| $x$ | CoM position in the direction of interest | m |
| $v$ | CoM velocity in that direction (**belt-corrected in AP**, §0.1) | m/s |
| $\omega_0$ | inverted-pendulum eigenfrequency | rad/s |
| $g$ | gravitational acceleration, 9.81 | m/s² |
| $\ell$ | effective pendulum length | m |
| $u_{max}$ | boundary of the base of support in that direction | m |
| $x_{CoM}$ | extrapolated centre of mass | m |
| $MoS$ | margin of stability | m |

### What the value means

- **MoS > 0** — the XCoM lies inside the base of support. You could stop here
  without falling.
- **MoS < 0** — the XCoM is outside. Another step is mechanically required.
- **AP MoS is routinely negative during walking, and that is normal.** Walking is
  controlled falling forward; a negative AP margin is what propulsion looks like.
  Reading it as instability is a common misinterpretation.
- **Larger is not simply better.** Less stable populations frequently adopt *larger*
  margins as compensation, which inverts the expected correlation. Interpret
  alongside step width and the foot-placement model (§8.2).

### Base of support from the mesh

This is where the foot meshes change the measurement rather than refine it. The
conventional $u_{max}$ is the ankle joint centre or a toe marker — both *internal*
points, several centimetres inside the true foot border. Using the mesh:

- **ML boundary** — the lateral-most vertex of the stance foot.
- **AP boundary** — the most anterior vertex.

Both come from the **sole** of the participant's MOVE4D boot (the lowest vertex
in every 10 mm cell of the boot's footprint), posed on Theia's foot and bent at
the MTP by the toe angle — the same geometry `detect_gait_events.py` uses, so
events and margins see the same boot. The ankle-based value is computed
alongside (`mos_ml_contact_ankle`), since quantifying that offset is a
methodological result in itself.

### Parameter justification

| Parameter | Value | Why |
|---|---|---|
| $\ell$ | per-frame CoM-to-ankle distance | Hof's original uses leg length; a per-frame distance is the more literal pendulum length. Constant-$\ell$ reported as a sensitivity check since the choice shifts absolute MoS. |
| CoM | the **system** CoM, body + load (§0.3) | The load moves the CoM; Theia's body CoM does not include it. |
| Velocity | markers below 0.5 Hz, integrated GRF / system mass above | Complementary filter, same cutoff on both branches. Velocity error reaches the xCoM divided by $\omega_0$; on the synthetic trial the fused velocity is 4–8 mm/s RMS from the truth against 25 for markers alone. |
| AP velocity | belt-corrected | §0.1. Without this the AP margin is meaningless on a treadmill. |
| Timing | at foot contact **and** minimum over **single support** | Before the other foot's toe-off and after its heel strike, the stance boot alone is not the base of support; a minimum over the whole stance picks up late stance, when the xCoM is far past a foot the body has already left. |
| Normalisation | also per leg length (`*_per_leg`) | For comparison between people. |
| Directions | ML and AP separately | They mean different things (see above) and must not be combined into a resultant. |

### Healthy values

| Quantity | Healthy adults |
|---|---|
| ML MoS at foot contact | ≈ 0.03 – 0.10 m ~ |
| AP MoS during single support | typically negative ~ |

Strongly dependent on walking speed, step width and how $u_{max}$ was defined —
which is precisely why the mesh-vs-marker comparison is worth reporting.

### References

- ✔ Hof AL, Gazendam MGJ, Sinke WE (2005). The condition for dynamic stability. *J Biomech* 38:1–8. doi:10.1016/j.jbiomech.2004.03.025
- ~ Hof AL (2008). The 'extrapolated center of mass' concept suggests a simple control of balance in walking. *Hum Mov Sci* 27(1):112–125.
- ~ Hak L, Houdijk H, Beek PJ, van Dieën JH (2013). Steps to take to enhance gait stability: the effect of stride frequency, stride length, and walking speed on local dynamic stability and margins of stability. *PLoS One* 8(12):e82842.
- ~ McAndrew Young PM, Dingwell JB (2012). Voluntary changes in step width and step length during human walking affect dynamic margins of stability. *Gait Posture* 36(2):219–224.

---

## 5. Margin of instability and the trip risk integral

### What it measures

Defined by [Schulz 2017 ✔], which is also the paper that motivates using digitised
shoe surfaces rather than markers — the same thing your foot meshes provide. The
logic is that trip risk depends on two things at once: how close the foot is to the
ground, **and** how badly the body would be destabilised if the foot did catch.
MFC alone captures only the first.

### Mathematics

Three modifications to the standard MoS produce the **margin of instability**:

1. Use the **continuous MoS trajectory during swing**, not the single
   double-support minimum.
2. Set $u_{max}$ to the **most anterior toe tip on either foot** — the stance toe
   during early swing, the swing toe during late swing. This is where the anterior
   boundary of the base of support *would* be if the swing foot contacted the
   ground.
3. Clamp stable values to zero and negate:

$$MoI(t) = \max\left(-MoS_{AP}(t),\ 0\right)$$

The **trip risk trajectory** is the dimensionless ratio of instability to
clearance, both in mm:

$$\text{trip risk}(t) = \frac{MoI(t)}{MFC(t)}$$

The **trip risk integral** is its integral across the swing phase:

$$TRI = \int_{t_1}^{t_2} \frac{MoI(t)}{MFC(t)}\, dt$$

### Variables

| Symbol | Meaning | Units |
|---|---|---|
| $MoI(t)$ | margin of instability, continuous through swing | mm |
| $MFC(t)$ | minimum foot clearance over the whole foot, continuous | mm |
| $t_1, t_2$ | integration bounds: peak acceleration and peak deceleration of the MFC point | s |
| $TRI$ | trip risk integral | s |

### What the value means

- **MoI = 0** means the body is in a mechanically stable configuration at that
  instant — foot contact would be recoverable.
- **Higher MoI** = greater destabilisation if the foot caught at that instant.
- **Higher TRI = greater trip risk**, the opposite direction from MFC, where higher
  means *safer*. Schulz's central finding is that MFC and TRI move in opposite
  directions with gait speed: as speed rises (which raises real trip-fall risk),
  MFC indicates *less* risk while TRI indicates *more*. That is the argument that
  TRI is the better risk measure.

### Parameter justification

| Parameter | Value | Why |
|---|---|---|
| $u_{max}$ | most anterior vertex of either foot mesh | Schulz used the most anterior toe *marker*; the mesh gives the actual anterior-most surface point, which is what would strike an obstacle. |
| Direction | AP only | MoI is defined in the direction of progression. **Requires the belt correction of §0.1** — without it MoI on a treadmill is meaningless. |
| Positive clamp | MoS > 0 → 0 | By definition: a stable configuration contributes no trip risk. |
| $MFC(t)$ | min over the swing boot's sole | Schulz used all digitised shoe points; the posed boot sole is the direct equivalent. |
| Floor | the fitted belt plane | The DICE plates are pitched ~0.84°, 13 mm over a stance, and Theia's floor height differs between standing and walking, so the belt surface is fitted to the boots in mid-stance: one slope, one offset per boot. Clearance is height above it. |
| Pendulum | CoM to the **stance** (other) ankle | During a swing the body pivots over the stance leg. |
| Integration bounds | peak acceleration to peak deceleration of the MFC point | Excludes the spikes at lift-off and landing, when the foot is intentionally near the ground. |
| Time normalisation | **none** | Schulz deliberately did not normalise, so slower swings yield larger TRI. Preserved here for comparability. |
| Integration step | 1/100 s | Data rate. Schulz used 1/120 s at 120 Hz. |

**Note on this protocol:** Schulz emphasises gait speed as a confound for TRI, and
shows the apparent group effect in his fallers was driven by their slower
self-selected speed. Your fixed belt speed removes that confound by design, which
is an advantage — but it also means you cannot reproduce his speed effect.

### Healthy values

Schulz reports TRI by group, speed and surface rather than a single normative
number, and TRI depends on swing duration because time is not normalised. Use his
figures for magnitude context; compare within your dataset.

### References

- ✔ Schulz BW (2017). A new measure of trip risk integrating minimum foot clearance and dynamic stability across the swing phase of gait. *J Biomech* 55:107–112. doi:10.1016/j.jbiomech.2017.02.024
- ✔ Hof AL, Gazendam MGJ, Sinke WE (2005). *J Biomech* 38:1–8.
- ✔ (cited within Schulz) Loverro KL, Mueske NM, Hamel KA (2013). Location of minimum foot clearance on the shoe and with respect to the obstacle changes with locomotor task. *J Biomech* 46:1842–1850.
- ✔ (cited within Schulz) Thies SB, Jones RK, Kenney LPJ, Howard D, Baker R (2011). Effects of ramp negotiation, paving type and shoe sole geometry on toe clearance in young adults. *J Biomech* 44:2679–2684.

---

## 6. Minimum foot clearance

### What it measures

The smallest gap between foot and ground during swing — the literal mechanical
margin for tripping.

### Mathematics

Per swing frame, over all vertices $\mathbf{p}_j$ of the swing foot mesh, with a
planar floor at $z = 0$:

$$MFC(t) = \min_j\ p_{j,z}(t)$$

MFC for the stride is the **local** minimum of this trajectory during mid-swing,
subject to the event criteria below.

### How it is found (`gait/stability.py`)

1. **The boot sole, not joint centres.** The heel and toe joint centres are
   internal points centimetres above the sole; the posed boot gives the true
   lowest surface point, above the fitted belt plane.
2. **Local minimum, not global.** [Schulz 2017 ✔] operationalises a valid MFC event
   as: (a) a local minimum — lower than the preceding and following two frames;
   (b) foot speed within the upper quartile for that swing, which eliminates
   spurious events immediately after foot-off; (c) the rear of the sole is not
   lower, which prevents false detections at the heel. A swing with no such
   minimum is a non-MTC cycle and gives NaN; the count is in the summary.
3. **Which vertex was lowest**, and whether it is on the toe cap, are recorded
   (`mfc_vertex`, `mfc_on_toes`). Schulz's own work shows the MFC point moves
   around the shoe with speed and surface.

Where more than one candidate satisfies the criteria, Schulz uses the smaller
value; the count of local minima per swing is also retained, since he raises
whether multiple minima themselves indicate risk.

### Limitation, stated up front

The boot bends at one hinge (the MTP, by Theia's toe angle); the rest of the sole
is rigid with the foot. A rigid toe cap went 30–40 mm through the belt at every
push-off; with the hinge, the deepest stance penetration on the synthetic trial
is 3–4 mm. What remains below the belt surface is pose error, and it is counted
(`mfc_below_belt_pct`).

### Healthy values

| Quantity | Healthy adults |
|---|---|
| MFC | ≈ 1 – 2 cm ~ |
| MFC SD | ≈ 0.5 cm ~ |

For trip risk the **variability and the low tail matter more than the mean** — a
single abnormally low clearance is what catches an obstacle. Report the
distribution, not just central tendency.

### References

- ✔ Winter DA (1992). Foot trajectory in human gait: a precise and multifactorial motor control task. *Phys Ther* 72:45–46.
- ✔ Schulz BW (2017). *J Biomech* 55:107–112.
- ✔ (cited within Schulz) Schulz BW (2011). Minimum toe clearance adaptations to floor surface irregularity and gait speed. *J Biomech* 44:1277–1284.
- ✔ (cited within Schulz) Byju AG, Nussbaum MA, Madigan ML (2016). Alternative measures of toe trajectory more accurately predict the probability of tripping than minimum toe clearance. *J Biomech* 49(16):4016–4021.
- ✔ (cited within Schulz) Santhiranayagam BK, Sparrow WA, Lai DTH, Begg R (2016). Non-MTC gait cycles: an adaptive toe trajectory control strategy in older adults. *Gait Posture* 53:73–79.

---

## 7. Local dynamic stability

`gait/lds.py`, following Bruijn's LocalDynamicStability toolbox: 150 strides
time-normalised to 100 samples each, delay-embedded (τ = 10, dE = 5), Rosenstein
divergence over 10 strides, λ_S fitted over 0–0.5 stride and λ_L over 4–10.
Validated in `lds_validation.py`, which now runs the shipped divergence code:
Lorenz −3%, Rössler −15% of their known exponents, and a periodic signal gives
λ_L ≈ 0 with a λ_S noise floor of 0.2–0.45 per stride.

Two additions: a long trial holds several **windows** of 150 strides (up to 4),
each gets its own λ and the trial value is their mean; and the curve is kept per
reference stride so strides can be **bootstrapped** in blocks (§0.4). The
interval says how much λ depends on which strides happened to be walked, not the
estimator's own bias. How it relates to the rest:

- **Lyapunov λ** measures how fast nearby trajectories diverge, assuming nothing
  about periodicity. Continuous, local, model-free.
- **DFA α** measures memory in a stride-indexed series. Different question, and
  independent of λ.
- **MoS** is an instantaneous mechanical quantity. A person can have a comfortable
  margin and a high λ at the same time.

**Dynamic mode decomposition has been dropped** from this analysis — there is no
established gait literature giving it normative values or fall-risk validation, so
it would produce numbers with nothing to compare them against.

---

## 8. Supplementary metrics

### 8.1 Goal-equivalent manifold analysis

**What it measures.** Splits step-to-step variability into the part that threatens
the task goal and the part that does not. Two people with identical total
variability can have very different goal-relevant error. This is the principled
answer to "is this variability bad?", and a fixed-speed treadmill is the ideal
setting because the goal — maintain belt speed — is unambiguous.

**Mathematics.** With stride length $L$ and stride time $T$, the goal is constant
speed $v^* = L/T$. The goal-equivalent manifold is the set of $(L, T)$ satisfying
$L = v^* T$. $T$ and $L$ are first divided by their means, so both axes are
dimensionless and the manifold is the 45° line — rotating seconds and metres
together would make the split depend on the units. Each stride's deviation is
then decomposed into a component **along** the manifold (goal-equivalent,
$\delta_\parallel$) and **across** it (goal-relevant, $\delta_\perp$). Each is
reported as an SD (% of the mean stride, with a block bootstrap CI), a lag-1
autocorrelation, and a DFA α with its interval.

**Stride length on a treadmill** is how far the foot travelled over the belt:
belt speed × stride time + how much further forward it landed than last time
[Dingwell et al. 2010 ✔]. Two heel-to-heel step lengths do **not** add up to
it: at the other foot's heel strike this heel has already lifted, which adds
30–50 mm to each step (100 mm per stride on the synthetic trial).

**What the value means.** Strong correction of $\delta_\perp$ with weak correction
of $\delta_\parallel$ is the signature of a controller exploiting redundancy
efficiently [Dingwell et al. 2010 ✔].

- ✔ Dingwell JB, John J, Cusumano JP (2010). *PLoS Comput Biol* 6(7):e1000856.

### 8.2 Mediolateral foot placement control

**What it measures.** How much of where the foot lands is explained by body state —
i.e. whether foot placement is actively used for balance at all. Complements MoS by
asking about the *controller* rather than the margin.

**Mathematics.** Regress the next lateral foot placement on CoM position and
velocity at midstance:

$$z_{\text{foot}} = \beta_0 + \beta_1 z_{CoM} + \beta_2 \dot{z}_{CoM} + \varepsilon$$

The **$R^2$ is the measure**, per limb, with a block bootstrap CI. Coefficients
$\beta_1, \beta_2$ describe the gain. Everything is across the walking direction
and relative to the stance ankle, so a slow drift across the belt cannot appear
on both sides of the regression; the CoM is the system CoM.

**Healthy values.** [Wang & Srinivasan 2014 ✔] report that a linear function of hip
position and velocity at midstance explains **over 80%** of next lateral foot
position variance. AP is substantially lower.

- ✔ Wang Y, Srinivasan M (2014). Stepping in the direction of the fall: the next foot placement can be predicted from current upper body state in steady-state walking. *Biol Lett* 10(9):20140405.

### 8.3 Autocorrelation-based regularity and symmetry

**What it measures.** Step regularity, stride regularity and their ratio, from the
unbiased autocorrelation of trunk acceleration. Cheaper and more robust than HR,
and a useful cross-check — if the two disagree, something in the stride
segmentation is worth inspecting.

**Mathematics.** With unbiased autocorrelation $A_d$ at lag $d$:

- $A_{d1}$ = the autocorrelation **peak** near one step → step regularity
- $A_{d2}$ = the peak near one stride → stride regularity
- symmetry = $A_{d1}/A_{d2}$

The peaks are searched within ±25% of the median step and stride. Reading the
value at a fixed lag misses the peak whenever the stride drifts: 6% off, a
perfectly regular signal reads 0.94 instead of 1.

**What the value means.** Coefficients near 1 mean each step/stride closely repeats
the last. A symmetry ratio near 1 means the two steps of a stride are equivalent.

- ✔ Moe-Nilssen R, Helbostad JL (2004). Estimation of gait cycle characteristics by trunk accelerometry. *J Biomech* 37:121–126.

### 8.4 Symmetry angle

**What it measures.** Inter-limb asymmetry, without the reference-limb artefact of
the classic symmetry index — which divides by one limb's value and therefore
depends on which limb you picked, and diverges as that denominator approaches zero.

**Mathematics.** For a variable with left value $X_L$ and right value $X_R$:

$$SA = \frac{45° - \arctan\!\left(X_L / X_R\right)}{90°} \times 100\%$$

with a correction when $SA > 100\%$. Bounded and reference-free.

**What the value means.** 0% = perfect symmetry. Larger magnitude = greater
asymmetry, with the sign indicating direction.

- ~ Zifchock RA, Davis I, Higginson J, Royer T (2008). The symmetry angle: a novel, robust method of quantifying asymmetry. *Gait Posture* 27(4):622–627.

### 8.5 Walk ratio

**What it measures.** Step length per unit cadence. Remarkably constant within a
person across speeds, which makes deviation from it a sensitive marker of altered
gait control. Nearly free to compute.

$$WR = \frac{\text{step length (m)}}{\text{cadence (steps/min)}}$$

**Healthy values.** ≈ 0.006 m per step/min ~, fairly stable across speeds in
healthy adults; reduced in neurological impairment ~.

**Caveat for this protocol:** with cadence fixed by the metronome and speed fixed by
the belt, the walk ratio is largely determined by the protocol rather than the
walker. Report it, but do not interpret it as a free control variable here.

### 8.6 Kinetics and CoM work

`gait/kinetics.py`, per stance, **only where both its heel strike and toe-off
were trusted GRF events of the same belt contact** — one boot on that belt, the
other off it, the CoP under the boot (see the gait events section of the
README). The event record names the belt, so a clean crossover step is read from
the belt it landed on. Anything else carries two feet's force and is left NaN.

- **Impulses**: braking and propulsive (the AP force along the walking
  direction, split by sign) and vertical; BW·s.
- **Peaks**: F1, trough, F2, peak braking and propulsive force; BW.
- **Loading rate**: least-squares slope of the vertical force between 20% and
  80% of F1; BW/s.
- **CoP range** along and across the walking direction (15 Hz), **free moment**
  peak |T_z|.
- **CoM work, individual limbs method** [Donelan, Kram & Kuo 2002 ~]: each
  leg's GRF · system CoM velocity (belt frame, so the stance foot is still as
  in overground walking), integrated over collision (heel strike → other
  toe-off, negative), rebound and preload (single support, positive and
  negative) and push-off (other heel strike → toe-off, positive). J and J/kg of
  system mass. Load carriage raises push-off and collision work, which is the
  metabolic cost of step-to-step transitions.

Forces are the GRF on the body in Theia's axes: the plate's rotation from its
fitted corners, then the lab → Theia registration, with the sign that makes the
vertical force hold the walker up. The horizontal axes are checked against the
marker CoM acceleration (0.5–5 Hz correlation, in the summary); a strongly
negative one is flipped and reported. Normalised per body weight (`_bw`) and per
total weight (`_tw`): load raises the first; the second shows whether the
pattern changed beyond carrying more.

### 8.7 Posture and joint motion

`gait/kinematics.py`: per stride, mean and range of the thorax segment angle
(trunk lean, the main postural response to a load), the pelvis segment angle,
and the range of motion of the hip, knee and ankle (first, sagittal component).
Time-normalised waveforms of these and of the GRF go to `{trial}_waveforms.npz`.

---

## 9. Shared construction: one stride series for every metric

DFA, entropy, the GEM analysis and the foot-placement model all consume the **same**
stride-indexed series, and all share the same "how many strides, which window"
decisions that the LDS section already handles carefully. They are built once, from
one function, so the sample definition is guaranteed identical across metrics.

Without this, α computed on 798 strides would be compared against λ computed on
150, and part of any difference would be sample size rather than physiology.

| Decision | Value | Why |
|---|---|---|
| Warm-up excluded | 60 s after the first event (`WARMUP_S`) | The belt ramp and acclimatisation; applies to every summary value. |
| Limb | each limb's own strides, both reported | Series for DFA, entropy, GEM and foot placement are per limb; the trunk measures and LDS use the right limb's strides (`LIMB_FOR_TRUNK`). |
| Gap handling | a series under 95% complete is not used | DFA and entropy both accept a gappy array and return a number. |
| Series length | the shortest trial's, for every trial (`SERIES_LENGTH`) | λ, α and entropy are all N-dependent; the N used is in every row. |

---

## 10. Validation

`gait_metrics_validation.py` mirrors what `lds_validation.py` does for λ: it
establishes each estimator's bias on signals whose answer is known, before any of
them are used to compare conditions.

| Estimator | Validated against | Expected |
|---|---|---|
| DFA α | synthetic fGn of known α; shuffled surrogates | recovers α; shuffle → 0.5 |
| SampEn | sine (regular), white noise (maximal), logistic map | ordering and direction correct |
| MSE | white vs 1/f noise | white falls with scale, 1/f stays flat |
| Harmonic ratio | synthetic 2-per-stride waveform with imposed asymmetry | HR falls monotonically with asymmetry |
| MoS / MoI / TRI | analytic inverted pendulum with known $x_{CoM}$; a constructed swing | exact |
| CoM fusion | known trajectory, noisy markers, offset force | beats markers alone, unbiased below 0.5 Hz |
| DFA interval | fGn at N = 486 | covers the truth ~95% |
| Block bootstrap | persistent AR(1) | ~95% coverage, where single-stride resampling gives ~66% |
| Regularity, GEM, foot placement, symmetry | constructed signals | exact |

End to end, `test_gait_analysis.py` runs detection and the whole analysis on a
synthetic trial with a known load, CoM, belt, clearance and forces, and checks
the load, CoM, velocity, stance timing (−0.1 ± 0.4 ms), step width, MFC,
margins and the kinetic balances against the truth.

`lds_validation.py` finds λ off by a system-dependent 3–15%, with no fit window
fixing both test systems at once. That result is the reason for this step: each
estimator's bias should be known before differences between conditions are
interpreted.
