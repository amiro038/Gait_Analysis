# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:26:25 2026

@author: amiro038
"""

###############################################################################
# This code is going to be for some basic spatiotemporal metrics
# Stride time - done
# stance time - done
# swing time - done
# stance percentage - done
# swing percentage - done
# single support time -done
# single support percentage -done
# double support time - done
# double support percentage - done
# stride length -done
# step length - done
# step width - done
# minimum foot clearance - done
# gait velocity - done
# cadence - done
# local dynamic stability (short and long term Lyapunov exponents) - done
# margin of stability
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =============================================================================
# %% Setting up folders one for now but set up for for loop later
# =============================================================================

gait_event_file = r"C:\Users\alexm\Box\Military Project\Data\Analysis\DICE_Treadmill\gait_event_outputs\D05_C1_Treadmill_1.3mpers 108bpm_merged_events.csv"
gait_event_path = Path(gait_event_file)

events = pd.read_csv(gait_event_file)

base = gait_event_path.parents[1]
metrics_path = base / "Theia_csv_outputs" / f"{(gait_event_path.stem).replace('_merged_events', '')}_metrics.csv"

kinematic_data = pd.read_csv(metrics_path, sep=None, engine='python', header=1, skiprows=[2,3,4])

# =============================================================================
# %% reshaping the long format event list into one row per contact
# =============================================================================

events = events.sort_values('frame_100hz').reset_index(drop=True)

#flag the toe off frames then backfill within limb so each heel strike grabs
#the next toe off of that same limb
events['_toe_off_frame']  = events['frame_100hz'].where(events['event_type'] == 'toe_off')
events['_toe_off_source'] = events['source'].where(events['event_type'] == 'toe_off')
events['toe_off_kinematic_frame_100hz'] = events.groupby('support_limb')['_toe_off_frame'].bfill()
events['toe_off_source']                = events.groupby('support_limb')['_toe_off_source'].bfill()

gait_event_data = (events[events['event_type'] == 'heel_strike']
                   .rename(columns={'frame_100hz': 'initial_contact_kinematic_frame_100hz',
                                    'source': 'heel_strike_source'})
                   [['support_limb',
                     'initial_contact_kinematic_frame_100hz',
                     'toe_off_kinematic_frame_100hz',
                     'heel_strike_source', 'toe_off_source']]
                   .reset_index(drop=True))

#brain check
reused = (gait_event_data['toe_off_kinematic_frame_100hz'].duplicated(keep=False) &
          gait_event_data['toe_off_kinematic_frame_100hz'].notna())
for x in gait_event_data.index[reused]:
    print(f" Toe off reused across contacts at index {x}, likely a missed event")

for x in range(len(gait_event_data)):
    if gait_event_data['support_limb'][x] == gait_event_data['support_limb'].shift()[x]:
        print(f" Two consecutive contacts on the same limb at index {x}, likely a missed stride")

# =============================================================================
# %% organizing gait events to make future calculations easier
# =============================================================================

#Set up all the values that you'll need down the line
gait_event_data['next_ipsi_heelstrike'] = gait_event_data.groupby('support_limb')['initial_contact_kinematic_frame_100hz'].shift(-1)
gait_event_data['prev_ipsi_heelstrike'] = gait_event_data.groupby('support_limb')['initial_contact_kinematic_frame_100hz'].shift(1)
gait_event_data['next_ipsi_toeoff'] = gait_event_data.groupby('support_limb')['toe_off_kinematic_frame_100hz'].shift(-1)
gait_event_data['prev_ipsi_toeoff'] = gait_event_data.groupby('support_limb')['toe_off_kinematic_frame_100hz'].shift(1)

run = (gait_event_data['support_limb'] != gait_event_data['support_limb'].shift()).cumsum()
idx = pd.Series(gait_event_data.index, index=gait_event_data.index)
first = idx.groupby(run).first()
last  = idx.groupby(run).last()
gait_event_data['prev_contra'] = run.sub(1).map(last)
gait_event_data['next_contra'] = run.add(1).map(first)

gait_event_data['prev_contra_heelstrike'] = gait_event_data['prev_contra'].map(gait_event_data['initial_contact_kinematic_frame_100hz'])
gait_event_data['next_contra_heelstrike'] = gait_event_data['next_contra'].map(gait_event_data['initial_contact_kinematic_frame_100hz'])
gait_event_data['prev_contra_toeoff']     = gait_event_data['prev_contra'].map(gait_event_data['toe_off_kinematic_frame_100hz'])
gait_event_data['next_contra_toeoff']     = gait_event_data['next_contra'].map(gait_event_data['toe_off_kinematic_frame_100hz'])

gait_event_data['stance_time'] = np.where(
    ~np.isnan(gait_event_data['toe_off_kinematic_frame_100hz']) & ~np.isnan(gait_event_data['initial_contact_kinematic_frame_100hz']) &
    (gait_event_data['toe_off_kinematic_frame_100hz'] > gait_event_data['initial_contact_kinematic_frame_100hz']),
    (gait_event_data['toe_off_kinematic_frame_100hz'] - gait_event_data['initial_contact_kinematic_frame_100hz']) /100,
    np.nan
    )

gait_event_data['step_time'] = np.where(
    ~np.isnan(gait_event_data['next_contra_heelstrike']) & ~np.isnan(gait_event_data['initial_contact_kinematic_frame_100hz']) &
    (gait_event_data['next_contra_heelstrike'] > gait_event_data['initial_contact_kinematic_frame_100hz']),
    (gait_event_data['next_contra_heelstrike'] - gait_event_data['initial_contact_kinematic_frame_100hz']) /100,
    np.nan
    )

gait_event_data['swing_time'] = np.where(
    ~np.isnan(gait_event_data['next_ipsi_heelstrike']) & ~np.isnan(gait_event_data['toe_off_kinematic_frame_100hz']) &
    (gait_event_data['next_ipsi_heelstrike'] > gait_event_data['toe_off_kinematic_frame_100hz']),
    (gait_event_data['next_ipsi_heelstrike'] - gait_event_data['toe_off_kinematic_frame_100hz']) /100,
    np.nan
    )

gait_event_data['stride_time'] = np.where(
    ~np.isnan(gait_event_data['next_ipsi_heelstrike']) & ~np.isnan(gait_event_data['initial_contact_kinematic_frame_100hz']) &
    (gait_event_data['next_ipsi_heelstrike'] > gait_event_data['initial_contact_kinematic_frame_100hz']),
    (gait_event_data['next_ipsi_heelstrike'] - gait_event_data['initial_contact_kinematic_frame_100hz'])/100,
    np.nan
    )

gait_event_data['stance_percentage'] = np.where(
    ~np.isnan(gait_event_data['stance_time']) & ~np.isnan(gait_event_data['stride_time']) &
    (gait_event_data['stride_time'] > gait_event_data['stance_time']),
    gait_event_data['stance_time'] / gait_event_data['stride_time'] * 100,
    np.nan
    )

gait_event_data['swing_percentage'] = np.where(
    ~np.isnan(gait_event_data['swing_time']) & ~np.isnan(gait_event_data['stride_time']) &
    (gait_event_data['stride_time'] > gait_event_data['swing_time']),
    gait_event_data['swing_time'] / gait_event_data['stride_time'] * 100,
    np.nan
    )

#brain check
for x in range(len(gait_event_data)):
    if np.round(gait_event_data['swing_percentage'][x] + gait_event_data['stance_percentage'][x]) != 100 and ~np.isnan(np.round(gait_event_data['swing_percentage'][x] + gait_event_data['stance_percentage'][x])):
        print(f" Something is wrong in percentages at index {x}")

gait_event_data['single_support_time'] = np.where(
    ~np.isnan(gait_event_data['next_contra_heelstrike']) & ~np.isnan(gait_event_data['prev_contra_toeoff']) &
    (gait_event_data['next_contra_heelstrike'] > gait_event_data['prev_contra_toeoff']),
    (gait_event_data['next_contra_heelstrike'] - gait_event_data['prev_contra_toeoff'])/100,
    np.nan
    )

gait_event_data['double_support_time'] = np.where(
    ~np.isnan(gait_event_data['prev_contra_toeoff']) & ~np.isnan(gait_event_data['initial_contact_kinematic_frame_100hz']) &
    ~np.isnan(gait_event_data['toe_off_kinematic_frame_100hz']) & ~np.isnan(gait_event_data['next_contra_heelstrike']) &
    (gait_event_data['toe_off_kinematic_frame_100hz'] > gait_event_data['next_contra_heelstrike']) &
    (gait_event_data['prev_contra_toeoff'] > gait_event_data['initial_contact_kinematic_frame_100hz']),
    ((gait_event_data['prev_contra_toeoff'] - gait_event_data['initial_contact_kinematic_frame_100hz']) + (gait_event_data['toe_off_kinematic_frame_100hz'] - gait_event_data['next_contra_heelstrike']))/100,
    np.nan
    )

gait_event_data['single_support_percentage'] = np.where(
    ~np.isnan(gait_event_data['stance_time']) & ~np.isnan(gait_event_data['single_support_time']) &
    (gait_event_data['stance_time'] > gait_event_data['single_support_time']),
    gait_event_data['single_support_time'] / gait_event_data['stance_time'] * 100,
    np.nan
    )

gait_event_data['double_support_percentage'] = np.where(
    ~np.isnan(gait_event_data['stance_time']) & ~np.isnan(gait_event_data['double_support_time']) &
    (gait_event_data['stance_time'] > gait_event_data['double_support_time']),
    gait_event_data['double_support_time'] / gait_event_data['stance_time'] * 100,
    np.nan
    )

#cadence in steps per minute. step_time is already in seconds (the frame
#difference above is divided by the 100 Hz sample rate), so the conversion is
#just 60 / step_time. The previous version divided by an extra factor of 100,
#which inflated cadence by 100x (~10,900 steps/min instead of ~109).
gait_event_data['cadence'] = np.where(
    ~np.isnan(gait_event_data['step_time']),
    60 / gait_event_data['step_time'],
    np.nan
    )
#brain check
for x in range(len(gait_event_data)):
    if np.round(gait_event_data['single_support_percentage'][x] + gait_event_data['double_support_percentage'][x]) != 100 and ~np.isnan(np.round(gait_event_data['single_support_percentage'][x] + gait_event_data['double_support_percentage'][x])):
        print(f" Something is wrong in percentages at index {x}")

plt.figure()
plt.plot(gait_event_data['stride_time'])
plt.title('Stride time (s)')

plt.figure()
plt.plot(gait_event_data['stance_time'])
plt.title('Stance (s)')

plt.figure()
plt.plot(gait_event_data['single_support_percentage'])
plt.title('Single support (%)')


kinematic_flag = ((gait_event_data['heel_strike_source'] == 'kinematic') |
                  (gait_event_data['toe_off_source'] == 'kinematic'))

plt.figure()
plt.plot(gait_event_data['stance_time'])
plt.plot(gait_event_data.index[kinematic_flag],
         gait_event_data['stance_time'][kinematic_flag],
         'x', color='red', markersize=8, linestyle='none', label='kinematic source')
plt.title('Stance (s)')
plt.legend()


#brain check the temporal metrics
print(gait_event_data[['stance_percentage', 'swing_percentage', 'cadence']].describe())
print(gait_event_data[['single_support_percentage', 'double_support_percentage']].describe())
# =============================================================================
# %% now time for spatial metrics
# =============================================================================

frame = gait_event_data['next_contra_heelstrike']

rx = kinematic_data['Right_Heel_Position'].reindex(frame).to_numpy()
ry = kinematic_data['Right_Heel_Position.1'].reindex(frame).to_numpy()
lx = kinematic_data['Left_Heel_Position'].reindex(frame).to_numpy()
ly = kinematic_data['Left_Heel_Position.1'].reindex(frame).to_numpy()

#contralateral limb is the one landing so it leads
lead_is_left = (gait_event_data['support_limb'] == 'R').to_numpy()

lead_ap  = np.where(lead_is_left, ly, ry)
trail_ap = np.where(lead_is_left, ry, ly)
lead_ml  = np.where(lead_is_left, lx, rx)
trail_ml = np.where(lead_is_left, rx, lx)

gait_event_data['step_length'] = np.where(
    ~np.isnan(lead_ap) & ~np.isnan(trail_ap),
    np.abs(lead_ap - trail_ap),
    np.nan
    )

gait_event_data['step_width'] = np.where(
    ~np.isnan(lead_ml) & ~np.isnan(trail_ml),
    np.abs(lead_ml - trail_ml),
    np.nan
    )

gait_event_data['stride_length'] = np.where(
    ~np.isnan(gait_event_data['step_length']) &
    ~np.isnan(gait_event_data['next_contra'].map(gait_event_data['step_length'])),
    gait_event_data['step_length'] + gait_event_data['next_contra'].map(gait_event_data['step_length']),
    np.nan
    )

gait_event_data['stride_velocity'] = np.where(
    ~np.isnan(gait_event_data['stride_time']) & ~np.isnan(gait_event_data['stride_length']),
    gait_event_data['stride_length']/gait_event_data['stride_time'],
    np.nan
    )

#brain check
print(gait_event_data[['step_length', 'step_width', 'stride_length']].describe())
print(gait_event_data.groupby('support_limb')[['step_length', 'step_width']].agg(['mean', 'std', 'count']))

plt.figure()
plt.plot(gait_event_data['step_length'])
plt.title('Step length (m)')

plt.figure()
plt.plot(gait_event_data['step_width'])
plt.title('Step width (m)')

plt.figure()
plt.plot(gait_event_data['stride_length'])
plt.title('Stride length (m)')

#Lets try to do some minimum foot clearance with what we have so far
foot_clearance = pd.DataFrame()
foot_clearance['frame'] = kinematic_data['Unnamed: 0']
foot_clearance['Left_min_z'] = np.where(
    ~np.isnan(kinematic_data['Left_Heel_Position.2']) & ~np.isnan(kinematic_data['Left_Toes_Position.2']),
    np.min([kinematic_data['Left_Heel_Position.2'], kinematic_data['Left_Toes_Position.2']], axis=0),
    np.nan
    )
foot_clearance['Right_min_z'] = np.where(
    ~np.isnan(kinematic_data['Right_Heel_Position.2']) & ~np.isnan(kinematic_data['Right_Toes_Position.2']),
    np.min([kinematic_data['Right_Heel_Position.2'], kinematic_data['Right_Toes_Position.2']], axis=0),
    np.nan
    )

#toe AP velocity in m/s, .1 is the AP axis and the data is 100 Hz
foot_clearance['Left_toe_ap_velocity']  = np.gradient(kinematic_data['Left_Toes_Position.1'].to_numpy()) * 100
foot_clearance['Right_toe_ap_velocity'] = np.gradient(kinematic_data['Right_Toes_Position.1'].to_numpy()) * 100

#map event frames onto row positions in the kinematic data
frame_position = pd.Series(np.arange(len(foot_clearance)), index=foot_clearance['frame'].to_numpy())

gait_event_data['minimum_foot_clearance'] = np.nan
gait_event_data['minimum_foot_clearance_frame'] = np.nan

for x in range(len(gait_event_data)):
    toe_off = gait_event_data['toe_off_kinematic_frame_100hz'][x]
    next_hs = gait_event_data['next_ipsi_heelstrike'][x]

    if np.isnan(toe_off) or np.isnan(next_hs) or next_hs <= toe_off:
        continue
    if int(toe_off) not in frame_position.index or int(next_hs) not in frame_position.index:
        print(f" Swing frames not found in kinematic data at index {x}")
        continue

    start = frame_position[int(toe_off)]
    stop  = frame_position[int(next_hs)]

    side = 'Left' if gait_event_data['support_limb'][x] == 'L' else 'Right'
    swing_z   = foot_clearance[f'{side}_min_z'].to_numpy()[start:stop+1]
    swing_vel = foot_clearance[f'{side}_toe_ap_velocity'].to_numpy()[start:stop+1]

    #keep only the frames where the toe is in the fastest quarter of that swing
    upper_quartile = swing_vel >= np.nanpercentile(swing_vel, 75)
    candidate = np.where(upper_quartile & ~np.isnan(swing_z), swing_z, np.nan)

    if np.all(np.isnan(candidate)):
        continue

    gait_event_data.loc[x, 'minimum_foot_clearance'] = np.nanmin(candidate)
    gait_event_data.loc[x, 'minimum_foot_clearance_frame'] = foot_clearance['frame'].to_numpy()[start + np.nanargmin(candidate)]

plt.figure()
plt.plot(gait_event_data['minimum_foot_clearance'])
plt.title('Minimum foot clearance (m)')

# =============================================================================
# %% Local dynamic stability of the trunk (short- and long-term Lyapunov)
# =============================================================================
# Uses kinematic_data and gait_event_data that already exist above.
# Nothing is re-read from disk.
#
# -----------------------------------------------------------------------------
# METHOD SUMMARY
# -----------------------------------------------------------------------------
# The maximum finite-time Lyapunov exponent is estimated with the algorithm of
# Rosenstein et al. (1993) [Rosenstein1993], applied to a state space
# reconstructed by delay embedding (Takens 1981 [Takens1981]) or built directly
# from measured physical states. This is the standard approach in the gait
# literature following Dingwell & Cusumano (2000) [Dingwell2000].
#
#   1.  One signal is conditioned and differentiated in REAL TIME, before any
#       stride normalisation, using a single Savitzky-Golay stage
#       [Savitzky1964]. Filtering demonstrably changes lambda, so the filter is
#       one stage, is identical for every trial, and its effective bandwidth is
#       recorded in the output [Raffalt2020, Mehdizadeh2019].
#   2.  The series is cut to a fixed number of strides (default 150) starting
#       after a warm-up period. lambda depends on the amount of data, so the
#       stride count must be identical across every trial you intend to compare
#       [Bruijn2009a].
#   3.  Optionally each stride is resampled to a fixed number of samples, so
#       that samples-per-stride does not vary with cadence [Dingwell2000,
#       Dingwell2006]. This is ON by default because this protocol manipulates
#       cadence, which would otherwise confound lambda directly.
#   4.  tau and dE are HARD CODED and FIXED for the whole dataset. They are not
#       re-estimated per trial, because per-trial parameters make the
#       reconstructed geometry condition-dependent and therefore make lambda
#       non-comparable [Bruijn2009a, vanSchooten2013, Raffalt2019].
#   5.  For every reference point the nearest neighbour is found subject to a
#       Theiler exclusion window of one stride, so that temporally correlated
#       points are not mistaken for dynamical neighbours [Theiler1986].
#   6.  The mean log divergence curve <ln d(i)> is built over a SINGLE FIXED
#       SET of neighbour pairs that survives the entire curve length, so the
#       ensemble does not change with lag.
#   7.  lambda_S is the slope over the first half stride [Bruijn2009a]; a second
#       fit over 0-1 stride is reported alongside it as a sensitivity check
#       [Dingwell2000, England2007]. lambda_L is fit over 4-10 strides
#       [Dingwell2000] but should be treated with caution: its reliability and
#       construct validity are poor [Bruijn2009a, Bruijn2013].
#   8.  Uncertainty is reported: a bootstrap CI over reference points and a
#       stride-based split-half difference [Kang2006, Reynard2014].
#
# -----------------------------------------------------------------------------
# WHAT YOU MUST SET BEFORE USING THIS ON REAL DATA
# -----------------------------------------------------------------------------
#   LDS_CONFIG['tau']           <- from the dataset-level parameter script
#   LDS_CONFIG['dE']            <- from the dataset-level parameter script
#   LDS_CONFIG['leg_length_m']  <- per participant, from anthropometrics
#
#   tau is expressed in samples OF THE SERIES THAT IS ANALYSED. With
#   time_normalize = True that means NORMALISED samples (default 100 per
#   stride), NOT samples at 100 Hz. The dataset-level script that chooses tau
#   must therefore apply the identical conditioning and normalisation before
#   computing AMI, or the value will not transfer.
#
# -----------------------------------------------------------------------------
# SIGNAL SPEC: ('Base_Column', 'axis', 'derivative')
#      axis       'X' -> Base_Column        (ML on this collection)
#                 'Y' -> Base_Column.1      (AP)
#                 'Z' -> Base_Column.2      (vertical)
#      derivative 'pos' | 'vel' | 'acc'
#
# EMBED    'delay' = Takens delay embedding, one COMMON tau for every channel
#                    (1 signal = univariate, N signals = multivariate,
#                    total dim = N * dE)
#          'none'  = the listed signals ARE the state variables
#
# SCALING  'none'   raw units. Safe only when every channel shares a unit, or
#                   when there is a single channel: lambda is invariant to a
#                   uniform rescaling of the whole state space (it shifts the
#                   intercept of <ln d(i)>, not the slope). PER-CHANNEL scaling
#                   is NOT uniform and does change the estimate.
#          'zscore' each channel / a reference SD. Use a FIXED reference SD
#                   (config['zscore_reference_sd']); dividing by each trial's
#                   own SD gives every condition a different geometry.
#          'nondim' one length scale (leg length) and ONE time scale, after
#                   Hof (1996) [Hof1996]:
#                       linear pos * 1/L      angular pos * 1
#                       linear vel * T/L      angular vel * T
#                       linear acc * T^2/L    angular acc * T^2
#                   The previous version mixed two time scales (stride time for
#                   angles, sqrt(L/g) for linear channels), which made the
#                   relative channel weighting inconsistent inside any state
#                   space containing both.
#
# DETREND  applies to 'pos' channels only.
#          None       leave as is
#          'linear'   remove a straight line over the window
#          'highpass' zero-lag Butterworth high pass on the full record.
#                     Preferred on a treadmill: a linear detrend removes net
#                     drift but leaves low-frequency postural wander, which has
#                     far more power than the gait-cycle oscillation and makes
#                     the nearest-neighbour search collapse onto temporally
#                     adjacent points.
# =============================================================================

LDS_REFERENCES = """
References for the local dynamic stability analysis.
NOTE: compiled from the literature by hand. Verify volume/page numbers and DOIs
against your reference manager before these go into a manuscript.

[Takens1981]     Takens, F. (1981). Detecting strange attractors in turbulence.
                 In: Dynamical Systems and Turbulence, Warwick 1980. Lecture
                 Notes in Mathematics, vol 898. Springer, 366-381.
[Savitzky1964]   Savitzky, A., Golay, M.J.E. (1964). Smoothing and
                 differentiation of data by simplified least squares
                 procedures. Analytical Chemistry, 36(8), 1627-1639.
[Wolf1985]       Wolf, A., Swift, J.B., Swinney, H.L., Vastano, J.A. (1985).
                 Determining Lyapunov exponents from a time series. Physica D,
                 16(3), 285-317.
[Theiler1986]    Theiler, J. (1986). Spurious dimension from correlation
                 algorithms applied to limited time-series data. Physical
                 Review A, 34(3), 2427-2432.
[Fraser1986]     Fraser, A.M., Swinney, H.L. (1986). Independent coordinates
                 for strange attractors from mutual information. Physical
                 Review A, 33(2), 1134-1140.
[Kennel1992]     Kennel, M.B., Brown, R., Abarbanel, H.D.I. (1992). Determining
                 embedding dimension for phase-space reconstruction using a
                 geometrical construction. Physical Review A, 45(6), 3403-3411.
[Theiler1992]    Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., Farmer,
                 J.D. (1992). Testing for nonlinearity in time series: the
                 method of surrogate data. Physica D, 58(1-4), 77-94.
[Rosenstein1993] Rosenstein, M.T., Collins, J.J., De Luca, C.J. (1993). A
                 practical method for calculating largest Lyapunov exponents
                 from small data sets. Physica D, 65(1-2), 117-134.
[Kantz1994]      Kantz, H. (1994). A robust method to estimate the maximal
                 Lyapunov exponent of a time series. Physics Letters A, 185(1),
                 77-87.
[Hof1996]        Hof, A.L. (1996). Scaling gait data to body size. Gait &
                 Posture, 4(3), 222-223.
[Dingwell2000]   Dingwell, J.B., Cusumano, J.P. (2000). Nonlinear time series
                 analysis of normal and pathological human walking. Chaos,
                 10(4), 848-863.
[Small2001]      Small, M., Yu, D., Harrison, R.G. (2001). Surrogate test for
                 pseudoperiodic time series data. Physical Review Letters,
                 87(18), 188101.
[Sprott2003]     Sprott, J.C. (2003). Chaos and Time-Series Analysis. Oxford
                 University Press. (Source of the reference Lorenz exponent
                 lambda_1 = 0.9056 used by the self test.)
[Kraskov2004]    Kraskov, A., Stoegbauer, H., Grassberger, P. (2004).
                 Estimating mutual information. Physical Review E, 69(6),
                 066138.
[Kang2006]       Kang, H.G., Dingwell, J.B. (2006). Intra-session reliability
                 of local dynamic stability of walking. Gait & Posture, 24(3),
                 386-390.
[Dingwell2006]   Dingwell, J.B., Marin, L.C. (2006). Kinematic variability and
                 local dynamic stability of upper body motions when walking at
                 different speeds. Journal of Biomechanics, 39(3), 444-452.
[England2007]    England, S.A., Granata, K.P. (2007). The influence of gait
                 speed on local dynamic stability of walking. Gait & Posture,
                 25(2), 172-178.
[Gates2009]      Gates, D.H., Dingwell, J.B. (2009). Comparison of different
                 state space definitions for local dynamic stability analyses.
                 Journal of Biomechanics, 42(9), 1345-1349.
[Bruijn2009a]    Bruijn, S.M., van Dieen, J.H., Meijer, O.G., Beek, P.J.
                 (2009). Statistical precision and sensitivity of measures of
                 dynamic gait stability. Journal of Neuroscience Methods,
                 178(2), 327-333.
[Bruijn2009b]    Bruijn, S.M., van Dieen, J.H., Meijer, O.G., Beek, P.J.
                 (2009). Is slow walking more stable? Journal of Biomechanics,
                 42(10), 1506-1512.
[Winter2009]     Winter, D.A. (2009). Biomechanics and Motor Control of Human
                 Movement, 4th ed. Wiley. (Residual analysis for cutoff
                 selection.)
[Sloot2011]      Sloot, L.H., van Schooten, K.S., Bruijn, S.M., Kingma, H.,
                 Pijnappels, M., van Dieen, J.H. (2011). Sensitivity of local
                 dynamic stability of over-ground walking to balance impairment
                 due to galvanic vestibular stimulation. Annals of Biomedical
                 Engineering, 39(5), 1563-1569.
[Terrier2011]    Terrier, P., Deriaz, O. (2011). Kinematic variability, fractal
                 dynamics and local dynamic stability of treadmill walking.
                 Journal of NeuroEngineering and Rehabilitation, 8, 12.
[Bruijn2013]     Bruijn, S.M., Meijer, O.G., Beek, P.J., van Dieen, J.H.
                 (2013). Assessing the stability of human locomotion: a review
                 of current measures. Journal of the Royal Society Interface,
                 10(83), 20120999.
[vanSchooten2013] van Schooten, K.S., Rispens, S.M., Pijnappels, M.,
                 Daffertshofer, A., van Dieen, J.H. (2013). Assessing gait
                 stability: the influence of state space reconstruction on
                 inter- and intra-day reliability of local dynamic stability
                 during over-ground walking. Journal of Biomechanics, 46(1),
                 137-141.
[Reynard2014]    Reynard, F., Terrier, P. (2014). Local dynamic stability of
                 treadmill walking: intrasession and week-to-week
                 repeatability. Journal of Biomechanics, 47(1), 74-80.
[Stenum2014]     Stenum, J., Bruijn, S.M., Jensen, B.R. (2014). The effect of
                 walking speed on local dynamic stability is sensitive to
                 calculation methods. Journal of Biomechanics, 47(15),
                 3776-3779.
[Mehdizadeh2018] Mehdizadeh, S. (2018). The largest Lyapunov exponent of gait
                 in young and elderly individuals: a systematic review. Gait &
                 Posture, 60, 241-250.
[Mehdizadeh2019] Mehdizadeh, S. (2019). A robust method to estimate the largest
                 Lyapunov exponent of noisy signals: a revision to the
                 Rosenstein's algorithm. Journal of Biomechanics, 85, 84-91.
[Raffalt2019]    Raffalt, P.C., Kent, J.A., Wurdeman, S.R., Stergiou, N.
                 (2019). Selection procedures for the largest Lyapunov exponent
                 in gait biomechanics. Annals of Biomedical Engineering, 47(4),
                 913-923.
[Raffalt2020]    Raffalt, P.C., Senderling, B., Stergiou, N. (2020). Filtering
                 affects the calculation of the largest Lyapunov exponent.
                 Computers in Biology and Medicine, 122, 103786.
[Kanko2021]      Kanko, R.M., Laende, E.K., Davis, E.M., Selbie, W.S., Deluzio,
                 K.J. (2021). Concurrent assessment of gait kinematics using
                 marker-based and markerless motion capture. Journal of
                 Biomechanics, 127, 110665.
"""

import json
import platform

import scipy
from scipy.signal import butter, filtfilt, savgol_filter, savgol_coeffs, freqz
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

# -----------------------------------------------------------------------------
# %% settings
# -----------------------------------------------------------------------------
# Everything the analysis depends on lives in this one dict, and the whole dict
# is written into the output file so any result can be traced back to the
# settings that produced it.

LDS_CONFIG = {

    # --- acquisition ---------------------------------------------------------
    'fs': 100.0,                 # Hz, matches the frame_100hz event convention
    'limb': 'R',                 # which limb's heel strikes define the strides

    # --- analysis window -----------------------------------------------------
    # Discard the start of the trial: treadmill gait needs an acclimatisation
    # period and the first strides are gait initiation, not steady state.
    'warmup_seconds': 60.0,
    'start_stride': None,        # None -> derived from warmup_seconds
    # Fixed stride count. lambda is a function of the amount of data, so this
    # MUST be identical for every trial being compared [Bruijn2009a].
    'n_strides': 150,

    # --- signal conditioning -------------------------------------------------
    # ONE smoothing stage only. v3 cascaded a 10 Hz Butterworth into a
    # Savitzky-Golay differentiator, giving an unquantified combined response;
    # filtering changes lambda, so the bandwidth has to be single valued and
    # reportable [Raffalt2020]. The effective -3 dB cutoff of the setting below
    # is computed at run time and written to the output.
    'savgol_window': 11,         # odd, samples
    'savgol_poly': 3,
    'highpass_hz': 0.2,          # for detrend='highpass' on position channels.
                                 # Well below stride frequency (~0.9 Hz here) so
                                 # the gait-cycle component is not attenuated.
    'highpass_order': 2,
    'angles_in_degrees': True,   # any column containing 'Angle' -> radians

    # --- missing data QC -----------------------------------------------------
    # v3 accepted a channel with up to 50% NaN and linearly interpolated it. An
    # interpolated span is artificially smooth and locally low dimensional and
    # will drag lambda down, so the tolerance here is tight and is evaluated on
    # the analysis window rather than on the whole record.
    'max_gap_frames': 5,         # longest run of consecutive NaN allowed
    'max_nan_fraction': 0.02,    # total NaN fraction allowed in the window
    'edge_pad_seconds': 5.0,     # QC margin either side of the window;
                                 # also covers the high-pass transient

    # --- stride time normalisation -------------------------------------------
    # ON by default. This protocol imposes cadence, and without normalisation a
    # slower cadence gives more samples per stride, a denser attractor and a
    # different lambda for purely methodological reasons [Dingwell2000,
    # Stenum2014]. It also removes the lag-to-stride smearing that otherwise
    # grows with lag in proportion to stride time variability, which is what
    # makes lambda_L partly a proxy for stride time variability.
    'time_normalize': True,
    'samples_per_stride_norm': 100,

    # --- embedding: FIXED ACROSS THE WHOLE DATASET ---------------------------
    # These are deliberately hard coded. Re-estimating tau per trial (as v3 did
    # with tau_mode='auto') gives every participant and condition its own
    # reconstructed geometry, and lambda is not invariant to tau at finite
    # sample size [Bruijn2009a, vanSchooten2013, Raffalt2019].
    #
    # !! PLACEHOLDERS !! Replace with the dataset-wide values from the separate
    # parameter script, and report in the manuscript how they were chosen.
    # Units: samples of the ANALYSED series. With time_normalize=True that is
    # normalised samples (100 per stride), not 100 Hz samples.
    'tau': 10,
    'dE': 5,

    # --- divergence curve ----------------------------------------------------
    'n_strides_curve': 10,       # length of the divergence curve, in strides
    # lambda_S over the first half stride, following Bruijn et al. The 0-1
    # stride convention of Dingwell is reported alongside as a sensitivity
    # check so the reader can see the result is not knife-edge on this choice.
    'short_window': (0.0, 0.5),          # [Bruijn2009a]
    # (0.0, 1.0) is the Dingwell convention [Dingwell2000, England2007].
    # (0.1, 0.5) is not a literature convention: it is a diagnostic. The
    # divergence curve has a steep transient at the very start caused by
    # nearest neighbours being selected as a minimum and therefore starting
    # anomalously close (see lds_self_test). Comparing it against the 0-0.5
    # fit shows how much of lambda_S is that artifact rather than dynamics.
    'short_window_sensitivity': [(0.0, 1.0), (0.1, 0.5)],
    'long_window': (4.0, 10.0),          # [Dingwell2000]
    # Exclude temporally correlated neighbours [Theiler1986]. One stride is the
    # usual choice in gait; it forces the neighbour to come from a different
    # stride rather than from a few samples away on the same trajectory.
    'theiler_strides': 1.0,
    'nn_k_initial': 200,         # neighbours requested per query, grown if needed
    'nn_k_max': 2000,
    # Reference points whose nearest valid neighbour is unusually far away do
    # not measure local divergence, they measure global attractor geometry.
    # Set to None to keep all of them and only report the distribution.
    'max_d0_percentile': 95.0,
    'max_ref_points': 6000,      # subsample reference points, None = use all
    'random_seed': 0,

    # --- scaling references --------------------------------------------------
    # leg_length_m MUST be set per participant. v3 hard coded 0.90 with a
    # comment saying "set per participant", which in a batch would silently be
    # wrong for everyone, and because it is a per-channel scale factor it
    # changes which points are nearest neighbours and therefore changes lambda.
    'leg_length_m': None,
    # Nondimensionalising by the trial's own stride time makes the state space
    # geometry condition-dependent. For between-condition comparisons set this
    # to one fixed reference (e.g. the participant's baseline condition).
    'nondim_reference_stride_time_s': None,
    # Same argument for z-scoring: {('Base','Axis','kind'): sd}. None falls back
    # to this trial's own SD and raises a QC warning.
    'zscore_reference_sd': None,

    # --- uncertainty ---------------------------------------------------------
    'n_bootstrap': 200,          # bootstrap over reference points, 0 = off
    'bootstrap_ci': 95.0,
    'run_split_half': True,      # first 75 strides vs last 75 strides

    # --- optional extras -----------------------------------------------------
    # Validate the implementation against systems with known exponents. Cheap,
    # and it converts "I wrote a Rosenstein implementation" into "my
    # implementation recovers known exponents". Also reports the value the
    # pipeline returns for a clean periodic signal, i.e. your noise floor.
    'run_selftest': True,
    # Surrogate data test [Theiler1992, Small2001]: does lambda reflect
    # deterministic local divergence or just the noise floor? Expensive, so run
    # it once per dataset rather than on every trial. Set to e.g. 20 to enable.
    'n_surrogates': 0,
    'surrogate_kind': 'stride_shuffle',   # 'stride_shuffle' | 'phase_randomise'
    # Sanity check only: compare this trial's AMI minimum / FNN knee against the
    # hard-coded tau and dE. It reports a mismatch, it NEVER changes tau or dE.
    'run_parameter_check': True,
    'param_check_signal': ('Trunk_Position', 'X', 'vel'),
    'ami_max_lag': 100,
    'ami_bins': 32,
    'fnn_max_dim': 12,
    'fnn_rtol': 15.0,            # [Kennel1992]
    'fnn_atol': 2.0,
}

# -----------------------------------------------------------------------------
# %% state space definitions
# -----------------------------------------------------------------------------
# ONE state space carries 'primary': True. Everything else is a pre-declared
# sensitivity analysis. Nine state spaces x two exponents is eighteen outcomes
# per trial; without a designated primary the output table is an invitation to
# report whichever one is significant. Declare the primary in the protocol,
# and handle multiplicity explicitly if you report the rest.
#
# The primary is trunk ML velocity: velocity-based trunk state spaces are the
# best supported and most reliable choice in the gait literature
# [Gates2009, vanSchooten2013, Bruijn2013], and the mediolateral direction is
# the one most sensitive to balance impairment [Sloot2011, Bruijn2013].

LDS_STATE_SPACES = {

    'A1_trunkVel_ML_delay': {
        'signals': [('Trunk_Position', 'X', 'vel')],
        'embed': 'delay', 'dE': None, 'scaling': 'none', 'detrend': None,
        'primary': True,
        'note': 'Primary outcome. Single channel, so scaling="none" is safe: '
                'lambda is invariant to a uniform rescale of the state space.'},

    'A2_trunkVel_AP_delay': {
        'signals': [('Trunk_Position', 'Y', 'vel')],
        'embed': 'delay', 'dE': None, 'scaling': 'none', 'detrend': None},

    'A3_trunkVel_VT_delay': {
        'signals': [('Trunk_Position', 'Z', 'vel')],
        'embed': 'delay', 'dE': None, 'scaling': 'none', 'detrend': None},

    'B1_trunkAcc_ML_delay': {
        'signals': [('Trunk_Position', 'X', 'acc')],
        'embed': 'delay', 'dE': None, 'scaling': 'none', 'detrend': None,
        'note': 'Acceleration is a second derivative of the measured position '
                'and so is the noisiest channel here; expect it to be the most '
                'filter-sensitive [Raffalt2020].'},

    'C1_thoraxAngle_sag_delay': {
        'signals': [('Thorax_Seg_Angle', 'X', 'pos')],
        'embed': 'delay', 'dE': None, 'scaling': 'none', 'detrend': None},

    'D1_trunkVel_3D_multivar': {
        'signals': [('Trunk_Position', 'X', 'vel'),
                    ('Trunk_Position', 'Y', 'vel'),
                    ('Trunk_Position', 'Z', 'vel')],
        'embed': 'delay', 'dE': 3, 'scaling': 'none', 'detrend': None,
        'note': 'dE is overridden to 3 to keep the total dimension at 9. Note '
                'the FNN diagnostic is univariate and does not inform this '
                'choice; it needs its own dataset-level justification. All '
                'three channels share one unit and one common tau.'},

    'E1_thoraxAngle_6D_physical': {
        'signals': [('Thorax_Seg_Angle', 'X', 'pos'),
                    ('Thorax_Seg_Angle', 'Y', 'pos'),
                    ('Thorax_Seg_Angle', 'Z', 'pos'),
                    ('Thorax_Seg_Angle', 'X', 'vel'),
                    ('Thorax_Seg_Angle', 'Y', 'vel'),
                    ('Thorax_Seg_Angle', 'Z', 'vel')],
        'embed': 'none', 'dE': None, 'scaling': 'nondim', 'detrend': None,
        'note': 'The velocity channels are filtered derivatives of the position '
                'channels, so the six coordinates are not independent.'},

    # --- position-based spaces: sensitivity only ------------------------------
    # Absolute trunk position on a treadmill wanders at frequencies well below
    # the stride. Even after a high pass these spaces are the most likely to be
    # dominated by postural drift rather than gait dynamics. Reported for
    # completeness; not recommended as an outcome.
    'E2_trunkPosVel_6D_physical': {
        'signals': [('Trunk_Position', 'X', 'pos'),
                    ('Trunk_Position', 'Y', 'pos'),
                    ('Trunk_Position', 'Z', 'pos'),
                    ('Trunk_Position', 'X', 'vel'),
                    ('Trunk_Position', 'Y', 'vel'),
                    ('Trunk_Position', 'Z', 'vel')],
        'embed': 'none', 'dE': None, 'scaling': 'nondim', 'detrend': 'highpass'},

    'E3_trunk12D_physical': {
        'signals': [('Trunk_Position', 'X', 'pos'),
                    ('Trunk_Position', 'Y', 'pos'),
                    ('Trunk_Position', 'Z', 'pos'),
                    ('Thorax_Seg_Angle', 'X', 'pos'),
                    ('Thorax_Seg_Angle', 'Y', 'pos'),
                    ('Thorax_Seg_Angle', 'Z', 'pos'),
                    ('Trunk_Position', 'X', 'vel'),
                    ('Trunk_Position', 'Y', 'vel'),
                    ('Trunk_Position', 'Z', 'vel'),
                    ('Thorax_Seg_Angle', 'X', 'vel'),
                    ('Thorax_Seg_Angle', 'Y', 'vel'),
                    ('Thorax_Seg_Angle', 'Z', 'vel')],
        'embed': 'none', 'dE': None, 'scaling': 'nondim', 'detrend': 'highpass',
        'note': 'Mixes linear and angular channels, so the single-time-scale '
                'nondimensionalisation matters here more than anywhere else.'},
}

# -----------------------------------------------------------------------------
# %% helpers: signal conditioning
# -----------------------------------------------------------------------------

def _savgol_cutoff_hz(window, poly, fs):
    """-3 dB cutoff of the Savitzky-Golay smoother, so the effective bandwidth
    of the one conditioning stage is a reportable number [Raffalt2020]."""
    c = savgol_coeffs(window, poly, deriv=0, use='conv')
    w, h = freqz(c, worN=8192, fs=fs)
    mag = np.abs(h)
    below = np.flatnonzero(mag < mag[0] / np.sqrt(2.0))
    return float(w[below[0]]) if below.size else float(fs / 2.0)


def _nan_runs(mask):
    """Start/stop (exclusive) index pairs for each run of True."""
    if not mask.any():
        return np.empty((0, 2), dtype=int)
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.column_stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)])


def _fill_gaps(x):
    """PCHIP fill of short gaps. Shape preserving, unlike the linear
    interpolation in v3, which puts a corner at every gap edge that the
    smoother then has to deal with."""
    ok = np.isfinite(x)
    if ok.all():
        return x
    idx = np.arange(len(x))
    out = PchipInterpolator(idx[ok], x[ok], extrapolate=False)(idx)
    first, last = idx[ok][0], idx[ok][-1]
    out[:first] = x[first]          # PCHIP will not extrapolate; hold the ends
    out[last + 1:] = x[last]
    return out


def _prepare_signal(kinematic_data, spec, detrend, cfg, i0, i1):
    """
    Condition one channel and return it over the analysis window.

    Order of operations matters. Gap QC, filtering and differentiation all
    happen on the full record in REAL TIME, before the window is cut and before
    any stride normalisation, so that (a) the filter transient never lands
    inside the analysed window and (b) the derivative is a true time derivative
    rather than a derivative with respect to normalised phase [Dingwell2000].
    """
    base_col, axis, kind = spec
    col = base_col + {'X': '', 'Y': '.1', 'Z': '.2'}[axis.upper()]
    qc = {'channel': f'{base_col}_{axis}|{kind}', 'detrend': str(detrend)}

    if col not in kinematic_data.columns:
        qc['status'] = 'missing_column'
        return None, qc

    x = pd.to_numeric(kinematic_data[col], errors='coerce').to_numpy(float)

    # --- missing data QC, evaluated on the window plus a filter margin -------
    pad = int(round(cfg['edge_pad_seconds'] * cfg['fs']))
    lo, hi = max(0, i0 - pad), min(len(x), i1 + 1 + pad)
    bad = ~np.isfinite(x[lo:hi])
    runs = _nan_runs(bad)
    qc['longest_gap_frames'] = int((runs[:, 1] - runs[:, 0]).max()) if len(runs) else 0
    qc['nan_fraction'] = float(bad.mean())

    if qc['longest_gap_frames'] > cfg['max_gap_frames']:
        qc['status'] = 'rejected_long_gap'
        return None, qc
    if qc['nan_fraction'] > cfg['max_nan_fraction']:
        qc['status'] = 'rejected_nan_fraction'
        return None, qc

    x = _fill_gaps(x)

    if cfg['angles_in_degrees'] and 'Angle' in base_col:
        x = np.deg2rad(x)

    # --- high pass on position channels, on the full record ------------------
    if kind == 'pos' and detrend == 'highpass':
        b, a = butter(cfg['highpass_order'],
                      cfg['highpass_hz'] / (cfg['fs'] / 2.0), btype='high')
        x = filtfilt(b, a, x)
    elif kind != 'pos' and detrend in ('highpass', 'linear'):
        qc['note'] = 'detrend requested on a non-position channel, ignored'

    # --- ONE smoothing / differentiation stage [Savitzky1964] ----------------
    order = {'pos': 0, 'vel': 1, 'acc': 2}[kind]
    x = savgol_filter(x, cfg['savgol_window'], cfg['savgol_poly'],
                      deriv=order, delta=1.0 / cfg['fs'])

    x = x[i0: i1 + 1]

    if kind == 'pos' and detrend == 'linear':
        t = np.arange(len(x))
        x = x - np.polyval(np.polyfit(t, x, 1), t)

    qc['status'] = 'ok'
    qc['sd'] = float(np.std(x))
    return x, qc


def _time_normalise(x, hs_rows, n_out):
    """
    Resample every stride to n_out samples [Dingwell2000, Dingwell2006].

    Each stride is resampled over the half-open interval [HS_k, HS_k+1), so
    consecutive strides join without the one-sample phase gap that a linspace
    over [0, len-1] leaves behind. Returns None if any stride is too short to
    resample: v3 silently dropped such strides, which puts a discontinuity in
    the concatenated series and quietly changes the stride count.
    """
    pieces = []
    for k in range(len(hs_rows) - 1):
        seg = x[hs_rows[k]: hs_rows[k + 1] + 1]   # inclusive of the next HS
        if len(seg) < 5:
            return None
        grid = np.linspace(0.0, len(seg) - 1.0, n_out, endpoint=False)
        pieces.append(np.interp(grid, np.arange(len(seg)), seg))
    return np.concatenate(pieces)


# -----------------------------------------------------------------------------
# %% helpers: state space construction
# -----------------------------------------------------------------------------

def _scale_channels(chans, specs, scaling, cfg, stride_time_s, notes):
    """
    Apply channel scaling.

    lambda is invariant to a UNIFORM rescale of the whole state space, because
    that only shifts the intercept of <ln d(i)>. Per-channel scaling is not
    uniform: it changes which points are nearest neighbours and therefore
    changes the estimate. Any per-channel factor must therefore be identical
    across every trial being compared, which is why the reference SD and the
    reference stride time are configuration items rather than trial statistics.
    """
    out = [c.copy() for c in chans]
    kinds = [s[2] for s in specs]

    if scaling == 'none':
        if len(set(kinds)) > 1:
            notes.append('mixed derivative orders with scaling="none": '
                         'Euclidean distance mixes units')
        return out

    if scaling == 'zscore':
        ref = cfg.get('zscore_reference_sd')
        for i, spec in enumerate(specs):
            if ref is not None and spec in ref:
                sd = float(ref[spec])
            else:
                sd = float(np.std(out[i]))
                notes.append(f'zscore fell back to this trial\'s own SD for '
                             f'{spec}: conditions are not on a common geometry')
            if sd > 0:
                out[i] = out[i] / sd
        return out

    if scaling == 'nondim':
        # Hof (1996) style scaling, but with ONE length scale and ONE time
        # scale for every channel. v3 used stride time for angular channels and
        # sqrt(L/g) for linear channels, so a state space holding both had two
        # different implied time scales and an inconsistent relative weighting.
        leg = cfg.get('leg_length_m')
        if leg is None or not np.isfinite(leg) or leg <= 0:
            raise ValueError(
                "LDS_CONFIG['leg_length_m'] must be set to this participant's "
                "leg length before any state space using scaling='nondim' can "
                "be computed.")
        T = cfg.get('nondim_reference_stride_time_s')
        if T is None:
            T = stride_time_s
            notes.append('nondim used this trial\'s own stride time as the '
                         'time scale: set nondim_reference_stride_time_s to a '
                         'fixed reference for between-condition comparisons')
        for i, spec in enumerate(specs):
            order = {'pos': 0, 'vel': 1, 'acc': 2}[spec[2]]
            factor = T ** order if 'Angle' in spec[0] else T ** order / leg
            out[i] = out[i] * factor
        return out

    raise ValueError(f"unknown scaling '{scaling}'")


def _build_state_matrix(chans, embed, dE, tau):
    """Delay embedding [Takens1981] with ONE common tau for every channel, or
    the measured states used directly."""
    if embed != 'delay':
        return np.column_stack(chans)
    span = (dE - 1) * tau
    M = len(chans[0]) - span
    if M < 500:
        return None
    cols = [c[k * tau: k * tau + M] for c in chans for k in range(dE)]
    return np.column_stack(cols)


# -----------------------------------------------------------------------------
# %% helpers: neighbours and the divergence curve
# -----------------------------------------------------------------------------

def _query(tree, pts, k):
    try:
        return tree.query(pts, k=k, workers=-1)
    except TypeError:                     # older scipy
        return tree.query(pts, k=k)


def _nearest_neighbours(Y, theiler, last_valid, k_init, k_max):
    """
    Nearest neighbour of every usable reference point, excluding the Theiler
    window [Theiler1986].

    Two constraints, both needed for a fixed pair set:
      |j - i| > theiler     the neighbour must come from a different stride
      j <= last_valid       the neighbour must survive the whole curve

    k is grown until every reference point resolves, instead of v3's fixed
    k = 200 which failed silently and biased the reference set toward regions
    where the trajectory happens to be locally sparse.
    """
    M = Y.shape[0]
    tree = cKDTree(Y)
    nn = np.full(M, -1, dtype=np.int64)
    d0 = np.full(M, np.nan)

    pending = np.arange(min(last_valid + 1, M))
    k = int(min(M, max(2, k_init)))
    k_used = k

    while len(pending):
        dist, idx = _query(tree, Y[pending], k=k)
        resolved = np.zeros(len(pending), dtype=bool)
        for col in range(1, k):
            cand = idx[:, col]
            good = (~resolved) & (np.abs(cand - pending) > theiler) & (cand <= last_valid)
            nn[pending[good]] = cand[good]
            d0[pending[good]] = dist[good, col]
            resolved |= good
        pending = pending[~resolved]
        k_used = k
        if not len(pending) or k >= min(M, k_max):
            break
        k = int(min(M, k_max, k * 2))

    return nn, d0, int(len(pending)), int(k_used)


def _divergence(Y, theiler, n_lags, cfg, rng):
    """
    Rosenstein mean log divergence curve [Rosenstein1993], computed over a
    SINGLE FIXED SET of neighbour pairs.

    This is the main correction to v3. There the pair set shrank as the lag
    grew (pairs running off the end of the trajectory were dropped) and zero
    distances were removed lag by lag, so <ln d(i)> was an average over a
    different ensemble at every i. Reference points near the end of the trial
    dropped out first, which meant the long-lag part of the curve - exactly
    where lambda_L is fit - came from a systematically earlier subset of the
    trial than the short-lag part.

    Returns the curve plus the per-pair log distance matrix, which lets the
    bootstrap resample reference points without recomputing any distances.
    """
    M = Y.shape[0]
    last_valid = M - 1 - n_lags
    if last_valid < 200:
        return None

    nn, d0, n_unresolved, k_used = _nearest_neighbours(
        Y, theiler, last_valid, cfg['nn_k_initial'], cfg['nn_k_max'])

    ref = np.flatnonzero(nn >= 0)
    info = {
        'n_points': int(M),
        'theiler_samples': int(theiler),
        'n_lags': int(n_lags),
        'nn_k_used': k_used,
        'frac_refs_unresolved': float(1.0 - len(ref) / max(1, last_valid + 1)),
    }

    # A zero initial separation means two identical state vectors. In markerless
    # data that is usually a frozen or gap-filled section, not a recurrence.
    info['n_zero_initial_distance'] = int(np.sum(d0[ref] <= 0))
    ref = ref[d0[ref] > 0]

    # Reference points whose nearest valid neighbour is far away do not measure
    # LOCAL divergence; their separation reflects global attractor geometry and
    # tends to wander or contract rather than grow.
    info['d0_threshold'] = np.nan
    info['n_refs_dropped_far_d0'] = 0
    if cfg['max_d0_percentile'] is not None and len(ref):
        thr = float(np.percentile(d0[ref], cfg['max_d0_percentile']))
        keep = d0[ref] <= thr
        info['d0_threshold'] = thr
        info['n_refs_dropped_far_d0'] = int((~keep).sum())
        ref = ref[keep]

    if len(ref) < 200:
        return None

    if cfg['max_ref_points'] is not None and len(ref) > cfg['max_ref_points']:
        ref = np.sort(rng.choice(ref, int(cfg['max_ref_points']), replace=False))

    a, b = ref, nn[ref]
    tiny = float(np.finfo(float).tiny)
    L = np.empty((len(ref), n_lags + 1), dtype=np.float32)
    n_zero_any = 0
    for i in range(n_lags + 1):
        d = np.linalg.norm(Y[a + i] - Y[b + i], axis=1)
        n_zero_any += int(np.sum(d <= 0))
        L[:, i] = np.log(np.maximum(d, tiny))

    curve = L.mean(axis=0).astype(float)

    # Saturation level of the curve: the mean log distance between unrelated
    # points on the attractor. Reported because a fixed fit window couples slope
    # to intercept - a trial whose neighbours start further apart begins closer
    # to saturation and will show a shallower slope for reasons that have
    # nothing to do with stability.
    n_pair = int(min(20000, 4 * M))
    ii = rng.integers(0, M, n_pair)
    jj = rng.integers(0, M, n_pair)
    far = np.abs(ii - jj) > theiler
    ln_attractor = float(np.mean(np.log(np.maximum(
        np.linalg.norm(Y[ii[far]] - Y[jj[far]], axis=1), tiny))))

    plateau_strides_samples = np.nan
    if ln_attractor > curve[0]:
        thr = curve[0] + 0.95 * (ln_attractor - curve[0])
        hit = np.flatnonzero(curve >= thr)
        if hit.size:
            plateau_strides_samples = float(hit[0])

    info.update({
        'n_ref_points': int(len(ref)),
        'n_zero_distance_events': int(n_zero_any),
        'mean_ln_d0': float(curve[0]),
        'median_d0': float(np.median(d0[ref])),
        'ln_attractor_size': ln_attractor,
        'plateau_onset_samples': plateau_strides_samples,
    })
    return {'curve': curve, 'L': L, 'info': info}


# -----------------------------------------------------------------------------
# %% helpers: fitting the divergence curve
# -----------------------------------------------------------------------------

def _fit_design(scale, window, n_lags):
    """Least squares design for a slope fit over `window`, expressed in units
    of which there are `scale` samples (strides for gait, time units for the
    self test). The pseudo-inverse is reused by the bootstrap."""
    j0 = int(round(window[0] * scale))
    j1 = int(min(round(window[1] * scale), n_lags))
    if j1 - j0 < 4:
        return None
    x = np.arange(j0, j1 + 1) / scale
    A = np.column_stack([x, np.ones_like(x)])
    return {'j0': j0, 'j1': j1, 'x': x, 'pinv': np.linalg.pinv(A)}


def _fit_curve(curve, design):
    """Slope, R2 and shape QC over one fit window.

    polyfit always returns a slope whether or not the curve has a linear
    scaling region, so the shape diagnostics travel with the number: R2, the
    fraction of the window over which the curve actually increases, and the
    ratio of the slope in the first half of the window to the second.
    """
    seg = curve[design['j0']: design['j1'] + 1]
    x = design['x']
    coef = design['pinv'] @ seg
    pred = coef[0] * x + coef[1]
    ss_res = float(np.sum((seg - pred) ** 2))
    ss_tot = float(np.sum((seg - seg.mean()) ** 2))

    half = len(seg) // 2
    ratio = np.nan
    if half >= 3:
        s1 = np.polyfit(x[:half], seg[:half], 1)[0]
        s2 = np.polyfit(x[half:], seg[half:], 1)[0]
        if abs(s2) > 1e-12:
            ratio = float(s1 / s2)

    return {
        'slope': float(coef[0]),
        'r2': float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        'frac_increasing': float(np.mean(np.diff(seg) > 0)),
        'slope_ratio_halves': ratio,
        'fit_lo_units': float(design['x'][0]),
        'fit_hi_units': float(design['x'][-1]),
        'n_fit_points': int(len(seg)),
    }


def _bootstrap_curves(L, n_boot, rng):
    """Bootstrap the whole divergence curve by resampling REFERENCE POINTS.

    v3 reported a bare point estimate with no uncertainty at all, so there was
    no way to tell whether a between-condition difference exceeded within-trial
    noise. Implemented as multinomial weights times the stored log distance
    matrix, which is algebraically identical to resampling rows but avoids
    copying a 24 MB array n_boot times.
    """
    if not n_boot:
        return None
    n = L.shape[0]
    counts = rng.multinomial(n, np.full(n, 1.0 / n),
                             size=int(n_boot)).astype(np.float32)
    return ((counts @ L) / n).astype(float)


def _bootstrap_slopes(boot_curves, design):
    """Slopes of the bootstrapped curves over one fit window."""
    if boot_curves is None:
        return np.empty(0)
    seg = boot_curves[:, design['j0']: design['j1'] + 1]
    return seg @ design['pinv'][0]


# -----------------------------------------------------------------------------
# %% helpers: embedding parameter diagnostics
# -----------------------------------------------------------------------------
# These exist so this trial can be checked against the hard-coded tau and dE,
# and so the SEPARATE dataset-level script can import them and use exactly the
# same estimators. They must never be used to set tau or dE for one trial.

def lds_average_mutual_information(x, max_lag, n_bins=32, equiprobable=True):
    """AMI as a function of lag [Fraser1986].

    Equiprobable (quantile) binning by default. v3 used equal-width bins, which
    on a skewed channel puts most of the data in a handful of bins and makes
    the AMI curve noisy enough to produce a spurious early minimum.
    """
    x = np.asarray(x, float)
    if equiprobable:
        edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
    else:
        edges = np.histogram_bin_edges(x, bins=n_bins)
    ami = np.zeros(max_lag + 1)
    for t in range(max_lag + 1):
        H, _, _ = np.histogram2d(x[: len(x) - t], x[t:], bins=[edges, edges])
        P = H / H.sum()
        Px, Py = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        ami[t] = float(np.sum(P[nz] * np.log2(P[nz] / (Px @ Py)[nz])))
    return ami


def lds_first_ami_minimum(ami, min_lag=1, smooth=3):
    """First local minimum of the AMI curve [Fraser1986].

    The comparison on the left is strict. v3 used `<=`, so a flat stretch of
    the curve triggered the minimum and returned a lag that was too short.
    """
    a = np.convolve(ami, np.ones(smooth) / smooth, mode='same') if smooth > 1 else ami
    for t in range(max(1, min_lag), len(a) - 1):
        if a[t] < a[t - 1] and a[t] <= a[t + 1]:
            return int(t)
    return int(np.argmin(a[min_lag:]) + min_lag)


def lds_false_nearest_neighbours(x, tau, max_dim, rtol=15.0, atol=2.0,
                                 theiler=0, k_init=50, k_max=500):
    """Global false nearest neighbours [Kennel1992], WITH a Theiler exclusion.

    v3 took the plain nearest neighbour. At 100 Hz the trajectory is heavily
    oversampled relative to the stride, so that neighbour is almost always the
    temporally adjacent sample, whose extra coordinate differs by a negligible
    amount by construction. The false-neighbour fraction is then driven toward
    zero at every dimension and the diagnostic carries no information.
    """
    x = np.asarray(x, float)
    sd = float(np.std(x))
    frac = np.full(max_dim, np.nan)
    for d in range(1, max_dim + 1):
        M = len(x) - d * tau
        if M < 500:
            continue
        Y = np.column_stack([x[k * tau: k * tau + M] for k in range(d)])
        nn, r, _, _ = _nearest_neighbours(Y, theiler, M - 1, k_init, k_max)
        ok = (nn >= 0) & np.isfinite(r) & (r > 0)
        if ok.sum() < 100:
            continue
        i = np.flatnonzero(ok)
        extra = np.abs(x[i + d * tau] - x[nn[i] + d * tau])
        c1 = (extra / r[i]) > rtol
        c2 = (np.sqrt(r[i] ** 2 + extra ** 2) / sd) > atol
        frac[d - 1] = float(np.mean(c1 | c2))
    return frac


def lds_fnn_knee(frac, threshold=0.10, delta=0.02):
    """First dimension where the false neighbour fraction drops below
    `threshold` and stops changing by more than `delta`."""
    for k in range(len(frac) - 1):
        if np.isfinite(frac[k]) and frac[k] < threshold and \
           np.isfinite(frac[k + 1]) and abs(frac[k + 1] - frac[k]) < delta:
            return int(k + 1)
    finite = np.flatnonzero(np.isfinite(frac))
    return int(finite[-1] + 1) if finite.size else np.nan


# -----------------------------------------------------------------------------
# %% helpers: surrogate data
# -----------------------------------------------------------------------------

def _surrogate_stride_shuffle(chans, n_strides, samples_per_stride, rng):
    """Shuffle the order of whole strides.

    Preserves the intra-stride waveform exactly and destroys all inter-stride
    structure, so it is the natural null for gait: it asks whether lambda_L
    reflects anything beyond stride-to-stride independence [Small2001]. Needs
    equal-length strides, i.e. time normalisation.
    """
    sps = int(samples_per_stride)
    perm = rng.permutation(n_strides)
    return [c[: n_strides * sps].reshape(n_strides, sps)[perm].ravel()
            for c in chans]


def _surrogate_phase_randomise(chans, rng):
    """Fourier phase randomisation [Theiler1992]: preserves the power spectrum
    and therefore the linear autocorrelation, destroys nonlinear structure.
    Note it also Gaussianises the waveform, which for gait is a strong change."""
    out = []
    for c in chans:
        n = len(c)
        F = np.fft.rfft(c)
        ph = rng.uniform(0.0, 2.0 * np.pi, len(F))
        ph[0] = 0.0
        if n % 2 == 0:
            ph[-1] = 0.0
        out.append(np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=n))
    return out


# -----------------------------------------------------------------------------
# %% helpers: implementation self test
# -----------------------------------------------------------------------------

def _lorenz_x(n, dt=0.01, transient=5000, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """RK4 integration of the classic Lorenz system."""
    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    s = np.array([1.0, 1.0, 1.0])
    out = np.empty(n)
    for i in range(transient + n):
        k1 = f(s)
        k2 = f(s + dt * k1 / 2.0)
        k3 = f(s + dt * k2 / 2.0)
        k4 = f(s + dt * k3)
        s = s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if i >= transient:
            out[i - transient] = s[0]
    return out


def _slope_from_series(x, tau, dE, theiler, n_lags, scale, window, cfg, rng):
    """Run the full Rosenstein pipeline on one univariate series and return the
    slope over `window`, expressed per unit of `scale` samples."""
    Y = _build_state_matrix([x], 'delay', dE, tau)
    if Y is None:
        return np.nan, None
    div = _divergence(Y, theiler, n_lags, cfg, rng)
    if div is None:
        return np.nan, None
    design = _fit_design(scale, window, n_lags)
    if design is None:
        return np.nan, div['curve']
    return _fit_curve(div['curve'], design)['slope'], div['curve']


def lds_self_test(cfg):
    """
    Validate the implementation against systems whose exponent is known, and
    measure the noise floor of this exact pipeline.

    1. Lorenz (sigma=10, rho=28, beta=8/3), reference lambda_1 = 0.9056
       [Sprott2003]. NOTE the fit runs over 1-3 time units, not from zero. The
       Rosenstein curve has a steep initial transient before it reaches the
       true scaling region, because the nearest neighbour of each reference
       point is the MINIMUM over many candidates and is therefore anomalously
       close; the pair separation regresses toward the typical separation
       before genuine exponential divergence takes over. Fitting from t = 0
       returns roughly 1.7 on this system, about double the true value.
       Recovering ~0.85-0.91 from the post-transient region is the evidence
       that the algorithm itself is correct.

    2. A clean periodic signal has lambda = 0. Running the pipeline on a
       periodic signal with the same embedding, stride count and fit window as
       the gait analysis gives the value this pipeline returns when there is no
       divergence at all. Its divergence curve is flat after roughly a quarter
       of a cycle, so whatever lambda_S it reports is entirely the
       neighbour-selection transient above. Compare it against your real
       lambda_S: it is the floor, and it is not small
       [Mehdizadeh2019, Raffalt2019].
    """
    rng = np.random.default_rng(cfg['random_seed'])
    test_cfg = dict(cfg)
    test_cfg['max_ref_points'] = 3000
    results = []

    # --- Lorenz, fit in the post-transient scaling region --------------------
    dt = 0.01
    x = _lorenz_x(20000, dt=dt)
    lam, _ = _slope_from_series(x, tau=11, dE=5, theiler=75, n_lags=400,
                                scale=1.0 / dt, window=(1.0, 3.0),
                                cfg=test_cfg, rng=rng)
    ref = 0.9056
    ok = bool(np.isfinite(lam) and abs(lam - ref) <= 0.15 * ref)
    results.append({'system': 'Lorenz x (reference lambda_1 = 0.9056)',
                    'fit_window': '1-3 time units',
                    'estimate': lam, 'reference': ref,
                    'units': 'per time unit',
                    'verdict': 'PASS' if ok else 'CHECK PIPELINE'})

    # --- periodic signals at the gait settings -------------------------------
    sps = float(cfg['samples_per_stride_norm'] if cfg['time_normalize'] else 100)
    n_str = int(cfg['n_strides'])
    theiler = int(round(cfg['theiler_strides'] * sps))
    n_lags = int(round(cfg['n_strides_curve'] * sps))
    phase = np.arange(n_str * int(sps)) / sps
    clean = np.sin(2.0 * np.pi * phase) + 0.3 * np.sin(4.0 * np.pi * phase)
    sw = tuple(cfg['short_window'])
    for noise in (0.001, 0.01, 0.05):
        y = clean + rng.normal(0.0, noise * np.std(clean), len(clean))
        lam_s, curve = _slope_from_series(
            y, tau=int(cfg['tau']), dE=int(cfg['dE']), theiler=theiler,
            n_lags=n_lags, scale=sps, window=sw, cfg=test_cfg, rng=rng)
        # slope well past the transient: for a periodic signal this must be ~0
        late = np.nan
        if curve is not None:
            d_late = _fit_design(sps, (1.0, cfg['n_strides_curve']), n_lags)
            if d_late is not None:
                late = _fit_curve(curve, d_late)['slope']
        results.append({'system': f'periodic + {noise*100:g}% noise '
                                  f'(true lambda = 0)',
                        'fit_window': f'{sw[0]:g}-{sw[1]:g} strides',
                        'estimate': lam_s, 'reference': 0.0,
                        'units': 'per stride',
                        'verdict': f'noise floor; post-transient slope '
                                   f'{late:+.4f}'})

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# %% main entry point
# -----------------------------------------------------------------------------

def compute_lds(kinematic_data, gait_event_data, cfg, state_spaces, trial_id):
    """
    Local dynamic stability for one trial.

    Written as a function on purpose. In v3 the whole analysis ran at module
    level and the state space dictionary was OVERWRITTEN in place by the channel
    availability filter, so in a loop over trials any state space dropped for
    trial 1 was gone permanently and every later trial silently analysed a
    smaller set. Nothing here mutates its arguments.

    Returns (results DataFrame, {state space: divergence curve}, qc dict).
    """
    cfg = dict(cfg)
    rng = np.random.default_rng(cfg['random_seed'])
    qc = {'trial': trial_id, 'warnings': [], 'channels': []}

    # --- the frame index must be trustworthy before anything else -----------
    frame_col = pd.to_numeric(kinematic_data['Unnamed: 0'], errors='coerce').to_numpy(float)
    if not np.all(np.isfinite(frame_col)):
        raise ValueError('kinematic frame column contains non-numeric values')
    if pd.Series(frame_col).duplicated().any():
        raise ValueError('kinematic frame column contains duplicates, so the '
                         'frame -> row mapping is ambiguous')
    if np.any(np.diff(frame_col) <= 0):
        raise ValueError('kinematic frame column is not strictly increasing')
    frame_position = pd.Series(np.arange(len(kinematic_data)), index=frame_col)

    # --- choose the analysis window -----------------------------------------
    hs_all = (gait_event_data.loc[gait_event_data['support_limb'] == cfg['limb'],
                                  'initial_contact_kinematic_frame_100hz']
              .dropna().astype(int).sort_values().to_numpy())
    if len(hs_all) < 3:
        raise ValueError(f'not enough heel strikes on limb {cfg["limb"]}')

    # Discard the acclimatisation period: treadmill gait is not steady state at
    # the start of a trial, and v3 began at stride 0.
    if cfg['start_stride'] is None:
        cutoff = hs_all[0] + cfg['warmup_seconds'] * cfg['fs']
        start = int(np.searchsorted(hs_all, cutoff))
    else:
        start = int(cfg['start_stride'])

    n_strides = int(cfg['n_strides'])
    qc['n_strides_requested'] = n_strides
    qc['stride_count_short'] = False
    if len(hs_all) < start + n_strides + 1:
        n_strides = len(hs_all) - start - 1
        qc['stride_count_short'] = True
        qc['warnings'].append(
            f'only {n_strides} strides available after the warm-up, not '
            f'{cfg["n_strides"]}. lambda depends on the amount of data, so this '
            f'trial is NOT comparable to trials analysed at the full stride '
            f'count [Bruijn2009a].')
    if n_strides < 20:
        raise ValueError(f'only {n_strides} usable strides on limb {cfg["limb"]}')

    hs = hs_all[start: start + n_strides + 1]
    if hs[0] not in frame_position.index or hs[-1] not in frame_position.index:
        raise ValueError('LDS window frames are not present in the kinematic data')

    i0 = int(frame_position[hs[0]])
    i1 = int(frame_position[hs[-1]])
    if np.any(np.diff(frame_col[i0:i1 + 1]) != 1):
        raise ValueError('kinematic frames are not contiguous inside the LDS '
                         'window, so sample counts and lags would be wrong')

    stride_samples = np.diff(hs)
    stride_time_s = float(np.mean(stride_samples) / cfg['fs'])
    stride_time_cv = float(np.std(stride_samples, ddof=1) / np.mean(stride_samples) * 100.0)

    qc.update({
        'limb': cfg['limb'],
        'start_stride': start,
        'n_strides_used': n_strides,
        'first_frame': int(hs[0]), 'last_frame': int(hs[-1]),
        'n_samples_raw': i1 - i0 + 1,
        'mean_stride_time_s': stride_time_s,
        'stride_time_cv_pct': stride_time_cv,
    })

    # --- condition every channel any state space asks for --------------------
    # The detrend mode is part of the cache key, because the same channel can be
    # wanted both with and without a high pass by different state spaces.
    needed = []
    for ss in state_spaces.values():
        det = ss.get('detrend')
        for spec in ss['signals']:
            key = (spec, det if spec[2] == 'pos' else None)
            if key not in needed:
                needed.append(key)
    if cfg['run_parameter_check']:
        pkey = (cfg['param_check_signal'], None)
        if pkey not in needed:
            needed.append(pkey)

    signals = {}
    for spec, det in needed:
        x, cqc = _prepare_signal(kinematic_data, spec, det, cfg, i0, i1)
        qc['channels'].append(cqc)
        if x is None:
            qc['warnings'].append(f'channel {cqc["channel"]} '
                                  f'({cqc["status"]}): state spaces using it '
                                  f'will be skipped')
        else:
            signals[(spec, det)] = x

    qc['savgol_cutoff_hz'] = _savgol_cutoff_hz(
        cfg['savgol_window'], cfg['savgol_poly'], cfg['fs'])

    # --- stride time normalisation -------------------------------------------
    hs_rows = np.array([int(frame_position[f]) - i0 for f in hs])
    if cfg['time_normalize']:
        n_out = int(cfg['samples_per_stride_norm'])
        for key in list(signals):
            z = _time_normalise(signals[key], hs_rows, n_out)
            if z is None:
                qc['warnings'].append(
                    f'stride too short to resample for {key[0]}, channel dropped')
                signals.pop(key)
            else:
                signals[key] = z
        samples_per_stride = float(n_out)
        stride_bounds = np.arange(n_strides + 1) * n_out
    else:
        samples_per_stride = float(np.mean(np.diff(hs_rows)))
        stride_bounds = hs_rows - hs_rows[0]
        qc['warnings'].append(
            'time_normalize is off: samples per stride varies with cadence, '
            'and the lag-to-stride mapping smears in proportion to stride time '
            'variability [Dingwell2000, Stenum2014].')

    qc['samples_per_stride'] = samples_per_stride
    theiler = int(round(cfg['theiler_strides'] * samples_per_stride))
    n_lags = int(round(cfg['n_strides_curve'] * samples_per_stride))

    primary_name = next((k for k, v in state_spaces.items()
                         if v.get('primary')), None)

    # --- provenance carried on every row -------------------------------------
    provenance = {
        'trial': trial_id,
        'analysis_datetime': pd.Timestamp.now().isoformat(timespec='seconds'),
        'python': platform.python_version(),
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'pandas': pd.__version__,
        'limb': cfg['limb'], 'start_stride': start,
        'n_strides': n_strides,
        'stride_count_short': qc['stride_count_short'],
        'first_frame': int(hs[0]), 'last_frame': int(hs[-1]),
        'mean_stride_time_s': stride_time_s,
        'stride_time_cv_pct': stride_time_cv,
        'time_normalized': cfg['time_normalize'],
        'samples_per_stride': samples_per_stride,
        'tau_samples': int(cfg['tau']),
        'theiler_samples': theiler,
        'savgol_window': cfg['savgol_window'],
        'savgol_poly': cfg['savgol_poly'],
        'savgol_cutoff_hz': qc['savgol_cutoff_hz'],
        'highpass_hz': cfg['highpass_hz'],
        'leg_length_m': cfg['leg_length_m'],
        'random_seed': cfg['random_seed'],
        'config_json': json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in cfg.items()
                                   if k != 'zscore_reference_sd'}, default=str),
    }

    fit_specs = [('S', tuple(cfg['short_window'])),
                 ('L', tuple(cfg['long_window']))]
    for n, w in enumerate(cfg.get('short_window_sensitivity') or [], start=1):
        fit_specs.append((f'S_alt{n}', tuple(w)))

    results, curves = [], {}

    for name, ss in state_spaces.items():
        det = ss.get('detrend')
        keys = [(spec, det if spec[2] == 'pos' else None) for spec in ss['signals']]
        if not all(k in signals for k in keys):
            qc['warnings'].append(f'{name}: skipped, a channel was unavailable')
            continue

        notes = []
        chans = _scale_channels([signals[k] for k in keys], ss['signals'],
                                ss.get('scaling', 'none'), cfg, stride_time_s, notes)
        dE = int(ss.get('dE') or cfg['dE'])
        tau = int(cfg['tau'])

        Y = _build_state_matrix(chans, ss['embed'], dE, tau)
        if Y is None:
            qc['warnings'].append(f'{name}: window too short for tau/dE, skipped')
            continue
        div = _divergence(Y, theiler, n_lags, cfg, rng)
        if div is None:
            qc['warnings'].append(f'{name}: too few usable neighbour pairs, skipped')
            continue

        curve = div['curve']
        row = dict(provenance)
        row.update({
            'state_space': name,
            'is_primary': bool(ss.get('primary', False)),
            'signals': ' + '.join(f'{s[0]}_{s[1]}|{s[2]}' for s in ss['signals']),
            'embed': ss['embed'],
            'scaling': ss.get('scaling', 'none'),
            'detrend': str(det),
            'dE_per_channel': dE if ss['embed'] == 'delay' else np.nan,
            'total_dimension': int(Y.shape[1]),
            'qc_notes': '; '.join(notes),
        })
        row.update(div['info'])
        row['plateau_onset_strides'] = (div['info']['plateau_onset_samples']
                                        / samples_per_stride)
        # A plateau before the long-term window means the curve has already
        # saturated at the attractor diameter and lambda_L is fitting the
        # flattening, not a divergence rate.
        row['plateau_before_long_window'] = bool(
            np.isfinite(row['plateau_onset_strides']) and
            row['plateau_onset_strides'] < cfg['long_window'][0])

        boot_curves = (_bootstrap_curves(div['L'], cfg['n_bootstrap'], rng)
                       if cfg['n_bootstrap'] else None)

        for tag, win in fit_specs:
            design = _fit_design(samples_per_stride, win, n_lags)
            if design is None:
                row[f'lambda_{tag}_per_stride'] = np.nan
                continue
            f = _fit_curve(curve, design)
            row[f'lambda_{tag}_per_stride'] = f['slope']
            row[f'R2_{tag}'] = f['r2']
            row[f'frac_increasing_{tag}'] = f['frac_increasing']
            row[f'slope_ratio_halves_{tag}'] = f['slope_ratio_halves']
            row[f'fit_window_{tag}_strides'] = f'{f["fit_lo_units"]:.3f}-{f["fit_hi_units"]:.3f}'
            row[f'n_fit_points_{tag}'] = f['n_fit_points']

            if tag in ('S', 'L') and boot_curves is not None:
                b = _bootstrap_slopes(boot_curves, design)
                alpha = (100.0 - cfg['bootstrap_ci']) / 2.0
                row[f'lambda_{tag}_ci_lo'] = float(np.percentile(b, alpha))
                row[f'lambda_{tag}_ci_hi'] = float(np.percentile(b, 100 - alpha))
                row[f'lambda_{tag}_boot_sd'] = float(np.std(b, ddof=1))

        # Per second is only meaningful on real-time data. Converting a
        # per-stride exponent from time-normalised data back to per second
        # simply reintroduces the cadence dependence that normalisation removed.
        if cfg['time_normalize']:
            row['lambda_S_per_s'] = np.nan
            row['lambda_L_per_s'] = np.nan
            row['per_second_valid'] = False
        else:
            row['lambda_S_per_s'] = row['lambda_S_per_stride'] / stride_time_s
            row['lambda_L_per_s'] = row['lambda_L_per_stride'] / stride_time_s
            row['per_second_valid'] = True

        # --- split half over strides -----------------------------------------
        # Precision of lambda_S within this trial. NOTE the halves each use half
        # the strides, and lambda is a function of the amount of data, so these
        # two values should be compared to EACH OTHER, never to the full series
        # value [Bruijn2009a, Kang2006].
        if cfg['run_split_half'] and n_strides >= 40:
            h = n_strides // 2
            half_lams = []
            for lo, hi in [(0, h), (h, n_strides)]:
                sub = [c[stride_bounds[lo]: stride_bounds[hi]] for c in chans]
                Ys = _build_state_matrix(sub, ss['embed'], dE, tau)
                ds = _divergence(Ys, theiler, n_lags, cfg, rng) if Ys is not None else None
                if ds is None:
                    half_lams.append(np.nan)
                    continue
                dsg = _fit_design(samples_per_stride, tuple(cfg['short_window']), n_lags)
                half_lams.append(_fit_curve(ds['curve'], dsg)['slope']
                                 if dsg is not None else np.nan)
            row['lambda_S_half1'] = half_lams[0]
            row['lambda_S_half2'] = half_lams[1]
            row['lambda_S_half_abs_diff'] = float(abs(half_lams[0] - half_lams[1]))

        # --- surrogate data test ---------------------------------------------
        if cfg['n_surrogates'] and (primary_name is None or name == primary_name):
            sur = []
            design = _fit_design(samples_per_stride, tuple(cfg['short_window']), n_lags)
            for _ in range(int(cfg['n_surrogates'])):
                if cfg['surrogate_kind'] == 'stride_shuffle' and cfg['time_normalize']:
                    sc = _surrogate_stride_shuffle(chans, n_strides,
                                                   samples_per_stride, rng)
                else:
                    sc = _surrogate_phase_randomise(chans, rng)
                Ysu = _build_state_matrix(sc, ss['embed'], dE, tau)
                dsu = _divergence(Ysu, theiler, n_lags, cfg, rng) if Ysu is not None else None
                if dsu is not None and design is not None:
                    sur.append(_fit_curve(dsu['curve'], design)['slope'])
            if sur:
                sur = np.asarray(sur, float)
                row['lambda_S_surrogate_mean'] = float(sur.mean())
                row['lambda_S_surrogate_sd'] = float(sur.std(ddof=1))
                row['lambda_S_surrogate_kind'] = cfg['surrogate_kind']
                row['lambda_S_surrogate_z'] = float(
                    (row['lambda_S_per_stride'] - sur.mean()) /
                    sur.std(ddof=1)) if sur.std(ddof=1) > 0 else np.nan

        results.append(row)
        band_lo = band_hi = None
        if boot_curves is not None:
            alpha = (100.0 - cfg['bootstrap_ci']) / 2.0
            band_lo = np.percentile(boot_curves, alpha, axis=0)
            band_hi = np.percentile(boot_curves, 100 - alpha, axis=0)
        curves[name] = {'curve': curve, 'ci_lo': band_lo, 'ci_hi': band_hi,
                        'ln_attractor_size': div['info']['ln_attractor_size'],
                        'samples_per_stride': samples_per_stride}
        del div, boot_curves

    return pd.DataFrame(results), curves, qc


# -----------------------------------------------------------------------------
# %% run it
# -----------------------------------------------------------------------------

lds_trial_id = gait_event_path.stem.replace('_merged_events', '')

# Leg length is needed by every scaling='nondim' state space. Set it from the
# participant's anthropometrics rather than leaving the v3 hard-coded 0.90.
if LDS_CONFIG['leg_length_m'] is None:
    LDS_CONFIG['leg_length_m'] = 0.90
    print(" ! LDS_CONFIG['leg_length_m'] is a placeholder (0.90 m). Set it "
          "from this participant's anthropometrics before reporting anything "
          "from a scaling='nondim' state space.")

print('=' * 78)
print(' Local dynamic stability')
print(f' Trial : {lds_trial_id}')
print('=' * 78)

# --- 1. does the implementation recover known exponents? ---------------------
if LDS_CONFIG['run_selftest']:
    print('\n Implementation self test')
    lds_selftest = lds_self_test(LDS_CONFIG)
    print(lds_selftest.to_string(index=False))
    lorenz_row = lds_selftest.iloc[0]
    if not lorenz_row['within_25pct']:
        print(' ! the Lorenz estimate is more than 25% from the reference '
              'value, check the pipeline before trusting any gait result')
    print('  The periodic rows are the noise floor of this pipeline: a signal '
          'with no chaos\n  in it still returns these values for lambda_S '
          '[Mehdizadeh2019].')

# --- 2. the main analysis ----------------------------------------------------
lds_results, lds_curves, lds_qc = compute_lds(
    kinematic_data, gait_event_data, LDS_CONFIG, LDS_STATE_SPACES, lds_trial_id)

print(f"\n Limb          : {lds_qc['limb']}")
print(f" Strides       : {lds_qc['n_strides_used']} starting at stride "
      f"{lds_qc['start_stride']} (frames {lds_qc['first_frame']} to "
      f"{lds_qc['last_frame']})")
print(f" Mean stride   : {lds_qc['mean_stride_time_s']:.3f} s  "
      f"(CV {lds_qc['stride_time_cv_pct']:.2f}%)")
print(f" Time normalise: {LDS_CONFIG['time_normalize']}  "
      f"({lds_qc['samples_per_stride']:.1f} samples/stride)")
print(f" Embedding     : tau = {LDS_CONFIG['tau']} samples, dE = "
      f"{LDS_CONFIG['dE']}  (FIXED, not estimated from this trial)")
print(f" Conditioning  : Savitzky-Golay {LDS_CONFIG['savgol_window']}/"
      f"{LDS_CONFIG['savgol_poly']}, effective -3 dB at "
      f"{lds_qc['savgol_cutoff_hz']:.1f} Hz")

for lds_w in lds_qc['warnings']:
    print(f' ! {lds_w}')

# --- 3. is this trial consistent with the dataset-wide tau and dE? -----------
# Sanity check ONLY. It reports a mismatch, it never changes tau or dE: doing
# that per trial is what makes lambda non-comparable in the first place
# [Bruijn2009a, vanSchooten2013].
if LDS_CONFIG['run_parameter_check']:
    lds_frame_position = pd.Series(
        np.arange(len(kinematic_data)),
        index=pd.to_numeric(kinematic_data['Unnamed: 0'], errors='coerce').to_numpy())
    lds_chk_i0 = int(lds_frame_position[lds_qc['first_frame']])
    lds_chk_i1 = int(lds_frame_position[lds_qc['last_frame']])

    lds_px, _ = _prepare_signal(kinematic_data, LDS_CONFIG['param_check_signal'],
                                None, LDS_CONFIG, lds_chk_i0, lds_chk_i1)

    # The check has to run on exactly the representation the analysis uses,
    # otherwise the tau it reports is not on the same scale as the fixed value.
    if lds_px is not None and LDS_CONFIG['time_normalize']:
        lds_hs_chk = (gait_event_data.loc[
            gait_event_data['support_limb'] == LDS_CONFIG['limb'],
            'initial_contact_kinematic_frame_100hz'].dropna().astype(int)
            .sort_values().to_numpy())
        lds_hs_chk = lds_hs_chk[(lds_hs_chk >= lds_qc['first_frame']) &
                                (lds_hs_chk <= lds_qc['last_frame'])]
        lds_px = _time_normalise(lds_px, lds_hs_chk - lds_qc['first_frame'],
                                 LDS_CONFIG['samples_per_stride_norm'])

    if lds_px is None:
        print('\n Parameter check skipped: the check channel was unavailable')
    else:
        lds_ami = lds_average_mutual_information(
            lds_px, LDS_CONFIG['ami_max_lag'], LDS_CONFIG['ami_bins'])
        lds_tau_trial = lds_first_ami_minimum(lds_ami)
        lds_theiler_chk = int(round(LDS_CONFIG['theiler_strides'] *
                                    lds_qc['samples_per_stride']))
        lds_fnn = lds_false_nearest_neighbours(
            lds_px, int(LDS_CONFIG['tau']), LDS_CONFIG['fnn_max_dim'],
            LDS_CONFIG['fnn_rtol'], LDS_CONFIG['fnn_atol'],
            theiler=lds_theiler_chk)
        lds_de_trial = lds_fnn_knee(lds_fnn)

        print(f"\n Parameter check on {LDS_CONFIG['param_check_signal']}")
        print(f'   this trial AMI first minimum : tau = {lds_tau_trial} '
              f"(fixed value in use: {LDS_CONFIG['tau']})")
        print(f'   this trial FNN knee          : dE  = {lds_de_trial} '
              f"(fixed value in use: {LDS_CONFIG['dE']})")
        if abs(lds_tau_trial - LDS_CONFIG['tau']) > 0.5 * LDS_CONFIG['tau']:
            print('   ! this trial sits far from the dataset-wide tau, worth '
                  'a look at the AMI curve, but DO NOT change tau for this '
                  'trial alone')

        plt.figure(figsize=(9.5, 3.4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(len(lds_ami)), lds_ami, color='#2a78d6', linewidth=2)
        plt.axvline(LDS_CONFIG['tau'], color='#eb6834', ls='--', linewidth=2,
                    label=f"fixed tau = {LDS_CONFIG['tau']}")
        plt.axvline(lds_tau_trial, color='#4a3aa7', ls=':', linewidth=2,
                    label=f'this trial = {lds_tau_trial}')
        plt.xlabel('lag (samples)')
        plt.ylabel('AMI (bits)')
        plt.title('Average mutual information', fontsize=10)
        plt.legend(fontsize=8, frameon=False)
        plt.grid(alpha=0.15)
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(1, LDS_CONFIG['fnn_max_dim'] + 1), 100 * lds_fnn,
                 'o-', color='#2a78d6', linewidth=2, markersize=5)
        plt.axvline(LDS_CONFIG['dE'], color='#eb6834', ls='--', linewidth=2,
                    label=f"fixed dE = {LDS_CONFIG['dE']}")
        plt.axvline(lds_de_trial, color='#4a3aa7', ls=':', linewidth=2,
                    label=f'this trial = {lds_de_trial}')
        plt.xlabel('embedding dimension')
        plt.ylabel('false nearest neighbours (%)')
        plt.title('Global FNN (Theiler corrected)', fontsize=10)
        plt.legend(fontsize=8, frameon=False)
        plt.grid(alpha=0.15)
        plt.tight_layout()

# -----------------------------------------------------------------------------
# %% results table, output files and figures
# -----------------------------------------------------------------------------

if len(lds_results) == 0:
    raise RuntimeError(
        'No state space could be computed for this trial. See the warnings '
        'above and the QC log: usually a missing kinematic column or a data '
        'gap inside the analysis window.')

if not lds_results['is_primary'].any():
    print(' ! the PRIMARY state space was dropped for this trial. Do not '
          'substitute one of the sensitivity analyses for it; fix the input '
          'or exclude the trial.')

lds_show = ['state_space', 'is_primary', 'total_dimension',
            'lambda_S_per_stride', 'lambda_S_ci_lo', 'lambda_S_ci_hi', 'R2_S',
            'lambda_L_per_stride', 'R2_L', 'frac_refs_unresolved',
            'plateau_before_long_window']
lds_show = [c for c in lds_show if c in lds_results.columns]
print('\n' + lds_results[lds_show].to_string(index=False))

lds_primary = lds_results[lds_results['is_primary']] if len(lds_results) else lds_results
if len(lds_primary):
    lds_p = lds_primary.iloc[0]
    print(f"\n PRIMARY OUTCOME  {lds_p['state_space']}")
    print(f"   lambda_S = {lds_p['lambda_S_per_stride']:.4f} per stride "
          f"[{lds_p.get('lambda_S_ci_lo', np.nan):.4f}, "
          f"{lds_p.get('lambda_S_ci_hi', np.nan):.4f}] "
          f"({LDS_CONFIG['bootstrap_ci']:.0f}% bootstrap CI)")
    if 'lambda_S_half1' in lds_p:
        print(f"   split half: {lds_p['lambda_S_half1']:.4f} vs "
              f"{lds_p['lambda_S_half2']:.4f} "
              f"(|diff| = {lds_p['lambda_S_half_abs_diff']:.4f}); the halves "
              f"use half the strides each so compare them only to each other")
    print('   Every other state space in the table is a pre-declared '
          'sensitivity analysis.\n   Handle multiplicity explicitly if you '
          'report them [Bruijn2013].')

# QC flags worth acting on before this row goes into a group analysis
for _, lds_r in lds_results.iterrows():
    if np.isfinite(lds_r.get('R2_S', np.nan)) and lds_r['R2_S'] < 0.90:
        print(f"   ! {lds_r['state_space']}: R2 = {lds_r['R2_S']:.2f} on the "
              f"lambda_S fit, the curve may have no linear scaling region")
    if lds_r.get('frac_refs_unresolved', 0) > 0.01:
        print(f"   ! {lds_r['state_space']}: "
              f"{100*lds_r['frac_refs_unresolved']:.1f}% of reference points "
              f"found no neighbour outside the Theiler window")
    if lds_r.get('n_zero_distance_events', 0) > 0:
        print(f"   ! {lds_r['state_space']}: "
              f"{lds_r['n_zero_distance_events']} zero distances, check for "
              f"duplicated or frozen frames in the kinematic data")
    if lds_r.get('plateau_before_long_window', False):
        print(f"   ! {lds_r['state_space']}: the curve saturates before "
              f"{LDS_CONFIG['long_window'][0]:g} strides, lambda_L is fitting "
              f"the plateau rather than a divergence rate [Bruijn2013]")

lds_out_dir = base / 'LDS_outputs'
lds_out_dir.mkdir(parents=True, exist_ok=True)

lds_out_path = lds_out_dir / f'{lds_trial_id}_LDS.csv'
lds_results.to_csv(lds_out_path, index=False)
print(f'\n LDS results -> {lds_out_path}')

# The divergence curves are saved too, so the fit windows can be changed or the
# curve shape re-inspected later without re-running the whole analysis.
lds_curve_table = pd.DataFrame({
    'strides': np.arange(len(next(iter(lds_curves.values()))['curve']))
               / lds_qc['samples_per_stride']}) if lds_curves else pd.DataFrame()
for lds_name, lds_c in lds_curves.items():
    lds_curve_table[f'{lds_name}__mean_ln_d'] = lds_c['curve']
    if lds_c['ci_lo'] is not None:
        lds_curve_table[f'{lds_name}__ci_lo'] = lds_c['ci_lo']
        lds_curve_table[f'{lds_name}__ci_hi'] = lds_c['ci_hi']
lds_curve_path = lds_out_dir / f'{lds_trial_id}_LDS_divergence_curves.csv'
lds_curve_table.to_csv(lds_curve_path, index=False)
print(f' Divergence curves -> {lds_curve_path}')

lds_qc_path = lds_out_dir / f'{lds_trial_id}_LDS_qc.json'
with open(lds_qc_path, 'w') as lds_f:
    json.dump(lds_qc, lds_f, indent=2, default=str)
print(f' QC log -> {lds_qc_path}')

# --- divergence curve figures -------------------------------------------------
# One panel per state space. Blue is the measured curve with its bootstrap band,
# orange and violet are the two slope fits, grey marks the saturation level the
# curve is heading for. Line style differs per element so the panels stay
# readable without colour.
if lds_curves:
    lds_ncol = min(3, len(lds_curves))
    lds_nrow = int(np.ceil(len(lds_curves) / float(lds_ncol)))
    plt.figure(figsize=(4.6 * lds_ncol, 3.5 * lds_nrow))
    for lds_panel, (lds_name, lds_c) in enumerate(lds_curves.items(), start=1):
        lds_curve = lds_c['curve']
        lds_sps = lds_c['samples_per_stride']
        lds_ax = np.arange(len(lds_curve)) / lds_sps
        plt.subplot(lds_nrow, lds_ncol, lds_panel)

        if lds_c['ci_lo'] is not None:
            plt.fill_between(lds_ax, lds_c['ci_lo'], lds_c['ci_hi'],
                             color='#2a78d6', alpha=0.18, linewidth=0,
                             label=f"{LDS_CONFIG['bootstrap_ci']:.0f}% CI")
        plt.plot(lds_ax, lds_curve, color='#2a78d6', linewidth=2,
                 label='<ln d(i)>')
        plt.axhline(lds_c['ln_attractor_size'], color='#8a8a85', ls=':',
                    linewidth=1.5, label='attractor size')

        for lds_win, lds_col, lds_ls, lds_lab in [
                (tuple(LDS_CONFIG['short_window']), '#eb6834', '--', 'S'),
                (tuple(LDS_CONFIG['long_window']), '#4a3aa7', '-.', 'L')]:
            lds_m = ((lds_ax >= lds_win[0]) & (lds_ax <= lds_win[1]) &
                     np.isfinite(lds_curve))
            if lds_m.sum() > 5:
                lds_fit = np.polyfit(lds_ax[lds_m], lds_curve[lds_m], 1)
                plt.plot(lds_ax[lds_m], np.polyval(lds_fit, lds_ax[lds_m]),
                         ls=lds_ls, color=lds_col, linewidth=2,
                         label=f'lambda_{lds_lab} = {lds_fit[0]:.3f}/stride')

        plt.title(lds_name + ('  [PRIMARY]' if
                  LDS_STATE_SPACES.get(lds_name, {}).get('primary') else ''),
                  fontsize=9)
        plt.xlabel('time (strides)')
        plt.ylabel('<ln d(i)>')
        plt.legend(fontsize=7, loc='lower right', frameon=False)
        plt.grid(alpha=0.15)
    plt.tight_layout()
