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
# Maximum finite time Lyapunov exponent by the Rosenstein algorithm
# [Rosenstein1993] on a Takens delay embedding [Takens1981], following the gait
# convention set by Dingwell [Dingwell2000].
#
# The choices that matter, and why:
#   tau and dE are FIXED for the whole dataset, not fitted per trial. Per trial
#     parameters give each condition its own geometry and make lambda
#     non comparable [Bruijn2009a, vanSchooten2013].
#   The stride count is FIXED. lambda depends on the amount of data, so every
#     trial you compare has to use the same number of strides [Bruijn2009a].
#   Strides are resampled to a fixed length, so samples per stride does not
#     track cadence, which this protocol manipulates [Dingwell2000, Stenum2014].
#   One Savitzky Golay stage does the smoothing and the differentiation.
#     Filtering changes lambda, so the bandwidth has to be single valued
#     [Savitzky1964, Raffalt2020].
#   Neighbours inside one stride are excluded [Theiler1986].
#   The divergence curve uses ONE fixed set of neighbour pairs for its whole
#     length, so <ln d(i)> is not an average over a different ensemble at every
#     lag.
#
# How accurate is lambda? Validated in lds_validation.py against Lorenz and
# Rossler: the estimator is off by a system dependent 4-19% and no fit window
# fixes both at once. Treat lambda as an operational measure of short term
# divergence to compare between conditions analysed identically, not as a
# physical constant, and do not compare absolute values against papers using
# other settings [Bruijn2013, Raffalt2019].
#
# Two things to watch in the output:
#   The curve has a steep transient at the start, because each neighbour is
#     picked as a minimum and so starts unusually close. A periodic signal with
#     no divergence at all still returns lambda_S of about 0.3 per stride here.
#     lambda_S_post excludes the transient so you can see the difference.
#   headroom_* says where a fit window sits between the starting separation and
#     the attractor size. Near 0 means you are fitting the transient, and a flat
#     value means the curve has already saturated and lambda_L is fitting a
#     plateau rather than a divergence rate.
#
# References
#   [Takens1981]      Takens (1981) Lecture Notes in Mathematics 898, 366-381.
#   [Savitzky1964]    Savitzky & Golay (1964) Anal Chem 36(8), 1627-1639.
#   [Theiler1986]     Theiler (1986) Phys Rev A 34(3), 2427-2432.
#   [Rosenstein1993]  Rosenstein et al. (1993) Physica D 65(1-2), 117-134.
#   [Dingwell2000]    Dingwell & Cusumano (2000) Chaos 10(4), 848-863.
#   [Kang2006]        Kang & Dingwell (2006) Gait Posture 24(3), 386-390.
#   [Dingwell2006]    Dingwell & Marin (2006) J Biomech 39(3), 444-452.
#   [England2007]     England & Granata (2007) Gait Posture 25(2), 172-178.
#   [Gates2009]       Gates & Dingwell (2009) J Biomech 42(9), 1345-1349.
#   [Bruijn2009a]     Bruijn et al. (2009) J Neurosci Methods 178(2), 327-333.
#   [Sloot2011]       Sloot et al. (2011) Ann Biomed Eng 39(5), 1563-1569.
#   [Bruijn2013]      Bruijn et al. (2013) J R Soc Interface 10(83), 20120999.
#   [vanSchooten2013] van Schooten et al. (2013) J Biomech 46(1), 137-141.
#   [Stenum2014]      Stenum et al. (2014) J Biomech 47(15), 3776-3779.
#   [Raffalt2019]     Raffalt et al. (2019) Ann Biomed Eng 47(4), 913-923.
#   [Raffalt2020]     Raffalt et al. (2020) Comput Biol Med 122, 103786.
#   Verify page numbers against your reference manager before publishing.
# =============================================================================

from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

# -----------------------------------------------------------------------------
# %% settings
# -----------------------------------------------------------------------------

lds_fs            = 100.0   # Hz, matches the frame_100hz event convention
lds_limb          = 'R'     # which limb's heel strikes define the strides
lds_warmup_s      = 60.0    # skip the acclimatisation period at the start
lds_n_strides     = 150     # fixed, must be the same for every trial compared

lds_savgol_win    = 11      # odd, samples. One stage, smooths and differentiates
lds_savgol_poly   = 3       # this setting is -3 dB at 9.9 Hz
lds_max_gap_frames = 5      # longest run of NaN allowed inside the window
lds_edge_pad_s    = 5.0     # gap check margin either side of the window

lds_normalise     = True    # resample every stride to lds_samples_per_stride
lds_samples_per_stride = 100

# !! FIXED FOR THE WHOLE DATASET, from the separate parameter script !!
# Units are samples of the analysed series, so with lds_normalise = True these
# are NORMALISED samples (100 per stride), not 100 Hz samples.
lds_tau           = 10
lds_dE            = 5

lds_curve_strides = 10      # length of the divergence curve, in strides
lds_theiler_strides = 1.0   # exclude neighbours within this many strides
lds_k             = 200     # neighbours requested per point
lds_max_ref       = 6000    # reference points used, None = all
lds_n_boot        = 200     # bootstrap resamples for the CI, 0 = off
lds_seed          = 0
lds_plot_strides  = 20      # strides drawn in the attractor figure

# Sweep a fixed length window across the whole trial. The bootstrap above
# resamples reference points INSIDE one window, so it only measures ensemble
# noise. This asks the different question of how much lambda depends on WHICH
# strides you analysed, and whether it drifts over the trial. Primary state
# space only, it costs roughly a second per window.
lds_sweep         = True
lds_window_step   = 25      # strides between consecutive sweep windows

# fit windows in strides. S is the primary, the others are sensitivity checks
lds_fit_windows = {
    'S':      (0.0, 0.5),    # [Bruijn2009a]
    'S_1':    (0.0, 1.0),    # Dingwell convention [Dingwell2000, England2007]
    'S_post': (0.1, 0.5),    # transient excluded, diagnostic, not a convention
    'L':      (4.0, 10.0),   # [Dingwell2000], see the caution above
}

# -----------------------------------------------------------------------------
# %% state space definitions
# -----------------------------------------------------------------------------
# signal spec is ('Base_Column', axis, derivative)
#   axis        'X' -> Base_Column    'Y' -> Base_Column.1    'Z' -> Base_Column.2
#   derivative  'pos' | 'vel' | 'acc'
# embed  'delay' = Takens embedding, one common tau, total dim = channels * dE
#        'none'  = the listed signals ARE the state variables
# scale  'none'   raw units. Safe for one channel, or for channels sharing a
#                 unit. lambda is invariant to a UNIFORM rescale of the state
#                 space, but NOT to a per channel one.
#        'zscore' per channel, so no longer automatically comparable between
#                 conditions unless you divide by a fixed reference SD.
#
# trunkVel_ML is the primary. Velocity based trunk state spaces are the best
# supported choice [Gates2009, vanSchooten2013] and ML is the direction most
# sensitive to balance impairment [Sloot2011, Bruijn2013]. The rest are
# pre declared sensitivity analyses, so handle multiplicity if you report them.

lds_state_spaces = {
    'trunkVel_ML':    {'signals': [('Trunk_Position', 'X', 'vel')],
                       'embed': 'delay', 'dE': lds_dE, 'scale': 'none'},

    'trunkVel_AP':    {'signals': [('Trunk_Position', 'Y', 'vel')],
                       'embed': 'delay', 'dE': lds_dE, 'scale': 'none'},

    'trunkVel_VT':    {'signals': [('Trunk_Position', 'Z', 'vel')],
                       'embed': 'delay', 'dE': lds_dE, 'scale': 'none'},

    # All three velocity components as the state directly. No delay embedding:
    # the three measured components already span the trunk velocity state, and
    # embedding them as well pushes the dimension to 9 or 15, which thins the
    # neighbours out and costs headroom. Check the headroom column, this space
    # is usually the best conditioned of the lot.
    'trunkVel_3D':    {'signals': [('Trunk_Position', 'X', 'vel'),
                                   ('Trunk_Position', 'Y', 'vel'),
                                   ('Trunk_Position', 'Z', 'vel')],
                       'embed': 'none', 'dE': None, 'scale': 'none'},

    'thoraxAngle_ML': {'signals': [('Thorax_Seg_Angle', 'X', 'pos')],
                       'embed': 'delay', 'dE': lds_dE, 'scale': 'none'},
}

lds_primary = 'trunkVel_ML'

# -----------------------------------------------------------------------------
# %% pick the analysis window from the heel strikes that already exist
# -----------------------------------------------------------------------------

lds_hs_all = (gait_event_data.loc[gait_event_data['support_limb'] == lds_limb,
                                  'initial_contact_kinematic_frame_100hz']
              .dropna().astype(int).sort_values().to_numpy())

#skip the acclimatisation period, treadmill gait is not steady state at the start
lds_start = int(np.searchsorted(lds_hs_all, lds_hs_all[0] + lds_warmup_s * lds_fs))

lds_hs_avail = lds_hs_all[lds_start:]

lds_short = False
if len(lds_hs_avail) < lds_n_strides + 1:
    lds_short = True
    lds_n_strides = len(lds_hs_avail) - 1
    print(f" Only {lds_n_strides} strides after the warm up. lambda depends on "
          f"the amount of data, so this trial is NOT comparable to trials run "
          f"at the full stride count")

#how much of the record to condition. The reported result always uses the first
#lds_n_strides strides; with the sweep on we also need everything after them
lds_n_span = (len(lds_hs_avail) - 1) if lds_sweep else lds_n_strides
lds_hs = lds_hs_avail[: lds_n_span + 1]

#map event frames onto row positions in the kinematic data
lds_frame = pd.to_numeric(kinematic_data['Unnamed: 0'], errors='coerce').to_numpy()
lds_frame_position = pd.Series(np.arange(len(kinematic_data)), index=lds_frame)

lds_i0 = int(lds_frame_position[lds_hs[0]])
lds_i1 = int(lds_frame_position[lds_hs[-1]])

#brain check, the frame to row mapping has to be unique and gap free or every
#lag in the divergence curve is off
if pd.Series(lds_frame).duplicated().any():
    raise ValueError(" Kinematic frame column has duplicates")
if np.any(np.diff(lds_frame[lds_i0:lds_i1 + 1]) != 1):
    raise ValueError(" Kinematic frames are not contiguous inside the LDS window")

lds_stride_samples = np.diff(lds_hs[: lds_n_strides + 1])
lds_stride_time = np.mean(lds_stride_samples) / lds_fs
lds_stride_cv = np.std(lds_stride_samples, ddof=1) / np.mean(lds_stride_samples) * 100

#row positions of every heel strike inside the window, relative to lds_i0
lds_hs_rows = lds_hs - lds_hs[0]

print("=" * 76)
print(f" Local dynamic stability")
print(f" Limb          : {lds_limb}")
print(f" Strides       : {lds_n_strides} starting at stride {lds_start} "
      f"(of {lds_n_span} available after the warm up)")
print(f" Mean stride   : {lds_stride_time:.3f} s  (CV {lds_stride_cv:.2f}%)")
print(f" Embedding     : tau = {lds_tau}, dE = {lds_dE}  (fixed, not fitted here)")
print(f" Time normalise: {lds_normalise}")
print("=" * 76)

# -----------------------------------------------------------------------------
# %% build every signal that any state space asks for
# -----------------------------------------------------------------------------

lds_specs = []
for lds_ss in lds_state_spaces.values():
    for lds_spec in lds_ss['signals']:
        if lds_spec not in lds_specs:
            lds_specs.append(lds_spec)

lds_signals = {}

for lds_spec in lds_specs:
    lds_base, lds_axis, lds_kind = lds_spec
    lds_col = lds_base + {'X': '', 'Y': '.1', 'Z': '.2'}[lds_axis]

    if lds_col not in kinematic_data.columns:
        print(f" Column {lds_col} not in kinematic_data, state spaces using it "
              f"will be skipped")
        continue

    lds_x = pd.to_numeric(kinematic_data[lds_col], errors='coerce').to_numpy(float)

    #gap check on the window plus a margin for the filter. Reject rather than
    #fill anything long: an interpolated span is artificially smooth and locally
    #low dimensional, and it will drag lambda down
    lds_pad = int(lds_edge_pad_s * lds_fs)
    lds_bad = ~np.isfinite(lds_x[max(0, lds_i0 - lds_pad): lds_i1 + 1 + lds_pad])
    lds_edges = np.diff(np.concatenate(([0], lds_bad.astype(int), [0])))
    lds_gap = 0
    if lds_bad.any():
        lds_gap = int((np.flatnonzero(lds_edges == -1) -
                       np.flatnonzero(lds_edges == 1)).max())
    if lds_gap > lds_max_gap_frames:
        print(f" {lds_col} has a {lds_gap} frame gap in the window, "
              f"state spaces using it will be skipped")
        continue

    #fill the short gaps that are left
    if not np.isfinite(lds_x).all():
        lds_ok = np.isfinite(lds_x)
        lds_x = np.interp(np.arange(len(lds_x)), np.flatnonzero(lds_ok), lds_x[lds_ok])

    if 'Angle' in lds_base:
        lds_x = np.deg2rad(lds_x)

    #one stage, on the FULL record and in real time, so the filter transient
    #never lands inside the window and the derivative is a time derivative
    lds_order = {'pos': 0, 'vel': 1, 'acc': 2}[lds_kind]
    lds_x = savgol_filter(lds_x, lds_savgol_win, lds_savgol_poly,
                          deriv=lds_order, delta=1.0 / lds_fs)

    lds_signals[lds_spec] = lds_x[lds_i0: lds_i1 + 1]

# -----------------------------------------------------------------------------
# %% stride time normalisation, fixed samples per stride
# -----------------------------------------------------------------------------
# Each stride is resampled over [HS_k, HS_k+1), so consecutive strides join
# without the one sample phase gap a linspace over [0, len-1] would leave.

if lds_normalise:
    for lds_spec in list(lds_signals):
        lds_x = lds_signals[lds_spec]
        lds_pieces = []
        for lds_s in range(len(lds_hs_rows) - 1):
            lds_seg = lds_x[lds_hs_rows[lds_s]: lds_hs_rows[lds_s + 1] + 1]
            lds_grid = np.linspace(0, len(lds_seg) - 1, lds_samples_per_stride,
                                   endpoint=False)
            lds_pieces.append(np.interp(lds_grid, np.arange(len(lds_seg)), lds_seg))
        lds_signals[lds_spec] = np.concatenate(lds_pieces)
    lds_spstride = float(lds_samples_per_stride)
    lds_stride_start = np.arange(lds_n_span + 1) * int(lds_spstride)
else:
    lds_spstride = float(np.mean(np.diff(lds_hs_rows)))
    lds_stride_start = lds_hs_rows
    print(" Time normalisation is off, so samples per stride varies with cadence")

lds_theiler = int(round(lds_theiler_strides * lds_spstride))
lds_n_lags = int(round(lds_curve_strides * lds_spstride))

# -----------------------------------------------------------------------------
# %% divergence curve and Lyapunov exponents, one state space at a time
# -----------------------------------------------------------------------------

lds_rng = np.random.default_rng(lds_seed)
lds_results = []
lds_curves = {}

print("")
for lds_name in lds_state_spaces:
    lds_ss = lds_state_spaces[lds_name]

    if not all(lds_s in lds_signals for lds_s in lds_ss['signals']):
        print(f"  {lds_name} skipped, a channel was not available")
        continue
    print(f"  {lds_name} ...")

    #gather the channels and scale them. Only the reported window, the sweep
    #further down walks the same channels across the rest of the trial
    lds_chans = [lds_signals[lds_s][lds_stride_start[0]:
                                    lds_stride_start[lds_n_strides]].copy()
                 for lds_s in lds_ss['signals']]
    if lds_ss['scale'] == 'zscore':
        for lds_c in range(len(lds_chans)):
            lds_chans[lds_c] = lds_chans[lds_c] / np.std(lds_chans[lds_c])
        print("    ! zscored on this trial's own SD, so this state space is not "
              "on a common geometry with other conditions")

    #build the state space, one common tau for every channel
    if lds_ss['embed'] == 'delay':
        lds_de = int(lds_ss['dE'])
        lds_M = len(lds_chans[0]) - (lds_de - 1) * lds_tau
        lds_Y = np.column_stack([lds_c[lds_k * lds_tau: lds_k * lds_tau + lds_M]
                                 for lds_c in lds_chans
                                 for lds_k in range(lds_de)])
    else:
        lds_de = np.nan
        lds_Y = np.column_stack(lds_chans)
        lds_M = lds_Y.shape[0]

    #nearest neighbours outside the Theiler window. Both the reference point and
    #its neighbour have to survive the whole curve, so <ln d(i)> stays an average
    #over ONE fixed set of pairs at every lag
    lds_last = lds_M - 1 - lds_n_lags
    lds_self = np.arange(lds_last + 1)

    lds_idx = cKDTree(lds_Y).query(lds_Y[:lds_last + 1], k=lds_k)[1]

    lds_nn = np.full(lds_last + 1, -1)
    for lds_col in range(1, lds_k):
        lds_cand = lds_idx[:, lds_col]
        lds_take = ((lds_nn < 0) & (lds_cand <= lds_last) &
                    (np.abs(lds_cand - lds_self) > lds_theiler))
        lds_nn[lds_take] = lds_cand[lds_take]

    lds_ref = np.flatnonzero(lds_nn >= 0)
    lds_unresolved = 1 - len(lds_ref) / (lds_last + 1)
    if lds_unresolved > 0.01:
        print(f"    ! {100 * lds_unresolved:.1f}% of points found no neighbour "
              f"outside the Theiler window, raise lds_k")

    if lds_max_ref is not None and len(lds_ref) > lds_max_ref:
        lds_ref = np.sort(lds_rng.choice(lds_ref, lds_max_ref, replace=False))

    #mean log divergence curve.
    #
    #Expect a ripple at exactly one cycle per stride on top of the rise. It is
    #not noise and it is not a bug. The nearest neighbour of a point is almost
    #always at the SAME phase of the gait cycle a few strides away (measured:
    #~83% of pairs are within 2 normalised samples of the same phase), so the
    #pair travels round the loop together. The loop is not equally thick all the
    #way round, because stride to stride variability depends on where you are in
    #the cycle, so their separation breathes once per stride. Most of it cancels
    #when averaging over reference points at all phases; what survives is about
    #1% of the rise across the lambda_S window. It only looks large on the
    #plateau, where the y axis is expanded. See the attractor figure at the end.
    lds_lnd = np.empty((len(lds_ref), lds_n_lags + 1))
    for lds_i in range(lds_n_lags + 1):
        lds_dd = np.linalg.norm(lds_Y[lds_ref + lds_i] -
                                lds_Y[lds_nn[lds_ref] + lds_i], axis=1)
        lds_lnd[:, lds_i] = np.log(np.maximum(lds_dd, 1e-300))
    lds_curve = lds_lnd.mean(axis=0)

    #saturation level, the mean log distance between unrelated points. A fixed
    #fit window couples slope to intercept, so both get reported
    lds_p1 = lds_rng.integers(0, lds_M, 20000)
    lds_p2 = lds_rng.integers(0, lds_M, 20000)
    lds_far = np.abs(lds_p1 - lds_p2) > lds_theiler
    lds_attractor = np.mean(np.log(np.linalg.norm(
        lds_Y[lds_p1[lds_far]] - lds_Y[lds_p2[lds_far]], axis=1)))
    lds_span = lds_attractor - lds_curve[0]

    #bootstrap the curve by resampling reference points, as multinomial weights
    #so nothing gets copied
    lds_boot = None
    if lds_n_boot:
        lds_w = lds_rng.multinomial(len(lds_ref),
                                    np.full(len(lds_ref), 1 / len(lds_ref)),
                                    size=lds_n_boot)
        lds_boot = lds_w @ lds_lnd / len(lds_ref)

    #fit every window
    lds_row = {'trial': gait_event_path.stem.replace('_merged_events', ''),
               'state_space': lds_name,
               'is_primary': lds_name == lds_primary,
               'signals': ' + '.join(f'{lds_s[0]}_{lds_s[1]}|{lds_s[2]}'
                                     for lds_s in lds_ss['signals']),
               'embed': lds_ss['embed'], 'scale': lds_ss['scale'],
               'limb': lds_limb, 'start_stride': lds_start,
               'n_strides': lds_n_strides, 'stride_count_short': lds_short,
               'first_frame': lds_hs[0], 'last_frame': lds_hs[-1],
               'mean_stride_time_s': lds_stride_time,
               'stride_time_cv_pct': lds_stride_cv,
               'time_normalized': lds_normalise,
               'samples_per_stride': lds_spstride,
               'tau': lds_tau, 'dE': lds_de,
               'total_dimension': lds_Y.shape[1],
               'n_points': lds_M, 'n_ref_points': len(lds_ref),
               'frac_unresolved': lds_unresolved,
               'mean_ln_d0': lds_curve[0],
               'ln_attractor_size': lds_attractor,
               'headroom_nats': lds_span,
               'savgol_win': lds_savgol_win, 'savgol_poly': lds_savgol_poly,
               'seed': lds_seed}

    for lds_tag in lds_fit_windows:
        lds_win = lds_fit_windows[lds_tag]
        lds_j0 = int(round(lds_win[0] * lds_spstride))
        lds_j1 = min(int(round(lds_win[1] * lds_spstride)), lds_n_lags)
        lds_t = np.arange(lds_j0, lds_j1 + 1) / lds_spstride
        lds_seg = lds_curve[lds_j0: lds_j1 + 1]

        lds_slope = np.polyfit(lds_t, lds_seg, 1)[0]
        lds_row[f'lambda_{lds_tag}'] = lds_slope
        lds_row[f'R2_{lds_tag}'] = np.corrcoef(lds_t, lds_seg)[0, 1] ** 2
        lds_row[f'window_{lds_tag}'] = f'{lds_win[0]:g}-{lds_win[1]:g}'
        #where this window sits between the starting separation and saturation
        lds_row[f'headroom_{lds_tag}_lo'] = (lds_curve[lds_j0] - lds_curve[0]) / lds_span
        lds_row[f'headroom_{lds_tag}_hi'] = (lds_curve[lds_j1] - lds_curve[0]) / lds_span

        if lds_boot is not None:
            lds_bs = np.polyfit(lds_t, lds_boot[:, lds_j0: lds_j1 + 1].T, 1)[0]
            lds_row[f'lambda_{lds_tag}_lo'] = np.percentile(lds_bs, 2.5)
            lds_row[f'lambda_{lds_tag}_hi'] = np.percentile(lds_bs, 97.5)
            lds_row[f'lambda_{lds_tag}_sd'] = np.std(lds_bs, ddof=1)

    #per second only means something on real time data. Converting a per stride
    #exponent from normalised data back to per second just puts the cadence
    #dependence back in
    lds_row['lambda_S_per_s'] = (np.nan if lds_normalise
                                 else lds_row['lambda_S'] / lds_stride_time)

    #brain check
    if lds_row['R2_S'] < 0.90:
        print(f"    ! R2 = {lds_row['R2_S']:.2f} on the lambda_S fit, the curve "
              f"may have no linear region here")
    if lds_row['headroom_L_hi'] - lds_row['headroom_L_lo'] < 0.02:
        print("    ! the curve is flat across the lambda_L window, so lambda_L "
              "is fitting a plateau, not a divergence rate")

    #keep what the figures need. For a delay space only column 0 is the real
    #signal, the rest are lagged copies of it
    if lds_ss['embed'] == 'delay':
        lds_labels = [f'{lds_ss["signals"][0][0]} {lds_ss["signals"][0][1]} (t)',
                      '(t + Td)', '(t + 2Td)']
    else:
        lds_labels = [f'{lds_s[1]} {lds_s[2]}' for lds_s in lds_ss['signals'][:3]]

    #a handful of neighbour pairs starting at the same point in the cycle, so
    #panel C is a tight bundle rather than segments scattered round the loop
    lds_phase = lds_ref % int(lds_spstride)
    lds_pick = lds_ref[lds_phase == int(0.25 * lds_spstride)][:8]
    if len(lds_pick) < 3:
        lds_pick = lds_ref[:8]

    lds_results.append(lds_row)
    lds_curves[lds_name] = {'curve': lds_curve, 'attractor': lds_attractor,
                            'boot': lds_boot, 'labels': lds_labels,
                            'embed': lds_ss['embed'], 'Y': lds_Y,
                            'pair_ref': lds_pick, 'pair_nn': lds_nn[lds_pick]}

lds_results = pd.DataFrame(lds_results)

# -----------------------------------------------------------------------------
# %% results table, output file and figures
# -----------------------------------------------------------------------------

if len(lds_results) == 0:
    raise ValueError(" No state space could be computed for this trial, see the "
                     "messages above. Usually a missing column or a data gap "
                     "inside the analysis window")

print("\n" + lds_results[['state_space', 'total_dimension', 'headroom_nats',
                          'lambda_S', 'lambda_S_lo', 'lambda_S_hi', 'R2_S',
                          'lambda_S_post', 'lambda_L']].to_string(index=False))

if not lds_results['is_primary'].any():
    print(f"\n ! the primary state space ({lds_primary}) was dropped for this "
          f"trial. Do not swap one of the sensitivity analyses in for it, fix "
          f"the input or exclude the trial")
else:
    lds_p = lds_results[lds_results['is_primary']].iloc[0]
    print(f"\n PRIMARY  {lds_p['state_space']}")
    print(f"   lambda_S      = {lds_p['lambda_S']:.4f} per stride "
          f"[{lds_p['lambda_S_lo']:.4f}, {lds_p['lambda_S_hi']:.4f}] 95% CI")
    print(f"   lambda_S_post = {lds_p['lambda_S_post']:.4f} with the transient "
          f"excluded, the gap is the neighbour selection artifact")
    print("   everything else in the table is a sensitivity analysis, handle "
          "multiplicity if you report it")

lds_out_path = base / "LDS_outputs" / f"{gait_event_path.stem.replace('_merged_events', '')}_LDS.csv"
lds_out_path.parent.mkdir(parents=True, exist_ok=True)
lds_results.to_csv(lds_out_path, index=False)
print(f"\n LDS results -> {lds_out_path}")

#save the curves too, so fit windows can be changed later without re-running
lds_curve_table = pd.DataFrame({'strides': np.arange(lds_n_lags + 1) / lds_spstride})
for lds_name in lds_curves:
    lds_curve_table[lds_name] = lds_curves[lds_name]['curve']
lds_curve_path = lds_out_path.with_name(lds_out_path.stem + '_curves.csv')
lds_curve_table.to_csv(lds_curve_path, index=False)
print(f" Divergence curves -> {lds_curve_path}")

# -----------------------------------------------------------------------------
# %% sweep a fixed length window across the trial
# -----------------------------------------------------------------------------
# The bootstrap in the main loop resamples reference points INSIDE one window,
# so it only measures how precisely the ensemble average is pinned down. It
# cannot tell you whether a different 150 strides would have given a different
# answer. This walks the same window across the whole trial and asks exactly
# that, and at the same time shows whether lambda drifts as the trial goes on.
#
# The window LENGTH is held fixed, because lambda depends on the amount of data
# [Bruijn2009a]. Only the starting stride moves.
#
# This repeats the neighbour and curve code from the main loop in a shorter
# form. If you change the recipe up there, change it here too.

lds_sweep_results = []

if lds_sweep and lds_primary in lds_curves:
    lds_ss = lds_state_spaces[lds_primary]
    lds_sw_chans = [lds_signals[lds_s].copy() for lds_s in lds_ss['signals']]
    if lds_ss['scale'] == 'zscore':
        for lds_c in range(len(lds_sw_chans)):
            lds_sw_chans[lds_c] = lds_sw_chans[lds_c] / np.std(lds_sw_chans[lds_c])

    lds_sw_starts = np.arange(0, lds_n_span - lds_n_strides + 1, lds_window_step)
    print(f"\n Sweeping a {lds_n_strides} stride window across {lds_primary}: "
          f"{len(lds_sw_starts)} positions, step {lds_window_step} strides")

    for lds_w0 in lds_sw_starts:
        lds_wc = [lds_c[lds_stride_start[lds_w0]:
                        lds_stride_start[lds_w0 + lds_n_strides]]
                  for lds_c in lds_sw_chans]

        if lds_ss['embed'] == 'delay':
            lds_de = int(lds_ss['dE'])
            lds_M = len(lds_wc[0]) - (lds_de - 1) * lds_tau
            lds_Y = np.column_stack([lds_c[lds_k * lds_tau: lds_k * lds_tau + lds_M]
                                     for lds_c in lds_wc
                                     for lds_k in range(lds_de)])
        else:
            lds_Y = np.column_stack(lds_wc)
            lds_M = lds_Y.shape[0]

        lds_last = lds_M - 1 - lds_n_lags
        lds_self = np.arange(lds_last + 1)
        lds_idx = cKDTree(lds_Y).query(lds_Y[:lds_last + 1], k=lds_k)[1]

        lds_nn = np.full(lds_last + 1, -1)
        for lds_col in range(1, lds_k):
            lds_cand = lds_idx[:, lds_col]
            lds_take = ((lds_nn < 0) & (lds_cand <= lds_last) &
                        (np.abs(lds_cand - lds_self) > lds_theiler))
            lds_nn[lds_take] = lds_cand[lds_take]

        lds_ref = np.flatnonzero(lds_nn >= 0)
        if lds_max_ref is not None and len(lds_ref) > lds_max_ref:
            lds_ref = np.sort(lds_rng.choice(lds_ref, lds_max_ref, replace=False))

        lds_curve = np.empty(lds_n_lags + 1)
        for lds_i in range(lds_n_lags + 1):
            lds_dd = np.linalg.norm(lds_Y[lds_ref + lds_i] -
                                    lds_Y[lds_nn[lds_ref] + lds_i], axis=1)
            lds_curve[lds_i] = np.mean(np.log(np.maximum(lds_dd, 1e-300)))

        lds_sw_row = {'trial': lds_results['trial'][0],
                      'state_space': lds_primary,
                      'window_index': int(lds_w0),
                      'start_stride': int(lds_start + lds_w0),
                      'n_strides': lds_n_strides}
        for lds_tag in ('S', 'L'):
            lds_win = lds_fit_windows[lds_tag]
            lds_j0 = int(round(lds_win[0] * lds_spstride))
            lds_j1 = min(int(round(lds_win[1] * lds_spstride)), lds_n_lags)
            lds_t = np.arange(lds_j0, lds_j1 + 1) / lds_spstride
            lds_seg = lds_curve[lds_j0: lds_j1 + 1]
            lds_sw_row[f'lambda_{lds_tag}'] = np.polyfit(lds_t, lds_seg, 1)[0]
            lds_sw_row[f'R2_{lds_tag}'] = np.corrcoef(lds_t, lds_seg)[0, 1] ** 2
        lds_sweep_results.append(lds_sw_row)

    lds_sweep_results = pd.DataFrame(lds_sweep_results)
    lds_sw_lam = lds_sweep_results['lambda_S'].to_numpy()

    #windows that share no strides with each other are the only independent ones
    lds_indep = lds_sweep_results[
        lds_sweep_results['window_index'] % lds_n_strides == 0]
    lds_win_sd = lds_indep['lambda_S'].std(ddof=1)
    lds_boot_sd = lds_results.loc[lds_results['is_primary'], 'lambda_S_sd'].iloc[0]

    print(f"   lambda_S over windows  : median {np.median(lds_sw_lam):.4f}, "
          f"IQR {np.percentile(lds_sw_lam, 25):.4f} to "
          f"{np.percentile(lds_sw_lam, 75):.4f}, "
          f"full range {lds_sw_lam.min():.4f} to {lds_sw_lam.max():.4f}")
    print(f"   non overlapping windows: {len(lds_indep)}, mean "
          f"{lds_indep['lambda_S'].mean():.4f}, SD {lds_win_sd:.4f}")

    #Drift in units you can read. Deliberately no p value: consecutive windows
    #share up to 149 of their 150 strides, so they are nowhere near independent
    #and any test on them would be far too optimistic. Carry the slope to the
    #group analysis and test it there, across participants.
    lds_trend = np.polyfit(lds_sweep_results['start_stride'], lds_sw_lam, 1)[0]
    print(f"   drift across the trial : {100 * lds_trend:+.4f} per 100 strides "
          f"(descriptive, no test, the windows overlap)")

    #the comparison this whole section exists for
    print(f"   reference point bootstrap SD {lds_boot_sd:.4f}  vs  window to "
          f"window SD {lds_win_sd:.4f}  ({lds_win_sd / lds_boot_sd:.1f}x)")
    if len(lds_indep) < 5:
        print(f"      (that SD is from only {len(lds_indep)} independent windows, "
              f"so it is itself rough.\n       Pool it across trials before "
              f"leaning on it)")
    if lds_win_sd > 2 * lds_boot_sd:
        print("   -> which strides you analyse matters more than the ensemble "
              "averaging does.\n      Quote the window to window spread, not "
              "the bootstrap CI, as this trial's uncertainty")

    lds_sweep_path = lds_out_path.with_name(lds_out_path.stem + '_sweep.csv')
    lds_sweep_results.to_csv(lds_sweep_path, index=False)
    print(f" Window sweep -> {lds_sweep_path}")

    # --- figure: does lambda depend on where you start? ----------------------
    plt.figure(figsize=(9.0, 4.2))
    lds_p = lds_results[lds_results['is_primary']].iloc[0]

    plt.axhspan(lds_p['lambda_S_lo'], lds_p['lambda_S_hi'], color='#2a78d6',
                alpha=0.15, linewidth=0,
                label='95% CI of the reported window (reference points only)')
    plt.plot(lds_sweep_results['start_stride'], lds_sw_lam, '-o', color='#2a78d6',
             linewidth=1.8, markersize=4, label='lambda_S of each window')
    plt.plot(lds_indep['start_stride'], lds_indep['lambda_S'], 'o',
             color='#eb6834', markersize=9, fillstyle='none', markeredgewidth=1.8,
             label='non overlapping windows')
    plt.axhline(np.median(lds_sw_lam), color='#8a8a85', ls=':', linewidth=1.5,
                label=f'median {np.median(lds_sw_lam):.3f}')

    plt.xlabel('first stride of the window')
    plt.ylabel('lambda_S (per stride)')
    plt.title(f'{lds_primary}: a {lds_n_strides} stride window moved across the '
              f'trial', fontsize=10)
    plt.legend(fontsize=8, frameon=False)
    plt.grid(alpha=0.15)
    plt.tight_layout()

# -----------------------------------------------------------------------------
# %% one figure per state space, four panels showing how it is built
# -----------------------------------------------------------------------------
# (A) the conditioned signal that goes in
# (B) the reconstructed state space, one colour per stride
# (C) a few neighbour pairs, showing d(0) at the start and d(i) later
# (D) the divergence curve those pairs average into, and the fits
#
# Panel B is also the picture that explains the ripple in panel D: the loop is
# not equally thick all the way round, so a pair travelling round it together
# has its separation breathe once per stride.

lds_seg_strides = 0.25       # segment length drawn in panel C

for lds_name in lds_curves:
    lds_c = lds_curves[lds_name]
    lds_Yp = lds_c['Y']
    lds_curve = lds_c['curve']
    lds_ax = np.arange(len(lds_curve)) / lds_spstride

    lds_fig = plt.figure(figsize=(10.5, 8.0))
    lds_fig.suptitle(lds_name + ('   [PRIMARY]' if lds_name == lds_primary else ''),
                     fontsize=12)

    # --- (A) the conditioned signal ------------------------------------------
    #for a delay space only column 0 is the signal, the others are lagged copies
    lds_nch = 1 if lds_c['embed'] == 'delay' else min(3, lds_Yp.shape[1])
    lds_n_show = int(8 * lds_spstride)

    plt.subplot(2, 2, 1)
    for lds_ch in range(lds_nch):
        plt.plot(np.arange(lds_n_show) / lds_spstride, lds_Yp[:lds_n_show, lds_ch],
                 color=['#2a78d6', '#eb6834', '#1baf7a'][lds_ch], linewidth=1.5,
                 label=lds_c['labels'][lds_ch])
    plt.xlabel('time (strides)')
    plt.ylabel('signal')
    plt.title('(A) conditioned signal', fontsize=10, loc='left')
    if lds_nch > 1:
        plt.legend(fontsize=7, frameon=False)
    plt.grid(alpha=0.15)

    # --- (B) the reconstructed state space -----------------------------------
    lds_axis3d = plt.subplot(2, 2, 2, projection='3d')
    lds_colours = plt.cm.Blues(np.linspace(0.40, 1.0, lds_plot_strides))
    for lds_s in range(lds_plot_strides):
        lds_a = int(lds_s * lds_spstride)
        lds_b = int((lds_s + 1) * lds_spstride) + 1
        if lds_b > len(lds_Yp):
            break
        lds_axis3d.plot(lds_Yp[lds_a:lds_b, 0], lds_Yp[lds_a:lds_b, 1],
                        lds_Yp[lds_a:lds_b, 2], color=lds_colours[lds_s],
                        linewidth=0.8)
    lds_axis3d.set_xlabel(lds_c['labels'][0], fontsize=7, labelpad=-4)
    lds_axis3d.set_ylabel(lds_c['labels'][1], fontsize=7, labelpad=-4)
    lds_axis3d.set_zlabel(lds_c['labels'][2], fontsize=7, labelpad=-4)
    lds_axis3d.tick_params(labelsize=6, pad=-2)
    lds_axis3d.set_title(f'(B) state space, {lds_plot_strides} strides, '
                         f'light to dark', fontsize=10, loc='left')

    # --- (C) neighbour pairs separating --------------------------------------
    #grey is the whole attractor in this projection, for context. Blue is a
    #reference trajectory, orange is the neighbour the search paired it with.
    #They sit next to each other in state space but are many strides apart in
    #time, and panel D is the average of ln of the dotted separations.
    plt.subplot(2, 2, 3)
    lds_seg = int(lds_seg_strides * lds_spstride)

    plt.plot(lds_Yp[:lds_n_show, 0], lds_Yp[:lds_n_show, 1], color='#d8d8d4',
             linewidth=0.6, zorder=1)

    for lds_q in range(min(3, len(lds_c['pair_ref']))):
        lds_r0 = lds_c['pair_ref'][lds_q]
        lds_n0 = lds_c['pair_nn'][lds_q]
        lds_rt = lds_Yp[lds_r0: lds_r0 + lds_seg + 1]
        lds_nt = lds_Yp[lds_n0: lds_n0 + lds_seg + 1]

        plt.plot(lds_rt[:, 0], lds_rt[:, 1], color='#2a78d6', linewidth=1.6,
                 zorder=3, label='reference' if lds_q == 0 else None)
        plt.plot(lds_nt[:, 0], lds_nt[:, 1], color='#eb6834', linewidth=1.6,
                 zorder=3, label='its neighbour' if lds_q == 0 else None)

        #separation at the start of the segment and at the end of it
        plt.plot([lds_rt[0, 0], lds_nt[0, 0]], [lds_rt[0, 1], lds_nt[0, 1]],
                 color='#4a3aa7', linewidth=1.6, zorder=4,
                 label='d(0)' if lds_q == 0 else None)
        plt.plot([lds_rt[-1, 0], lds_nt[-1, 0]], [lds_rt[-1, 1], lds_nt[-1, 1]],
                 color='#4a3aa7', linewidth=1.6, linestyle=':', zorder=4,
                 label='d(i)' if lds_q == 0 else None)
        plt.plot([lds_rt[0, 0], lds_nt[0, 0]], [lds_rt[0, 1], lds_nt[0, 1]], 'o',
                 color='#4a3aa7', markersize=3.5, zorder=5)

    plt.xlabel(lds_c['labels'][0])
    plt.ylabel(lds_c['labels'][1])
    plt.title(f'(C) 3 neighbour pairs over {lds_seg_strides:g} stride',
              fontsize=10, loc='left')
    plt.legend(fontsize=7, frameon=False)
    plt.grid(alpha=0.15)

    # --- (D) the divergence curve --------------------------------------------
    plt.subplot(2, 2, 4)
    if lds_c['boot'] is not None:
        plt.fill_between(lds_ax, np.percentile(lds_c['boot'], 2.5, axis=0),
                         np.percentile(lds_c['boot'], 97.5, axis=0),
                         color='#2a78d6', alpha=0.18, linewidth=0, label='95% CI')
    plt.plot(lds_ax, lds_curve, color='#2a78d6', linewidth=2, label='<ln d(i)>')
    plt.axhline(lds_c['attractor'], color='#8a8a85', ls=':', linewidth=1.5,
                label='attractor size')
    for lds_tag, lds_colour, lds_style in [('S', '#eb6834', '--'),
                                           ('L', '#4a3aa7', '-.')]:
        lds_win = lds_fit_windows[lds_tag]
        lds_m = (lds_ax >= lds_win[0]) & (lds_ax <= lds_win[1])
        lds_fit = np.polyfit(lds_ax[lds_m], lds_curve[lds_m], 1)
        plt.plot(lds_ax[lds_m], np.polyval(lds_fit, lds_ax[lds_m]), ls=lds_style,
                 color=lds_colour, linewidth=2,
                 label=f'lambda_{lds_tag} = {lds_fit[0]:.3f}/stride')
    plt.xlabel('time (strides)')
    plt.ylabel('<ln d(i)>')
    plt.title('(D) divergence curve', fontsize=10, loc='left')
    plt.legend(fontsize=7, loc='lower right', frameon=False)
    plt.grid(alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
