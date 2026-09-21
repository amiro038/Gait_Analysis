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

# =============================================================================
# %% Kinetics: load the force file, check its quality, re-derive gait events
# =============================================================================
# The split-belt treadmill gives one force plate per limb, so there is no plate
# hit to disambiguate. What it does need is care about three things that this
# cell measures rather than assumes:
#
#   1. The unloaded baseline is NOT zero. On this hardware an unloaded belt
#      reads about 8 N and peaks near 25 N. A 20 N threshold therefore sits
#      INSIDE the noise and chatters, producing spurious contacts. The
#      threshold is set as a fraction of body weight and then checked against
#      the measured noise floor.
#
#   2. The handrails are instrumented. Their channels carry a small non-zero
#      offset (2-3 N) that is a baseline, not contact. It is removed here, and
#      genuine contact is flagged, because any hand force breaks the
#      CoM-from-GRF calculation in the next cell.
#
#   3. Body mass comes from the data. The mean total vertical GRF over a
#      steady walking trial is mg, so mass does not have to be entered by hand
#      and cannot be mistyped.
#
# The threshold sweep at the end is the diagnostic that matters. Stride time CV
# should be about 2-3% for metronome paced walking. If it is much higher at a
# given threshold, that threshold is chattering, and the sweep shows you where
# the plateau is instead of leaving you to trust one hardcoded number.
#
# References
#   [Hof2005]     Hof, Gazendam & Sinke (2005) J Biomech 38, 1-8.
#   [Schulz2017]  Schulz (2017) J Biomech 55, 107-112.
# =============================================================================

# -----------------------------------------------------------------------------
# %% kinetics settings
# -----------------------------------------------------------------------------

kin_force_folder   = "Force_csv_outputs"  # sibling of Theia_csv_outputs under base
kin_fs             = 1000.0    # Hz, force sampling rate (checked against the file)
kin_kinematic_fs   = 100.0     # Hz, matches the frame_100hz event convention

# Contact thresholds as a fraction of body weight. 7% / 3% clears the measured
# noise floor on this hardware with room to spare; the check below refuses to
# continue quietly if it does not.
kin_on_frac        = 0.07      # rising threshold, fraction of body weight
kin_off_frac       = 0.03      # falling threshold, hysteresis
kin_min_stance_s   = 0.25      # shorter "contacts" than this are threshold chatter
kin_min_swing_s    = 0.15      # shorter "flights" than this are a dropout, not swing

kin_handrail_contact_n = 15.0  # N above baseline that counts as real hand contact
kin_sweep_fracs    = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]   # threshold diagnostic

# -----------------------------------------------------------------------------
# %% find and load the force file
# -----------------------------------------------------------------------------
# The force export names the trial with underscores where the event file uses a
# space, so both spellings are tried rather than assuming one.

kin_stem = gait_event_path.stem.replace('_merged_events', '')
kin_candidates = [
    base / kin_force_folder / f"{kin_stem}.csv",
    base / kin_force_folder / f"{kin_stem.replace(' ', '_')}.csv",
    base / kin_force_folder / f"{kin_stem.replace('_', ' ')}.csv",
]
kin_path = next((p for p in kin_candidates if p.exists()), None)

if kin_path is None:
    raise FileNotFoundError(
        "No force file found. Tried:\n  " + "\n  ".join(str(p) for p in kin_candidates)
        + "\nSet kin_force_folder to the folder holding the force exports.")

print("\n" + "=" * 74)
print("  KINETICS")
print("=" * 74)
print(f"  file: {kin_path.name}")

force_data = pd.read_csv(kin_path, index_col=0)

# -----------------------------------------------------------------------------
# %% sampling rate, time base and synchronisation to the kinematics
# -----------------------------------------------------------------------------

kin_time = force_data['Left belt_TIME'].to_numpy()
kin_fs_measured = 1.0 / np.median(np.diff(kin_time))

if abs(kin_fs_measured - kin_fs) > 1.0:
    print(f"  ! force file is {kin_fs_measured:.0f} Hz, not the {kin_fs:.0f} Hz "
          f"configured. Using the measured rate.")
    kin_fs = kin_fs_measured

if not np.allclose(force_data['Left belt_TIME'], force_data['Right belt_TIME']):
    print("  ! the two belts have different time columns, they should be identical")

kin_n_samples = len(force_data)
kin_duration  = kin_time[-1] - kin_time[0]
print(f"  {kin_n_samples} samples, {kin_duration:.1f} s at {kin_fs:.0f} Hz")

# Event frames are 1-based at 100 Hz, so frame f corresponds to force time
# (f - 1) / 100. This mapping was checked against GRF-detected contacts and
# agrees to about 3 ms once the threshold offset is removed.
def kin_frame_to_sample(frame_100hz):
    """100 Hz kinematic frame number -> index into the force arrays."""
    return np.round((np.asarray(frame_100hz, float) - 1.0)
                    * kin_fs / kin_kinematic_fs).astype(int)

def kin_sample_to_frame(sample):
    """Index into the force arrays -> 100 Hz kinematic frame number."""
    return np.asarray(sample, float) * kin_kinematic_fs / kin_fs + 1.0

# -----------------------------------------------------------------------------
# %% body mass from the vertical GRF
# -----------------------------------------------------------------------------
# Over a steady walking trial the mean total vertical force is body weight. No
# quiet-standing period is needed and nothing has to be typed in.

kin_grf_z_total = (force_data['Left belt_Force_Z'].to_numpy()
                   + force_data['Right belt_Force_Z'].to_numpy())
kin_body_weight = float(np.mean(kin_grf_z_total))
kin_body_mass   = kin_body_weight / 9.81

print(f"  body weight from mean total vertical GRF: {kin_body_weight:.1f} N "
      f"({kin_body_mass:.1f} kg)")

# -----------------------------------------------------------------------------
# %% handrails: remove the baseline, flag genuine contact
# -----------------------------------------------------------------------------
# Any hand force is an external force on the body, so it invalidates the
# CoM-from-GRF calculation in the next cell. The channels carry a small
# constant offset that must not be mistaken for contact, so the baseline is
# taken as the median of each channel and removed first.

handrail_force = {}
for kin_side in ('Left', 'Right'):
    kin_v = np.column_stack([
        force_data[f'{kin_side} handrail_Force_{a}'].to_numpy() for a in 'XYZ'])
    kin_baseline = np.median(kin_v, axis=0)
    handrail_force[kin_side] = kin_v - kin_baseline
    print(f"  {kin_side} handrail baseline removed: "
          f"({kin_baseline[0]:+.2f}, {kin_baseline[1]:+.2f}, {kin_baseline[2]:+.2f}) N")

handrail_magnitude = {s: np.linalg.norm(v, axis=1) for s, v in handrail_force.items()}
handrail_contact = np.zeros(kin_n_samples, bool)
for kin_side in ('Left', 'Right'):
    kin_touch = handrail_magnitude[kin_side] > kin_handrail_contact_n
    handrail_contact |= kin_touch
    print(f"  {kin_side} handrail: residual |F| max {handrail_magnitude[kin_side].max():.1f} N, "
          f"{100 * kin_touch.mean():.2f}% of samples over {kin_handrail_contact_n:.0f} N")

if handrail_contact.any():
    print(f"  ! handrail contact on {100 * handrail_contact.mean():.2f}% of samples. "
          f"Those frames are excluded from the CoM fusion.")
else:
    print("  no handrail contact detected, the GRF is the only external force")

# -----------------------------------------------------------------------------
# %% measure the unloaded noise floor, then set the contact thresholds
# -----------------------------------------------------------------------------
# The noise floor is estimated from samples that are unambiguously unloaded
# (below 5% body weight). The rising threshold has to clear it, otherwise the
# detector will fire on noise.

kin_grf_z = {s: force_data[f'{s} belt_Force_Z'].to_numpy() for s in ('Left', 'Right')}

# -----------------------------------------------------------------------------
# %% contact detection
# -----------------------------------------------------------------------------

def kin_detect_contacts(vertical_force, on_n, off_n,
                        min_stance_s=None, min_swing_s=None, fs=None):
    """Heel strike and toe off samples from one belt's vertical force.

    A Schmitt trigger (rise above on_n, fall below off_n) rather than a single
    threshold, because a single threshold chatters when the signal hovers near
    it. Contacts shorter than min_stance_s are discarded as chatter, and gaps
    shorter than min_swing_s are treated as a momentary dropout within one
    contact rather than a real swing phase, so the two halves are merged.

    Returns a list of (heel_strike_sample, toe_off_sample).
    """
    fs = kin_fs if fs is None else fs
    min_stance_s = kin_min_stance_s if min_stance_s is None else min_stance_s
    min_swing_s = kin_min_swing_s if min_swing_s is None else min_swing_s

    loaded = np.zeros(len(vertical_force), bool)
    kin_state = False
    for kin_i, kin_val in enumerate(vertical_force):
        kin_state = kin_val > off_n if kin_state else kin_val > on_n
        loaded[kin_i] = kin_state

    kin_rises = np.where(~loaded[:-1] & loaded[1:])[0] + 1
    kin_falls = np.where(loaded[:-1] & ~loaded[1:])[0] + 1

    kin_pairs = []
    for kin_h in kin_rises:
        kin_after = kin_falls[kin_falls > kin_h]
        if len(kin_after) and (kin_after[0] - kin_h) / fs >= min_stance_s:
            kin_pairs.append((int(kin_h), int(kin_after[0])))

    kin_merged = []
    for kin_h, kin_t in kin_pairs:
        if kin_merged and (kin_h - kin_merged[-1][1]) / fs < min_swing_s:
            kin_merged[-1] = (kin_merged[-1][0], kin_t)
        else:
            kin_merged.append((kin_h, kin_t))
    return kin_merged


# Two passes. A percentile of "samples below X" just returns X when the noise
# reaches that far, so the first pass finds contacts with a threshold that is
# safely clear of any plausible noise, and the second measures the floor only
# during confirmed mid-swing, where the belt is genuinely unloaded.
kin_noise_floor = {}
for kin_side, kin_z in kin_grf_z.items():
    kin_provisional = kin_detect_contacts(kin_z, 0.15 * kin_body_weight,
                                          0.08 * kin_body_weight)
    kin_swing_mask = np.zeros(len(kin_z), bool)
    for kin_a, kin_b in zip(kin_provisional[:-1], kin_provisional[1:]):
        kin_lo, kin_hi = kin_a[1], kin_b[0]          # toe off -> next heel strike
        kin_pad = int(0.2 * (kin_hi - kin_lo))       # middle 60% of swing only
        kin_swing_mask[kin_lo + kin_pad:kin_hi - kin_pad] = True
    kin_quiet = kin_z[kin_swing_mask]
    kin_noise_floor[kin_side] = float(np.percentile(kin_quiet, 99.9))
    print(f"  {kin_side} belt unloaded (mid-swing, n={kin_swing_mask.sum()}): "
          f"mean {kin_quiet.mean():.1f} N, 99.9th pct {kin_noise_floor[kin_side]:.1f} N, "
          f"max {kin_quiet.max():.1f} N")

# A plate that is much noisier than its partner is a hardware problem, not a
# parameter choice, and it shows up downstream as one limb needing a kinematic
# event fallback. Worth knowing before any left-right comparison is believed.
kin_floor_ratio = (max(kin_noise_floor.values())
                   / max(min(kin_noise_floor.values()), 1e-9))
if kin_floor_ratio > 1.8:
    kin_noisy = max(kin_noise_floor, key=kin_noise_floor.get)
    print(f"  ! the {kin_noisy} belt's noise floor is {kin_floor_ratio:.1f}x the other's "
          f"({kin_noise_floor[kin_noisy]:.0f} N vs "
          f"{min(kin_noise_floor.values()):.0f} N).")
    print(f"    That is a plate problem, not a threshold problem. It forces a higher")
    print(f"    threshold for both limbs and is the usual reason one limb falls back")
    print(f"    to kinematic event detection. Worth re-zeroing or servicing the plate.")

kin_on_n  = kin_on_frac * kin_body_weight
kin_off_n = kin_off_frac * kin_body_weight
kin_worst_floor = max(kin_noise_floor.values())

print(f"  thresholds: on {kin_on_n:.0f} N ({100*kin_on_frac:.0f}% BW), "
      f"off {kin_off_n:.0f} N ({100*kin_off_frac:.0f}% BW)")
if kin_on_n < 1.5 * kin_worst_floor:
    print(f"  ! the rising threshold ({kin_on_n:.0f} N) is close to the noise "
          f"floor ({kin_worst_floor:.0f} N). Raise kin_on_frac.")

# crosstalk between belts would make one plate read the other's load
for kin_side, kin_other in (('Left', 'Right'), ('Right', 'Left')):
    kin_off_mask = kin_grf_z[kin_side] < 0.03 * kin_body_weight
    if kin_off_mask.sum() > 100:
        kin_slope = np.polyfit(kin_grf_z[kin_other][kin_off_mask],
                               kin_grf_z[kin_side][kin_off_mask], 1)[0]
        print(f"  {kin_side} belt crosstalk from {kin_other}: {100*kin_slope:.2f}%")

force_contacts = {s: kin_detect_contacts(kin_grf_z[s], kin_on_n, kin_off_n)
                  for s in ('Left', 'Right')}

kin_contact_quality = {}
print("\n  contacts detected")
for kin_side, kin_con in force_contacts.items():
    kin_stance = np.array([(t - h) / kin_fs for h, t in kin_con])
    kin_stride = np.diff([h for h, _ in kin_con]) / kin_fs
    print(f"    {kin_side:5s} n={len(kin_con):4d}  "
          f"stance {kin_stance.mean():.3f} +- {kin_stance.std():.3f} s  "
          f"stride {kin_stride.mean():.3f} +- {kin_stride.std():.3f} s "
          f"(CV {100*kin_stride.std()/kin_stride.mean():.2f}%)")
    kin_contact_quality[kin_side] = {'n': len(kin_con),
                                     'stance_mean': kin_stance.mean(),
                                     'stance_sd': kin_stance.std(),
                                     'stride_cv': 100*kin_stride.std()/kin_stride.mean()}

# Stance time SD is the sharper of the two: a noisy plate blurs toe off much
# more than it blurs the stride period, which is set by the metronome anyway.
if len(kin_contact_quality) == 2:
    kin_sd_l = kin_contact_quality['Left']['stance_sd']
    kin_sd_r = kin_contact_quality['Right']['stance_sd']
    if max(kin_sd_l, kin_sd_r) > 2.0 * min(kin_sd_l, kin_sd_r):
        kin_worse = 'Left' if kin_sd_l > kin_sd_r else 'Right'
        print(f"    ! {kin_worse} stance time is {max(kin_sd_l,kin_sd_r)/min(kin_sd_l,kin_sd_r):.1f}x "
              f"more variable than the other limb ({1000*max(kin_sd_l,kin_sd_r):.0f} vs "
              f"{1000*min(kin_sd_l,kin_sd_r):.0f} ms SD).")
        print(f"      Check this is the walker and not the plate before reporting "
              f"any stance-time asymmetry.")

# -----------------------------------------------------------------------------
# %% threshold sensitivity, the diagnostic that catches a chattering belt
# -----------------------------------------------------------------------------
# Stride time CV should be about 2-3% under a metronome. A threshold that is
# too low reads noise as contact and inflates CV badly. Look for the plateau:
# if CV is still falling at the chosen threshold, the threshold is too low.

print("\n  threshold sweep (stride time CV should plateau near 2-3%)")
print("    %BW   on (N)    Left n   Left CV    Right n   Right CV")
kin_sweep_rows = []
for kin_frac in kin_sweep_fracs:
    kin_row = {'fraction_bw': kin_frac, 'on_n': kin_frac * kin_body_weight}
    kin_text = f"    {100*kin_frac:4.0f}  {kin_frac*kin_body_weight:7.0f}"
    for kin_side in ('Left', 'Right'):
        kin_con = kin_detect_contacts(kin_grf_z[kin_side],
                                      kin_frac * kin_body_weight,
                                      min(kin_off_n, 0.5 * kin_frac * kin_body_weight))
        if len(kin_con) > 2:
            kin_sd = np.diff([h for h, _ in kin_con]) / kin_fs
            kin_cv = 100 * kin_sd.std() / kin_sd.mean()
        else:
            kin_cv = np.nan
        kin_row[f'{kin_side}_n'] = len(kin_con)
        kin_row[f'{kin_side}_stride_cv'] = kin_cv
        kin_text += f"   {len(kin_con):6d}   {kin_cv:7.2f}%"
    kin_sweep_rows.append(kin_row)
    print(kin_text)

kin_threshold_sweep = pd.DataFrame(kin_sweep_rows)

# -----------------------------------------------------------------------------
# %% agreement with the existing event file
# -----------------------------------------------------------------------------
# The merged event file already carries GRF-derived events with a kinematic
# fallback. Comparing against a freshly detected set shows whether the fallback
# events are systematically offset, which matters because mixed timing sources
# add noise to every stride interval they touch.

print("\n  agreement with the merged event file")
kin_event_check = []
for kin_side, kin_limb in (('Left', 'L'), ('Right', 'R')):
    kin_file_ev = events[(events['event_type'] == 'heel_strike')
                         & (events['support_limb'] == kin_limb)]
    kin_file_t = (kin_file_ev['frame_100hz'].to_numpy() - 1.0) / kin_kinematic_fs
    kin_file_src = kin_file_ev['source'].to_numpy()
    kin_file_t = kin_file_t[kin_file_t <= kin_duration]
    kin_file_src = kin_file_src[:len(kin_file_t)]

    kin_mine_t = np.array([h for h, _ in force_contacts[kin_side]]) / kin_fs
    if not len(kin_mine_t) or not len(kin_file_t):
        continue
    kin_offsets = np.array([(kin_file_t - x)[np.argmin(np.abs(kin_file_t - x))]
                            for x in kin_mine_t]) * 1000.0
    kin_n_kinematic = int((kin_file_src != 'GRF').sum())
    print(f"    {kin_side:5s} file {len(kin_file_t):4d} events "
          f"({kin_n_kinematic} kinematic-sourced), detected {len(kin_mine_t):4d}, "
          f"offset {np.median(kin_offsets):+.1f} +- {kin_offsets.std():.1f} ms")
    kin_event_check.append({'side': kin_side, 'n_file': len(kin_file_t),
                            'n_detected': len(kin_mine_t),
                            'n_kinematic_source': kin_n_kinematic,
                            'median_offset_ms': float(np.median(kin_offsets)),
                            'offset_sd_ms': float(kin_offsets.std())})

kin_event_agreement = pd.DataFrame(kin_event_check)

# Any limb whose events came partly from kinematics has a mixed timing source.
# That is worth knowing before any symmetry metric is interpreted, because it
# makes an instrumentation difference look like a biomechanical one.
kin_fallback = events[events['source'] != 'GRF']
if len(kin_fallback):
    print(f"\n  ! {len(kin_fallback)} of {len(events)} events in the file came from "
          f"kinematics, not GRF:")
    for (kin_lm, kin_ty), kin_grp in kin_fallback.groupby(['support_limb', 'event_type']):
        print(f"      {kin_lm} {kin_ty}: {len(kin_grp)}")
    print("    If these are concentrated on one limb, that limb's stance and swing")
    print("    times carry a different timing bias from the other, and any left-right")
    print("    comparison is partly measuring the detector rather than the walker.")

# =============================================================================
# %% Centre of mass: fuse the segmental model with the force plates
# =============================================================================
# Every margin of stability number depends on CoM VELOCITY, and velocity is the
# weak link. There are two ways to get it and each is bad in a different band:
#
#   Differentiating the segmental-model CoM position is trustworthy at low
#     frequency (it cannot drift, it is tied to the markers) but differentiation
#     amplifies noise, so it is poor at high frequency exactly where the
#     within-stride dynamics live.
#
#   Integrating the GRF is trustworthy at high frequency (force is measured
#     directly, at 1000 Hz, with no differentiation) but any residual offset
#     integrates into a drifting velocity, so it is poor at low frequency.
#
# A complementary filter takes each where it is good: low frequencies from the
# kinematics, high frequencies from the force, with matched cutoffs so the two
# halves sum to unity and nothing is double counted or lost.
#
#     v_fused = lowpass(v_kinematic) + [ v_force - lowpass(v_force) ]
#
# Two things make this honest here. Body mass is taken as the mean total
# vertical GRF over the trial, so the mean of the force-derived acceleration is
# zero BY CONSTRUCTION and there is no net velocity ramp to remove. And the
# handrails are instrumented, so frames where the hand is loaded -- where the
# GRF is no longer the only external force and the fusion's assumption fails --
# are known rather than guessed at.
#
# The cutoff is the one real choice. Below it you are trusting the markers,
# above it the force plates. It is set below the stride fundamental so that
# within-stride dynamics come from the force, and the sweep printed at the end
# shows how little the answer moves over a sensible range of cutoffs.
#
# References
#   [Shimba1984]  Shimba (1984) J Biomech 17(1), 53-60. CoM from force plates.
#   [Maus2011]    Maus et al. (2011) J Exp Biol 214(21), 3511. Combining forces
#                 and kinematics for consistent CoM trajectories.
#   [Hof2005]     Hof, Gazendam & Sinke (2005) J Biomech 38, 1-8.
# =============================================================================

from scipy.signal import butter, filtfilt

# -----------------------------------------------------------------------------
# %% CoM settings
# -----------------------------------------------------------------------------

com_crossover_hz   = 0.5    # complementary filter cutoff, see the sweep below
com_filter_order   = 2
com_antialias_hz   = 40.0   # lowpass before decimating 1000 Hz -> 100 Hz
com_belt_speed_ms  = 1.3    # nominal; measured from the kinematics below
com_stance_window  = (0.30, 0.70)   # fraction of stance used as "foot flat"
com_cutoff_sweep   = [0.25, 0.5, 1.0, 2.0]

# Which axis is which. The force file and the Theia export agree: X is
# mediolateral, Y is anterior-posterior, Z is vertical.
com_ml_axis, com_ap_axis, com_vt_axis = 0, 1, 2

# -----------------------------------------------------------------------------
# %% force-derived CoM acceleration, resampled to the kinematic rate
# -----------------------------------------------------------------------------
# Newton's second law on the whole body: the sum of the external forces equals
# mass times CoM acceleration. Gravity is already in the measured vertical GRF
# as the mean, so subtracting body weight leaves the acceleration.

com_grf_total = np.column_stack([
    force_data[f'Left belt_Force_{a}'].to_numpy()
    + force_data[f'Right belt_Force_{a}'].to_numpy() for a in 'XYZ'])

# Over a steady treadmill trial the walker has no net acceleration in ANY axis,
# so the mean of each force channel is a zero offset rather than physiology.
# Vertical is handled by taking body weight as that mean; the horizontal
# channels get the same treatment for the same reason. On this hardware the
# horizontal offsets are ~3 N, which is small but integrates to metres per
# second of spurious velocity if it is left in. The complementary high-pass
# would hide it, but removing it explicitly is honest and makes the three axes
# consistent.
com_force_offset = com_grf_total.mean(axis=0)
com_force_offset[com_vt_axis] = kin_body_weight        # vertical mean IS mg
print("\n  force channel offsets removed (N): "
      f"ML {com_force_offset[com_ml_axis]:+.2f}, "
      f"AP {com_force_offset[com_ap_axis]:+.2f}, "
      f"VT {com_force_offset[com_vt_axis]:+.1f} (= body weight)")
if max(abs(com_force_offset[com_ml_axis]), abs(com_force_offset[com_ap_axis])) > 10.0:
    print("    ! a horizontal offset over 10 N is large. Check the plate zeroing;")
    print("      it biases propulsive and braking impulses, which are not high-passed.")

com_accel_force_1k = (com_grf_total - com_force_offset) / kin_body_mass

def com_decimate(signal_1k, factor=None, antialias_hz=None, fs=None):
    """1000 Hz -> 100 Hz with an anti-alias lowpass first.

    Taking every 10th sample without filtering folds everything above 50 Hz
    back into the band we care about. Heel strike transients are broadband, so
    this is not a hypothetical problem.
    """
    fs = kin_fs if fs is None else fs
    antialias_hz = com_antialias_hz if antialias_hz is None else antialias_hz
    factor = int(round(fs / kin_kinematic_fs)) if factor is None else factor
    com_b, com_a = butter(4, antialias_hz / (fs / 2), 'low')
    com_smooth = filtfilt(com_b, com_a, signal_1k, axis=0)
    return com_smooth[::factor]

com_accel_force = com_decimate(com_accel_force_1k)
handrail_contact_100hz = com_decimate(handrail_contact.astype(float)) > 0.05

print("\n" + "-" * 74)
print("  CENTRE OF MASS")
print("-" * 74)
print(f"  GRF-derived CoM acceleration: {com_accel_force.shape[0]} samples at "
      f"{kin_kinematic_fs:.0f} Hz")
print(f"    mean {com_accel_force.mean(axis=0)} m/s^2  (should be ~0 in all axes)")

# -----------------------------------------------------------------------------
# %% kinematic CoM, and the measured belt speed
# -----------------------------------------------------------------------------

com_have_kinematics = 'Whole_body_COG' in kinematic_data.columns
if com_have_kinematics:
    com_pos_kin = np.column_stack([
        kinematic_data['Whole_body_COG'].to_numpy(),
        kinematic_data['Whole_body_COG.1'].to_numpy(),
        kinematic_data['Whole_body_COG.2'].to_numpy()])
    print(f"  kinematic CoM: {len(com_pos_kin)} frames, "
          f"{100*np.isfinite(com_pos_kin).all(axis=1).mean():.1f}% finite")
else:
    com_pos_kin = None
    print("  ! no Whole_body_COG column, the fusion cannot run for this trial")

# The belt speed CANNOT be read off the centre of pressure. During stance the
# CoP travels heel to toe ALONG the foot at the same time as the foot travels
# backwards WITH the belt, and the CoP alone cannot separate the two. The foot
# itself can: during foot-flat mid-stance it is stationary relative to the belt,
# so its velocity in the lab frame IS the belt velocity.
com_belt_measured = np.nan
com_belt_samples = []
if 'Left_Heel_Position.1' in kinematic_data.columns:
    for com_side, com_col in (('Left', 'Left_Heel_Position.1'),
                              ('Right', 'Right_Heel_Position.1')):
        com_heel_ap = kinematic_data[com_col].to_numpy()
        for com_h, com_t in force_contacts[com_side]:
            com_a = int(kin_sample_to_frame(com_h + com_stance_window[0] * (com_t - com_h))) - 1
            com_b = int(kin_sample_to_frame(com_h + com_stance_window[1] * (com_t - com_h))) - 1
            if com_a < 0 or com_b >= len(com_heel_ap) or com_b - com_a < 5:
                continue
            com_seg = com_heel_ap[com_a:com_b]
            if not np.isfinite(com_seg).all():
                continue
            com_belt_samples.append(
                np.polyfit(np.arange(len(com_seg)) / kin_kinematic_fs, com_seg, 1)[0])

if len(com_belt_samples) > 10:
    com_belt_samples = np.array(com_belt_samples)
    com_belt_measured = float(np.median(np.abs(com_belt_samples)))
    print(f"  belt speed measured from heel AP velocity during foot-flat: "
          f"{com_belt_measured:.3f} m/s "
          f"(IQR {np.percentile(np.abs(com_belt_samples), 25):.3f}-"
          f"{np.percentile(np.abs(com_belt_samples), 75):.3f}, "
          f"n={len(com_belt_samples)} stances)")
    com_belt_sign = -np.sign(np.median(com_belt_samples))   # walking direction
    if abs(com_belt_measured - com_belt_speed_ms) > 0.1:
        print(f"    ! that differs from the configured {com_belt_speed_ms:.2f} m/s by "
              f"{abs(com_belt_measured - com_belt_speed_ms):.3f} m/s. The measured "
              f"value is used; check the protocol if the gap is large.")
    com_belt_speed_used = com_belt_measured
else:
    com_belt_sign = 1.0
    com_belt_speed_used = com_belt_speed_ms
    print(f"  ! could not measure belt speed from the kinematics, using the "
          f"configured {com_belt_speed_ms:.2f} m/s")

# -----------------------------------------------------------------------------
# %% the complementary fusion
# -----------------------------------------------------------------------------

def com_fuse_velocity(position_kin, accel_force, crossover_hz=None,
                      fs=None, order=None):
    """Complementary blend of a differentiated position and an integrated force.

    Returns velocity in the same frame as `position_kin`. The two branches are
    filtered with the SAME cutoff and summed, so their transfer functions add
    to one at every frequency: nothing is counted twice and nothing is lost.
    """
    fs = kin_kinematic_fs if fs is None else fs
    crossover_hz = com_crossover_hz if crossover_hz is None else crossover_hz
    order = com_filter_order if order is None else order

    com_n = min(len(position_kin), len(accel_force))
    com_p, com_acc = position_kin[:com_n], accel_force[:com_n]

    com_v_kin = np.gradient(com_p, 1.0 / fs, axis=0)
    # Trapezoidal, not np.cumsum. A rectangle-rule integral lands half a sample
    # late, which at stride harmonics costs about 10 mm/s of velocity error --
    # invisible next to noisy markers but pure loss when they are clean. The
    # trapezoid is centred on the sample grid and cuts that to about 0.2 mm/s.
    com_v_force = np.zeros_like(com_acc)
    com_v_force[1:] = np.cumsum((com_acc[:-1] + com_acc[1:]) / 2, axis=0) / fs
    com_v_force -= com_v_force.mean(axis=0)

    com_b, com_a_coef = butter(order, crossover_hz / (fs / 2), 'low')
    com_low = filtfilt(com_b, com_a_coef, com_v_kin, axis=0)
    com_force_low = filtfilt(com_b, com_a_coef, com_v_force, axis=0)
    return com_low + (com_v_force - com_force_low)

if com_have_kinematics:
    com_n_use = min(len(com_pos_kin), len(com_accel_force))
    com_velocity_kin = np.gradient(com_pos_kin[:com_n_use], 1.0 / kin_kinematic_fs, axis=0)
    com_velocity = com_fuse_velocity(com_pos_kin, com_accel_force)
    com_position = com_pos_kin[:com_n_use]

    # In the belt frame the walker is travelling forward at belt speed even
    # though the lab-frame mean is ~0. Every AP margin needs this.
    com_velocity_belt = com_velocity.copy()
    com_velocity_belt[:, com_ap_axis] += com_belt_sign * com_belt_speed_used

    print(f"  fused at {com_crossover_hz:.2f} Hz crossover")
    print(f"    lab-frame AP velocity: kinematic mean {com_velocity_kin[:, com_ap_axis].mean():+.3f}, "
          f"fused {com_velocity[:, com_ap_axis].mean():+.3f} m/s (both ~0 on a treadmill)")
    print(f"    belt-frame AP velocity mean {com_velocity_belt[:, com_ap_axis].mean():+.3f} m/s")

    # How much did the fusion actually change things? Reported as the RMS
    # difference from the kinematics-only velocity, per axis.
    com_delta = com_velocity - com_velocity_kin
    print(f"    RMS change vs differentiating the markers alone: "
          f"ML {1000*com_delta[:, 0].std():.1f}, AP {1000*com_delta[:, 1].std():.1f}, "
          f"VT {1000*com_delta[:, 2].std():.1f} mm/s")

    print("\n  crossover sensitivity (AP velocity RMS difference from the 0.5 Hz result)")
    for com_fc in com_cutoff_sweep:
        com_alt = com_fuse_velocity(com_pos_kin, com_accel_force, crossover_hz=com_fc)
        com_d = (com_alt - com_velocity)[:, com_ap_axis]
        print(f"    {com_fc:4.2f} Hz: {1000*com_d.std():6.1f} mm/s")

    if handrail_contact_100hz[:com_n_use].any():
        print(f"  ! {handrail_contact_100hz[:com_n_use].sum()} frames have handrail "
              f"contact; the GRF is not the only external force there and the "
              f"fused velocity is unreliable on those frames.")
else:
    com_velocity = com_velocity_belt = com_position = None

# =============================================================================
# %% Margin of stability, with the base of support taken from the foot meshes
# =============================================================================
# [Hof2005]:  XCoM = x + v / omega_0,   omega_0 = sqrt(g / l),   MoS = u_max - XCoM
#
# The CoM is not just somewhere, it is going somewhere. If active control
# stopped now, inverted pendulum dynamics would carry the body to the
# extrapolated CoM. MoS is how much base of support is left beyond that point.
#
# What changes here is u_max. It is normally the ankle joint centre or a toe
# marker, both of which are points INSIDE the foot, several centimetres from
# the border that would actually stop you falling. The posed foot meshes give
# the real boundary, and both are computed so the difference is quantified
# rather than assumed away.
#
# Sign convention, chosen so it does not depend on which way the lab axes point:
#   ML  u_max is the most LATERAL vertex of the stance foot, where lateral
#       means away from the other foot. Positive MoS = XCoM inside the foot.
#   AP  u_max is the most ANTERIOR vertex, anterior being the direction of
#       travel, which is measured from the belt not assumed.
#
# Two readings are reported per step, because they answer different questions:
#   at foot contact      the conventional value, how much margin the step bought
#   minimum over stance  the worst moment within the step [McAndrewYoung2012]
#
# AP MoS is routinely NEGATIVE in walking and that is not a pathology. Walking
# is controlled falling forward; a negative anterior margin is what propulsion
# looks like. Only the ML margin reads as "how close to falling sideways".
#
# References
#   [Hof2005]             Hof, Gazendam & Sinke (2005) J Biomech 38, 1-8.
#   [Hof2008]             Hof (2008) Hum Mov Sci 27(1), 112-125.
#   [McAndrewYoung2012]   McAndrew Young & Dingwell (2012) Gait Posture 36(2), 219-224.
#   [Hak2013]             Hak et al. (2013) PLoS One 8(12), e82842.
# =============================================================================

# -----------------------------------------------------------------------------
# %% margin of stability settings
# -----------------------------------------------------------------------------

mos_gravity          = 9.81
mos_pendulum_mode    = 'per_frame'   # 'per_frame' CoM-to-ankle, or 'leg_length'
mos_mesh_pickle_suffix = '_mesh.pkl' # written next to the metrics csv by apply_binding
mos_stance_fraction  = (0.0, 1.0)    # portion of stance searched for the minimum
mos_report_marker_bos = True         # also compute the marker-based boundary

# -----------------------------------------------------------------------------
# %% load the posed foot meshes for this trial
# -----------------------------------------------------------------------------
# apply_binding.py writes <TRIAL>_mesh.pkl beside the metrics csv: a DataFrame
# indexed by frame with a (segment, vertex, axis) column MultiIndex, in Theia
# world metres. If it is missing the cell still runs, on markers alone, and
# says so rather than silently changing what it measures.

mos_mesh_path = metrics_path.with_name(metrics_path.stem + mos_mesh_pickle_suffix)

print("\n" + "-" * 74)
print("  MARGIN OF STABILITY")
print("-" * 74)

if mos_mesh_path.exists():
    mesh_positions = pd.read_pickle(mos_mesh_path)
    mos_have_mesh = True
    print(f"  mesh: {mos_mesh_path.name}  {mesh_positions.shape[0]} frames, "
          f"{mesh_positions.shape[1] // 3} vertices")
else:
    mesh_positions = None
    mos_have_mesh = False
    print(f"  ! {mos_mesh_path.name} not found. Falling back to marker-based")
    print(f"    boundaries, which sit inside the foot and will overstate the")
    print(f"    margin. Run apply_binding.py on this trial to fix that.")

# Per-foot vertex blocks, as plain arrays (frames, vertices, 3), so the
# per-frame extreme is one vectorised reduction rather than a pandas lookup.
mos_foot_vertices = {}
if mos_have_mesh:
    for mos_seg, mos_side in (('left_foot', 'Left'), ('right_foot', 'Right')):
        if mos_seg in mesh_positions.columns.get_level_values('segment'):
            mos_block = mesh_positions[mos_seg]
            mos_n_vert = mos_block.shape[1] // 3
            mos_foot_vertices[mos_side] = (mos_block.to_numpy()
                                           .reshape(len(mos_block), mos_n_vert, 3))
            print(f"    {mos_side}: {mos_n_vert} vertices")

# -----------------------------------------------------------------------------
# %% pull the joint positions the cell needs into plain arrays
# -----------------------------------------------------------------------------

kinematic_positions = {}
for mos_joint in ('Ankle', 'Toes', 'Heel', 'Hip', 'Knee'):
    for mos_side in ('Left', 'Right'):
        mos_col = f'{mos_side}_{mos_joint}_Position'
        if mos_col in kinematic_data.columns:
            kinematic_positions[f'{mos_side}_{mos_joint}'] = np.column_stack([
                kinematic_data[mos_col].to_numpy(),
                kinematic_data[f'{mos_col}.1'].to_numpy(),
                kinematic_data[f'{mos_col}.2'].to_numpy()])

# A constant fallback pendulum length, for the sensitivity comparison
mos_leg_length = np.nan
if 'Left_Hip' in kinematic_positions and 'Left_Ankle' in kinematic_positions:
    mos_leg_length = float(np.nanmedian(np.linalg.norm(
        kinematic_positions['Left_Hip'] - kinematic_positions['Left_Ankle'], axis=1)))
    print(f"  leg length (hip to ankle, median): {mos_leg_length:.3f} m")

# -----------------------------------------------------------------------------
# %% geometry helpers
# -----------------------------------------------------------------------------

def mos_lateral_unit(stance_side, frame):
    """Unit vector in the ground plane pointing laterally for the stance foot.

    Defined as the direction from the contralateral foot toward the stance
    foot, so it does not depend on whether lab +X points left or right, and it
    follows the walker if the treadmill heading drifts.
    """
    mos_other = 'Right' if stance_side == 'Left' else 'Left'
    mos_a = kinematic_positions[f'{mos_other}_Ankle'][frame]
    mos_b = kinematic_positions[f'{stance_side}_Ankle'][frame]
    mos_v = np.array([mos_b[com_ml_axis] - mos_a[com_ml_axis],
                      mos_b[com_ap_axis] - mos_a[com_ap_axis]])
    mos_n = np.linalg.norm(mos_v)
    if not np.isfinite(mos_n) or mos_n < 1e-6:
        return np.array([1.0, 0.0])
    return mos_v / mos_n


def mos_boundary(stance_side, frame, direction_2d):
    """Furthest extent of the stance foot along `direction_2d`, in metres.

    Uses the mesh when it is available -- the real surface boundary -- and the
    ankle and toe markers otherwise, which are interior points.
    """
    if mos_have_mesh and stance_side in mos_foot_vertices:
        mos_v = mos_foot_vertices[stance_side]
        if frame < len(mos_v):
            mos_pts = mos_v[frame][:, [com_ml_axis, com_ap_axis]]
            if np.isfinite(mos_pts).all():
                return float(np.max(mos_pts @ direction_2d))
    mos_cands = []
    for mos_name in ('Ankle', 'Toes', 'Heel'):
        mos_key = f'{stance_side}_{mos_name}'
        if mos_key in kinematic_positions:
            mos_p = kinematic_positions[mos_key][frame]
            if np.isfinite(mos_p).all():
                mos_cands.append([mos_p[com_ml_axis], mos_p[com_ap_axis]])
    if not mos_cands:
        return np.nan
    return float(np.max(np.array(mos_cands) @ direction_2d))


def mos_pendulum_length(stance_side, frame):
    """Effective pendulum length: CoM height above the stance ankle."""
    if mos_pendulum_mode == 'leg_length' and np.isfinite(mos_leg_length):
        return mos_leg_length
    mos_ank = kinematic_positions[f'{stance_side}_Ankle'][frame]
    if not np.isfinite(mos_ank).all() or frame >= len(com_position):
        return np.nan
    mos_l = np.linalg.norm(com_position[frame] - mos_ank)
    return mos_l if mos_l > 0.2 else np.nan

# -----------------------------------------------------------------------------
# %% the margin, per step
# -----------------------------------------------------------------------------

mos_rows = []
for mos_side in ('Left', 'Right'):
    for mos_hs_sample, mos_to_sample in force_contacts[mos_side]:
        mos_hs = int(kin_sample_to_frame(mos_hs_sample)) - 1
        mos_to = int(kin_sample_to_frame(mos_to_sample)) - 1
        if mos_hs < 0 or mos_to >= len(com_velocity_belt) or mos_to <= mos_hs:
            continue

        mos_a = mos_hs + int(mos_stance_fraction[0] * (mos_to - mos_hs))
        mos_b = mos_hs + int(mos_stance_fraction[1] * (mos_to - mos_hs))
        mos_ml_series, mos_ap_series = [], []
        mos_ml_marker_series = []

        for mos_f in range(mos_a, min(mos_b + 1, len(com_position))):
            mos_l = mos_pendulum_length(mos_side, mos_f)
            if not np.isfinite(mos_l):
                mos_ml_series.append(np.nan); mos_ap_series.append(np.nan)
                mos_ml_marker_series.append(np.nan); continue
            mos_w0 = np.sqrt(mos_gravity / mos_l)

            # extrapolated CoM in the ground plane, belt-corrected in AP
            mos_xcom = np.array([
                com_position[mos_f, com_ml_axis]
                + com_velocity_belt[mos_f, com_ml_axis] / mos_w0,
                com_position[mos_f, com_ap_axis]
                + com_velocity_belt[mos_f, com_ap_axis] / mos_w0])

            mos_lat = mos_lateral_unit(mos_side, mos_f)
            mos_ml_series.append(mos_boundary(mos_side, mos_f, mos_lat)
                                 - float(mos_xcom @ mos_lat))

            mos_fwd = np.array([0.0, com_belt_sign])
            mos_ap_series.append(mos_boundary(mos_side, mos_f, mos_fwd)
                                 - float(mos_xcom @ mos_fwd))

            if mos_report_marker_bos and mos_have_mesh:
                mos_ank = kinematic_positions[f'{mos_side}_Ankle'][mos_f]
                mos_ml_marker_series.append(
                    float(np.array([mos_ank[com_ml_axis],
                                    mos_ank[com_ap_axis]]) @ mos_lat)
                    - float(mos_xcom @ mos_lat))

        mos_ml_series = np.array(mos_ml_series, float)
        mos_ap_series = np.array(mos_ap_series, float)
        if not np.isfinite(mos_ml_series).any():
            continue

        mos_rows.append({
            'side': mos_side,
            'heel_strike_frame_100hz': mos_hs + 1,
            'toe_off_frame_100hz': mos_to + 1,
            'mos_ml_contact': mos_ml_series[0],
            'mos_ml_min': np.nanmin(mos_ml_series),
            'mos_ap_contact': mos_ap_series[0],
            'mos_ap_min': np.nanmin(mos_ap_series),
            'mos_ml_contact_marker_bos': (np.array(mos_ml_marker_series)[0]
                                          if mos_ml_marker_series else np.nan),
            'pendulum_length': mos_pendulum_length(mos_side, mos_hs),
        })

margin_of_stability = pd.DataFrame(mos_rows)

if len(margin_of_stability):
    print(f"\n  {len(margin_of_stability)} steps")
    for mos_side, mos_grp in margin_of_stability.groupby('side'):
        print(f"    {mos_side:5s}  ML at contact {1000*mos_grp['mos_ml_contact'].mean():6.1f} "
              f"+- {1000*mos_grp['mos_ml_contact'].std():5.1f} mm   "
              f"ML min over stance {1000*mos_grp['mos_ml_min'].mean():6.1f} mm")
        print(f"           AP at contact {1000*mos_grp['mos_ap_contact'].mean():6.1f} "
              f"+- {1000*mos_grp['mos_ap_contact'].std():5.1f} mm   "
              f"(negative is normal, walking falls forward)")

    if mos_have_mesh and margin_of_stability['mos_ml_contact_marker_bos'].notna().any():
        mos_diff = 1000 * (margin_of_stability['mos_ml_contact']
                           - margin_of_stability['mos_ml_contact_marker_bos'])
        print(f"\n  mesh boundary vs ankle joint centre: the mesh gives a margin "
              f"{mos_diff.mean():.1f} +- {mos_diff.std():.1f} mm larger")
        print(f"    That gap is the distance from the ankle joint centre out to the")
        print(f"    real lateral border of the shoe. Against a margin of "
              f"{1000*margin_of_stability['mos_ml_contact'].mean():.0f} mm it is not a "
              f"rounding difference.")
else:
    print("  ! no steps produced a margin, check the CoM and contact inputs")

# =============================================================================
# %% Minimum foot clearance, margin of instability and the trip risk integral
# =============================================================================
# [Schulz2017]. Minimum foot clearance says how close the foot came to the
# ground; it does not say what would have happened if the foot had caught. Trip
# risk needs both, because a low clearance in a mechanically stable
# configuration is recoverable and the same clearance in an unstable one is not.
#
# MINIMUM FOOT CLEARANCE
# Schulz digitised the soles of the shoes and took the minimum distance over
# ALL digitised points. The posed mesh is the direct equivalent, and it matters:
# the heel and toe joint centres used previously are interior points, so they
# measure the clearance of somewhere inside the foot rather than of the sole.
#
# A valid MFC event is operationally defined [Schulz2011, Schulz2017] as
#   1. a local minimum, lower than the preceding and following two frames
#   2. foot speed in the upper quartile for that swing, which rejects the
#      spurious minima just after foot off
#   3. heel clearance not smaller than toe clearance, which rejects false
#      detections at midfoot where the two segments meet
# If more than one candidate qualifies, the smaller is used. Swings with NO
# qualifying local minimum return NaN rather than a global minimum -- those
# non-MTC cycles are real and Schulz devotes a figure to them.
#
# MARGIN OF INSTABILITY
# The AP margin of stability, with three modifications:
#   1. the CONTINUOUS trajectory through swing, not the stance minimum
#   2. u_max is the most anterior toe point on EITHER foot -- the stance toe in
#      early swing, the swing toe in late swing -- because that is where the
#      anterior boundary of the base of support would be if the swing foot
#      came down
#   3. stable (positive) values are clamped to zero and the result negated, so
#      MoI >= 0 and larger means more destabilising
#
#       MoI(t) = max( -MoS_AP(t), 0 )
#
# TRIP RISK INTEGRAL
#       trip risk(t) = MoI(t) / MFC(t)          both in mm, dimensionless
#       TRI = integral of trip risk over the swing phase
#
# The bounds are peak acceleration to peak deceleration of the MFC point, which
# excludes the spikes at lift-off and landing where the foot is intentionally
# close to the ground. Time is deliberately NOT normalised, so a longer swing
# gives a larger TRI; that is Schulz's choice and it is kept for comparability.
#
# Direction of the effect: higher MFC means LESS risk, higher TRI means MORE.
# Schulz's central result is that the two move oppositely with gait speed, and
# that TRI is the one that tracks real trip-fall risk.
#
# References
#   [Schulz2011]  Schulz (2011) J Biomech 44, 1277-1284.
#   [Schulz2017]  Schulz (2017) J Biomech 55, 107-112.
#   [Winter1992]  Winter (1992) Phys Ther 72, 45-46.
#   [Hof2005]     Hof, Gazendam & Sinke (2005) J Biomech 38, 1-8.
# =============================================================================

# -----------------------------------------------------------------------------
# %% trip risk settings
# -----------------------------------------------------------------------------

tri_floor_height    = 0.0     # treadmill belt, flat, z = 0 in the Theia frame
tri_speed_quantile  = 0.75    # "upper quartile" foot speed gate
tri_local_window    = 2       # frames either side that a local minimum must beat
tri_min_clearance_mm = 1.0    # guard: MoI/MFC blows up as MFC approaches zero
tri_swing_trim      = 0.02    # fraction of swing trimmed at each end before
                              # searching, removes the contact frames themselves

# -----------------------------------------------------------------------------
# %% per-frame foot geometry from the mesh
# -----------------------------------------------------------------------------
# Three quantities per foot per frame: the lowest point above the floor, which
# vertex that was, and the most anterior extent. All are single reductions over
# the vertex axis.

print("\n" + "-" * 74)
print("  MINIMUM FOOT CLEARANCE AND TRIP RISK")
print("-" * 74)

foot_clearance_mesh = {}
foot_lowest_vertex = {}
foot_anterior_mesh = {}

if mos_have_mesh:
    for tri_side, tri_verts in mos_foot_vertices.items():
        tri_height = tri_verts[:, :, com_vt_axis] - tri_floor_height
        foot_clearance_mesh[tri_side] = np.nanmin(tri_height, axis=1)
        foot_lowest_vertex[tri_side] = np.nanargmin(
            np.where(np.isfinite(tri_height), tri_height, np.inf), axis=1)
        foot_anterior_mesh[tri_side] = np.nanmax(
            com_belt_sign * tri_verts[:, :, com_ap_axis], axis=1)
        print(f"  {tri_side}: clearance {1000*np.nanmin(foot_clearance_mesh[tri_side]):.1f} "
              f"to {1000*np.nanmax(foot_clearance_mesh[tri_side]):.0f} mm over the trial")
else:
    print("  ! no mesh, MFC falls back to the heel and toe joint centres, which")
    print("    are interior points and overstate the clearance by the distance")
    print("    from the joint centre down to the sole.")
    for tri_side in ('Left', 'Right'):
        tri_stack = [kinematic_positions[f'{tri_side}_{j}'][:, com_vt_axis]
                     for j in ('Heel', 'Toes') if f'{tri_side}_{j}' in kinematic_positions]
        if tri_stack:
            foot_clearance_mesh[tri_side] = np.nanmin(np.vstack(tri_stack), axis=0) - tri_floor_height
            foot_anterior_mesh[tri_side] = com_belt_sign * kinematic_positions[
                f'{tri_side}_Toes'][:, com_ap_axis]

# Anterior boundary of the base of support: the furthest forward point on
# EITHER foot, per Schulz's definition.
tri_n_frames = min(len(v) for v in foot_anterior_mesh.values())
bos_anterior = np.nanmax(np.vstack([foot_anterior_mesh[s][:tri_n_frames]
                                    for s in foot_anterior_mesh]), axis=0)

# -----------------------------------------------------------------------------
# %% the margin of instability trajectory
# -----------------------------------------------------------------------------

def tri_margin_of_instability(frames):
    """MoI in mm over `frames`: the AP margin, clamped at zero and negated."""
    tri_out = np.full(len(frames), np.nan)
    for tri_i, tri_f in enumerate(frames):
        if tri_f >= len(com_position) or tri_f >= tri_n_frames:
            continue
        tri_l = np.nan
        for tri_side in ('Left', 'Right'):
            tri_l = mos_pendulum_length(tri_side, tri_f)
            if np.isfinite(tri_l):
                break
        if not np.isfinite(tri_l):
            continue
        tri_w0 = np.sqrt(mos_gravity / tri_l)
        tri_xcom = (com_belt_sign * com_position[tri_f, com_ap_axis]
                    + com_belt_sign * com_velocity_belt[tri_f, com_ap_axis] / tri_w0)
        tri_mos = bos_anterior[tri_f] - tri_xcom
        tri_out[tri_i] = 1000.0 * max(-tri_mos, 0.0)
    return tri_out

# -----------------------------------------------------------------------------
# %% per swing: MFC event, MoI, trip risk and its integral
# -----------------------------------------------------------------------------

tri_rows = []
for tri_side in ('Left', 'Right'):
    tri_contacts = force_contacts[tri_side]
    for tri_k in range(len(tri_contacts) - 1):
        tri_to_frame = int(kin_sample_to_frame(tri_contacts[tri_k][1])) - 1
        tri_hs_frame = int(kin_sample_to_frame(tri_contacts[tri_k + 1][0])) - 1
        if tri_to_frame < 0 or tri_hs_frame >= tri_n_frames or tri_hs_frame <= tri_to_frame:
            continue

        tri_pad = int(tri_swing_trim * (tri_hs_frame - tri_to_frame))
        tri_frames = np.arange(tri_to_frame + tri_pad, tri_hs_frame - tri_pad + 1)
        if len(tri_frames) < 10:
            continue

        tri_clear_m = foot_clearance_mesh[tri_side][tri_frames]
        if not np.isfinite(tri_clear_m).any():
            continue

        # --- foot speed, for the upper-quartile gate --------------------------
        if mos_have_mesh and tri_side in mos_foot_vertices:
            tri_centroid = np.nanmean(mos_foot_vertices[tri_side][tri_frames], axis=1)
        else:
            tri_centroid = kinematic_positions[f'{tri_side}_Toes'][tri_frames]
        tri_speed = np.linalg.norm(
            np.gradient(tri_centroid, 1.0 / kin_kinematic_fs, axis=0), axis=1)
        tri_fast = tri_speed >= np.nanquantile(tri_speed, tri_speed_quantile)

        # --- heel clearance, for the midfoot rejection ------------------------
        tri_heel_key = f'{tri_side}_Heel'
        if tri_heel_key in kinematic_positions:
            tri_heel_clear = (kinematic_positions[tri_heel_key][tri_frames, com_vt_axis]
                              - tri_floor_height)
        else:
            tri_heel_clear = np.full(len(tri_frames), np.inf)

        # --- local minima that satisfy all three criteria ---------------------
        tri_candidates = []
        for tri_i in range(tri_local_window, len(tri_clear_m) - tri_local_window):
            tri_v = tri_clear_m[tri_i]
            if not np.isfinite(tri_v):
                continue
            tri_before = tri_clear_m[tri_i - tri_local_window:tri_i]
            tri_after = tri_clear_m[tri_i + 1:tri_i + tri_local_window + 1]
            # <= rather than <, because a minimum that falls between two
            # samples gives two equal neighbouring values and a strict test
            # then finds nothing at all. Requiring a strict decrease on at
            # least one side still rules out a flat run being called a
            # minimum, and adjacent duplicates are collapsed below.
            if not (np.all(tri_v <= tri_before) and np.all(tri_v <= tri_after)):
                continue
            if not (np.any(tri_v < tri_before) and np.any(tri_v < tri_after)):
                continue
            if not tri_fast[tri_i]:
                continue
            if tri_heel_clear[tri_i] < tri_v:
                continue
            tri_candidates.append(tri_i)

        # collapse runs of adjacent indices (a flat minimum) to their first
        tri_candidates = [c for j, c in enumerate(tri_candidates)
                          if j == 0 or c != tri_candidates[j - 1] + 1]

        tri_has_event = len(tri_candidates) > 0
        if tri_has_event:
            tri_idx = min(tri_candidates, key=lambda i: tri_clear_m[i])
            tri_mfc_m = float(tri_clear_m[tri_idx])
            tri_mfc_frame = int(tri_frames[tri_idx])
            tri_mfc_vertex = (int(foot_lowest_vertex[tri_side][tri_mfc_frame])
                              if mos_have_mesh else -1)
        else:
            tri_idx = tri_mfc_frame = tri_mfc_vertex = -1
            tri_mfc_m = np.nan

        # --- the MFC point's own kinematics, which set the integration bounds -
        # Schulz tracks the single point identified as the MFC point, not
        # whichever vertex happens to be lowest at each instant; tracking the
        # latter would make the velocity jump whenever the lowest point moved.
        tri_t1 = tri_t2 = np.nan
        if tri_has_event and mos_have_mesh and tri_side in mos_foot_vertices:
            tri_point = mos_foot_vertices[tri_side][tri_frames, tri_mfc_vertex, :]
            tri_pt_vel = np.gradient(tri_point, 1.0 / kin_kinematic_fs, axis=0)
            tri_pt_speed = np.linalg.norm(tri_pt_vel, axis=1)
            tri_pt_acc = np.gradient(tri_pt_speed, 1.0 / kin_kinematic_fs)
            tri_a = int(np.nanargmax(tri_pt_acc))          # peak acceleration
            tri_b = int(np.nanargmin(tri_pt_acc))          # peak deceleration
            if tri_b > tri_a:
                tri_t1, tri_t2 = tri_a, tri_b

        # --- MoI, trip risk, TRI ---------------------------------------------
        tri_moi = tri_margin_of_instability(tri_frames)
        tri_clear_mm = np.maximum(1000.0 * tri_clear_m, tri_min_clearance_mm)
        tri_risk = tri_moi / tri_clear_mm

        tri_integral = np.nan
        if np.isfinite(tri_t1) and np.isfinite(tri_t2):
            tri_seg = tri_risk[int(tri_t1):int(tri_t2) + 1]
            if np.isfinite(tri_seg).any():
                tri_integral = float(np.nansum(tri_seg) / kin_kinematic_fs)

        tri_rows.append({
            'side': tri_side,
            'toe_off_frame_100hz': tri_to_frame + 1,
            'next_heel_strike_frame_100hz': tri_hs_frame + 1,
            'swing_time': (tri_hs_frame - tri_to_frame) / kin_kinematic_fs,
            'mfc_m': tri_mfc_m,
            'mfc_frame_100hz': tri_mfc_frame + 1 if tri_has_event else np.nan,
            'mfc_vertex': tri_mfc_vertex if tri_has_event else np.nan,
            'n_local_minima': len(tri_candidates),
            'has_mfc_event': tri_has_event,
            'moi_peak_mm': np.nanmax(tri_moi) if np.isfinite(tri_moi).any() else np.nan,
            'moi_mean_mm': np.nanmean(tri_moi) if np.isfinite(tri_moi).any() else np.nan,
            'trip_risk_peak': np.nanmax(tri_risk) if np.isfinite(tri_risk).any() else np.nan,
            'trip_risk_integral': tri_integral,
            'integration_window_s': ((tri_t2 - tri_t1) / kin_kinematic_fs
                                     if np.isfinite(tri_t1) else np.nan),
        })

trip_risk = pd.DataFrame(tri_rows)

# -----------------------------------------------------------------------------
# %% report
# -----------------------------------------------------------------------------

if len(trip_risk):
    print(f"\n  {len(trip_risk)} swings")
    for tri_side, tri_grp in trip_risk.groupby('side'):
        tri_with = tri_grp[tri_grp['has_mfc_event']]
        print(f"    {tri_side:5s}  MFC {1000*tri_with['mfc_m'].mean():5.1f} "
              f"+- {1000*tri_with['mfc_m'].std():4.1f} mm  "
              f"(5th pct {1000*tri_with['mfc_m'].quantile(0.05):5.1f} mm)")
        print(f"           MoI peak {tri_grp['moi_peak_mm'].mean():6.1f} mm   "
              f"TRI {tri_grp['trip_risk_integral'].mean():.4f} "
              f"+- {tri_grp['trip_risk_integral'].std():.4f} s")
        print(f"           {len(tri_grp) - len(tri_with)} of {len(tri_grp)} swings had "
              f"no qualifying MFC event, "
              f"{(tri_grp['n_local_minima'] > 1).sum()} had more than one")

    # For trip risk the tail matters more than the mean: one abnormally low
    # clearance is what catches an obstacle.
    tri_all = trip_risk[trip_risk['has_mfc_event']]['mfc_m']
    if len(tri_all) > 10:
        print(f"\n  MFC distribution over all swings: median {1000*tri_all.median():.1f} mm, "
              f"IQR {1000*tri_all.quantile(0.25):.1f}-{1000*tri_all.quantile(0.75):.1f}, "
              f"min {1000*tri_all.min():.1f} mm")

    if mos_have_mesh and trip_risk['mfc_vertex'].notna().any():
        tri_vc = trip_risk['mfc_vertex'].dropna().astype(int).value_counts()
        print(f"  the lowest point was one of {len(tri_vc)} distinct vertices; the most "
              f"frequent accounted for {100*tri_vc.iloc[0]/tri_vc.sum():.0f}% of swings")
        print(f"    A single dominant vertex means the foot presents the same point")
        print(f"    every stride. A spread means it moves, which is Schulz's argument")
        print(f"    against treating MFC as one fixed point on the shoe.")
else:
    print("  ! no swings produced a trip risk value")

# =============================================================================
# %% Kinetic metrics: impulses, GRF descriptors, centre of pressure
# =============================================================================
# The metric family that only exists because the treadmill is instrumented.
# Everything here is per limb, which the split belts give directly with no
# plate-hit problem to solve.
#
# CONVENTIONS, ESTABLISHED FROM THE DATA RATHER THAN ASSUMED
# The moment columns are in N*m about each plate's own origin, and the CoP
# columns are already in a shared lab frame offset from those origins. That is
# not documented anywhere in the export, so it is derived below: regressing the
# reported CoP against -My/Fz and Mx/Fz recovers the offsets with r = 1.0000
# and reconstructs the CoP to 0.0000 mm. On this hardware the belt centres come
# out 559 mm apart, which is the geometry you would expect.
#
# Getting this wrong matters. The free moment is the vertical torque that is
# NOT explained by the horizontal forces acting at the CoP, so it needs the CoP
# expressed relative to the moment origin. Use the lab-frame CoP by mistake and
# the free moment picks up a spurious term of tens of N*m, against a real
# signal of about 5.
#
# WHAT EACH MEASURE IS FOR
#   braking / propulsive impulse   the AP force integrated over the part of
#                                  stance where it opposes / drives travel.
#                                  Reduced propulsion is one of the better
#                                  established ageing markers, and impulse
#                                  symmetry is a far better asymmetry measure
#                                  than any kinematic one.
#   F1, trough, F2                 weight acceptance peak, mid-stance unloading
#                                  and push-off peak of the vertical GRF.
#   loading rate                   how fast load is accepted, 20-80% of F1.
#   CoP path                       how the point of application travels. The ML
#                                  excursion is clean; the AP excursion on a
#                                  treadmill also contains belt travel, because
#                                  the foot rides the belt while the CoP
#                                  progresses along the foot, so it is reported
#                                  with that caveat attached.
#   free moment                    transverse-plane torque against the ground,
#                                  related to rotational control and to
#                                  tibial loading.
# =============================================================================

from scipy.signal import butter, filtfilt   # repeated so the cell runs alone

# -----------------------------------------------------------------------------
# %% kinetic metric settings
# -----------------------------------------------------------------------------

kmx_loading_band    = (0.20, 0.80)   # fraction of F1 used for the loading rate
kmx_peak_search     = (0.05, 0.95)   # fraction of stance searched for F1 / F2
kmx_loaded_n        = 200.0          # N, "solidly loaded" for the CoP calibration
kmx_min_stance_frames = 50           # at 1000 Hz

# The CoP must be filtered before anything is derived from it. At 1000 Hz the
# sample-to-sample CoP step on this hardware is about 2.5 mm of noise, so a
# path length accumulates roughly 1800 mm over a stance in which the CoP
# actually moved 3 mm -- the raw path length is 99.8% noise and means nothing.
# Range is robust to this, path length is not, so path length is computed from
# the filtered signal and the noise level is reported so the choice is visible.
kmx_cop_filter_hz   = 15.0
kmx_force_filter_hz = 50.0           # for derived metrics only, NOT for events

print("\n" + "-" * 74)
print("  KINETIC METRICS")
print("-" * 74)

# -----------------------------------------------------------------------------
# %% derive the plate origin offsets, then the free moment
# -----------------------------------------------------------------------------
# copX = -My/Fz + offset_x and copY = Mx/Fz + offset_y, so a regression of the
# reported CoP on those ratios recovers each plate's offset. Solved per belt
# because the two plates sit either side of the midline.

def kmx_lowpass(signal, cutoff_hz, fs=None, order=4):
    """Zero-phase Butterworth. Used for derived metrics only; gait events are
    detected on the raw signal, where the thresholds were tuned."""
    fs = kin_fs if fs is None else fs
    kmx_b, kmx_a = butter(order, cutoff_hz / (fs / 2), 'low')
    return filtfilt(kmx_b, kmx_a, signal)

cop_filtered = {}
for kmx_side in ('Left', 'Right'):
    cop_filtered[kmx_side] = {
        a: kmx_lowpass(force_data[f'{kmx_side} belt_COP_{a}'].to_numpy(),
                       kmx_cop_filter_hz) for a in 'XY'}
    kmx_raw = np.column_stack([force_data[f'{kmx_side} belt_COP_{a}'].to_numpy()
                               for a in 'XY'])
    kmx_fil = np.column_stack([cop_filtered[kmx_side][a] for a in 'XY'])
    kmx_loaded = force_data[f'{kmx_side} belt_Force_Z'].to_numpy() > kmx_loaded_n
    kmx_step_raw = np.median(np.linalg.norm(np.diff(kmx_raw[kmx_loaded], axis=0), axis=1))
    kmx_step_fil = np.median(np.linalg.norm(np.diff(kmx_fil[kmx_loaded], axis=0), axis=1))
    print(f"  {kmx_side} CoP sample step when loaded: {kmx_step_raw:.3f} mm raw, "
          f"{kmx_step_fil:.3f} mm after a {kmx_cop_filter_hz:.0f} Hz lowpass")

kmx_cop_offset = {}
free_moment = {}
for kmx_side in ('Left', 'Right'):
    kmx_f = {a: force_data[f'{kmx_side} belt_Force_{a}'].to_numpy() for a in 'XYZ'}
    kmx_m = {a: force_data[f'{kmx_side} belt_Moment_{a}'].to_numpy() for a in 'XYZ'}
    # The offset is a geometric constant of the hardware, so it is calibrated
    # against the RAW CoP -- the filtered signal would make the check depend on
    # the filter choice rather than on the moment convention.
    kmx_raw_c = {a: force_data[f'{kmx_side} belt_COP_{a}'].to_numpy() for a in 'XY'}
    kmx_c = {a: cop_filtered[kmx_side][a] for a in 'XY'}

    kmx_ok = kmx_f['Z'] > kmx_loaded_n
    kmx_ox = float(np.median(kmx_raw_c['X'][kmx_ok]
                             + 1000.0 * kmx_m['Y'][kmx_ok] / kmx_f['Z'][kmx_ok]))
    kmx_oy = float(np.median(kmx_raw_c['Y'][kmx_ok]
                             - 1000.0 * kmx_m['X'][kmx_ok] / kmx_f['Z'][kmx_ok]))
    kmx_cop_offset[kmx_side] = (kmx_ox, kmx_oy)

    # check the offsets actually reproduce the reported CoP
    kmx_pred_x = -1000.0 * kmx_m['Y'][kmx_ok] / kmx_f['Z'][kmx_ok] + kmx_ox
    kmx_resid = float(np.max(np.abs(kmx_pred_x - kmx_raw_c['X'][kmx_ok])))
    print(f"  {kmx_side} plate origin offset ({kmx_ox:+.1f}, {kmx_oy:+.1f}) mm, "
          f"CoP reconstruction residual {kmx_resid:.4f} mm")
    if kmx_resid > 1.0:
        print(f"    ! residual over 1 mm means the moment convention is not what")
        print(f"      this assumes. Free moments below will be wrong; check the export.")

    # free moment: the vertical torque not explained by the horizontal forces
    # acting at the CoP, with the CoP taken relative to the moment origin
    kmx_rx = (kmx_c['X'] - kmx_ox) / 1000.0
    kmx_ry = (kmx_c['Y'] - kmx_oy) / 1000.0
    free_moment[kmx_side] = kmx_m['Z'] - (kmx_rx * kmx_f['Y'] - kmx_ry * kmx_f['X'])

kmx_belt_separation = abs(kmx_cop_offset['Left'][0] - kmx_cop_offset['Right'][0])
print(f"  belt centres {kmx_belt_separation:.0f} mm apart")

# -----------------------------------------------------------------------------
# %% which AP sign is braking?
# -----------------------------------------------------------------------------
# Travel direction is known from the belt, so braking is the AP force opposing
# it. Rather than trust that, the pattern is checked: the AP force should be
# predominantly braking in the first half of stance and propulsive in the
# second. If it is not, the axis or sign assumption is wrong.

kmx_ap_sign_check = []
for kmx_side in ('Left', 'Right'):
    kmx_fy = force_data[f'{kmx_side} belt_Force_Y'].to_numpy()
    kmx_first, kmx_second = [], []
    for kmx_h, kmx_t in force_contacts[kmx_side]:
        kmx_mid = (kmx_h + kmx_t) // 2
        kmx_first.append(kmx_fy[kmx_h:kmx_mid].mean())
        kmx_second.append(kmx_fy[kmx_mid:kmx_t].mean())
    kmx_ap_sign_check.append((kmx_side, np.mean(kmx_first), np.mean(kmx_second)))
    print(f"  {kmx_side} AP force: first half of stance {np.mean(kmx_first):+.1f} N, "
          f"second half {np.mean(kmx_second):+.1f} N")

kmx_forward_sign = com_belt_sign
if kmx_ap_sign_check[0][2] < kmx_ap_sign_check[0][1]:
    kmx_forward_sign = -com_belt_sign
    print(f"  ! the AP pattern is reversed from what the belt direction implies;")
    print(f"    using the force pattern, which is the more direct evidence.")
print(f"  propulsion is the {'+' if kmx_forward_sign > 0 else '-'}Y direction")

# -----------------------------------------------------------------------------
# %% per-stance kinetic measures
# -----------------------------------------------------------------------------

kmx_rows = []
for kmx_side in ('Left', 'Right'):
    kmx_fx = force_data[f'{kmx_side} belt_Force_X'].to_numpy()
    kmx_fy = force_data[f'{kmx_side} belt_Force_Y'].to_numpy() * kmx_forward_sign
    kmx_fz = force_data[f'{kmx_side} belt_Force_Z'].to_numpy()
    kmx_cx = cop_filtered[kmx_side]['X']
    kmx_cy = cop_filtered[kmx_side]['Y']
    kmx_free = free_moment[kmx_side]

    for kmx_h, kmx_t in force_contacts[kmx_side]:
        if kmx_t - kmx_h < kmx_min_stance_frames:
            continue
        kmx_n = kmx_t - kmx_h
        kmx_vz = kmx_fz[kmx_h:kmx_t]
        kmx_vy = kmx_fy[kmx_h:kmx_t]
        kmx_dt = 1.0 / kin_fs

        # --- AP impulses -----------------------------------------------------
        kmx_brake = float(np.sum(kmx_vy[kmx_vy < 0]) * kmx_dt)      # negative
        kmx_propel = float(np.sum(kmx_vy[kmx_vy > 0]) * kmx_dt)     # positive

        # --- vertical GRF landmarks ------------------------------------------
        kmx_a = int(kmx_peak_search[0] * kmx_n)
        kmx_b = int(kmx_peak_search[1] * kmx_n)
        kmx_mid = kmx_n // 2
        kmx_f1_idx = kmx_a + int(np.argmax(kmx_vz[kmx_a:kmx_mid])) if kmx_mid > kmx_a else kmx_a
        kmx_f2_idx = kmx_mid + int(np.argmax(kmx_vz[kmx_mid:kmx_b])) if kmx_b > kmx_mid else kmx_mid
        kmx_trough_idx = (kmx_f1_idx + int(np.argmin(kmx_vz[kmx_f1_idx:kmx_f2_idx]))
                          if kmx_f2_idx > kmx_f1_idx else kmx_f1_idx)
        kmx_f1, kmx_f2 = float(kmx_vz[kmx_f1_idx]), float(kmx_vz[kmx_f2_idx])
        kmx_trough = float(kmx_vz[kmx_trough_idx])

        # --- loading rate, 20-80% of the first peak --------------------------
        kmx_lr = np.nan
        kmx_rise = kmx_vz[:kmx_f1_idx + 1]
        if len(kmx_rise) > 5 and kmx_f1 > 0:
            kmx_lo = np.argmax(kmx_rise >= kmx_loading_band[0] * kmx_f1)
            kmx_hi = np.argmax(kmx_rise >= kmx_loading_band[1] * kmx_f1)
            if kmx_hi > kmx_lo:
                kmx_lr = float((kmx_rise[kmx_hi] - kmx_rise[kmx_lo])
                               / ((kmx_hi - kmx_lo) * kmx_dt))

        # --- centre of pressure ----------------------------------------------
        kmx_px, kmx_py = kmx_cx[kmx_h:kmx_t], kmx_cy[kmx_h:kmx_t]
        kmx_path = float(np.sum(np.hypot(np.diff(kmx_px), np.diff(kmx_py))))

        kmx_rows.append({
            'side': kmx_side,
            'heel_strike_frame_100hz': int(kin_sample_to_frame(kmx_h)),
            'stance_time': kmx_n / kin_fs,
            'braking_impulse_ns': kmx_brake,
            'propulsive_impulse_ns': kmx_propel,
            'net_ap_impulse_ns': kmx_brake + kmx_propel,
            'braking_impulse_bw_s': kmx_brake / kin_body_weight,
            'propulsive_impulse_bw_s': kmx_propel / kin_body_weight,
            'vertical_impulse_bw_s': float(np.sum(kmx_vz) * kmx_dt) / kin_body_weight,
            'grf_peak1_bw': kmx_f1 / kin_body_weight,
            'grf_trough_bw': kmx_trough / kin_body_weight,
            'grf_peak2_bw': kmx_f2 / kin_body_weight,
            'loading_rate_bw_s': kmx_lr / kin_body_weight if np.isfinite(kmx_lr) else np.nan,
            'cop_ml_range_mm': float(np.ptp(kmx_px)),
            'cop_ap_range_mm': float(np.ptp(kmx_py)),
            'cop_path_mm': kmx_path,          # filtered; see kmx_cop_filter_hz
            'free_moment_peak_nm': float(np.max(np.abs(kmx_free[kmx_h:kmx_t]))),
            'free_moment_range_nm': float(np.ptp(kmx_free[kmx_h:kmx_t])),
        })

kinetic_metrics = pd.DataFrame(kmx_rows)

# -----------------------------------------------------------------------------
# %% report and symmetry
# -----------------------------------------------------------------------------

if len(kinetic_metrics):
    print(f"\n  {len(kinetic_metrics)} stances")
    for kmx_side, kmx_g in kinetic_metrics.groupby('side'):
        print(f"    {kmx_side:5s} vGRF  F1 {kmx_g['grf_peak1_bw'].mean():.3f}  "
              f"trough {kmx_g['grf_trough_bw'].mean():.3f}  "
              f"F2 {kmx_g['grf_peak2_bw'].mean():.3f} BW   "
              f"loading rate {kmx_g['loading_rate_bw_s'].mean():.1f} BW/s")
        print(f"          impulse  braking {kmx_g['braking_impulse_bw_s'].mean():+.4f}  "
              f"propulsive {kmx_g['propulsive_impulse_bw_s'].mean():+.4f}  "
              f"net {kmx_g['net_ap_impulse_ns'].mean()/kin_body_weight:+.4f} BW*s")
        print(f"          CoP  ML range {kmx_g['cop_ml_range_mm'].mean():.0f} mm  "
              f"path {kmx_g['cop_path_mm'].mean():.0f} mm   "
              f"free moment peak {kmx_g['free_moment_peak_nm'].mean():.1f} N*m")

    # Net AP impulse must be near zero over a steady trial: the walker is not
    # accelerating, so propulsion has to cancel braking. A large net value
    # means a force-plate offset, not a physiological finding.
    kmx_net = kinetic_metrics['net_ap_impulse_ns'].mean()
    kmx_prop = kinetic_metrics['propulsive_impulse_ns'].mean()
    print(f"\n  net AP impulse per stance {kmx_net:+.3f} N*s "
          f"({100*abs(kmx_net)/abs(kmx_prop):.1f}% of the propulsive impulse)")
    if abs(kmx_net) > 0.1 * abs(kmx_prop):
        print(f"    ! that should be near zero at steady speed. Check the AP zero offset.")

    # Symmetry angle [Zifchock2008]: bounded and reference-free, unlike the
    # classic symmetry index which depends on which limb is the denominator.
    print("\n  symmetry angle (0% = symmetric, sign gives direction)")
    for kmx_var in ('propulsive_impulse_ns', 'braking_impulse_ns', 'grf_peak1_bw',
                    'grf_peak2_bw', 'vertical_impulse_bw_s'):
        kmx_l = kinetic_metrics[kinetic_metrics['side'] == 'Left'][kmx_var].mean()
        kmx_r = kinetic_metrics[kinetic_metrics['side'] == 'Right'][kmx_var].mean()
        if not (np.isfinite(kmx_l) and np.isfinite(kmx_r)) or kmx_r == 0:
            continue
        kmx_sa = (45.0 - np.degrees(np.arctan2(kmx_l, kmx_r))) / 90.0 * 100.0
        if kmx_sa > 100.0:
            kmx_sa -= 200.0
        print(f"    {kmx_var:26s} L {kmx_l:+9.4f}  R {kmx_r:+9.4f}  SA {kmx_sa:+6.2f}%")
else:
    print("  ! no stances produced kinetic metrics")

# =============================================================================
# %% One stride series, shared by every stride-to-stride metric
# =============================================================================
# DFA, entropy, the goal-equivalent manifold analysis and the foot placement
# model all consume a stride-indexed series, and all of them are N-DEPENDENT.
# Build them separately and you end up comparing alpha on 756 strides against
# lambda on 150, where part of any difference is sample size rather than
# physiology. So they are built once, here, and every later cell draws from
# this table.
#
# WHY A FIXED N MATTERS MORE THAN A LARGE N
# alpha, sample entropy and lambda all drift with series length. If one
# condition ran 15 minutes and another 10, the longer trial gets a
# systematically different alpha for a reason that has nothing to do with the
# walker. The fix is the same one the LDS cell already applies to stride count:
# truncate every trial to a COMMON length, chosen as the shortest trial in the
# dataset, and state it.
#
#   at 108 bpm the cadence is 54 strides/min, so
#     15 min trial -> 810 strides, 756 after a 60 s warm-up
#     10 min trial -> 540 strides, 486 after a 60 s warm-up
#
# 486 is below the 600 that [Damouras2010] recommend for a stable alpha, so on
# 10 minute trials alpha carries wider confidence intervals. That is a reason
# to report the interval, not a reason to use a different N per trial.
#
# References
#   [Damouras2010]  Damouras et al. (2010) Gait Posture 31(3), 336-340.
#   [Dingwell2010]  Dingwell, John & Cusumano (2010) PLoS Comput Biol 6(7), e1000856.
# =============================================================================

# -----------------------------------------------------------------------------
# %% stride series settings
# -----------------------------------------------------------------------------

ss_limb        = 'Right'   # which limb's heel strikes define a stride
ss_warmup_s    = 60.0      # treadmill acclimatisation, excluded
ss_fixed_n     = None      # None = use everything available. SET THIS to the
                           # shortest trial in the dataset before comparing
                           # trials, e.g. 480 for a 10 minute protocol.
ss_match_tolerance = 5     # frames, for joining per-step tables onto strides

print("\n" + "-" * 74)
print("  STRIDE SERIES")
print("-" * 74)

# -----------------------------------------------------------------------------
# %% the stride index, from the force events
# -----------------------------------------------------------------------------
# Force events rather than the merged event file, because they are one
# consistent detector rather than a GRF/kinematic mixture, and mixed timing
# sources put a step change into every interval they touch.

ss_contacts = force_contacts[ss_limb]
ss_hs_frames = np.array([int(kin_sample_to_frame(h)) for h, _ in ss_contacts])
ss_to_frames = np.array([int(kin_sample_to_frame(t)) for _, t in ss_contacts])

ss_keep = ss_hs_frames > ss_warmup_s * kin_kinematic_fs
ss_hs_frames, ss_to_frames = ss_hs_frames[ss_keep], ss_to_frames[ss_keep]
ss_contacts = [c for c, k in zip(ss_contacts, ss_keep) if k]

print(f"  limb {ss_limb}, {ss_warmup_s:.0f} s warm-up excluded")
print(f"  {len(ss_hs_frames)} strides available after the warm-up")

stride_series = pd.DataFrame({
    'stride': np.arange(len(ss_hs_frames)),
    'heel_strike_frame_100hz': ss_hs_frames,
    'toe_off_frame_100hz': ss_to_frames,
})
stride_series['stride_time'] = np.append(
    np.diff(ss_hs_frames) / kin_kinematic_fs, np.nan)
stride_series['stance_time'] = (ss_to_frames - ss_hs_frames) / kin_kinematic_fs
stride_series['swing_time'] = stride_series['stride_time'] - stride_series['stance_time']
stride_series['stance_percent'] = (100 * stride_series['stance_time']
                                   / stride_series['stride_time'])
stride_series['cadence_spm'] = 120.0 / stride_series['stride_time']

# -----------------------------------------------------------------------------
# %% stride length and velocity in the belt frame
# -----------------------------------------------------------------------------
# On a treadmill the foot returns to roughly the same lab position each stride,
# so stride length is not a lab-frame displacement. It is the belt travel over
# the stride plus whatever net progression the walker made:
#     stride length = belt speed * stride time + (lab displacement of the foot)

if f'{ss_limb}_Heel' in kinematic_positions:
    ss_heel_ap = kinematic_positions[f'{ss_limb}_Heel'][:, com_ap_axis]
    ss_disp = np.full(len(ss_hs_frames), np.nan)
    for ss_i in range(len(ss_hs_frames) - 1):
        ss_a, ss_b = ss_hs_frames[ss_i] - 1, ss_hs_frames[ss_i + 1] - 1
        if 0 <= ss_a < len(ss_heel_ap) and 0 <= ss_b < len(ss_heel_ap):
            ss_disp[ss_i] = ss_heel_ap[ss_b] - ss_heel_ap[ss_a]
    stride_series['stride_length'] = (com_belt_speed_used * stride_series['stride_time']
                                      + com_belt_sign * ss_disp)
    stride_series['stride_velocity'] = (stride_series['stride_length']
                                        / stride_series['stride_time'])

# step width: the ML distance between the two feet at each heel strike
if f'{ss_limb}_Heel' in kinematic_positions:
    ss_other = 'Left' if ss_limb == 'Right' else 'Right'
    if f'{ss_other}_Heel' in kinematic_positions:
        ss_ml_self = kinematic_positions[f'{ss_limb}_Heel'][:, com_ml_axis]
        ss_ml_other = kinematic_positions[f'{ss_other}_Heel'][:, com_ml_axis]
        ss_idx = np.clip(ss_hs_frames - 1, 0, len(ss_ml_self) - 1)
        stride_series['step_width'] = np.abs(ss_ml_self[ss_idx] - ss_ml_other[ss_idx])

# -----------------------------------------------------------------------------
# %% join the per-step tables onto the stride index
# -----------------------------------------------------------------------------

def ss_join(table, columns, frame_column='heel_strike_frame_100hz', prefix=''):
    """Attach per-step values to the stride table, matched on `frame_column`.

    The stride table carries both the heel strike and the toe off frame, and
    the two must be matched on the SAME event: joining a toe-off-keyed table
    onto heel strike frames misses by a whole stance phase and silently
    produces an all-NaN column.
    """
    if table is None or not len(table):
        return
    ss_sub = table[table['side'] == ss_limb] if 'side' in table.columns else table
    ss_sub = ss_sub[[frame_column] + [c for c in columns if c in ss_sub.columns]]
    if len(ss_sub) == 0:
        return
    if frame_column not in stride_series.columns:
        print(f"  ! cannot join on {frame_column}, it is not in the stride table")
        return
    ss_left = stride_series[[frame_column]].copy()
    ss_left['_row'] = np.arange(len(ss_left))
    ss_merged = pd.merge_asof(
        ss_left.sort_values(frame_column),
        ss_sub.sort_values(frame_column).rename(
            columns={frame_column: frame_column + '_r'}),
        left_on=frame_column, right_on=frame_column + '_r',
        direction='nearest', tolerance=ss_match_tolerance)
    ss_merged = ss_merged.sort_values('_row')
    for ss_c in columns:
        if ss_c in ss_merged.columns:
            stride_series[prefix + ss_c] = ss_merged[ss_c].to_numpy()
            ss_hit = np.isfinite(pd.to_numeric(ss_merged[ss_c],
                                               errors='coerce')).mean()
            if ss_hit < 0.5:
                print(f"  ! {ss_c} matched only {100*ss_hit:.0f}% of strides on "
                      f"{frame_column}; check the key and ss_match_tolerance")

ss_join(kinetic_metrics, ['propulsive_impulse_bw_s', 'braking_impulse_bw_s',
                          'vertical_impulse_bw_s', 'grf_peak1_bw', 'grf_peak2_bw',
                          'loading_rate_bw_s', 'cop_ml_range_mm',
                          'free_moment_peak_nm'])
ss_join(margin_of_stability, ['mos_ml_contact', 'mos_ml_min',
                              'mos_ap_contact', 'mos_ap_min'])
ss_join(trip_risk, ['mfc_m', 'trip_risk_integral', 'moi_peak_mm'],
        frame_column='toe_off_frame_100hz')

# -----------------------------------------------------------------------------
# %% truncate to a common length
# -----------------------------------------------------------------------------

ss_available = len(stride_series)
if ss_fixed_n is not None:
    if ss_available < ss_fixed_n:
        print(f"  ! only {ss_available} strides, fewer than the {ss_fixed_n} this")
        print(f"    dataset is standardised to. This trial is NOT comparable on any")
        print(f"    N-dependent metric (alpha, entropy, lambda). Either shorten")
        print(f"    ss_fixed_n for the whole dataset or exclude this trial.")
    else:
        stride_series = stride_series.iloc[:ss_fixed_n].copy()
        print(f"  truncated to the dataset-wide {ss_fixed_n} strides "
              f"({ss_available} were available)")
else:
    print(f"  using all {ss_available} strides. SET ss_fixed_n before comparing")
    print(f"    trials: alpha, entropy and lambda all drift with series length, so")
    print(f"    a 756-stride trial is not comparable with a 486-stride one.")

# -----------------------------------------------------------------------------
# %% what is usable
# -----------------------------------------------------------------------------
# A series with gaps is not a series. Anything that will be handed to DFA or
# entropy has to be reported with its completeness, because those estimators
# silently accept a NaN-filled array and return a number.

ss_candidates = ['stride_time', 'stance_time', 'swing_time', 'stance_percent',
                 'stride_length', 'stride_velocity', 'step_width',
                 'mos_ml_contact', 'mos_ap_contact', 'mfc_m',
                 'propulsive_impulse_bw_s', 'braking_impulse_bw_s',
                 'grf_peak1_bw', 'grf_peak2_bw', 'trip_risk_integral']

print(f"\n  series                      n valid   complete   mean        CV")
stride_series_usable = []
for ss_c in ss_candidates:
    if ss_c not in stride_series.columns:
        continue
    ss_v = stride_series[ss_c].to_numpy(float)
    ss_ok = np.isfinite(ss_v)
    if ss_ok.sum() < 10:
        continue
    ss_pct = 100 * ss_ok.mean()
    ss_mean = np.nanmean(ss_v)
    ss_cv = 100 * np.nanstd(ss_v) / abs(ss_mean) if ss_mean else np.nan
    print(f"    {ss_c:26s} {ss_ok.sum():5d}    {ss_pct:5.1f}%   "
          f"{ss_mean:9.4f}  {ss_cv:6.2f}%")
    if ss_pct > 95:
        stride_series_usable.append(ss_c)

print(f"\n  {len(stride_series_usable)} series are over 95% complete and will be")
print(f"    carried into the stride-to-stride metrics:")
print(f"    {', '.join(stride_series_usable)}")

# The metronome constrains stride TIMING directly, so alpha on stride time
# measures how tightly the walker locks to the beat. Series the metronome does
# not set are the better primary outcomes.
ss_cued = {'stride_time', 'stance_time', 'swing_time', 'cadence_spm'}
ss_uncued = [c for c in stride_series_usable if c not in ss_cued]
print(f"\n  with a metronome, prefer the series it does not directly constrain:")
print(f"    {', '.join(ss_uncued[:8])}")

# =============================================================================
# %% Long-range correlations: detrended fluctuation analysis
# =============================================================================
# DFA asks whether a stride is statistically related to strides HUNDREDS of
# strides earlier. Not how much gait varies -- how the variation is organised
# in time. It is blind to magnitude: shuffle a series and its SD is unchanged
# while alpha collapses to 0.5.
#
#   integrate      Y(k) = sum_{i<=k} (x_i - mean(x))
#   segment        non-overlapping boxes of length n, forwards AND backwards so
#                  no data is discarded when n does not divide N
#   detrend        least squares polynomial of order 1 within each box
#   fluctuate      F(n) = RMS of the residuals over all boxes
#   scale          F(n) ~ n^alpha, alpha is the slope in log-log space
#
#   alpha < 0.5   anti-persistent, a long stride is corrected by a short one
#   alpha = 0.5   uncorrelated, no memory
#   0.5 < a < 1   persistent, deviations tend to be followed by more of the same
#   alpha ~ 1.0   1/f, scale free
#   alpha > 1.0   non-stationary, usually a trend rather than a property of
#                 control -- check for drift before interpreting
#
# THE BOX RANGE IS THE CHOICE THAT MATTERS
# [Damouras2010] examined this directly and recommends 16 <= n <= N/9. The
# widely used 4 to N/4 inflates alpha: small boxes are dominated by the
# detrending fit itself, and boxes past N/9 contain too few segments to average.
#
# TWO CHECKS RUN EVERY TIME
#   a shuffled surrogate must return alpha = 0.5. If it does not, the
#     implementation is wrong, and this catches it on the real data rather than
#     only in the validation file.
#   a second estimator, GPH on the periodogram, gives d and hence alpha = d+0.5
#     by a completely different route. DFA alone cannot resolve the fGn/fBm
#     ambiguity, so agreement between the two is worth having and disagreement
#     is worth knowing about.
#
# References
#   [Peng1994]      Peng et al. (1994) Phys Rev E 49, 1685-1689.
#   [Hausdorff1996] Hausdorff et al. (1996) J Appl Physiol 80(5), 1448-1457.
#   [Damouras2010]  Damouras et al. (2010) Gait Posture 31(3), 336-340.
#   [Ravi2020]      Ravi et al. (2020) Front Physiol 11, 562.
#   [GPH1983]       Geweke & Porter-Hudak (1983) J Time Ser Anal 4(4), 221-238.
# =============================================================================

# -----------------------------------------------------------------------------
# %% DFA settings
# -----------------------------------------------------------------------------

dfa_order        = 1        # DFA-1, linear detrending within each box
dfa_min_box      = 16       # [Damouras2010]
dfa_max_box_frac = 1.0 / 9  # [Damouras2010]
dfa_n_boxes      = 20       # how many box sizes to sample, log spaced
dfa_n_surrogate  = 50       # shuffles for the surrogate check
dfa_seed         = 0
dfa_gph_power    = 0.5      # GPH bandwidth: m = N**dfa_gph_power

# -----------------------------------------------------------------------------
# %% the estimators
# -----------------------------------------------------------------------------

def dfa_fluctuation(series, box_sizes, order=None):
    """F(n) for each box size. Forward and backward boxes at every scale."""
    order = dfa_order if order is None else order
    dfa_x = np.asarray(series, float)
    dfa_x = dfa_x[np.isfinite(dfa_x)]
    dfa_y = np.cumsum(dfa_x - dfa_x.mean())
    dfa_n_total = len(dfa_y)

    dfa_out = np.full(len(box_sizes), np.nan)
    for dfa_i, dfa_n in enumerate(box_sizes):
        dfa_n = int(dfa_n)
        dfa_count = dfa_n_total // dfa_n
        if dfa_count < 2:
            continue
        # forward from the start, and backward from the end, so a series whose
        # length is not a multiple of n does not lose its tail
        dfa_fwd = dfa_y[:dfa_count * dfa_n].reshape(dfa_count, dfa_n)
        dfa_bwd = dfa_y[dfa_n_total - dfa_count * dfa_n:].reshape(dfa_count, dfa_n)
        dfa_seg = np.vstack([dfa_fwd, dfa_bwd])

        dfa_t = np.arange(dfa_n)
        dfa_coef = np.polyfit(dfa_t, dfa_seg.T, order)
        dfa_trend = np.polyval(dfa_coef, dfa_t[:, None]).T
        dfa_out[dfa_i] = np.sqrt(np.mean((dfa_seg - dfa_trend) ** 2))
    return dfa_out


def dfa_alpha(series, min_box=None, max_box=None, order=None, n_boxes=None):
    """Scaling exponent, plus the fit quality and the curve it came from."""
    min_box = dfa_min_box if min_box is None else min_box
    n_boxes = dfa_n_boxes if n_boxes is None else n_boxes
    dfa_x = np.asarray(series, float)
    dfa_x = dfa_x[np.isfinite(dfa_x)]
    dfa_N = len(dfa_x)
    max_box = int(dfa_max_box_frac * dfa_N) if max_box is None else int(max_box)

    if max_box <= min_box or dfa_N < 4 * min_box:
        return dict(alpha=np.nan, se=np.nan, r2=np.nan, n=dfa_N,
                    boxes=np.array([]), fluctuation=np.array([]))

    dfa_boxes = np.unique(np.round(np.logspace(
        np.log10(min_box), np.log10(max_box), n_boxes)).astype(int))
    dfa_f = dfa_fluctuation(dfa_x, dfa_boxes, order=order)
    dfa_ok = np.isfinite(dfa_f) & (dfa_f > 0)
    if dfa_ok.sum() < 4:
        return dict(alpha=np.nan, se=np.nan, r2=np.nan, n=dfa_N,
                    boxes=dfa_boxes, fluctuation=dfa_f)

    dfa_lx = np.log10(dfa_boxes[dfa_ok])
    dfa_ly = np.log10(dfa_f[dfa_ok])
    dfa_slope, dfa_icept = np.polyfit(dfa_lx, dfa_ly, 1)

    dfa_pred = dfa_slope * dfa_lx + dfa_icept
    dfa_resid = dfa_ly - dfa_pred
    dfa_dof = max(len(dfa_lx) - 2, 1)
    dfa_se = float(np.sqrt(np.sum(dfa_resid ** 2) / dfa_dof
                           / np.sum((dfa_lx - dfa_lx.mean()) ** 2)))
    dfa_r2 = float(1 - np.sum(dfa_resid ** 2)
                   / np.sum((dfa_ly - dfa_ly.mean()) ** 2))
    return dict(alpha=float(dfa_slope), se=dfa_se, r2=dfa_r2, n=dfa_N,
                boxes=dfa_boxes[dfa_ok], fluctuation=dfa_f[dfa_ok])


def dfa_gph(series, power=None):
    """Geweke-Porter-Hudak estimate of the fractional differencing parameter d.

    A completely different route to the same property: regress the log
    periodogram on log(4 sin^2(w/2)) over the lowest frequencies. For
    fractional Gaussian noise alpha = d + 0.5, so this is an independent check
    on DFA rather than a restatement of it.
    """
    power = dfa_gph_power if power is None else power
    dfa_x = np.asarray(series, float)
    dfa_x = dfa_x[np.isfinite(dfa_x)]
    dfa_N = len(dfa_x)
    dfa_m = int(dfa_N ** power)
    if dfa_m < 4:
        return dict(d=np.nan, alpha=np.nan, m=dfa_m)

    dfa_per = np.abs(np.fft.rfft(dfa_x - dfa_x.mean())) ** 2 / (2 * np.pi * dfa_N)
    dfa_w = 2 * np.pi * np.arange(len(dfa_per)) / dfa_N
    dfa_j = np.arange(1, min(dfa_m, len(dfa_per) - 1) + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dfa_reg = np.log(4 * np.sin(dfa_w[dfa_j] / 2) ** 2)
        dfa_resp = np.log(dfa_per[dfa_j])
    # a periodogram ordinate can be exactly zero, which the log sends to -inf
    dfa_ok = np.isfinite(dfa_reg) & np.isfinite(dfa_resp)
    if dfa_ok.sum() < 4:
        return dict(d=np.nan, alpha=np.nan, m=dfa_m)
    dfa_slope = np.polyfit(dfa_reg[dfa_ok], dfa_resp[dfa_ok], 1)[0]
    dfa_d = float(-dfa_slope)
    return dict(d=dfa_d, alpha=dfa_d + 0.5, m=dfa_m)


def dfa_surrogate_alpha(series, n_surrogate=None, seed=None):
    """alpha of shuffled copies. Shuffling destroys order but not the values,
    so alpha must fall to 0.5 while the SD is untouched."""
    n_surrogate = dfa_n_surrogate if n_surrogate is None else n_surrogate
    dfa_rng = np.random.default_rng(dfa_seed if seed is None else seed)
    dfa_x = np.asarray(series, float)
    dfa_x = dfa_x[np.isfinite(dfa_x)]
    dfa_vals = [dfa_alpha(dfa_rng.permutation(dfa_x))['alpha']
                for _ in range(n_surrogate)]
    return np.array(dfa_vals, float)

# -----------------------------------------------------------------------------
# %% run it over every usable series
# -----------------------------------------------------------------------------

print("\n" + "-" * 74)
print("  LONG-RANGE CORRELATIONS (DFA)")
print("-" * 74)

dfa_N_used = len(stride_series)
dfa_max_box_used = int(dfa_max_box_frac * dfa_N_used)
print(f"  N = {dfa_N_used} strides, boxes {dfa_min_box} to {dfa_max_box_used} "
      f"(Damouras: 16 to N/9)")
if dfa_N_used < 600:
    print(f"  ! under the 600 strides [Damouras2010] recommend. alpha is still")
    print(f"    estimable but its confidence interval is wider; report it.")
if dfa_max_box_used <= dfa_min_box * 2:
    print(f"  ! the box range spans less than one octave, alpha is not meaningful")

dfa_rows = []
for dfa_name in stride_series_usable:
    dfa_series = stride_series[dfa_name].to_numpy(float)
    dfa_res = dfa_alpha(dfa_series)
    if not np.isfinite(dfa_res['alpha']):
        continue
    dfa_g = dfa_gph(dfa_series)
    dfa_sur = dfa_surrogate_alpha(dfa_series)
    dfa_rows.append({
        'series': dfa_name,
        'n': dfa_res['n'],
        'alpha': dfa_res['alpha'],
        'alpha_se': dfa_res['se'],
        'r2': dfa_res['r2'],
        'alpha_gph': dfa_g['alpha'],
        'surrogate_alpha_mean': float(np.nanmean(dfa_sur)),
        'surrogate_alpha_sd': float(np.nanstd(dfa_sur)),
        'cued_by_metronome': dfa_name in ss_cued,
    })

dfa_results = pd.DataFrame(dfa_rows)

if len(dfa_results):
    print(f"\n  series                      alpha   +-SE     R2    GPH    "
          f"surrogate      cued")
    for _, dfa_r in dfa_results.iterrows():
        dfa_flag = '  yes' if dfa_r['cued_by_metronome'] else '   no'
        print(f"    {dfa_r['series']:26s} {dfa_r['alpha']:5.3f}  "
              f"{dfa_r['alpha_se']:.3f}  {dfa_r['r2']:.3f}  "
              f"{dfa_r['alpha_gph']:5.3f}  "
              f"{dfa_r['surrogate_alpha_mean']:.3f}+-{dfa_r['surrogate_alpha_sd']:.3f}"
              f"{dfa_flag}")

    # the surrogate check, on the real data
    dfa_bad = dfa_results[np.abs(dfa_results['surrogate_alpha_mean'] - 0.5) > 0.05]
    if len(dfa_bad):
        print(f"\n  ! {len(dfa_bad)} series have a shuffled surrogate away from 0.5.")
        print(f"    A shuffle destroys order and nothing else, so alpha must go to")
        print(f"    0.5. Anything else means the estimator is misbehaving at this N.")
    else:
        print(f"\n  surrogate check passed: every shuffled series returned "
              f"alpha ~ 0.5")

    # DFA against GPH. The validation puts GPH's own scatter at 0.15-0.21 for
    # N in the 500-750 range, against 0.06-0.10 for DFA, so GPH is a coarse
    # sanity check on the ballpark rather than a precise second opinion. Only a
    # large gap is informative.
    dfa_gap = np.abs(dfa_results['alpha'] - dfa_results['alpha_gph'])
    print(f"  DFA vs GPH: median gap {np.nanmedian(dfa_gap):.3f}, "
          f"worst {np.nanmax(dfa_gap):.3f}  (GPH scatter alone is ~0.15 at this N)")
    if np.nanmedian(dfa_gap) > 0.25:
        print(f"    ! the two estimators disagree. Usually that means the series is")
        print(f"      not a clean fGn -- check for a trend or a level shift.")

    print(f"\n  reminder: this protocol paces cadence with a metronome, so alpha on")
    print(f"  the cued series measures how tightly the walker locks to the beat.")
    print(f"  The uncued series are the better primary outcomes.")
else:
    print("  ! no series produced an alpha")

# =============================================================================
# %% Entropy: sample entropy and refined composite multiscale entropy
# =============================================================================
# Sample entropy asks how predictable a series is. Take every template of m
# consecutive points; count how many other templates match it within a
# tolerance r; then count how many of those still match when extended to m+1
# points. The negative log of that ratio is the entropy.
#
#     SampEn(m, r, N) = -ln( A / B )
#
#   B  pairs of length-m templates within Chebyshev distance r
#   A  pairs that are STILL within r when extended to length m+1
#
# Self-matches are excluded, which is the difference from approximate entropy
# and removes ApEn's bias toward calling everything regular.
#
# THE INTERPRETATION TRAP
# High entropy gets called "complex", but white noise has MAXIMAL sample
# entropy and no complexity at all. That is why multiscale entropy exists:
# coarse-grain the series over longer and longer windows and recompute. White
# noise starts high and falls away steeply; a genuinely structured signal holds
# its entropy across scales. The CURVE is the complexity measure, not the
# single-scale value.
#
# The refined composite variant averages the match COUNTS over all tau
# coarse-graining phases before taking the logarithm, rather than averaging
# tau separate entropies. That is much steadier at long scales, where any one
# phase has few points left [Wu2014].
#
# THERE ARE NO TRANSFERABLE NORMATIVE VALUES
# SampEn is defined only relative to its m, r, N and preprocessing, and those
# vary across the literature. Compare conditions analysed identically; never
# compare an absolute value against another paper. [Yentes2021] requires m, r
# and N to be reported every time, and warns against assuming one r transfers
# between conditions -- so r is swept here rather than fixed at 0.2 SD and
# hoped for.
#
# References
#   [Richman2000]  Richman & Moorman (2000) Am J Physiol Heart Circ Physiol
#                  278(6), H2039-H2049.
#   [Costa2002]    Costa, Goldberger & Peng (2002) Phys Rev Lett 89, 068102.
#   [Yentes2013]   Yentes et al. (2013) Ann Biomed Eng 41(2), 349-365.
#   [Wu2014]       Wu et al. (2014) Phys Lett A 378(20), 1369-1374.
#   [Yentes2021]   Yentes & Raffalt (2021) Ann Biomed Eng 49(3), 979-990.
# =============================================================================

from scipy.spatial import cKDTree      # repeated so the cell runs alone

# -----------------------------------------------------------------------------
# %% entropy settings
# -----------------------------------------------------------------------------

ent_m          = 2                       # template length
ent_r          = 0.2                     # tolerance, as a fraction of the SD
ent_r_sweep    = [0.10, 0.15, 0.20, 0.25, 0.30]   # [Yentes2021]
ent_m_sweep    = [2, 3]
ent_normalise  = True                    # z-score before computing
ent_mse_scales = 10                      # coarse-graining scales for RCMSE
ent_min_n      = 200                     # [Yentes2013]: below this, unreliable

# -----------------------------------------------------------------------------
# %% the estimators
# -----------------------------------------------------------------------------

def ent_match_counts(series, m, r):
    """Counts of matching template pairs at lengths m and m+1.

    Returned rather than the entropy itself, because the refined composite
    multiscale variant has to sum counts across coarse-graining phases BEFORE
    taking the logarithm.

    Counting is done with a KD-tree under the Chebyshev metric. The obvious
    double loop is O(N^2) and is fine for a few hundred strides but hopeless on
    a continuous signal, where multiscale entropy needs tens of thousands of
    points at every scale and phase.

    A length-(m+1) match implies a length-m match, since the Chebyshev distance
    over m+1 coordinates is never smaller than over the first m. So A is simply
    the pair count in the (m+1)-dimensional tree; there is no need to filter
    the m-dimensional matches a second time.
    """
    ent_x = np.asarray(series, float)
    ent_x = ent_x[np.isfinite(ent_x)]
    ent_n = len(ent_x)
    if ent_n < m + 2:
        return 0, 0

    # the SAME N-m start indices at both lengths, so the counts are comparable
    ent_k = ent_n - m
    ent_starts = np.arange(ent_k)
    ent_tpl_m = ent_x[ent_starts[:, None] + np.arange(m)[None, :]]
    ent_tpl_m1 = ent_x[ent_starts[:, None] + np.arange(m + 1)[None, :]]

    ent_tree_m = cKDTree(ent_tpl_m)
    ent_tree_m1 = cKDTree(ent_tpl_m1)
    # count_neighbors counts ordered pairs and includes i == j, so subtract the
    # k self-matches and halve to get unordered distinct pairs
    ent_b = int((ent_tree_m.count_neighbors(ent_tree_m, r, p=np.inf) - ent_k) // 2)
    ent_a = int((ent_tree_m1.count_neighbors(ent_tree_m1, r, p=np.inf) - ent_k) // 2)
    return ent_a, ent_b


def ent_sample_entropy(series, m=None, r=None, normalise=None):
    """SampEn. r is a fraction of the series SD unless normalise is False."""
    m = ent_m if m is None else m
    r = ent_r if r is None else r
    normalise = ent_normalise if normalise is None else normalise

    ent_x = np.asarray(series, float)
    ent_x = ent_x[np.isfinite(ent_x)]
    if len(ent_x) < m + 2:
        return np.nan
    ent_sd = np.std(ent_x)
    if ent_sd == 0:
        return np.nan
    if normalise:
        ent_x = (ent_x - ent_x.mean()) / ent_sd
        ent_tol = r
    else:
        ent_tol = r * ent_sd

    ent_a, ent_b = ent_match_counts(ent_x, m, ent_tol)
    if ent_a == 0 or ent_b == 0:
        return np.nan            # no matches: undefined, not zero
    return float(-np.log(ent_a / ent_b))


def ent_coarse_grain(series, scale, phase=0):
    """Non-overlapping means of `scale` points, starting at `phase`."""
    ent_x = np.asarray(series, float)[phase:]
    ent_k = len(ent_x) // scale
    if ent_k < 1:
        return np.array([])
    return ent_x[:ent_k * scale].reshape(ent_k, scale).mean(axis=1)


def ent_rcmse(series, scales=None, m=None, r=None):
    """Refined composite multiscale entropy.

    At each scale the match counts are summed over all `scale` coarse-graining
    phases before the log is taken, rather than averaging separate entropies.
    Far steadier at long scales, where one phase leaves few points.
    """
    scales = ent_mse_scales if scales is None else scales
    m = ent_m if m is None else m
    r = ent_r if r is None else r

    ent_x = np.asarray(series, float)
    ent_x = ent_x[np.isfinite(ent_x)]
    if len(ent_x) < m + 2:
        return np.full(scales, np.nan)
    ent_sd = np.std(ent_x)
    if ent_sd == 0:
        return np.full(scales, np.nan)
    # tolerance is fixed on the ORIGINAL series, not recomputed per scale --
    # coarse-graining reduces the SD, so a per-scale r would change what is
    # being asked at every scale and the curve would not be comparable
    ent_x = (ent_x - ent_x.mean()) / ent_sd
    ent_out = np.full(scales, np.nan)
    for ent_s in range(1, scales + 1):
        ent_a_sum = ent_b_sum = 0
        for ent_p in range(ent_s):
            ent_cg = ent_coarse_grain(ent_x, ent_s, ent_p)
            if len(ent_cg) < m + 2:
                continue
            ent_a, ent_b = ent_match_counts(ent_cg, m, r)
            ent_a_sum += ent_a
            ent_b_sum += ent_b
        if ent_a_sum > 0 and ent_b_sum > 0:
            ent_out[ent_s - 1] = float(-np.log(ent_a_sum / ent_b_sum))
    return ent_out

# -----------------------------------------------------------------------------
# %% run it over the stride series
# -----------------------------------------------------------------------------

print("\n" + "-" * 74)
print("  ENTROPY")
print("-" * 74)
print(f"  m = {ent_m}, r = {ent_r} x SD, N = {len(stride_series)}, "
      f"z-scored = {ent_normalise}")
if len(stride_series) < ent_min_n:
    print(f"  ! N is under {ent_min_n}. [Yentes2013] shows both algorithms become")
    print(f"    extremely parameter sensitive below that.")

ent_rows = []
for ent_name in stride_series_usable:
    ent_series = stride_series[ent_name].to_numpy(float)
    ent_row = {'series': ent_name, 'n': int(np.isfinite(ent_series).sum()),
               'm': ent_m, 'r': ent_r}
    ent_row['sampen'] = ent_sample_entropy(ent_series)
    for ent_rv in ent_r_sweep:
        ent_row[f'sampen_r{ent_rv:.2f}'] = ent_sample_entropy(ent_series, r=ent_rv)
    for ent_mv in ent_m_sweep:
        ent_row[f'sampen_m{ent_mv}'] = ent_sample_entropy(ent_series, m=ent_mv)
    ent_rows.append(ent_row)

entropy_results = pd.DataFrame(ent_rows)

if len(entropy_results):
    print(f"\n  sample entropy, and how it moves with r")
    print(f"    series                     SampEn   " +
          "".join(f"r={v:.2f} " for v in ent_r_sweep))
    for _, ent_r_row in entropy_results.iterrows():
        ent_txt = f"    {ent_r_row['series']:26s} {ent_r_row['sampen']:6.3f}   "
        for ent_rv in ent_r_sweep:
            ent_txt += f"{ent_r_row[f'sampen_r{ent_rv:.2f}']:5.3f}  "
        print(ent_txt)

    # How sensitive is the ranking to r? If the order of the series changes
    # with r, no single r tells you anything durable about them.
    ent_ranks = {}
    for ent_rv in ent_r_sweep:
        ent_col = entropy_results[f'sampen_r{ent_rv:.2f}']
        if ent_col.notna().sum() > 2:
            ent_ranks[ent_rv] = ent_col.rank().to_numpy()
    if len(ent_ranks) > 1:
        ent_keys = list(ent_ranks)
        ent_corrs = [np.corrcoef(ent_ranks[ent_keys[0]], ent_ranks[k])[0, 1]
                     for k in ent_keys[1:]]
        print(f"\n  rank correlation with the r={ent_keys[0]:.2f} ordering: "
              f"{np.round(ent_corrs, 3)}")
        if np.nanmin(ent_corrs) < 0.8:
            print(f"    ! the ordering of the series changes with r. Report the sweep,")
            print(f"      not a single r, and do not read much into one value.")
        else:
            print(f"    the ordering is stable across r, so the single-r value is")
            print(f"    reporting something about the series rather than about r.")

# -----------------------------------------------------------------------------
# %% multiscale entropy on the continuous trunk acceleration
# -----------------------------------------------------------------------------
# On a stride series there are only a few hundred points, so coarse-graining
# runs out quickly. The continuous signal has enough samples for the curve to
# mean something.

ent_mse_signal = None
for ent_col in ('Low_Back_Joint_Acc', 'Trunk_Joint_Acc'):
    if ent_col in kinematic_data.columns:
        ent_mse_signal = np.column_stack([
            kinematic_data[ent_col].to_numpy(),
            kinematic_data[f'{ent_col}.1'].to_numpy(),
            kinematic_data[f'{ent_col}.2'].to_numpy()])
        ent_mse_name = ent_col
        break

multiscale_entropy = None
if ent_mse_signal is not None:
    ent_start = int(ss_warmup_s * kin_kinematic_fs)
    ent_seg = ent_mse_signal[ent_start:]
    ent_curves = {}
    for ent_ax, ent_label in enumerate(('ML', 'AP', 'VT')):
        ent_curves[ent_label] = ent_rcmse(ent_seg[:, ent_ax])
    multiscale_entropy = pd.DataFrame(ent_curves,
                                      index=np.arange(1, ent_mse_scales + 1))
    multiscale_entropy.index.name = 'scale'

    print(f"\n  refined composite MSE on {ent_mse_name}, "
          f"{len(ent_seg)} samples after the warm-up")
    print(f"    scale   ML     AP     VT")
    for ent_s, ent_row2 in multiscale_entropy.iterrows():
        print(f"    {ent_s:5d}  {ent_row2['ML']:5.3f}  {ent_row2['AP']:5.3f}  "
              f"{ent_row2['VT']:5.3f}")
    ent_area = multiscale_entropy.sum()
    print(f"    area under the curve: ML {ent_area['ML']:.2f}  "
          f"AP {ent_area['AP']:.2f}  VT {ent_area['VT']:.2f}")
    print(f"    The AREA is the complexity measure, not the scale-1 value. A")
    print(f"    curve that falls away steeply is noise-like; one that holds up")
    print(f"    across scales has structure at multiple time scales.")
else:
    print("\n  ! no trunk acceleration column found, MSE skipped")

# =============================================================================
# %% Harmonic ratio and improved harmonic ratio
# =============================================================================
# Smoothness and rhythmic regularity of trunk motion WITHIN a stride. This is a
# waveform-shape measure, so it is independent of everything in the DFA and
# entropy cells, which are about variation BETWEEN strides.
#
# WHY ODD AND EVEN
# One stride is two steps. If the two steps produce the same trunk
# acceleration, the signal repeats twice per stride and its energy lands on the
# EVEN harmonics of stride frequency. Asymmetry, jerkiness or an irregular
# event leaks energy into the ODD harmonics.
#
#     HR(AP, vertical) = sum(even harmonics) / sum(odd harmonics)
#
# Mediolateral is the exception and it is the part people get backwards. You
# sway left over one step and right over the other, so ML completes ONE cycle
# per stride, not two: ML is odd-dominant and the ratio flips.
#
#     HR(ML) = sum(odd harmonics) / sum(even harmonics)
#
# HR IS NOT A SYMMETRY MEASURE
# It is widely described as one. [Pasciuto2017] showed that is wrong -- a
# perfectly symmetric but jerky gait scores low -- and that HR's unboundedness
# makes group means unstable when a denominator is small. Their improved
# harmonic ratio relates the intrinsic harmonic power to the TOTAL power,
# giving a bounded 0-100% index, and is reported alongside.
#
# THE FRAME
# Accelerations are rotated by a single yaw taken from the mean trunk heading
# over the trial, not by the instantaneous pelvis orientation. Rotating frame
# by frame would mix pelvis rotation into the acceleration, which is not what
# this measure is about.
#
# References
#   [Menz2003]      Menz, Lord & Fitzpatrick (2003) Gait Posture 18(1), 35-46.
#   [Bellanca2013]  Bellanca et al. (2013) J Biomech 46(4), 828-831.
#   [Pasciuto2017]  Pasciuto et al. (2017) J Biomech 53, 84-89.
# =============================================================================

# -----------------------------------------------------------------------------
# %% harmonic ratio settings
# -----------------------------------------------------------------------------

hr_n_harmonics  = 20        # the established convention
hr_min_samples  = 40        # a stride shorter than this cannot carry 20 harmonics
hr_signal_order = ('Low_Back_Joint_Acc', 'Trunk_Joint_Acc')

print("\n" + "-" * 74)
print("  HARMONIC RATIO")
print("-" * 74)

# -------------------------------------------------------------------------
# %% per-stride spectrum
# -------------------------------------------------------------------------

def hr_harmonics(segment, n_harmonics=None):
    """Amplitudes of harmonics 1..n of the stride frequency.

    The segment is exactly one stride, so DFT bin 1 IS the stride
    fundamental and bin k is the k-th harmonic. No windowing: a window
    would smear energy between the odd and even bins that the ratio is
    built on.
    """
    n_harmonics = hr_n_harmonics if n_harmonics is None else n_harmonics
    hr_seg = np.asarray(segment, float)
    hr_seg = hr_seg - hr_seg.mean()
    hr_spec = np.abs(np.fft.rfft(hr_seg))
    if len(hr_spec) <= n_harmonics:
        return None
    return hr_spec[1:n_harmonics + 1]


def hr_ratio(amplitudes, odd_dominant=False):
    """Harmonic ratio and the Pasciuto improved harmonic ratio.

    `odd_dominant` is True for mediolateral, where the walker completes one
    cycle per stride rather than two.
    """
    hr_k = np.arange(1, len(amplitudes) + 1)
    hr_odd = amplitudes[hr_k % 2 == 1]
    hr_even = amplitudes[hr_k % 2 == 0]
    hr_intrinsic = hr_odd if odd_dominant else hr_even
    hr_other = hr_even if odd_dominant else hr_odd

    hr_value = (hr_intrinsic.sum() / hr_other.sum()
                if hr_other.sum() > 0 else np.nan)
    hr_total_power = np.sum(amplitudes ** 2)
    hr_i = (100.0 * np.sum(hr_intrinsic ** 2) / hr_total_power
            if hr_total_power > 0 else np.nan)
    return hr_value, hr_i

# -----------------------------------------------------------------------------
# %% signal and frame
# -----------------------------------------------------------------------------

hr_signal = None
for hr_col in hr_signal_order:
    if hr_col in kinematic_data.columns:
        hr_signal = np.column_stack([
            kinematic_data[hr_col].to_numpy(),
            kinematic_data[f'{hr_col}.1'].to_numpy(),
            kinematic_data[f'{hr_col}.2'].to_numpy()])
        hr_signal_name = hr_col
        break

if hr_signal is None:
    print("  ! no trunk acceleration column, harmonic ratio skipped")
    harmonic_ratio = pd.DataFrame()
else:
    print(f"  signal: {hr_signal_name}, "
          f"{100*np.isfinite(hr_signal).all(axis=1).mean():.1f}% finite")

    # A single yaw from the mean heading. On a treadmill the heading is fixed,
    # so this is close to identity -- it exists to remove any lab-to-walker
    # misalignment rather than to track the pelvis.
    hr_heading = np.array([0.0, com_belt_sign])          # travel direction, ML/AP
    hr_yaw = np.arctan2(hr_heading[0], hr_heading[1])
    hr_c, hr_s = np.cos(hr_yaw), np.sin(hr_yaw)
    hr_rot = np.array([[hr_c, -hr_s], [hr_s, hr_c]])
    hr_acc = hr_signal.copy()
    hr_acc[:, [com_ml_axis, com_ap_axis]] = (
        hr_signal[:, [com_ml_axis, com_ap_axis]] @ hr_rot.T)
    print(f"  heading yaw applied: {np.degrees(hr_yaw):+.1f} deg")

    hr_axis_spec = [(com_ap_axis, 'AP', False),
                    (com_vt_axis, 'VT', False),
                    (com_ml_axis, 'ML', True)]

    hr_rows = []
    for hr_i_stride in range(len(stride_series) - 1):
        hr_a = int(stride_series['heel_strike_frame_100hz'].iloc[hr_i_stride]) - 1
        hr_b = int(stride_series['heel_strike_frame_100hz'].iloc[hr_i_stride + 1]) - 1
        if hr_b - hr_a < hr_min_samples or hr_b > len(hr_acc):
            continue
        hr_seg_all = hr_acc[hr_a:hr_b]
        if not np.isfinite(hr_seg_all).all():
            continue

        hr_row = {'stride': int(stride_series['stride'].iloc[hr_i_stride]),
                  'n_samples': hr_b - hr_a}
        hr_good = True
        for hr_ax, hr_label, hr_odd_dom in hr_axis_spec:
            hr_amp = hr_harmonics(hr_seg_all[:, hr_ax])
            if hr_amp is None:
                hr_good = False
                break
            hr_v, hr_ih = hr_ratio(hr_amp, odd_dominant=hr_odd_dom)
            hr_row[f'hr_{hr_label}'] = hr_v
            hr_row[f'ihr_{hr_label}'] = hr_ih
        if hr_good:
            hr_rows.append(hr_row)

    harmonic_ratio = pd.DataFrame(hr_rows)

    if len(harmonic_ratio):
        print(f"\n  {len(harmonic_ratio)} strides")
        print(f"    axis   HR median (IQR)              iHR median (IQR)")
        for _, hr_label, hr_odd_dom in hr_axis_spec:
            hr_v = harmonic_ratio[f'hr_{hr_label}']
            hr_ih = harmonic_ratio[f'ihr_{hr_label}']
            hr_dom = 'odd/even' if hr_odd_dom else 'even/odd'
            print(f"    {hr_label}    {hr_v.median():5.2f} "
                  f"({hr_v.quantile(0.25):.2f}-{hr_v.quantile(0.75):.2f})  {hr_dom}"
                  f"      {hr_ih.median():5.1f}% "
                  f"({hr_ih.quantile(0.25):.1f}-{hr_ih.quantile(0.75):.1f})")

        # The median is reported rather than the mean because HR is an
        # unbounded ratio: one stride with a small denominator drags a mean
        # anywhere. This says how far apart the two are.
        for _, hr_label, _ in hr_axis_spec:
            hr_v = harmonic_ratio[f'hr_{hr_label}']
            if hr_v.median() > 0 and abs(hr_v.mean() - hr_v.median()) > 0.3 * hr_v.median():
                print(f"    ! {hr_label} HR mean {hr_v.mean():.2f} is far from its "
                      f"median {hr_v.median():.2f}: the tail is doing the work.")
                print(f"      This is the unboundedness [Pasciuto2017] describes. "
                      f"Use iHR, which is bounded 0-100%.")

# =============================================================================
# %% Supplementary metrics
# =============================================================================
# Five measures that each answer something the main set does not.
#
# GOAL-EQUIVALENT MANIFOLD [Dingwell2010]
# The one that changes how the rest is read. At a fixed belt speed the task
# goal is unambiguous: keep stride length over stride time equal to belt speed.
# In the (T, L) plane that goal is a LINE, L = v* T, and every stride's
# deviation splits into a component ALONG the line, which does not threaten the
# goal, and one PERPENDICULAR to it, which does.
#
#     delta_par  = ( dT + v* dL ) / sqrt(1 + v*^2)      goal-equivalent
#     delta_perp = ( -v* dT + dL ) / sqrt(1 + v*^2)     goal-relevant
#
# Two walkers with identical total variability can have completely different
# goal-relevant error. The lag-1 autocorrelation of each component says how
# hard the controller works to correct it: strongly negative means tightly
# corrected stride to stride, near zero means left alone.
#
# FOOT PLACEMENT CONTROL [Wang2014]
# Regress where the foot lands on the CoM state at midstance. The R^2 IS the
# measure: it says how much of foot placement is driven by body state, which
# is a statement about the controller rather than about the margin. Healthy
# mediolateral values are above 0.8.
#
# AUTOCORRELATION REGULARITY AND SYMMETRY [MoeNilssen2004]
# From the unbiased autocorrelation of trunk acceleration: the coefficient at
# the one-step lag is step regularity, at the one-stride lag is stride
# regularity, and their ratio is symmetry. Cheaper and more robust than the
# harmonic ratio, and a useful cross-check on it.
#
# SYMMETRY ANGLE [Zifchock2008]
# Bounded and reference-free, unlike the classic symmetry index, which divides
# by one limb and therefore depends on which limb you picked and diverges as
# that denominator approaches zero.
#
# WALK RATIO
# Step length over cadence. Nearly constant within a person across speeds,
# which makes deviation from it sensitive. With BOTH speed and cadence fixed by
# the protocol it is largely determined by the protocol here, so it is reported
# but should not be read as a free control variable.
#
# References
#   [Dingwell2010]    Dingwell, John & Cusumano (2010) PLoS Comput Biol 6(7), e1000856.
#   [Wang2014]        Wang & Srinivasan (2014) Biol Lett 10(9), 20140405.
#   [MoeNilssen2004]  Moe-Nilssen & Helbostad (2004) J Biomech 37, 121-126.
#   [Zifchock2008]    Zifchock et al. (2008) Gait Posture 27(4), 622-627.
# =============================================================================

sup_midstance_fraction = 0.5    # point in stance where the CoM state is read
sup_autocorr_max_lag_s = 3.0

print("\n" + "-" * 74)
print("  SUPPLEMENTARY METRICS")
print("-" * 74)

# -----------------------------------------------------------------------------
# %% goal-equivalent manifold
# -----------------------------------------------------------------------------

def sup_gem_decompose(stride_time, stride_length):
    """Split each stride's deviation into goal-equivalent and goal-relevant.

    The goal is constant speed, so the manifold is the line L = v* T with
    v* the mean speed. Returns the two components and the direction vectors,
    which are orthogonal by construction.
    """
    sup_t = np.asarray(stride_time, float)
    sup_l = np.asarray(stride_length, float)
    sup_ok = np.isfinite(sup_t) & np.isfinite(sup_l)
    if sup_ok.sum() < 10:
        return None
    sup_t, sup_l = sup_t[sup_ok], sup_l[sup_ok]

    sup_vstar = sup_l.mean() / sup_t.mean()
    sup_dt = sup_t - sup_t.mean()
    sup_dl = sup_l - sup_l.mean()
    sup_norm = np.sqrt(1.0 + sup_vstar ** 2)
    return dict(
        v_star=float(sup_vstar),
        parallel=(sup_dt + sup_vstar * sup_dl) / sup_norm,
        perpendicular=(-sup_vstar * sup_dt + sup_dl) / sup_norm,
        n=int(sup_ok.sum()))


def sup_lag1(series):
    """Lag-1 autocorrelation. Negative means stride-to-stride correction."""
    sup_x = np.asarray(series, float)
    sup_x = sup_x[np.isfinite(sup_x)]
    if len(sup_x) < 10:
        return np.nan
    sup_x = sup_x - sup_x.mean()
    sup_d = np.sum(sup_x ** 2)
    return float(np.sum(sup_x[:-1] * sup_x[1:]) / sup_d) if sup_d > 0 else np.nan

goal_equivalent = None
if {'stride_time', 'stride_length'} <= set(stride_series.columns):
    goal_equivalent = sup_gem_decompose(stride_series['stride_time'],
                                        stride_series['stride_length'])
if goal_equivalent:
    sup_par_sd = float(np.std(goal_equivalent['parallel']))
    sup_perp_sd = float(np.std(goal_equivalent['perpendicular']))
    sup_par_ac = sup_lag1(goal_equivalent['parallel'])
    sup_perp_ac = sup_lag1(goal_equivalent['perpendicular'])
    print(f"\n  goal-equivalent manifold  (v* = {goal_equivalent['v_star']:.3f} m/s, "
          f"n = {goal_equivalent['n']})")
    print(f"    goal-equivalent (along the manifold)  SD {1000*sup_par_sd:6.1f} mm   "
          f"lag-1 autocorr {sup_par_ac:+.3f}")
    print(f"    goal-relevant  (across it)            SD {1000*sup_perp_sd:6.1f} mm   "
          f"lag-1 autocorr {sup_perp_ac:+.3f}")
    print(f"    ratio of goal-relevant to goal-equivalent SD: "
          f"{sup_perp_sd/sup_par_sd:.3f}")
    if sup_perp_ac < sup_par_ac - 0.1:
        print(f"    The goal-relevant component is corrected more strongly than the")
        print(f"    goal-equivalent one, which is the signature of a controller")
        print(f"    exploiting the redundancy rather than fighting all variability.")
    else:
        print(f"    The two components are corrected about equally, which is NOT")
        print(f"    what [Dingwell2010] report for healthy treadmill walking.")

# -----------------------------------------------------------------------------
# %% mediolateral foot placement control
# -----------------------------------------------------------------------------

def sup_foot_placement_model(com_ml, com_ml_velocity, foot_ml):
    """Least squares foot placement on CoM state. Returns R^2 and the gains."""
    sup_ok = (np.isfinite(com_ml) & np.isfinite(com_ml_velocity)
              & np.isfinite(foot_ml))
    if sup_ok.sum() < 20:
        return None
    sup_x = np.column_stack([np.ones(sup_ok.sum()),
                             np.asarray(com_ml)[sup_ok],
                             np.asarray(com_ml_velocity)[sup_ok]])
    sup_y = np.asarray(foot_ml)[sup_ok]
    sup_beta, *_ = np.linalg.lstsq(sup_x, sup_y, rcond=None)
    sup_pred = sup_x @ sup_beta
    sup_ss_res = np.sum((sup_y - sup_pred) ** 2)
    sup_ss_tot = np.sum((sup_y - sup_y.mean()) ** 2)
    return dict(r2=float(1 - sup_ss_res / sup_ss_tot) if sup_ss_tot > 0 else np.nan,
                intercept=float(sup_beta[0]), gain_position=float(sup_beta[1]),
                gain_velocity=float(sup_beta[2]), n=int(sup_ok.sum()))

foot_placement = None
sup_other = 'Left' if ss_limb == 'Right' else 'Right'
if (com_position is not None and f'{sup_other}_Heel' in kinematic_positions):
    sup_com_ml, sup_com_v, sup_foot_ml = [], [], []
    for sup_i in range(len(stride_series)):
        sup_hs = int(stride_series['heel_strike_frame_100hz'].iloc[sup_i]) - 1
        sup_to = int(stride_series['toe_off_frame_100hz'].iloc[sup_i]) - 1
        sup_mid = int(sup_hs + sup_midstance_fraction * (sup_to - sup_hs))
        # the NEXT contralateral contact is the foot placement being predicted
        sup_next = [h for h, _ in force_contacts[sup_other]
                    if kin_sample_to_frame(h) - 1 > sup_hs]
        if not sup_next or sup_mid >= len(com_position):
            continue
        sup_land = int(kin_sample_to_frame(sup_next[0])) - 1
        if sup_land >= len(kinematic_positions[f'{sup_other}_Heel']):
            continue
        sup_com_ml.append(com_position[sup_mid, com_ml_axis])
        sup_com_v.append(com_velocity[sup_mid, com_ml_axis])
        sup_foot_ml.append(
            kinematic_positions[f'{sup_other}_Heel'][sup_land, com_ml_axis])
    foot_placement = sup_foot_placement_model(
        np.array(sup_com_ml), np.array(sup_com_v), np.array(sup_foot_ml))

if foot_placement:
    print(f"\n  mediolateral foot placement on CoM state at midstance "
          f"(n = {foot_placement['n']})")
    print(f"    R^2 = {foot_placement['r2']:.3f}   "
          f"position gain {foot_placement['gain_position']:+.3f}, "
          f"velocity gain {foot_placement['gain_velocity']:+.3f} s")
    print(f"    [Wang2014] report over 0.8 for healthy mediolateral placement.")
    if foot_placement['r2'] < 0.5:
        print(f"    ! well under that. Either foot placement is not being driven by")
        print(f"      CoM state here, or the midstance timing needs checking.")

# -----------------------------------------------------------------------------
# %% autocorrelation regularity and symmetry
# -----------------------------------------------------------------------------

def sup_unbiased_autocorr(signal, max_lag):
    """Unbiased autocorrelation, normalised so lag 0 is 1.

    Unbiased -- divide by the number of overlapping samples at each lag, not by
    N -- because the biased form tapers toward zero with lag and would make
    stride regularity look worse than step regularity for that reason alone.
    """
    sup_x = np.asarray(signal, float)
    sup_x = sup_x[np.isfinite(sup_x)]
    sup_x = sup_x - sup_x.mean()
    sup_n = len(sup_x)
    sup_out = np.full(max_lag + 1, np.nan)
    for sup_k in range(max_lag + 1):
        if sup_n - sup_k < 10:
            continue
        sup_out[sup_k] = np.sum(sup_x[:sup_n - sup_k] * sup_x[sup_k:]) / (sup_n - sup_k)
    return sup_out / sup_out[0] if sup_out[0] else sup_out

gait_regularity = None
if hr_signal is not None and len(stride_series) > 10:
    sup_stride_s = float(np.nanmedian(stride_series['stride_time']))
    sup_step_lag = int(round(0.5 * sup_stride_s * kin_kinematic_fs))
    sup_stride_lag = int(round(sup_stride_s * kin_kinematic_fs))
    sup_max_lag = int(sup_autocorr_max_lag_s * kin_kinematic_fs)
    sup_start = int(ss_warmup_s * kin_kinematic_fs)

    sup_rows = []
    for sup_ax, sup_label in ((com_ml_axis, 'ML'), (com_ap_axis, 'AP'),
                              (com_vt_axis, 'VT')):
        sup_ac = sup_unbiased_autocorr(hr_acc[sup_start:, sup_ax], sup_max_lag)
        if not np.isfinite(sup_ac[sup_step_lag:sup_stride_lag + 1]).any():
            continue
        sup_d1 = float(sup_ac[sup_step_lag])
        sup_d2 = float(sup_ac[sup_stride_lag])
        sup_rows.append({'axis': sup_label, 'step_regularity': sup_d1,
                         'stride_regularity': sup_d2,
                         'symmetry': sup_d1 / sup_d2 if sup_d2 else np.nan})
    gait_regularity = pd.DataFrame(sup_rows)

if gait_regularity is not None and len(gait_regularity):
    print(f"\n  autocorrelation regularity (step lag {sup_step_lag}, "
          f"stride lag {sup_stride_lag} samples)")
    print(f"    axis   step reg   stride reg   symmetry")
    for _, sup_r in gait_regularity.iterrows():
        print(f"    {sup_r['axis']:5s}  {sup_r['step_regularity']:8.3f}  "
              f"{sup_r['stride_regularity']:10.3f}  {sup_r['symmetry']:9.3f}")
    print(f"    Coefficients near 1 mean each step or stride closely repeats the")
    print(f"    last; a symmetry ratio near 1 means the two steps are equivalent.")

# -----------------------------------------------------------------------------
# %% symmetry angle and walk ratio
# -----------------------------------------------------------------------------

def sup_symmetry_angle(left, right):
    """Zifchock symmetry angle, percent. 0 is symmetric, sign gives direction."""
    if not (np.isfinite(left) and np.isfinite(right)) or right == 0:
        return np.nan
    sup_sa = (45.0 - np.degrees(np.arctan2(left, right))) / 90.0 * 100.0
    return sup_sa - 200.0 if sup_sa > 100.0 else sup_sa

sup_sym_rows = []
for sup_var, sup_src in (('stance_time', None), ('swing_time', None),
                         ('propulsive_impulse_bw_s', kinetic_metrics),
                         ('mos_ml_contact', margin_of_stability),
                         ('mfc_m', trip_risk)):
    if sup_src is None or 'side' not in getattr(sup_src, 'columns', []):
        continue
    if sup_var not in sup_src.columns:
        continue
    sup_l = sup_src[sup_src['side'] == 'Left'][sup_var].mean()
    sup_r = sup_src[sup_src['side'] == 'Right'][sup_var].mean()
    sup_sym_rows.append({'variable': sup_var, 'left': sup_l, 'right': sup_r,
                         'symmetry_angle_pct': sup_symmetry_angle(sup_l, sup_r)})

symmetry_angles = pd.DataFrame(sup_sym_rows)
if len(symmetry_angles):
    print(f"\n  symmetry angle (0% symmetric, bounded, reference-free)")
    for _, sup_r in symmetry_angles.iterrows():
        print(f"    {sup_r['variable']:26s} L {sup_r['left']:+9.4f}  "
              f"R {sup_r['right']:+9.4f}   SA {sup_r['symmetry_angle_pct']:+6.2f}%")

walk_ratio = np.nan
if {'stride_length', 'cadence_spm'} <= set(stride_series.columns):
    sup_step_len = stride_series['stride_length'] / 2.0
    walk_ratio = float(np.nanmedian(sup_step_len / stride_series['cadence_spm']))
    print(f"\n  walk ratio {walk_ratio:.5f} m per step/min "
          f"(step length {np.nanmedian(sup_step_len):.3f} m, "
          f"cadence {np.nanmedian(stride_series['cadence_spm']):.1f} steps/min)")
    print(f"    Both speed and cadence are fixed by this protocol, so the walk")
    print(f"    ratio is largely set by the protocol rather than by the walker.")

print(f"\n  note: whole-body angular momentum is NOT computed here. It needs")
print(f"  segment masses and inertia tensors mapped onto Theia's segment")
print(f"  definitions, which is an anthropometric table this script does not")
print(f"  have. Supply it and WBAM is a natural addition, with the GRF moment")
print(f"  about the CoM as an independent check via dH/dt = M_ext.")
