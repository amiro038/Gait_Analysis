# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 07:53:21 2026

@author: alexm
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


# =============================================================================
# %% Importing libraries 
# =============================================================================
import sys 
sys.path.append(r"C:\Users\alexm\Box\Military Project\Codes\Python\Qualisys_Processing\Treadmill")
import pandas as pd 
import numpy as np 
from pathlib import Path 
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.spatial import cKDTree
from scipy.interpolate import PchipInterpolator, CubicSpline
import json
import apply_binding as ab
from matplotlib.patches import Patch
# =============================================================================
# %% Defining functions 
# =============================================================================
def normalize(indat, numpoints, method='pchip'):
    """
    This function transforms the data into x samples evenly spaced
    throughout the dataset using a specified interpolation method.

    Parameters:
    indat (ndarray): Input data array.
    numpoints (int): Number of points for normalization.
    method (str): Interpolation method ('pchip' or 'spline').

    Returns:
    normdat (ndarray): Normalized data array.
    """
    # Ensure 2D shape (nframes, ncols)
    indat = np.asarray(indat)
    if indat.ndim == 1:
        indat = indat.reshape(-1, 1)

    nframes, ncols = indat.shape
    index = np.linspace(1, nframes, num=nframes)
    cycle = np.linspace(1, nframes, num=numpoints)

    normdat = np.zeros((numpoints, ncols))

    for i in range(ncols):
        if method == 'pchip':
            interpolator = PchipInterpolator(index, indat[:, i])
        elif method == 'spline':
            interpolator = CubicSpline(index, indat[:, i])
        else:
            raise ValueError("Invalid method. Use 'pchip' or 'spline'.")

        normdat[:, i] = interpolator(cycle)

    if ncols == 1:
        return normdat[:, 0]   # return 1-D
    return normdat





# =============================================================================
# %% Setting up folders and files
# will be a for loop later on 
# =============================================================================
gait_event_file = r"C:\Users\alexm\Box\Military Project\Data\Analysis\DICE_Treadmill\gait_event_outputs\D05_C1_Treadmill_1.3mpers 108bpm_merged_events.csv"
gait_event_path = Path(gait_event_file)

events = pd.read_csv(gait_event_file)

base = gait_event_path.parents[1]
metrics_path = base / "Theia_csv_outputs" / f"{(gait_event_path.stem).replace('_merged_events', '')}_metrics.csv"

kinematic_data = pd.read_csv(metrics_path, sep=None, engine='python', header=1, skiprows=[2,3,4])

# =============================================================================
# ----Force data
# =============================================================================
# The split-belt treadmill gives one plate per limb, so there is no plate hit
# to disambiguate. It is loaded here because the margin of stability depends on
# CoM VELOCITY, and velocity from markers alone is the weak link in that whole
# calculation: differentiating marker positions amplifies noise exactly where
# the within-stride dynamics live. Integrating the GRF is clean there and
# drifts instead at low frequency, so the two get blended in the MoS cell.
#
# The force export names the trial with underscores where the event file uses a
# space, so both spellings are tried.

force_folder = "FP_renamed"     # sibling of Theia_csv_outputs under base
force_fs = 1000                        # Hz, checked against the file below
handrail_contact_n = 15.0              # N above baseline that counts as contact

force_stem = gait_event_path.stem.replace('_merged_events', '')
force_candidates = [base / force_folder / f"{force_stem}.csv",
                    base / force_folder / f"{force_stem.replace(' ', '_')}.csv",
                    base / force_folder / f"{force_stem.replace('_', ' ')}.csv"]
force_path = next((p for p in force_candidates if p.exists()), None)

if force_path is None:
    raise FileNotFoundError(
        "No force file found. Tried:\n  "
        + "\n  ".join(str(p) for p in force_candidates)
        + "\nSet force_folder to wherever the force exports live.")

force_data = pd.read_csv(force_path, index_col=0)

force_time = force_data['Left belt_TIME'].to_numpy()
force_fs_measured = 1.0 / np.median(np.diff(force_time))
if abs(force_fs_measured - force_fs) > 1.0:
    print(f" Force file is {force_fs_measured:.0f} Hz, not the {force_fs} Hz set "
          f"above. Using the measured rate")
    force_fs = force_fs_measured

print(f" Force: {force_path.name}, {len(force_data)} samples, "
      f"{force_time[-1] - force_time[0]:.1f} s at {force_fs:.0f} Hz")

#body weight is the mean total vertical GRF over a steady walking trial, so
#mass comes from the data and cannot be mistyped
force_grf = np.column_stack([force_data['Left belt_Force_' + a].to_numpy()
                             + force_data['Right belt_Force_' + a].to_numpy()
                             for a in 'XYZ'])
body_weight = float(np.mean(force_grf[:, 2]))
body_mass = body_weight / 9.81
print(f" Body weight {body_weight:.1f} N ({body_mass:.1f} kg) from the mean vertical GRF")

#the handrails are instrumented. Their channels carry a small constant offset
#that is NOT contact, so the baseline is removed first. Any real hand force is
#an external force on the body and breaks the CoM-from-GRF calculation, so
#those frames get flagged rather than silently used
handrail_contact = np.zeros(len(force_data), bool)
for side in ('Left', 'Right'):
    rail = np.column_stack([force_data[f'{side} handrail_Force_{a}'].to_numpy()
                            for a in 'XYZ'])
    rail = rail - np.median(rail, axis=0)
    handrail_contact |= np.linalg.norm(rail, axis=1) > handrail_contact_n

if handrail_contact.any():
    print(f" Handrail contact on {100 * handrail_contact.mean():.2f}% of samples; "
          f"the fused CoM velocity is unreliable there")
else:
    print(" No handrail contact, the GRF is the only external force")

#brain check: over a steady trial the walker has no net acceleration in ANY
#axis, so a non-zero horizontal mean is a plate zero offset, not physiology
for a, name in enumerate(('ML', 'AP')):
    if abs(force_grf[:, a].mean()) > 10.0:
        print(f" {name} force averages {force_grf[:, a].mean():+.1f} N over the "
              f"trial, which should be ~0. Check the plate zeroing")

# =============================================================================
# %% Organizing input and output formats 
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

# =============================================================================
# %% Calculating temporal metrics 
# =============================================================================
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
print(gait_event_data[['stride_time', 'swing_percentage', 'cadence']].describe())
print(gait_event_data[['single_support_percentage', 'double_support_percentage']].describe())

# =============================================================================
# %% Now spatial 
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

# #Lets try to do some minimum foot clearance with what we have so far
# foot_clearance = pd.DataFrame()
# foot_clearance['frame'] = kinematic_data['Unnamed: 0']
# foot_clearance['Left_min_z'] = np.where(
#     ~np.isnan(kinematic_data['Left_Heel_Position.2']) & ~np.isnan(kinematic_data['Left_Toes_Position.2']),
#     np.min([kinematic_data['Left_Heel_Position.2'], kinematic_data['Left_Toes_Position.2']], axis=0),
#     np.nan
#     )
# foot_clearance['Right_min_z'] = np.where(
#     ~np.isnan(kinematic_data['Right_Heel_Position.2']) & ~np.isnan(kinematic_data['Right_Toes_Position.2']),
#     np.min([kinematic_data['Right_Heel_Position.2'], kinematic_data['Right_Toes_Position.2']], axis=0),
#     np.nan
#     )

# #toe AP velocity in m/s, .1 is the AP axis and the data is 100 Hz
# foot_clearance['Left_toe_ap_velocity']  = np.gradient(kinematic_data['Left_Toes_Position.1'].to_numpy()) * 100
# foot_clearance['Right_toe_ap_velocity'] = np.gradient(kinematic_data['Right_Toes_Position.1'].to_numpy()) * 100

# #map event frames onto row positions in the kinematic data
# frame_position = pd.Series(np.arange(len(foot_clearance)), index=foot_clearance['frame'].to_numpy())

# gait_event_data['minimum_foot_clearance'] = np.nan
# gait_event_data['minimum_foot_clearance_frame'] = np.nan

# for x in range(len(gait_event_data)):
#     toe_off = gait_event_data['toe_off_kinematic_frame_100hz'][x]
#     next_hs = gait_event_data['next_ipsi_heelstrike'][x]

#     if np.isnan(toe_off) or np.isnan(next_hs) or next_hs <= toe_off:
#         continue
#     if int(toe_off) not in frame_position.index or int(next_hs) not in frame_position.index:
#         print(f" Swing frames not found in kinematic data at index {x}")
#         continue

#     start = frame_position[int(toe_off)]
#     stop  = frame_position[int(next_hs)]

#     side = 'Left' if gait_event_data['support_limb'][x] == 'L' else 'Right'
#     swing_z   = foot_clearance[f'{side}_min_z'].to_numpy()[start:stop+1]
#     swing_vel = foot_clearance[f'{side}_toe_ap_velocity'].to_numpy()[start:stop+1]

#     #keep only the frames where the toe is in the fastest quarter of that swing
#     upper_quartile = swing_vel >= np.nanpercentile(swing_vel, 75)
#     candidate = np.where(upper_quartile & ~np.isnan(swing_z), swing_z, np.nan)

#     if np.all(np.isnan(candidate)):
#         continue

#     gait_event_data.loc[x, 'minimum_foot_clearance'] = np.nanmin(candidate)
#     gait_event_data.loc[x, 'minimum_foot_clearance_frame'] = foot_clearance['frame'].to_numpy()[start + np.nanargmin(candidate)]

# plt.figure()
# plt.plot(gait_event_data['minimum_foot_clearance'])
# plt.title('Minimum foot clearance (m)')

# =============================================================================
# %% Local dynamic stability
# =============================================================================
# makestatelocal.m      -> time normalise the whole block to 100 samples per
#                          stride (spline), THEN delay embed with tau in
#                          normalised samples
# lds_calc.m            -> Rosenstein (1993): nearest neighbour outside
#                          +-0.5 stride, track for ws strides, nanmean of
#                          ln(distance), fit 0-0.5 and 4-10 strides
# lds_calc_mehdizadeh.m -> same with n_neighbours > 1
# https://github.com/SjoerdBruijn/LocalDynamicStability

#Settings
n_strides = 150 
warmup_seconds = 60 #skip the first minute since there's the speed up portion in there 
lds_limb = 'R' #Gonna focus on the right side
samples_per_Stride = 100 #What we're going to normalize to (Bruijn: 100 on average per stride)
kinematic_fs = 100 #Hz, sampling rate of the kinematic data, used for the warm up

tau = 10 #In normalised samples, this is hard set, find the better number with the whole dataset 
embedding_dimensions = 5 #Same as Tau 

ws = 10 #strides of divergence to track (lds_calc.m ws)
period = 1 #dominant period in strides (lds_calc.m period)
n_neighbours = 1 #1 = lds_calc.m, >1 = lds_calc_mehdizadeh.m

fit_window = {
    'S': (0.0, 0.5), 
    'L': (4.0, 10.0),
    }

#Define the state space 
# signal spec is ('Base_Column', axis)
#   axis        'X' -> Base_Column    'Y' -> Base_Column.1    'Z' -> Base_Column.2
# embed  'delay' = Takens embedding, one common tau, total dim = channels * dE
#        'none'  = the listed signals ARE the state variables
#        Bruijn's code embeds one signal. Multi channel spaces embed each
#        channel the same way and stack the columns
# scale  'none'   raw units. Safe for one channel, or for channels sharing a
#                 unit. lambda is invariant to a UNIFORM rescale of the state
#                 space, but NOT to a per channel one. (Bruijn does not scale)
#        'zscore' per channel, so no longer automatically comparable between
#                 conditions unless you divide by a fixed reference SD.
state_spaces = {
    'trunkVel_ML': {'signals': [('Trunk_Linear_Velocity', 'X')],
                    'embed': 'delay', 'dE': embedding_dimensions, 'scale': 'none'},

    'trunkVel_AP': {'signals': [('Trunk_Linear_Velocity', 'Y')],
                    'embed': 'delay', 'dE': embedding_dimensions, 'scale': 'none'},

    'trunkVel_VT': {'signals': [('Trunk_Linear_Velocity', 'Z')],
                    'embed': 'delay', 'dE': embedding_dimensions, 'scale': 'none'},
    'trunkVel_3D_delayed': {'signals': [('Trunk_Linear_Velocity', 'X'), 
                                ('Trunk_Linear_Velocity', 'Y'), 
                                ('Trunk_Linear_Velocity', 'Z')], 
                    'embed': 'delay', 'dE': 3, 'scale': 'none'},
    'trunkVel_3D': {'signals': [('Trunk_Linear_Velocity', 'X'), 
                                ('Trunk_Linear_Velocity', 'Y'), 
                                ('Trunk_Linear_Velocity', 'Z')], 
                    'embed': 'delay', 'dE': 2, 'scale': 'none'}, 
    'trunk_kin_3D': {'signals': [('Trunk_Linear_Velocity', 'X'), 
                                ('Trunk_Linear_Velocity', 'Y'), 
                                ('Trunk_Linear_Velocity', 'Z'), 
                                ('Trunk_Joint_Velocity', 'X'), 
                                ('Trunk_Joint_Velocity', 'Y'), 
                                ('Trunk_Joint_Velocity', 'Z')], 
                    'embed': 'none', 'dE': None, 'scale': 'zscore'},
    'trunk_kin_3D_delayed': {'signals': [('Trunk_Linear_Velocity', 'X'), 
                                ('Trunk_Linear_Velocity', 'Y'), 
                                ('Trunk_Linear_Velocity', 'Z'), 
                                ('Trunk_Joint_Velocity', 'X'), 
                                ('Trunk_Joint_Velocity', 'Y'), 
                                ('Trunk_Joint_Velocity', 'Z')], 
                    'embed': 'delay', 'dE': 2, 'scale': 'zscore'},
    }

lds_primary = 'trunkVel_AP'

# =============================================================================
# ----Set up kinematic data range 
# =============================================================================
lds_hs_all = (gait_event_data.loc[gait_event_data['support_limb'] == lds_limb,
                                  'initial_contact_kinematic_frame_100hz']
              .dropna().astype(int).sort_values().to_numpy())

lds_start_index, lds_start_frame = min(
    enumerate(lds_hs_all), 
    key=lambda x: abs(x[1] - warmup_seconds * kinematic_fs)
)

#Brain check that you have enough strides to keep going, This should never be a problem, you have 9min worth of gait data
lds_short = False
if len(lds_hs_all) - lds_start_index < n_strides + 1: 
    lds_short = True
    n_strides = len(lds_hs_all) - lds_start_index - 1
    print(f" Only {n_strides} strides after the warm up. lambda depends on "
          f"the amount of data, so this trial is NOT comparable to trials run "
          f"at the full stride count")
    
lds_end_index = lds_start_index + n_strides
lds_end_frame = lds_hs_all[lds_end_index]

#Heel strikes are FRAME numbers but raw[...] indexes ROWS, so map one onto the
#other and make sure no frames are missing inside the window
lds_frame = pd.to_numeric(kinematic_data['Unnamed: 0'], errors='coerce').to_numpy()
lds_frame_to_row = pd.Series(np.arange(len(kinematic_data)), index=lds_frame)
lds_start_row = int(lds_frame_to_row[lds_start_frame])
lds_end_row = int(lds_frame_to_row[lds_end_frame])
if np.any(np.diff(lds_frame[lds_start_row:lds_end_row + 1]) != 1):
    raise ValueError(" Kinematic frames are not contiguous inside the LDS window")

#As in makestatelocal.m, the whole block from the first to the last heel strike
#becomes n_strides * 100 samples
n_samples = n_strides * samples_per_Stride

#lds_calc.m settings, in normalised samples
ws_samples = int(round(ws * samples_per_Stride))
half_period = int(round(0.5 * period * samples_per_Stride))

#Make the state spaces and calculate lambda
lds_results = []

for ss_name, ss in state_spaces.items():
    signals = ss['signals']
    dE = ss['dE'] if ss['embed'] == 'delay' else 1
    n_embed = dE 

    ncols = len(signals) * n_embed
    nrows = n_samples - tau * (n_embed - 1) #embedding loses the last (dE-1)*tau rows

    mat = np.zeros((nrows, ncols))
    col_idx = 0
    skip = False

    for base_col, axis in signals: #not 'base', that would overwrite your output path
        suffix = {'X': '', 'Y': '.1', 'Z': '.2'}[axis]
        colname = base_col + suffix

        if colname not in kinematic_data.columns:
            print(f"Column {colname} missing → skipping state space {ss_name}")
            skip = True
            break

        raw = pd.to_numeric(kinematic_data[colname], errors='coerce').to_numpy(float)
        segment = raw[lds_start_row:lds_end_row + 1] #first to last heel strike, inclusive

        if not np.isfinite(segment).all():
            print(f"{colname} has NaN inside the window → skipping state space {ss_name}")
            skip = True
            break

        #normalise first. 'spline' matches interp1(...,'spline') in makestatelocal.m
        normalised = normalize(segment, n_samples, 'spline')
        if ss['scale'] == 'zscore':
            normalised = normalised / np.std(normalised)

        #then embed, delay in normalised samples
        for t in range(n_embed):
            start = t * tau
            end   = start + nrows
            mat[:, col_idx] = normalised[start:end]
            col_idx += 1

    if skip:
        continue

    state_spaces[ss_name]['Matrix'] = mat

    # -------------------------------------------------------------------------
    # Divergence curve, lds_calc.m
    # -------------------------------------------------------------------------
    #pad with NaN so pairs that run off the end drop out of the mean
    padded = np.vstack([mat, np.full((ws_samples, ncols), np.nan)])
    log_sum = np.zeros(ws_samples) #running sum and count = nanmean over points,
    log_cnt = np.zeros(ws_samples) #without holding an nrows x ws matrix

    for i_t in range(nrows):
        difference = np.sum((mat - mat[i_t]) ** 2, axis=1)

        #discard everything within half a period of i_t (includes i_t itself)
        start_index = max(0, i_t - half_period)
        stop_index = min(nrows - 1, i_t + half_period)
        difference[start_index:stop_index + 1] = np.inf

        #nearest neighbour(s)
        if n_neighbours == 1:
            index = [int(np.argmin(difference))]
        else:
            index = np.argpartition(difference, n_neighbours)[:n_neighbours]
            index = index[np.argsort(difference[index])]

        #track the divergence and store
        for i_nn in index:
            dist = np.sqrt(np.sum((padded[i_t:i_t + ws_samples] -
                                   padded[i_nn:i_nn + ws_samples]) ** 2, axis=1))
            with np.errstate(divide='ignore'):
                ln_dist = np.log(dist)
            ok = np.isfinite(ln_dist)
            log_sum[ok] += ln_dist[ok]
            log_cnt[ok] += 1

    divergence = log_sum / log_cnt
    state_spaces[ss_name]['divergence'] = divergence

    # -------------------------------------------------------------------------
    # Least squares fits, lds_calc.m
    # -------------------------------------------------------------------------
    #Same indexing as MATLAB: S fits lags 0..49 against t = 0.01..0.50,
    #L fits lags 399..999 against t = 4.00..10.00. The one sample offset moves
    #the intercept, not the slope
    lds_row = {'trial': gait_event_path.stem.replace('_merged_events', ''),
               'state_space': ss_name,
               'is_primary': ss_name == lds_primary,
               'n_strides': n_strides,
               'stride_count_short': lds_short,
               'first_frame': lds_start_frame, 'last_frame': lds_end_frame,
               'mean_stride_time_s': (lds_end_frame - lds_start_frame) / n_strides / kinematic_fs,
               'tau': tau, 'dE': ss['dE'], 'total_dimension': ncols,
               'n_neighbours': n_neighbours}

    for tag, (w0, w1) in fit_window.items():
        L_start = max(int(round(w0 * samples_per_Stride)), 1)
        L_stop = min(int(round(w1 * samples_per_Stride)), ws_samples)
        t_fit = np.arange(L_start, L_stop + 1) / samples_per_Stride
        y_fit = divergence[L_start - 1:L_stop]
        state_spaces[ss_name][f'fit_{tag}'] = np.polyfit(t_fit, y_fit, 1)
        lds_row[f'lambda_{tag}'] = state_spaces[ss_name][f'fit_{tag}'][0]
        lds_row[f'R2_{tag}'] = np.corrcoef(t_fit, y_fit)[0, 1] ** 2

    lds_results.append(lds_row)
    print(f"  {ss_name:22s} lambda_S = {lds_row['lambda_S']:.4f}  "
          f"lambda_L = {lds_row['lambda_L']:.4f}  R2_S = {lds_row['R2_S']:.2f}")

lds_results = pd.DataFrame(lds_results)

# # =============================================================================
# # ----Save
# # =============================================================================
# lds_out_path = base / "LDS_outputs" / f"{lds_results['trial'][0]}_LDS.csv"
# lds_out_path.parent.mkdir(parents=True, exist_ok=True)
# lds_results.to_csv(lds_out_path, index=False)

# #curves too, so fit windows can be changed later without re-running
# lds_curves = pd.DataFrame({'strides': np.arange(1, ws_samples + 1) / samples_per_Stride})
# for ss_name, ss in state_spaces.items():
#     if 'divergence' in ss:
#         lds_curves[ss_name] = ss['divergence']
# lds_curves.to_csv(lds_out_path.with_name(lds_out_path.stem + '_curves.csv'), index=False)
# print(f" LDS results -> {lds_out_path}")

# =============================================================================
# ----Plot, as lds_calc.m does with plotje = 1
# =============================================================================
lds_done = [ss_name for ss_name in state_spaces if 'divergence' in state_spaces[ss_name]]
n_cols = 3
n_rows = int(np.ceil(len(lds_done) / n_cols))
plt.figure(figsize=(4.2 * n_cols, 3.4 * n_rows))

t_plot = np.arange(1, ws_samples + 1) / samples_per_Stride
for p, ss_name in enumerate(lds_done):
    ss = state_spaces[ss_name]
    plt.subplot(n_rows, n_cols, p + 1)
    plt.plot(t_plot, ss['divergence'], label='Divergence Curve')
    for tag, colour in [('S', 'm'), ('L', 'r')]:
        w0, w1 = fit_window[tag]
        mask = (t_plot >= max(w0, 1 / samples_per_Stride)) & (t_plot <= w1)
        plt.plot(t_plot[mask], np.polyval(ss[f'fit_{tag}'], t_plot[mask]), colour,
                 label=f"Lambda {tag}: {ss[f'fit_{tag}'][0]:.3f}")
    plt.title(ss_name + ('  [PRIMARY]' if ss_name == lds_primary else ''), fontsize=9)
    plt.xlabel('Time (strides)')
    plt.ylabel('Ln(divergence)')
    plt.legend(fontsize=7, frameon=False, loc='lower right')

plt.tight_layout()

# =============================================================================
# %% Minimum foot clearance 
# =============================================================================

mos_gravity          = 9.81
mos_pendulum_mode    = 'per_frame'   # 'per_frame' CoM-to-ankle, or 'leg_length'
mos_posed_suffix = '_posed.npz'      # written next to the metrics csv by apply_binding
mos_binding_path = Path(r"C:\Users\alexm\Box\Military Project\Data\Analysis\DICE_Treadmill\Foot bindings\D05_foot_mesh_binding.npz")
mos_stance_fraction  = (0.0, 1.0)    # portion of stance searched for the minimum
mos_report_marker_bos = True         # also compute the marker-based boundary

# apply_binding.py writes <TRIAL>_mesh.pkl beside the metrics csv: a DataFrame
# indexed by frame with a (segment, vertex, axis) column MultiIndex, in Theia
# world metres. If it is missing the cell still runs, on markers alone, and
# says so rather than silently changing what it measures.

mos_posed_path = metrics_path.with_name(metrics_path.stem + mos_posed_suffix)

print("\n" + "-" * 74)
print("  MARGIN OF STABILITY")
print("-" * 74)

# The .npz holds poses, not positions: vertices are one matrix multiply away,
# so they are rebuilt here rather than read. Selection is by SEGMENT, matching
# what mesh_positions[segment] used to give -- not by mesh, see below.
mos_foot_vertices = {}
mos_have_mesh = False

if mos_posed_path.exists() and mos_binding_path.exists():
    mos_bind = np.load(mos_binding_path, allow_pickle=False)
    mos_seg_names = json.loads(str(mos_bind["meta"]))["segments"]
    mos_seg_of = mos_bind["vertex_segment"]
    mos_have_mesh = True

    for mos_seg, mos_side in (('left_foot', 'Left'), ('right_foot', 'Right')):
        if mos_seg not in mos_seg_names:
            continue
        mos_idx = np.flatnonzero(mos_seg_of == mos_seg_names.index(mos_seg))
        mos_foot_vertices[mos_side], _ = ab.vertex_tracks(
            str(mos_posed_path), vertices=mos_idx,
            binding=str(mos_binding_path), dtype=np.float32)

    mos_n_frames = len(next(iter(mos_foot_vertices.values())))
    print(f"  mesh: {mos_posed_path.name}  {mos_n_frames} frames, "
          f"{sum(v.shape[1] for v in mos_foot_vertices.values())} vertices")
    for mos_side, mos_v in mos_foot_vertices.items():
        print(f"    {mos_side}: {mos_v.shape[1]} vertices")
else:
    mos_missing = mos_posed_path if not mos_posed_path.exists() else mos_binding_path
    print(f"  ! {mos_missing.name} not found. Falling back to marker-based")
    print("    boundaries, which sit inside the foot and will overstate the")
    print("    margin. Run apply_binding.py on this trial to fix that.")

foot_clearance = pd.DataFrame()
foot_clearance['frame'] = kinematic_data['Unnamed: 0']
foot_clearance['Left_min_z'] = np.nanmin(mos_foot_vertices['Left'][:, :, 2], axis=1)
foot_clearance['Right_min_z'] = np.nanmin(mos_foot_vertices['Right'][:, :, 2], axis=1)

# =============================================================================
# ----Plotting the minimum foot position 
# =============================================================================
n_plot = 5000
plot_frames = pd.to_numeric(foot_clearance['frame'], errors='coerce').to_numpy()[:n_plot]
plot_z = foot_clearance['Right_min_z'].to_numpy()[:n_plot]
f0, f1 = plot_frames[0], plot_frames[-1]

right = gait_event_data[gait_event_data['support_limb'] == 'R']

fig, ax = plt.subplots(figsize=(12, 4))
for hs, to, next_hs in zip(right['initial_contact_kinematic_frame_100hz'],
                           right['toe_off_kinematic_frame_100hz'],
                           right['next_ipsi_heelstrike']):
    #stance: heel strike -> toe off
    if not np.isnan(hs) and not np.isnan(to) and to > hs and to >= f0 and hs <= f1:
        ax.axvspan(hs, to, color='red', alpha=0.15, lw=0)
    #swing: toe off -> next heel strike, same limb
    if not np.isnan(to) and not np.isnan(next_hs) and next_hs > to and next_hs >= f0 and to <= f1:
        ax.axvspan(to, next_hs, color='green', alpha=0.15, lw=0)

ax.plot(plot_frames, plot_z, color='k', lw=1)
ax.set_xlim(f0, f1)
ax.set_xlabel('Frame')
ax.set_ylabel('Min z (m)')
ax.set_title('Minimum mesh position right foot')
ax.legend(handles=[Patch(color='red', alpha=0.15, label='Stance'),
                   Patch(color='green', alpha=0.15, label='Swing')],
          loc='upper right')
plt.tight_layout()

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

###### Hmm there are negative values in here.... I know I can't accomodate for the bending of the toes (or maybe I can? does Theia do that?)
# so there is a grain of salt to take here. Maybe I only use a set of vertices under the heel to toes? 
#To be figured out 
print([gait_event_data[['minimum_foot_clearance']]].describe())
# =============================================================================
# %% Margin of Stability
# =============================================================================
# Hof, Gazendam & Sinke (2005) J Biomech 38, 1-8:
#     xCoM = x + v / w0,   w0 = sqrt(g / l),   MoS = u_max - xCoM
#
# The CoM is not just somewhere, it is going somewhere. If control stopped now,
# inverted pendulum dynamics would carry the body to the extrapolated CoM, and
# MoS is how much base of support is left beyond that point.
#
# Two things this cell has to get right that are easy to get wrong:
#
# 1. THE BELT. On a treadmill the CoM has near-zero mean AP velocity in the lab
#    frame while the feet cycle backwards. Feed the lab-frame velocity into
#    xCoM and the AP margin is meaningless. Belt speed is MEASURED here from
#    the stance foot, which is stationary relative to the belt during foot
#    flat, so its lab velocity IS the belt velocity. It cannot be read off the
#    centre of pressure: the CoP travels heel to toe along the foot at the same
#    time, and the two cannot be separated.
#
# 2. THE BOUNDARY. u_max is normally the ankle joint centre or a toe marker,
#    both of which sit INSIDE the foot, centimetres from the border that would
#    actually stop a fall. The posed mesh gives the real boundary. The marker
#    version is computed alongside so the difference is a number, not a guess.
#
# Sign conventions are derived rather than assumed, so nothing depends on which
# way the lab axes happen to point:
#   ML  lateral = the direction from the swing foot toward the stance foot
#   AP  anterior = the direction of travel, taken from the belt measurement
#
# On the AP margin, be careful which instant is meant. At heel strike the newly
# loaded foot is out AHEAD of the body, so the anterior boundary is beyond the
# xCoM and MoS_AP at contact is normally POSITIVE. It is the minimum over
# stance that goes negative, during single support, as the xCoM passes the
# stance toe -- that is the "walking is controlled falling forward" part. A
# positive value at contact is not a red flag; a positive MINIMUM is.

# --- participant values ------------------------------------------------------
# All optional. Leave any at None and the cell derives what it can and says so.
#   mass          only a CROSS-CHECK. Mass already comes from the mean vertical
#                 GRF, which cannot be mistyped, so a disagreement here points
#                 at a plate calibration problem rather than at a typo.
#   leg length    greater trochanter to floor, or ASIS to medial malleolus.
#                 Two uses: Hof's original omega_0 takes leg length rather than
#                 a per-frame CoM-to-ankle distance, and margins normalised to
#                 leg length are what make them comparable BETWEEN people.
#   height        the other common normaliser; also a sanity check on leg length
#                 (leg length is usually 0.50-0.55 of height).
#   foot length   an independent check on the mesh: the ankle-to-toe distance
#                 the mesh reports should land near it.
participant_mass_kg    = 80
participant_leg_length = 0.8769      # m
participant_height     = 1.7688      # m
participant_foot_length = None     # m

mos_savgol_window = 11     # matches the LDS cell: one stage, smooth + derivative
mos_savgol_poly   = 3
mos_flat_foot     = (0.30, 0.70)   # portion of stance used as "foot flat"
mos_crossover_hz  = 0.5    # markers below this, force plates above
mos_antialias_hz  = 40.0   # before decimating 1000 Hz -> 100 Hz

# =============================================================================
# ----Participant values, and what they are checked against
# =============================================================================
if participant_mass_kg is not None:
    mos_mass_gap = 100 * abs(participant_mass_kg - body_mass) / participant_mass_kg
    print(f"  mass: {participant_mass_kg:.1f} kg entered, {body_mass:.1f} kg from "
          f"the GRF ({mos_mass_gap:.1f}% apart)")
    if mos_mass_gap > 3.0:
        print(f" Over 3% apart. The GRF value is the one used everywhere; a gap "
              f"this size usually means the plates need re-calibrating")

if participant_leg_length is not None and participant_height is not None:
    mos_ratio = participant_leg_length / participant_height
    if not (0.45 < mos_ratio < 0.60):
        print(f" Leg length is {mos_ratio:.2f} of height, outside the usual "
              f"0.50-0.55. Check one of the two")

# =============================================================================
# ----Belt speed, measured from the stance foot
# =============================================================================
mos_belt_slopes = []
for x in range(len(gait_event_data)):
    hs = gait_event_data['initial_contact_kinematic_frame_100hz'][x]
    to = gait_event_data['toe_off_kinematic_frame_100hz'][x]
    if np.isnan(hs) or np.isnan(to) or to <= hs:
        continue
    if int(hs) not in frame_position.index or int(to) not in frame_position.index:
        continue
    start = frame_position[int(hs)]
    stop  = frame_position[int(to)]
    a = start + int(mos_flat_foot[0] * (stop - start))
    b = start + int(mos_flat_foot[1] * (stop - start))
    if b - a < 5:
        continue
    side = 'Left' if gait_event_data['support_limb'][x] == 'L' else 'Right'
    seg = kinematic_data[f'{side}_Heel_Position.1'].to_numpy()[a:b]
    if not np.isfinite(seg).all():
        continue
    mos_belt_slopes.append(np.polyfit(np.arange(len(seg)) / kinematic_fs, seg, 1)[0])

mos_belt_slopes = np.array(mos_belt_slopes)
mos_belt_speed = float(np.median(np.abs(mos_belt_slopes)))
mos_travel_sign = -float(np.sign(np.median(mos_belt_slopes)))   # foot goes backwards

print(f"  belt speed from the stance foot: {mos_belt_speed:.3f} m/s "
      f"(IQR {np.percentile(np.abs(mos_belt_slopes), 25):.3f}-"
      f"{np.percentile(np.abs(mos_belt_slopes), 75):.3f}, "
      f"n = {len(mos_belt_slopes)} stances)")
print(f"  travel direction: {'+' if mos_travel_sign > 0 else '-'}Y")

#brain check: the spread across stances says whether the belt really is steady
if np.std(np.abs(mos_belt_slopes)) > 0.1 * mos_belt_speed:
    print(f" Belt speed varies by {100*np.std(np.abs(mos_belt_slopes))/mos_belt_speed:.0f}% "
          f"across stances, which is more than a steady belt should")

# =============================================================================
# ----CoM position, and velocity fused from markers and force plates
# =============================================================================
# Velocity is the weak link in the whole margin calculation, and there are two
# ways to get it, each bad in a different band:
#
#   differentiating the marker CoM is trustworthy at low frequency -- it cannot
#     drift -- but differentiation amplifies noise, so it is poor high up,
#     which is exactly where the within-stride dynamics are
#   integrating the GRF is trustworthy at high frequency -- force is measured
#     directly at 1000 Hz with no differentiation -- but any residual offset
#     integrates into a drifting velocity, so it is poor low down
#
# A complementary filter takes each where it is good, with the SAME cutoff on
# both branches so their transfer functions sum to one and nothing is counted
# twice or lost:
#
#     v = lowpass(v_markers) + [ v_force - lowpass(v_force) ]
#
# It is worth the trouble. Velocity error propagates into the extrapolated CoM
# as error / w0, which for a 1 m pendulum is error / 3.13, so 140 mm/s of
# velocity error is 45 mm of xCoM error against an ML margin only 30-100 mm
# wide. On synthetic data with 2 mm marker noise this cuts that from 45 mm to
# about 2.5 mm.

mos_com = np.column_stack([kinematic_data['Whole_body_COG'].to_numpy(),
                           kinematic_data['Whole_body_COG.1'].to_numpy(),
                           kinematic_data['Whole_body_COG.2'].to_numpy()])

#CoM acceleration straight from Newton. Subtracting each column's mean removes
#gravity from the vertical channel (its mean IS mg) and the plate zero offset
#from the horizontals, which is the right treatment for both: over a steady
#trial the walker has no net acceleration in any axis
mos_accel_1k = (force_grf - force_grf.mean(axis=0)) / body_mass

#anti-alias BEFORE decimating. Taking every 10th sample without filtering folds
#everything above 50 Hz back into the band that matters, and heel strike
#transients are broadband
mos_aa_b, mos_aa_a = butter(4, mos_antialias_hz / (force_fs / 2), 'low')
mos_step = int(round(force_fs / kinematic_fs))
mos_accel_100 = filtfilt(mos_aa_b, mos_aa_a, mos_accel_1k, axis=0)[::mos_step]
mos_handrail_100 = handrail_contact[::mos_step]

#align force samples onto kinematic ROWS through the frame numbers, rather than
#assuming the two files start together. Frame f sits at force sample (f-1)*10,
#which was checked against GRF-detected contacts and agrees to about 3 ms
mos_frame_no = pd.to_numeric(kinematic_data['Unnamed: 0'], errors='coerce').to_numpy()
mos_force_row = np.full(len(kinematic_data), -1, int)
mos_ok_row = np.isfinite(mos_frame_no)
mos_force_row[mos_ok_row] = (mos_frame_no[mos_ok_row] - 1).astype(int)
mos_ok_row &= (mos_force_row >= 0) & (mos_force_row < len(mos_accel_100))

mos_accel = np.full((len(kinematic_data), 3), np.nan)
mos_accel[mos_ok_row] = mos_accel_100[mos_force_row[mos_ok_row]]
mos_handrail_row = np.zeros(len(kinematic_data), bool)
mos_handrail_row[mos_ok_row] = mos_handrail_100[mos_force_row[mos_ok_row]]

print(f"  {mos_ok_row.sum()} of {len(kinematic_data)} kinematic frames have force data")
if mos_ok_row.sum() < 0.95 * len(kinematic_data):
    print(f" Over 5% of kinematic frames fall outside the force file. Those steps "
          f"will use markers alone; check the two exports cover the same trial")

#filtfilt cannot cross a NaN, so fill first and remember where
mos_gap = ~np.isfinite(mos_com).all(axis=1) | ~np.isfinite(mos_accel).all(axis=1)
if mos_gap.any():
    print(f" Filling {mos_gap.sum()} frames with gaps in the CoM or force before "
          f"filtering")
    mos_index = np.arange(len(mos_com))
    for mos_a in range(3):
        for mos_arr in (mos_com, mos_accel):
            mos_good = np.isfinite(mos_arr[:, mos_a])
            if mos_good.any():
                mos_arr[~mos_good, mos_a] = np.interp(
                    mos_index[~mos_good], mos_index[mos_good], mos_arr[mos_good, mos_a])

#one Savitzky-Golay stage smooths and differentiates the markers, same as the
#LDS cell, so the bandwidth is single valued
mos_v_markers = savgol_filter(mos_com, mos_savgol_window, mos_savgol_poly,
                              deriv=1, delta=1.0 / kinematic_fs, axis=0)

#trapezoidal, not np.cumsum. A rectangle-rule integral lands half a sample
#late, which at stride harmonics is about 10 mm/s of velocity error -- small
#against noisy markers but pure loss when they are clean. The trapezoid is
#centred on the sample grid and cuts that to about 0.2 mm/s.
mos_v_force = np.zeros_like(mos_accel)
mos_v_force[1:] = np.cumsum((mos_accel[:-1] + mos_accel[1:]) / 2, axis=0) / kinematic_fs
mos_v_force -= mos_v_force.mean(axis=0)

mos_cb, mos_ca = butter(2, mos_crossover_hz / (kinematic_fs / 2), 'low')
mos_com_velocity = (filtfilt(mos_cb, mos_ca, mos_v_markers, axis=0)
                    + mos_v_force - filtfilt(mos_cb, mos_ca, mos_v_force, axis=0))

# in the belt frame the walker travels forward even though the lab-frame mean
# is about zero. Every AP margin needs this.
mos_com_velocity_belt = mos_com_velocity.copy()
mos_com_velocity_belt[:, 1] += mos_travel_sign * mos_belt_speed

print(f"  CoM AP velocity: lab mean {np.nanmean(mos_com_velocity[:, 1]):+.3f} m/s "
      f"(should be ~0), belt frame {np.nanmean(mos_com_velocity_belt[:, 1]):+.3f} m/s")

#how much did the force plates actually change things? This is the number that
#propagates into every margin below, divided by w0
mos_delta = 1000 * np.nanstd(mos_com_velocity - mos_v_markers, axis=0)
print(f"  fusion moved the velocity by {mos_delta[0]:.1f} / {mos_delta[1]:.1f} / "
      f"{mos_delta[2]:.1f} mm/s RMS (ML / AP / VT) vs markers alone")
print(f"    at a 1 m pendulum that is {mos_delta[0]/np.sqrt(9.81):.1f} mm of xCoM "
      f"in ML, against a margin of a few centimetres")

mos_ankle = {s: np.column_stack([kinematic_data[f'{s}_Ankle_Position'].to_numpy(),
                                 kinematic_data[f'{s}_Ankle_Position.1'].to_numpy(),
                                 kinematic_data[f'{s}_Ankle_Position.2'].to_numpy()])
             for s in ('Left', 'Right')}

# =============================================================================
# ----Margin of stability, per step
# =============================================================================
gait_event_data['mos_ml_contact'] = np.nan
gait_event_data['mos_ml_min'] = np.nan
gait_event_data['mos_ap_contact'] = np.nan
gait_event_data['mos_ap_min'] = np.nan
gait_event_data['mos_ml_contact_marker'] = np.nan
gait_event_data['mos_pendulum_length'] = np.nan
gait_event_data['mos_handrail_contact'] = False

mos_forward = np.array([0.0, mos_travel_sign])
mos_width = []

for x in range(len(gait_event_data)):
    hs = gait_event_data['initial_contact_kinematic_frame_100hz'][x]
    to = gait_event_data['toe_off_kinematic_frame_100hz'][x]

    if np.isnan(hs) or np.isnan(to) or to <= hs:
        continue
    if int(hs) not in frame_position.index or int(to) not in frame_position.index:
        continue

    start = frame_position[int(hs)]
    stop  = frame_position[int(to)]
    a = start + int(mos_stance_fraction[0] * (stop - start))
    b = start + int(mos_stance_fraction[1] * (stop - start))
    if b <= a:
        continue

    side  = 'Left' if gait_event_data['support_limb'][x] == 'L' else 'Right'
    other = 'Right' if side == 'Left' else 'Left'

    #lateral has to be PERPENDICULAR to the direction of travel. The raw
    #ankle-to-ankle vector will not do: at heel strike the feet are about 0.7 m
    #apart in AP and only 0.15 m in ML, so that vector points roughly 97%
    #FORWARD, and the "lateral" boundary then lands on the toe instead of the
    #lateral border of the shoe -- which shows up as an ankle-to-boundary gap
    #of ~230 mm (a foot LENGTH) rather than the 50-80 mm it should be.
    #
    #Travel is (0, mos_travel_sign) in the (ML, AP) plane, so the horizontal
    #perpendicular is (mos_travel_sign, 0). Which of its two signs points away
    #from the midline is decided by where the stance ankle sits along it, so
    #nothing depends on whether lab +X is left or right.
    mos_perp = np.array([mos_travel_sign, 0.0])
    mos_offset = (mos_ankle[side][start, :2] - mos_ankle[other][start, :2]) @ mos_perp
    if not np.isfinite(mos_offset) or abs(mos_offset) < 1e-6:
        continue
    mos_lat = mos_perp * np.sign(mos_offset)
    mos_width.append(abs(mos_offset))

    #pendulum length is the CoM to stance ankle distance, per frame
    mos_length = np.linalg.norm(mos_com[a:b + 1] - mos_ankle[side][a:b + 1], axis=1)
    if mos_pendulum_mode == 'leg_length':
        #Hof's original omega_0 uses leg length. Prefer the measured value; fall
        #back to the median CoM-to-ankle distance if none was entered
        mos_length = np.full_like(mos_length, participant_leg_length
                                  if participant_leg_length else np.nanmedian(mos_length))
    mos_w0 = np.sqrt(mos_gravity / mos_length)

    #extrapolated CoM in the ground plane
    mos_xcom = mos_com[a:b + 1, :2] + mos_com_velocity_belt[a:b + 1, :2] / mos_w0[:, None]

    #boundary from the mesh: furthest vertex along each direction
    if mos_have_mesh and side in mos_foot_vertices:
        mos_verts = mos_foot_vertices[side][a:b + 1, :, :2]
        mos_u_ml = np.nanmax(mos_verts @ mos_lat, axis=1)
        mos_u_ap = np.nanmax(mos_verts @ mos_forward, axis=1)
    else:
        mos_u_ml = mos_ankle[side][a:b + 1, :2] @ mos_lat
        mos_u_ap = mos_ankle[side][a:b + 1, :2] @ mos_forward

    mos_ml = mos_u_ml - mos_xcom @ mos_lat
    mos_ap = mos_u_ap - mos_xcom @ mos_forward

    if not np.isfinite(mos_ml).any():
        continue

    #a loaded handrail is an external force the GRF does not see, so the fused
    #velocity is not trustworthy on those frames
    if mos_handrail_row[a:b + 1].any():
        gait_event_data.loc[x, 'mos_handrail_contact'] = True

    gait_event_data.loc[x, 'mos_ml_contact'] = mos_ml[0]
    gait_event_data.loc[x, 'mos_ml_min'] = np.nanmin(mos_ml)
    gait_event_data.loc[x, 'mos_ap_contact'] = mos_ap[0]
    gait_event_data.loc[x, 'mos_ap_min'] = np.nanmin(mos_ap)
    gait_event_data.loc[x, 'mos_pendulum_length'] = mos_length[0]

    #the same margin with the ankle joint centre as the boundary, so the cost
    #of using an interior point instead of the real border is measurable
    if mos_report_marker_bos and mos_have_mesh:
        gait_event_data.loc[x, 'mos_ml_contact_marker'] = (
            mos_ankle[side][a, :2] @ mos_lat - mos_xcom[0] @ mos_lat)

# =============================================================================
# ----Brain checks and plots
# =============================================================================
print(f"\n  {gait_event_data['mos_ml_contact'].notna().sum()} of "
      f"{len(gait_event_data)} steps produced a margin")
print(gait_event_data.groupby('support_limb')[
    ['mos_ml_contact']].agg(['mean', 'std', 'count']))
print(gait_event_data.groupby('support_limb')[
    ['mos_ml_min']].agg(['mean', 'std', 'count']))
print(gait_event_data.groupby('support_limb')[
    ['mos_ap_contact']].agg(['mean', 'std', 'count']))
#ML at contact should be positive and a few centimetres. AP should be negative.
mos_ml_mean = gait_event_data['mos_ml_contact'].mean()
if not (0.0 < mos_ml_mean < 0.20):
    print(f" ML margin averages {1000*mos_ml_mean:.0f} mm, outside the 0-200 mm "
          f"a healthy adult should show. Check the lateral direction and the belt")
#positive at contact is expected; it is the stance MINIMUM that should go
#negative as the xCoM passes the stance toe
if gait_event_data['mos_ap_min'].mean() > 0:
    print(f" AP margin never goes negative (stance minimum averages "
          f"{1000*gait_event_data['mos_ap_min'].mean():+.0f} mm). During single "
          f"support the xCoM should pass the stance toe, so check the travel "
          f"direction and the belt speed")

#step width from the ankles. If this is not a few centimetres the lateral
#direction is wrong, which is the failure that makes ML look like AP
mos_width = np.array(mos_width)
print(f"  step width from the ankles: {1000*np.median(mos_width):.0f} mm "
      f"(IQR {1000*np.percentile(mos_width,25):.0f}-"
      f"{1000*np.percentile(mos_width,75):.0f})")
if not (0.02 < np.median(mos_width) < 0.40):
    print(f" That is not a plausible step width. The ML axis assumption is wrong")

if mos_report_marker_bos and mos_have_mesh:
    mos_gap = 1000 * (gait_event_data['mos_ml_contact']
                      - gait_event_data['mos_ml_contact_marker'])
    print(f"\n  mesh boundary vs ankle joint centre: the mesh gives a margin "
          f"{mos_gap.mean():.1f} +- {mos_gap.std():.1f} mm larger")
    print(f"    That gap is the distance from the ankle joint centre out to the "
          f"real lateral border of the shoe.")
    if mos_gap.mean() > 120:
        print(f" A gap over ~120 mm means the boundary is finding the TOE, not the "
              f"lateral border, which happens when the lateral direction has "
              f"picked up an AP component. Expect 50-80 mm.")

if gait_event_data['mos_handrail_contact'].any():
    print(f"\n {gait_event_data['mos_handrail_contact'].sum()} steps had handrail "
          f"contact during stance. Their margins rest on a fused velocity whose "
          f"assumption (GRF is the only external force) does not hold")

#Between-subject comparison needs a normaliser: a 40 mm margin means something
#different on a 0.75 m leg than on a 0.95 m one. Within one person it changes
#nothing, so both are kept.
mos_normaliser = participant_leg_length or participant_height
if mos_normaliser:
    mos_tag = 'leg length' if participant_leg_length else 'height'
    for mos_col in ('mos_ml_contact', 'mos_ml_min', 'mos_ap_contact', 'mos_ap_min'):
        gait_event_data[mos_col + '_norm'] = gait_event_data[mos_col] / mos_normaliser
    print(f"\n  margins also stored normalised to {mos_tag} ({mos_normaliser:.3f} m), "
          f"as *_norm columns")
    print(f"    ML at contact {gait_event_data['mos_ml_contact_norm'].mean():.4f} "
          f"of {mos_tag}")
else:
    print(f"\n  no leg length or height entered, so margins are in metres only. "
          f"Enter one before comparing between participants")

#the mesh should agree with a tape measure on foot length
if participant_foot_length and mos_have_mesh:
    mos_len_seen = []
    for mos_s in ('Left', 'Right'):
        if mos_s in mos_foot_vertices:
            mos_v0 = mos_foot_vertices[mos_s][len(mos_foot_vertices[mos_s]) // 2]
            mos_len_seen.append(np.ptp(mos_v0[:, 1]))
    mos_len_seen = float(np.mean(mos_len_seen))
    print(f"  mesh foot length {1000*mos_len_seen:.0f} mm vs "
          f"{1000*participant_foot_length:.0f} mm entered "
          f"({100*abs(mos_len_seen-participant_foot_length)/participant_foot_length:.1f}% apart)")
    if abs(mos_len_seen - participant_foot_length) > 0.03:
        print(f" Over 30 mm apart. Either the scan is not this participant's shoe "
              f"or the mesh scaling is off")

plt.figure()
plt.plot(gait_event_data['mos_ml_contact'], label='at contact')
plt.plot(gait_event_data['mos_ml_min'], label='minimum over stance')
plt.axhline(0, color='k', lw=0.8)
plt.title('Mediolateral margin of stability (m)')
plt.xlabel('Step')
plt.legend()

plt.figure()
plt.plot(gait_event_data['mos_ap_contact'], label='at contact')
plt.plot(gait_event_data['mos_ap_min'], label='minimum over stance')
plt.axhline(0, color='k', lw=0.8)
plt.title('Anteroposterior margin of stability (m)  -- negative is normal')
plt.xlabel('Step')
plt.legend()





plt.figure()
plt.plot(kinematic_data['Left_Ankle_Position'][:5000])
plt.plot(kinematic_data['Left_Ankle_Position.1'][:5000])
plt.plot(kinematic_data['Left_Ankle_Position.2'][:5000])

plt.figure()
plt.plot(force_data['Left belt_Force_X'][:50000])
plt.plot(force_data['Left belt_Force_Y'][:50000])
plt.plot(force_data['Left belt_Force_Z'][:50000])

# =============================================================================
# ----Birds-eye video of a representative stride
# =============================================================================
# The margin of stability lives in the ground plane -- both the xCoM and the
# base of support boundary are horizontal positions -- so a birds-eye view is
# the only one that shows it without foreshortening.
#
# WHY THE BELT FRAME
# Drawn in the lab frame the feet sweep 0.9 m backwards during stance while the
# body stays put, which is unreadable. Positions are shifted forward by
# belt_speed * t so the foot stays planted during stance and the body advances,
# exactly as overground walking looks. This does NOT distort any margin: the
# margin is a DIFFERENCE between two positions at the same instant, so adding
# the same translation to both leaves it unchanged.
#
# WHAT IS DRAWN, and why each earns its place
#   foot outlines      convex hull of the mesh vertices in the ground plane,
#                      filled while loaded, open while swinging
#   base of support    hull of whatever is loaded -- one foot in single
#                      support, both during double support
#   CoM and xCoM       with the arrow between them, which IS v / omega_0. The
#                      whole concept is that arrow, so it is drawn rather than
#                      described
#   the margins        ML and AP drawn as the actual measured segments, from
#                      the xCoM out to the boundary, so a negative margin is
#                      visibly an xCoM outside the polygon
#   centre of pressure where the load actually is. This is the one that shows
#                      the BoS shrinking as the heel lifts, which the whole-foot
#                      hull does not capture -- the known limitation, made
#                      visible instead of written down
#   the time series    MoS_ML and MoS_AP across the stride with a cursor, so
#                      the birds-eye frame and the number are tied together,
#                      with the contact and minimum instants marked
# =============================================================================

from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D

vid_limb        = 'R'          # which limb's stride to show
vid_fps         = 20           # 111 frames at 100 Hz -> about 5.5 s, slow enough to read
vid_dpi         = 110
vid_trail       = True         # leave footprints behind
vid_show_cop    = True         # needs the force data; auto-aligns, see below
vid_pad         = 0.25         # m of margin around the action
# The base of support is the CONTACT area, so strictly the boundary should come
# from the sole rather than the whole shoe silhouette. On this participant's
# mesh that is only a 0.4-4.2 mm difference, because the upper barely overhangs
# the sole -- so it is principled rather than important. None uses every vertex.
vid_sole_height = 0.02         # m above the foot's lowest point, or None
vid_out = base / "MoS_outputs" / f"{gait_event_path.stem.replace('_merged_events','')}_MoS_stride"

# =============================================================================
# ----Pick the most representative stride
# =============================================================================
# "Median" needs a definition. A stride that is median on stride time may be
# unusual on step width, so strides are scored by how far they sit from the
# median on ALL the variables that matter, in IQR units so the variables are
# comparable, and the smallest total wins.

vid_vars = [v for v in ('stride_time', 'step_width', 'mos_ml_contact',
                        'mos_ap_contact', 'mos_ml_min')
            if v in gait_event_data.columns]

vid_pool = gait_event_data[(gait_event_data['support_limb'] == vid_limb)
                           & gait_event_data[vid_vars].notna().all(axis=1)
                           & gait_event_data['next_ipsi_heelstrike'].notna()].copy()

vid_score = np.zeros(len(vid_pool))
for vid_v in vid_vars:
    vid_col = vid_pool[vid_v].to_numpy(float)
    vid_iqr = np.percentile(vid_col, 75) - np.percentile(vid_col, 25)
    if vid_iqr > 0:
        vid_score += np.abs(vid_col - np.median(vid_col)) / vid_iqr
vid_pool['typicality'] = vid_score

vid_row = vid_pool.loc[vid_pool['typicality'].idxmin()]
vid_idx = int(vid_pool['typicality'].idxmin())

print("\n" + "-" * 74)
print("  REPRESENTATIVE STRIDE")
print("-" * 74)
print(f"  stride at index {vid_idx}, scored on {', '.join(vid_vars)}")
for vid_v in vid_vars:
    vid_col = gait_event_data.loc[gait_event_data['support_limb'] == vid_limb, vid_v]
    print(f"    {vid_v:18s} {vid_row[vid_v]:8.4f}   median {vid_col.median():8.4f}")

vid_hs = int(vid_row['initial_contact_kinematic_frame_100hz'])
vid_to = int(vid_row['toe_off_kinematic_frame_100hz'])
vid_next = int(vid_row['next_ipsi_heelstrike'])
vid_start, vid_stop = frame_position[vid_hs], frame_position[vid_next]
vid_frames = np.arange(vid_start, vid_stop + 1)
print(f"  frames {vid_hs} to {vid_next} ({len(vid_frames)} samples, "
      f"{len(vid_frames)/kinematic_fs:.2f} s), toe off at {vid_to}")

# =============================================================================
# ----Belt-frame positions, and the contact state of each foot
# =============================================================================
vid_shift = mos_travel_sign * mos_belt_speed * (vid_frames - vid_frames[0]) / kinematic_fs

vid_side = 'Left' if vid_limb == 'L' else 'Right'
vid_other = 'Right' if vid_side == 'Left' else 'Left'

#loaded state per foot, from the event table rather than a force threshold, so
#the picture agrees with the numbers the cell above produced
vid_loaded = {s: np.zeros(len(vid_frames), bool) for s in ('Left', 'Right')}
for vid_x in range(len(gait_event_data)):
    vid_s = 'Left' if gait_event_data['support_limb'][vid_x] == 'L' else 'Right'
    vid_a = gait_event_data['initial_contact_kinematic_frame_100hz'][vid_x]
    vid_b = gait_event_data['toe_off_kinematic_frame_100hz'][vid_x]
    if np.isnan(vid_a) or np.isnan(vid_b):
        continue
    vid_in = (vid_frames >= frame_position.get(int(vid_a), -1)) & \
             (vid_frames <= frame_position.get(int(vid_b), -1))
    vid_loaded[vid_s] |= vid_in

#foot outlines in the ground plane, one hull per foot per frame
vid_hull = {s: [] for s in ('Left', 'Right')}
for vid_s in ('Left', 'Right'):
    for vid_i, vid_f in enumerate(vid_frames):
        vid_v = mos_foot_vertices[vid_s][vid_f]
        if vid_sole_height is not None:
            vid_keep = (vid_v[:, 2] - vid_v[:, 2].min()) <= vid_sole_height
            vid_v = vid_v[vid_keep] if vid_keep.sum() >= 3 else vid_v
        vid_p = vid_v[:, :2].copy()
        vid_p[:, 1] += vid_shift[vid_i]
        vid_h = ConvexHull(vid_p)
        vid_hull[vid_s].append(vid_p[vid_h.vertices])

# =============================================================================
# ----CoM, xCoM and the margins over the stride
# =============================================================================
vid_com = mos_com[vid_frames, :2].copy()
vid_com[:, 1] += vid_shift
vid_vel = mos_com_velocity_belt[vid_frames, :2]

vid_len = np.linalg.norm(mos_com[vid_frames] - mos_ankle[vid_side][vid_frames], axis=1)
vid_w0 = np.sqrt(mos_gravity / vid_len)
vid_xcom = vid_com + vid_vel / vid_w0[:, None]

vid_perp = np.array([mos_travel_sign, 0.0])
vid_offset = (mos_ankle[vid_side][vid_start, :2]
              - mos_ankle[vid_other][vid_start, :2]) @ vid_perp
vid_lat = vid_perp * np.sign(vid_offset)
vid_fwd = np.array([0.0, mos_travel_sign])

vid_ml = np.full(len(vid_frames), np.nan)
vid_ap = np.full(len(vid_frames), np.nan)
for vid_i in range(len(vid_frames)):
    vid_pts = vid_hull[vid_side][vid_i]
    vid_ml[vid_i] = np.max(vid_pts @ vid_lat) - vid_xcom[vid_i] @ vid_lat
    vid_ap[vid_i] = np.max(vid_pts @ vid_fwd) - vid_xcom[vid_i] @ vid_fwd

vid_stance = vid_frames <= frame_position[vid_to]
vid_min_i = int(np.nanargmin(np.where(vid_stance, vid_ml, np.nan)))
print(f"  ML margin {1000*vid_ml[0]:.0f} mm at contact, "
      f"minimum {1000*vid_ml[vid_min_i]:.0f} mm at "
      f"{100*vid_min_i/np.sum(vid_stance):.0f}% of stance")

# =============================================================================
# ----Centre of pressure, aligned to the mesh
# =============================================================================
# The force file's CoP is in the plate frame, and whether that frame coincides
# with Theia's world frame is not documented anywhere. Rather than assume, the
# constant offset that best matches the CoP to the loaded foot's centroid over
# the whole trial is solved for, and the residual reported. A small residual
# says the two systems agree about where the foot is, which is a free
# end-to-end check on the binding AND the plate alignment.

vid_cop = None
#`dir()` only sees the current scope, which is not the same thing as the
#notebook globals when a cell is run on its own -- so ask globals() directly
vid_have_force = 'force_data' in globals()
vid_cop_cols = (vid_have_force and all(f'{s} belt_COP_{a}' in force_data.columns
                                       for s in ('Left', 'Right') for a in 'XY'))
if vid_show_cop and not vid_have_force:
    print("  CoP overlay OFF: force_data is not loaded. Run the force cell first")
elif vid_show_cop and not vid_cop_cols:
    print("  CoP overlay OFF: the force file has no belt_COP_X / _Y columns")
if vid_show_cop and vid_cop_cols:
    vid_step = int(round(force_fs / kinematic_fs))
    vid_cop_xy, vid_cen_xy = [], []
    for vid_x in range(0, len(gait_event_data), 5):     # every 5th step is plenty
        vid_s = 'Left' if gait_event_data['support_limb'][vid_x] == 'L' else 'Right'
        vid_a = gait_event_data['initial_contact_kinematic_frame_100hz'][vid_x]
        vid_b = gait_event_data['toe_off_kinematic_frame_100hz'][vid_x]
        if np.isnan(vid_a) or np.isnan(vid_b):
            continue
        vid_r0, vid_r1 = frame_position.get(int(vid_a), -1), frame_position.get(int(vid_b), -1)
        if vid_r0 < 0 or vid_r1 < 0:
            continue
        vid_mid = (vid_r0 + vid_r1) // 2
        vid_fs_row = int(vid_mid * vid_step)
        if vid_fs_row >= len(force_data):
            continue
        vid_cop_xy.append([force_data[f'{vid_s} belt_COP_X'].to_numpy()[vid_fs_row] / 1000,
                           force_data[f'{vid_s} belt_COP_Y'].to_numpy()[vid_fs_row] / 1000])
        vid_cen_xy.append(mos_foot_vertices[vid_s][vid_mid][:, :2].mean(axis=0))
    print(f"  CoP alignment used {len(vid_cop_xy)} sampled stances")
    if len(vid_cop_xy) <= 20:
        print(f"  CoP overlay OFF: only {len(vid_cop_xy)} stances could be sampled, "
              f"too few to solve the plate-to-mocap offset")
    if len(vid_cop_xy) > 20:
        vid_cop_xy = np.array(vid_cop_xy)
        vid_cen_xy = np.array(vid_cen_xy)
        vid_cop_offset = np.median(vid_cen_xy - vid_cop_xy, axis=0)
        vid_resid = np.linalg.norm(vid_cen_xy - vid_cop_xy - vid_cop_offset, axis=1)
        print(f"  CoP to mesh alignment: offset ({1000*vid_cop_offset[0]:+.0f}, "
              f"{1000*vid_cop_offset[1]:+.0f}) mm, residual "
              f"{1000*np.median(vid_resid):.0f} mm median")
        if np.median(vid_resid) > 0.05:
            print(f"    Over 50 mm. The plates and the mocap do not agree about where")
            print(f"    the foot is; treat the CoP overlay as indicative only")
        vid_cop = np.full((len(vid_frames), 2), np.nan)
        vid_cop_skipped = 0
        for vid_i, vid_f in enumerate(vid_frames):
            if not vid_loaded[vid_side][vid_i]:
                continue
            vid_fs_row = int(vid_f * vid_step)
            if vid_fs_row >= len(force_data):
                vid_cop_skipped += 1
                continue
            vid_cop[vid_i] = [force_data[f'{vid_side} belt_COP_X'].to_numpy()[vid_fs_row] / 1000
                              + vid_cop_offset[0],
                              force_data[f'{vid_side} belt_COP_Y'].to_numpy()[vid_fs_row] / 1000
                              + vid_cop_offset[1] + vid_shift[vid_i]]

    if vid_cop is not None:
        vid_n_cop = int(np.isfinite(vid_cop).all(axis=1).sum())
        print(f"  CoP drawn on {vid_n_cop} of {len(vid_frames)} frames "
              f"({int(vid_loaded[vid_side].sum())} are stance)")
        if vid_n_cop == 0:
            print(f"  ! none drawn. {vid_cop_skipped} frames fell past the end of "
                  f"the force file, so it is shorter than the kinematics")

# =============================================================================
# ----Render
# =============================================================================
#Frame on the FEET and the CoM, which are always where the walker is. The xCoM
#is included only if it is within a sane distance of them: if the margins are
#wrong for some other reason the xCoM can land metres away, and letting it set
#the limits turns the whole picture into an unreadable sliver rather than
#showing you the stride and the problem.
vid_all = np.vstack([np.vstack(vid_hull['Left']), np.vstack(vid_hull['Right']),
                     vid_com])
vid_lo, vid_hi = vid_all.min(axis=0), vid_all.max(axis=0)
vid_near = np.all((vid_xcom > vid_lo - 1.0) & (vid_xcom < vid_hi + 1.0), axis=1)
if vid_near.any():
    vid_lo = np.minimum(vid_lo, vid_xcom[vid_near].min(axis=0))
    vid_hi = np.maximum(vid_hi, vid_xcom[vid_near].max(axis=0))
if not vid_near.all():
    print(f"  ! the xCoM is off the map on {np.sum(~vid_near)} of "
          f"{len(vid_near)} frames -- the view is framed on the feet instead. "
          f"That usually means the CoM and the feet are in different frames")
vid_lo, vid_hi = vid_lo - vid_pad, vid_hi + vid_pad

if vid_cop is not None and np.isfinite(vid_cop).all(axis=1).any():
    vid_seen = vid_cop[np.isfinite(vid_cop).all(axis=1)]
    if not np.all((vid_seen >= vid_lo) & (vid_seen <= vid_hi)):
        print(f"  ! the CoP falls outside the drawn area, so it will not be "
              f"visible. Its range is ML {vid_seen[:,0].min():.2f}..{vid_seen[:,0].max():.2f}, "
              f"AP {vid_seen[:,1].min():.2f}..{vid_seen[:,1].max():.2f} m against a "
              f"view of ML {vid_lo[0]:.2f}..{vid_hi[0]:.2f}, AP {vid_lo[1]:.2f}..{vid_hi[1]:.2f}")

fig = plt.figure(figsize=(13, 7.5))
ax_map = fig.add_axes((0.06, 0.38, 0.91, 0.54))
ax_ts = fig.add_axes((0.06, 0.08, 0.91, 0.21))
vid_t = (vid_frames - vid_frames[0]) / kinematic_fs

def vid_draw(i):
    ax_map.clear()
    #footprints already laid down
    if vid_trail:
        for vid_j in range(0, i, 4):
            for vid_s, vid_c in (('Left', '#c9d6e8'), ('Right', '#e8d3c9')):
                if vid_loaded[vid_s][vid_j]:
                    ax_map.add_patch(MplPolygon(vid_hull[vid_s][vid_j][:, ::-1], closed=True,
                                                facecolor=vid_c, edgecolor='none',
                                                alpha=0.25, zorder=1))
    #base of support: the hull of everything loaded right now
    vid_bos = [vid_hull[s][i] for s in ('Left', 'Right') if vid_loaded[s][i]]
    if vid_bos:
        vid_stack = np.vstack(vid_bos)
        vid_h = ConvexHull(vid_stack)
        vid_poly = vid_stack[vid_h.vertices][:, ::-1]
        ax_map.add_patch(MplPolygon(vid_poly, closed=True, facecolor='#f2e5b8',
                                    edgecolor='none', alpha=0.55, zorder=2))
        #outline drawn ON TOP of the feet, otherwise a filled stance foot hides
        #it and the double-support polygon never shows
        ax_map.add_patch(MplPolygon(vid_poly, closed=True, facecolor='none',
                                    edgecolor='#8a6d1f', lw=2.0, ls='--', zorder=5))
    for vid_s, vid_c in (('Left', '#2a5d9f'), ('Right', '#a8442a')):
        ax_map.add_patch(MplPolygon(
            vid_hull[vid_s][i][:, ::-1], closed=True,
            facecolor=vid_c if vid_loaded[vid_s][i] else 'none',
            edgecolor=vid_c, lw=2.0,
            alpha=0.75 if vid_loaded[vid_s][i] else 1.0, zorder=3))

    #the margins, drawn as the segments actually measured
    vid_b_ml = np.max(vid_hull[vid_side][i] @ vid_lat)
    vid_b_ap = np.max(vid_hull[vid_side][i] @ vid_fwd)
    vid_x = vid_xcom[i]
    vid_end_ml = vid_x + (vid_b_ml - vid_x @ vid_lat) * vid_lat
    vid_end_ap = vid_x + (vid_b_ap - vid_x @ vid_fwd) * vid_fwd
    ax_map.plot([vid_x[1], vid_end_ml[1]], [vid_x[0], vid_end_ml[0]],
                color='#1b7a3d', lw=3.0, zorder=6, solid_capstyle='butt')
    ax_map.plot([vid_x[1], vid_end_ap[1]], [vid_x[0], vid_end_ap[0]],
                color='#7a4fb5', lw=3.0, zorder=6, solid_capstyle='butt')

    #CoM -> xCoM. That arrow IS v / omega_0
    ax_map.annotate('', xy=vid_xcom[i][::-1], xytext=vid_com[i][::-1], zorder=7,
                    arrowprops=dict(arrowstyle='-|>', color='#111111', lw=1.8))
    ax_map.plot(vid_com[i][1], vid_com[i][0], 'o', ms=11, color='#111111', zorder=8)
    ax_map.plot(vid_xcom[i][1], vid_xcom[i][0], 'o', ms=11, mfc='none',
                mec='#111111', mew=2.2, zorder=8)

    if vid_cop is not None and np.isfinite(vid_cop[i]).all():
        ax_map.plot(vid_cop[i][1], vid_cop[i][0], 'X', ms=11, color='#d94f04',
                    mew=0, zorder=9)

    vid_phase = 'STANCE' if vid_loaded[vid_side][i] else 'SWING'
    vid_note = ''
    if i == 0:
        vid_note = '   <- heel strike, MoS at contact read here'
    elif i == vid_min_i:
        vid_note = '   <- minimum MoS over stance'
    elif vid_frames[i] == frame_position[vid_to]:
        vid_note = '   <- toe off'
    ax_map.set_title(f"{vid_side} stride, {vid_phase}   t = {vid_t[i]:.2f} s"
                     f"   MoS_ML {1000*vid_ml[i]:+.0f} mm   "
                     f"MoS_AP {1000*vid_ap[i]:+.0f} mm{vid_note}", fontsize=11)
    ax_map.set_xlim(vid_lo[1], vid_hi[1])
    ax_map.set_ylim(vid_lo[0], vid_hi[0])
    ax_map.set_aspect('equal')
    ax_map.set_xlabel('direction of travel, belt frame (m)  ->')
    ax_map.set_ylabel('mediolateral (m)')
    ax_map.grid(alpha=0.15)

    ax_ts.clear()
    ax_ts.plot(vid_t, 1000 * vid_ml, color='#1b7a3d', lw=1.8, label='MoS ML')
    ax_ts.plot(vid_t, 1000 * vid_ap, color='#7a4fb5', lw=1.8, label='MoS AP')
    ax_ts.axhline(0, color='k', lw=0.8)
    ax_ts.axvline(vid_t[i], color='#888888', lw=1.4)
    ax_ts.axvline(vid_t[0], color='#1b7a3d', ls=':', lw=1.2)
    ax_ts.axvline(vid_t[vid_min_i], color='#1b7a3d', ls='--', lw=1.2)
    ax_ts.axvspan(vid_t[0], vid_t[np.sum(vid_stance) - 1], color='#cccccc', alpha=0.25)
    ax_ts.annotate('contact', (vid_t[0], 0), xytext=(4, 6), textcoords='offset points',
                   fontsize=8, color='#1b7a3d')
    ax_ts.annotate('minimum', (vid_t[vid_min_i], 1000 * vid_ml[vid_min_i]),
                   xytext=(4, -14), textcoords='offset points', fontsize=8,
                   color='#1b7a3d')
    ax_ts.set_xlim(vid_t[0], vid_t[-1])
    ax_ts.set_xlabel('time through the stride (s)   shaded = stance')
    ax_ts.set_ylabel('margin (mm)')
    ax_ts.legend(fontsize=8, ncol=2, frameon=True, framealpha=0.9,
                 loc='lower left')
    ax_ts.grid(alpha=0.15)

fig.legend(handles=[
    Line2D([], [], color='#2a5d9f', lw=2, label='left foot'),
    Line2D([], [], color='#a8442a', lw=2, label='right foot'),
    Line2D([], [], color='#b59f4a', lw=2, label='base of support'),
    Line2D([], [], color='#111111', marker='o', ls='none', label='CoM'),
    Line2D([], [], color='#111111', marker='o', mfc='none', ls='none', label='xCoM'),
    Line2D([], [], color='#d94f04', marker='X', ls='none', label='centre of pressure'),
    Line2D([], [], color='#1b7a3d', lw=2.5, label='ML margin'),
    Line2D([], [], color='#7a4fb5', lw=2.5, label='AP margin'),
], loc='upper center', ncol=8, fontsize=8, frameon=False,
    bbox_to_anchor=(0.5, 1.0))

vid_out.parent.mkdir(parents=True, exist_ok=True)
try:
    from matplotlib.animation import FFMpegWriter
    import shutil
    vid_exe = shutil.which('ffmpeg')
    if vid_exe is None:
        import imageio_ffmpeg
        vid_exe = imageio_ffmpeg.get_ffmpeg_exe()
    plt.rcParams['animation.ffmpeg_path'] = vid_exe
    vid_writer, vid_ext = FFMpegWriter(fps=vid_fps, bitrate=3000), '.mp4'
except Exception:
    from matplotlib.animation import PillowWriter
    vid_writer, vid_ext = PillowWriter(fps=vid_fps), '.gif'
    print("  no ffmpeg, writing a gif instead (pip install imageio-ffmpeg for mp4)")

vid_path = str(vid_out) + vid_ext
with vid_writer.saving(fig, vid_path, vid_dpi):
    for vid_i in range(len(vid_frames)):
        vid_draw(vid_i)
        vid_writer.grab_frame()
print(f"  wrote {vid_path}")

#a still of the two instants the numbers come from, for a figure
for vid_i, vid_tag in ((0, 'contact'), (vid_min_i, 'minimum')):
    vid_draw(vid_i)
    fig.savefig(f"{vid_out}_{vid_tag}.png", dpi=150)
    print(f"  wrote {vid_out}_{vid_tag}.png")



























# =============================================================================
# %% Margin of instability and trip risk
# =============================================================================
# Schulz (2017) J Biomech 55, 107-112.
#
# Minimum foot clearance says how CLOSE the foot came to the ground. It does
# not say what would have happened if the foot had caught. Trip risk needs
# both, because a low clearance in a mechanically stable configuration is
# recoverable and the same clearance in an unstable one is not.
#
# MARGIN OF INSTABILITY is the AP margin of stability with three changes:
#   1. the CONTINUOUS trajectory through swing, not a stance minimum
#   2. the boundary is the most anterior point on EITHER foot, because that is
#      where the front of the base of support WOULD be if the swing foot came
#      down right now
#   3. stable (positive) values clamped to zero, then negated, so MoI >= 0 and
#      bigger means more destabilising
#
#       MoI(t) = max( -MoS_AP(t), 0 )        in mm
#       trip risk(t) = MoI(t) / MFC(t)       dimensionless
#       TRI = integral of trip risk over swing
#
# Time is deliberately NOT normalised, so a longer swing gives a larger TRI.
# That is Schulz's choice and it is kept so the numbers stay comparable.
#
# Direction of the effect: higher MFC means LESS risk, higher TRI means MORE.
# Schulz's central result is that the two move OPPOSITELY with gait speed, and
# that TRI is the one that tracks real trip-related falls.
# =============================================================================

tri_local_window    = 2      # frames either side that a local minimum must beat
tri_speed_quantile  = 0.75   # Schulz's "upper quartile" foot speed gate
tri_min_clearance_mm = 1.0   # MoI/MFC blows up as MFC approaches zero
tri_swing_trim      = 0.02   # fraction of swing trimmed at each end

print("\n" + "-" * 74)
print("  MARGIN OF INSTABILITY AND TRIP RISK")
print("-" * 74)

# Anterior boundary of the base of support: the furthest forward point on
# EITHER foot, per Schulz. mos_foot_vertices is already in Theia world metres.
tri_n_frames = min(len(v) for v in mos_foot_vertices.values())
tri_anterior = np.nanmax(np.vstack([
    mos_travel_sign * mos_foot_vertices[s][:tri_n_frames, :, 1].max(axis=1)
    for s in mos_foot_vertices]), axis=0)

gait_event_data['moi_peak_mm'] = np.nan
gait_event_data['moi_mean_mm'] = np.nan
gait_event_data['trip_risk_peak'] = np.nan
gait_event_data['trip_risk_integral'] = np.nan
gait_event_data['tri_window_s'] = np.nan
gait_event_data['mfc_local_minima'] = np.nan
gait_event_data['mfc_vertex'] = np.nan

tri_example = None       # keep one swing's traces for the figure below
tri_reversed = 0         # swings where the acceleration bounds came out reversed

for x in range(len(gait_event_data)):
    toe_off = gait_event_data['toe_off_kinematic_frame_100hz'][x]
    next_hs = gait_event_data['next_ipsi_heelstrike'][x]
    if np.isnan(toe_off) or np.isnan(next_hs) or next_hs <= toe_off:
        continue
    if int(toe_off) not in frame_position.index or int(next_hs) not in frame_position.index:
        continue

    start = frame_position[int(toe_off)]
    stop = frame_position[int(next_hs)]
    pad = int(tri_swing_trim * (stop - start))
    swing = np.arange(start + pad, stop - pad + 1)
    if len(swing) < 10 or swing[-1] >= tri_n_frames:
        continue

    side = 'Left' if gait_event_data['support_limb'][x] == 'L' else 'Right'

    # --- clearance over the whole shoe, per frame ----------------------------
    clearance = np.nanmin(mos_foot_vertices[side][swing, :, 2], axis=1)
    lowest_vertex = np.nanargmin(
        np.where(np.isfinite(mos_foot_vertices[side][swing, :, 2]),
                 mos_foot_vertices[side][swing, :, 2], np.inf), axis=1)

    # --- foot speed, for Schulz's upper-quartile gate ------------------------
    centroid = np.nanmean(mos_foot_vertices[side][swing], axis=1)
    speed = np.linalg.norm(np.gradient(centroid, 1.0 / kinematic_fs, axis=0), axis=1)
    fast = speed >= np.nanquantile(speed, tri_speed_quantile)

    # --- rear-of-shoe clearance, to reject false minima at the midfoot -------
    # Schulz compares the TOE segment's clearance against the HEEL segment's.
    # Taking the heel MARKER height instead mixes a joint centre with a mesh
    # minimum: the marker sits well above the sole, so the test either never
    # fires or fires on everything depending on the offset. Splitting the shoe
    # front/rear along its own long axis keeps both sides on the same footing.
    ap_local = (mos_foot_vertices[side][swing, :, 1]
                - np.nanmean(mos_foot_vertices[side][swing, :, 1], axis=1)[:, None])
    is_rear = (mos_travel_sign * ap_local) < 0
    rear_z = np.where(is_rear, mos_foot_vertices[side][swing, :, 2], np.inf).min(axis=1)

    # --- local minima meeting all three criteria -----------------------------
    candidates = []
    for i in range(tri_local_window, len(clearance) - tri_local_window):
        before = clearance[i - tri_local_window:i]
        after = clearance[i + 1:i + tri_local_window + 1]
        # <= not <, because a minimum falling between two samples gives two
        # equal neighbours and a strict test then finds nothing at all
        if not (np.all(clearance[i] <= before) and np.all(clearance[i] <= after)):
            continue
        if not (np.any(clearance[i] < before) and np.any(clearance[i] < after)):
            continue
        if not fast[i] or rear_z[i] < clearance[i]:
            continue
        candidates.append(i)
    # collapse runs of adjacent indices (a flat minimum) to their first
    candidates = [c for j, c in enumerate(candidates)
                  if j == 0 or c != candidates[j - 1] + 1]

    gait_event_data.loc[x, 'mfc_local_minima'] = len(candidates)
    if not candidates:
        continue                  # a real non-MTC cycle: NaN, not a global min
    mfc_i = min(candidates, key=lambda i: clearance[i])
    gait_event_data.loc[x, 'mfc_vertex'] = int(lowest_vertex[mfc_i])

    # --- the margin of instability through swing -----------------------------
    pend = np.linalg.norm(mos_com[swing] - mos_ankle[side][swing], axis=1)
    w0 = np.sqrt(mos_gravity / pend)
    xcom_ap = (mos_travel_sign * mos_com[swing, 1]
               + mos_travel_sign * mos_com_velocity_belt[swing, 1] / w0)
    moi = 1000.0 * np.maximum(-(tri_anterior[swing] - xcom_ap), 0.0)

    clear_mm = np.maximum(1000.0 * clearance, tri_min_clearance_mm)
    risk = moi / clear_mm

    # --- integration bounds: peak acceleration to peak deceleration ----------
    # of the MFC POINT. Tracking the single identified point, not whichever
    # vertex is momentarily lowest, which would jump when the low point moves.
    point = mos_foot_vertices[side][swing, int(lowest_vertex[mfc_i]), :]
    point_speed = np.linalg.norm(np.gradient(point, 1.0 / kinematic_fs, axis=0), axis=1)
    point_acc = np.gradient(point_speed, 1.0 / kinematic_fs)
    # Schulz takes peak acceleration first, terminal deceleration second. If a
    # noisy point returns them the other way round, ORDER them rather than drop
    # the swing silently -- the window is still lift-off to landing either way,
    # and the count of reversals is reported below as a data-quality signal.
    a_i, b_i = int(np.nanargmax(point_acc)), int(np.nanargmin(point_acc))
    if b_i < a_i:
        a_i, b_i = b_i, a_i
        tri_reversed += 1

    gait_event_data.loc[x, 'moi_peak_mm'] = np.nanmax(moi)
    gait_event_data.loc[x, 'moi_mean_mm'] = np.nanmean(moi)
    gait_event_data.loc[x, 'trip_risk_peak'] = np.nanmax(risk)
    if b_i > a_i:
        gait_event_data.loc[x, 'trip_risk_integral'] = np.nansum(risk[a_i:b_i + 1]) / kinematic_fs
        gait_event_data.loc[x, 'tri_window_s'] = (b_i - a_i) / kinematic_fs

    # keep a mid-trial swing for the explanatory figure
    if tri_example is None and x > len(gait_event_data) // 2 and b_i > a_i:
        tri_example = dict(t=(swing - swing[0]) / kinematic_fs, clear=clear_mm,
                           moi=moi, risk=risk, acc=point_acc, a=a_i, b=b_i,
                           mfc_i=mfc_i, side=side, index=x)

# --- report -------------------------------------------------------------
tri_has = gait_event_data['trip_risk_integral'].notna()
print(f"  {tri_has.sum()} of {len(gait_event_data)} swings produced a TRI")
print(gait_event_data.groupby('support_limb')[
    ['minimum_foot_clearance', 'moi_peak_mm', 'trip_risk_integral']].agg(['mean', 'std']))
if tri_reversed:
    print(f"  ! {tri_reversed} swings had the acceleration bounds reversed and were "
          f"re-ordered. A few is normal; many means the MFC point is noisy")
print(f"  swings with no qualifying MFC event: "
      f"{int((gait_event_data['mfc_local_minima'] == 0).sum())}; "
      f"with more than one: {int((gait_event_data['mfc_local_minima'] > 1).sum())}")

# For trip risk the TAIL matters more than the mean: one abnormally low
# clearance is what catches an obstacle.
tri_mfc = gait_event_data['minimum_foot_clearance'].dropna()
print(f"  MFC distribution: median {1000*tri_mfc.median():.1f} mm, "
      f"5th pct {1000*tri_mfc.quantile(0.05):.1f} mm, min {1000*tri_mfc.min():.1f} mm")

# =============================================================================
# ----Figure: how TRI is built, on one swing
# =============================================================================
if tri_example is not None:
    e = tri_example
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(e['t'], e['clear'], color='#1b7a3d', lw=1.8)
    axes[0].plot(e['t'][e['mfc_i']], e['clear'][e['mfc_i']], 'o', ms=9,
                 color='#1b7a3d', label=f"MFC = {e['clear'][e['mfc_i']]:.0f} mm")
    axes[0].set_ylabel('clearance (mm)')
    axes[0].set_title(f"How the trip risk integral is built  -- {e['side']} swing, "
                      f"index {e['index']}", fontsize=11)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].plot(e['t'], e['moi'], color='#a8442a', lw=1.8)
    axes[1].set_ylabel('MoI (mm)')
    axes[1].annotate('zero where the body is stable;\npositive = a catch here '
                     'would destabilise', (0.02, 0.95), xycoords='axes fraction',
                     va='top', fontsize=8, color='#555555')

    axes[2].plot(e['t'], e['acc'], color='#7a4fb5', lw=1.5)
    axes[2].plot(e['t'][e['a']], e['acc'][e['a']], '^', ms=9, color='#7a4fb5')
    axes[2].plot(e['t'][e['b']], e['acc'][e['b']], 'v', ms=9, color='#7a4fb5')
    axes[2].set_ylabel('MFC point\nacceleration (m/s$^2$)')
    axes[2].annotate('peak accel -> peak decel sets the integration window,\n'
                     'excluding the lift-off and landing spikes',
                     (0.02, 0.95), xycoords='axes fraction', va='top',
                     fontsize=8, color='#555555')

    axes[3].plot(e['t'], e['risk'], color='#111111', lw=1.8)
    axes[3].fill_between(e['t'][e['a']:e['b'] + 1], 0, e['risk'][e['a']:e['b'] + 1],
                         color='#d94f04', alpha=0.35,
                         label=f"TRI = {np.nansum(e['risk'][e['a']:e['b']+1])/kinematic_fs:.4f} s")
    axes[3].set_ylabel('MoI / MFC')
    axes[3].set_xlabel('time through swing (s)')
    axes[3].legend(fontsize=9, frameon=False)
    for ax in axes:
        ax.axvline(e['t'][e['a']], color='#bbbbbb', lw=1, ls='--')
        ax.axvline(e['t'][e['b']], color='#bbbbbb', lw=1, ls='--')
        ax.grid(alpha=0.15)
    plt.tight_layout()

# MFC distribution: the tail is the part that matters
plt.figure(figsize=(9, 3.6))
plt.hist(1000 * tri_mfc, bins=60, color='#2a5d9f', alpha=0.8)
plt.axvline(1000 * tri_mfc.median(), color='k', lw=1.5, label='median')
plt.axvline(1000 * tri_mfc.quantile(0.05), color='#d94f04', lw=1.5,
            label='5th percentile, the trip-relevant tail')
plt.xlabel('minimum foot clearance (mm)')
plt.ylabel('swings')
plt.title('MFC distribution -- the LOW TAIL is what catches an obstacle')
plt.legend(fontsize=8, frameon=False)
plt.tight_layout()


# =============================================================================
# %% Kinetic metrics
# =============================================================================
# The metric family that only exists because the treadmill is instrumented.
# Everything is per limb, which the split belts give directly.
#
# CONVENTIONS ARE DERIVED FROM THE DATA, NOT ASSUMED
# Nothing in the export documents the moment convention. Regressing the
# reported CoP on -My/Fz and Mx/Fz recovers each plate's origin offset, and the
# reconstruction residual says whether the assumption holds. It matters for the
# free moment, which needs the CoP RELATIVE TO THE MOMENT ORIGIN: use the
# lab-frame CoP by mistake and you add a spurious term of tens of N*m to a real
# signal of about 5.
#
# THE CoP MUST BE FILTERED BEFORE ANYTHING IS DERIVED FROM IT
# At 1000 Hz the sample-to-sample CoP step is a couple of mm of noise, so a raw
# path length accumulates ~1800 mm over a stance in which the CoP actually
# moved 3 mm. Range is robust to that; path length is not.
#
# WHAT EACH MEASURE IS FOR
#   braking / propulsive impulse  AP force integrated over the part of stance
#       where it opposes / drives travel. Reduced propulsion is one of the
#       better established ageing markers, and impulse symmetry is a far better
#       asymmetry measure than any kinematic one.
#   F1, trough, F2   weight acceptance peak, mid-stance unloading, push-off peak
#   loading rate     how fast load is accepted, 20-80% of F1
#   free moment      transverse-plane torque against the ground
# =============================================================================

kmx_loading_band = (0.20, 0.80)   # fraction of F1 used for the loading rate
kmx_loaded_n     = 200.0          # N, "solidly loaded", for the CoP calibration
kmx_cop_filter_hz = 15.0          # see the note above
kmx_norm_points  = 101            # samples per stance for the ensemble figure

print("\n" + "-" * 74)
print("  KINETIC METRICS")
print("-" * 74)

# --- filter the CoP, and derive each plate's origin offset -------------------
cop_filtered = {}
free_moment = {}
kmx_cop_offset = {}
for side in ('Left', 'Right'):
    b_cop, a_cop = butter(4, kmx_cop_filter_hz / (force_fs / 2), 'low')
    raw = {a: force_data[f'{side} belt_COP_{a}'].to_numpy() for a in 'XY'}
    cop_filtered[side] = {a: filtfilt(b_cop, a_cop, raw[a]) for a in 'XY'}

    F = {a: force_data[f'{side} belt_Force_{a}'].to_numpy() for a in 'XYZ'}
    M = {a: force_data[f'{side} belt_Moment_{a}'].to_numpy() for a in 'XYZ'}
    ok = F['Z'] > kmx_loaded_n
    # the offset is a geometric constant, so calibrate on the RAW CoP: using
    # the filtered one would make the check depend on the filter choice
    ox = float(np.median(raw['X'][ok] + 1000.0 * M['Y'][ok] / F['Z'][ok]))
    oy = float(np.median(raw['Y'][ok] - 1000.0 * M['X'][ok] / F['Z'][ok]))
    kmx_cop_offset[side] = (ox, oy)
    resid = float(np.max(np.abs(-1000.0 * M['Y'][ok] / F['Z'][ok] + ox - raw['X'][ok])))
    print(f"  {side} plate origin ({ox:+.1f}, {oy:+.1f}) mm, "
          f"CoP reconstruction residual {resid:.4f} mm")
    if resid > 1.0:
        print(f"   Residual over 1 mm: the moment convention is not what this "
              f"assumes, so the free moments below will be wrong")

    # free moment = the vertical torque NOT explained by the horizontal forces
    # acting at the CoP, with the CoP taken relative to the moment origin
    rx = (cop_filtered[side]['X'] - ox) / 1000.0
    ry = (cop_filtered[side]['Y'] - oy) / 1000.0
    free_moment[side] = M['Z'] - (rx * F['Y'] - ry * F['X'])

print(f"  belt centres {abs(kmx_cop_offset['Left'][0]-kmx_cop_offset['Right'][0]):.0f} mm apart")

# --- which AP sign is propulsion? check the pattern, do not assume -----------
# Braking should dominate the first half of stance and propulsion the second.
kmx_first, kmx_second = [], []
for x in range(len(gait_event_data)):
    hs = gait_event_data['initial_contact_kinematic_frame_100hz'][x]
    to = gait_event_data['toe_off_kinematic_frame_100hz'][x]
    if np.isnan(hs) or np.isnan(to) or to <= hs:
        continue
    side = 'Left' if gait_event_data['support_limb'][x] == 'L' else 'Right'
    fy = force_data[f'{side} belt_Force_Y'].to_numpy()
    a = int((hs - 1) * force_fs / kinematic_fs)
    b = int((to - 1) * force_fs / kinematic_fs)
    if b <= a or b >= len(fy):
        continue
    kmx_first.append(fy[a:(a + b) // 2].mean())
    kmx_second.append(fy[(a + b) // 2:b].mean())

kmx_forward_sign = mos_travel_sign
if np.mean(kmx_second) < np.mean(kmx_first):
    kmx_forward_sign = -mos_travel_sign
    print(f"  ! the AP force pattern is reversed from what the belt implies; "
          f"using the force pattern, which is the more direct evidence")
print(f"  AP force: first half of stance {np.mean(kmx_first):+.1f} N, "
      f"second half {np.mean(kmx_second):+.1f} N -> propulsion is "
      f"{'+' if kmx_forward_sign > 0 else '-'}Y")

# --- per stance --------------------------------------------------------------
for c in ('braking_impulse_bw_s', 'propulsive_impulse_bw_s', 'vertical_impulse_bw_s',
          'grf_peak1_bw', 'grf_trough_bw', 'grf_peak2_bw', 'loading_rate_bw_s',
          'cop_ml_range_mm', 'cop_ap_range_mm', 'free_moment_peak_nm'):
    gait_event_data[c] = np.nan

kmx_ensemble = {'Left': [], 'Right': []}      # normalised vGRF, for the figure
kmx_cop_paths = {'Left': [], 'Right': []}

for x in range(len(gait_event_data)):
    hs = gait_event_data['initial_contact_kinematic_frame_100hz'][x]
    to = gait_event_data['toe_off_kinematic_frame_100hz'][x]
    if np.isnan(hs) or np.isnan(to) or to <= hs:
        continue
    side = 'Left' if gait_event_data['support_limb'][x] == 'L' else 'Right'
    a = int((hs - 1) * force_fs / kinematic_fs)
    b = int((to - 1) * force_fs / kinematic_fs)
    if b <= a + 50 or b >= len(force_data):
        continue

    fz = force_data[f'{side} belt_Force_Z'].to_numpy()[a:b]
    fy = force_data[f'{side} belt_Force_Y'].to_numpy()[a:b] * kmx_forward_sign
    dt = 1.0 / force_fs

    # impulses: the AP force split by sign, so braking and propulsion separate
    gait_event_data.loc[x, 'braking_impulse_bw_s'] = np.sum(fy[fy < 0]) * dt / body_weight
    gait_event_data.loc[x, 'propulsive_impulse_bw_s'] = np.sum(fy[fy > 0]) * dt / body_weight
    gait_event_data.loc[x, 'vertical_impulse_bw_s'] = np.sum(fz) * dt / body_weight

    # vertical GRF landmarks: first peak before mid-stance, second after
    mid = len(fz) // 2
    i1 = int(np.argmax(fz[:mid])); i2 = mid + int(np.argmax(fz[mid:]))
    gait_event_data.loc[x, 'grf_peak1_bw'] = fz[i1] / body_weight
    gait_event_data.loc[x, 'grf_peak2_bw'] = fz[i2] / body_weight
    gait_event_data.loc[x, 'grf_trough_bw'] = fz[i1:i2].min() / body_weight if i2 > i1 else np.nan

    # loading rate over 20-80% of the first peak
    rise = fz[:i1 + 1]
    if len(rise) > 5 and fz[i1] > 0:
        lo = int(np.argmax(rise >= kmx_loading_band[0] * fz[i1]))
        hi = int(np.argmax(rise >= kmx_loading_band[1] * fz[i1]))
        if hi > lo:
            gait_event_data.loc[x, 'loading_rate_bw_s'] = (
                (rise[hi] - rise[lo]) / ((hi - lo) * dt) / body_weight)

    px = cop_filtered[side]['X'][a:b]; py = cop_filtered[side]['Y'][a:b]
    gait_event_data.loc[x, 'cop_ml_range_mm'] = np.ptp(px)
    gait_event_data.loc[x, 'cop_ap_range_mm'] = np.ptp(py)
    gait_event_data.loc[x, 'free_moment_peak_nm'] = np.max(np.abs(free_moment[side][a:b]))

    kmx_ensemble[side].append(normalize(fz / body_weight, kmx_norm_points))
    if len(kmx_cop_paths[side]) < 40:
        kmx_cop_paths[side].append(np.column_stack([px - px[0], py - py[0]]))

print(f"\n  {gait_event_data['grf_peak1_bw'].notna().sum()} stances")
print(gait_event_data.groupby('support_limb')[
    ['grf_peak1_bw', 'grf_trough_bw', 'grf_peak2_bw',
     'propulsive_impulse_bw_s', 'braking_impulse_bw_s']].mean())

# Net AP impulse must be near zero at steady speed: propulsion has to cancel
# braking. A large net value is a plate offset, not a finding.
kmx_net = (gait_event_data['propulsive_impulse_bw_s']
           + gait_event_data['braking_impulse_bw_s']).mean()
kmx_prop = gait_event_data['propulsive_impulse_bw_s'].mean()
print(f"  net AP impulse {kmx_net:+.5f} BW*s "
      f"({100*abs(kmx_net/kmx_prop):.1f}% of the propulsive impulse)")
if abs(kmx_net) > 0.1 * abs(kmx_prop):
    print(f"   That should be near zero at steady speed -- check the AP zero offset")

# =============================================================================
# ----Figures: ensemble GRF, impulse split, CoP butterfly
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

pct = np.linspace(0, 100, kmx_norm_points)
for side, colour in (('Left', '#2a5d9f'), ('Right', '#a8442a')):
    if not kmx_ensemble[side]:
        continue
    E = np.vstack(kmx_ensemble[side])
    axes[0].fill_between(pct, np.percentile(E, 5, axis=0), np.percentile(E, 95, axis=0),
                         color=colour, alpha=0.20, lw=0)
    axes[0].plot(pct, E.mean(axis=0), color=colour, lw=2, label=f'{side} (n={len(E)})')
axes[0].axhline(1.0, color='k', lw=0.8, ls=':')
axes[0].set_xlabel('% of stance'); axes[0].set_ylabel('vertical GRF (BW)')
axes[0].set_title('Vertical GRF, mean and 5-95%', fontsize=10)
axes[0].legend(fontsize=8, frameon=False); axes[0].grid(alpha=0.15)
axes[0].annotate('F1\nweight\nacceptance', (25, 1.15), fontsize=7, ha='center', color='#555')
axes[0].annotate('trough\nunloading', (50, 0.72), fontsize=7, ha='center', color='#555')
axes[0].annotate('F2\npush-off', (75, 1.12), fontsize=7, ha='center', color='#555')

axes[1].scatter(gait_event_data['braking_impulse_bw_s'],
                gait_event_data['propulsive_impulse_bw_s'],
                c=np.where(gait_event_data['support_limb'] == 'L', 0, 1),
                cmap='coolwarm', s=10, alpha=0.6)
lim = np.nanmax(np.abs(gait_event_data['braking_impulse_bw_s']))
axes[1].plot([-lim, 0], [lim, 0], 'k--', lw=1, label='perfect cancellation')
axes[1].set_xlabel('braking impulse (BW*s)'); axes[1].set_ylabel('propulsive impulse (BW*s)')
axes[1].set_title('Braking vs propulsion, each stance', fontsize=10)
axes[1].legend(fontsize=8, frameon=False); axes[1].grid(alpha=0.15)

for side, colour in (('Left', '#2a5d9f'), ('Right', '#a8442a')):
    for p in kmx_cop_paths[side]:
        axes[2].plot(p[:, 0], p[:, 1], color=colour, lw=0.6, alpha=0.35)
axes[2].set_xlabel('CoP mediolateral (mm)'); axes[2].set_ylabel('CoP anteroposterior (mm)')
axes[2].set_title('CoP paths, aligned to heel strike', fontsize=10)
axes[2].grid(alpha=0.15)
plt.tight_layout()

# =============================================================================
# %% Stride series
# =============================================================================
# Everything from here on is a STRIDE-TO-STRIDE metric: alpha, entropy, the
# harmonic ratio, the goal-equivalent manifold. They all read a series indexed
# by stride, and every one of them is N-DEPENDENT. Build them ad hoc and you
# end up comparing alpha on 756 strides against entropy on 150, where part of
# any difference between conditions is sample size rather than the walker.
#
# So the series are cut ONCE, here, and every later cell reads this table.
# Nothing new is computed: the per-step columns are already on gait_event_data,
# this just takes one limb's rows, drops the warm-up, and truncates.
#
# WHY A FIXED N MATTERS MORE THAN A LARGE N
# alpha, sample entropy and lambda all drift with series length, so if one
# condition ran 15 min and another 10, the longer trial gets a different alpha
# for a reason that has nothing to do with gait. Truncate every trial in the
# dataset to a COMMON length -- the shortest one -- and state it.
#
#   at 108 bpm the cadence is 54 strides/min, so
#     15 min trial -> 810 strides, 756 after a 60 s warm-up
#     10 min trial -> 540 strides, 486 after a 60 s warm-up
#
# 486 is below the 600 that Damouras recommends for a stable alpha, so on the
# 10 min trials alpha carries a wider confidence interval. That is a reason to
# report the interval, not a reason to use a different N per trial.
#
# THE METRONOME
# It paces cadence, so it sets stride TIMING directly. alpha on stride time
# measures how tightly the walker locks to the beat, which is a real finding
# but a different one. The series the metronome does not constrain (step width,
# clearance, margins, impulses) are the better primary outcomes.
#
# References
#   Damouras et al. (2010) Gait Posture 31(3), 336-340.
#   Dingwell, John & Cusumano (2010) PLoS Comput Biol 6(7), e1000856.
# =============================================================================

ss_limb     = 'R'      # which limb's steps define the series, same as lds_limb
ss_warmup_s = 60.0     # treadmill acclimatisation, excluded
ss_fixed_n  = None     # None = use everything. SET THIS to the shortest trial
                       # in the dataset before comparing trials, e.g. 486.

print("\n" + "-" * 74)
print("  STRIDE SERIES")
print("-" * 74)

# one limb's rows, past the warm-up, in time order. reset_index matters: every
# later cell indexes this table positionally as a series
ss_keep = ((gait_event_data['support_limb'] == ss_limb)
           & (gait_event_data['initial_contact_kinematic_frame_100hz']
              > ss_warmup_s * kinematic_fs))
stride_series = (gait_event_data.loc[ss_keep]
                 .sort_values('initial_contact_kinematic_frame_100hz')
                 .reset_index(drop=True))
ss_all = (gait_event_data.loc[gait_event_data['support_limb'] == ss_limb]
          .sort_values('initial_contact_kinematic_frame_100hz'))

print(f"  limb {ss_limb}, {ss_warmup_s:.0f} s warm-up excluded")
print(f"  {len(stride_series)} strides available ({len(ss_all)} before the warm-up cut)")

# --- truncate to the dataset-wide length -------------------------------------
ss_available = len(stride_series)
if ss_fixed_n is None:
    print(f"  using all {ss_available}. SET ss_fixed_n before comparing trials:")
    print(f"    alpha, entropy and lambda all drift with N, so a 756-stride trial")
    print(f"    is not comparable with a 486-stride one")
elif ss_available < ss_fixed_n:
    print(f"  ! only {ss_available} strides, fewer than the {ss_fixed_n} this dataset")
    print(f"    is standardised to. This trial is NOT comparable on any N-dependent")
    print(f"    metric -- either lower ss_fixed_n for the whole dataset, or drop it")
else:
    stride_series = stride_series.iloc[:ss_fixed_n].copy()
    print(f"  truncated to the dataset-wide {ss_fixed_n} strides "
          f"({ss_available} were available)")

# --- which series are actually usable ----------------------------------------
# A series with gaps is not a series. DFA and entropy both accept a NaN-riddled
# array and hand back a number, so completeness is checked here, once, and only
# the series that pass are carried forward.

ss_candidates = ['stride_time', 'stance_time', 'swing_time', 'stance_percentage',
                 'double_support_percentage', 'step_length', 'step_width',
                 'stride_length', 'stride_velocity', 'minimum_foot_clearance',
                 'mos_ml_contact', 'mos_ml_min', 'mos_ap_contact',
                 'trip_risk_integral', 'propulsive_impulse_bw_s',
                 'braking_impulse_bw_s', 'grf_peak1_bw', 'grf_peak2_bw',
                 'loading_rate_bw_s', 'cop_ml_range_mm', 'free_moment_peak_nm']

ss_complete = {}
print(f"\n  series                       n valid   complete       mean        CV")
stride_series_usable = []
for ss_c in ss_candidates:
    if ss_c not in stride_series.columns:
        continue
    ss_v = pd.to_numeric(stride_series[ss_c], errors='coerce').to_numpy(float)
    ss_ok = np.isfinite(ss_v)
    if ss_ok.sum() < 10:
        continue
    ss_pct = 100 * ss_ok.mean()
    ss_mean = np.nanmean(ss_v)
    ss_cv = 100 * np.nanstd(ss_v) / abs(ss_mean) if ss_mean else np.nan
    ss_complete[ss_c] = (ss_pct, ss_cv)
    # a column with no variance is not a series. Every estimator downstream
    # will still return a number for it -- DFA fits a slope to the floating
    # point residue, entropy finds every template matching every other -- so it
    # is caught once, here, rather than producing a plausible-looking result
    ss_flat = np.nanstd(ss_v) <= 1e-10 * max(abs(ss_mean), 1.0)
    print(f"    {ss_c:27s} {ss_ok.sum():5d}    {ss_pct:5.1f}%   {ss_mean:10.4f}  "
          f"{ss_cv:6.2f}%{'   constant, dropped' if ss_flat else ''}")
    if ss_pct > 95 and not ss_flat:
        stride_series_usable.append(ss_c)

print(f"\n  {len(stride_series_usable)} series are over 95% complete and carry forward")

# the metronome constrains timing directly, so flag those separately
ss_cued = {'stride_time', 'stance_time', 'swing_time', 'cadence'}
ss_uncued = [c for c in stride_series_usable if c not in ss_cued]
print(f"  with a metronome, prefer the {len(ss_uncued)} it does not constrain:")
print(f"    {', '.join(ss_uncued[:6])}{' ...' if len(ss_uncued) > 6 else ''}")

# =============================================================================
# ----Figures: what the series is, what is in it, and how variable each one is
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

# panel 1: the cut itself, drawn on the stride time trace
ss_x_all = np.arange(len(ss_all))
ss_t_all = pd.to_numeric(ss_all['stride_time'], errors='coerce').to_numpy(float)
ss_n_dropped = len(ss_all) - ss_available
axes[0].plot(ss_x_all, ss_t_all, color='#bbbbbb', lw=0.8)
axes[0].plot(ss_x_all[ss_n_dropped:ss_n_dropped + len(stride_series)],
             ss_t_all[ss_n_dropped:ss_n_dropped + len(stride_series)],
             color='#2a5d9f', lw=0.9)
axes[0].axvspan(0, ss_n_dropped, color='#cc6633', alpha=0.12, lw=0)
axes[0].annotate(f'{ss_warmup_s:.0f} s warm-up\n({ss_n_dropped} strides)',
                 (ss_n_dropped / 2, np.nanmax(ss_t_all)), fontsize=7,
                 ha='center', va='top', color='#cc6633')
if ss_fixed_n is not None and ss_available > ss_fixed_n:
    axes[0].axvline(ss_n_dropped + ss_fixed_n, color='k', ls='--', lw=1)
    axes[0].annotate(f'ss_fixed_n = {ss_fixed_n}',
                     (ss_n_dropped + ss_fixed_n, np.nanmin(ss_t_all)),
                     fontsize=7, rotation=90, va='bottom')
axes[0].set_xlabel(f'{ss_limb} limb step number')
axes[0].set_ylabel('stride time (s)')
axes[0].set_title(f'The series: {len(stride_series)} strides kept', fontsize=10)
axes[0].grid(alpha=0.15)

# panel 2: every usable series on one stride axis, z-scored and stacked, so it
# is obvious they share an index and roughly how each one behaves
ss_show = ss_uncued[:6] if len(ss_uncued) >= 3 else stride_series_usable[:6]
for ss_i, ss_c in enumerate(ss_show):
    ss_v = pd.to_numeric(stride_series[ss_c], errors='coerce').to_numpy(float)
    ss_z = (ss_v - np.nanmean(ss_v)) / np.nanstd(ss_v)
    axes[1].plot(np.arange(len(ss_z)), ss_z + 5 * ss_i, lw=0.6, color='#2a5d9f')
    axes[1].annotate(ss_c, (0, 5 * ss_i + 2.0), fontsize=7, color='#444')
axes[1].set_yticks([])
axes[1].set_xlabel('stride number')
axes[1].set_title('Each series, z-scored and offset', fontsize=10)
axes[1].grid(alpha=0.15, axis='x')

# panel 3: variability, with completeness written on each bar
ss_names = list(ss_complete)
ss_cvs = [ss_complete[c][1] for c in ss_names]
ss_cols = ['#2a5d9f' if c in stride_series_usable else '#bbbbbb' for c in ss_names]
axes[2].barh(np.arange(len(ss_names)), ss_cvs, color=ss_cols)
for ss_i, ss_c in enumerate(ss_names):
    axes[2].annotate(f"{ss_complete[ss_c][0]:.0f}%", (ss_cvs[ss_i], ss_i),
                     fontsize=6, va='center', xytext=(3, 0),
                     textcoords='offset points', color='#555')
axes[2].set_yticks(np.arange(len(ss_names)))
axes[2].set_yticklabels(ss_names, fontsize=7)
axes[2].invert_yaxis()
axes[2].set_xlabel('coefficient of variation (%)')
axes[2].set_title('Variability, labelled with completeness\n'
                  '(grey = under 95% complete, dropped)', fontsize=10)
axes[2].grid(alpha=0.15, axis='x')

plt.tight_layout()

# =============================================================================
# %% Long-range correlations
# =============================================================================
# Detrended fluctuation analysis asks whether a stride is statistically related
# to strides HUNDREDS of strides earlier. Not how MUCH gait varies -- how the
# variation is organised in time. It is blind to magnitude: shuffle a series
# and its SD is unchanged while alpha collapses to 0.5.
#
#   integrate   Y(k) = sum of (x_i - mean) up to i = k, so the series becomes a
#               random-walk-like profile
#   segment     non-overlapping boxes of length n, forwards AND backwards so
#               nothing is discarded when n does not divide N
#   detrend     least squares straight line within each box (DFA-1)
#   fluctuate   F(n) = RMS of the residuals over all boxes
#   scale       F(n) ~ n^alpha, and alpha is the slope in log-log space
#
#   alpha < 0.5   anti-persistent, a long stride is corrected by a short one
#   alpha = 0.5   uncorrelated, no memory at all
#   0.5 < a < 1   persistent, a deviation tends to be followed by more of the
#                 same -- the healthy range for stride time, ~0.75-0.85
#   alpha ~ 1.0   1/f, scale free
#   alpha > 1.0   non-stationary. Usually a drift in the trial rather than a
#                 property of control, so check the raw trace before believing it
#
# THE BOX RANGE IS THE ONE CHOICE THAT REALLY MATTERS
# Damouras examined it directly: 16 <= n <= N/9. The widely used 4 to N/4
# inflates alpha, because small boxes are dominated by the detrending fit
# itself and boxes past N/9 hold too few segments to average.
#
# TWO CHECKS RUN ON EVERY SERIES, NOT JUST IN A TEST FILE
#   a shuffled surrogate must come back at alpha = 0.5. Shuffling destroys
#     order and nothing else, so anything else means the estimator is
#     misbehaving at this N.
#   a second estimator, GPH on the periodogram, reaches the same property by a
#     completely different route (alpha = d + 0.5). Its own scatter is ~0.15 at
#     these N against ~0.07 for DFA, so it is a ballpark check, and only a
#     large gap is informative.
#
# References
#   Peng et al. (1994) Phys Rev E 49, 1685-1689.
#   Hausdorff et al. (1996) J Appl Physiol 80(5), 1448-1457.
#   Damouras et al. (2010) Gait Posture 31(3), 336-340.
#   Geweke & Porter-Hudak (1983) J Time Ser Anal 4(4), 221-238.
# =============================================================================

dfa_order        = 1        # DFA-1, a straight line removed inside each box
dfa_min_box      = 16       # Damouras
dfa_max_box_frac = 1.0 / 9  # Damouras
dfa_n_boxes      = 20       # log-spaced box sizes sampled between those limits
dfa_n_surrogate  = 50       # shuffles per series for the surrogate check
dfa_gph_power    = 0.5      # GPH bandwidth, m = N ** this
dfa_rng = np.random.default_rng(0)

# The one function this cell defines: it is called once per series, another 50
# times per series for the surrogates, and again for the figures.
def dfa_alpha(series, min_box=dfa_min_box, max_box=None):
    """DFA scaling exponent, with the fluctuation curve it was fitted to.

    Returns alpha, its standard error, the R2 of the log-log fit, and the
    (boxes, F) curve so the figure can draw exactly what was fitted.
    """
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    n_total = len(x)
    max_box = int(dfa_max_box_frac * n_total) if max_box is None else int(max_box)
    empty = dict(alpha=np.nan, se=np.nan, r2=np.nan, n=n_total,
                 boxes=np.array([]), fluct=np.array([]), profile=np.array([]))
    if max_box <= min_box or n_total < 4 * min_box:
        return empty
    # a series with no variance is all floating point residue once the mean is
    # removed, and DFA will happily fit a slope to that residue. Refuse it
    if np.std(x) <= 1e-10 * max(abs(x.mean()), 1.0):
        return empty

    # the integrated profile, which is what actually gets detrended
    profile = np.cumsum(x - x.mean())
    boxes = np.unique(np.round(np.logspace(
        np.log10(min_box), np.log10(max_box), dfa_n_boxes)).astype(int))

    fluct = np.full(len(boxes), np.nan)
    for i, n in enumerate(boxes):
        count = n_total // n
        if count < 2:
            continue
        # forward from the start AND backward from the end, so a series whose
        # length is not a multiple of n does not lose its tail
        seg = np.vstack([profile[:count * n].reshape(count, n),
                         profile[n_total - count * n:].reshape(count, n)])
        t = np.arange(n)
        trend = np.polyval(np.polyfit(t, seg.T, dfa_order), t[:, None]).T
        fluct[i] = np.sqrt(np.mean((seg - trend) ** 2))

    ok = np.isfinite(fluct) & (fluct > 0)
    if ok.sum() < 4:
        return empty

    lx, ly = np.log10(boxes[ok]), np.log10(fluct[ok])
    slope, icept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + icept)
    se = float(np.sqrt(np.sum(resid ** 2) / max(len(lx) - 2, 1)
                       / np.sum((lx - lx.mean()) ** 2)))
    r2 = float(1 - np.sum(resid ** 2) / np.sum((ly - ly.mean()) ** 2))
    return dict(alpha=float(slope), se=se, r2=r2, n=n_total,
                boxes=boxes[ok], fluct=fluct[ok], profile=profile)

print("\n" + "-" * 74)
print("  LONG-RANGE CORRELATIONS (DFA)")
print("-" * 74)

dfa_n_used = len(stride_series)
dfa_max_box = int(dfa_max_box_frac * dfa_n_used)
print(f"  N = {dfa_n_used} strides, boxes {dfa_min_box} to {dfa_max_box} "
      f"(Damouras: 16 to N/9)")
if dfa_n_used < 600:
    print(f"  ! under the 600 strides Damouras recommends. alpha is still")
    print(f"    estimable but its confidence interval is wider -- report it")
if dfa_max_box <= dfa_min_box * 2:
    print(f"  ! the box range spans less than one octave, alpha is not meaningful")

# --- one alpha per usable series, each with its own surrogate and GPH check ---
dfa_rows = []
dfa_curves = {}
for dfa_name in stride_series_usable:
    dfa_x = pd.to_numeric(stride_series[dfa_name], errors='coerce').to_numpy(float)
    dfa_res = dfa_alpha(dfa_x)
    if not np.isfinite(dfa_res['alpha']):
        continue
    dfa_curves[dfa_name] = dfa_res

    # surrogate: same values, order destroyed, so alpha must fall to 0.5
    dfa_clean = dfa_x[np.isfinite(dfa_x)]
    dfa_sur = np.array([dfa_alpha(dfa_rng.permutation(dfa_clean))['alpha']
                        for _ in range(dfa_n_surrogate)], float)

    # GPH: regress the log periodogram on log(4 sin^2(w/2)) over the lowest
    # frequencies. alpha = d + 0.5 for fractional Gaussian noise
    dfa_m = int(len(dfa_clean) ** dfa_gph_power)
    dfa_per = np.abs(np.fft.rfft(dfa_clean - dfa_clean.mean())) ** 2 / (2 * np.pi * len(dfa_clean))
    dfa_w = 2 * np.pi * np.arange(len(dfa_per)) / len(dfa_clean)
    dfa_j = np.arange(1, min(dfa_m, len(dfa_per) - 1) + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dfa_reg = np.log(4 * np.sin(dfa_w[dfa_j] / 2) ** 2)
        dfa_resp = np.log(dfa_per[dfa_j])
    dfa_fin = np.isfinite(dfa_reg) & np.isfinite(dfa_resp)   # an ordinate can be 0
    dfa_gph = (0.5 - np.polyfit(dfa_reg[dfa_fin], dfa_resp[dfa_fin], 1)[0]
               if dfa_fin.sum() >= 4 else np.nan)

    dfa_rows.append({'series': dfa_name, 'n': dfa_res['n'],
                     'alpha': dfa_res['alpha'], 'alpha_se': dfa_res['se'],
                     'r2': dfa_res['r2'], 'alpha_gph': dfa_gph,
                     'surrogate_mean': float(np.nanmean(dfa_sur)),
                     'surrogate_sd': float(np.nanstd(dfa_sur)),
                     'cued_by_metronome': dfa_name in ss_cued})

dfa_results = pd.DataFrame(dfa_rows)

if len(dfa_results):
    print(f"\n  series                      alpha   +-SE     R2    GPH   "
          f"surrogate     cued")
    for _, r in dfa_results.iterrows():
        print(f"    {r['series']:26s} {r['alpha']:5.3f}  {r['alpha_se']:.3f}  "
              f"{r['r2']:.3f}  {r['alpha_gph']:5.3f}  "
              f"{r['surrogate_mean']:.3f}+-{r['surrogate_sd']:.3f}"
              f"{'   yes' if r['cued_by_metronome'] else '    no'}")

    dfa_bad = dfa_results[np.abs(dfa_results['surrogate_mean'] - 0.5) > 0.05]
    if len(dfa_bad):
        print(f"\n  ! {len(dfa_bad)} series have a shuffled surrogate away from 0.5:")
        print(f"    {', '.join(dfa_bad['series'])}")
        print(f"    A shuffle destroys order and nothing else, so alpha must go to")
        print(f"    0.5. Anything else means the estimator misbehaves at this N")
    else:
        print(f"\n  surrogate check passed: every shuffled series returned alpha ~ 0.5")

    dfa_gap = np.abs(dfa_results['alpha'] - dfa_results['alpha_gph'])
    print(f"  DFA vs GPH: median gap {np.nanmedian(dfa_gap):.3f}, "
          f"worst {np.nanmax(dfa_gap):.3f}  (GPH scatter alone is ~0.15 at this N)")
    if np.nanmedian(dfa_gap) > 0.25:
        print(f"    ! the two estimators disagree, which usually means the series is")
        print(f"      not a clean fGn. Check it for a trend or a level shift")

    print(f"\n  reminder: the metronome paces cadence, so alpha on the cued series")
    print(f"  measures how tightly the walker locks to the beat. The uncued series")
    print(f"  are the better primary outcomes")
else:
    print("  ! no series produced an alpha")

# =============================================================================
# ----Figures: the profile and its boxes, the log-log fit, the surrogate, alpha
# =============================================================================
# The primary series for the figures: the first uncued one if there is one,
# since that is what the protocol is actually about
# and it has to be one that actually produced an alpha, not just one that was
# nominally usable
dfa_show = next((c for c in dfa_curves if c not in ss_cued),
                next(iter(dfa_curves), None))

if dfa_show is not None:
    dfa_res = dfa_curves[dfa_show]
    dfa_x = pd.to_numeric(stride_series[dfa_show], errors='coerce').to_numpy(float)
    dfa_clean = dfa_x[np.isfinite(dfa_x)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))

    # panel 1: the integrated profile with two box sizes drawn on it, and the
    # straight line that gets removed inside each box. This IS the algorithm
    dfa_prof = dfa_res['profile']
    axes[0, 0].plot(dfa_prof, color='#333', lw=0.9, label='Y(k), integrated profile')
    for dfa_n, dfa_col, dfa_off in ((dfa_res['boxes'][0], '#2a5d9f', 0),
                                    (dfa_res['boxes'][-1], '#cc6633', 0)):
        dfa_count = len(dfa_prof) // dfa_n
        for dfa_b in range(dfa_count):
            dfa_seg = dfa_prof[dfa_b * dfa_n:(dfa_b + 1) * dfa_n]
            dfa_t = np.arange(dfa_n)
            dfa_fit = np.polyval(np.polyfit(dfa_t, dfa_seg, dfa_order), dfa_t)
            axes[0, 0].plot(dfa_b * dfa_n + dfa_t, dfa_fit, color=dfa_col, lw=1.4,
                            label=f'local fits, n = {dfa_n}' if dfa_b == 0 else None)
        for dfa_b in range(dfa_count + 1):
            axes[0, 0].axvline(dfa_b * dfa_n, color=dfa_col, lw=0.3, alpha=0.35)
    axes[0, 0].set_xlabel('stride number')
    axes[0, 0].set_ylabel('cumulative deviation from the mean')
    axes[0, 0].set_title(f'Step 1-3: integrate {dfa_show}, box it, detrend each box',
                         fontsize=10)
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[0, 0].grid(alpha=0.15)

    # panel 2: F(n) against n in log-log, with the fitted slope = alpha
    axes[0, 1].loglog(dfa_res['boxes'], dfa_res['fluct'], 'o', ms=5,
                      color='#2a5d9f', label='F(n)')
    dfa_lx = np.log10(dfa_res['boxes'])
    dfa_fitline = np.polyval(np.polyfit(dfa_lx, np.log10(dfa_res['fluct']), 1), dfa_lx)
    axes[0, 1].loglog(dfa_res['boxes'], 10 ** dfa_fitline, '-', color='#cc6633', lw=2,
                      label=f"alpha = {dfa_res['alpha']:.3f} +- {dfa_res['se']:.3f}"
                            f"\nR2 = {dfa_res['r2']:.3f}")
    # the two reference slopes, anchored at the first point so they are comparable
    for dfa_ref, dfa_lbl, dfa_ls in ((0.5, 'alpha = 0.5, no memory', ':'),
                                     (1.0, 'alpha = 1.0, 1/f', '--')):
        axes[0, 1].loglog(dfa_res['boxes'],
                          dfa_res['fluct'][0] * (dfa_res['boxes'] / dfa_res['boxes'][0]) ** dfa_ref,
                          dfa_ls, color='#888', lw=1, label=dfa_lbl)
    axes[0, 1].set_xlabel('box size n (strides)')
    axes[0, 1].set_ylabel('fluctuation F(n)')
    axes[0, 1].set_title('Step 4-5: F(n) ~ n^alpha, alpha is the slope', fontsize=10)
    axes[0, 1].legend(fontsize=7, frameon=False)
    axes[0, 1].grid(alpha=0.15, which='both')

    # panel 3: the real series against a shuffled copy. Same values, same SD,
    # different alpha -- which is the whole point of the measure
    dfa_shuf = dfa_rng.permutation(dfa_clean)
    dfa_sres = dfa_alpha(dfa_shuf)
    axes[1, 0].plot(dfa_clean, lw=0.6, color='#2a5d9f',
                    label=f"as walked, alpha = {dfa_res['alpha']:.3f}")
    axes[1, 0].plot(dfa_shuf, lw=0.6, color='#cc6633', alpha=0.65,
                    label=f"shuffled, alpha = {dfa_sres['alpha']:.3f}")
    axes[1, 0].set_xlabel('stride number')
    axes[1, 0].set_ylabel(dfa_show)
    axes[1, 0].set_title(f'Identical values, identical SD '
                         f'({np.nanstd(dfa_clean):.4f}), different order',
                         fontsize=10)
    axes[1, 0].legend(fontsize=7, frameon=False)
    axes[1, 0].grid(alpha=0.15)

    # panel 4: alpha for every series, with the surrogate beside it
    dfa_ord = dfa_results.sort_values('alpha')
    dfa_y = np.arange(len(dfa_ord))
    axes[1, 1].barh(dfa_y, dfa_ord['alpha'], xerr=dfa_ord['alpha_se'],
                    color=['#bbbbbb' if c else '#2a5d9f'
                           for c in dfa_ord['cued_by_metronome']],
                    error_kw=dict(lw=0.8))
    axes[1, 1].plot(dfa_ord['surrogate_mean'], dfa_y, 'kx', ms=5,
                    label='shuffled surrogate')
    axes[1, 1].axvline(0.5, color='#888', ls=':', lw=1)
    axes[1, 1].axvline(1.0, color='#888', ls='--', lw=1)
    axes[1, 1].annotate('no memory', (0.5, -0.4), fontsize=6, rotation=90,
                        ha='right', va='bottom', color='#666')
    axes[1, 1].annotate('1/f', (1.0, -0.4), fontsize=6, rotation=90,
                        ha='right', va='bottom', color='#666')
    axes[1, 1].set_yticks(dfa_y)
    axes[1, 1].set_yticklabels(dfa_ord['series'], fontsize=7)
    axes[1, 1].set_xlabel('alpha')
    axes[1, 1].set_title('alpha per series (grey = paced by the metronome)',
                         fontsize=10)
    axes[1, 1].legend(fontsize=7, frameon=False, loc='lower right')
    axes[1, 1].grid(alpha=0.15, axis='x')

    plt.tight_layout()

# =============================================================================
# %% Entropy
# =============================================================================
# Sample entropy asks how PREDICTABLE a series is. Take every template of m
# consecutive points; count how many other templates match it within a
# tolerance r; then count how many of those still match when the template is
# extended by one point. The negative log of that ratio is the entropy.
#
#     SampEn(m, r, N) = -ln( A / B )
#
#   B   pairs of length-m templates within Chebyshev distance r of each other
#   A   pairs that are STILL within r when extended to length m+1
#
# So a low SampEn means that knowing m points tells you the next one: the
# series is regular. A high SampEn means it does not. Self-matches are
# excluded, which is the difference from approximate entropy and removes
# ApEn's bias toward calling everything regular.
#
# THE INTERPRETATION TRAP
# High entropy gets called "complex", but white noise has MAXIMAL sample
# entropy and no complexity whatsoever. That is why multiscale entropy exists:
# coarse-grain the series over longer and longer windows and recompute. White
# noise starts high and falls away steeply; a genuinely structured signal holds
# its entropy across scales. The CURVE, or the area under it, is the complexity
# measure -- never the single-scale value.
#
# The refined composite variant used below sums the match COUNTS over all tau
# coarse-graining phases before taking the logarithm, rather than averaging tau
# separate entropies. Far steadier at long scales, where any one phase has few
# points left.
#
# THERE ARE NO TRANSFERABLE NORMATIVE VALUES
# SampEn is defined only relative to its m, r, N and preprocessing, and those
# vary across the literature. Compare conditions analysed identically; never
# compare an absolute value against another paper. Yentes & Raffalt require m,
# r and N to be reported every time and warn that one r does not transfer
# between conditions -- so r is SWEPT here rather than fixed at 0.2 SD and hoped
# for, and what matters is whether the ranking of the series survives the sweep.
#
# References
#   Richman & Moorman (2000) Am J Physiol Heart Circ Physiol 278(6), H2039-H2049.
#   Costa, Goldberger & Peng (2002) Phys Rev Lett 89, 068102.
#   Yentes et al. (2013) Ann Biomed Eng 41(2), 349-365.
#   Wu et al. (2014) Phys Lett A 378(20), 1369-1374.
#   Yentes & Raffalt (2021) Ann Biomed Eng 49(3), 979-990.
# =============================================================================

ent_m          = 2       # template length
ent_r          = 0.2     # tolerance, as a fraction of the series SD
ent_r_sweep    = [0.10, 0.15, 0.20, 0.25, 0.30]    # Yentes & Raffalt
ent_m_sweep    = [2, 3]
ent_mse_scales = 10      # coarse-graining scales for the multiscale curve
ent_min_n      = 200     # Yentes 2013: below this both algorithms get fragile
ent_mse_max_s  = 300     # seconds of the continuous signal used for the curve.
                         # 300 s at 100 Hz leaves 3000 points at scale 10, well
                         # past the ~750 the curve needs to settle, and keeps
                         # the cell to seconds rather than minutes

# Two functions, both used far more than three times: the counter runs once per
# scale per phase per series, and the entropy runs across the whole r and m sweep.
def ent_match_counts(series, m, r):
    """Matching template pairs at lengths m and m+1, as raw COUNTS.

    Counts rather than the entropy itself, because the refined composite
    multiscale variant has to sum them across coarse-graining phases BEFORE
    taking the logarithm.

    A KD-tree under the Chebyshev metric does the counting. The obvious double
    loop is O(N^2), fine for a few hundred strides but hopeless on the
    continuous signal, where the multiscale curve needs tens of thousands of
    points at every scale and phase.

    A length-(m+1) match implies a length-m match, because the Chebyshev
    distance over m+1 coordinates is never smaller than over the first m. So A
    is just the pair count in the (m+1)-dimensional tree.
    """
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < m + 2:
        return 0, 0
    # the SAME n-m start indices at both lengths, so the counts are comparable
    k = n - m
    starts = np.arange(k)
    tree_m = cKDTree(x[starts[:, None] + np.arange(m)[None, :]])
    tree_m1 = cKDTree(x[starts[:, None] + np.arange(m + 1)[None, :]])
    # count_neighbors counts ORDERED pairs and includes i == j, so take off the
    # k self-matches and halve to get unordered distinct pairs
    b = int((tree_m.count_neighbors(tree_m, r, p=np.inf) - k) // 2)
    a = int((tree_m1.count_neighbors(tree_m1, r, p=np.inf) - k) // 2)
    return a, b


def ent_sampen(series, m=ent_m, r=ent_r):
    """Sample entropy. The series is z-scored, so r is directly in SD units."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    if len(x) < m + 2:
        return np.nan
    sd = np.std(x)
    if sd <= 1e-10 * max(abs(x.mean()), 1.0):
        return np.nan      # a constant series: every template matches, A == B
    a, b = ent_match_counts((x - x.mean()) / sd, m, r)
    if a == 0 or b == 0:
        return np.nan        # no matches: undefined, NOT zero
    return float(-np.log(a / b))

print("\n" + "-" * 74)
print("  ENTROPY")
print("-" * 74)
print(f"  m = {ent_m}, r = {ent_r} x SD, N = {len(stride_series)}, z-scored")
if len(stride_series) < ent_min_n:
    print(f"  ! N is under {ent_min_n}. Yentes 2013 shows both algorithms become")
    print(f"    extremely parameter sensitive below that")

# a white-noise series of the same length: the ceiling SampEn can reach here.
# It has a closed form, -ln(2*Phi(r/sqrt(2)) - 1), but the empirical value at
# THIS N is the one worth comparing against
ent_ceiling = ent_sampen(np.random.default_rng(0).normal(size=len(stride_series)))

# --- one row per series, swept over r and m ----------------------------------
ent_rows = []
for ent_name in stride_series_usable:
    ent_x = pd.to_numeric(stride_series[ent_name], errors='coerce').to_numpy(float)
    ent_row = {'series': ent_name, 'n': int(np.isfinite(ent_x).sum()),
               'm': ent_m, 'r': ent_r, 'sampen': ent_sampen(ent_x)}
    for ent_rv in ent_r_sweep:
        ent_row[f'sampen_r{ent_rv:.2f}'] = ent_sampen(ent_x, r=ent_rv)
    for ent_mv in ent_m_sweep:
        ent_row[f'sampen_m{ent_mv}'] = ent_sampen(ent_x, m=ent_mv)
    ent_rows.append(ent_row)

entropy_results = pd.DataFrame(ent_rows)

if len(entropy_results):
    print(f"\n  white noise at this N reaches {ent_ceiling:.3f} -- that is the ceiling")
    print(f"\n    series                     SampEn   "
          + "".join(f"r={v:.2f} " for v in ent_r_sweep))
    for _, ent_row in entropy_results.iterrows():
        ent_txt = f"    {ent_row['series']:26s} {ent_row['sampen']:6.3f}   "
        for ent_rv in ent_r_sweep:
            ent_txt += f"{ent_row[f'sampen_r{ent_rv:.2f}']:5.3f}  "
        print(ent_txt)

    # Does the ORDERING of the series survive the sweep? If it does not, no
    # single r says anything durable about them and only the sweep is reportable
    ent_ranks = {v: entropy_results[f'sampen_r{v:.2f}'].rank().to_numpy()
                 for v in ent_r_sweep
                 if entropy_results[f'sampen_r{v:.2f}'].notna().sum() > 2}
    ent_keys = list(ent_ranks)
    if len(ent_keys) > 1:
        ent_corrs = [np.corrcoef(ent_ranks[ent_keys[0]], ent_ranks[k])[0, 1]
                     for k in ent_keys[1:]]
        print(f"\n  rank correlation with the r = {ent_keys[0]:.2f} ordering: "
              f"{np.round(ent_corrs, 3)}")
        if np.nanmin(ent_corrs) < 0.8:
            print(f"    ! the ordering changes with r. Report the sweep, not a single")
            print(f"      value, and do not read much into any one of them")
        else:
            print(f"    the ordering is stable across r, so a single-r value is")
            print(f"    reporting something about the series and not about r")

# --- the multiscale curve, on the continuous trunk acceleration --------------
# A stride series has a few hundred points, so coarse-graining runs out after
# two or three scales. The continuous signal has enough samples for the curve
# to mean something.

ent_signal = None
for ent_col in ('Low_Back_Joint_Acc', 'Trunk_Joint_Acc'):
    if ent_col in kinematic_data.columns:
        ent_signal = np.column_stack([kinematic_data[ent_col].to_numpy(),
                                      kinematic_data[f'{ent_col}.1'].to_numpy(),
                                      kinematic_data[f'{ent_col}.2'].to_numpy()])
        ent_signal_name = ent_col
        break

multiscale_entropy = None
if ent_signal is not None:
    ent_seg = ent_signal[int(ss_warmup_s * kinematic_fs):][
        :int(ent_mse_max_s * kinematic_fs)]
    # white noise goes through the IDENTICAL loop as a reference column rather
    # than being computed separately, so the two cannot drift apart. This is
    # the whole argument for multiscale entropy: noise starts at the ceiling
    # and falls away, structure holds up
    ent_inputs = {'ML': ent_seg[:, 0], 'AP': ent_seg[:, 1], 'VT': ent_seg[:, 2],
                  'white noise': np.random.default_rng(1).normal(size=len(ent_seg))}

    ent_curves = {}
    for ent_label, ent_x in ent_inputs.items():
        ent_x = ent_x[np.isfinite(ent_x)]
        # the tolerance is fixed on the ORIGINAL series and never recomputed per
        # scale: coarse-graining shrinks the SD, so re-normalising would ask a
        # different question at every scale and the curve would stay flat at the
        # ceiling instead of showing anything
        ent_x = (ent_x - ent_x.mean()) / np.std(ent_x)
        ent_out = np.full(ent_mse_scales, np.nan)
        for ent_s in range(1, ent_mse_scales + 1):
            ent_a_sum = ent_b_sum = 0
            for ent_p in range(ent_s):          # every coarse-graining phase
                ent_cut = ent_x[ent_p:]
                ent_k = len(ent_cut) // ent_s
                if ent_k < ent_m + 2:
                    continue
                ent_cg = ent_cut[:ent_k * ent_s].reshape(ent_k, ent_s).mean(axis=1)
                ent_a, ent_b = ent_match_counts(ent_cg, ent_m, ent_r)
                ent_a_sum += ent_a
                ent_b_sum += ent_b
            if ent_a_sum > 0 and ent_b_sum > 0:
                ent_out[ent_s - 1] = float(-np.log(ent_a_sum / ent_b_sum))
        ent_curves[ent_label] = ent_out

    multiscale_entropy = pd.DataFrame(ent_curves,
                                      index=np.arange(1, ent_mse_scales + 1))
    multiscale_entropy.index.name = 'scale'

    print(f"\n  refined composite MSE on {ent_signal_name}, "
          f"{len(ent_seg)} samples ({len(ent_seg)/kinematic_fs:.0f} s) after the warm-up")
    print(f"    scale   ML     AP     VT     white noise")
    for ent_s, ent_row in multiscale_entropy.iterrows():
        print(f"    {ent_s:5d}  {ent_row['ML']:5.3f}  {ent_row['AP']:5.3f}  "
              f"{ent_row['VT']:5.3f}     {ent_row['white noise']:5.3f}")
    ent_area = multiscale_entropy.sum()
    print(f"    area under the curve: ML {ent_area['ML']:.2f}  "
          f"AP {ent_area['AP']:.2f}  VT {ent_area['VT']:.2f}  "
          f"(white noise {ent_area['white noise']:.2f})")
    print(f"    The AREA is the complexity measure, not the scale-1 value. A curve")
    print(f"    that falls away as steeply as the noise reference is noise-like;")
    print(f"    one that holds up across scales has structure at several time")
    print(f"    scales at once")
else:
    print("\n  ! no trunk acceleration column found, the multiscale curve is skipped")

# =============================================================================
# ----Figures: what a match IS, the r sweep, the multiscale curve, the ranking
# =============================================================================
ent_show = next((c for c in stride_series_usable if c not in ss_cued),
                stride_series_usable[0] if stride_series_usable else None)

if ent_show is not None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))

    # panel 1: the actual counting. Pick one template, draw every length-m
    # series that matches it within r, then show which of those SURVIVE being
    # extended to m+1. The ratio of the two counts is the entropy
    ent_x = pd.to_numeric(stride_series[ent_show], errors='coerce').to_numpy(float)
    ent_x = ent_x[np.isfinite(ent_x)]
    ent_z = (ent_x - ent_x.mean()) / np.std(ent_x)
    ent_k = len(ent_z) - ent_m
    ent_tpl = ent_z[np.arange(ent_k)[:, None] + np.arange(ent_m + 1)[None, :]]

    # a representative template: the one with the median number of matches
    ent_d_m = np.max(np.abs(ent_tpl[:, None, :ent_m] - ent_tpl[None, :, :ent_m]), axis=2)
    ent_hits = (ent_d_m <= ent_r).sum(axis=1) - 1
    ent_i = int(np.argsort(ent_hits)[len(ent_hits) // 2])

    ent_match_m = np.where((ent_d_m[ent_i] <= ent_r))[0]
    ent_match_m = ent_match_m[ent_match_m != ent_i]
    ent_d_m1 = np.max(np.abs(ent_tpl[ent_match_m] - ent_tpl[ent_i]), axis=1)
    ent_match_m1 = ent_match_m[ent_d_m1 <= ent_r]

    ent_t = np.arange(ent_m + 1)
    for ent_j in ent_match_m:
        axes[0, 0].plot(ent_t[:ent_m], ent_tpl[ent_j, :ent_m], color='#bbbbbb',
                        lw=0.7, alpha=0.7)
        axes[0, 0].plot(ent_t[ent_m - 1:], ent_tpl[ent_j, ent_m - 1:], color='#bbbbbb',
                        lw=0.5, alpha=0.35, ls=':')
    for ent_j in ent_match_m1:
        axes[0, 0].plot(ent_t, ent_tpl[ent_j], color='#2a5d9f', lw=0.8, alpha=0.75)
    axes[0, 0].plot(ent_t, ent_tpl[ent_i], color='#cc6633', lw=2.6, marker='o',
                    label=f'the template (stride {ent_i})')
    axes[0, 0].fill_between(ent_t, ent_tpl[ent_i] - ent_r, ent_tpl[ent_i] + ent_r,
                            color='#cc6633', alpha=0.12, lw=0,
                            label=f'tolerance +- r = {ent_r} SD')
    axes[0, 0].axvline(ent_m - 0.5, color='k', lw=0.6, ls='--')
    axes[0, 0].annotate(f'grey + blue: {len(ent_match_m)} match over m = {ent_m}  (B)',
                        (0.03, 0.97), xycoords='axes fraction', fontsize=8,
                        va='top', color='#555')
    axes[0, 0].annotate(f'blue only: {len(ent_match_m1)} survive to m+1 = {ent_m + 1}  (A)',
                        (0.03, 0.91), xycoords='axes fraction', fontsize=8,
                        va='top', color='#2a5d9f')
    axes[0, 0].annotate(f'one template: -ln(A/B) = '
                        f'{-np.log(max(len(ent_match_m1), 1) / max(len(ent_match_m), 1)):.3f}\n'
                        f'over all templates: SampEn = '
                        f'{ent_sampen(ent_x):.3f}',
                        (0.03, 0.83), xycoords='axes fraction', fontsize=8,
                        va='top', color='#333')
    axes[0, 0].set_xticks(ent_t)
    axes[0, 0].set_xlabel('position within the template (strides)')
    axes[0, 0].set_ylabel(f'{ent_show} (SD units)')
    axes[0, 0].set_title('What a "match" is: templates within r, before and after\n'
                         'the extension to m+1', fontsize=10)
    axes[0, 0].legend(fontsize=7, frameon=False, loc='lower right')
    axes[0, 0].grid(alpha=0.15)

    # panel 2: the r sweep. Parallel lines mean the ranking is r-independent;
    # crossing lines mean no single r is reportable
    for _, ent_row in entropy_results.iterrows():
        ent_vals = [ent_row[f'sampen_r{v:.2f}'] for v in ent_r_sweep]
        axes[0, 1].plot(ent_r_sweep, ent_vals, 'o-', ms=3, lw=1,
                        color='#cc6633' if ent_row['series'] == ent_show else '#2a5d9f',
                        alpha=1.0 if ent_row['series'] == ent_show else 0.35,
                        label=ent_show if ent_row['series'] == ent_show else None)
    axes[0, 1].axvline(ent_r, color='k', ls='--', lw=0.8)
    axes[0, 1].annotate(f'r = {ent_r} reported', (ent_r, axes[0, 1].get_ylim()[1]),
                        fontsize=7, rotation=90, ha='right', va='top')
    axes[0, 1].set_xlabel('tolerance r (fraction of SD)')
    axes[0, 1].set_ylabel('sample entropy')
    axes[0, 1].set_title('Every series across the r sweep\n'
                         '(crossing lines = no single r is reportable)', fontsize=10)
    axes[0, 1].legend(fontsize=7, frameon=False)
    axes[0, 1].grid(alpha=0.15)

    # panel 3: the multiscale curve against white noise, which is the whole
    # argument for multiscale entropy in the first place
    if multiscale_entropy is not None:
        ent_scales = multiscale_entropy.index.to_numpy()
        for ent_label, ent_colour in (('ML', '#2a5d9f'), ('AP', '#cc6633'),
                                      ('VT', '#4a8f4a')):
            axes[1, 0].plot(ent_scales, multiscale_entropy[ent_label], 'o-', ms=4,
                            color=ent_colour,
                            label=f"{ent_label}, area {multiscale_entropy[ent_label].sum():.2f}")
        axes[1, 0].plot(ent_scales, multiscale_entropy['white noise'], 's--', ms=3,
                        color='#999',
                        label=f"white noise, area "
                              f"{multiscale_entropy['white noise'].sum():.2f}")
        axes[1, 0].set_xlabel('coarse-graining scale (samples averaged)')
        axes[1, 0].set_ylabel('sample entropy')
        axes[1, 0].set_title(f'Refined composite MSE, {ent_signal_name}\n'
                             'the AREA is the complexity, not scale 1', fontsize=10)
        axes[1, 0].legend(fontsize=7, frameon=False)
        axes[1, 0].grid(alpha=0.15)
    else:
        axes[1, 0].axis('off')

    # panel 4: the ranking at the reported r, against the white-noise ceiling
    ent_ord = entropy_results.dropna(subset=['sampen']).sort_values('sampen')
    axes[1, 1].barh(np.arange(len(ent_ord)), ent_ord['sampen'],
                    color=['#bbbbbb' if s in ss_cued else '#2a5d9f'
                           for s in ent_ord['series']])
    axes[1, 1].axvline(ent_ceiling, color='#cc6633', ls='--', lw=1.2)
    axes[1, 1].annotate('white noise at this N', (ent_ceiling, -0.4), fontsize=6,
                        rotation=90, ha='right', va='bottom', color='#cc6633')
    axes[1, 1].set_yticks(np.arange(len(ent_ord)))
    axes[1, 1].set_yticklabels(ent_ord['series'], fontsize=7)
    axes[1, 1].set_xlabel(f'sample entropy (m = {ent_m}, r = {ent_r} SD)')
    axes[1, 1].set_title('Regular at the bottom, unpredictable at the top\n'
                         '(grey = paced by the metronome)', fontsize=10)
    axes[1, 1].grid(alpha=0.15, axis='x')

    plt.tight_layout()
