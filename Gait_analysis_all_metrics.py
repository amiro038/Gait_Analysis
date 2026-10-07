# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 07:53:21 2026
Revised on Tue Oct 06 2026 (version 4)

@author: alexm
"""

###############################################################################
# GAIT ANALYSIS, ALL METRICS -- DICE split-belt treadmill, military load
#
# One trial, top to bottom, one cell per metric family. Run it cell by cell
# in Spyder (Ctrl+Enter) or all at once (F5). Step 1, detect_gait_events.py,
# must have been run on the trial first: this script reads its events.
#
# Version 4 checks every metric against the guideline or the paper that
# defines it. Each cell says which, and every change from version 3 is marked
#
#     # REVISED: what changed, and why
#
# so the two versions can be compared line by line. Each cell ends with brain
# checks, and the last cell compares every key metric with the values healthy
# adults show in the literature.
#
# Metrics (cell by cell):
#   temporal          stride, stance, swing, step time; single and double
#                     support as % of the gait cycle; cadence
#   spatial           step length and width along / across the belt's
#                     direction of travel; stride length and speed over the
#                     belt; walk ratio
#   local dynamic stability   short- and long-term Lyapunov exponents
#   minimum foot clearance    on the scanned boot, Schulz's three criteria
#   margin of stability       Hof (2005): at heel strike, single-support
#                             minimum, double-support minimum (Schulz)
#   margin of instability and trip risk integral (Schulz 2017)
#   kinetics          GRF peaks, loading rate, impulses, CoP, free moment
#   stride-to-stride  DFA, sample and multiscale entropy, harmonic ratio,
#                     goal-equivalent manifold, foot placement, regularity,
#                     symmetry angle
###############################################################################


# =============================================================================
# %% Importing libraries
# =============================================================================
import sys
import warnings
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0       # one figure per check, ~30 in all
from scipy.signal import savgol_filter, butter, filtfilt, sosfiltfilt
from scipy.spatial import cKDTree
from scipy.interpolate import PchipInterpolator, CubicSpline

# the folder with this script and detect_gait_events.py. The boot geometry and
# the force-plate geometry are read with step 1's own functions, so the events
# and the metrics are computed on exactly the same boot and the same plates
code_folder = Path(r"C:\Users\amiro038\Box\Military Project\Codes\Python\Qualisys_Processing\Treadmill")
for code_path in (code_folder, Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()):
    if str(code_path) not in sys.path:
        sys.path.append(str(code_path))
import detect_gait_events as dge

# nanmean / nanmin over a stride with no valid sample returns NaN on purpose
# (the stride is then simply missing); numpy also warns each time
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0')
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 30)


# =============================================================================
# %% Defining functions
# =============================================================================
# Four small tools, each used dozens of times below. Everything else is
# written out where it is used, so each cell reads top to bottom.

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


def xyz(name):
    """A three-component Theia signal as an (frames, 3) array, or None.
    The export repeats the name for each component, which pandas reads as
    Name, Name.1, Name.2."""
    if name not in kinematic_data.columns:
        return None
    return np.column_stack([pd.to_numeric(kinematic_data[c], errors='coerce').to_numpy(float)
                            for c in (name, f'{name}.1', f'{name}.2')])


def at(signal, rows):
    """signal (frames, ...) read at FRACTIONAL rows by linear interpolation.
    REVISED: events are known to a fraction of a frame (force plate events to
    1 ms), and reading positions at the nearest whole frame threw that away."""
    rows = np.asarray(rows, float)
    i = np.clip(np.floor(np.nan_to_num(rows)).astype(int), 0, len(signal) - 2)
    w = np.clip(np.nan_to_num(rows) - i, 0, 1).reshape(rows.shape + (1,) * (signal.ndim - 1))
    out = signal[i] * (1 - w) + signal[i + 1] * w
    out[~np.isfinite(rows)] = np.nan
    return out


def fill_gaps(x):
    """Fill missing samples by straight lines so a filter can cross them, and
    return where they were so they can be put back afterwards."""
    x = np.array(x, float)
    flat = x.reshape(len(x), -1)
    gap = ~np.isfinite(flat).all(axis=1)
    n = np.arange(len(x))
    for j in range(flat.shape[1]):
        ok = np.isfinite(flat[:, j])
        if 2 <= ok.sum() < len(ok):
            flat[~ok, j] = np.interp(n[~ok], n[ok], flat[ok, j])
    return flat.reshape(x.shape), gap


# =============================================================================
# %% Settings
# =============================================================================
# Every number that shapes a result, with the cell that uses it. Change it
# here, never inside a cell.

# --- the trial (or: python Gait_analysis_all_metrics.py <events file>) -----
gait_event_file = r"C:\Users\amiro038\Box\Military Project\Data\Analysis\DICE_Treadmill\gait_event_outputs\D05_C1_Treadmill_1.3mpers 108bpm_merged_events.csv"
if len(sys.argv) > 1 and sys.argv[1].lower().endswith('.csv'):
    gait_event_file = sys.argv[1]
make_figures = '--no-figures' not in sys.argv     # batch runs switch them off
make_video = '--video' in sys.argv                 # birds-eye MoS video (slow); or set True
save_outputs = '--no-save' not in sys.argv        # CSVs in base / gait_analysis_outputs

# --- the participant (participants.csv overrides these if it exists) --------
# body mass WITHOUT the load: the load is then weighed in the quiet standing
participant_mass_kg     = 80
participant_leg_length  = 0.8769     # m, greater trochanter to floor
participant_height      = 1.7688     # m
participant_foot_length = None       # m

# --- sampling ----------------------------------------------------------------
kinematic_fs = 100            # Hz, Theia
force_fs = 1000               # Hz, checked against the force file
gravity = 9.81

# --- steady state --------------------------------------------------------------
# REVISED: version 3 used the warm-up only for the stride series. Every metric
# now uses only steady walking. The trial starts with a quiet standing and a
# belt ramp, and people need time to settle into treadmill walking (Meyer et
# al. 2019 report that several gait parameters keep changing for minutes;
# verify the exact duration before quoting it). 60 s from the first event is
# the minimum; a longer value is safer for the variability measures.
warmup_seconds = 60

# --- belt speed and direction (heel flat on the belt) -----------------------
# REVISED: 20-45% of stance. The heel is flat on the belt from the end of
# loading response (~17% of stance) to heel rise (~50% of stance, Perry &
# Burnfield 2010); version 3's 30-70% included the heel rising.
belt_flat_fraction = (0.20, 0.45)

# --- gross event errors ------------------------------------------------------------
# A step whose timing or length is this many robust SDs (1.4826 x MAD) from the
# median of its foot is a missed, doubled or misplaced event, not gait. It is
# taken out of steady walking for every metric (D05: a step length of -0.38 m,
# a cadence of 144 steps/min). Hausdorff-type screens use 3 SD of the mean;
# a robust 5 only catches what cannot be a real step.
outlier_robust_z = 5.0
# metronome-paced trial? None = read it from the trial name ("...bpm")
metronome_paced = None

# --- forces ------------------------------------------------------------------
force_filter_hz = 50.0        # peaks and landmarks (impulses use the raw force)
handrail_contact_n = 15.0     # N above its baseline that counts as a hand
cop_min_force_n = 150.0       # below this the CoP is mostly noise
quiet_search_s = 30.0         # quiet standing searched for in the first 30 s
quiet_window_s = 2.0          # ... as the steadiest 2 s
quiet_max_cv = 0.02           # total vertical force steady within 2%
quiet_min_share = 0.15        # both feet loaded (each belt >= 15%)
quiet_max_heel_travel = 0.02  # heels still (moved < 20 mm)
force_align_gravity = True    # turn the forces so their mean over steady walking
                              # is vertical (a body that does not accelerate is
                              # pushed straight up on average): removes a plate /
                              # registration tilt (D05: ~1 deg, a net braking impulse)
force_max_tilt_deg = 3.0      # a larger tilt is reported, not corrected
load_min_kg = 1.0             # below this: no load
load_place_min_kg = 10.0      # below this the load is boots / clothing / a light
                              # vest: its position cannot be solved (the equation
                              # multiplies CoP errors by m / m_load, x18 for 4.7 kg),
                              # so it is spread like the body: CoM unchanged
load_height_m = 0.0           # load CoM above Trunk_Position (not observable)
load_max_offset_m = 0.40      # a load further than this is not believed

# --- boots and the belt --------------------------------------------------------
belt_floor_fraction = (0.20, 0.45)   # foot flat, used to fit the belt surface
toe_max_bend_deg = 60.0              # the toe cap never bends further than this
sole_flat_in_stance = True           # turn each boot about the ankle so its sole lies flat
                                     # in foot-flat (heel and forefoot on the belt together)
sole_flat_max_deg = 6.0              # a larger correction is reported, not applied
toe_bend_step_deg = 0.5              # resolution of the contact-constrained bend
toe_use_theia_extension = True       # also lift the toe cap by Theia's toe EXTENSION
                                     # (never flexion), as step 1 does; False = the
                                     # belt alone bends it

# --- minimum foot clearance (Schulz 2011, 2017) -----------------------------
mfc_local_window = 2          # frames either side that a local minimum must beat
mfc_speed_quantile = 0.75     # toe speed in the upper quartile of that swing
mfc_swing_trim = 0.02         # fraction of swing trimmed at each end

# --- margin of stability (Hof 2005) ------------------------------------------
mos_pendulum_mode = 'per_frame'   # 'per_frame' CoM-to-ankle, or 'leg_length'
mos_savgol_window = 11
mos_savgol_poly = 3
mos_crossover_hz = 0.5        # markers below this, force plates above
mos_antialias_hz = 40.0       # before taking the force from 1000 to 100 Hz

# --- trip risk (Schulz 2017) ---------------------------------------------------
tri_min_clearance_mm = 1.0    # MoI/MFC is undefined at zero clearance
tri_deriv_window = 7          # frames: Savitzky-Golay (order 3) for the MFC point's
                              # velocity and acceleration, over the swing +- 3 frames

# --- kinetics ------------------------------------------------------------------
kmx_loading_band = (0.20, 0.80)   # loading rate between 20% and 80% of F1
kmx_cop_filter_hz = 15.0
kmx_norm_points = 101

# --- local dynamic stability (Bruijn's LocalDynamicStability toolbox) -----
n_strides = 150
lds_limb = 'R'
samples_per_Stride = 100
tau = 10                      # normalised samples; fix it for the whole dataset
embedding_dimensions = 5
ws = 10                       # strides the divergence is followed
period = 1
fit_window = {'S': (0.0, 0.5), 'L': (4.0, 10.0)}

# --- stride series and variability ---------------------------------------------
ss_limb = 'R'
ss_fixed_n = None             # SET to the shortest trial in the dataset
dfa_min_box = 16              # Damouras et al. (2010)
dfa_max_box_frac = 1.0 / 9
dfa_n_boxes = 20
dfa_n_surrogate = 50
dfa_n_boot = 200              # simulated series for alpha's 95% interval
ent_m = 2
ent_r = 0.2
ent_r_sweep = [0.10, 0.15, 0.20, 0.25, 0.30]
ent_mse_scales = 20           # REVISED from 10: 0.01-0.20 s at 100 Hz
ent_mse_max_s = 300
hr_n_harmonics = 20
sup_midstance_fraction = 0.5
regularity_search = 0.25      # peak searched within +-25% of the expected lag

limbs = ('L', 'R')
side_name = {'L': 'Left', 'R': 'Right'}
other_limb = {'L': 'R', 'R': 'L'}


# =============================================================================
# %% Setting up folders and files
# =============================================================================
gait_event_path = Path(gait_event_file)
trial = gait_event_path.stem.replace('_merged_events', '').replace('_event_qa', '')
participant = trial.split('_')[0]
base = gait_event_path.parents[1]          # ...\DICE_Treadmill
event_folder = gait_event_path.parent

print("=" * 74)
print(f"  {trial}")
print("=" * 74)

# --- the Theia export ----------------------------------------------------------
# REVISED: the file is tab separated, so it is read with the fast C engine
# instead of letting the python engine sniff the separator on 90 000 rows.
# The trial name is spelled with a space in some exports and an underscore in
# others, so both are tried
metrics_path = dge.find_file(base / "Theia_csv_outputs", trial, "_metrics.csv")
kinematic_data = pd.read_csv(metrics_path, sep='\t', header=1, skiprows=[2, 3, 4],
                             low_memory=False)
kinematic_data = kinematic_data.apply(pd.to_numeric, errors='coerce')
# a Visual3D export can end with empty rows: cut after the last row with data
kin_has_data = np.flatnonzero(kinematic_data.drop(columns='Unnamed: 0').notna().any(axis=1).to_numpy())
kinematic_data = (kinematic_data.iloc[:kin_has_data[-1] + 1]
                  .dropna(subset=['Unnamed: 0']).reset_index(drop=True))
kin_frames = kinematic_data['Unnamed: 0'].to_numpy(float)
n_frames = len(kin_frames)
if np.any(np.diff(kin_frames) != 1):
    raise ValueError(" Theia frames are not contiguous; every cell below "
                     "assumes row = frame - first frame")

#map event frames onto row positions in the kinematic data. REVISED: this is
#now one subtraction, valid for fractional frames too (version 3 used
#reindex(frame) in the spatial cell, which reads ROW `frame`, one frame late)
first_frame = kin_frames[0]
def frame_to_row(frame):
    return np.asarray(frame, float) - first_frame
frame_position = pd.Series(np.arange(n_frames), index=kin_frames.astype(int))
print(f" Theia: {metrics_path.name}, {n_frames} frames ({n_frames / kinematic_fs:.0f} s)")

# --- the force export ----------------------------------------------------------
force_path = dge.find_file(base / "FP_renamed", trial, ".csv")
force_data = pd.read_csv(force_path, index_col=0)
force_time = force_data['Left belt_TIME'].to_numpy()
force_fs_measured = 1.0 / np.median(np.diff(force_time))
if abs(force_fs_measured - force_fs) > 1.0:
    print(f" Force file is {force_fs_measured:.0f} Hz, not the {force_fs} Hz set "
          f"above. Using the measured rate")
    force_fs = force_fs_measured
force_step = int(round(force_fs / kinematic_fs))     # force samples per frame
print(f" Force: {force_path.name}, {len(force_data)} samples, "
      f"{force_time[-1] - force_time[0]:.1f} s at {force_fs:.0f} Hz")

# --- what the event detection measured: plate registration, belt surface ----
summary_path = event_folder / f"{trial}_event_summary.json"
event_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
if not event_summary:
    print(" ! no event summary from detect_gait_events.py: the plates are taken")
    print("   to be in Theia's frame. Re-run step 1 with the current version")

# --- the two clocks: the force data moved as step 1 moved them ------------------
# REVISED (after D05, where Theia's events came 14 ms before the plates'):
# step 1 measures the offset between the force and Theia clocks and moves the
# force data by it; the same shift is applied here, so the events (in step
# 1's aligned force time) and the force samples agree. The clocks check in
# the Forces cell then reads what is LEFT, which should be ~0.
force_shift_ms = int(event_summary.get('force_shift_ms', 0) or 0)
if force_shift_ms:
    shift_n = int(round(force_shift_ms * force_fs / 1000))
    shift_cols = [c for c in force_data.columns if not c.endswith('_TIME')]
    force_data[shift_cols] = force_data[shift_cols].shift(shift_n).bfill().ffill()
    print(f" force data moved {force_shift_ms:+d} ms to Theia's clock (measured in step 1)")

# --- the plates' measured corners (C3D parameters) ---------------------------
plate_file = next((p for p in (base / "force_plates_DICE_treadmill.txt",
                               dge.PLATE_FILE) if Path(p).exists()), None)
if plate_file is None:
    raise FileNotFoundError("force_plates_DICE_treadmill.txt not found next to "
                            "the data or next to detect_gait_events.py")

# --- the participant's boots ---------------------------------------------------
binding_path = next((p for p in (base / "Foot bindings" / f"{participant}_foot_mesh_binding.npz",
                                 dge.binding_for(trial)) if Path(p).exists()), None)

# --- participant values: participants.csv if it is there ---------------------
for participants_file in (base / "participants.csv", code_folder / "participants.csv"):
    if participants_file.exists():
        participants = pd.read_csv(participants_file).set_index('participant')
        if participant in participants.index:
            row_p = participants.loc[participant]
            participant_mass_kg = float(row_p.get('body_mass_kg', participant_mass_kg))
            participant_leg_length = float(row_p.get('leg_length_m', participant_leg_length))
            participant_height = float(row_p.get('height_m', participant_height))
            if pd.notna(row_p.get('foot_length_m', np.nan)):
                participant_foot_length = float(row_p['foot_length_m'])
            print(f" participant {participant} from {participants_file.name}: "
                  f"{participant_mass_kg:.1f} kg without the load, "
                  f"leg {participant_leg_length:.3f} m")
        break

#brain check: leg length is usually 0.50-0.55 of height
if participant_leg_length and participant_height:
    if not (0.45 < participant_leg_length / participant_height < 0.60):
        print(f" Leg length is {participant_leg_length / participant_height:.2f} of "
              f"height, outside the usual 0.50-0.55. Check one of the two")


# =============================================================================
# %% Organizing input and output formats
# =============================================================================
# One row per heel strike, holding every event that step needs.
#
# REVISED:
#  * events are read from {trial}_event_qa.csv when it exists: it has each
#    event to a fraction of a frame (force-plate events to 1 ms), the belt that
#    carried it and how it was found. merged_events.csv has whole frames only.
#    A 10 ms frame is 1.5% of a stance, more than many effects of load.
#  * each event a step needs is FOUND by time (the first toe-off of this foot
#    after the heel strike, the other foot's first toe-off after it, ...)
#    instead of by shifting rows. Version 3's row shifts went wrong after any
#    missed event: one missing toe-off moved every later "next contra" event
#    by one row.
#  * the step that ENDS at this heel strike belongs to this foot (its step
#    length is this heel ahead of the other heel). Version 3 stored the step
#    STARTING here, which is the other foot's step, on this foot's row.
events_qa_path = event_folder / f"{trial}_event_qa.csv"
if events_qa_path.exists():
    events = pd.read_csv(events_qa_path)
    # frame_float is a ROW (0 = first frame); turn it into a frame number
    events['frame'] = first_frame + events['frame_float'].astype(float)
    events['grf_sample'] = pd.to_numeric(events['grf_frame_1000hz'], errors='coerce')
    print(f" events: {events_qa_path.name} (sub-frame times, with the belt)")
else:
    events = pd.read_csv(gait_event_path)
    events['frame'] = events['frame_100hz'].astype(float)
    events['grf_sample'] = np.nan
    events['belt'] = ''
    events['contact_label'] = ''
    print(f" events: {gait_event_path.name} (whole frames only; no belt, so the")
    print("   kinetics cannot tell a crossover step from a clean one)")
# the force sample of an event: exact for force-plate events, frame x 10 else
events['grf_sample'] = events['grf_sample'].fillna(
    pd.Series(frame_to_row(events['frame']) * force_step, index=events.index))
events = events.sort_values('frame').reset_index(drop=True)

hs_frames = {b: events.loc[(events['support_limb'] == b) & (events['event_type'] == 'heel_strike'), 'frame'].to_numpy()
             for b in limbs}
to_frames = {b: events.loc[(events['support_limb'] == b) & (events['event_type'] == 'toe_off'), 'frame'].to_numpy()
             for b in limbs}

def first_after(times, t):
    """The first time in a sorted array later than each t (NaN if none)."""
    i = np.searchsorted(times, t, side='right')
    return np.where(i < len(times), times[np.minimum(i, len(times) - 1)], np.nan)

def last_before(times, t):
    """The last time in a sorted array earlier than each t (NaN if none)."""
    i = np.searchsorted(times, t, side='left') - 1
    return np.where(i >= 0, times[np.maximum(i, 0)], np.nan)

rows_out = []
for b in limbs:
    o = other_limb[b]
    hs_rows = events[(events['support_limb'] == b) & (events['event_type'] == 'heel_strike')]
    to_rows = events[(events['support_limb'] == b) & (events['event_type'] == 'toe_off')].set_index('frame')
    hs = hs_rows['frame'].to_numpy()
    d = pd.DataFrame({'support_limb': b, 'initial_contact_kinematic_frame_100hz': hs})
    d['toe_off_kinematic_frame_100hz'] = first_after(to_frames[b], hs)
    d['next_ipsi_heelstrike'] = first_after(hs_frames[b], hs)
    d['prev_ipsi_heelstrike'] = last_before(hs_frames[b], hs)
    # a toe-off after the next heel strike means this foot's toe-off was missed
    missed = d['toe_off_kinematic_frame_100hz'] > d['next_ipsi_heelstrike']
    d.loc[missed, 'toe_off_kinematic_frame_100hz'] = np.nan
    d['contra_toeoff'] = first_after(to_frames[o], hs)            # ends initial double support
    d['next_contra_heelstrike'] = first_after(hs_frames[o], hs)   # starts terminal double support
    d['prev_contra_heelstrike'] = last_before(hs_frames[o], hs)   # the step that ends here started there
    d['heel_strike_source'] = hs_rows['source'].to_numpy()
    d['heel_strike_belt'] = hs_rows['belt'].fillna('').astype(str).to_numpy()
    d['heel_strike_grf_sample'] = hs_rows['grf_sample'].to_numpy()
    d['contact_label'] = hs_rows['contact_label'].fillna('').astype(str).to_numpy()
    d['toe_off_source'] = to_rows['source'].reindex(d['toe_off_kinematic_frame_100hz']).to_numpy()
    d['toe_off_belt'] = to_rows['belt'].fillna('').astype(str).reindex(d['toe_off_kinematic_frame_100hz']).to_numpy()
    d['toe_off_grf_sample'] = to_rows['grf_sample'].reindex(d['toe_off_kinematic_frame_100hz']).to_numpy()
    rows_out.append(d)
gait_event_data = (pd.concat(rows_out).sort_values('initial_contact_kinematic_frame_100hz')
                   .reset_index(drop=True))
del rows_out

# the five events of a step must come in their natural order; anything else
# is a missed or doubled event, and every metric needing them is left missing
hs_ = gait_event_data['initial_contact_kinematic_frame_100hz']
gait_event_data['events_in_order'] = (
    (hs_ < gait_event_data['contra_toeoff'])
    & (gait_event_data['contra_toeoff'] < gait_event_data['next_contra_heelstrike'])
    & (gait_event_data['next_contra_heelstrike'] < gait_event_data['toe_off_kinematic_frame_100hz'])
    & (gait_event_data['toe_off_kinematic_frame_100hz'] < gait_event_data['next_ipsi_heelstrike']))

# steady walking: after the warm-up, counted from the first event
gait_event_data['steady'] = hs_ >= events['frame'].min() + warmup_seconds * kinematic_fs
steady = gait_event_data['steady'].to_numpy()

#brain check
print(f" {len(gait_event_data)} heel strikes, {int(steady.sum())} after the "
      f"{warmup_seconds} s warm-up")
print(" event sources: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in
                                       events['source'].value_counts(normalize=True).items()))
n_bad = int((~gait_event_data['events_in_order'] & steady).sum())
if n_bad:
    print(f" {n_bad} steady steps have their events out of order (a missed or "
          f"doubled event); their timing and support metrics are left missing")
if gait_event_data['toe_off_kinematic_frame_100hz'].isna().sum() > 2:
    print(f" {gait_event_data['toe_off_kinematic_frame_100hz'].isna().sum()} heel "
          f"strikes have no toe-off before the next heel strike (missed events)")


# =============================================================================
# %% Calculating temporal metrics
# =============================================================================
# All from the events, in seconds:
#   stance        toe-off - heel strike            (this foot)
#   swing         next heel strike - toe-off       (this foot)
#   stride        next heel strike - heel strike   (this foot)
#   step          heel strike - the other foot's previous heel strike
#   cadence       60 / step time, steps per minute
#   double support  initial (heel strike -> other toe-off)
#                 + terminal (other heel strike -> toe-off)
#   single support  other toe-off -> other heel strike (the other foot swings)
#
# REVISED: single and double support are now a % of the STRIDE (the gait
# cycle), the convention of the normative literature (Perry & Burnfield 2010:
# ~20% double support, ~40% single support at normal speed). Version 3 divided
# them by the stance time, which gives numbers that no reference uses.
# Supports are only computed when the five events are in their natural order.
g = gait_event_data
hs = g['initial_contact_kinematic_frame_100hz']
to = g['toe_off_kinematic_frame_100hz']
nxt = g['next_ipsi_heelstrike']
cto = g['contra_toeoff']
chs = g['next_contra_heelstrike']
pchs = g['prev_contra_heelstrike']
in_order = g['events_in_order']

def span(a, b, ok=True):
    """(b - a) in seconds, only where b comes after a (and ok)."""
    return ((b - a) / kinematic_fs).where((b > a) & ok)

g['stance_time'] = span(hs, to)
g['swing_time'] = span(to, nxt)
g['stride_time'] = span(hs, nxt)
g['step_time'] = span(pchs, hs)
g['cadence'] = 60 / g['step_time']
g['initial_double_support_time'] = span(hs, cto, in_order)
g['terminal_double_support_time'] = span(chs, to, in_order)
g['double_support_time'] = g['initial_double_support_time'] + g['terminal_double_support_time']
g['single_support_time'] = span(cto, chs, in_order)
for k in ('stance', 'swing', 'single_support', 'double_support'):
    g[f'{k}_percentage'] = 100 * g[f'{k}_time'] / g['stride_time']

#brain check: stance + swing = 100%, and single + double support = stance
bad = (g['stance_percentage'] + g['swing_percentage'] - 100).abs() > 0.5
if bad.any():
    print(f" Something is wrong in percentages at {int(bad.sum())} strides")
bad = (g['single_support_percentage'] + g['double_support_percentage']
       - g['stance_percentage']).abs() > 0.5
if bad.any():
    print(f" Single + double support differs from stance at {int(bad.sum())} strides")

kinematic_flag = ((g['heel_strike_source'] != 'GRF') | (g['toe_off_source'] != 'GRF'))

print("\n" + "-" * 74)
print("  TEMPORAL (steady walking)")
print("-" * 74)
print(g.loc[steady, ['stride_time', 'stance_percentage', 'swing_percentage', 'cadence']].describe().round(3))
print(g.loc[steady, ['single_support_percentage', 'double_support_percentage']].describe().round(2))

if make_figures:
    plt.figure()
    plt.plot(g['stride_time'])
    plt.title('Stride time (s)')

    plt.figure()
    plt.plot(g['stance_time'])
    plt.plot(g.index[kinematic_flag], g['stance_time'][kinematic_flag],
             'x', color='red', markersize=8, linestyle='none', label='not a force-plate event')
    plt.axvline(np.argmax(steady), color='k', ls=':', label='end of warm-up')
    plt.title('Stance (s)')
    plt.legend()

    plt.figure()
    plt.plot(g['single_support_percentage'])
    plt.title('Single support (% of the gait cycle)')


# =============================================================================
# %% Belt speed and direction of travel
# =============================================================================
# REVISED: moved here from the margin-of-stability cell, because the spatial
# metrics need it too, and generalised from "AP = lab Y" to the measured
# direction of travel.
#
# During foot flat the stance heel is fixed on the belt, so its velocity IS
# the belt's velocity. A straight line is fitted to the heel's horizontal
# position over 20-45% of each stance; its slope is the heel velocity. The
# direction of travel is minus the median direction of those velocities (the
# feet move backwards), and the belt speed their median size.
#
# Why the direction matters: AP and ML must be along and across the belt.
# Theia's Y need not line up with it, and a 1.3 deg misalignment moves 17 mm of
# a 0.75 m step length into the step width -- the size of a load effect.
heel = {b: xyz(f'{side_name[b]}_Heel_Position') for b in limbs}
ankle = {b: xyz(f'{side_name[b]}_Ankle_Position') for b in limbs}

belt_velocities = []
for x in range(len(g)):
    if not np.isfinite(g['stance_time'][x]):
        continue
    a = int(frame_to_row(hs[x] + belt_flat_fraction[0] * (to[x] - hs[x])))
    b_ = int(frame_to_row(hs[x] + belt_flat_fraction[1] * (to[x] - hs[x])))
    seg = heel[g['support_limb'][x]][a:b_, :2]
    if b_ - a >= 5 and np.isfinite(seg).all():
        belt_velocities.append(np.polyfit(np.arange(len(seg)) / kinematic_fs, seg, 1)[0])
belt_velocities = np.array(belt_velocities)
backwards = np.median(belt_velocities / np.linalg.norm(belt_velocities, axis=1, keepdims=True), axis=0)
forward = -backwards / np.linalg.norm(backwards)          # direction of travel
lateral = np.array([-forward[1], forward[0]])              # to the walker's left
belt_speeds = -belt_velocities @ forward
belt_speed = float(np.median(belt_speeds))
forward3, lateral3 = np.r_[forward, 0.0], np.r_[lateral, 0.0]

print("\n" + "-" * 74)
print("  BELT")
print("-" * 74)
print(f"  belt speed from the stance heels: {belt_speed:.3f} m/s "
      f"(IQR {np.percentile(belt_speeds, 25):.3f}-{np.percentile(belt_speeds, 75):.3f}, "
      f"n = {len(belt_speeds)} stances)")
print(f"  direction of travel ({forward[0]:+.4f}, {forward[1]:+.4f}) in Theia's X-Y, "
      f"{np.degrees(np.arctan2(forward[0], abs(forward[1]))):+.2f} deg from the Y axis")
#brain check: the trial name says the speed it was set to
if 'mpers' in trial:
    try:
        set_speed = float(trial.split('mpers')[0].split('_')[-1])
        if abs(belt_speed - set_speed) > 0.05 * set_speed:
            print(f"  ! measured {belt_speed:.3f} m/s against {set_speed} set on the treadmill")
    except ValueError:
        pass
if np.std(belt_speeds) > 0.1 * belt_speed:
    print(f"  ! belt speed varies by {100 * np.std(belt_speeds) / belt_speed:.0f}% across "
          f"stances, more than a steady belt should")


# =============================================================================
# %% Now spatial
# =============================================================================
# Both heels at the heel strike, at the exact (fractional) event time:
#   step length = (this heel - other heel) . direction of travel
#   step width  = |(this heel - other heel) . lateral|
#
# STRIDE LENGTH ON A TREADMILL (Dingwell, John & Cusumano 2010):
#   L = belt speed x stride time + (where the heel landed now - where it landed
#       at the previous heel strike of the same foot, along the belt)
# i.e. how far the foot travelled over the belt, as it would over ground.
#
# REVISED:
#  * heels read at the exact event time and the right ROW. Version 3 used
#    kinematic_data[...].reindex(frame), which reads row `frame` -- one frame
#    after the event, ~13 mm of heel travel at 1.3 m/s.
#  * along and across the measured direction of travel, not Theia's Y and X.
#  * stride length is no longer two step lengths added. On a belt the rear
#    heel has already started to lift and moved relative to the belt by the
#    time the other foot lands, so heel-to-heel step lengths overestimate the
#    stride (by 30-50 mm each). The belt-based stride length makes the mean
#    stride speed equal the belt speed, which the brain check below tests.
#  * the step on each row is the one that ENDS with this foot's heel strike.
rows_hs = frame_to_row(hs.to_numpy())
step_length = np.full(len(g), np.nan)
step_width = np.full(len(g), np.nan)
heel_advance = np.full(len(g), np.nan)
for b in limbs:
    m = (g['support_limb'] == b).to_numpy()
    d = at(heel[b], rows_hs[m]) - at(heel[other_limb[b]], rows_hs[m])
    step_length[m] = d[:, :2] @ forward
    step_width[m] = np.abs(d[:, :2] @ lateral)
    # how much further forward this heel landed than at its previous landing
    heel_advance[m] = (at(heel[b], frame_to_row(nxt.to_numpy()[m]))
                       - at(heel[b], rows_hs[m]))[:, :2] @ forward

g['step_length'] = step_length
g['step_width'] = step_width
g['stride_length'] = belt_speed * g['stride_time'] + heel_advance
g['stride_velocity'] = g['stride_length'] / g['stride_time']
g['walk_ratio'] = g['step_length'] / g['cadence']

#brain check
print("\n" + "-" * 74)
print("  SPATIAL (steady walking)")
print("-" * 74)
print(g.loc[steady, ['step_length', 'step_width', 'stride_length', 'stride_velocity']].describe().round(4))
print(g.loc[steady].groupby('support_limb')[['step_length', 'step_width']].agg(['mean', 'std', 'count']).round(4))
spatial_speed_gap = g.loc[steady, 'stride_velocity'].mean() / belt_speed - 1
print(f"  mean stride speed over the belt is {100 * spatial_speed_gap:+.2f}% from the belt speed")
if abs(spatial_speed_gap) > 0.02:
    print("  ! over 2%: the walker drifted on the belt, or the heels are mis-tracked")
# what version 3's definition would have given, for the record. On a treadmill
# two heel-to-heel steps are SHORTER than the stride over the belt: at the
# other foot's heel strike the trailing heel has already risen and rolled
# forward over the forefoot (D05: ~50 mm per step)
spatial_two_steps = 2 * g.loc[steady, 'step_length'].mean() - g.loc[steady, 'stride_length'].mean()
print(f"  (two heel-to-heel step lengths add up to {1000 * abs(spatial_two_steps):.0f} mm "
      f"{'less' if spatial_two_steps < 0 else 'more'} than the stride over the belt)")
if not (0.02 < g.loc[steady, 'step_width'].median() < 0.40):
    print("  ! that is not a plausible step width: check the direction of travel")

# --- gross event errors --------------------------------------------------------
# REVISED, new: a missed, doubled or misplaced event makes a step that cannot be
# gait (a negative step length, a 0.42 s step at a 0.56 s metronome). Such steps
# are taken out of steady walking for EVERY metric, so a single bad event does
# not inflate a CV, a DFA alpha or a margin. Robust: median and MAD of each foot.
qa_vars = ['stride_time', 'step_time', 'stance_time', 'swing_time', 'step_length', 'step_width']
gait_outlier = np.zeros(len(g), bool)
qa_counts = {}
for c_ in qa_vars:
    v_ = g[c_].to_numpy(float)
    bad_ = np.zeros(len(g), bool)
    for b in limbs:
        m_ = steady & (g['support_limb'] == b).to_numpy() & np.isfinite(v_)
        if m_.sum() < 10:
            continue
        med_ = np.median(v_[m_])
        mad_ = 1.4826 * np.median(np.abs(v_[m_] - med_))
        if mad_ > 0:
            bad_ |= m_ & (np.abs(v_ - med_) > outlier_robust_z * mad_)
    if c_ == 'step_length':
        bad_ |= steady & (v_ <= 0)
    qa_counts[c_] = int(bad_.sum())
    gait_outlier |= bad_
g['gait_outlier'] = gait_outlier
print(f"  gross event errors (> {outlier_robust_z:g} robust SD from the foot's median): "
      f"{int(gait_outlier.sum())} of {int(steady.sum())} steady steps "
      f"({100 * gait_outlier.sum() / max(steady.sum(), 1):.1f}%) taken out of steady walking")
if gait_outlier.any():
    print("    flagged by " + ", ".join(f"{k_} {n_}" for k_, n_ in qa_counts.items() if n_))
if gait_outlier.sum() > 0.02 * steady.sum():
    print("  ! over 2%: look at the events before trusting the variability measures")
steady = steady & ~gait_outlier
gait_event_data['steady'] = steady

if make_figures:
    plt.figure()
    plt.plot(g['step_length'])
    plt.title('Step length (m)')

    plt.figure()
    plt.plot(g['step_width'])
    plt.title('Step width (m)')

    plt.figure()
    plt.plot(g['stride_length'])
    plt.title('Stride length over the belt (m)')


# =============================================================================
# %% Forces: into Theia's frame, and body + load weighed
# =============================================================================
# REVISED, a new cell. Version 3 used the plate forces as exported: no baseline
# removed, plate X/Y taken to be Theia's X/Y, and the "body weight" taken as
# the mean vertical force over the whole trial. Three problems:
#
#  1. BASELINE. An unloaded plate does not read zero and its reading drifts
#     (6-8 N on DICE). It is re-measured every 10 s, as the event detection
#     does (detect_gait_events.belt_baseline).
#  2. AXES. The forces and CoP are in each plate's own frame. The plate's
#     rotation comes from its four measured corners (C3D parameters), and the
#     lab -> Theia rotation and offset from the event detection, which
#     measured them by matching the CoP to the boot meshes in single support
#     ({trial}_event_summary.json). Without them, an AP force can end up in
#     the ML velocity of the margin of stability. The CoM-acceleration check
#     in the MoS cell then confirms the axes independently.
#  3. THE LOAD. The mean vertical force while walking is the weight of body
#     AND load. The load differs between conditions and Theia cannot see it,
#     so the plates weigh body + load in the quiet standing that opens each
#     trial (standing still, nothing accelerates, so the force IS the weight),
#     and the body mass entered for the participant gives the load:
#         m_system = W / g,     m_load = m_system - m_body
#     Kinetics are then given per BODY weight (the convention) and per TOTAL
#     weight (does the pattern change beyond carrying more?), and the CoM used
#     for stability is that of body + load (MoS cell).
plates = {b: dge.fit_plate(p) for b, p in dge.read_plates(plate_file).items()}
lab_to_theia = np.asarray(event_summary.get('lab_to_theia', [[1.0, 0, 0], [0, 1.0, 0]]), float)
belt_name = {'L': 'Left belt', 'R': 'Right belt'}
sos_force = butter(4, force_filter_hz, fs=force_fs, output='sos')

grf_raw, grf, cop_theia, force_loaded, force_baseline = {}, {}, {}, {}, {}
for b in limbs:
    F = np.column_stack([force_data[f'{belt_name[b]}_Force_{a}'].to_numpy(float) for a in 'XYZ'])
    # 1. baseline: the vertical from the drift-tracking baseline, the horizontals
    #    from the median of the unloaded samples
    base_z, noise_z = dge.belt_baseline(F[:, 2])
    force_loaded[b] = sosfiltfilt(sos_force, F[:, 2]) - base_z > dge.FORCE_THRESHOLD_N
    F[:, 2] -= base_z
    F[:, :2] -= np.median(F[~force_loaded[b], :2], axis=0)
    force_baseline[b] = float(np.median(base_z))
    # 2. plate frame -> lab (fitted corners) -> Theia (registration), with the
    #    sign chosen so the vertical force holds the walker up
    lab = F @ plates[b]['R'].T
    lab *= 1.0 if np.mean(lab[force_loaded[b], 2]) > 0 else -1.0
    grf_raw[b] = np.column_stack([lab[:, :2] @ lab_to_theia[:, :2].T, lab[:, 2]])
    grf[b] = sosfiltfilt(sos_force, grf_raw[b], axis=0)
    # 3. CoP into Theia, kept only where the foot loads the belt solidly
    cp = np.column_stack([force_data[f'{belt_name[b]}_COP_{a}'].to_numpy(float) for a in 'XY'])
    if dge.COP_FRAME == 'plate':
        cp_lab = (np.column_stack([cp, np.zeros(len(cp))]) @ plates[b]['R'].T + plates[b]['t'])[:, :2] / 1000
    else:
        cp_lab = cp / 1000
    cop_theia[b] = dge.apply_2d(lab_to_theia, cp_lab)
    cop_theia[b][~(grf_raw[b][:, 2] > cop_min_force_n)] = np.nan
n_force = len(force_data)

# the handrails, if the export has them: a hand on a rail is an external force
# the plates do not see, so those steps are flagged
handrail_contact = np.zeros(n_force, bool)
for side in ('Left', 'Right'):
    rail_cols = [f'{side} handrail_Force_{a}' for a in 'XYZ']
    if all(c in force_data.columns for c in rail_cols):
        rail = force_data[rail_cols].to_numpy(float)
        rail = rail - np.median(rail, axis=0)
        handrail_contact |= np.linalg.norm(rail, axis=1) > handrail_contact_n

print("\n" + "-" * 74)
print("  FORCES")
print("-" * 74)
print(f"  plate baselines {force_baseline['L']:.1f} / {force_baseline['R']:.1f} N removed; "
      f"lab -> Theia {'from the event summary' if event_summary else 'ASSUMED identity'}")
print(f"  handrail contact on {100 * handrail_contact.mean():.2f}% of samples")

steady_samples = slice(int(frame_to_row(hs[steady].min()) * force_step),
                       int(frame_to_row(hs[steady].max()) * force_step))

# --- the forces' tilt: on average they must push straight up ----------------------
# Over minutes of steady walking the body does not accelerate, so the mean of
# the total GRF is the weight, VERTICAL, whatever the treadmill's incline. A
# mean that leans is a tilt between the plates' axes and Theia's (corner
# survey, registration, or an export already levelled and turned again):
# sin(1 deg) of the vertical force leaks into AP, ~10 N at mid-stance, and a
# net braking impulse of ~0.01 BW*s per stance (a quarter of the
# propulsive impulse) appears that no walker
# produces. The leftover shear offsets were removed above, so what remains is
# a rotation, and it is undone as one.
force_tilt_deg, force_tilt_ap_deg, force_tilt_ml_deg = 0.0, 0.0, 0.0
force_mean = np.mean((grf_raw['L'] + grf_raw['R'])[steady_samples], axis=0)
force_tilt_ap_deg = float(np.degrees(np.arctan2(force_mean[:2] @ forward, force_mean[2])))
force_tilt_ml_deg = float(np.degrees(np.arctan2(force_mean[:2] @ lateral, force_mean[2])))
force_tilt_deg = float(np.degrees(np.arccos(np.clip(force_mean[2] / np.linalg.norm(force_mean), -1, 1))))
print(f"  mean GRF over steady walking leans {force_tilt_ap_deg:+.2f} deg along travel and "
      f"{force_tilt_ml_deg:+.2f} deg across (should be 0: the body does not accelerate)")
if not force_align_gravity:
    print("    left as it is (force_align_gravity = False)")
elif handrail_contact[steady_samples].mean() > 0.01:
    print("    not corrected: hands on the rails carry part of the weight")
elif force_tilt_deg > force_max_tilt_deg:
    print(f"  ! over {force_max_tilt_deg:g} deg: that is not a small misalignment. Not corrected;")
    print("    check the plate corners and the lab -> Theia registration")
elif force_tilt_deg > 0.01:
    k_ = np.cross(force_mean / np.linalg.norm(force_mean), [0.0, 0.0, 1.0])
    sin_ = np.linalg.norm(k_)
    k_ /= sin_
    K_ = np.array([[0, -k_[2], k_[1]], [k_[2], 0, -k_[0]], [-k_[1], k_[0], 0]])
    R_align = np.eye(3) + sin_ * K_ + (1 - np.cos(np.radians(force_tilt_deg))) * K_ @ K_
    for b in limbs:
        grf_raw[b] = grf_raw[b] @ R_align.T
        grf[b] = grf[b] @ R_align.T
    print(f"    turned back by {force_tilt_deg:.2f} deg (force_align_gravity)")

# --- quiet standing: weigh body + load -----------------------------------------
fz_total = grf['L'][:, 2] + grf['R'][:, 2]
first_event_sample = int(frame_to_row(events['frame'].min()) * force_step)
qs_end = min(first_event_sample, int(quiet_search_s * force_fs), n_force)
qs_win = int(quiet_window_s * force_fs)
qs_best, qs_cv = None, np.inf
for s0 in range(0, max(qs_end - qs_win, 0), int(force_fs // 10)):     # every 0.1 s
    seg = fz_total[s0:s0 + qs_win]
    if np.mean(seg) < 200:
        continue
    share = grf['L'][s0:s0 + qs_win, 2] / seg
    cv = np.std(seg) / np.mean(seg)
    if cv >= min(qs_cv, quiet_max_cv) or share.min() < quiet_min_share or share.max() > 1 - quiet_min_share:
        continue
    # both heels where they were 2 s earlier (medians of 0.2 s at each end, so
    # tracking noise does not count as movement)
    r0, r1 = s0 // force_step, (s0 + qs_win) // force_step
    k = kinematic_fs // 5
    still = all(np.linalg.norm(np.nanmedian(heel[b_][r1 - k:r1], axis=0)
                               - np.nanmedian(heel[b_][r0:r0 + k], axis=0)) < quiet_max_heel_travel
                for b_ in limbs)
    if still:
        qs_best, qs_cv = s0, cv

walking_weight = float(np.mean(fz_total[steady_samples]))
if qs_best is None:
    print("  ! no quiet standing found before the first step: the system mass is")
    print("    the mean vertical GRF while walking, and the load cannot be placed")
    system_weight = walking_weight
    quiet_rows = None
    quiet_cop = None
else:
    qs_slice = slice(qs_best, qs_best + qs_win)
    system_weight = float(np.mean(fz_total[qs_slice]))
    quiet_rows = (qs_best // force_step, (qs_best + qs_win) // force_step)
    # the force-weighted CoP of both belts while standing, to place the load
    with np.errstate(invalid='ignore'):
        quiet_cop = np.nanmean(np.nansum([cop_theia[b][qs_slice] * grf[b][qs_slice, 2:3]
                                          for b in limbs], axis=0) / fz_total[qs_slice, None], axis=0)
    print(f"  quiet standing {qs_best / force_fs:.1f}-{(qs_best + qs_win) / force_fs:.1f} s: "
          f"{system_weight:.0f} N = {system_weight / gravity:.1f} kg (CV {100 * qs_cv:.2f}%)")
    #brain check: over steady walking the body does not accelerate on average,
    #so the mean vertical force must be the same weight
    print(f"  mean vertical GRF while walking is {100 * (walking_weight / system_weight - 1):+.2f}% from it")
    if abs(walking_weight / system_weight - 1) > 0.02:
        print("  ! over 2% apart: plate drift, or the standing was not still")

system_mass = system_weight / gravity
body_mass = float(participant_mass_kg) if participant_mass_kg else np.nan
body_weight = body_mass * gravity
load_kg = system_mass - body_mass
print(f"  body {body_mass:.1f} kg (entered)  +  load {load_kg:+.1f} kg  =  {system_mass:.1f} kg weighed")
if np.isfinite(load_kg) and load_kg < -load_min_kg:
    print("  ! the plates weigh LESS than the body mass entered: check the entry or the plate zero")
if not np.isfinite(body_mass):
    print("  ! no body mass entered: per-body-weight kinetics use the system weight instead")
    body_mass, body_weight = system_mass, system_weight

# --- timing: do the force plates and Theia run on the same clock? ------------------
# REVISED (after D05), a check. The force file and the Theia export are taken
# to start together. If they do not, every force-plate event is read at the
# wrong kinematic instant (at 1.3 m/s a 20 ms offset is 26 mm of belt travel).
# Newton gives the test: the system CoM's vertical acceleration from the
# plates, GRF / m - g, and from Theia, the second derivative of
# Whole_body_COG, are the same signal. The lag that best lines them up
# (0.5-8 Hz, both filtered without phase shift) is the offset between the
# two clocks. Positive = Theia's signal comes LATER than the force's.
sync_lag_ms, sync_r = np.nan, np.nan
sync_com = xyz('Whole_body_COG')
if sync_com is not None:
    sync_r0 = int(frame_to_row(hs[steady].min()))
    sync_r1 = min(int(frame_to_row(hs[steady].max())), n_force // force_step - 1)
    sync_az = savgol_filter(fill_gaps(sync_com[:, 2])[0], 11, 3, deriv=2, delta=1.0 / kinematic_fs)
    sync_fz = sosfiltfilt(butter(4, 40.0, fs=force_fs, output='sos'),
                          grf_raw['L'][:, 2] + grf_raw['R'][:, 2])
    sync_af = sync_fz[np.arange(n_frames)[:sync_r1 + 1] * force_step] / system_mass - gravity
    sync_band = butter(2, [0.5, 8.0], btype='band', fs=kinematic_fs, output='sos')
    sync_x = sosfiltfilt(sync_band, sync_af[sync_r0:sync_r1])
    sync_y = sosfiltfilt(sync_band, sync_az[sync_r0:sync_r1])
    sync_k = np.arange(-15, 16)
    sync_c = np.array([np.corrcoef(sync_x[max(0, -k):len(sync_x) - max(0, k)],
                                   sync_y[max(0, k):len(sync_y) - max(0, -k)])[0, 1] for k in sync_k])
    i_ = int(np.argmax(sync_c))
    sync_frac = 0.0
    if 0 < i_ < len(sync_k) - 1:
        den = sync_c[i_ - 1] - 2 * sync_c[i_] + sync_c[i_ + 1]
        sync_frac = 0.5 * (sync_c[i_ - 1] - sync_c[i_ + 1]) / den if den != 0 else 0.0
    sync_lag_ms = 1000.0 * (sync_k[i_] + sync_frac) / kinematic_fs
    sync_r = float(sync_c[i_])
    print(f"  clocks: Theia's CoM acceleration matches the plates' best {sync_lag_ms:+.0f} ms later "
          f"(r = {sync_r:.2f}; 0 = in sync)")
    if abs(sync_lag_ms) > 10:
        print(f"  ! the two recordings are about {abs(sync_lag_ms):.0f} ms apart. Every force-plate event is")
        print("    read at the wrong kinematic instant: re-run step 1 (detect_gait_events.py,")
        print("    FORCE_SHIFT_MS = 'auto') so the force data are moved to Theia's clock")


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
#
# This follows the current recommendations (Bruijn et al. 2013 review;
# Mehdizadeh 2018 review): a FIXED number of strides for every trial (lambda
# depends on N), time normalised to 100 samples per stride, tau and dE fixed
# across the whole dataset, and the short-term exponent lambda_S as the
# primary outcome. The trunk velocity is the most studied state space.
#
# REVISED:
#  * the window starts at the first right heel strike after the warm-up
#    (version 3 took the heel strike nearest frame 6000, which counted from
#    the start of the recording, not from the start of walking)
#  * the LINEAR trunk velocity is rotated into the walker's axes (ML, AP, VT
#    along the measured direction of travel) before it is used, so 'AP' is
#    along the belt and not along Theia's Y
#  * the z-scored spaces are kept but labelled: dividing each channel by its
#    own SD makes lambda incomparable between conditions (the SDs differ)
n_neighbours = 1   # 1 = lds_calc.m, >1 = lds_calc_mehdizadeh.m

#Define the state space
# signal spec is ('Base_Column', axis)
#   axis 'X','Y','Z' -> for Trunk_Linear_Velocity: the walker's ML, AP, VT
#                       for any other column: the export's own X, Y, Z
# embed  'delay' = Takens embedding, one common tau, total dim = channels * dE
#        'none'  = the listed signals ARE the state variables
# scale  'none'   raw units (lambda is invariant to a UNIFORM rescale)
#        'zscore' per channel: NOT comparable between conditions
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
    'trunk_kin_3D (zscored)': {'signals': [('Trunk_Linear_Velocity', 'X'),
                                           ('Trunk_Linear_Velocity', 'Y'),
                                           ('Trunk_Linear_Velocity', 'Z'),
                                           ('Trunk_Joint_Velocity', 'X'),
                                           ('Trunk_Joint_Velocity', 'Y'),
                                           ('Trunk_Joint_Velocity', 'Z')],
                               'embed': 'none', 'dE': None, 'scale': 'zscore'},
    }
lds_primary = 'trunkVel_AP'

# the trunk velocity in the walker's axes: ML (to the left), AP (along the
# direction of travel), VT
lds_trunk_vel = xyz('Trunk_Linear_Velocity')
lds_walker = None
if lds_trunk_vel is not None:
    lds_walker = np.column_stack([lds_trunk_vel[:, :2] @ lateral,
                                  lds_trunk_vel[:, :2] @ forward,
                                  lds_trunk_vel[:, 2]])

def lds_column(base_col, axis):
    """One channel of a state space, as described above."""
    k = 'XYZ'.index(axis)
    if base_col == 'Trunk_Linear_Velocity':
        return None if lds_walker is None else lds_walker[:, k]
    sig = xyz(base_col)
    return None if sig is None else sig[:, k]

# =============================================================================
# ----Set up kinematic data range
# =============================================================================
lds_hs_all = np.sort(g.loc[(g['support_limb'] == lds_limb) & g['steady'],
                           'initial_contact_kinematic_frame_100hz'].dropna().to_numpy())
lds_hs_rows = np.round(frame_to_row(lds_hs_all)).astype(int)

#Brain check that you have enough strides to keep going
lds_short = False
lds_n = n_strides
if len(lds_hs_rows) < n_strides + 1:
    lds_short = True
    lds_n = len(lds_hs_rows) - 1
    print(f" Only {lds_n} strides after the warm up. lambda depends on the amount")
    print(f" of data, so this trial is NOT comparable to trials run at {n_strides}")

lds_start_row, lds_end_row = int(lds_hs_rows[0]), int(lds_hs_rows[lds_n])
lds_start_frame, lds_end_frame = lds_hs_all[0], lds_hs_all[lds_n]

#As in makestatelocal.m, the whole block from the first to the last heel strike
#becomes lds_n * 100 samples
n_samples = lds_n * samples_per_Stride

#lds_calc.m settings, in normalised samples
ws_samples = int(round(ws * samples_per_Stride))
half_period = int(round(0.5 * period * samples_per_Stride))

#Make the state spaces and calculate lambda
print("\n" + "-" * 74)
print(f"  LOCAL DYNAMIC STABILITY ({lds_n} strides of the {side_name[lds_limb].lower()} foot)")
print("-" * 74)
lds_results = []

for ss_name, ss in state_spaces.items():
    signals = ss['signals']
    n_embed = ss['dE'] if ss['embed'] == 'delay' else 1

    ncols = len(signals) * n_embed
    nrows = n_samples - tau * (n_embed - 1) #embedding loses the last (dE-1)*tau rows

    mat = np.zeros((nrows, ncols))
    col_idx = 0
    skip = False

    for base_col, axis in signals:
        raw = lds_column(base_col, axis)
        if raw is None:
            print(f"  {base_col} missing -> skipping state space {ss_name}")
            skip = True
            break
        segment = raw[lds_start_row:lds_end_row + 1] #first to last heel strike, inclusive
        if not np.isfinite(segment).all():
            print(f"  {base_col} has NaN inside the window -> skipping state space {ss_name}")
            skip = True
            break

        #normalise first. 'spline' matches interp1(...,'spline') in makestatelocal.m
        normalised = normalize(segment, n_samples, 'spline')
        if ss['scale'] == 'zscore':
            normalised = normalised / np.std(normalised)

        #then embed, delay in normalised samples
        for t_ in range(n_embed):
            start = t_ * tau
            mat[:, col_idx] = normalised[start:start + nrows]
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
    #L fits lags 399..999 against t = 4.00..10.00
    lds_row = {'trial': trial, 'state_space': ss_name,
               'is_primary': ss_name == lds_primary,
               'n_strides': lds_n, 'stride_count_short': lds_short,
               'first_frame': lds_start_frame, 'last_frame': lds_end_frame,
               'mean_stride_time_s': (lds_end_frame - lds_start_frame) / lds_n / kinematic_fs,
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
    print(f"  {ss_name:24s} lambda_S = {lds_row['lambda_S']:.4f}  "
          f"lambda_L = {lds_row['lambda_L']:.4f}  R2_S = {lds_row['R2_S']:.2f}")

lds_results = pd.DataFrame(lds_results)

# =============================================================================
# ----Plot, as lds_calc.m does with plotje = 1
# =============================================================================
lds_done = [ss_name for ss_name in state_spaces if 'divergence' in state_spaces[ss_name]]
if make_figures and lds_done:
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
# %% Boots on the feet and the belt surface
# =============================================================================
# The base of support (margin of stability) and the foot clearance are about
# the real boot, so the participant's scanned boots are posed on Theia's feet
# in every frame: vertex(t) = R_foot(t) @ vertex_local + p_foot(t).
#
# REVISED, three things:
#
#  1. THE TOE CAP. A boot's toe cap bends up at the flex line when the belt
#     pushes it (push-off) and springs back to its moulded, scanned shape once
#     it is unloaded; a boot cannot curl its toes down. So the toe cap is
#     turned up about the MTP by the LARGER of
#       - Theia's toe EXTENSION, never its flexion (toe_use_theia_extension),
#         as the event detection does: Theia's toe segment is fitted to the
#         image of the boot's toe, so its extension carries information;
#       - the smallest bend that keeps every toe vertex out of the belt.
#     The toe can then never put the boot through the belt. Version 3 bent
#     the toe cap by Theia's angle in BOTH directions, every frame.
#     D05, first real run: Theia's toe angle in swing is -1.5 to +11 deg
#     (5-95%), mostly extension, and the RIGID boot already reaches the belt
#     in mid-swing (median 0-2 mm). So the low clearance there is not the toe
#     hinge: it is Theia's foot pose in swing relative to stance, or the
#     binding. The force plates are the check, printed below: at a
#     force-plate heel strike the boot's lowest point must be ON the belt,
#     and at a force-plate toe-off too.
#     The cell also poses the boot rigid, bent by the belt alone, and bent by
#     Theia's angle both ways (version 3), and reports each model's clearance.
#  2. ONLY THE SOLE. The lowest vertex in every 10 mm cell of the footprint,
#     up to 35 mm above the lowest (detect_gait_events.boot_sole). Only the
#     underside can touch the belt or an obstacle, and ~300 vertices instead
#     of ~2500 per boot make every frame of a 15-min trial affordable.
#  3. THE BELT SURFACE is fitted along the measured direction of travel (not
#     Theia's Y): z = offset[boot] + slope * (distance along travel), to the
#     lowest sole point in foot-flat (20-45% of stance). One slope for the
#     pitched belt, one offset per boot to absorb a height bias in either
#     boot's pose.
if binding_path is None:
    raise FileNotFoundError(f"no foot binding for {participant}: build it with build_foot_binding.py")
boot_verts, pose_names = dge.read_binding(binding_path)

foot_pose = {}
for b in limbs:
    cols = [pose_names[b]] + [f'{pose_names[b]}.{i}' for i in range(1, 16)]
    P = kinematic_data[cols].to_numpy(float).reshape(-1, 4, 4).copy()
    P[:, :3, :3] = dge.orthonormalise(P[:, :3, :3])
    P[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
    P[~np.isfinite(P[:, :3, :]).all(axis=(1, 2))] = np.nan
    foot_pose[b] = P

# --- the sole, its toe cap, and the hinge ---------------------------------------
sole_local, sole_is_toe, sole_is_front, toe_hinge_point, toe_lift_sign, boot_length = {}, {}, {}, {}, {}, {}
sole_fwd_local, sole_tilt_deg = {}, {}
for b in limbs:
    P = foot_pose[b]
    R, t = P[:, :3, :3], P[:, :3, 3]
    ok = np.isfinite(P).all(axis=(1, 2))
    sole_local[b], long_axis = dge.boot_sole(boot_verts[b], P)
    # the MTP in the foot's own frame: <Side>_Toes_Position seen from the foot
    mtp_world = xyz(f'{side_name[b]}_Toes_Position')
    mtp_local = np.einsum('fji,fj->fi', R, mtp_world - t)
    m = np.nanmedian(mtp_local[ok & np.isfinite(mtp_local).all(axis=1)], axis=0)
    toe_hinge_point[b] = m
    # the toe cap: sole vertices ahead of the MTP along the boot
    up_local = np.median(R[ok][:, 2, :], axis=0)
    up_local /= np.linalg.norm(up_local)
    fwd_local = long_axis - (long_axis @ up_local) * up_local
    fwd_local /= np.linalg.norm(fwd_local)
    along = (sole_local[b] - m) @ fwd_local
    boot_length[b] = float(np.ptp(sole_local[b] @ fwd_local))
    sole_fwd_local[b] = fwd_local
    sole_is_toe[b] = along > 0
    # front and rear halves of the sole, for Schulz's toe-vs-heel criterion
    along_sole = sole_local[b] @ fwd_local
    sole_is_front[b] = along_sole > 0.5 * (along_sole.min() + along_sole.max())
    # which way about the foot's X axis lifts the toe tip
    tip = sole_local[b][np.argmax(along)] - m
    toe_lift_sign[b] = 1.0 if up_local @ np.cross([1.0, 0, 0], tip) > 0 else -1.0

def rot_x(angles):
    """Rotations about the foot's X axis, one per angle (radians)."""
    c, s = np.cos(angles), np.sin(angles)
    R_ = np.zeros((len(angles), 3, 3))
    R_[:, 0, 0] = 1.0
    R_[:, 1, 1], R_[:, 1, 2], R_[:, 2, 1], R_[:, 2, 2] = c, -s, s, c
    return R_

# --- the sole flat in foot-flat ---------------------------------------------------------
# REVISED (after D05), a check and a correction. In foot-flat (20-45% of
# stance) a boot's heel and forefoot are BOTH on the belt, so the posed sole
# must lie flat there. If the heel sits above the forefoot (or below), the
# boot is tilted on Theia's foot: the binding, or Theia's foot angle in
# walking against the static pose it was built from. A 2 deg toe-down tilt
# puts the heel ~8 mm up at heel strike and the toe a few mm too low in
# swing. The tilt is measured as the turn about the ankle (the foot frame's
# X axis) that brings the heel's lowest point level with the forefoot's
# (toe cap excluded, it curls up), and undone (sole_flat_in_stance).
# Synthetic trial, built from the same mesh: 0.0-0.5 deg.
tilt_grid = np.radians(np.arange(-8.0, 8.0 + 1e-9, 0.1))
for b in limbs:
    v_ = sole_local[b]
    al_ = v_ @ sole_fwd_local[b]
    al_ = (al_ - al_.min()) / np.ptp(al_)
    heel_ = al_ < 0.25
    fore_ = (al_ > 0.55) & ~sole_is_toe[b]
    rows_ = []
    for x in np.flatnonzero(steady & (g['support_limb'] == b).to_numpy())[::3]:
        if np.isfinite(hs[x]) and np.isfinite(to[x]):
            r0 = int(np.ceil(frame_to_row(hs[x] + belt_floor_fraction[0] * (to[x] - hs[x]))))
            r1 = int(np.floor(frame_to_row(hs[x] + belt_floor_fraction[1] * (to[x] - hs[x]))))
            rows_.extend(range(r0, r1 + 1))
    up_ = foot_pose[b][np.array(rows_, int), 2, :3]                # world up, seen from the foot
    up_ = up_[np.isfinite(up_).all(axis=1)]
    sole_tilt_deg[b] = np.nan
    if len(up_) < 50 or not heel_.any() or not fore_.any():
        continue
    gaps_ = []
    for a_ in tilt_grid:
        vr_ = v_ @ rot_x(np.array([toe_lift_sign[b] * a_]))[0].T
        z_ = up_ @ vr_.T                                           # height of each vertex, per frame
        gaps_.append(np.median(z_[:, heel_].min(axis=1) - z_[:, fore_].min(axis=1)))
    gaps_ = np.array(gaps_)
    gap0 = float(np.interp(0.0, tilt_grid, gaps_))
    # the turn that levels them (the gap falls as the toe is turned up)
    order_ = np.argsort(gaps_)
    tilt_ = float(np.degrees(np.interp(0.0, gaps_[order_], tilt_grid[order_])))
    sole_tilt_deg[b] = tilt_
    applied_ = sole_flat_in_stance and abs(tilt_) <= sole_flat_max_deg
    print(f"  {side_name[b]} boot in foot-flat: heel {1000 * gap0:+.1f} mm above the forefoot (0 = flat); "
          f"levelled by turning the toe {'up' if tilt_ > 0 else 'down'} {abs(tilt_):.1f} deg"
          + ("" if applied_ else "  [NOT applied]"))
    if applied_:
        sole_local[b] = v_ @ rot_x(np.array([toe_lift_sign[b] * np.radians(tilt_)]))[0].T

# --- the belt surface, from the rigid soles in foot flat --------------------------
belt_pts = {b: [] for b in limbs}
for x in range(len(g)):
    if not (steady[x] and np.isfinite(g['stance_time'][x])):
        continue
    b = g['support_limb'][x]
    r0 = int(np.ceil(frame_to_row(hs[x] + belt_floor_fraction[0] * (to[x] - hs[x]))))
    r1 = int(np.floor(frame_to_row(hs[x] + belt_floor_fraction[1] * (to[x] - hs[x]))))
    if x % 3:                       # every third stance is plenty
        continue
    P = foot_pose[b][r0:r1 + 1]
    W = sole_local[b] @ P[:, :3, :3].transpose(0, 2, 1) + P[:, None, :3, 3]
    k = np.nanargmin(np.where(np.isfinite(W[..., 2]), W[..., 2], np.inf), axis=1)
    belt_pts[b].append(W[np.arange(len(W)), k])
belt_pts = {b: np.vstack(v) for b, v in belt_pts.items()}
belt_pts = {b: v[np.isfinite(v).all(axis=1)] for b, v in belt_pts.items()}
belt_keep = {b: np.ones(len(v), bool) for b, v in belt_pts.items()}
for _ in range(4):              # least squares, then drop > 3 MAD and refit
    A_ = np.vstack([np.column_stack([np.full(belt_keep[b].sum(), b == 'L'),
                                     np.full(belt_keep[b].sum(), b == 'R'),
                                     belt_pts[b][belt_keep[b], :2] @ forward])
                    for b in limbs]).astype(float)
    z_ = np.concatenate([belt_pts[b][belt_keep[b], 2] for b in limbs])
    off_l, off_r, belt_slope = np.linalg.lstsq(A_, z_, rcond=None)[0]
    belt_offset = {'L': off_l, 'R': off_r}
    for b in limbs:
        r_ = belt_pts[b][:, 2] - belt_offset[b] - belt_slope * (belt_pts[b][:, :2] @ forward)
        belt_keep[b] = np.abs(r_) < 3 * 1.4826 * np.median(np.abs(r_)) + 1e-4

def height_above_belt(W, b):
    """Height of world points W (..., 3) above the pitched belt under boot b."""
    return W[..., 2] - belt_offset[b] - belt_slope * (W[..., :2] @ forward)

# --- pose the soles: the toe cap bent only as far as the belt requires -----------
# Heights are linear in the world position, h = a . w - offset, with
# a = (-slope f_x, -slope f_y, 1), so for each frame h(vertex) = (R^T a) . v +
# a . p - offset. Every candidate bend of the toe cap is tried at once (a grid
# of 0.5 deg steps) and the smallest one with no toe vertex below the belt kept.
belt_normal = np.array([-belt_slope * forward[0], -belt_slope * forward[1], 1.0])
bend_grid = np.radians(np.arange(0.0, toe_max_bend_deg + 1e-9, toe_bend_step_deg))
sole_world, sole_height, toe_bend_deg, toe_bend_contact_deg, theia_toe_deg = {}, {}, {}, {}, {}
clear_rigid, clear_theia_toe, clear_contact = {}, {}, {}
for b in limbs:
    P = foot_pose[b]
    m = toe_hinge_point[b]
    toe = sole_is_toe[b]
    d_toe = sole_local[b][toe] - m
    bent_local = m + np.einsum('gij,tj->gti', rot_x(toe_lift_sign[b] * bend_grid), d_toe)  # (G, T, 3)
    q = np.einsum('fji,j->fi', P[:, :3, :3], belt_normal)          # R^T a, per frame
    c = P[:, :3, 3] @ belt_normal - belt_offset[b]
    toe_angle_col = f'{side_name[b]}_Toes_Joint_Angle'
    theia_toe = (np.radians(dge.TOE_SIGN * (kinematic_data[toe_angle_col].to_numpy(float)
                                           - dge.TOE_REFERENCE_DEG))
                 if toe_angle_col in kinematic_data.columns else None)
    theia_toe_deg[b] = np.degrees(theia_toe) if theia_toe is not None else np.full(n_frames, np.nan)
    # Theia's EXTENSION only, as a lift of the toe cap (dge's convention:
    # TOE_SIGN x (angle - reference) > 0 is extension, checked on D05)
    theia_lift = (np.clip(np.nan_to_num(theia_toe), 0.0, np.radians(toe_max_bend_deg))
                  if (theia_toe is not None and toe_use_theia_extension) else np.zeros(n_frames))
    bend = np.full(n_frames, np.nan)
    bend_contact = np.full(n_frames, np.nan)
    sole_world[b] = np.full((n_frames, len(sole_local[b]), 3), np.nan, np.float32)
    clear_rigid[b] = np.full(n_frames, np.nan)
    clear_contact[b] = np.full(n_frames, np.nan)
    if theia_toe is not None:
        clear_theia_toe[b] = np.full(n_frames, np.nan)
    # in chunks of frames, so a 15-min trial fits in memory
    for s0 in range(0, n_frames, 2000):
        sl = slice(s0, min(s0 + 2000, n_frames))
        Pc = P[sl]
        good = np.isfinite(Pc).all(axis=(1, 2))
        # 1. the smallest bend that keeps every toe vertex out of the belt
        h_toe = np.einsum('fk,gtk->fgt', q[sl], bent_local) + c[sl, None, None]
        enough = h_toe.min(axis=2) >= 0                                # (F, G)
        first_ok = np.where(enough.any(axis=1), np.argmax(enough, axis=1), len(bend_grid) - 1)
        bend_c = np.where(good, bend_grid[first_ok], np.nan)
        bend_contact[sl] = bend_c
        # ... and the larger of that and Theia's extension
        bend_f = np.where(good, np.maximum(np.nan_to_num(bend_c), theia_lift[sl]), np.nan)
        bend[sl] = bend_f
        # 2. every sole vertex in the world: rigid, toe cap bent by the belt
        #    alone, and toe cap bent by the model used
        W = sole_local[b] @ Pc[:, :3, :3].transpose(0, 2, 1) + Pc[:, None, :3, 3]
        H_rigid = height_above_belt(W, b)
        clear_rigid[b][sl] = np.nanmin(H_rigid, axis=1) if good.any() else np.nan
        Rc = rot_x(toe_lift_sign[b] * np.nan_to_num(bend_c))
        W_toe_c = np.einsum('fij,ftj->fti', Pc[:, :3, :3], m + np.einsum('fij,tj->fti', Rc, d_toe)) + Pc[:, None, :3, 3]
        clear_contact[b][sl] = np.minimum(np.nanmin(H_rigid[:, ~toe], axis=1),
                                          np.nanmin(height_above_belt(W_toe_c, b), axis=1))
        Rb = rot_x(toe_lift_sign[b] * np.nan_to_num(bend_f))
        W_toe = np.einsum('fij,ftj->fti', Pc[:, :3, :3], m + np.einsum('fij,tj->fti', Rb, d_toe)) + Pc[:, None, :3, 3]
        # 3. for comparison only: the toe cap bent by Theia's toe angle, both
        #    ways, as apply_binding.py posed it for version 3
        if theia_toe is not None:
            Rt = rot_x(np.nan_to_num(theia_toe[sl]))
            W_th = W.copy()
            W_th[:, toe] = np.einsum('fij,ftj->fti', Pc[:, :3, :3], m + np.einsum('fij,tj->fti', Rt, d_toe)) + Pc[:, None, :3, 3]
            clear_theia_toe[b][sl] = np.nanmin(height_above_belt(W_th, b), axis=1)
        W[:, toe] = W_toe
        sole_world[b][sl] = W
    toe_bend_deg[b] = np.degrees(bend)
    toe_bend_contact_deg[b] = np.degrees(bend_contact)
    sole_height[b] = height_above_belt(sole_world[b], b).astype(np.float32)

# per frame: the lowest sole point (the foot's clearance) and which one it is
foot_clearance = pd.DataFrame({'frame': kin_frames})
for b in limbs:
    hb = np.where(np.isfinite(sole_height[b]), sole_height[b], np.inf)
    foot_clearance[f'{side_name[b]}_min_z'] = np.where(np.isfinite(hb.min(axis=1)), hb.min(axis=1), np.nan)
    foot_clearance[f'{side_name[b]}_lowest_vertex'] = hb.argmin(axis=1)

print("\n" + "-" * 74)
print("  BOOTS AND THE BELT")
print("-" * 74)
print(f"  {binding_path.name}: {len(sole_local['L'])} / {len(sole_local['R'])} sole points "
      f"(L / R), {sole_is_toe['L'].sum()} / {sole_is_toe['R'].sum()} on the toe cap")
print(f"  belt surface: {1000 * belt_offset['L']:.1f} / {1000 * belt_offset['R']:.1f} mm under "
      f"the left / right boot, {np.degrees(np.arctan(belt_slope)):+.2f} deg along travel "
      f"(C3D plates: ~0.8 deg)")

# --- which model of the toe puts the boot through the belt in swing? -------------
swing_rows = {b: np.zeros(n_frames, bool) for b in limbs}
stance_rows = {b: np.zeros(n_frames, bool) for b in limbs}
mid_swings = {b: [] for b in limbs}      # 30-80% of each swing, where the MFC is
for x in range(len(g)):
    b = g['support_limb'][x]
    if steady[x] and np.isfinite(g['swing_time'][x]):
        r0, r1 = frame_to_row(to[x]), frame_to_row(nxt[x])
        pad = mfc_swing_trim * (r1 - r0)
        swing_rows[b][int(np.ceil(r0 + pad)):int(np.floor(r1 - pad)) + 1] = True
        mid_swings[b].append(slice(int(np.ceil(r0 + 0.3 * (r1 - r0))), int(np.floor(r0 + 0.8 * (r1 - r0))) + 1))
    if steady[x] and np.isfinite(g['stance_time'][x]):
        stance_rows[b][int(np.ceil(frame_to_row(hs[x]))):int(np.floor(frame_to_row(to[x]))) + 1] = True
es_surface = event_summary.get('belt_surface')
if es_surface and 'walking_direction' in event_summary:
    # compared WHERE THE FEET ARE: the offsets are heights at distance 0 along
    # travel, which can be a metre away, so on their own they mean little
    es_fwd = np.asarray(event_summary['walking_direction'], float)[:2]
    es_slope = np.tan(np.radians(es_surface['slope_deg']))
    gap_ = []
    for b in limbs:
        p_ = belt_pts[b][belt_keep[b]]
        mine = belt_offset[b] + belt_slope * (p_[:, :2] @ forward)
        theirs = es_surface['floor_m'][b] + es_slope * (p_[:, :2] @ es_fwd)
        gap_.append(1000 * np.median(theirs - mine))
    print(f"  (the event detection's fit, {es_surface['slope_deg']:+.2f} deg, sits {gap_[0]:+.1f} / {gap_[1]:+.1f} mm"
          f" from this one under the feet in foot-flat; it is fitted to every frame's lowest point,")
    print("   push-off and landing included, so this one, from foot-flat only, is used here)")
toe_models = [("Theia's toe angle both ways (v3)", clear_theia_toe),
              ('rigid boot', clear_rigid),
              ('toe bent by the belt alone', clear_contact),
              ('toe bent by Theia extension or belt (used)' if toe_use_theia_extension
               else 'toe bent by the belt alone (used)',
               {b: foot_clearance[f'{side_name[b]}_min_z'].to_numpy() for b in limbs})]
print("  each model of the toe cap, steady swings: % of swing frames BELOW the belt,")
print("  and the lowest sole point in mid-swing (30-80% of each swing), mm")
print("      foot   model                                      below   median  5th pct  minimum")
for b in limbs:
    for label, hh in toe_models:
        if b not in hh or not mid_swings[b]:
            continue
        h = hh[b]
        lows = 1000 * np.array([np.nanmin(h[sl]) if np.isfinite(h[sl]).any() else np.nan for sl in mid_swings[b]])
        print(f"      {side_name[b]:5s}  {label:42s} {100 * np.mean(h[swing_rows[b]] < 0):5.1f}%  "
              f"{np.nanmedian(lows):7.1f}  {np.nanpercentile(lows, 5):7.1f}  {np.nanmin(lows):7.1f}")
    th_sw = theia_toe_deg[b][swing_rows[b]]
    if np.isfinite(th_sw).any():
        print(f"             Theia's toe angle in swing: median {np.nanmedian(th_sw):+.1f} deg, "
              f"5-95% {np.nanpercentile(th_sw, 5):+.1f} to {np.nanpercentile(th_sw, 95):+.1f} deg "
              f"(+ = extension)")

# --- the force plates as ground truth -------------------------------------------------
# At a force-plate heel strike the boot touches the belt, and at a force-plate
# toe-off it leaves it: within a frame or two of each, the boot's lowest point
# must be ON the belt. (Within +-20 ms, because at 100 Hz the frame before a
# landing can still be several mm up.) At toe-off the toe is the last point
# on the belt, so the boot posed with Theia's toe angle checks that angle.
anchor_win = 2                                    # frames either side of the event
anchor_hs = {b: [] for b in limbs}
anchor_to = {b: [] for b in limbs}
for x in range(len(g)):
    if not steady[x]:
        continue
    b = g['support_limb'][x]
    for src, ev, store, hh in (('heel_strike_source', hs, anchor_hs, clear_rigid),
                               ('toe_off_source', to, anchor_to, clear_theia_toe)):
        if g[src][x] != 'GRF' or not np.isfinite(ev[x]) or b not in hh:
            continue
        r_ = frame_to_row(ev[x])
        w_ = hh[b][max(int(np.floor(r_)) - anchor_win, 0):int(np.ceil(r_)) + anchor_win + 1]
        if np.isfinite(w_).any():
            store[b].append(1000 * float(np.nanmin(w_)))
# and WHEN, relative to each force-plate event, Theia's boot actually touches
# down (the rigid boot's lowest point first within 3 mm of the belt) and lifts
# off (the boot, toe bent by the belt, last within 3 mm), searched +-100 ms
touch_ms = {b: [] for b in limbs}
lift_ms = {b: [] for b in limbs}
for x in range(len(g)):
    if not steady[x]:
        continue
    b = g['support_limb'][x]
    for src, ev, store, hh, first in (('heel_strike_source', hs, touch_ms, clear_rigid, True),
                                      ('toe_off_source', to, lift_ms, clear_contact, False)):
        if g[src][x] != 'GRF' or not np.isfinite(ev[x]):
            continue
        r_ = frame_to_row(ev[x])
        r0_ = max(int(np.floor(r_)) - 10, 0)
        w_ = hh[b][r0_:int(np.ceil(r_)) + 11]
        on_ = np.flatnonzero(w_ <= 0.003)
        if not len(on_) or not np.isfinite(w_).all():
            continue
        j_ = on_[0] if first else on_[-1]
        # the crossing of 3 mm, between frame j_ and its neighbour
        k_ = j_ - 1 if first else j_ + 1
        if 0 <= k_ < len(w_) and w_[k_] > 0.003:
            j_ = k_ + (j_ - k_) * (w_[k_] - 0.003) / (w_[k_] - w_[j_])
        store[b].append(10.0 * (r0_ + j_ - r_) * 100.0 / kinematic_fs)
print("  at the FORCE-PLATE events (steady steps), the lowest point within +-20 ms; 0 = on the belt:")
for b in limbs:
    hs_, to_ = np.array(anchor_hs[b]), np.array(anchor_to[b])
    line = f"      {side_name[b]:5s}"
    if len(hs_):
        line += (f"  heel strike, rigid boot {np.median(hs_):+5.1f} mm "
                 f"(IQR {np.percentile(hs_, 25):+.1f} to {np.percentile(hs_, 75):+.1f})")
    if len(to_):
        line += (f";  toe-off, toe by Theia's angle {np.median(to_):+5.1f} mm "
                 f"(IQR {np.percentile(to_, 25):+.1f} to {np.percentile(to_, 75):+.1f})")
    print(line)
print("  (below 0: Theia poses the boot too LOW at that instant, and probably through the")
print("   swing next to it; above 0: too high. Within +-3 mm is as good as the belt fit.")
print("   Toe-off near 0 says Theia's toe angle follows the real toe cap as it leaves the")
print("   belt, the evidence for using its extension in swing)")
print("  when Theia's boot touches down and lifts off, against the force plates:")
for b in limbs:
    if touch_ms[b] or lift_ms[b]:
        print(f"      {side_name[b]:5s}  Theia's boot touches down {np.nanmedian(touch_ms[b]):+4.0f} ms after the "
              f"force-plate heel strike, lifts off {np.nanmedian(lift_ms[b]):+4.0f} ms after its toe-off")
print("  (the same delay at both events, and the same as the clocks check in FORCES, is a")
print("   clock offset; a late touch-down with an on-time lift-off is Theia smoothing the landing)")

if make_figures:
    # ----Plotting the minimum foot position (version 3's figure, three models)
    n_plot = min(5000, n_frames)
    fig, ax = plt.subplots(figsize=(12, 4))
    b = 'R'
    for x in np.flatnonzero((g['support_limb'] == b).to_numpy()):
        if np.isfinite(g['stance_time'][x]) and hs[x] < first_frame + n_plot:
            ax.axvspan(hs[x], to[x], color='red', alpha=0.10, lw=0)
    if b in clear_theia_toe:
        ax.plot(kin_frames[:n_plot], 1000 * clear_theia_toe[b][:n_plot], color='#bbbbbb', lw=1,
                label="toe cap bent by Theia's toe angle (v3)")
    ax.plot(kin_frames[:n_plot], 1000 * foot_clearance['Right_min_z'].to_numpy()[:n_plot], color='k', lw=1,
            label='model used (Theia extension or the belt, whichever lifts more)')
    ax.plot(kin_frames[:n_plot], 1000 * clear_rigid[b][:n_plot], color='#2a5d9f', lw=0.8, alpha=0.6,
            label='rigid boot')
    ax.axhline(0, color='#d94f04', lw=1)
    ax.set_xlim(kin_frames[0], kin_frames[n_plot - 1])
    ax.set_xlabel('Frame')
    ax.set_ylabel('Lowest sole point above the belt (mm)')
    ax.set_title('Minimum boot height above the belt, right foot (shaded = stance)')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()


# =============================================================================
# %% Minimum foot clearance
# =============================================================================
# Schulz (2011; 2017, J Biomech 55, 107-112), on the digitised shoe:
#   clearance(t)  = lowest point of the boot above the belt, every frame
#   minimum toe clearance (MTC) = a local minimum of the TOE (front) half's
#   clearance in mid-swing that meets all three criteria:
#     1. lower than the 2 frames before and after it
#     2. the toe in the upper quartile of its speed in that swing (rules out
#        the dip just after toe-off)
#     3. the heel (rear) half not lower than the toe half (rules out a false
#        minimum at the midfoot as the low point moves from toe to heel)
#   The lowest qualifying minimum if there are several. A swing with none is
#   a "non-MTC" cycle and is left MISSING, not filled with the global minimum
#   (which is at toe-off or heel strike, when the foot is meant to be low).
#
# REVISED:
#  * one definition. Version 3 computed MFC twice: a global minimum within the
#    fastest quarter of the swing (the column that went into the histogram and
#    the stride series), and Schulz's criteria in the trip-risk cell.
#  * the toe speed is measured in the BELT frame. On a treadmill the lab-frame
#    speed of the swing foot is its speed over the belt minus the belt speed,
#    which shifts the upper quartile; Schulz walked overground.
#  * criterion 3 compares the front and rear halves of the scanned sole, as
#    Schulz's toe and heel shoe segments.
gait_event_data['minimum_foot_clearance'] = np.nan
gait_event_data['minimum_foot_clearance_frame'] = np.nan
gait_event_data['mfc_local_minima'] = np.nan
gait_event_data['mfc_on_toe_cap'] = np.nan
gait_event_data['swing_min_clearance'] = np.nan

for x in range(len(g)):
    if not (steady[x] and np.isfinite(g['swing_time'][x])):
        continue
    b = g['support_limb'][x]
    r0, r1 = frame_to_row(to[x]), frame_to_row(nxt[x])
    pad = mfc_swing_trim * (r1 - r0)
    swing = np.arange(int(np.ceil(r0 + pad)), int(np.floor(r1 - pad)) + 1)
    if len(swing) < 10:
        continue
    H = sole_height[b][swing]                                   # (n, V)
    if not np.isfinite(H).all():
        continue
    front, rear = sole_is_front[b], ~sole_is_front[b]
    toe_clear = H[:, front].min(axis=1)
    heel_clear = H[:, rear].min(axis=1)
    # toe speed over the belt: the front half's centroid, plus the belt velocity
    toe_centroid = sole_world[b][swing][:, front].mean(axis=1)
    toe_vel = np.gradient(toe_centroid, 1.0 / kinematic_fs, axis=0) + belt_speed * forward3
    toe_speed = np.linalg.norm(toe_vel, axis=1)
    fast = toe_speed >= np.quantile(toe_speed, mfc_speed_quantile)

    candidates = []
    for i in range(mfc_local_window, len(toe_clear) - mfc_local_window):
        before = toe_clear[i - mfc_local_window:i]
        after = toe_clear[i + 1:i + mfc_local_window + 1]
        # <= not <, so a minimum falling between two samples is still found
        if not (np.all(toe_clear[i] <= before) and np.all(toe_clear[i] <= after)):
            continue
        if not (np.any(toe_clear[i] < before) and np.any(toe_clear[i] < after)):
            continue
        if fast[i] and heel_clear[i] >= toe_clear[i]:
            candidates.append(i)
    candidates = [c_ for j, c_ in enumerate(candidates) if j == 0 or c_ != candidates[j - 1] + 1]
    gait_event_data.loc[x, 'mfc_local_minima'] = len(candidates)
    gait_event_data.loc[x, 'swing_min_clearance'] = H.min()
    if not candidates:
        continue                                # a real non-MTC cycle: missing
    i = min(candidates, key=lambda c_: toe_clear[c_])
    low_vertex = int(np.argmin(H[i]))
    gait_event_data.loc[x, 'minimum_foot_clearance'] = toe_clear[i]
    gait_event_data.loc[x, 'minimum_foot_clearance_frame'] = kin_frames[swing[i]]
    gait_event_data.loc[x, 'mfc_on_toe_cap'] = float(sole_is_toe[b][low_vertex])   # 1 = toe cap

mfc = gait_event_data.loc[steady, 'minimum_foot_clearance']
print("\n" + "-" * 74)
print("  MINIMUM FOOT CLEARANCE (Schulz's criteria, steady swings)")
print("-" * 74)
print(f"  {mfc.notna().sum()} of {int((steady & g['swing_time'].notna()).sum())} swings have an MTC event; "
      f"{int((gait_event_data.loc[steady, 'mfc_local_minima'] == 0).sum())} have none (non-MTC), "
      f"{int((gait_event_data.loc[steady, 'mfc_local_minima'] > 1).sum())} more than one")
print(f"  MFC median {1000 * mfc.median():.1f} mm (IQR {1000 * mfc.quantile(0.25):.1f}-"
      f"{1000 * mfc.quantile(0.75):.1f}), 5th percentile {1000 * mfc.quantile(0.05):.1f} mm, "
      f"SD {1000 * mfc.std():.1f} mm")
print(g.loc[steady].groupby('support_limb')['minimum_foot_clearance'].agg(['mean', 'std', 'count']).round(4))
#brain check: literature, healthy young adults: 10-30 mm (Begg et al. 2007;
#Barrett et al. 2010 review), SD 2-6 mm; Schulz (2017) whole-shoe MTC at this
#speed, overground: 3.15 x speed (leg lengths/s) + 7.51 mm
mfc_expected = 3.15 * belt_speed / participant_leg_length + 7.51
print(f"  Schulz (2017) regression at {belt_speed / participant_leg_length:.2f} leg lengths/s: "
      f"~{mfc_expected:.0f} mm")
mfc_neg = int((mfc < 0).sum())
if mfc_neg:
    print(f"  ! {mfc_neg} MTC events are below the belt. The toe cap cannot cause that")
    print("    any more, so it is the pose of the FOOT itself: check the tracking there")
if mfc.median() < 0.005:
    print("  ! a median under 5 mm is lower than any healthy group reported. Read the")
    print("    force-plate check in BOOTS AND THE BELT: if the boot is posed below the")
    print("    belt at the force-plate events, Theia places the swinging foot too low,")
    print("    and MFC and TRI from this trial measure that, not the walker")


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
# Reported as recommended by Curtze, Buurke & McCrum (2024, "Notes on the
# margin of stability", J Biomech -- verify the details before citing): the
# CoM definition, the base-of-support boundary, the instant at which it is
# read, the pendulum length and the velocity frame are all stated, and
# margins are also given normalised to leg length:
#   CoM        the SYSTEM CoM, body + load (below)
#   velocity   fused markers + force plates, in the BELT frame
#   l          CoM to stance ankle, per frame (or leg length)
#   boundary   the scanned boot's sole: its outer edge (ML) and front (AP);
#              the ankle joint centre version is kept for the literature
#   instants   heel strike (the most widely reported value); the minimum over
#              single support (the stance boot alone is the base of support);
#              and, for AP, the minimum over double support (Schulz 2017's
#              definition of the standard MoS)
#
# REVISED:
#  * the CoM of body AND load. A 20-40 kg pack moves the CoM by several
#    centimetres, and Theia's Whole_body_COG is the body alone.
#  * the forces are the ones put into Theia's frame (Forces cell), and the
#    system mass is the one weighed in the quiet standing.
#  * AP and ML along and across the measured direction of travel.
#  * the minima are over single support, where the stance boot alone is the
#    base of support. Version 3 took the whole stance, so its AP "minimum"
#    was read at toe-off against the TRAILING boot, after the other boot had
#    landed ahead of it and become the real front of the base of support.
#    Even over single support the AP margin is negative in every healthy step
#    (the xCoM passes the stance boot's front: each step catches the body),
#    so mos_ap_min sits around -0.5 to -0.8 m at 1.3 m/s. The values to put
#    beside the literature are the heel-strike value and Schulz's minimum
#    over double support (Schulz 2017: "the single minimum value per gait
#    cycle during double support").
#  * AP at heel strike uses the front of the LANDING boot, and the double-
#    support minimum is added.

print("\n" + "-" * 74)
print("  MARGIN OF STABILITY")
print("-" * 74)

# =============================================================================
# ----The system (body + load) centre of mass
# =============================================================================
# Standing still, the CoP is vertically below the system CoM. Horizontally:
#     m * x_CoP = m_body * x_body + m_load * x_load
#     x_load    = (m * x_CoP - m_body * x_body) / m_load
# Height: standing tells nothing about it, so the load sits at
# Trunk_Position (+ load_height_m). Left-right: on the mid-line -- the equation
# multiplies any CoP error by m / m_load (~5 for 20 kg on 80 kg), and packs and
# vests are symmetric. The load is stored in a trunk frame (Low_Back -> Neck
# up, right, forward) and carried with the trunk through the trial:
#     x_CoM = (m_body x_body + m_load x_load) / m
mos_com_body = xyz('Whole_body_COG')
mos_com = mos_com_body.copy()
load_offset_local = np.full(3, np.nan)
trunk_pos = xyz('Trunk_Position')
# REVISED (after D05): a load under load_place_min_kg (boots, clothing, a light
# vest; D05 C1 weighed 4.7 kg) is not placed. The equation above multiplies any
# CoP or CoM error by m / m_load (x18 at 4.7 kg), so its answer is noise, and
# putting it on the trunk instead moved the CoM 19 mm on a guess. It is spread
# like the body: the CoM stays Theia's, the mass is the one weighed.
if np.isfinite(load_kg) and load_min_kg <= load_kg < load_place_min_kg:
    print(f"  load {load_kg:.1f} kg is under {load_place_min_kg:g} kg (boots, clothing): spread like the "
          f"body, the CoM is Theia's and the mass the one weighed ({system_mass:.1f} kg)")
elif (quiet_rows is not None and np.isfinite(load_kg) and load_kg >= load_place_min_kg
        and trunk_pos is not None and np.isfinite(quiet_cop).all()):
    lo, hi = xyz('Low_Back_Position'), xyz('Neck_Position')
    up = np.tile([0.0, 0.0, 1.0], (n_frames, 1))
    if lo is not None and hi is not None:
        u_ = hi - lo
        ok = np.isfinite(u_).all(axis=1)
        up[ok] = u_[ok] / np.linalg.norm(u_[ok], axis=1, keepdims=True)
    right = -lateral3
    x_axis = right - (up @ right)[:, None] * up
    x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
    trunk_R = np.stack([x_axis, np.cross(up, x_axis), up], axis=2)    # columns: right, forward, up
    a_, b_ = quiet_rows
    load_xy = (system_mass * quiet_cop - body_mass * np.nanmean(mos_com_body[a_:b_, :2], axis=0)) / load_kg
    load_p = np.r_[load_xy, np.nanmean(trunk_pos[a_:b_, 2]) + load_height_m]
    load_offset_local = np.nanmedian(np.einsum('fji,fj->fi', trunk_R[a_:b_], load_p - trunk_pos[a_:b_]), axis=0)
    load_offset_local[0] = 0.0                                          # on the mid-line
    if np.hypot(*(load_p[:2] - np.nanmean(trunk_pos[a_:b_, :2], axis=0))) > load_max_offset_m:
        print(f" ! the load would sit over {1000 * load_max_offset_m:.0f} mm from the trunk: "
              f"not believed, placed on the trunk")
        load_offset_local[:2] = 0.0
    load_position = trunk_pos + np.einsum('fij,j->fi', trunk_R, load_offset_local)
    mos_com = (body_mass * mos_com_body + load_kg * load_position) / system_mass
    print(f"  load CoM {1000 * load_offset_local[1]:+.0f} mm forward of Trunk_Position "
          f"(negative = behind), on the mid-line, {1000 * load_height_m:+.0f} mm up (assumed); "
          f"system CoM {1000 * np.nanmedian(np.linalg.norm(mos_com - mos_com_body, axis=1)):.0f} mm "
          f"from the body's")
else:
    print("  no load placed (no quiet standing, no body mass, or no load): the CoM is Theia's body CoM")

# =============================================================================
# ----CoM velocity, fused from markers and force plates
# =============================================================================
# Velocity is the weak link in the whole margin calculation:
#   differentiating the marker CoM is good at low frequency but amplifies
#     noise exactly where the within-stride dynamics are
#   integrating GRF / m is exact at high frequency (Newton, 1000 Hz, no
#     differentiation) but drifts slowly with any offset
# A complementary filter takes each where it is good, with the SAME cutoff on
# both branches so their transfer functions sum to one:
#     v = lowpass(v_markers) + [ v_force - lowpass(v_force) ]
# Velocity error enters the xCoM divided by w0 (~3.2 /s): 30 mm/s is ~9 mm of
# margin, against margins of a few centimetres.
mos_com_filled, mos_com_gap = fill_gaps(mos_com)
mos_v_markers = savgol_filter(mos_com_filled, mos_savgol_window, mos_savgol_poly,
                              deriv=1, delta=1.0 / kinematic_fs, axis=0)

grf_total_raw = grf_raw['L'] + grf_raw['R']
mos_accel_1k = (grf_total_raw - grf_total_raw.mean(axis=0)) / system_mass
mos_aa = butter(4, mos_antialias_hz, fs=force_fs, output='sos')
mos_accel_1k = sosfiltfilt(mos_aa, mos_accel_1k, axis=0)
mos_force_rows = np.arange(n_frames) * force_step
mos_have = mos_force_rows < n_force
mos_accel = np.full((n_frames, 3), np.nan)
mos_accel[mos_have] = mos_accel_1k[mos_force_rows[mos_have]]
mos_handrail_row = np.zeros(n_frames, bool)
mos_handrail_row[mos_have] = handrail_contact[mos_force_rows[mos_have]]
if (~mos_have).mean() > 0.05:
    print(f" ! {100 * (~mos_have).mean():.0f}% of kinematic frames have no force data: markers alone there")
mos_accel, _ = fill_gaps(mos_accel)

# REVISED: the axes check. GRF / m must equal the CoM's acceleration, so the
# two are correlated per axis (band-passed 0.5-5 Hz). This tests the plate ->
# Theia rotation, the sign of each axis and the mass, before they are trusted
mos_band = butter(2, [0.5, 5.0], btype='band', fs=kinematic_fs, output='sos')
mos_a_markers = savgol_filter(mos_com_filled, mos_savgol_window, mos_savgol_poly,
                              deriv=2, delta=1.0 / kinematic_fs, axis=0)
mos_axes_r = {}
for k, name in enumerate(('X', 'Y', 'Z')):
    mos_axes_r[name] = float(np.corrcoef(sosfiltfilt(mos_band, mos_a_markers[:, k]),
                                         sosfiltfilt(mos_band, mos_accel[:, k]))[0, 1])
    if k < 2 and mos_axes_r[name] < -0.5:
        print(f"  ! GRF {name} runs AGAINST the CoM acceleration (r = {mos_axes_r[name]:.2f}): flipped")
        mos_accel[:, k] *= -1
        for b in limbs:
            grf[b][:, k] *= -1
            grf_raw[b][:, k] *= -1

mos_v_force = np.zeros_like(mos_accel)
mos_v_force[1:] = np.cumsum((mos_accel[:-1] + mos_accel[1:]) / 2, axis=0) / kinematic_fs
mos_v_force -= mos_v_force.mean(axis=0)
mos_cb, mos_ca = butter(2, mos_crossover_hz / (kinematic_fs / 2), 'low')
mos_com_velocity = (filtfilt(mos_cb, mos_ca, mos_v_markers, axis=0)
                    + mos_v_force - filtfilt(mos_cb, mos_ca, mos_v_force, axis=0))
mos_bad = np.convolve(mos_com_gap, np.ones(mos_savgol_window), 'same') > 0
mos_com_velocity[mos_bad] = np.nan       # untracked frames stay missing

# in the belt frame the walker travels forward even though the lab-frame mean
# is about zero. Every AP margin needs this.
mos_com_velocity_belt = mos_com_velocity + belt_speed * forward3

print("  GRF / mass vs marker CoM acceleration (0.5-5 Hz): r = "
      + " / ".join(f"{v:.2f}" for v in mos_axes_r.values()) + "  (X / Y / Z)")
if min(mos_axes_r.values()) < 0.7:
    print("  ! weak agreement: the force axes or the mass are off, and the fused")
    print("    velocity inherits it. Check the event summary's registration")
print(f"  CoM speed along travel: lab {np.nanmean(mos_com_velocity @ forward3):+.3f} m/s "
      f"(should be ~0), belt frame {np.nanmean(mos_com_velocity_belt @ forward3):+.3f} m/s")
mos_delta = 1000 * np.nanstd(mos_com_velocity - mos_v_markers, axis=0)
print(f"  fusion moved the velocity by {mos_delta[0]:.1f} / {mos_delta[1]:.1f} / "
      f"{mos_delta[2]:.1f} mm/s RMS (X / Y / Z) against the markers alone")

# the boot's extent along the direction of travel and across it, every frame
boot_front = {b: np.nanmax(sole_world[b][..., :2] @ forward, axis=1) for b in limbs}
boot_rear = {b: np.nanmin(sole_world[b][..., :2] @ forward, axis=1) for b in limbs}
boot_left = {b: np.nanmax(sole_world[b][..., :2] @ lateral, axis=1) for b in limbs}
boot_right = {b: np.nanmin(sole_world[b][..., :2] @ lateral, axis=1) for b in limbs}

# =============================================================================
# ----Margin of stability, per step
# =============================================================================
for col in ('mos_ml_contact', 'mos_ml_min', 'mos_ml_min_at_pct', 'mos_ap_contact',
            'mos_ap_min', 'mos_ap_ds_min', 'mos_ml_contact_ankle', 'mos_pendulum_length'):
    gait_event_data[col] = np.nan
gait_event_data['mos_handrail_contact'] = False

for x in range(len(g)):
    if not (np.isfinite(g['stance_time'][x])):
        continue
    b, o = g['support_limb'][x], other_limb[g['support_limb'][x]]
    rows = np.arange(int(np.ceil(frame_to_row(hs[x]))), int(np.floor(frame_to_row(to[x]))) + 1)
    rows = rows[rows < n_frames]
    if len(rows) < 3:
        continue
    # outward = away from the other ankle at heel strike, across the belt, so
    # the sign does not depend on which way Theia's axes point
    side = np.sign((ankle[b][rows[0], :2] - ankle[o][rows[0], :2]) @ lateral)
    if not np.isfinite(side) or side == 0:
        continue
    pend = np.linalg.norm(mos_com[rows] - ankle[b][rows], axis=1)
    if mos_pendulum_mode == 'leg_length':
        pend = np.full(len(rows), participant_leg_length)
    w0 = np.sqrt(gravity / pend)
    xcom = mos_com[rows, :2] + mos_com_velocity_belt[rows, :2] / w0[:, None]
    edge = boot_left[b][rows] if side > 0 else -boot_right[b][rows]
    ml = edge - side * (xcom @ lateral)
    ap = boot_front[b][rows] - xcom @ forward
    if not np.isfinite(ml).any():
        continue
    single = (rows >= frame_to_row(cto[x])) & (rows <= frame_to_row(chs[x]))
    double = rows < frame_to_row(cto[x])                     # initial double support
    gait_event_data.loc[x, 'mos_ml_contact'] = ml[0]
    gait_event_data.loc[x, 'mos_ap_contact'] = ap[0]
    gait_event_data.loc[x, 'mos_pendulum_length'] = pend[0]
    gait_event_data.loc[x, 'mos_ml_contact_ankle'] = side * (ankle[b][rows[0], :2] @ lateral) - side * (xcom[0] @ lateral)
    if single.sum() >= 3 and np.isfinite(ml[single]).any():
        k = np.flatnonzero(single)[np.nanargmin(ml[single])]
        gait_event_data.loc[x, 'mos_ml_min'] = ml[k]
        gait_event_data.loc[x, 'mos_ml_min_at_pct'] = 100 * k / (len(rows) - 1)
        gait_event_data.loc[x, 'mos_ap_min'] = np.nanmin(ap[single])
    if double.sum() >= 2:
        gait_event_data.loc[x, 'mos_ap_ds_min'] = np.nanmin(ap[double])
    gait_event_data.loc[x, 'mos_handrail_contact'] = bool(mos_handrail_row[rows].any())

# =============================================================================
# ----Brain checks and plots
# =============================================================================
mos_cols = ['mos_ml_contact', 'mos_ml_min', 'mos_ap_contact', 'mos_ap_min', 'mos_ap_ds_min']
print(f"  {gait_event_data.loc[steady, 'mos_ml_contact'].notna().sum()} steady steps produced a margin")
print((1000 * gait_event_data.loc[steady].groupby('support_limb')[mos_cols].mean()).round(1).rename(
    columns=lambda c_: c_ + '_mm'))
mos_ml_mean = gait_event_data.loc[steady, 'mos_ml_contact'].mean()
if not (0.0 < mos_ml_mean < 0.20):
    print(f"  ! ML margin averages {1000 * mos_ml_mean:.0f} mm, outside the 0-200 mm a "
          f"healthy adult should show. Check the lateral direction and the belt")
if gait_event_data.loc[steady, 'mos_ap_min'].mean() > 0:
    print("  ! the AP margin never goes negative in single support: during single")
    print("    support the xCoM should pass the stance boot's front. Check the belt speed")
mos_gap = 1000 * (gait_event_data.loc[steady, 'mos_ml_contact'] - gait_event_data.loc[steady, 'mos_ml_contact_ankle'])
print(f"  boot edge vs ankle joint centre: the boot gives a margin {mos_gap.mean():.1f} +- "
      f"{mos_gap.std():.1f} mm larger (the ankle-to-edge distance; expect 30-100 mm for a boot)")
if gait_event_data['mos_handrail_contact'][steady].any():
    print(f"  {int(gait_event_data['mos_handrail_contact'][steady].sum())} steady steps had a hand "
          f"on a rail: their margins rest on a velocity whose assumption does not hold")
# expected size of the extrapolation: belt speed / w0
mos_v_w0 = belt_speed / np.sqrt(gravity / gait_event_data.loc[steady, 'mos_pendulum_length'].median())
print(f"  v / w0 at this speed: {1000 * mos_v_w0:.0f} mm (how far ahead of the CoM the xCoM sits)")

#Between-subject comparison needs a normaliser: a 40 mm margin means something
#different on a 0.75 m leg than on a 0.95 m one
if participant_leg_length:
    for col in mos_cols:
        gait_event_data[col + '_norm'] = gait_event_data[col] / participant_leg_length

if make_figures:
    plt.figure()
    plt.plot(gait_event_data['mos_ml_contact'], label='at heel strike')
    plt.plot(gait_event_data['mos_ml_min'], label='minimum over single support')
    plt.plot(gait_event_data['mos_ml_contact_ankle'], label='at heel strike, ankle boundary', alpha=0.5)
    plt.axhline(0, color='k', lw=0.8)
    plt.title('Mediolateral margin of stability (m)')
    plt.xlabel('Step')
    plt.legend()

    plt.figure()
    plt.plot(gait_event_data['mos_ap_contact'], label='at heel strike (front of the landing boot)')
    plt.plot(gait_event_data['mos_ap_ds_min'], label='minimum over double support (Schulz)')
    plt.plot(gait_event_data['mos_ap_min'], label='minimum over single support')
    plt.axhline(0, color='k', lw=0.8)
    plt.title('Anteroposterior margin of stability (m)  -- negative in single support is normal')
    plt.xlabel('Step')
    plt.legend()


# =============================================================================
# ----Birds-eye video of a representative stride
# =============================================================================
# The margin of stability lives in the ground plane, so a birds-eye view is the
# only one that shows it without foreshortening. Drawn in the BELT frame
# (positions shifted forward by belt speed x time) so the stance foot stays
# planted and the body advances, as over ground. Margins are differences
# between positions at the same instant, so the shift does not change them.
# REVISED: drawn in the walker's axes (along / across the direction of travel)
# and from the boot SOLE (the contact area), and only when make_video = True
# (it takes minutes).
if make_video:
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.animation import FFMpegWriter, PillowWriter
    import shutil

    vid_limb = 'R'
    vid_vars = ['stride_time', 'step_width', 'mos_ml_contact', 'mos_ap_contact', 'mos_ml_min']
    vid_pool = gait_event_data[(gait_event_data['support_limb'] == vid_limb) & gait_event_data['steady']
                               & gait_event_data[vid_vars + ['next_ipsi_heelstrike']].notna().all(axis=1)]
    vid_score = sum(np.abs(vid_pool[v] - vid_pool[v].median())
                    / max(vid_pool[v].quantile(0.75) - vid_pool[v].quantile(0.25), 1e-9) for v in vid_vars)
    vid_x = int(vid_score.idxmin())
    vid_rows = np.arange(int(np.ceil(frame_to_row(hs[vid_x]))), int(np.floor(frame_to_row(nxt[vid_x]))) + 1)
    vid_shift = belt_speed * (vid_rows - vid_rows[0]) / kinematic_fs
    vid_side, vid_other = vid_limb, other_limb[vid_limb]
    vid_stance = {b: stance_rows[b][vid_rows] for b in limbs}

    def vid_plane(p):          # world xy -> (along travel + belt shift, across)
        return np.column_stack([p[..., :2] @ forward, p[..., :2] @ lateral])

    vid_hull = {b: [] for b in limbs}
    for b in limbs:
        for i, r in enumerate(vid_rows):
            pp = vid_plane(sole_world[b][r]) + [vid_shift[i], 0]
            vid_hull[b].append(pp[ConvexHull(pp).vertices])
    vid_com = vid_plane(mos_com[vid_rows]) + np.column_stack([vid_shift, np.zeros(len(vid_rows))])
    vid_w0 = np.sqrt(gravity / np.linalg.norm(mos_com[vid_rows] - ankle[vid_side][vid_rows], axis=1))
    vid_xcom = vid_com + vid_plane(mos_com_velocity_belt[vid_rows]) / vid_w0[:, None]
    vid_out = base / "MoS_outputs" / f"{trial}_MoS_stride"
    vid_out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 6))
    ax_map = fig.add_axes((0.07, 0.12, 0.9, 0.8))
    vid_all = np.vstack([np.vstack(vid_hull['L']), np.vstack(vid_hull['R']), vid_com])
    vid_lo, vid_hi = vid_all.min(axis=0) - 0.25, vid_all.max(axis=0) + 0.25

    def vid_draw(i):
        ax_map.clear()
        for b, colour in (('L', '#2a78d6'), ('R', '#eb6834')):
            ax_map.add_patch(MplPolygon(vid_hull[b][i], closed=True,
                                        facecolor=colour if vid_stance[b][i] else 'none',
                                        edgecolor=colour, lw=2, alpha=0.6 if vid_stance[b][i] else 1))
        ax_map.annotate('', xy=vid_xcom[i], xytext=vid_com[i],
                        arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))
        ax_map.plot(*vid_com[i], 'o', color='k', ms=9)
        ax_map.plot(*vid_xcom[i], 'o', mfc='none', mec='k', ms=9, mew=2)
        ax_map.set_xlim(vid_lo[0], vid_hi[0])
        ax_map.set_ylim(vid_lo[1], vid_hi[1])
        ax_map.set_aspect('equal')
        ax_map.set_xlabel('along the direction of travel, belt frame (m)')
        ax_map.set_ylabel('to the left (m)')
        ax_map.set_title(f"{side_name[vid_side]} stride, t = {(vid_rows[i] - vid_rows[0]) / kinematic_fs:.2f} s  "
                         f"(filled = on the belt; dot = CoM, ring = xCoM)")

    vid_exe = shutil.which('ffmpeg')
    vid_writer, vid_ext = ((FFMpegWriter(fps=20, bitrate=3000), '.mp4') if vid_exe
                           else (PillowWriter(fps=20), '.gif'))
    with vid_writer.saving(fig, str(vid_out) + vid_ext, 110):
        for i in range(len(vid_rows)):
            vid_draw(i)
            vid_writer.grab_frame()
    print(f"  wrote {vid_out}{vid_ext}")


# =============================================================================
# %% Margin of instability and trip risk
# =============================================================================
# Schulz (2017) J Biomech 55, 107-112.
#
# Minimum foot clearance says how CLOSE the foot came to the ground. It does
# not say what would have happened if the foot had caught. Trip risk needs
# both: a low clearance in a stable configuration is recoverable, the same
# clearance in an unstable one is not.
#
# MARGIN OF INSTABILITY is the AP margin of stability with three changes:
#   1. the CONTINUOUS trajectory through the swing
#   2. the boundary is the most anterior point of EITHER boot (the stance toe
#      early in swing, the swing toe late), where the front of the base of
#      support WOULD be if the swing foot came down now
#   3. stable (positive) margins set to zero, then negated: MoI >= 0
#
#       MoI(t)   = max( xCoM_AP(t) - front of either boot(t), 0 )     mm
#       risk(t)  = MoI(t) / MFC(t)      MFC(t) = the whole boot's clearance
#       TRI      = integral of risk(t) dt, between the peak ACCELERATION of
#                  the MFC point after lift-off and its peak DECELERATION
#                  before landing                                      s
#
# Time is NOT normalised: a longer swing gives a larger TRI (Schulz's choice).
# Higher MFC means LESS risk, higher TRI MORE; Schulz's central result is that
# they move OPPOSITELY with speed.
#
# REVISED:
#  * THE INTEGRATION WINDOW (why version 3 gave TRI = 0). Schulz's Fig. 2
#    plots the RESULTANT acceleration of the point of MFC on the shoe (always
#    positive): one peak just after lift-off (the foot accelerating) and one
#    just before landing (the foot decelerating), and integrates between
#    them -- nearly the whole swing. Version 3 took the global maximum and
#    minimum of d|v|/dt anywhere in the swing; the landing impact gave both,
#    so the window sat in the last ~40 ms, after MoI had already returned to
#    zero. Now: the point of MFC is the sole vertex that is lowest in each
#    frame (Schulz's "point of MFC"; its speed starts and ends near zero, as
#    in his figure), the acceleration peak is searched BEFORE that point's
#    speed peak and the deceleration peak AFTER it.
#  * the velocity and acceleration are Savitzky-Golay derivatives (7 frames,
#    order 3) of each sole point's path over the swing PLUS 3 frames either
#    side. A plain np.gradient is one-sided at the swing's first frame,
#    where the boot is still on the belt; its spurious "peak" there pulled
#    the lift-off spike (MoI / 1 mm) into the integral whenever the event
#    times were whole frames (TRI +25% on the synthetic trial).
#  * the pendulum is the STANCE leg's (CoM to the other foot's ankle).
#    Version 3 used the swinging foot's ankle.
#  * velocities in the belt frame; the boot fronts from the scanned sole.
#
# WHAT TO EXPECT: MoI is zero just after toe-off (the stance boot is still well
# ahead of the xCoM), peaks in mid-swing as the stance boot falls behind, and
# returns to zero once the swing boot reaches past the xCoM. Its size scales
# with v / w0, so it grows quickly with speed: Schulz's Fig. 2 example
# (MoI 350-700 mm, MFC point at 8 m/s) is a FAST walk; at 1.3 m/s, v/w0 is
# ~400 mm and the MoI peak is a few hundred mm. Schulz's regression on
# speed (no obstacles, Fig. 4): TRI ~ 3.18 x speed (leg lengths/s) - 2.02.
for col in ('moi_peak_mm', 'moi_mean_mm', 'trip_risk_peak', 'trip_risk_integral', 'tri_window_s'):
    gait_event_data[col] = np.nan

tri_example = None
tri_window_edge = 0
for x in range(len(g)):
    if not (steady[x] and np.isfinite(g['swing_time'][x])):
        continue
    b, o = g['support_limb'][x], other_limb[g['support_limb'][x]]
    swing = np.arange(int(np.ceil(frame_to_row(to[x]))), int(np.floor(frame_to_row(nxt[x]))) + 1)
    swing = swing[swing < n_frames]
    if len(swing) < 10:
        continue
    H = sole_height[b][swing]
    if not np.isfinite(H).all() or not np.isfinite(mos_com_velocity_belt[swing]).all():
        continue
    clear_mm = 1000.0 * H.min(axis=1)                          # whole-boot MFC trajectory
    low = H.argmin(axis=1)                                     # the point of MFC, per frame

    # the margin of instability, the STANCE leg's pendulum (the other foot)
    pend = np.linalg.norm(mos_com[swing] - ankle[o][swing], axis=1)
    w0 = np.sqrt(gravity / pend)
    xcom_ap = mos_com[swing, :2] @ forward + (mos_com_velocity_belt[swing, :2] @ forward) / w0
    anterior = np.maximum(boot_front[b][swing], boot_front[o][swing])
    moi = 1000.0 * np.maximum(xcom_ap - anterior, 0.0)
    risk = moi / np.maximum(clear_mm, tri_min_clearance_mm)

    # the point of MFC: velocity and resultant acceleration of the vertex that
    # is lowest in each frame, each read off that vertex's own trajectory,
    # differentiated over the swing plus a few frames either side
    tri_pad = tri_deriv_window // 2
    padded = np.clip(np.arange(swing[0] - tri_pad, swing[-1] + tri_pad + 1), 0, n_frames - 1)
    Wpad = sole_world[b][padded].astype(float)
    if not np.isfinite(Wpad).all():
        Wpad = fill_gaps(Wpad)[0]
    inner = slice(tri_pad, tri_pad + len(swing))
    vel = savgol_filter(Wpad, tri_deriv_window, 3, deriv=1, delta=1.0 / kinematic_fs, axis=0)[inner] \
        + belt_speed * forward3
    acc = savgol_filter(Wpad, tri_deriv_window, 3, deriv=2, delta=1.0 / kinematic_fs, axis=0)[inner]
    k_ = np.arange(len(swing))
    point_speed = np.linalg.norm(vel[k_, low], axis=1)
    point_acc = np.linalg.norm(acc[k_, low], axis=1)
    i_peak = int(np.argmax(point_speed))
    if i_peak < 2 or i_peak > len(swing) - 3:
        tri_window_edge += 1
        continue
    a_i = int(np.argmax(point_acc[:i_peak]))                    # lift-off acceleration
    b_i = i_peak + int(np.argmax(point_acc[i_peak:]))           # landing deceleration

    gait_event_data.loc[x, 'moi_peak_mm'] = np.nanmax(moi)
    gait_event_data.loc[x, 'moi_mean_mm'] = np.nanmean(moi[a_i:b_i + 1])
    gait_event_data.loc[x, 'trip_risk_peak'] = np.nanmax(risk[a_i:b_i + 1])
    gait_event_data.loc[x, 'trip_risk_integral'] = np.nansum(risk[a_i:b_i + 1]) / kinematic_fs
    gait_event_data.loc[x, 'tri_window_s'] = (b_i - a_i) / kinematic_fs

    # keep a typical mid-trial swing for the explanatory figure
    if tri_example is None and x > len(g) // 2:
        tri_example = dict(t=(swing - swing[0]) / kinematic_fs, clear=clear_mm, moi=moi, risk=risk,
                           speed=point_speed, acc=point_acc, a=a_i, b=b_i, side=b, index=x,
                           mfc=gait_event_data.loc[x, 'minimum_foot_clearance'],
                           mfc_t=(frame_to_row(gait_event_data.loc[x, 'minimum_foot_clearance_frame'])
                                  - swing[0]) / kinematic_fs)

# --- report ---------------------------------------------------------------------
tri_steady = gait_event_data.loc[steady]
print("\n" + "-" * 74)
print("  MARGIN OF INSTABILITY AND TRIP RISK (steady swings)")
print("-" * 74)
print(f"  {tri_steady['trip_risk_integral'].notna().sum()} swings produced a TRI"
      + (f"; {tri_window_edge} had the MFC point's speed peak at the edge of the swing" if tri_window_edge else ""))
print(tri_steady.groupby('support_limb')[['moi_peak_mm', 'trip_risk_integral', 'tri_window_s']].agg(['mean', 'std']).round(3))
tri_speed_ll = belt_speed / participant_leg_length
tri_expected = 3.18 * tri_speed_ll - 2.02
print(f"  Schulz (2017), no obstacles, at {tri_speed_ll:.2f} leg lengths/s: TRI ~ {tri_expected:.1f} "
      f"(his Fig. 4 shows roughly +-2 around the line)")
if tri_steady['trip_risk_integral'].median() < 0.05:
    print("  ! TRI is essentially zero: either the xCoM never passes the boots (very slow")
    print("    walking -- Schulz's floor below ~0.75 m/s), or the window or velocity is wrong")

if make_figures and tri_example is not None:
    e = tri_example
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(e['t'], e['clear'], color='#1b7a3d', lw=1.8)
    if np.isfinite(e['mfc']):
        axes[0].plot(e['mfc_t'], 1000 * e['mfc'], 'o', ms=9, color='#1b7a3d',
                     label=f"MTC = {1000 * e['mfc']:.0f} mm")
        axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_ylabel('clearance (mm)')
    axes[0].set_title(f"How the trip risk integral is built -- {side_name[e['side']]} swing, "
                      f"index {e['index']}", fontsize=11)
    axes[1].plot(e['t'], e['moi'], color='#a8442a', lw=1.8)
    axes[1].set_ylabel('MoI (mm)')
    axes[1].annotate('zero where the boots reach past the xCoM;\npositive = a catch here '
                     'would throw the body forward', (0.02, 0.95), xycoords='axes fraction',
                     va='top', fontsize=8, color='#555555')
    ax2 = axes[2]
    ax2.plot(e['t'], e['acc'], color='#7a4fb5', lw=1.5, label='resultant acceleration (m/s$^2$)')
    ax2.plot(e['t'][e['a']], e['acc'][e['a']], '^', ms=9, color='#7a4fb5')
    ax2.plot(e['t'][e['b']], e['acc'][e['b']], 'v', ms=9, color='#7a4fb5')
    ax2.set_ylabel('MFC point\nacceleration (m/s$^2$)')
    ax2b = ax2.twinx()
    ax2b.plot(e['t'], e['speed'], color='#333333', lw=1.2, label='speed (m/s)')
    ax2b.set_ylabel('speed (m/s)')
    ax2.annotate('window: peak acceleration after lift-off ->\npeak deceleration before landing',
                 (0.02, 0.95), xycoords='axes fraction', va='top', fontsize=8, color='#555555')
    axes[3].plot(e['t'], e['risk'], color='#111111', lw=1.8)
    axes[3].fill_between(e['t'][e['a']:e['b'] + 1], 0, e['risk'][e['a']:e['b'] + 1],
                         color='#d94f04', alpha=0.35,
                         label=f"TRI = {np.nansum(e['risk'][e['a']:e['b'] + 1]) / kinematic_fs:.2f} s")
    axes[3].set_ylabel('MoI / MFC')
    axes[3].set_xlabel('time through swing (s)')
    axes[3].legend(fontsize=9, frameon=False)
    for ax in axes:
        ax.axvline(e['t'][e['a']], color='#bbbbbb', lw=1, ls='--')
        ax.axvline(e['t'][e['b']], color='#bbbbbb', lw=1, ls='--')
        ax.grid(alpha=0.15)
    plt.tight_layout()

if make_figures:
    # MFC distribution: the tail is the part that matters
    plt.figure(figsize=(9, 3.6))
    plt.hist(1000 * mfc.dropna(), bins=60, color='#2a5d9f', alpha=0.8)
    plt.axvline(1000 * mfc.median(), color='k', lw=1.5, label='median')
    plt.axvline(1000 * mfc.quantile(0.05), color='#d94f04', lw=1.5,
                label='5th percentile, the trip-relevant tail')
    plt.axvline(mfc_expected, color='#1b7a3d', lw=1.5, ls='--',
                label='Schulz (2017) at this speed')
    plt.axvline(0, color='k', lw=0.8, ls=':')
    plt.xlabel('minimum toe clearance (mm)')
    plt.ylabel('swings')
    plt.title('MFC distribution -- the LOW TAIL is what catches an obstacle')
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()


# =============================================================================
# %% Kinetic metrics
# =============================================================================
# The metric family that only exists because the treadmill is instrumented.
# Everything is per limb, which the split belts give directly -- WHEN one
# foot is alone on one belt.
#
# WHAT EACH MEASURE IS FOR
#   F1, trough, F2   weight acceptance peak, mid-stance unloading, push-off
#                    peak (vertical GRF)
#   loading rate     average slope from 20% to 80% of F1
#   braking / propulsive impulse   AP force integrated where it opposes /
#                    drives travel; they cancel over a stride at steady speed
#   CoP excursion    along and across the direction of travel
#   free moment      transverse-plane torque against the ground
#
# REVISED:
#  * TRUSTED STANCES ONLY. A stance is used only if its heel strike AND toe-off
#    were both trusted force-plate events of the same belt contact (one boot
#    alone on that belt, the CoP under it: detect_gait_events.py). A step
#    onto the gap, a crossover or a shared belt carries two feet's force or
#    part of one; version 3 read every stance from the foot's own belt.
#    The belt is the one the event record names, so a clean crossover step
#    is read from the belt it actually landed on.
#  * the force is baseline-removed and in Theia's frame (Forces cell), and
#    AP is along the measured direction of travel (positive = forward).
#  * the force samples are the events' own (1 ms), not frame x 10.
#  * per BODY weight (_bw, the convention) AND per TOTAL weight (_tw, body +
#    load weighed in the quiet standing). Version 3's "body weight" was the
#    mean GRF of the whole trial, i.e. body + load. With a load, _bw rises
#    roughly in proportion to the added weight (Birrell et al. 2007); _tw
#    shows whether the pattern changed beyond carrying more.
#  * the loading rate is the slope of the FILTERED force (the 1000 Hz raw
#    force has several N of noise per sample at the 20% and 80% points).
print("\n" + "-" * 74)
print("  KINETIC METRICS")
print("-" * 74)

# --- the free moment, per belt (version 3's method, kept) ----------------------
# Nothing in the export documents the moment convention. Regressing the CoP on
# -My/Fz and Mx/Fz recovers each plate's moment origin, and the residual says
# whether the convention holds. It matters for the free moment, which needs the
# CoP relative to that origin: using the lab-frame CoP by mistake adds a
# spurious term of tens of N*m to a real signal of a few.
b_cop, a_cop = butter(4, kmx_cop_filter_hz / (force_fs / 2), 'low')
free_moment = {}
for b in limbs:
    side = belt_name[b]
    raw = {a: force_data[f'{side}_COP_{a}'].to_numpy(float) for a in 'XY'}
    F = {a: force_data[f'{side}_Force_{a}'].to_numpy(float) for a in 'XYZ'}
    M = {a: force_data[f'{side}_Moment_{a}'].to_numpy(float) for a in 'XYZ'}
    ok = force_loaded[b] & (np.abs(F['Z']) > 200)
    ox = float(np.median(raw['X'][ok] + 1000.0 * M['Y'][ok] / F['Z'][ok]))
    oy = float(np.median(raw['Y'][ok] - 1000.0 * M['X'][ok] / F['Z'][ok]))
    resid = float(np.median(np.abs(-1000.0 * M['Y'][ok] / F['Z'][ok] + ox - raw['X'][ok])))
    print(f"  {side} moment origin ({ox:+.1f}, {oy:+.1f}) mm, CoP reconstruction residual "
          f"{resid:.2f} mm (median)")
    if resid > 1.0:
        print("   ! over 1 mm: the moment convention is not what this assumes; the free")
        print("     moment below is unreliable")
    cop_f = {a: filtfilt(b_cop, a_cop, raw[a]) for a in 'XY'}
    rx, ry = (cop_f['X'] - ox) / 1000.0, (cop_f['Y'] - oy) / 1000.0
    free_moment[b] = filtfilt(b_cop, a_cop, M['Z'] - (rx * F['Y'] - ry * F['X']))

# --- which stances can be trusted ------------------------------------------------
kmx_has_belts = events_qa_path.exists()
kmx_trusted = ((g['heel_strike_source'] == 'GRF') & (g['toe_off_source'] == 'GRF')).to_numpy().copy()
if kmx_has_belts:
    kmx_belt = g['heel_strike_belt'].to_numpy()
    kmx_trusted &= np.isin(kmx_belt, limbs) & (g['heel_strike_belt'] == g['toe_off_belt']).to_numpy()
else:
    kmx_belt = g['support_limb'].to_numpy()
    print("  ! no event_qa.csv: every force-plate stance is read from the foot's own")
    print("    belt, so a crossover or a step on the gap cannot be excluded")
gait_event_data['kinetics_trusted'] = kmx_trusted

kmx_cols = ['braking_impulse', 'propulsive_impulse', 'vertical_impulse', 'grf_peak1',
            'grf_trough', 'grf_peak2', 'loading_rate', 'peak_braking', 'peak_propulsive']
for c_ in kmx_cols:
    gait_event_data[c_ + '_bw'] = np.nan
    gait_event_data[c_ + '_tw'] = np.nan
for c_ in ('cop_ap_range_mm', 'cop_ml_range_mm', 'free_moment_peak_nm'):
    gait_event_data[c_] = np.nan

kmx_ensemble = {'L': [], 'R': []}
kmx_ensemble_ap = {'L': [], 'R': []}
kmx_cop_paths = {'L': [], 'R': []}
kmx_first, kmx_second = [], []
kmx_past_toe, kmx_past_heel = [], []
dt = 1.0 / force_fs
for x in np.flatnonzero(kmx_trusted & steady):
    belt = kmx_belt[x]
    a = int(round(g['heel_strike_grf_sample'][x]))
    b_ = int(round(g['toe_off_grf_sample'][x]))
    if not (np.isfinite(g['heel_strike_grf_sample'][x]) and np.isfinite(g['toe_off_grf_sample'][x])):
        continue
    if b_ - a < 0.2 * force_fs or b_ > n_force:
        continue
    F_raw = grf_raw[belt][a:b_]
    F = grf[belt][a:b_]
    ap_raw, ap = F_raw @ forward3, F @ forward3
    vt = F[:, 2]
    kmx_first.append(ap[:len(ap) // 2].mean())
    kmx_second.append(ap[len(ap) // 2:].mean())

    # impulses: the raw force (integration needs no filter), split by sign
    vals = dict(braking_impulse=np.sum(np.clip(ap_raw, None, 0)) * dt,
                propulsive_impulse=np.sum(np.clip(ap_raw, 0, None)) * dt,
                vertical_impulse=np.sum(F_raw[:, 2]) * dt,
                peak_braking=-ap.min(), peak_propulsive=ap.max())
    # vertical landmarks: F1 before mid-stance, F2 after, the trough between
    mid = len(vt) // 2
    i1 = int(np.argmax(vt[:mid]))
    i2 = mid + int(np.argmax(vt[mid:]))
    vals.update(grf_peak1=vt[i1], grf_peak2=vt[i2],
                grf_trough=vt[i1:i2 + 1].min() if i2 > i1 else np.nan)
    # loading rate: average slope between 20% and 80% of F1
    lo = int(np.argmax(vt[:i1 + 1] >= kmx_loading_band[0] * vt[i1]))
    hi = int(np.argmax(vt[:i1 + 1] >= kmx_loading_band[1] * vt[i1]))
    vals['loading_rate'] = (vt[hi] - vt[lo]) / ((hi - lo) * dt) if hi > lo else np.nan
    for k_, v_ in vals.items():
        gait_event_data.loc[x, k_ + '_bw'] = v_ / body_weight
        gait_event_data.loc[x, k_ + '_tw'] = v_ / system_weight

    # CoP along and across the direction of travel, and the free moment.
    # REVISED: the CoP is a point on the PLATE, and the foot on top of it rides
    # the belt backwards ~0.9 m per stance, so in the lab the CoP "travels"
    # mostly with the belt. Adding the belt's travel back gives its path over
    # the belt -- along the foot, as over ground (150-220 mm). Version 3
    # reported the lab-frame range.
    cp = cop_theia[belt][a:b_]
    cp_ok = np.flatnonzero(np.isfinite(cp).all(axis=1))
    cp = cp[cp_ok] + belt_speed * (cp_ok / force_fs)[:, None] * forward
    if len(cp) > 30:
        cp = filtfilt(b_cop, a_cop, cp, axis=0)
        gait_event_data.loc[x, 'cop_ap_range_mm'] = 1000 * np.ptp(cp @ forward)
        gait_event_data.loc[x, 'cop_ml_range_mm'] = 1000 * np.ptp(cp @ lateral)
        if len(kmx_cop_paths[belt]) < 40:
            kmx_cop_paths[belt].append(1000 * np.column_stack([(cp - cp[0]) @ lateral, (cp - cp[0]) @ forward]))
        # REVISED (after D05), a check: the CoP must lie under the boot. How far,
        # at its most, does it pass the posed boot's toe and its heel?
        limb_ = g['support_limb'][x]
        rows_ = (a + cp_ok) / force_step
        along_ = filtfilt(b_cop, a_cop, cop_theia[belt][a:b_][cp_ok], axis=0) @ forward
        kmx_past_toe.append(1000 * np.nanmax(along_ - at(boot_front[limb_], rows_)))
        kmx_past_heel.append(1000 * np.nanmax(at(boot_rear[limb_], rows_) - along_))
    gait_event_data.loc[x, 'free_moment_peak_nm'] = np.nanmax(np.abs(free_moment[belt][a:b_]))

    kmx_ensemble[g['support_limb'][x]].append(normalize(vt / system_weight, kmx_norm_points))
    kmx_ensemble_ap[g['support_limb'][x]].append(normalize(ap / system_weight, kmx_norm_points))

kmx_steady = gait_event_data.loc[steady]
print(f"  {int(kmx_steady['kinetics_trusted'].sum())} of {len(kmx_steady)} steady stances trusted "
      f"for kinetics ({100 * kmx_steady['kinetics_trusted'].mean():.0f}%)")
print(kmx_steady.groupby('support_limb')[['grf_peak1_bw', 'grf_trough_bw', 'grf_peak2_bw',
                                          'grf_peak1_tw', 'loading_rate_bw',
                                          'propulsive_impulse_bw', 'braking_impulse_bw']].mean().round(3))

#brain checks
# 1. braking comes first in stance and propulsion second: if not, the direction
#    of travel or the AP axis is reversed
print(f"  AP force: first half of stance {np.mean(kmx_first):+.1f} N, second half "
      f"{np.mean(kmx_second):+.1f} N (braking then propulsion: - then +)")
if np.mean(kmx_first) > 0 or np.mean(kmx_second) < 0:
    print("  ! the AP pattern is reversed: check the direction of travel and the registration")
# 2. at steady speed propulsion cancels braking
kmx_net = (kmx_steady['propulsive_impulse_bw'] + kmx_steady['braking_impulse_bw']).mean()
kmx_prop = kmx_steady['propulsive_impulse_bw'].mean()
print(f"  net AP impulse {kmx_net:+.5f} BW*s ({100 * abs(kmx_net / kmx_prop):.1f}% of the propulsive impulse)"
      + (f"; the forces were turned by {force_tilt_ap_deg:+.2f} deg along travel (Forces cell)"
         if force_align_gravity and 0.01 < force_tilt_deg <= force_max_tilt_deg else ""))
if abs(kmx_net) > 0.1 * abs(kmx_prop):
    print("  ! that should be near zero at steady speed: check the AP baseline and the tilt")
# 3. the CoP stays under the boot, and its excursion fits the boot
if kmx_past_toe:
    kmx_toe, kmx_heel = float(np.nanmedian(kmx_past_toe)), float(np.nanmedian(kmx_past_heel))
    kmx_boot = 1000 * np.mean([boot_length[b] for b in limbs])
    print(f"  CoP vs the posed boot: at its most it passes the toe by {kmx_toe:+.0f} mm and the heel by "
          f"{kmx_heel:+.0f} mm (median of the stances; <= 0 = under the boot)")
    print(f"  CoP excursion along travel {kmx_steady['cop_ap_range_mm'].mean():.0f} mm = "
          f"{100 * kmx_steady['cop_ap_range_mm'].mean() / kmx_boot:.0f}% of the boot's sole ({kmx_boot:.0f} mm)")
    if kmx_toe > 15 and kmx_heel > 15:
        print("  ! past BOTH ends: the plate's CoP is probably computed at the force sensors,")
        print("    not at the belt surface. The error is height x F_AP / F_z: backwards while")
        print("    braking, forwards at push-off, so the excursion comes out too long. The")
        print("    other kinetics (forces, impulses) are not affected")
    elif max(kmx_toe, kmx_heel) > 15:
        print("  ! past one end only: an offset between the plates and Theia along travel")
# 4. each foot carries about half the weight over a stride
kmx_share = (kmx_steady['vertical_impulse_tw'] / kmx_steady['stride_time']).mean()
print(f"  vertical impulse per stance = {100 * kmx_share:.1f}% of total weight x stride time (expect ~50%)")
if abs(kmx_share - 0.5) > 0.03:
    print("  ! off by more than 3 points: the system mass or the vertical baseline is off")

if make_figures and kmx_ensemble['L']:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    pct = np.linspace(0, 100, kmx_norm_points)
    for b, colour in (('L', '#2a5d9f'), ('R', '#a8442a')):
        if not kmx_ensemble[b]:
            continue
        E = np.vstack(kmx_ensemble[b])
        axes[0].fill_between(pct, np.percentile(E, 5, axis=0), np.percentile(E, 95, axis=0),
                             color=colour, alpha=0.20, lw=0)
        axes[0].plot(pct, E.mean(axis=0), color=colour, lw=2, label=f'{side_name[b]} (n={len(E)})')
        axes[0].plot(pct, np.vstack(kmx_ensemble_ap[b]).mean(axis=0), color=colour, lw=1.2, ls='--')
    axes[0].axhline(1.0, color='k', lw=0.8, ls=':')
    axes[0].axhline(0.0, color='k', lw=0.6)
    axes[0].set_xlabel('% of stance')
    axes[0].set_ylabel('GRF (total weight)')
    axes[0].set_title('Vertical (solid) and AP (dashed) GRF, trusted stances', fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(alpha=0.15)

    axes[1].scatter(kmx_steady['braking_impulse_bw'], kmx_steady['propulsive_impulse_bw'],
                    c=np.where(kmx_steady['support_limb'] == 'L', 0, 1), cmap='coolwarm', s=10, alpha=0.6)
    lim = np.nanmax(np.abs(kmx_steady['braking_impulse_bw']))
    axes[1].plot([-lim, 0], [lim, 0], 'k--', lw=1, label='perfect cancellation')
    axes[1].set_xlabel('braking impulse (BW*s)')
    axes[1].set_ylabel('propulsive impulse (BW*s)')
    axes[1].set_title('Braking vs propulsion, each stance', fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=0.15)

    for b, colour in (('L', '#2a5d9f'), ('R', '#a8442a')):
        for p_ in kmx_cop_paths[b]:
            axes[2].plot(p_[:, 0], p_[:, 1], color=colour, lw=0.6, alpha=0.35)
    axes[2].set_xlabel('CoP across travel (mm, + left)')
    axes[2].set_ylabel('CoP along travel (mm)')
    axes[2].set_title('CoP paths, aligned to heel strike', fontsize=10)
    axes[2].grid(alpha=0.15)
    plt.tight_layout()


# =============================================================================
# %% Stride series
# =============================================================================
# Everything from here on is a STRIDE-TO-STRIDE metric: alpha, entropy, the
# harmonic ratio, the goal-equivalent manifold. They all read a series indexed
# by stride, and every one of them is N-DEPENDENT. So the series are cut ONCE,
# here, and every later cell reads this table.
#
# WHY A FIXED N MATTERS MORE THAN A LARGE N
# alpha, sample entropy and lambda all drift with series length. Truncate every
# trial in the dataset to a COMMON length -- the shortest one -- and state it.
#   at 108 bpm the cadence is 54 strides/min, so
#     15 min trial -> 810 strides, 756 after a 60 s warm-up
#     10 min trial -> 540 strides, 486 after a 60 s warm-up
#
# THE METRONOME
# It paces cadence, so it sets stride TIMING directly: alpha of stride time
# measures how tightly the walker follows the beat (typically < 0.5; Terrier &
# Deriaz 2012), not the free-walking persistence of ~0.75-0.9. The series the
# metronome does not constrain (step width, stride length, clearance,
# margins, impulses) are the better primary outcomes.
#
# References
#   Damouras et al. (2010) Gait Posture 31(3), 336-340.
#   Dingwell, John & Cusumano (2010) PLoS Comput Biol 6(7), e1000856.
#
# REVISED: the warm-up is the one used everywhere (counted from the first
# event, not from frame 0), and the timing series paced by the metronome now
# include step time and the support percentages.
print("\n" + "-" * 74)
print("  STRIDE SERIES")
print("-" * 74)

ss_all = gait_event_data.loc[gait_event_data['support_limb'] == ss_limb].sort_values(
    'initial_contact_kinematic_frame_100hz')
stride_series = ss_all.loc[ss_all['steady']].reset_index(drop=True)
print(f"  limb {ss_limb}, {warmup_seconds} s warm-up excluded")
print(f"  {len(stride_series)} strides available ({len(ss_all)} before the warm-up cut)")

ss_available = len(stride_series)
if ss_fixed_n is None:
    print(f"  using all {ss_available}. SET ss_fixed_n before comparing trials:")
    print("    alpha, entropy and lambda all drift with N")
elif ss_available < ss_fixed_n:
    print(f"  ! only {ss_available} strides, fewer than the {ss_fixed_n} this dataset is")
    print("    standardised to: NOT comparable on any N-dependent metric")
else:
    stride_series = stride_series.iloc[:ss_fixed_n].copy()
    print(f"  truncated to the dataset-wide {ss_fixed_n} strides ({ss_available} were available)")

# A series with gaps is not a series. DFA and entropy both accept a NaN-riddled
# array and hand back a number, so completeness is checked here, once
ss_candidates = ['stride_time', 'stance_time', 'swing_time', 'step_time',
                 'double_support_percentage', 'step_length', 'step_width',
                 'stride_length', 'stride_velocity', 'minimum_foot_clearance',
                 'mos_ml_contact', 'mos_ml_min', 'mos_ap_contact', 'trip_risk_integral',
                 'propulsive_impulse_bw', 'braking_impulse_bw', 'grf_peak1_bw',
                 'grf_peak2_bw', 'loading_rate_bw', 'cop_ml_range_mm', 'free_moment_peak_nm']
ss_complete = {}
stride_series_usable = []
print("\n  series                       n valid   complete       mean        CV")
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
    ss_flat = np.nanstd(ss_v) <= 1e-10 * max(abs(ss_mean), 1.0)
    print(f"    {ss_c:27s} {ss_ok.sum():5d}    {ss_pct:5.1f}%   {ss_mean:10.4f}  "
          f"{ss_cv:6.2f}%{'   constant, dropped' if ss_flat else ''}")
    if ss_pct > 95 and not ss_flat:
        stride_series_usable.append(ss_c)
print(f"\n  {len(stride_series_usable)} series are over 95% complete and carry forward")

# REVISED (after D05): stride length and stride speed are paced too. With the
# belt speed fixed, L = v T + heel advance, so a metronome that paces T paces L
# (D05: alpha 0.31 for stride length, against 0.6-0.9 uncued; Dingwell 2010)
ss_cued = {'stride_time', 'stance_time', 'swing_time', 'step_time', 'cadence',
           'stance_percentage', 'swing_percentage', 'double_support_percentage',
           'single_support_percentage', 'stride_length', 'stride_velocity'}
ss_uncued = [c_ for c_ in stride_series_usable if c_ not in ss_cued]
print(f"  with a metronome, prefer the {len(ss_uncued)} it does not pace:")
print(f"    {', '.join(ss_uncued[:6])}{' ...' if len(ss_uncued) > 6 else ''}")


# =============================================================================
# %% Long-range correlations
# =============================================================================
# Detrended fluctuation analysis (Peng et al. 1994) asks whether a stride is
# statistically related to strides HUNDREDS of strides earlier: not how MUCH
# gait varies -- how the variation is organised in time.
#
#   integrate   Y(k) = sum of (x_i - mean) up to i = k
#   segment     non-overlapping boxes of length n, forwards AND backwards
#   detrend     least squares straight line within each box (DFA-1)
#   fluctuate   F(n) = RMS of the residuals over all boxes
#   scale       F(n) ~ n^alpha, and alpha is the slope in log-log space
#
#   alpha < 0.5   anti-persistent (corrected stride to stride: a metronome,
#                 or a belt speed the walker must match)
#   alpha = 0.5   uncorrelated
#   0.5 < a < 1   persistent: healthy free-walking stride time ~0.75-0.90
#   alpha > 1.0   non-stationary
#
# THE BOX RANGE: 16 <= n <= N/9 (Damouras et al. 2010). The widely used 4 to
# N/4 inflates alpha.
#
# REVISED: a 95% CONFIDENCE INTERVAL for alpha. Version 3 reported the standard
# error of the log-log regression, which is not an uncertainty for alpha
# (neighbouring box sizes share data, so the residuals are not independent,
# and it comes out far too small). A block bootstrap would cut exactly the
# correlations alpha measures, so the interval is a PARAMETRIC bootstrap:
# series with the estimated alpha and the same N are simulated as exact
# fractional Gaussian noise (Davies & Harte 1987), DFA is rerun on each, and
# the basic interval [2a - q97.5, 2a - q2.5] corrects DFA's small bias at
# this N. At N ~ 500 it is about 0.3 wide: a difference between conditions
# smaller than ~0.15 is within one trial's uncertainty.
#
# References
#   Peng et al. (1994) Phys Rev E 49, 1685-1689.
#   Hausdorff et al. (1996) J Appl Physiol 80(5), 1448-1457.
#   Damouras et al. (2010) Gait Posture 31(3), 336-340.
#   Geweke & Porter-Hudak (1983) J Time Ser Anal 4(4), 221-238.
#   Davies & Harte (1987) Biometrika 74(1), 95-101.
dfa_order = 1
dfa_gph_power = 0.5
dfa_rng = np.random.default_rng(0)

def dfa_alpha(series, min_box=dfa_min_box, max_box=None):
    """DFA scaling exponent, with the fluctuation curve it was fitted to."""
    x_ = np.asarray(series, float)
    x_ = x_[np.isfinite(x_)]
    n_total = len(x_)
    max_box = int(dfa_max_box_frac * n_total) if max_box is None else int(max_box)
    empty = dict(alpha=np.nan, r2=np.nan, n=n_total, boxes=np.array([]),
                 fluct=np.array([]), profile=np.array([]))
    if max_box <= min_box or n_total < 4 * min_box:
        return empty
    if np.std(x_) <= 1e-10 * max(abs(x_.mean()), 1.0):
        return empty
    profile = np.cumsum(x_ - x_.mean())
    boxes = np.unique(np.round(np.logspace(np.log10(min_box), np.log10(max_box),
                                           dfa_n_boxes)).astype(int))
    fluct = np.full(len(boxes), np.nan)
    for i, n in enumerate(boxes):
        count = n_total // n
        if count < 2:
            continue
        seg = np.vstack([profile[:count * n].reshape(count, n),
                         profile[n_total - count * n:].reshape(count, n)])
        t_ = np.arange(n)
        trend = np.polyval(np.polyfit(t_, seg.T, dfa_order), t_[:, None]).T
        fluct[i] = np.sqrt(np.mean((seg - trend) ** 2))
    ok = np.isfinite(fluct) & (fluct > 0)
    if ok.sum() < 4:
        return empty
    lx, ly = np.log10(boxes[ok]), np.log10(fluct[ok])
    slope, icept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + icept)
    r2 = float(1 - np.sum(resid ** 2) / np.sum((ly - ly.mean()) ** 2))
    return dict(alpha=float(slope), r2=r2, n=n_total, boxes=boxes[ok],
                fluct=fluct[ok], profile=profile)

def fractional_gaussian_noise(n, hurst, rng):
    """Exact fGn (Davies & Harte 1987): the series whose DFA alpha is hurst."""
    k = np.arange(0, n + 1)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * hurst) - 2 * np.abs(k) ** (2 * hurst)
                   + np.abs(k + 1) ** (2 * hurst))
    eig = np.maximum(np.fft.fft(np.concatenate([gamma, gamma[-2:0:-1]])).real, 0.0)
    m_ = len(eig)
    z = rng.normal(size=m_) + 1j * rng.normal(size=m_)
    return np.fft.fft(np.sqrt(eig / (2 * m_)) * z).real[:n]

def dfa_interval(alpha_hat, n, rng):
    """Parametric bootstrap 95% interval for alpha (basic interval)."""
    if not np.isfinite(alpha_hat):
        return np.nan, np.nan
    reps = []
    for _ in range(dfa_n_boot):
        if alpha_hat < 1:
            sim = fractional_gaussian_noise(n, float(np.clip(alpha_hat, 0.02, 0.98)), rng)
        else:
            sim = np.cumsum(fractional_gaussian_noise(n, float(np.clip(alpha_hat - 1, 0.02, 0.98)), rng))
        reps.append(dfa_alpha(sim)['alpha'])
    q_lo, q_hi = np.nanquantile(reps, [0.025, 0.975])
    return float(2 * alpha_hat - q_hi), float(2 * alpha_hat - q_lo)

print("\n" + "-" * 74)
print("  LONG-RANGE CORRELATIONS (DFA)")
print("-" * 74)
dfa_n_used = len(stride_series)
print(f"  N = {dfa_n_used} strides, boxes {dfa_min_box} to {int(dfa_max_box_frac * dfa_n_used)} "
      f"(Damouras: 16 to N/9)")
if dfa_n_used < 288:
    print("  ! fewer than 288 strides: the box range spans less than an octave")

dfa_rows = []
dfa_curves = {}
for dfa_name in stride_series_usable:
    dfa_x = pd.to_numeric(stride_series[dfa_name], errors='coerce').to_numpy(float)
    dfa_res = dfa_alpha(dfa_x)
    if not np.isfinite(dfa_res['alpha']):
        continue
    dfa_curves[dfa_name] = dfa_res
    dfa_clean = dfa_x[np.isfinite(dfa_x)]
    # surrogate: same values, order destroyed, so alpha must fall to 0.5
    dfa_sur = np.array([dfa_alpha(dfa_rng.permutation(dfa_clean))['alpha']
                        for _ in range(dfa_n_surrogate)], float)
    # GPH: alpha = d + 0.5 from the low-frequency periodogram slope
    dfa_m = int(len(dfa_clean) ** dfa_gph_power)
    dfa_per = np.abs(np.fft.rfft(dfa_clean - dfa_clean.mean())) ** 2 / (2 * np.pi * len(dfa_clean))
    dfa_w = 2 * np.pi * np.arange(len(dfa_per)) / len(dfa_clean)
    dfa_j = np.arange(1, min(dfa_m, len(dfa_per) - 1) + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dfa_reg = np.log(4 * np.sin(dfa_w[dfa_j] / 2) ** 2)
        dfa_resp = np.log(dfa_per[dfa_j])
    dfa_fin = np.isfinite(dfa_reg) & np.isfinite(dfa_resp)
    dfa_gph = (0.5 - np.polyfit(dfa_reg[dfa_fin], dfa_resp[dfa_fin], 1)[0]
               if dfa_fin.sum() >= 4 else np.nan)
    dfa_lo, dfa_hi = dfa_interval(dfa_res['alpha'], len(dfa_clean), dfa_rng)
    dfa_rows.append({'series': dfa_name, 'n': dfa_res['n'], 'alpha': dfa_res['alpha'],
                     'alpha_ci_lo': dfa_lo, 'alpha_ci_hi': dfa_hi, 'r2': dfa_res['r2'],
                     'alpha_gph': dfa_gph, 'surrogate_mean': float(np.nanmean(dfa_sur)),
                     'surrogate_sd': float(np.nanstd(dfa_sur)),
                     'cued_by_metronome': dfa_name in ss_cued})
dfa_results = pd.DataFrame(dfa_rows)

if len(dfa_results):
    print("\n  series                      alpha   95% CI          R2    GPH   surrogate   cued")
    for _, r in dfa_results.iterrows():
        print(f"    {r['series']:26s} {r['alpha']:5.3f}  [{r['alpha_ci_lo']:.2f}, {r['alpha_ci_hi']:.2f}]  "
              f"{r['r2']:.3f}  {r['alpha_gph']:5.3f}  {r['surrogate_mean']:.3f}"
              f"{'   yes' if r['cued_by_metronome'] else '    no'}")
    dfa_bad = dfa_results[np.abs(dfa_results['surrogate_mean'] - 0.5) > 0.05]
    if len(dfa_bad):
        print(f"\n  ! {len(dfa_bad)} series have a shuffled surrogate away from 0.5: "
              f"{', '.join(dfa_bad['series'])}")
    else:
        print("\n  surrogate check passed: every shuffled series returned alpha ~ 0.5")
    dfa_gap = np.abs(dfa_results['alpha'] - dfa_results['alpha_gph'])
    print(f"  DFA vs GPH: median gap {np.nanmedian(dfa_gap):.3f} (GPH scatter alone is ~0.15 at this N)")
else:
    print("  ! no series produced an alpha")

if make_figures and dfa_curves:
    dfa_show = next((c_ for c_ in dfa_curves if c_ not in ss_cued), next(iter(dfa_curves)))
    dfa_res = dfa_curves[dfa_show]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].plot(dfa_res['profile'], color='#333', lw=0.9)
    for dfa_n, dfa_col in ((dfa_res['boxes'][0], '#2a5d9f'), (dfa_res['boxes'][-1], '#cc6633')):
        for dfa_b in range(len(dfa_res['profile']) // dfa_n):
            seg = dfa_res['profile'][dfa_b * dfa_n:(dfa_b + 1) * dfa_n]
            t_ = np.arange(dfa_n)
            axes[0].plot(dfa_b * dfa_n + t_, np.polyval(np.polyfit(t_, seg, 1), t_), color=dfa_col, lw=1.2)
    axes[0].set_xlabel('stride number')
    axes[0].set_title(f'{dfa_show}: integrated profile and local fits', fontsize=10)
    axes[1].loglog(dfa_res['boxes'], dfa_res['fluct'], 'o', color='#2a5d9f')
    axes[1].loglog(dfa_res['boxes'], 10 ** np.polyval(np.polyfit(np.log10(dfa_res['boxes']),
                                                                 np.log10(dfa_res['fluct']), 1),
                                                      np.log10(dfa_res['boxes'])),
                   color='#cc6633', label=f"alpha = {dfa_res['alpha']:.3f}")
    axes[1].set_xlabel('box size n (strides)')
    axes[1].set_ylabel('F(n)')
    axes[1].legend(frameon=False)
    dfa_ord = dfa_results.sort_values('alpha')
    yy = np.arange(len(dfa_ord))
    axes[2].errorbar(dfa_ord['alpha'], yy,
                     xerr=[dfa_ord['alpha'] - dfa_ord['alpha_ci_lo'], dfa_ord['alpha_ci_hi'] - dfa_ord['alpha']],
                     fmt='o', color='#2a5d9f', ms=4)
    axes[2].plot(dfa_ord['surrogate_mean'], yy, 'kx', ms=5, label='shuffled')
    axes[2].axvline(0.5, color='#888', ls=':')
    axes[2].set_yticks(yy)
    axes[2].set_yticklabels([f"{s}{' (paced)' if c_ else ''}" for s, c_ in
                             zip(dfa_ord['series'], dfa_ord['cued_by_metronome'])], fontsize=7)
    axes[2].set_xlabel('alpha with its 95% CI')
    axes[2].legend(frameon=False, fontsize=7)
    plt.tight_layout()


# =============================================================================
# %% Trunk acceleration signal
# =============================================================================
# Three metrics stand on a trunk acceleration: the multiscale entropy curve, the
# harmonic ratio and the autocorrelation regularity. The literature standard is
# an accelerometer at the lower back, which Theia does not give, so it is
# derived ONCE, here, so all three read the same signal.
#
# WHICH COLUMN: a LINEAR one. Trunk_Joint_Acceleration and the *_Joint_Acc
# family are ANGULAR (derivatives of joint angles, deg/s^2).
#
# HOW: low-pass 20 Hz, then a short Savitzky-Golay derivative (window 5,
# accurate to 18 Hz); the harmonic ratio differentiates exactly in the
# frequency domain instead (harmonic k of the acceleration = k w0 x harmonic k
# of the velocity).
#
# REVISED: rotated into the walker's axes (ML, AP, VT) along the measured
# direction of travel. Version 3 rotated by 0 or 180 deg (the sign of Y only).
trunk_lowpass_hz = 20.0
trunk_deriv_window = 5
trunk_deriv_poly = 3
trunk_sources = [('Trunk_Linear_Acceleration', 0), ('Low_Back_Linear_Acceleration', 0),
                 ('Trunk_Linear_Velocity', 1), ('Low_Back_Linear_Velocity', 1),
                 ('Low_Back_Position', 2), ('Trunk_Position', 2), ('Pelvis_Position', 2)]
print("\n" + "-" * 74)
print("  TRUNK ACCELERATION")
print("-" * 74)
trunk_acc, trunk_raw, trunk_name, trunk_spec_signal, trunk_spec_power = None, None, None, None, 0
for trunk_col, trunk_order in trunk_sources:
    trunk_raw = xyz(trunk_col)
    if trunk_raw is not None:
        trunk_name, trunk_deriv_order = trunk_col, trunk_order
        break
if trunk_raw is None:
    print("  ! no linear trunk column found: entropy curve, harmonic ratio and regularity skipped")
else:
    print(f"  source: {trunk_name}, differentiated {trunk_deriv_order} time(s)")
    # into the walker's axes first: ML (to the left), AP (along travel), VT
    trunk_raw = np.column_stack([trunk_raw[:, :2] @ lateral, trunk_raw[:, :2] @ forward, trunk_raw[:, 2]])
    trunk_fill, trunk_gap = fill_gaps(trunk_raw)
    trunk_reach = trunk_deriv_order * (trunk_deriv_window // 2)
    trunk_gap_wide = np.convolve(trunk_gap, np.ones(2 * trunk_reach + 1), 'same') > 0
    trunk_b, trunk_a = butter(4, trunk_lowpass_hz / (kinematic_fs / 2), 'low')
    trunk_acc = filtfilt(trunk_b, trunk_a, trunk_fill, axis=0)
    for _ in range(trunk_deriv_order):
        trunk_acc = savgol_filter(trunk_acc, trunk_deriv_window, trunk_deriv_poly,
                                  deriv=1, delta=1.0 / kinematic_fs, axis=0)
    trunk_acc[trunk_gap_wide] = np.nan
    trunk_spec_signal = filtfilt(trunk_b, trunk_a, trunk_fill, axis=0)
    trunk_spec_signal[trunk_gap] = np.nan
    trunk_spec_power = trunk_deriv_order
    trunk_rms = np.sqrt(np.nanmean(trunk_acc[steady_samples.start // force_step:steady_samples.stop // force_step] ** 2, axis=0))
    print(f"  RMS (steady walking)  ML {trunk_rms[0]:.2f}   AP {trunk_rms[1]:.2f}   VT {trunk_rms[2]:.2f}  m/s^2")
    if np.nanmax(trunk_rms) > 20 or np.nanmax(trunk_rms) < 0.2:
        print("  ! that is not a walking trunk acceleration (expect 1-4 m/s^2 RMS per axis)")


# =============================================================================
# %% Entropy
# =============================================================================
# Sample entropy (Richman & Moorman 2000): of the templates of m consecutive
# points that match within r, how many still match when extended to m+1?
#     SampEn(m, r, N) = -ln( A / B )
# Low = regular, high = unpredictable. White noise is the MOST unpredictable
# and the least complex, so the multiscale curve -- coarse-grain and recompute
# -- is the complexity measure, not a single value (Costa et al. 2002). The
# refined composite variant (Wu et al. 2014) sums the match counts over all
# coarse-graining phases before the log.
#
# NO TRANSFERABLE NORMATIVE VALUES: SampEn depends on m, r, N and the
# preprocessing (Yentes & Raffalt 2021): compare conditions analysed
# identically, report m, r and N, and sweep r.
#
# REVISED: the multiscale curve goes to scale 20 (0.20 s) instead of 10.
def ent_match_counts(series, m, r):
    """Matching template pairs at lengths m+1 (A) and m (B), by KD-tree."""
    x_ = np.asarray(series, float)
    x_ = x_[np.isfinite(x_)]
    n = len(x_)
    if n < m + 2:
        return 0, 0
    k = n - m
    starts = np.arange(k)
    tree_m = cKDTree(x_[starts[:, None] + np.arange(m)[None, :]])
    tree_m1 = cKDTree(x_[starts[:, None] + np.arange(m + 1)[None, :]])
    b_ = int((tree_m.count_neighbors(tree_m, r, p=np.inf) - k) // 2)
    a_ = int((tree_m1.count_neighbors(tree_m1, r, p=np.inf) - k) // 2)
    return a_, b_

def ent_sampen(series, m=ent_m, r=ent_r):
    """Sample entropy of the z-scored series (r in SD units); NaN, not 0, if
    nothing matches."""
    x_ = np.asarray(series, float)
    x_ = x_[np.isfinite(x_)]
    if len(x_) < m + 2 or np.std(x_) <= 1e-10 * max(abs(x_.mean()), 1.0):
        return np.nan
    a_, b_ = ent_match_counts((x_ - x_.mean()) / np.std(x_), m, r)
    return float(-np.log(a_ / b_)) if a_ > 0 and b_ > 0 else np.nan

print("\n" + "-" * 74)
print("  ENTROPY")
print("-" * 74)
print(f"  m = {ent_m}, r = {ent_r} x SD, N = {len(stride_series)}, z-scored")
if len(stride_series) < 200:
    print("  ! N under 200: both algorithms become very parameter-sensitive (Yentes 2013)")
ent_ceiling = ent_sampen(np.random.default_rng(0).normal(size=len(stride_series)))
ent_rows = []
for ent_name in stride_series_usable:
    ent_x = pd.to_numeric(stride_series[ent_name], errors='coerce').to_numpy(float)
    ent_row = {'series': ent_name, 'n': int(np.isfinite(ent_x).sum()), 'm': ent_m, 'r': ent_r,
               'sampen': ent_sampen(ent_x)}
    for ent_rv in ent_r_sweep:
        ent_row[f'sampen_r{ent_rv:.2f}'] = ent_sampen(ent_x, r=ent_rv)
    ent_rows.append(ent_row)
entropy_results = pd.DataFrame(ent_rows)
if len(entropy_results):
    print(f"  white noise at this N reaches {ent_ceiling:.3f}")
    print("    series                     SampEn   " + "".join(f"r={v:.2f} " for v in ent_r_sweep))
    for _, ent_row in entropy_results.iterrows():
        print(f"    {ent_row['series']:26s} {ent_row['sampen']:6.3f}   "
              + "".join(f"{ent_row[f'sampen_r{v:.2f}']:5.3f}  " for v in ent_r_sweep))

multiscale_entropy = None
if trunk_acc is not None:
    ent_seg = trunk_acc[steady_samples.start // force_step:][:int(ent_mse_max_s * kinematic_fs)]
    ent_inputs = {'ML': ent_seg[:, 0], 'AP': ent_seg[:, 1], 'VT': ent_seg[:, 2],
                  'white noise': np.random.default_rng(1).normal(size=len(ent_seg))}
    ent_curves = {}
    for ent_label, ent_x in ent_inputs.items():
        ent_x = ent_x[np.isfinite(ent_x)]
        # r is fixed on the ORIGINAL series: re-normalising per scale would
        # remove the change in variance the curve is meant to show
        ent_x = (ent_x - ent_x.mean()) / np.std(ent_x)
        ent_out = np.full(ent_mse_scales, np.nan)
        for ent_s in range(1, ent_mse_scales + 1):
            ent_a_sum = ent_b_sum = 0
            for ent_p in range(ent_s):
                ent_cut = ent_x[ent_p:]
                ent_k = len(ent_cut) // ent_s
                if ent_k < ent_m + 2:
                    continue
                ent_a, ent_b = ent_match_counts(ent_cut[:ent_k * ent_s].reshape(ent_k, ent_s).mean(axis=1),
                                                ent_m, ent_r)
                ent_a_sum += ent_a
                ent_b_sum += ent_b
            if ent_a_sum > 0 and ent_b_sum > 0:
                ent_out[ent_s - 1] = float(-np.log(ent_a_sum / ent_b_sum))
        ent_curves[ent_label] = ent_out
    multiscale_entropy = pd.DataFrame(ent_curves, index=np.arange(1, ent_mse_scales + 1))
    ent_area = multiscale_entropy.sum()
    print(f"  refined composite MSE, {trunk_name}, {len(ent_seg) / kinematic_fs:.0f} s: area "
          f"ML {ent_area['ML']:.2f}  AP {ent_area['AP']:.2f}  VT {ent_area['VT']:.2f}  "
          f"(white noise {ent_area['white noise']:.2f})")
    if make_figures:
        plt.figure(figsize=(6, 4))
        for ent_label, ent_colour in (('ML', '#2a5d9f'), ('AP', '#cc6633'), ('VT', '#4a8f4a')):
            plt.plot(multiscale_entropy.index, multiscale_entropy[ent_label], 'o-', ms=3,
                     color=ent_colour, label=ent_label)
        plt.plot(multiscale_entropy.index, multiscale_entropy['white noise'], 's--', ms=3,
                 color='#999', label='white noise')
        plt.xlabel('coarse-graining scale (samples)')
        plt.ylabel('sample entropy')
        plt.title('Refined composite MSE of the trunk acceleration')
        plt.legend(frameon=False)
        plt.tight_layout()


# =============================================================================
# %% Harmonic ratio
# =============================================================================
# One stride is two steps. If both produce the same trunk acceleration, AP and
# VT repeat TWICE per stride (energy on the even harmonics); ML sways once per
# stride (odd harmonics):
#     HR(AP, VT) = sum(even) / sum(odd)        HR(ML) = sum(odd) / sum(even)
# over the first 20 harmonics of stride frequency (Menz et al. 2003). HR is
# unbounded, so its MEDIAN over strides is reported, and the improved
# harmonic ratio (Pasciuto et al. 2017), the intrinsic share of the power,
# bounded 0-100%, beside it. One stride per DFT and no window, so bin k IS
# harmonic k.
#
# REVISED: strides are cut at the right ROWS (frame - first frame; version 3
# used frame - 1, right only if the export starts at frame 1).
hr_min_samples = 40
print("\n" + "-" * 74)
print("  HARMONIC RATIO")
print("-" * 74)
harmonic_ratio = pd.DataFrame()
if trunk_spec_signal is not None:
    hr_axes = [(1, 'AP', False), (2, 'VT', False), (0, 'ML', True)]
    hr_rows = []
    for x in range(len(stride_series)):
        hr_a, hr_b = stride_series['initial_contact_kinematic_frame_100hz'][x], stride_series['next_ipsi_heelstrike'][x]
        if not (np.isfinite(hr_a) and np.isfinite(hr_b)):
            continue
        hr_a, hr_b = int(round(frame_to_row(hr_a))), int(round(frame_to_row(hr_b)))
        if hr_b - hr_a < hr_min_samples or hr_b > n_frames:
            continue
        hr_seg = trunk_spec_signal[hr_a:hr_b]
        if not np.isfinite(hr_seg).all():
            continue
        hr_row = {'stride': x}
        hr_k = np.arange(1, hr_n_harmonics + 1)
        for hr_ax, hr_label, hr_odd_dom in hr_axes:
            hr_amp = np.abs(np.fft.rfft(hr_seg[:, hr_ax] - hr_seg[:, hr_ax].mean()))
            if len(hr_amp) <= hr_n_harmonics:
                hr_row = None
                break
            hr_amp = hr_amp[1:hr_n_harmonics + 1] * hr_k.astype(float) ** trunk_spec_power
            hr_odd, hr_even = hr_amp[hr_k % 2 == 1], hr_amp[hr_k % 2 == 0]
            hr_in, hr_out = (hr_odd, hr_even) if hr_odd_dom else (hr_even, hr_odd)
            hr_row[f'hr_{hr_label}'] = hr_in.sum() / hr_out.sum() if hr_out.sum() > 0 else np.nan
            hr_row[f'ihr_{hr_label}'] = (100.0 * np.sum(hr_in ** 2) / np.sum(hr_amp ** 2)
                                         if np.sum(hr_amp ** 2) > 0 else np.nan)
        if hr_row is not None:
            hr_rows.append(hr_row)
    harmonic_ratio = pd.DataFrame(hr_rows)
    if len(harmonic_ratio):
        print(f"  {len(harmonic_ratio)} strides")
        print("    axis   HR median (IQR)          iHR median (IQR)")
        for _, hr_label, hr_odd_dom in hr_axes:
            hr_v, hr_ih = harmonic_ratio[f'hr_{hr_label}'], harmonic_ratio[f'ihr_{hr_label}']
            print(f"    {hr_label}    {hr_v.median():5.2f} ({hr_v.quantile(0.25):.2f}-{hr_v.quantile(0.75):.2f})"
                  f"   {'odd/even' if hr_odd_dom else 'even/odd'}   {hr_ih.median():5.1f}% "
                  f"({hr_ih.quantile(0.25):.1f}-{hr_ih.quantile(0.75):.1f})")
        print("    (literature: accelerometers at L3/sacrum; Theia's trunk is a model fitted to")
        print("     video and smoother -- compare conditions first, the healthy range second)")
else:
    print("  ! no trunk signal, the harmonic ratio is skipped")


# =============================================================================
# %% Supplementary metrics
# =============================================================================
# GOAL-EQUIVALENT MANIFOLD (Dingwell, John & Cusumano 2010)
# At a fixed belt speed the task goal is L / T = belt speed. Stride time T and
# stride length L are first made DIMENSIONLESS by dividing by their means, so
# the goal line is the diagonal, and each stride's deviation splits into
#     along the line  (T^ + L^)/sqrt2   goal-EQUIVALENT, speed unchanged
#     across it       (L^ - T^)/sqrt2   goal-RELEVANT, a speed error
# SD, lag-1 autocorrelation and DFA alpha of each. Healthy treadmill walkers
# correct the goal-relevant part hard (alpha < 0.5) and leave the
# goal-equivalent part alone (alpha ~ 1).
#
# FOOT PLACEMENT CONTROL (Wang & Srinivasan 2014): regress where the other
# foot lands (across the belt) on the CoM position and velocity at this foot's
# mid-stance, all relative to the stance ankle. R^2 = how much of foot
# placement the body's state explains.
#
# REGULARITY (Moe-Nilssen & Helbostad 2004): the unbiased autocorrelation of
# the trunk acceleration, read at its peaks near one step and one stride.
#
# SYMMETRY ANGLE (Zifchock et al. 2008): bounded and reference-free.
# WALK RATIO: step length / cadence (Sekiya & Nagasaki 1998: ~0.0063).
#
# REVISED:
#  * GEM: T and L are divided by their means first. Version 3 rotated the raw
#    (seconds, metres) plane with the slope v* = 1.17 m/s, so "along" and
#    "across" depended on the units; and L is now the stride length over the
#    belt (two heel-to-heel steps do not make L / T equal the belt speed).
#  * foot placement: the system CoM (body + load), the fused velocity, at the
#    exact mid-stance and landing times, across the measured direction.
#  * regularity at the autocorrelation PEAKS within +-25% of a step and a
#    stride (version 3 read fixed lags: a 6% stride-time drift makes a
#    perfectly regular signal read 0.94).
print("\n" + "-" * 74)
print("  SUPPLEMENTARY METRICS")
print("-" * 74)

def lag1(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)] - np.nanmean(v)
    return float(np.sum(v[:-1] * v[1:]) / np.sum(v ** 2)) if np.sum(v ** 2) > 0 else np.nan

# --- goal-equivalent manifold ----------------------------------------------------
sup_gem_t = pd.to_numeric(stride_series['stride_time'], errors='coerce').to_numpy(float)
sup_gem_l = pd.to_numeric(stride_series['stride_length'], errors='coerce').to_numpy(float)
sup_gem_ok = np.isfinite(sup_gem_t) & np.isfinite(sup_gem_l)
goal_equivalent = None
if sup_gem_ok.sum() >= 4 * dfa_min_box:
    T_hat = sup_gem_t[sup_gem_ok] / sup_gem_t[sup_gem_ok].mean() - 1
    L_hat = sup_gem_l[sup_gem_ok] / sup_gem_l[sup_gem_ok].mean() - 1
    goal_equivalent = {'v_star': float(sup_gem_l[sup_gem_ok].mean() / sup_gem_t[sup_gem_ok].mean()),
                       'n': int(sup_gem_ok.sum()),
                       'parallel': (T_hat + L_hat) / np.sqrt(2),
                       'perpendicular': (L_hat - T_hat) / np.sqrt(2)}
    for k_ in ('parallel', 'perpendicular'):
        goal_equivalent[k_ + '_sd_pct'] = 100 * float(np.std(goal_equivalent[k_], ddof=1))
        goal_equivalent[k_ + '_lag1'] = lag1(goal_equivalent[k_])
        goal_equivalent[k_ + '_dfa'] = dfa_alpha(goal_equivalent[k_])['alpha']
    print(f"  goal-equivalent manifold (v* = {goal_equivalent['v_star']:.3f} m/s, n = {goal_equivalent['n']})")
    for k_, lbl in (('parallel', 'goal-equivalent'), ('perpendicular', 'goal-relevant  ')):
        print(f"    {lbl}  SD {goal_equivalent[k_ + '_sd_pct']:.2f}% of the mean stride, "
              f"lag-1 {goal_equivalent[k_ + '_lag1']:+.3f}, DFA alpha {goal_equivalent[k_ + '_dfa']:.3f}")
    if goal_equivalent['perpendicular_dfa'] < goal_equivalent['parallel_dfa']:
        print("    the speed error is corrected harder than the goal-equivalent deviations:")
        print("    a controller exploiting the redundancy (Dingwell et al. 2010)")
    else:
        print("    ! the speed error is NOT corrected harder; worth a second look")
else:
    print(f"  ! fewer than {4 * dfa_min_box} complete strides: the GEM is skipped")

# --- mediolateral foot placement on CoM state --------------------------------------
sup_stance = ss_limb
sup_other = other_limb[ss_limb]
foot_placement = None
sup_ok = (stride_series['toe_off_kinematic_frame_100hz'].notna()
          & stride_series['next_contra_heelstrike'].notna()).to_numpy()
if sup_ok.sum() >= 20:
    ssd = stride_series[sup_ok]
    sup_mid = frame_to_row(ssd['initial_contact_kinematic_frame_100hz']
                           + sup_midstance_fraction * (ssd['toe_off_kinematic_frame_100hz']
                                                       - ssd['initial_contact_kinematic_frame_100hz']))
    sup_ank = at(ankle[sup_stance], sup_mid)
    sup_z = (at(mos_com, sup_mid) - sup_ank)[:, :2] @ lateral
    sup_v = at(mos_com_velocity, sup_mid)[:, :2] @ lateral
    sup_y = (at(heel[sup_other], frame_to_row(ssd['next_contra_heelstrike'].to_numpy())) - sup_ank)[:, :2] @ lateral
    sup_fit = np.isfinite(sup_z) & np.isfinite(sup_v) & np.isfinite(sup_y)
    sup_X = np.column_stack([np.ones(sup_fit.sum()), sup_z[sup_fit], sup_v[sup_fit]])
    sup_beta = np.linalg.lstsq(sup_X, sup_y[sup_fit], rcond=None)[0]
    sup_pred = sup_X @ sup_beta
    sup_ss_tot = np.sum((sup_y[sup_fit] - sup_y[sup_fit].mean()) ** 2)
    foot_placement = {'r2': float(1 - np.sum((sup_y[sup_fit] - sup_pred) ** 2) / sup_ss_tot) if sup_ss_tot > 0 else np.nan,
                      'gain_position': float(sup_beta[1]), 'gain_velocity': float(sup_beta[2]),
                      'n': int(sup_fit.sum()), 'actual': sup_y[sup_fit], 'predicted': sup_pred}
    print(f"\n  foot placement on the CoM state at mid-stance (n = {foot_placement['n']}): "
          f"R^2 = {foot_placement['r2']:.3f}, gains {foot_placement['gain_position']:+.2f} (position), "
          f"{foot_placement['gain_velocity']:+.3f} s (velocity)")

# --- autocorrelation regularity, at the peaks ----------------------------------------
gait_regularity = None
if trunk_acc is not None and len(stride_series) > 10:
    sup_stride_rows = float(np.nanmedian(stride_series['stride_time'])) * kinematic_fs
    sup_max_lag = int(1.6 * sup_stride_rows)
    sup_rows_, sup_curves = [], {}
    r0 = int(round(frame_to_row(stride_series['initial_contact_kinematic_frame_100hz'].iloc[0])))
    r1 = int(round(frame_to_row(stride_series['next_ipsi_heelstrike'].dropna().iloc[-1])))
    for sup_ax, sup_label in ((0, 'ML'), (1, 'AP'), (2, 'VT')):
        sx = trunk_acc[r0:r1, sup_ax]
        sx = sx[np.isfinite(sx)] - np.nanmean(sx)
        n_ = len(sx)
        ac = np.array([np.sum(sx[:n_ - k] * sx[k:]) / (n_ - k) for k in range(sup_max_lag + 1)])
        ac = ac / ac[0]
        sup_curves[sup_label] = ac

        def peak_near(lag):
            lo_ = int(np.floor(lag * (1 - regularity_search)))
            hi_ = min(int(np.ceil(lag * (1 + regularity_search))), len(ac) - 1)
            i_ = lo_ + int(np.argmax(ac[lo_:hi_ + 1]))
            return float(ac[i_]), i_
        d1, l1 = peak_near(sup_stride_rows / 2)
        d2, l2 = peak_near(sup_stride_rows)
        sup_rows_.append({'axis': sup_label, 'step_regularity': d1, 'stride_regularity': d2,
                          'symmetry': d1 / d2 if d2 else np.nan, 'step_lag': l1, 'stride_lag': l2})
    gait_regularity = pd.DataFrame(sup_rows_)
    print(f"\n  autocorrelation regularity (peaks near {sup_stride_rows / 2:.0f} and {sup_stride_rows:.0f} samples)")
    print(gait_regularity.round(3).to_string(index=False))

# --- symmetry angle, from the per-step table -------------------------------------
sup_sym_rows = []
for sup_var in ('stance_time', 'swing_time', 'step_length', 'minimum_foot_clearance',
                'mos_ml_contact', 'propulsive_impulse_bw', 'grf_peak1_bw', 'trip_risk_integral'):
    sl_ = gait_event_data.loc[steady & (gait_event_data['support_limb'] == 'L'), sup_var].mean()
    sr_ = gait_event_data.loc[steady & (gait_event_data['support_limb'] == 'R'), sup_var].mean()
    if not (np.isfinite(sl_) and np.isfinite(sr_)) or sr_ == 0:
        continue
    sa = (45.0 - np.degrees(np.arctan2(sl_, sr_))) / 90.0 * 100.0
    sup_sym_rows.append({'variable': sup_var, 'left': sl_, 'right': sr_,
                         'symmetry_angle_pct': sa - 200.0 if sa > 100.0 else sa})
symmetry_angles = pd.DataFrame(sup_sym_rows)
if len(symmetry_angles):
    print("\n  symmetry angle (0% symmetric, positive = right larger)")
    for _, r in symmetry_angles.iterrows():
        print(f"    {r['variable']:26s} L {r['left']:+9.4f}  R {r['right']:+9.4f}   SA {r['symmetry_angle_pct']:+6.2f}%")

# --- walk ratio --------------------------------------------------------------------
walk_ratio = float(gait_event_data.loc[steady, 'walk_ratio'].mean())
print(f"\n  walk ratio {walk_ratio:.5f} m per step/min (both speed and cadence are fixed by")
print("  this protocol, so it is largely set by the protocol)")

if make_figures:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    if goal_equivalent is not None:
        axes[0].scatter(100 * (goal_equivalent['parallel'] - goal_equivalent['perpendicular']) / np.sqrt(2),
                        100 * (goal_equivalent['parallel'] + goal_equivalent['perpendicular']) / np.sqrt(2),
                        s=6, alpha=0.4, color='#2a5d9f')
        lim_ = 4
        axes[0].plot([-lim_, lim_], [-lim_, lim_], color='#cc6633', label='goal: L / T = belt speed')
        axes[0].set_aspect('equal')
        axes[0].set_xlabel('stride time, % from mean')
        axes[0].set_ylabel('stride length, % from mean')
        axes[0].set_title(f"GEM: SD along {goal_equivalent['parallel_sd_pct']:.2f}%, "
                          f"across {goal_equivalent['perpendicular_sd_pct']:.2f}%", fontsize=10)
        axes[0].legend(frameon=False, fontsize=8)
    if foot_placement is not None:
        axes[1].scatter(1000 * foot_placement['predicted'], 1000 * foot_placement['actual'], s=8, alpha=0.4)
        lo_, hi_ = 1000 * foot_placement['actual'].min(), 1000 * foot_placement['actual'].max()
        axes[1].plot([lo_, hi_], [lo_, hi_], 'k--', lw=1)
        axes[1].set_xlabel('predicted ML foot placement (mm)')
        axes[1].set_ylabel('actual ML foot placement (mm)')
        axes[1].set_title(f"Foot placement R^2 = {foot_placement['r2']:.2f}", fontsize=10)
    if gait_regularity is not None:
        for sup_label, colour in (('ML', '#cc6633'), ('AP', '#2a5d9f'), ('VT', '#4a8f4a')):
            axes[2].plot(np.arange(len(sup_curves[sup_label])) / kinematic_fs, sup_curves[sup_label],
                         color=colour, label=sup_label)
        for _, r in gait_regularity.iterrows():
            axes[2].plot([r['step_lag'] / kinematic_fs, r['stride_lag'] / kinematic_fs],
                         [r['step_regularity'], r['stride_regularity']], 'ko', ms=4)
        axes[2].set_xlabel('lag (s)')
        axes[2].set_ylabel('unbiased autocorrelation')
        axes[2].set_title('Regularity at the step and stride peaks', fontsize=10)
        axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    plt.tight_layout()


# =============================================================================
# %% Literature check
# =============================================================================
# Every key metric of the trial, over steady walking, beside the values
# healthy adults show in the literature at about this speed. A value outside
# its range is NOT an error by itself: the participant carries a load, wears
# boots, walks on a treadmill and steps to a metronome, and each of those
# shifts some metrics (all said in the "population" column). It IS the place
# to look first when something went wrong upstream: a 3 cm MFC that should be
# 1.5 cm, an MoS that is twice the healthy value, a vertical impulse that is
# not half the weight times the stride.
#
# Status: "confirmed" = the range was checked against the cited paper;
# "verify" = typical of the literature, check the paper before quoting it.
#
# Two values come from REGRESSIONS on this participant's speed instead of a
# fixed range (Schulz 2017, Fig. 4, no-obstacle surface; speed in leg lengths
# per second):
#     MTC ~ 3.15 v + 7.51 mm        TRI ~ 3.18 v - 2.02 s
# Schulz's subjects walked overground in shoes and no load, so these are the
# order of magnitude to expect, not a pass mark.
#
# REVISED: new cell. Version 3 printed each metric on its own and left the
# comparison to the reader.
print("\n" + "=" * 74)
print("  LITERATURE CHECK (steady walking)")
print("=" * 74)

lit_steady = gait_event_data.loc[steady]


def lit_mean(col, limb=None):
    v = lit_steady[col] if limb is None else lit_steady.loc[lit_steady['support_limb'] == limb, col]
    v = pd.to_numeric(v, errors='coerce')
    return float(v.mean()) if v.notna().any() else np.nan


def lit_sd(col):
    v = pd.to_numeric(lit_steady[col], errors='coerce')
    return float(v.std()) if v.notna().sum() > 2 else np.nan


def lit_cv(col):
    return 100 * lit_sd(col) / lit_mean(col)


def lit_kinetic(col):
    # kinetics only on the stances where both events came from the plate
    v = pd.to_numeric(lit_steady.loc[lit_steady['kinetics_trusted'].astype(bool), col], errors='coerce')
    return float(v.mean()) if v.notna().any() else np.nan


def lit_table_value(table, key_col, key, col):
    if table is None or len(table) == 0 or key_col not in table:
        return np.nan
    r = table.loc[table[key_col] == key, col]
    return float(r.iloc[0]) if len(r) else np.nan


lit_speed_ll = belt_speed / participant_leg_length
lit_paced = bool(metronome_paced) if metronome_paced is not None else ('bpm' in trial.lower())

# --- the trial's key values ------------------------------------------------------
# name: (value, unit)
key_metrics = {
    # temporal and spatial
    'stride_time_s': (lit_mean('stride_time'), 's'),
    'cadence_spm': (lit_mean('cadence'), 'steps/min'),
    'stance_pct': (lit_mean('stance_percentage'), '% cycle'),
    'double_support_pct': (lit_mean('double_support_percentage'), '% cycle'),
    'single_support_pct': (lit_mean('single_support_percentage'), '% cycle'),
    'step_length_m': (lit_mean('step_length'), 'm'),
    'step_width_m': (lit_mean('step_width'), 'm'),
    'stride_velocity_over_belt': (lit_mean('stride_velocity') / belt_speed, 'x belt speed'),
    'walk_ratio': (lit_mean('walk_ratio'), 'm/(steps/min)'),
    'stride_time_cv': (lit_cv('stride_time'), '%'),
    'stride_length_cv': (lit_cv('stride_length'), '%'),
    'step_width_sd_m': (lit_sd('step_width'), 'm'),
    # foot clearance and trip risk
    'mfc_mm': (1000 * lit_mean('minimum_foot_clearance'), 'mm'),
    'mfc_sd_mm': (1000 * lit_sd('minimum_foot_clearance'), 'mm'),
    'mfc_below_belt_pct': (100 * float((lit_steady['swing_min_clearance'] < 0).mean()), '% swings'),
    'trip_risk_integral_s': (lit_mean('trip_risk_integral'), 's'),
    'moi_peak_mm': (lit_mean('moi_peak_mm'), 'mm'),
    # margin of stability
    'mos_ml_contact_ankle_m': (lit_mean('mos_ml_contact_ankle'), 'm'),
    'mos_ml_contact_m': (lit_mean('mos_ml_contact'), 'm'),
    'mos_ml_min_m': (lit_mean('mos_ml_min'), 'm'),
    'mos_ap_contact_m': (lit_mean('mos_ap_contact'), 'm'),
    'mos_ap_ds_min_m': (lit_mean('mos_ap_ds_min'), 'm'),
    # kinetics, per TOTAL weight (body + load) against the unloaded pattern
    'grf_peak1_tw': (lit_kinetic('grf_peak1_tw'), 'TW'),
    'grf_trough_tw': (lit_kinetic('grf_trough_tw'), 'TW'),
    'grf_peak2_tw': (lit_kinetic('grf_peak2_tw'), 'TW'),
    'loading_rate_tw': (lit_kinetic('loading_rate_tw'), 'TW/s'),
    'peak_braking_tw': (lit_kinetic('peak_braking_tw'), 'TW'),
    'peak_propulsive_tw': (lit_kinetic('peak_propulsive_tw'), 'TW'),
    'braking_impulse_tw': (lit_kinetic('braking_impulse_tw'), 'TW s'),
    'propulsive_impulse_tw': (lit_kinetic('propulsive_impulse_tw'), 'TW s'),
    'vertical_impulse_per_stride': (lit_kinetic('vertical_impulse_tw') / lit_mean('stride_time'), 'TW s / stride s'),
    'cop_ap_range_mm': (lit_kinetic('cop_ap_range_mm'), 'mm'),
    'cop_ml_range_mm': (lit_kinetic('cop_ml_range_mm'), 'mm'),
    'free_moment_peak_nm': (lit_kinetic('free_moment_peak_nm'), 'N m'),
    # stride-to-stride
    'dfa_stride_time': (lit_table_value(dfa_results, 'series', 'stride_time', 'alpha'), '-'),
    'dfa_stride_length': (lit_table_value(dfa_results, 'series', 'stride_length', 'alpha'), '-'),
    'dfa_stride_velocity': (lit_table_value(dfa_results, 'series', 'stride_velocity', 'alpha'), '-'),
    'dfa_step_width': (lit_table_value(dfa_results, 'series', 'step_width', 'alpha'), '-'),
    'gem_perpendicular_dfa': (goal_equivalent['perpendicular_dfa'] if goal_equivalent else np.nan, '-'),
    'gem_parallel_dfa': (goal_equivalent['parallel_dfa'] if goal_equivalent else np.nan, '-'),
    'foot_placement_r2': (foot_placement['r2'] if foot_placement else np.nan, '-'),
    'lds_lambda_S_trunkVel_AP': (lit_table_value(lds_results, 'state_space', 'trunkVel_AP', 'lambda_S'), '1/stride'),
    'hr_AP': (float(harmonic_ratio['hr_AP'].median()) if len(harmonic_ratio) else np.nan, '-'),
    'hr_VT': (float(harmonic_ratio['hr_VT'].median()) if len(harmonic_ratio) else np.nan, '-'),
    'hr_ML': (float(harmonic_ratio['hr_ML'].median()) if len(harmonic_ratio) else np.nan, '-'),
    'ihr_AP': (float(harmonic_ratio['ihr_AP'].median()) if len(harmonic_ratio) else np.nan, '%'),
    'ihr_VT': (float(harmonic_ratio['ihr_VT'].median()) if len(harmonic_ratio) else np.nan, '%'),
    'ihr_ML': (float(harmonic_ratio['ihr_ML'].median()) if len(harmonic_ratio) else np.nan, '%'),
    'step_regularity_VT': (lit_table_value(gait_regularity, 'axis', 'VT', 'step_regularity'), '-'),
    'stride_regularity_VT': (lit_table_value(gait_regularity, 'axis', 'VT', 'stride_regularity'), '-'),
    'step_regularity_AP': (lit_table_value(gait_regularity, 'axis', 'AP', 'step_regularity'), '-'),
    'stride_regularity_AP': (lit_table_value(gait_regularity, 'axis', 'AP', 'stride_regularity'), '-'),
    'symmetry_angle_stance_time': (lit_table_value(symmetry_angles, 'variable', 'stance_time', 'symmetry_angle_pct'), '%'),
    'symmetry_angle_step_length': (lit_table_value(symmetry_angles, 'variable', 'step_length', 'symmetry_angle_pct'), '%'),
    'symmetry_angle_grf_peak1': (lit_table_value(symmetry_angles, 'variable', 'grf_peak1_bw', 'symmetry_angle_pct'), '%'),
}

# --- the healthy ranges -----------------------------------------------------------
# name: (low, high, population / condition, source, status)
literature = {
    'stride_time_s': (1.00, 1.15, 'healthy adults ~1.3 m/s (the 108-bpm metronome fixes it at 1.11 s)', 'Perry & Burnfield 2010; Oberg et al. 1993', 'verify'),
    'cadence_spm': (104, 120, 'healthy adults ~1.3 m/s (metronome: 108)', 'Oberg et al. 1993', 'verify'),
    'stance_pct': (58, 62, 'healthy adults, % of gait cycle', 'Perry & Burnfield 2010', 'verify'),
    'double_support_pct': (18, 24, 'healthy adults, both periods, % of gait cycle', 'Perry & Burnfield 2010', 'verify'),
    'single_support_pct': (38, 42, 'healthy adults, % of gait cycle', 'Perry & Burnfield 2010', 'verify'),
    'step_length_m': (0.65, 0.80, 'healthy adults ~1.3 m/s (scales with stature)', 'Oberg et al. 1993', 'verify'),
    'step_width_m': (0.08, 0.15, 'healthy young adults, treadmill, heel to heel', 'Owings & Grabiner 2004', 'verify'),
    'stride_velocity_over_belt': (0.98, 1.02, 'physics: on a treadmill the mean stride speed over the belt IS the belt speed', 'Dingwell et al. 2010', 'confirmed'),
    'walk_ratio': (0.0055, 0.0072, 'healthy adults, preferred speed (0.0063 +- 0.0007)', 'Sekiya & Nagasaki 1998', 'confirmed'),
    'stride_time_cv': (1.0, 3.0, 'healthy young adults', 'Hausdorff 2005', 'verify'),
    'stride_length_cv': (1.0, 3.0, 'healthy young adults', 'Hausdorff 2005', 'verify'),
    'step_width_sd_m': (0.010, 0.030, 'healthy young adults, treadmill', 'Owings & Grabiner 2004', 'verify'),
    'mfc_mm': (10, 30, 'healthy young adults (1-2 cm overground; treadmill studies up to 3 cm)', 'Begg et al. 2007; Barrett et al. 2010', 'confirmed'),
    'mfc_sd_mm': (2, 6, 'healthy young adults', 'Begg et al. 2007; Barrett et al. 2010', 'verify'),
    'mfc_below_belt_pct': (0, 0, 'physics: a swinging boot cannot be under the belt', '-', 'confirmed'),
    'mos_ml_contact_ankle_m': (0.03, 0.10, 'healthy adults, ankle/malleolus boundary; treadmill > overground', 'Hof et al. 2005; Rosenblum et al. 2021', 'verify'),
    'mos_ml_contact_m': (0.06, 0.15, 'BOOT EDGE: the ankle-boundary range plus half a boot (~0.03-0.05 m)', 'Hof et al. 2005 + boot geometry', 'verify'),
    'mos_ml_min_m': (0.05, 0.13, 'BOOT EDGE: the ankle-boundary range (0.02-0.08) plus half a boot', 'Hof et al. 2005 + boot geometry', 'verify'),
    'grf_peak1_tw': (1.05, 1.30, 'unloaded pattern; with a load it scales with total weight', 'Winter 2009; Birrell et al. 2007', 'verify'),
    'grf_trough_tw': (0.60, 0.85, 'unloaded pattern ~1.3 m/s', 'Winter 2009; Birrell et al. 2007', 'verify'),
    'grf_peak2_tw': (1.00, 1.25, 'unloaded pattern', 'Winter 2009; Birrell et al. 2007', 'verify'),
    'loading_rate_tw': (5.0, 15.0, 'shod walking, mean slope 20-80% of F1', 'gait literature; method dependent', 'verify'),
    'peak_braking_tw': (0.15, 0.25, 'healthy adults ~1.3 m/s', 'Winter 2009', 'verify'),
    'peak_propulsive_tw': (0.15, 0.25, 'healthy adults ~1.3 m/s', 'Winter 2009', 'verify'),
    'braking_impulse_tw': (-0.04, -0.02, 'healthy adults', 'Revi et al. 2020', 'verify'),
    'propulsive_impulse_tw': (0.02, 0.04, 'healthy adults', 'Revi et al. 2020', 'verify'),
    'vertical_impulse_per_stride': (0.47, 0.53, 'physics: each foot carries half the weight over a stride', 'Newton', 'confirmed'),
    'cop_ap_range_mm': (150, 220, 'healthy adults, path over the belt (~60-75% of foot length)', 'gait literature', 'verify'),
    'cop_ml_range_mm': (15, 40, 'healthy adults', 'gait literature', 'verify'),
    'free_moment_peak_nm': (2, 8, 'healthy adults walking', 'Holden & Cavanagh 1991; Li et al. 2001', 'verify'),
    'dfa_stride_time': ((0.20, 0.60, 'METRONOME-cued walking is anti-persistent', 'Hausdorff et al. 1996; Terrier et al. 2005; Terrier 2016', 'verify')
                        if lit_paced else
                        (0.75, 0.90, 'healthy adults, uncued', 'Hausdorff et al. 1996; Terrier et al. 2005', 'confirmed')),
    'dfa_stride_length': ((np.nan, np.nan, 'no range: metronome AND belt pace it (L = v T); the uncued treadmill value is 0.6-0.9', 'Dingwell et al. 2010', '-')
                          if lit_paced else
                          (0.60, 0.90, 'healthy young adults, treadmill (persistent)', 'Dingwell et al. 2010', 'confirmed')),
    'dfa_stride_velocity': (0.20, 0.40, 'healthy young adults, treadmill (anti-persistent)', 'Dingwell et al. 2010; Terrier 2012', 'confirmed'),
    'dfa_step_width': (0.60, 0.90, 'healthy young adults, treadmill (persistent)', 'Dingwell & Cusumano 2015', 'verify'),
    'gem_perpendicular_dfa': (0.20, 0.50, 'healthy young adults, treadmill', 'Dingwell et al. 2010', 'verify'),
    'gem_parallel_dfa': ((np.nan, np.nan, 'no range: with a metronome the goal-equivalent direction is paced too; uncued treadmill 0.7-1.0', 'Dingwell et al. 2010', '-')
                         if lit_paced else
                         (0.70, 1.00, 'healthy young adults, treadmill', 'Dingwell et al. 2010', 'verify')),
    'foot_placement_r2': (0.60, 0.90, 'healthy adults, ML, mid-stance (> 0.8 with pelvis state)', 'Wang & Srinivasan 2014', 'confirmed'),
    # REVISED (after D05, lambda_S = 2.3): no range. lambda_S depends on the
    # signal, the state space, the embedding, the time normalisation and the
    # fit window, so published values differ several-fold between methods
    # (Bruijn et al. 2013, review); none used this state space. Compare
    # conditions, with the same settings, never against a published number.
    'lds_lambda_S_trunkVel_AP': (np.nan, np.nan, 'no range: depends on the state space and settings; compare conditions', 'Bruijn et al. 2013 (review)', '-'),
    'hr_AP': (3.0, 4.0, 'healthy young adults, trunk accelerometer', 'Menz et al. 2003; Lowry et al. 2012', 'confirmed'),
    'hr_VT': (3.0, 4.0, 'healthy young adults, trunk accelerometer', 'Menz et al. 2003; Lowry et al. 2012', 'confirmed'),
    'hr_ML': (2.0, 2.7, 'healthy young adults, trunk accelerometer', 'Menz et al. 2003; Lowry et al. 2012', 'confirmed'),
    'ihr_AP': (85, 97, 'healthy young adults, sacrum', 'Pasciuto et al. 2017', 'verify'),
    'ihr_VT': (85, 97, 'healthy young adults, sacrum', 'Pasciuto et al. 2017', 'verify'),
    'ihr_ML': (70, 90, 'healthy young adults, sacrum', 'Pasciuto et al. 2017', 'verify'),
    'step_regularity_VT': (0.70, 0.95, 'healthy adults, trunk', 'Moe-Nilssen & Helbostad 2004; Kobsar et al. 2014', 'verify'),
    'stride_regularity_VT': (0.75, 0.95, 'healthy adults, trunk', 'Moe-Nilssen & Helbostad 2004; Kobsar et al. 2014', 'verify'),
    'step_regularity_AP': (0.60, 0.90, 'healthy adults, trunk', 'Moe-Nilssen & Helbostad 2004; Kobsar et al. 2014', 'verify'),
    'stride_regularity_AP': (0.65, 0.90, 'healthy adults, trunk', 'Moe-Nilssen & Helbostad 2004; Kobsar et al. 2014', 'verify'),
    'symmetry_angle_stance_time': (-3, 3, 'healthy adults', 'Zifchock et al. 2008', 'verify'),
    'symmetry_angle_step_length': (-3, 3, 'healthy adults', 'Zifchock et al. 2008', 'verify'),
    'symmetry_angle_grf_peak1': (-3, 3, 'healthy adults', 'Zifchock et al. 2008', 'verify'),
}

# --- the two regressions of Schulz (2017), at this participant's speed ---------------
# (expected value, population, source)
schulz_expected = {
    'mfc_mm': (3.15 * lit_speed_ll + 7.51,
               f'Schulz Fig. 4 regression at {lit_speed_ll:.2f} leg lengths/s, overground, no load'),
    'trip_risk_integral_s': (3.18 * lit_speed_ll - 2.02,
                             f'Schulz Fig. 4 regression at {lit_speed_ll:.2f} leg lengths/s, overground, no load'),
}

lit_rows = []
for k_, (v_, unit_) in key_metrics.items():
    lo_, hi_, pop_, src_, status_ = literature.get(k_, (np.nan, np.nan, '', '', ''))
    expected_ = schulz_expected.get(k_, (np.nan, ''))[0]
    if k_ in schulz_expected and not pop_:
        pop_, src_, status_ = schulz_expected[k_][1], 'Schulz 2017', 'verify'
    if not np.isfinite(v_):
        verdict_ = 'not computed'
    elif not (np.isfinite(lo_) and np.isfinite(hi_)):
        verdict_ = (f'{v_ / expected_:.2f} x expected' if np.isfinite(expected_) and expected_ > 0
                    else 'no range')
    elif v_ < lo_ - 1e-12:
        # within a tenth of the range's width outside: on the edge, not out
        verdict_ = 'edge, below' if v_ >= lo_ - 0.1 * (hi_ - lo_) else 'BELOW'
    elif v_ > hi_ + 1e-12:
        verdict_ = 'edge, above' if v_ <= hi_ + 0.1 * (hi_ - lo_) else 'ABOVE'
    else:
        verdict_ = 'within'
    lit_rows.append({'metric': k_, 'value': v_, 'unit': unit_, 'healthy_low': lo_, 'healthy_high': hi_,
                     'expected_schulz': expected_, 'verdict': verdict_, 'population': pop_,
                     'source': src_, 'status': status_})
literature_check = pd.DataFrame(lit_rows)

print(f"  belt speed {belt_speed:.3f} m/s = {lit_speed_ll:.2f} leg lengths/s, "
      f"load {load_kg:+.1f} kg\n")
print(f"  {'metric':30s} {'value':>10s}   {'healthy range':>19s}   verdict")
for _, r in literature_check.iterrows():
    rng_ = (f"{r['healthy_low']:.4g} to {r['healthy_high']:.4g}" if np.isfinite(r['healthy_low'])
            else (f"~{r['expected_schulz']:.3g} (Schulz)" if np.isfinite(r['expected_schulz']) else '-'))
    flag_ = '  <--' if r['verdict'] in ('BELOW', 'ABOVE') else ''
    print(f"  {r['metric']:30s} {r['value']:10.4g}   {rng_:>19s}   {r['verdict']}{flag_}")
lit_out = literature_check['verdict'].isin(['BELOW', 'ABOVE'])
print(f"\n  {int(lit_out.sum())} of {int(literature_check['healthy_low'].notna().sum())} metrics with a "
      f"range fall outside it; read the population column before reading anything into it:")
for _, r in literature_check[lit_out].iterrows():
    print(f"    {r['metric']:28s} {r['population']}")

if make_figures:
    # each value placed on its own healthy range: 0 = low end, 1 = high end
    lit_plot = literature_check[literature_check['healthy_low'].notna()
                                & (literature_check['healthy_high'] > literature_check['healthy_low'])
                                & literature_check['value'].notna()].reset_index(drop=True)
    lit_pos = ((lit_plot['value'] - lit_plot['healthy_low'])
               / (lit_plot['healthy_high'] - lit_plot['healthy_low'])).clip(-1.5, 2.5)
    plt.figure(figsize=(8, 0.26 * len(lit_plot) + 1.2))
    plt.axvspan(0, 1, color='#4a8f4a', alpha=0.12, label='healthy range')
    plt.scatter(lit_pos, np.arange(len(lit_plot)), s=22, zorder=3,
                color=np.where((lit_pos < 0) | (lit_pos > 1), '#cc3333', '#2a5d9f'))
    plt.yticks(np.arange(len(lit_plot)), lit_plot['metric'], fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel('position in the healthy range (0 = low end, 1 = high end; clipped at -1.5 and 2.5)')
    plt.title(f'{trial}: key metrics against the literature', fontsize=10)
    plt.grid(axis='x', alpha=0.2)
    plt.legend(frameon=False, fontsize=8, loc='lower right')
    plt.tight_layout()


# =============================================================================
# %% Saving
# =============================================================================
# Three tables, in base / gait_analysis_outputs:
#   {trial}_all_metrics_steps.csv     one row per step, every per-step column
#   {trial}_all_metrics_summary.csv   the literature check above
#   all_trials_all_metrics.csv        ONE ROW PER TRIAL, key metrics as columns;
#                                     re-running a trial replaces its row
# and, in a folder per trial, the tables of the stride-to-stride cells (LDS,
# DFA, entropy, harmonic ratio per stride, regularity, symmetry angles).
#
# REVISED: new cell. Version 3 left the saving commented out.
if save_outputs:
    out_folder = base / "gait_analysis_outputs"
    out_trial = out_folder / trial
    out_trial.mkdir(parents=True, exist_ok=True)

    gait_event_data.to_csv(out_folder / f"{trial}_all_metrics_steps.csv", index=False)
    literature_check.to_csv(out_folder / f"{trial}_all_metrics_summary.csv", index=False)
    for out_name, out_table in (('lds', lds_results), ('dfa', dfa_results), ('entropy', entropy_results),
                                ('harmonic_ratio', harmonic_ratio), ('regularity', gait_regularity),
                                ('symmetry_angles', symmetry_angles), ('multiscale_entropy', multiscale_entropy)):
        if out_table is not None and len(out_table):
            out_table.to_csv(out_trial / f"{trial}_{out_name}.csv",
                             index=(out_name == 'multiscale_entropy'))

    out_row = {'trial': trial, 'participant': participant, 'belt_speed_ms': belt_speed,
               'body_mass_kg': body_mass, 'load_kg': load_kg,
               'n_steps_steady': int(steady.sum()), 'n_strides_series': len(stride_series)}
    out_row.update({k_: v_ for k_, (v_, _) in key_metrics.items()})
    out_row.update({f'{k_}_expected_schulz': v_[0] for k_, v_ in schulz_expected.items()})
    out_all_path = out_folder / "all_trials_all_metrics.csv"
    out_all = pd.read_csv(out_all_path) if out_all_path.exists() else pd.DataFrame()
    if len(out_all) and 'trial' in out_all:
        out_all = out_all[out_all['trial'] != trial]
    out_all = pd.concat([out_all, pd.DataFrame([out_row])], ignore_index=True)
    out_all.to_csv(out_all_path, index=False)
    print(f"\n  saved to {out_folder}")
    print(f"    {trial}_all_metrics_steps.csv ({len(gait_event_data)} steps), "
          f"{trial}_all_metrics_summary.csv, all_trials_all_metrics.csv ({len(out_all)} trials)")

if make_figures:
    plt.show()
