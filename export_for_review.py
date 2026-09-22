# =============================================================================
# Export a review bundle
# =============================================================================
# Run this in the SAME Spyder session, after Gait_analysis_all_metrics.py has
# finished, so every variable is still live in the kernel.
#
# The point is that the raw trial is gigabytes and the things needed to check
# the results are not. Two tiers:
#
#   TIER 1  the tables. stride_series IS by construction the complete input to
#           every stride-to-stride metric, so with it any alpha, entropy, GEM
#           or symmetry number can be recomputed from scratch and compared.
#           A few hundred KB.
#   TIER 2  a short window of the RAW signals, so the per-step machinery
#           (kinetics, margin of stability, foot clearance, harmonic ratio) can
#           be run against real data instead of synthetic. Set the window below.
#
# Nothing here recomputes anything. It only writes out what already exists, so
# it cannot change a result.
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import io as _io

exp_window_s   = 30.0    # seconds of raw signal for tier 2. 0 disables it
exp_window_at  = 120.0   # start the window this far into the trial
exp_out = gait_event_path.parent / "review_bundle"
exp_out.mkdir(exist_ok=True)

print("\n" + "=" * 74)
print(f"  EXPORTING A REVIEW BUNDLE TO {exp_out}")
print("=" * 74)

# --- tier 1: every table, and the settings that produced them ----------------
# globals() rather than dir(): a name that a cell never reached simply is not
# here, and the export says so instead of failing
for exp_name in ('gait_event_data', 'stride_series', 'lds_results', 'dfa_results',
                 'entropy_results', 'multiscale_entropy', 'harmonic_ratio',
                 'gait_regularity', 'symmetry_angles', 'foot_clearance'):
    exp_obj = globals().get(exp_name)
    if isinstance(exp_obj, pd.DataFrame) and len(exp_obj):
        exp_obj.to_csv(exp_out / f"{exp_name}.csv", index=True)
        print(f"  {exp_name:22s} {exp_obj.shape[0]:5d} x {exp_obj.shape[1]:3d}")
    else:
        print(f"  {exp_name:22s} not present, skipped")

# the scalars and settings, so the numbers can be reproduced exactly
exp_scalars = {}
for exp_name in ('body_mass', 'body_weight', 'kinematic_fs', 'force_fs',
                 'participant_mass_kg', 'participant_leg_length',
                 'participant_height', 'participant_foot_length',
                 'mos_belt_speed', 'mos_travel_sign', 'mos_gravity',
                 'ss_limb', 'ss_warmup_s', 'ss_fixed_n', 'lds_limb', 'n_strides',
                 'dfa_min_box', 'dfa_max_box_frac', 'dfa_order',
                 'ent_m', 'ent_r', 'ent_mse_scales', 'hr_n_harmonics',
                 'walk_ratio', 'mos_have_mesh'):
    if exp_name in globals():
        exp_v = globals()[exp_name]
        if isinstance(exp_v, (int, float, str, bool, type(None), np.floating, np.integer)):
            exp_scalars[exp_name] = exp_v

# the goal-equivalent and foot placement results are dicts of mixed content, so
# only their scalar entries go in
for exp_src in ('goal_equivalent', 'foot_placement'):
    exp_d = globals().get(exp_src)
    if isinstance(exp_d, dict):
        for exp_k, exp_v in exp_d.items():
            if isinstance(exp_v, (int, float, np.floating, np.integer)):
                exp_scalars[f"{exp_src}.{exp_k}"] = float(exp_v)

pd.Series(exp_scalars).to_csv(exp_out / "settings_and_scalars.csv", header=False)
print(f"  {'settings and scalars':22s} {len(exp_scalars):5d} values")

# --- a text digest, so the numbers can be read without loading anything ------
exp_txt = _io.StringIO()
exp_txt.write(f"trial: {gait_event_path.stem}\n")
exp_txt.write(f"force file: {force_path.name}\n\n")
for exp_k, exp_v in exp_scalars.items():
    exp_txt.write(f"{exp_k:34s} {exp_v}\n")

for exp_name in ('gait_event_data', 'stride_series'):
    exp_obj = globals().get(exp_name)
    if not isinstance(exp_obj, pd.DataFrame) or not len(exp_obj):
        continue
    exp_num = exp_obj.select_dtypes(include=[np.number])
    exp_txt.write(f"\n\n{'=' * 70}\n{exp_name}: {exp_obj.shape[0]} rows\n{'=' * 70}\n")
    exp_desc = exp_num.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
    exp_desc['n_nan'] = exp_num.isna().sum()
    exp_txt.write(exp_desc.to_string(float_format=lambda v: f"{v:12.5f}"))
    # and the same split by limb, which is where an asymmetry or a one-sided
    # detection failure shows up
    if 'support_limb' in exp_obj.columns:
        exp_txt.write(f"\n\nby limb, means:\n")
        exp_txt.write(exp_num.groupby(exp_obj['support_limb']).mean().T.to_string(
            float_format=lambda v: f"{v:12.5f}"))

for exp_name in ('lds_results', 'dfa_results', 'entropy_results',
                 'multiscale_entropy', 'gait_regularity', 'symmetry_angles'):
    exp_obj = globals().get(exp_name)
    if isinstance(exp_obj, pd.DataFrame) and len(exp_obj):
        exp_txt.write(f"\n\n{'=' * 70}\n{exp_name}\n{'=' * 70}\n")
        exp_txt.write(exp_obj.to_string(float_format=lambda v: f"{v:10.4f}"))

(exp_out / "digest.txt").write_text(exp_txt.getvalue(), encoding='utf-8')
print(f"  {'digest.txt':22s} {len(exp_txt.getvalue())/1024:5.0f} KB")

# --- tier 2: a short window of the raw signals -------------------------------
# float32 and compressed: the values carry far more precision than the
# measurement does, and the window is for checking behaviour, not archiving
if exp_window_s > 0:
    exp_a = int(exp_window_at * kinematic_fs)
    exp_b = exp_a + int(exp_window_s * kinematic_fs)
    exp_b = min(exp_b, len(kinematic_data))
    exp_fa = int(exp_window_at * force_fs)
    exp_fb = exp_fa + int(exp_window_s * force_fs)
    exp_fb = min(exp_fb, len(force_data))

    exp_pack = {'window_start_frame_100hz': exp_a + 1,
                'window_end_frame_100hz': exp_b,
                'kinematic_fs': kinematic_fs, 'force_fs': force_fs}

    # only the kinematic columns anything in the script actually reads
    exp_cols = []
    for exp_base in ('Left_Heel_Position', 'Right_Heel_Position',
                     'Left_Ankle_Position', 'Right_Ankle_Position',
                     'Left_Toes_Position', 'Right_Toes_Position',
                     'Whole_body_COG', 'Low_Back_Joint_Acc', 'Trunk_Joint_Acc'):
        exp_cols += [c for c in (exp_base, exp_base + '.1', exp_base + '.2')
                     if c in kinematic_data.columns]
    exp_pack['kinematic_columns'] = np.array(exp_cols)
    exp_pack['kinematic'] = kinematic_data[exp_cols].iloc[exp_a:exp_b].to_numpy(np.float32)

    exp_fcols = [c for c in force_data.columns
                 if 'belt_' in c or 'handrail_' in c]
    exp_pack['force_columns'] = np.array(exp_fcols)
    exp_pack['force'] = force_data[exp_fcols].iloc[exp_fa:exp_fb].to_numpy(np.float32)

    # The foot meshes are 10208 vertices, which is 368 MB for a 3000-frame
    # trial -- hopeless to ship. But the vertices are POSES times a fixed
    # template, so exporting the poses for the window plus ONE copy of the
    # template reconstructs them exactly, in under a megabyte:
    #
    #     verts = ab.apply_poses_selected(poses, vertices_local,
    #                                     vertex_segment, idx)
    #
    # which is the same call the margin of stability cell makes.
    if 'mos_posed_path' in globals() and mos_posed_path.exists():
        exp_posed = np.load(mos_posed_path, allow_pickle=False)
        exp_pack['poses'] = exp_posed['poses'][exp_a:exp_b].astype(np.float32)
        exp_pack['pose_segments'] = exp_posed['segments']
        # the frame ids that go with them, so the alignment to the kinematics
        # can be checked rather than assumed
        if 'frames' in exp_posed.files:
            exp_pack['pose_frames'] = exp_posed['frames'][exp_a:exp_b]
        print(f"  {'poses':22s} {exp_pack['poses'].shape} "
              f"{exp_pack['poses'].nbytes/1e6:.2f} MB")

    if 'mos_binding_path' in globals() and mos_binding_path.exists():
        exp_bind = np.load(mos_binding_path, allow_pickle=False)
        exp_pack['vertices_local'] = exp_bind['vertices_local'].astype(np.float32)
        exp_pack['vertex_segment'] = exp_bind['vertex_segment']
        exp_pack['binding_meta'] = exp_bind['meta']
        print(f"  {'mesh template':22s} "
              f"{exp_pack['vertices_local'].shape[0]} vertices, "
              f"{exp_pack['vertices_local'].nbytes/1e6:.2f} MB (one copy, not per frame)")

    # and the already-posed vertex heights, as a cheap independent check that
    # the reconstruction above lands where this run put it
    if globals().get('mos_have_mesh') and 'mos_foot_vertices' in globals():
        for exp_side in ('Left', 'Right'):
            exp_pack[f'min_z_{exp_side}'] = np.nanmin(
                mos_foot_vertices[exp_side][exp_a:exp_b, :, 2], axis=1).astype(np.float32)

    # the events inside the window, so the frames line up
    exp_ev = gait_event_data[
        (gait_event_data['initial_contact_kinematic_frame_100hz'] >= exp_a + 1)
        & (gait_event_data['initial_contact_kinematic_frame_100hz'] <= exp_b)]
    exp_ev.to_csv(exp_out / "excerpt_events.csv", index=False)

    np.savez_compressed(exp_out / "excerpt.npz", **exp_pack)
    print(f"  {'excerpt.npz':22s} {exp_window_s:.0f} s from {exp_window_at:.0f} s, "
          f"{(exp_out / 'excerpt.npz').stat().st_size/1e6:5.1f} MB")

exp_total = sum(p.stat().st_size for p in exp_out.iterdir() if p.is_file())
print(f"\n  {len(list(exp_out.iterdir()))} files, {exp_total/1e6:.1f} MB total")
print(f"  {exp_out}")
if exp_total > 25e6:
    print(f"  ! over 25 MB. Lower exp_window_s and re-run, or send the CSVs alone")
