# -*- coding: utf-8 -*-
"""
Gait analysis of the DICE split-belt treadmill trials, step 2 of 2.

    step 1   detect_gait_events.py   every heel strike and toe-off
    step 2   run_gait_analysis.py    every metric, with confidence intervals

The paths are the ones set at the top of detect_gait_events.py, so they are
set in one place. The metrics live in the gait/ package, one module per
family:

    gait/trial.py           load a trial: events -> steps, boots, forces,
                            the load from the opening quiet standing, the
                            system CoM and its velocity
    gait/spatiotemporal.py  times, lengths, widths
    gait/stability.py       margin of stability, MFC, trip risk
    gait/kinetics.py        GRF metrics and CoM work, trusted stances only
    gait/kinematics.py      trunk lean, joint ranges of motion
    gait/variability.py     DFA, entropy, harmonic ratio, regularity, GEM,
                            foot placement, symmetry
    gait/lds.py             local dynamic stability
    gait/uncertainty.py     block bootstrap and DFA intervals
    gait/summary.py         everything into one long table
    gait/figures.py         QA figures

Run it with F5 after running detect_gait_events.py, or

    python run_gait_analysis.py                    # every trial with events
    python run_gait_analysis.py "D05_C1_Treadmill_1.3mpers 108bpm"

Per trial, in OUTPUT_FOLDER:

    {trial}_steps.csv         one row per heel strike, every per-step metric
    {trial}_summary.csv       one row per number: metric, limb, statistic,
                              value, 95% CI, n, bootstrap block length
    {trial}_waveforms.npz     time-normalised GRF and joint angles per stride,
                              LDS divergence curves, MSE curve
    {trial}_lds_windows.csv   lambda per window of strides
    {trial}_qa.png, {trial}_variability.png

and all_trials_summary.csv: every trial's summary stacked, ready for a mixed
model (participant random, condition fixed).
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import detect_gait_events as dge                              # noqa: E402
from gait import figures, kinematics, kinetics, spatiotemporal  # noqa: E402
from gait import stability, summary                           # noqa: E402
from gait import trial as gt                                  # noqa: E402

# %% ==========================================================================
#  CONFIG
# =============================================================================

FOLDERS = dict(force=dge.FORCE_FOLDER, kinematic=dge.KINEMATIC_FOLDER,
               events=dge.OUTPUT_FOLDER, binding=dge.BINDING_FILE,
               plates=dge.PLATE_FILE)
OUTPUT_FOLDER = dge.DATA_FOLDER / "gait_analysis_outputs"
PARTICIPANTS_FILE = HERE / "participants.csv"
TRIALS = None                  # None = every trial detect_gait_events wrote

# Stride series for DFA, entropy and GEM are cut to this many strides so
# every trial is compared at the same N. "shortest" = the shortest trial in
# this run (after the warm-up); an int fixes it; None uses all strides.
SERIES_LENGTH = "shortest"
LIMB_FOR_TRUNK = "R"           # strides of this limb for HR and LDS
RUN_LDS = True
FIGURES = True
VIDEO = False                  # birds-eye MoS video of a typical stride


# %% ==========================================================================

def read_participants(path):
    if not Path(path).exists():
        print(f"! {path} not found: no body mass, so the load is unknown")
        return {}
    p = pd.read_csv(path)
    return {r["participant"]: r.to_dict() for _, r in p.iterrows()}


def trial_stems():
    if TRIALS:
        return list(TRIALS)
    return sorted(p.name.replace("_merged_events.csv", "")
                  for p in Path(FOLDERS["events"]).glob("*_merged_events.csv"))


def steady_strides(stem):
    """Strides per limb after the warm-up, read off the event files."""
    ev = Path(FOLDERS["events"])
    qa = ev / f"{stem}_event_qa.csv"
    e = pd.read_csv(qa if qa.exists() else ev / f"{stem}_merged_events.csv")
    row = e["frame_float"] if "frame_float" in e else e["frame_100hz"]
    hs = e["event_type"] == "heel_strike"
    start = row.min() + gt.WARMUP_S * gt.FS
    return min(int(((e["support_limb"] == b) & hs & (row >= start)).sum()) - 1
               for b in ("L", "R"))


def analyse(stem, participant, series_length):
    t = gt.load_trial(stem, FOLDERS, participant)
    spatiotemporal.spatiotemporal(t)
    stability.margin_of_stability(t)
    stability.foot_clearance(t)
    kinetics.stance_kinetics(t)
    kinematics.stride_kinematics(t)
    print("  metrics per step done; summarising (bootstrap, DFA, entropy"
          + (", LDS" if RUN_LDS else "") + ")")
    table, extras = summary.summarise(t, series_length, LIMB_FOR_TRUNK,
                                      RUN_LDS)

    out = Path(OUTPUT_FOLDER)
    out.mkdir(parents=True, exist_ok=True)
    t.steps.to_csv(out / f"{stem}_steps.csv", index=False)
    table.to_csv(out / f"{stem}_summary.csv", index=False)
    extras["lds_windows"].to_csv(out / f"{stem}_lds_windows.csv", index=False)
    waves = {f"grf_{k}": np.asarray(v) for k, v in t.waves.items()}
    waves.update({f"angle_{k}": np.asarray(v)
                  for k, v in t.angle_waves.items()})
    waves.update({f"lds_{k}": v for k, v in extras["lds_curves"].items()})
    if extras["mse"] is not None:
        waves.update({f"mse_{c}": extras["mse"][c].to_numpy()
                      for c in extras["mse"]})
    np.savez_compressed(out / f"{stem}_waveforms.npz", **waves)
    if FIGURES:
        figures.qa_figure(t, out / f"{stem}_qa.png")
        figures.variability_figure(t, table, extras, series_length,
                                   out / f"{stem}_variability.png",
                                   LIMB_FOR_TRUNK)
    if VIDEO:
        print(f"  video: {figures.mos_video(t, out / f'{stem}_mos_stride')}")
    print(f"  saved to {out}")
    return t, table


def main(stems=None):
    stems = stems or trial_stems()
    people = read_participants(PARTICIPANTS_FILE)
    n = SERIES_LENGTH
    if n == "shortest":
        counts = {s: steady_strides(s) for s in stems}
        n = min(counts.values())
        print(f"series cut to {n} strides, the shortest trial's "
              f"({min(counts, key=counts.get)})")
    tables = []
    for stem in stems:
        who = stem.split("_")[0]
        try:
            tables.append(analyse(stem, people.get(who, {}), n)[1])
        except Exception as exc:                       # keep the batch going
            print(f"\n{stem}: FAILED -- {exc}")
            traceback.print_exc()
    if tables:
        allt = pd.concat(tables, ignore_index=True)
        allt.to_csv(Path(OUTPUT_FOLDER) / "all_trials_summary.csv",
                    index=False)
        print(f"\n{len(tables)} of {len(stems)} trials -> "
              f"{Path(OUTPUT_FOLDER) / 'all_trials_summary.csv'}")
    return tables


if __name__ == "__main__":
    main(sys.argv[1:] or None)
