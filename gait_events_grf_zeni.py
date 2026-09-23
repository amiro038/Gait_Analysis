# -*- coding: utf-8 -*-
"""
=============================================================================
 GAIT EVENTS: TRUSTED GRF FIRST, ZENI FOR EVERYTHING ELSE
=============================================================================

Every heel strike and toe-off, in the same files gait_events.py writes:

    1. GRF, but only where it can be trusted. A belt event is kept when the
       posed boot meshes show that belt carrying one foot, all of it and
       clear of the gap, with the other foot off it, the centre of pressure
       under the foot, and a force that rises or falls like a foot landing
       or leaving. The limb comes from the mesh, not the belt's name.

    2. Zeni et al. (2008) for every event the GRF could not vouch for:
       heel strike where the heel is furthest in front of the pelvis, toe-off
       where the toe is furthest behind it, along the walking direction.
       Each one is searched for only in the window the gait sequence
       (L HS -> R TO -> R HS -> L TO) leaves for it, and its timing is
       corrected by its median offset from this trial's trusted GRF events,
       per limb.

    3. Only if Zeni finds nothing there -- in practice a tracking dropout,
       where the kinematics are gone too -- a clean-looking belt contact the
       mesh could not check (source GRF_unverified), and failing that a
       time interpolated between the neighbouring events (interpolated).

Set the paths below and press F5, or

    python gait_events_grf_zeni.py                    # every trial
    python gait_events_grf_zeni.py STEM [STEM ...]    # just these

Per trial, in OUTPUT_FOLDER (see gait_events.py for the details):

    {trial}_merged_events.csv    event_type, support_limb, frame_100hz,
                                 source -- source is GRF, kinematic (Zeni),
                                 GRF_unverified or interpolated
    {trial}_event_qa.csv         how every event was found, stance/stride flags
    {trial}_grf_contacts.csv     every belt contact and why its events were
                                 or were not trusted
    {trial}_zeni_benchmark.csv   Zeni against held-out trusted GRF events,
                                 i.e. how far off the filled-in events are
                                 likely to be
    {trial}_event_registration.json

and zeni_benchmark_all_trials.csv across the run. Everything not set here --
thresholds, trust margins, the plate file -- is in gait_events.py's CONFIG.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fallback_detectors as fd      # noqa: E402
import gait_events as ge             # noqa: E402


# %%==========================================================================
#  CONFIG
# ============================================================================

DATA_FOLDER = Path(
    r"C:\Users\lcour081\Box\Military Project\Data\Analysis\DICE_Treadmill"
)
FORCE_FOLDER = DATA_FOLDER / "FP_renamed"
KINEMATIC_FOLDER = DATA_FOLDER / "Theia_csv_outputs"
OUTPUT_FOLDER = DATA_FOLDER / "gait_event_outputs"
TRIALS = None                  # None = every .csv in FORCE_FOLDER, or stems


def configure():
    """GRF, then Zeni only, then unverified GRF, then interpolation."""
    ge.FORCE_FOLDER = FORCE_FOLDER
    ge.KINEMATIC_FOLDER = KINEMATIC_FOLDER
    ge.OUTPUT_FOLDER = OUTPUT_FOLDER
    ge.TRIALS = TRIALS
    fd.DETECTORS = {"zeni_position": fd.zeni_position}
    ge.FALLBACK_ORDER = ["zeni_position"]
    ge.USE_UNVERIFIED_GRF = "last"
    ge.INTERPOLATE_MISSING = True
    ge.BENCHMARK_NAME = "zeni_benchmark"


def main(stems=None):
    configure()
    ge.main(stems)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
