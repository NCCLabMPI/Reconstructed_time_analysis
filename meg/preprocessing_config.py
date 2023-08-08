# Default settings for data processing and analysis.

from typing import Optional, Union, Iterable, List, Tuple, Dict, Callable, Literal

from numpy.typing import ArrayLike

import mne
from mne_bids import BIDSPath
import numpy as np

from mne_bids_pipeline.typing import PathLike, ArbitraryContrast


###############################################################################
# Config parameters
# -----------------
study_name = "ReconTime"
bids_root = r'C:\Users\alexander.lepauvre\Documents\PhD\Reconstructed_Time\meg_pilot\bids'  # Use this to specify a path here.
deriv_root = r'C:\Users\alexander.lepauvre\Documents\PhD\Reconstructed_Time\meg_pilot\bids\derivatives'  # Use this to specify a path here.
interactive = False
sessions = ["1"]
task = "visual"
runs = "all"
plot_psd_for_runs = "all"
subjects = "all"
process_empty_room = False
process_rest = False
ch_types = ["meg"]
data_type = "meg"
eog_channels = None

###############################################################################
# BREAK DETECTION
# ---------------
find_breaks = True
min_break_duration = 15.0

###############################################################################
# FREQUENCY FILTERING & RESAMPLING
# --------------------------------
l_freq = None
h_freq = 40.0
notch_freq = 50
raw_resample_sfreq = 250.0

###############################################################################
# ARTIFACT REMOVAL:
# -----------------
spatial_filter = "ica"
ica_max_iterations = 500
ica_l_freq = 1.0
ica_n_components = 0.99
ica_reject_components = "auto"

###############################################################################
# DECIMATION
# ----------
epochs_decim = 1

###############################################################################
# RENAME EXPERIMENTAL EVENTS
# --------------------------
rename_events = dict()

###############################################################################
# EPOCHING
# --------
epochs_tmin = -0.2
epochs_tmax = 3.0
baseline = (None, 0)
event_repeated = "drop"
conditions = ["visual_onset", "auditory_onset"]


