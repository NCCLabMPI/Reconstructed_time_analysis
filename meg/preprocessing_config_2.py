# Default settings for data processing and analysis.

###############################################################################
# Config parameters
# -----------------
study_name = "ReconTime"
bids_root = r'C:\Users\alexander.lepauvre\Documents\PhD\Reconstructed_Time\meg_pilot\bids'  # Use this to specify a path here.
deriv_root = r'C:\Users\alexander.lepauvre\Documents\PhD\Reconstructed_Time\meg_pilot\bids\derivatives'  # Use this to specify a path here.
interactive = False
sessions = ["1"]
task = "visual"
runs = ["01"]
plot_psd_for_runs = "all"
subjects = "all"
process_empty_room = False
process_rest = False
ch_types = ["meg"]
data_type = "meg"
eog_channels = None
config_validation = "ignore"

###############################################################################
# BREAK DETECTION
# ---------------
find_breaks = True
min_break_duration = 15.0
t_break_annot_start_after_previous_event = 5.0
t_break_annot_stop_before_next_event = 5.0

###############################################################################
# FREQUENCY FILTERING & RESAMPLING
# --------------------------------
l_freq = None
h_freq = 40.0
notch_freq = 50
raw_resample_sfreq = 250.0

###############################################################################
# DECIMATION
# ----------
epochs_decim = 1

###############################################################################
# RENAME EXPERIMENTAL EVENTS
# --------------------------
rename_events = dict()

###############################################################################
# HANDLING OF REPEATED EVENTS
# ---------------------------
event_repeated = "merge"

###############################################################################
# EPOCHING
# --------
conditions = ["visual_onset", "auditory_onset"]
epochs_tmin = -0.5
epochs_tmax = 3.0
task_is_rest = False
baseline = (None, 0)
contrasts = [
    {
        'name': 'facevsobject',
        'conditions': [
            'visual_onset/face',
            'visual_onset/object'
        ],
        'weights': [-1, 1]
    },
    {
        'name': 'highvslow',
        'conditions': [
            'auditory_onset/1000',
            'auditory_onset/1100'
        ],
        'weights': [-1, 1]
    }

]

# contrasts: Iterable[Union[Tuple[str, str], ArbitraryContrast]] = []
# """
# The conditions to contrast via a subtraction of ERPs / ERFs. The list elements
# can either be tuples or dictionaries (or a mix of both). Each element in the
# list corresponds to a single contrast.

# A tuple specifies a one-vs-one contrast, where the second condition is
# subtracted from the first.

# If a dictionary, must contain the following keys:

# - `name`: a custom name of the contrast
# - `conditions`: the conditions to contrast
# - `weights`: the weights associated with each condition.

# Pass an empty list to avoid calculation of any contrasts.

# For the contrasts to be computed, the appropriate conditions must have been
# epoched, and therefore the conditions should either match or be subsets of
# `conditions` above.

# ???+ example "Example"
#     Contrast the "left" and the "right" conditions by calculating
#     `left - right` at every time point of the evoked responses:
#     ```python
#     contrasts = [('left', 'right')]  # Note we pass a tuple inside the list!
#     ```

#     Contrast the "left" and the "right" conditions within the "auditory" and
#     the "visual" modality, and "auditory" vs "visual" regardless of side:
#     ```python
#     contrasts = [('auditory/left', 'auditory/right'),
#                  ('visual/left', 'visual/right'),
#                  ('auditory', 'visual')]
#     ```

#     Contrast the "left" and the "right" regardless of side, and compute an
#     arbitrary contrast with a gradient of weights:
#     ```python
#     contrasts = [
#         ('auditory/left', 'auditory/right'),
#         {
#             'name': 'gradedContrast',
#             'conditions': [
#                 'auditory/left',
#                 'auditory/right',
#                 'visual/left',
#                 'visual/right'
#             ],
#             'weights': [-1.5, -.5, .5, 1.5]
#         }
#     ]
#     ```
# """

###############################################################################
# ARTIFACT REMOVAL
# ----------------
spatial_filter = "ica"
ica_max_iterations = 500
ica_l_freq = 1.0
ica_n_components = 0.99
ica_reject_components = "auto"
ica_algorithm = "fastica"
ica_l_freq = 1.0
ica_max_iterations = 500

###############################################################################
# DECODING
# --------
decode = True
decoding_epochs_tmin = 0.0
decoding_epochs_tmax = None
decoding_metric = "roc_auc"
decoding_n_splits = 5
decoding_time_generalization = False
decoding_train_times = None
decoding_predict_times = None
decoding_train_on_all_data = False
decoding_predict_proba = False
decoding_regularize = None
decoding_contrast_name = None

###############################################################################
# GROUP AVERAGE SENSORS
# ---------------------
interpolate_bads_grand_average = True

###############################################################################
# TIME-FREQUENCY
# --------------

# time_frequency_conditions: Iterable[str] = []
# """
# The conditions to compute time-frequency decomposition on.

# ???+ example "Example"
#     ```python
#     time_frequency_conditions = ['left', 'right']
#     ```
# """

# time_frequency_freq_min: Optional[float] = 8
# """
# Minimum frequency for the time frequency analysis, in Hz.
# ???+ example "Example"
#     ```python
#     time_frequency_freq_min = 0.3  # 0.3 Hz
#     ```
# """

# time_frequency_freq_max: Optional[float] = 40
# """
# Maximum frequency for the time frequency analysis, in Hz.
# ???+ example "Example"
#     ```python
#     time_frequency_freq_max = 22.3  # 22.3 Hz
#     ```
# """

# time_frequency_cycles: Optional[Union[float, ArrayLike]] = None
# """
# The number of cycles to use in the Morlet wavelet. This can be a single number
# or one per frequency, where frequencies are calculated via
# `np.arange(time_frequency_freq_min, time_frequency_freq_max)`.
# If `None`, uses
# `np.arange(time_frequency_freq_min, time_frequency_freq_max) / 3`.
# """

# time_frequency_subtract_evoked: bool = False
# """
# Whether to subtract the evoked signal (averaged across all epochs) from the
# epochs before passing them to time-frequency analysis. Set this to `True` to
# highlight induced activity.

# !!! info
#      This also applies to CSP analysis.
# """

###############################################################################
# TIME-FREQUENCY CSP
# ------------------
decoding_csp = False
# decoding_csp: bool = False
# """
# Whether to run decoding via Common Spatial Patterns (CSP) analysis on the
# data. CSP takes as input data covariances that are estimated on different
# time and frequency ranges. This allows to obtain decoding scores defined over
# time and frequency.
# """

# decoding_csp_times: Optional[ArrayLike] = np.linspace(
#     max(0, epochs_tmin), epochs_tmax, num=6
# )
# """
# The edges of the time bins to use for CSP decoding.
# Must contain at least two elements. By default, 5 equally-spaced bins are
# created across the non-negative time range of the epochs.
# All specified time points must be contained in the epochs interval.
# If `None`, do not perform **time-frequency** analysis, and only run CSP on
# **frequency** data.

# ???+ example "Example"
#     Create 3 equidistant time bins (0–0.2, 0.2–0.4, 0.4–0.6 sec):
#     ```python
#     decoding_csp_times = np.linspace(start=0, stop=0.6, num=4)
#     ```
#     Create 2 time bins of different durations (0–0.4, 0.4–0.6 sec):
#     ```python
#     decoding_csp_times = [0, 0.4, 0.6]
#     ```
# """

# decoding_csp_freqs: Dict[str, ArrayLike] = {
#     "custom": [
#         time_frequency_freq_min,
#         (time_frequency_freq_max + time_frequency_freq_min) / 2,  # noqa: E501
#         time_frequency_freq_max,
#     ]
# }
# """
# The edges of the frequency bins to use for CSP decoding.

# This parameter must be a dictionary with:
# - keys specifying the unique identifier or "name" to use for the frequency
#   range to be treated jointly during statistical testing (such as "alpha" or
#   "beta"), and
# - values must be list-like objects containing at least two scalar values,
#   specifying the edges of the respective frequency bin(s), e.g., `[8, 12]`.

# Defaults to two frequency bins, one from
# [`time_frequency_freq_min`][mne_bids_pipeline._config.time_frequency_freq_min]
# to the midpoint between this value and
# [`time_frequency_freq_max`][mne_bids_pipeline._config.time_frequency_freq_max];
# and the other from that midpoint to `time_frequency_freq_max`.
# ???+ example "Example"
#     Create two frequency bins, one for 4–8 Hz, and another for 8–14 Hz, which
#     will be clustered together during statistical testing (in the
#     time-frequency plane):
#     ```python
#     decoding_csp_freqs = {
#         'custom_range': [4, 8, 14]
#     }
#     ```
#     Create the same two frequency bins, but treat them separately during
#     statistical testing (i.e., temporal clustering only):
#     ```python
#     decoding_csp_freqs = {
#         'theta': [4, 8],
#         'alpha': [8, 14]
#     }
#     ```
#     Create 5 equidistant frequency bins from 4 to 14 Hz:
#     ```python
#     decoding_csp_freqs = {
#         'custom_range': np.linspace(
#             start=4,
#             stop=14,
#             num=5+1  # We need one more to account for the endpoint!
#         )
#     }
# """

# time_frequency_baseline: Optional[Tuple[float, float]] = None
# """
# Baseline period to use for the time-frequency analysis. If `None`, no baseline.
# ???+ example "Example"
#     ```python
#     time_frequency_baseline = (None, 0)
#     ```
# """

# time_frequency_baseline_mode: str = "mean"
# """
# Baseline mode to use for the time-frequency analysis. Can be chosen among:
# "mean" or "ratio" or "logratio" or "percent" or "zscore" or "zlogratio".
# ???+ example "Example"
#     ```python
#     time_frequency_baseline_mode = 'mean'
#     ```
# """

# time_frequency_crop: Optional[dict] = None
# """
# Period and frequency range to crop the time-frequency analysis to.
# If `None`, no cropping.

# ???+ example "Example"
#     ```python
#     time_frequency_crop = dict(tmin=-0.3, tmax=0.5, fmin=5, fmax=20)
#     ```
# """

###############################################################################
# SOURCE ESTIMATION PARAMETERS
# ----------------------------
#
run_source_estimation = False
# run_source_estimation: bool = True
# """
# Whether to run source estimation processing steps if not explicitly requested.
# """

# use_template_mri: Optional[str] = None
# """
# Whether to use a template MRI subject such as FreeSurfer's `fsaverage` subject.
# This may come in handy if you don't have individual MR scans of your
# participants, as is often the case in EEG studies.

# Note that the template MRI subject must be available as a subject
# in your subjects_dir. You can use for example a scaled version
# of fsaverage that could get with
# [`mne.scale_mri`](https://mne.tools/stable/generated/mne.scale_mri.html).
# Scaling fsaverage can be a solution to problems that occur when the head of a
# subject is small compared to `fsaverage` and, therefore, the default
# coregistration mislocalizes MEG sensors inside the head.

# ???+ example "Example"
#     ```python
#     use_template_mri = "fsaverage"
#     ```
# """

# adjust_coreg: bool = False
# """
# Whether to adjust the coregistration between the MRI and the channels
# locations, possibly combined with the digitized head shape points.
# Setting it to True is mandatory if you use a template MRI subject
# that is different from `fsaverage`.

# ???+ example "Example"
#     ```python
#     adjust_coreg = True
#     ```
# """

# bem_mri_images: Literal["FLASH", "T1", "auto"] = "auto"
# """
# Which types of MRI images to use when creating the BEM model.
# If `'FLASH'`, use FLASH MRI images, and raise an exception if they cannot be
# found.

# ???+ info "Advice"
#     It is recommended to use the FLASH images if available, as the quality
#     of the extracted BEM surfaces will be higher.

# If `'T1'`, create the BEM surfaces from the T1-weighted images using the
# `watershed` algorithm.

# If `'auto'`, use FLASH images if available, and use the `watershed``
# algorithm with the T1-weighted images otherwise.

# *[FLASH MRI]: Fast low angle shot magnetic resonance imaging
# """

# recreate_bem: bool = False
# """
# Whether to re-create the BEM surfaces, even if existing surfaces have been
# found. If `False`, the BEM surfaces are only created if they do not exist
# already. `True` forces their recreation, overwriting existing BEM surfaces.
# """

# recreate_scalp_surface: bool = False
# """
# Whether to re-create the scalp surfaces used for visualization of the
# coregistration in the report and the lower-density coregistration surfaces.
# If `False`, the scalp surface is only created if it does not exist already.
# If `True`, forces a re-computation.
# """

# freesurfer_verbose: bool = False
# """
# Whether to print the complete output of FreeSurfer commands. Note that if
# `False`, no FreeSurfer output might be displayed at all!"""

# mri_t1_path_generator: Optional[Callable[[BIDSPath], BIDSPath]] = None
# """
# To perform source-level analyses, the Pipeline needs to generate a
# transformation matrix that translates coordinates from MEG and EEG sensor
# space to MRI space, and vice versa. This process, called "coregistration",
# requires access to both, the electrophyisiological recordings as well as
# T1-weighted MRI images of the same participant. If both are stored within
# the same session, the Pipeline (or, more specifically, MNE-BIDS) can find the
# respective files automatically.

# However, in certain situations, this is not possible. Examples include:

# - MRI was conducted during a different session than the electrophysiological
#   recording.
# - MRI was conducted in a single session, while electrophysiological recordings
#   spanned across several sessions.
# - MRI and electrophysiological data are stored in separate BIDS datasets to
#   allow easier storage and distribution in certain situations.

# To allow the Pipeline to find the correct MRI images and perform coregistration
# automatically, we provide a "hook" that allows you to provide a custom
# function whose output tells the Pipeline where to find the T1-weighted image.

# The function is expected to accept a single parameter: The Pipeline will pass
# a `BIDSPath` with the following parameters set based on the currently processed
# electrophysiological data:

# - the subject ID, `BIDSPath.subject`
# - the experimental session, `BIDSPath.session`
# - the BIDS root, `BIDSPath.root`

# This `BIDSPath` can then be modified – or an entirely new `BIDSPath` can be
# generated – and returned by the function, pointing to the T1-weighted image.

# !!! info
#     The function accepts and returns a single `BIDSPath`.

# ???+ example "Example"
#     The MRI session is different than the electrophysiological session:
#     ```python
#     def get_t1_from_meeg(bids_path):
#         bids_path.session = 'MRI'
#         return bids_path


#     mri_t1_path_generator = get_t1_from_meeg
#     ```

#     The MRI recording is stored in a different BIDS dataset than the
#     electrophysiological data:
#     ```python
#     def get_t1_from_meeg(bids_path):
#         bids_path.root = '/data/mri'
#         return bids_path


#     mri_t1_path_generator = get_t1_from_meeg
#     ```
# """

# mri_landmarks_kind: Optional[Callable[[BIDSPath], str]] = None
# """
# This config option allows to look for specific landmarks in the json
# sidecar file of the T1 MRI file. This can be useful when we have different
# fiducials coordinates e.g. the manually positioned fiducials or the
# fiducials derived for the coregistration transformation of a given session.

# ???+ example "Example"
#     We have one MRI session and we have landmarks with a kind
#     indicating how to find the landmarks for each session:

#     ```python
#     def mri_landmarks_kind(bids_path):
#         return f"ses-{bids_path.session}"
#     ```
# """

# spacing: Union[Literal["oct5", "oct6", "ico4", "ico5", "all"], int] = "oct6"
# """
# The spacing to use. Can be `'ico#'` for a recursively subdivided
# icosahedron, `'oct#'` for a recursively subdivided octahedron,
# `'all'` for all points, or an integer to use approximate
# distance-based spacing (in mm). See (the respective MNE-Python documentation)
# [https://mne.tools/dev/overview/cookbook.html#setting-up-the-source-space]
# for more info.
# """

# mindist: float = 5
# """
# Exclude points closer than this distance (mm) to the bounding surface.
# """

# loose: Union[float, Literal["auto"]] = 0.2
# """
# Value that weights the source variances of the dipole components
# that are parallel (tangential) to the cortical surface. If `0`, then the
# inverse solution is computed with **fixed orientation.**
# If `1`, it corresponds to **free orientation.**
# The default value, `'auto'`, is set to `0.2` for surface-oriented source
# spaces, and to `1.0` for volumetric, discrete, or mixed source spaces,
# unless `fixed is True` in which case the value 0. is used.
# """

# depth: Optional[Union[float, dict]] = 0.8
# """
# If float (default 0.8), it acts as the depth weighting exponent (`exp`)
# to use (must be between 0 and 1). None is equivalent to 0, meaning no
# depth weighting is performed. Can also be a `dict` containing additional
# keyword arguments to pass to :func:`mne.forward.compute_depth_prior`
# (see docstring for details and defaults).
# """

# inverse_method: Literal["MNE", "dSPM", "sLORETA", "eLORETA"] = "dSPM"
# """
# Use minimum norm, dSPM (default), sLORETA, or eLORETA to calculate the inverse
# solution.
# """

# noise_cov: Union[
#     Tuple[Optional[float], Optional[float]],
#     Literal["emptyroom", "rest", "ad-hoc"],
#     Callable[[BIDSPath], mne.Covariance],
# ] = (None, 0)
# """
# Specify how to estimate the noise covariance matrix, which is used in
# inverse modeling.

# If a tuple, it takes the form `(tmin, tmax)` with the time specified in
# seconds. If the first value of the tuple is `None`, the considered
# period starts at the beginning of the epoch. If the second value of the
# tuple is `None`, the considered period ends at the end of the epoch.
# The default, `(None, 0)`, includes the entire period before the event,
# which is typically the pre-stimulus period.

# If `'emptyroom'`, the noise covariance matrix will be estimated from an
# empty-room MEG recording. The empty-room recording will be automatically
# selected based on recording date and time. This cannot be used with EEG data.

# If `'rest'`, the noise covariance will be estimated from a resting-state
# recording (i.e., a recording with `task-rest` and without a `run` in the
# filename).

# If `'ad-hoc'`, a diagonal ad-hoc noise covariance matrix will be used.

# You can also pass a function that accepts a `BIDSPath` and returns an
# `mne.Covariance` instance. The `BIDSPath` will point to the file containing
# the generated evoked data.

# ???+ example "Example"
#     Use the period from start of the epoch until 100 ms before the experimental
#     event:
#     ```python
#     noise_cov = (None, -0.1)
#     ```

#     Use the time period from the experimental event until the end of the epoch:
#     ```python
#     noise_cov = (0, None)
#     ```

#     Use an empty-room recording:
#     ```python
#     noise_cov = 'emptyroom'
#     ```

#     Use a resting-state recording:
#     ```python
#     noise_cov = 'rest'
#     ```

#     Use an ad-hoc covariance:
#     ```python
#     noise_cov = 'ad-hoc'
#     ```

#     Use a custom covariance derived from raw data:
#     ```python
#     def noise_cov(bids_path):
#         bp = bids_path.copy().update(task='rest', run=None, suffix='meg')
#         raw_rest = mne_bids.read_raw_bids(bp)
#         raw.crop(tmin=5, tmax=60)
#         cov = mne.compute_raw_covariance(raw, rank='info')
#         return cov
#     ```
# """

# source_info_path_update: Optional[Dict[str, str]] = dict(suffix="ave")
# """
# When computing the forward and inverse solutions, by default the pipeline
# retrieves the `mne.Info` object from the cleaned evoked data. However, in
# certain situations you may wish to use a different `Info`.

# This parameter allows you to explicitly specify from which file to retrieve the
# `mne.Info` object. Use this parameter to supply a dictionary to
# `BIDSPath.update()` during the forward and inverse processing steps.

# ???+ example "Example"
#     Use the `Info` object stored in the cleaned epochs:
#     ```python
#     source_info_path_update = {'processing': 'clean',
#                                'suffix': 'epo'}
#     ```
# """

# inverse_targets: List[Literal["evoked"]] = ["evoked"]
# """

# On which data to apply the inverse operator. Currently, the only supported
# target is `'evoked'`. If no inverse computation should be done, pass an
# empty list, `[]`.

# ???+ example "Example"
#     Compute the inverse solution on evoked data:
#     ```python
#     inverse_targets = ['evoked']
#     ```

#     Don't compute an inverse solution:
#     ```python
#     inverse_targets = []
#     ```
# """

###############################################################################
# Report generation
# -----------------

# report_evoked_n_time_points: Optional[int] = None
# """
# Specifies the number of time points to display for each evoked
# in the report. If `None`, it defaults to the current default in MNE-Python.

# ???+ example "Example"
#     Only display 5 time points per evoked
#     ```python
#     report_evoked_n_time_points = 5
#     ```
# """

# report_stc_n_time_points: Optional[int] = None
# """
# Specifies the number of time points to display for each source estimates
# in the report. If `None`, it defaults to the current default in MNE-Python.

# ???+ example "Example"
#     Only display 5 images per source estimate:
#     ```python
#     report_stc_n_time_points = 5
#     ```
# """

###############################################################################
# Execution
# ---------
n_jobs = 4
# n_jobs: int = 1
# """
# Specifies how many subjects you want to process in parallel. If `1`, disables
# parallel processing.
# """

# parallel_backend: Literal["loky", "dask"] = "loky"
# """
# Specifies which backend to use for parallel job execution. `loky` is the
# default backend used by `joblib`. `dask` requires [`Dask`](https://dask.org) to
# be installed. Ignored if [`n_jobs`][mne_bids_pipeline._config.n_jobs] is set to
# `1`.
# """

# dask_open_dashboard: bool = False
# """
# Whether to open the Dask dashboard in the default webbrowser automatically.
# Ignored if `parallel_backend` is not `'dask'`.
# """

# dask_temp_dir: Optional[PathLike] = None
# """
# The temporary directory to use by Dask. Dask places lock-files in this
# directory, and also uses it to "spill" RAM contents to disk if the amount of
# free memory in the system hits a critical low. It is recommended to point this
# to a location on a fast, local disk (i.e., not a network-attached storage) to
# ensure good performance. The directory needs to be writable and will be created
# if it does not exist.

# If `None`, will use `.dask-worker-space` inside of
# [`deriv_root`][mne_bids_pipeline._config.deriv_root].
# """

# dask_worker_memory_limit: str = "10G"
# """
# The maximum amount of RAM per Dask worker.
# """

# random_state: Optional[int] = 42
# """
# You can specify the seed of the random number generator (RNG).
# This setting is passed to the ICA algorithm and to the decoding function,
# ensuring reproducible results. Set to `None` to avoid setting the RNG
# to a defined state.
# """

# shortest_event: int = 1
# """
# Minimum number of samples an event must last. If the
# duration is less than this, an exception will be raised.
# """

# log_level: Literal["info", "error"] = "info"
# """
# Set the pipeline logging verbosity.
# """

# mne_log_level: Literal["info", "error"] = "error"
# """
# Set the MNE-Python logging verbosity.
# """

# on_error: Literal["continue", "abort", "debug"] = "abort"
# """
# Whether to abort processing as soon as an error occurs, continue with all other
# processing steps for as long as possible, or drop you into a debugger in case
# of an error.

# !!! info
#     Enabling debug mode deactivates parallel processing.
# """

# memory_location: Optional[Union[PathLike, bool]] = True
# """
# If not None (or False), caching will be enabled and the cache files will be
# stored in the given directory. The default (True) will use a
# `'joblib'` subdirectory in the BIDS derivative root of the dataset.
# """

# memory_file_method: Literal["mtime", "hash"] = "mtime"
# """
# The method to use for cache invalidation (i.e., detecting changes). Using the
# "modified time" reported by the filesystem (`'mtime'`, default) is very fast
# but requires that the filesystem supports proper mtime reporting. Using file
# hashes (`'hash'`) is slower and requires reading all input files but should
# work on any filesystem.
# """

# memory_verbose: int = 0
# """
# The verbosity to use when using memory. The default (0) does not print, while
# 1 will print the function calls that will be cached. See the documentation for
# the joblib.Memory class for more information."""

# config_validation: Literal["raise", "warn", "ignore"] = "raise"
# """
# How strictly to validate the configuration. Errors are always raised for
# invalid entries (e.g., not providing `ch_types`). This setting controls
# how to handle *possibly* or *likely* incorrect entries, such as likely
# misspellings (e.g., providing `session` instead of `sessions`).
# """
