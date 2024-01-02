import mne
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy import interpolate
import matplotlib.pyplot as plt

show_checks = False


def create_bad_annotations(bad_indices, times, description, eye, orig_time):
    """
    This function takes in an array of indices marking each bad data according to whatever metric and converts it
    to mne annotations to be append to the raw object.
    :param bad_indices: (list) list of integers containing the index of samples marked as bad
    :param times: (np array) time vector from the mne raw object
    :param description: (string) description of the bad annotation
    :param eye: (string) left or right, for the eye to annotate
    :param orig_time: (POSIX Timestamp) start time of the recording. This is necessary to be able to concatenate with
    other annotations
    :return: (mne annotations object) generated annotations
    """
    # Convert to annotations:
    onsets, offsets = convert_onset_offset(bad_indices)
    # Convert samples to times:
    onsets = [times[ind] for ind in onsets]
    offsets = [times[ind] for ind in offsets]
    # Compute the duration:
    duration = [offset - onsets[ind] for ind, offset in enumerate(offsets)]
    desc = [description] * len(onsets)
    ch_names = [('xpos_' + eye, 'ypos_' + eye, 'pupil_' + eye)] * len(onsets)
    # Create the annotations:
    annot = mne.Annotations(
        onset=onsets,  # in seconds
        duration=duration,  # in seconds, too
        description=desc,
        ch_names=ch_names,
        orig_time=orig_time
    )
    return annot


def convert_onset_offset(bad_indices):
    """
    This function converts an array of indices into two array, marking the onset and offset of contiguous chunks of data
    marked as bad. This is useful to create mne annotations.
    :param bad_indices: (list) list of integers containing the index of samples marked as bad
    :return: onsets, offsets: lists marking the onset and offset of each contiguous segment
    """
    if not bad_indices:
        return [], []
    # Prepare lists to store onset and offsets
    onsets = [bad_indices[0]]
    offsets = []
    # Loop through each index:
    for i in range(1, len(bad_indices)):
        # If the previous index + 1 is not equal to the current index, then this is not a contiguous segment
        # and we are at an edge:
        if bad_indices[i] != bad_indices[i - 1] + 1:
            offsets.append(bad_indices[i - 1])
            onsets.append(bad_indices[i])
    offsets.append(bad_indices[-1])
    return onsets, offsets


def remove_bad_epochs(epochs, channels=None, nan_proportion_thresh=0.2):
    """
    This function identifies any epochs in which there is more than X% in any channel
    :param epochs:
    :param threshold:
    :return:
    """
    print("=" * 40)
    print("Removing bad epochs (more than {}% nan)".format(nan_proportion_thresh * 100))
    if channels is None:
        channels = ["BAD_blinks_left", "BAD_blinks_right"]
    # Extract the data:
    data = epochs.get_data(channels)
    data_combined = np.logical_and(data[:, 0, :], data[:, 0, :])
    # Compute the proportion of nan:
    nan_proportion = np.sum(data_combined, axis=1) / data.shape[2]
    # Extract the epochs that have more than X% nan:
    bad_epochs = np.where(np.max(nan_proportion, axis=0) > nan_proportion_thresh)[0]
    # Remove the bad epochs:
    epochs.drop(bad_epochs, reason='TOO_MANY_NANS')
    proportion_rejected = len(bad_epochs) / len(epochs)
    # Print the number of removed epochs:
    print("Removed {:2f}% epochs".format(100 * proportion_rejected))
    # Dropping the bad epochs if there were any:
    epochs.drop_bad()
    return epochs, proportion_rejected


def trend_line_departure(raw, threshold_factor=3, eyes=None, window_length_s=0.05, n_iter=4, polyorder=3):
    """
    This function performs an iterative procedure in which the trendline is computed using a savegol filter. Then,
    the unfiltered data are compared to the filtered one. Sample which deviates from more than X mad (3 is the default)
    are marked as bad in the MNE annotations. This is an iterative such that the same process is repeated N times. This
    is due to the fact that fitted trend line will change once outliers are removed and new outliers might emerge. This
    ensures that massive outliers do not hide smaller but nonetheless artifactual samples
    :param raw: (mne raw object) contains the pupil data for which to compute trend line departure
    :param threshold_factor: (float) factors for the MAD outlier detection (if set to 4, samples for which dilation
    speed deviates from 4 MAD or more are rejected).
    :param eyes: (string) left or right
    :param window_length_s: (float) length of the window for the savgol filter to compute the trend line
    :param n_iter: (int) number of iteration to perform
    :paranm polyorder: (int) polynomial order of the savegol filter.
    :return:
    """
    if eyes is None:
        eyes = ["left", "right"]
    print("=" * 40)
    print("Trend line departure")
    # Calculate the window length:
    window_length_n = int(window_length_s * raw.info["sfreq"])
    # Loop through each eye:
    for eye in eyes:
        bad_indices = []
        # Extract the raw data:
        raw_data = np.squeeze(raw.copy().get_data(picks='pupil_' + eye))
        for i in range(n_iter):
            # Filter the data:
            data_filt = savgol_filter(raw_data, window_length_n, polyorder)
            # Compute the difference between the data and the smoothed data:
            dev = np.abs(raw_data - data_filt)
            # Compute the median absolute deviation:
            inds = mad_outliers_ind(dev, threshold_factor=threshold_factor, axis=0)
            print("Removing {:2f}% samples in iter {}".format((len(inds) / raw_data.shape[0]) * 100, i))
            # Append the bad indices to the data:
            bad_indices.append(inds)
            # Set the bad samples to nan, so that they are excluded from the next computation:
            raw_data[inds] = np.nan
        # Set the bad indices in the annotations:
        bad_indices = list(np.unique(np.concatenate(bad_indices, axis=0)))
        # Convert to annotations:
        annot = create_bad_annotations(bad_indices, raw.times, "BAD_trend_line_departure", eye,
                                       raw.annotations.orig_time)
        # Combine annotations:
        raw.set_annotations(raw.annotations + annot)
    if show_checks:
        # Create a copy of the raw data to only select the bad annotations:
        raw_copy = raw.copy()
        bad_annot_ind = [ind for ind, val in enumerate(raw_copy.annotations.description) if "BAD_" in val]
        bad_annot = mne.Annotations(
            onset=raw_copy.annotations.onset[bad_annot_ind],  # in seconds
            duration=raw_copy.annotations.duration[bad_annot_ind],  # in seconds, too
            description=raw_copy.annotations.description[bad_annot_ind],
            ch_names=raw_copy.annotations.ch_names[bad_annot_ind],
            orig_time=raw_copy.annotations.orig_time
        )
        raw_copy.set_annotations(bad_annot)
        raw_copy.plot(scalings=dict(eyegaze=1e3), block=True)

    return raw


def dilation_filter(pupil_size, times, axis=-1):
    """
    This function computes a dilation filter according to the description found here:
    https://link.springer.com/article/10.3758/s13428-018-1075-y
    :param pupil_size: (np array)
    :return: (np array)
    """
    if len(pupil_size.shape) > 1:
        raise Exception("The pupil array has more than 1 dimension. This function currently supports one eye at a time")
    # Compute the forward diff:
    forward_diff = np.diff(pupil_size, axis=axis)
    # Pad with nan in the end:
    forward_diff = np.pad(forward_diff, (0, 1), 'constant', constant_values=np.nan)
    # Compute the backward diff:
    backward_diff = np.flip(np.diff(np.flip(pupil_size, axis=axis), axis=axis), axis=axis)
    # Pad with nan in the beginning:
    backward_diff = np.pad(backward_diff, (1, 0), 'constant', constant_values=np.nan)
    # Compute time interval:
    dt = times[1] - times[0]
    # Compute dilation speed:
    dilation_speed = np.nanmax([forward_diff / dt, backward_diff / dt], axis=0)
    return dilation_speed


def mad_outliers_ind(data, threshold_factor=4, axis=0):
    if len(data.shape) > 1:
        raise Exception("This function only supports 1D arrays")
    # Compute the MAD:
    mad = np.nanmedian(np.abs(data - np.nanmedian(data, axis=axis)), axis=axis)
    # Compute the threshold:
    thresh = np.nanmedian(data, axis=axis) + threshold_factor * mad
    # Find the outliers:
    outliers_ind = np.where(data > thresh)[0]
    return outliers_ind


def dilation_speed_rejection(raw, threshold_factor=4, eyes=None):
    """
    This function handles the annotations of samples based on a speed filter. Samples in which pupil dilation speed
    deviates from a MAD threshold are marked as bad, following
    https://link.springer.com/article/10.3758/s13428-018-1075-y. The detected segments are annotated as
    "BAD_speed_outlier" in the raw annotations
    :param raw: (mne raw object) contains the eyetracking data
    :param threshold_factor: (float) factors for the MAD outlier detection (if set to 4, samples for which dilation
    speed deviates from 4 MAD or more are rejected).
    :param eyes: (list) eyes to apply the filter to
    :return: raw (mne raw object) contains the data with additional annotations based on dilation speed filtering
    """
    print("=" * 40)
    print("Rejecting samples based on dilation speed")
    if eyes is None:
        eyes = ["left", "right"]

    # Loop through each eye:
    for eye in eyes:
        # Extract the pupil size from this eye:
        data = raw.copy().get_data(picks='pupil_' + eye)
        # Compute the dilation speed:
        dilation_speed = dilation_filter(np.squeeze(data), raw.times, axis=-1)
        # Extract the index of the outliers:
        outliers_ind = mad_outliers_ind(dilation_speed, threshold_factor=threshold_factor, axis=0)
        # Display some information about the proportion of outliers that were found:
        print("For {} eye: ".format(eye))
        print("{:2f}% of rejected samples ({} out of {})".format((outliers_ind.shape[0] / data.shape[-1]) *
                                                                 100,
                                                                 outliers_ind.shape[0], data.shape[-1]))
        # Convert to annotations:
        annot = create_bad_annotations(list(outliers_ind), raw.times, "BAD_speed_outlier", eye,
                                       raw.annotations.orig_time)
        # Combine annotations:
        raw.set_annotations(raw.annotations + annot)

    if show_checks:
        # Create a copy of the raw data to only select the bad annotations:
        raw_copy = raw.copy()
        bad_annot_ind = [ind for ind, val in enumerate(raw_copy.annotations.description) if "BAD_" in val]
        bad_annot = mne.Annotations(
            onset=raw_copy.annotations.onset[bad_annot_ind],  # in seconds
            duration=raw_copy.annotations.duration[bad_annot_ind],  # in seconds, too
            description=raw_copy.annotations.description[bad_annot_ind],
            ch_names=raw_copy.annotations.ch_names[bad_annot_ind],
            orig_time=raw_copy.annotations.orig_time
        )
        raw_copy.set_annotations(bad_annot)
        raw_copy.plot(scalings=dict(eyegaze=1e3), block=True)

    return raw


def create_metadata_from_events(epochs, metadata_column):
    """
    This function parses the events found in the epochs descriptions to create the meta data. The column of the meta
    data are generated based on the metadata column names. The column name must be a list in the same order as the
    strings describing the events. The name of the column must be the name of the overall condition, so say the
    specific column describes the category of the presented stim (faces, objects...), then the column should be called
    category. This will become obsolete here at some point, when the preprocessing is changed to generate the meta data
    directly
    :param epochs: (mne epochs object) epochs for which the meta data will be generated
    :param metadata_column: (list of strings) name of the column of the meta data. Must be in the same order
    as the events description + must be of the same length as the number of word in the events description
    :return: epochs (mne epochs object)
    """

    # Getting the event description of each single trial
    trials_descriptions = [[key for key in epochs.event_id.keys() if epochs.event_id[key] == event]
                           for event in epochs.events[:, 2]]
    trial_descriptions_parsed = [description[0].split(
        "/") for description in trials_descriptions]
    # Making sure that the dimensions of the trials description is consistent across all trials:
    if len(set([len(vals) for vals in trial_descriptions_parsed])) > 1:
        raise ValueError('dimension mismatch in event description!\nThe forward slash separated list found in the '
                         'epochs description has inconsistent length when parsed. Having different number of '
                         'descriptors for different trials is not yet supported. Please make sure that your events '
                         'description are set accordingly')
    if len(metadata_column) != len(trial_descriptions_parsed[0]):
        raise ValueError("The number of meta data columns you have passed doesn't match the number of descriptors for\n"
                         "each trials. Make sure you have matching numbers. In doubt, go and check the events file in\n"
                         "the BIDS directory")
    if len(trial_descriptions_parsed) != len(epochs):
        raise ValueError("Somehow, the number of trials descriptions found in the epochs object doesn't match the "
                         "number of trials in the same epochs. I have no idea how you managed that one champion, so I "
                         "can't really help here")

    # Convert the trials description to a pandas dataframe:
    epochs.metadata = pd.DataFrame.from_records(
        trial_descriptions_parsed, columns=metadata_column)

    return epochs


def epoch_data(raw, events, event_dict, events_of_interest=None, metadata_column=None, tmin=-0.5, tmax=2.0,
               baseline=None, picks="all", reject_by_annotation=False):
    """
    This function epochs the continuous data according to specified events of interest, i.e. not all the events get
    evoked, only those we are interested in!
    :param raw: (mne raw object) contains the data to epochs
    :param events: (array of int) ID of each event
    :param event_dict: (dictionary) description for each event UID
    :param events_of_interest: (list of strings) list of events that we wish to epochs. The name must match strings
    found in the event_dict keys
    :param metadata_column: (list of strings) name of the meta data table columns. The event descriptions must be
    encoded as \ separated values. Each string in the event dict key corresponds to a specific parameter from the
    experiment. These are then parsed as a meta data table accordingly
    :param tmin: (float) time from which to epoch (relative to event onset)
    :param tmax: (float) time until which to epoch (relative to event onset)
    :param baseline: (None or tuple) time to use as baseline. If set to None, no baseline correction applied
    :param picks: (list or "all") list of channels to epoch
    :param reject_by_annotation: (boolean) whether or not to reject trials based on annotations
    :return: mne epochs object: the epoched data
    """
    # First, extract the events of interest:
    if events_of_interest is not None:
        select_event_dict = {key: event_dict[key] for key in event_dict if any(substring in key
                                                                               for substring in events_of_interest)}
    else:
        select_event_dict = event_dict
    # Epochs the data accordingly:
    epochs = mne.Epochs(raw, events=events, event_id=select_event_dict, tmin=tmin,
                        tmax=tmax, baseline=baseline, verbose='ERROR', picks=picks,
                        reject_by_annotation=reject_by_annotation)
    # Reject trials whose end falls outside the recording:
    tend = raw.times[-1]
    # Find trials whose end is outside the recording:
    bad_trials = np.where((epochs.events[:, 0] / raw.info["sfreq"]) + tmax > tend)[0]
    # Drop those trials:
    epochs.drop(bad_trials)
    # Dropping the bad epochs if there were any:
    epochs.drop_bad()
    # Adding the meta data to the table. The meta data are created by parsing the events strings, as each substring
    # contains specific info about the trial:
    if metadata_column is not None:
        epochs = create_metadata_from_events(epochs, metadata_column)
    return epochs


def extract_eyelink_events(raw, description="blink", eyes=None):
    """
    This function extracts the eyelink events from the annotation. In the annotation, we have the onset and duration
    for each of the eyelink parser events. These are converted to continuous regressors, with ones where we have the
    event in question and zeros elsewhere. This is for handy processing down the line, where we can reject or regress
    those out.
    :param raw: (mne raw object) raw object containing the annotations and continuous recordings
    :param description: (string) identifier for the event in question (blink, saccades, fixations...). The description
    must match the description found in the raw object annotation
    :param eyes: (list or None) eye to use. By default, set to use both, which will create one channel per eye and per
    event. MONOCULAR NOT IMPLEMENTED
    :return: raw_new (mne raw object) raw object with the added channels encoding the events and their duration
    """
    # Create the new channels, one per eye:
    if eyes is None:
        eyes = ["left", "right"]

    desc_vectors = []
    for eye in eyes:
        # Extract the events
        evts_ind = np.where(raw.annotations.description == "_".join([description, eye]))[0]
        # Extract the onset and duration of the said event:
        evt_onset = raw.annotations.onset[evts_ind]
        evt_offset = evt_onset + raw.annotations.duration[evts_ind]
        # Convert to samples:
        onset = (evt_onset * raw.info["sfreq"]).astype(int)
        offset = (evt_offset * raw.info["sfreq"]).astype(int)

        # Set the regressor to 1 where the event is happening:
        desc_vector = np.zeros(raw.n_times)
        for i in range(len(onset)):
            desc_vector[onset[i]:offset[i]] = 1
        desc_vectors.append(desc_vector)

    # Add these two channels to the raw data:
    data = raw.get_data()
    data = np.concatenate([data, np.array(desc_vectors)])
    channels = raw.ch_names
    channels.extend(["_".join([description, eye]) for eye in eyes])
    info = mne.create_info(channels,
                           ch_types=["eeg"] * len(channels),
                           sfreq=raw.info["sfreq"])
    info.set_meas_date(raw.info['meas_date'])
    raw_new = mne.io.RawArray(data, info, verbose="WARNING")
    raw_new.set_annotations(raw.annotations)
    return raw_new
