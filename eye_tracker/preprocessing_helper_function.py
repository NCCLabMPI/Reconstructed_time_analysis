import mne
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def remove_bad_epochs(epochs, nan_proportion_thresh=0.2):
    """
    This function identifies any epochs in which there is more than X% in any channel
    :param epochs:
    :param threshold:
    :return:
    """
    # Extract the data:
    data = epochs.get_data(["LPupil", "RPupil"])
    # Compute the proportion of nan:
    nan_proportion = np.sum(np.isnan(data), axis=2) / data.shape[2]
    # Extract the epochs that have more than X% nan:
    bad_epochs = np.where(np.max(nan_proportion, axis=1) > nan_proportion_thresh)[0]
    # Remove the bad epochs:
    epochs.drop(bad_epochs, reason='TOO_MANY_NANS')
    print(epochs.drop_log)
    return epochs


def trend_line_departure(raw, threshold_factor=3, eyes=None, window_length_s=0.05, polyorder=3, n_iter=4):
    """
    This function identifies samples that deviate more than X MAD from the trend line
    :param raw:
    :param threshold_factor:
    :param eyes:
    :return:
    """
    if eyes is None:
        eyes = ["L", "R"]
    print("="*40)
    print("Trend line departure")
    # Convert the window length to samples:
    window_length_n = int(window_length_s * raw.info["sfreq"])
    # Loop through each eye:
    for eye in eyes:
        # Extract the data:
        for i in range(n_iter):
            if i == 0:
                data = np.squeeze(raw.copy().get_data(picks="{}Pupil".format(eye)))
            # Interpolate the data:
            data_interp = interp_nan(data.copy())
            # Smooth the data:
            data_filt = savgol_filter(data_interp, window_length_n, polyorder)
            # Compute the difference between the data and the smoothed data:
            dev = np.abs(data_interp - data_filt)
            # Compute the median absolute deviation:
            inds = mad_outliers_ind(dev, threshold_factor=threshold_factor, axis=0)
            print("Removing {:2f}% samples in iter {}".format((len(inds[0]) / data.shape[0]) * 100, i))
            # Set the data to nan:
            data[inds] = np.nan
        # Add back into raw:
        raw_data = raw.get_data()
        # Extract the index of the channel:
        ch_ind = np.where(np.array(raw.ch_names) == "{}Pupil".format(eye))[0]
        raw_data[ch_ind, :] = data
        # Recreate the raw object:
        raw_new = mne.io.RawArray(raw_data, raw.info, verbose="WARNING")
        raw_new.set_annotations(raw.annotations)
        raw = raw_new
    return raw


def remove_around_gap(raw, gap_reject_s=0.05, gap_duration_s=0.075, eyes=None):
    """
    This function removes data points around gaps (i.e. nan data points). When a nan is encountered, the n ms of data
    before and after are removed
    :param raw:
    :param gap_duration_s:
    :param eyes:
    :return:
    """
    print("="*40)
    print("Removing data around gaps")
    # Convert the gap duration to samples:
    gap_reject_n = int(gap_reject_s * raw.info["sfreq"])
    gap_duration_n = int(gap_duration_s * raw.info["sfreq"])
    if eyes is None:
        eyes = ["L", "R"]
    # Loop through each eye:
    for eye in eyes:
        # Extract the data from this eye:
        data = np.squeeze(raw.copy().get_data(picks="{}Pupil".format(eye)))
        # Find the onset and offsets of each gap:
        nan_onset_inds = np.where(np.diff(np.isnan(data).astype(float)) == 1)[0]
        nan_offset_inds = np.where(np.diff(np.isnan(data).astype(float)) == -1)[0]
        if len(nan_onset_inds) != len(nan_offset_inds):
            print(len(nan_onset_inds))
            print(len(nan_offset_inds))
            raise ValueError("The number of nan onsets and offsets is not equal!")
        # Store the number of removed samples:
        n_removed = 0
        # Loop through each nan value:
        for i, onset_ind in enumerate(nan_onset_inds):
            if nan_offset_inds[i] < onset_ind:
                raise ValueError("The offset is before the onset!")
            if nan_offset_inds[i] - onset_ind >= gap_duration_n:
                # Remove the data around the onset and the offset:
                data[onset_ind - gap_reject_n:onset_ind] = np.nan
                data[nan_offset_inds[i]:nan_offset_inds[i] + gap_reject_n] = np.nan
                n_removed += gap_reject_n * 2
        # Display the proportion of removed samples:
        print("Removed {:2f}% samples".format((n_removed / data.shape[0]) * 100))
        # Add back to the mne raw object:
        raw_data = raw.get_data()
        # Extract the index of the channel:
        ch_ind = np.where(np.array(raw.ch_names) == "{}Pupil".format(eye))[0]
        raw_data[ch_ind, :] = data
        # Recreate the raw object:
        raw_new = mne.io.RawArray(raw_data, raw.info, verbose="WARNING")
        raw_new.set_annotations(raw.annotations)
        raw = raw_new
    return raw


def set_pupil_nans(raw, eyes=None):
    """
    This function sets impossible values to Nans. Impossible values are negative values or zeros:
    :param raw:
    :param eyes:
    :return:
    """
    print("="*40)
    print("Setting impossible values to Nan")
    if eyes is None:
        eyes = ["L", "R"]
    # Loop through each eye:
    for eye in eyes:
        # Extract the data from this eye:
        data = raw.copy().get_data(picks="{}Pupil".format(eye))
        # Count the proportion of zeros and negative values:
        prop_rej = np.sum(data <= 0) / data.size
        print("Proportion of impossible values for eye {}: {:2f}%".format(eye, prop_rej * 100))
        # Find the samples that are equal to 0:
        data[data == 0] = np.nan
        # Set the negative samples to Nan:
        data[data < 0] = np.nan
        # Add back to the mne raw object:
        raw_data = raw.get_data()
        # Extract the index of the channel:
        ch_ind = np.where(np.array(raw.ch_names) == "{}Pupil".format(eye))[0]
        raw_data[ch_ind, :] = data
        # Recreate the raw object:
        raw_new = mne.io.RawArray(raw_data, raw.info, verbose="WARNING")
        raw_new.set_annotations(raw.annotations)
        raw = raw_new
    return raw


def interp_nan(data):
    """

    :param data:
    :return:
    """
    ok = ~np.isnan(data)
    xp = ok.ravel().nonzero()[0]
    fp = data[~np.isnan(data)]
    x = np.isnan(data).ravel().nonzero()[0]
    data[np.isnan(data)] = np.interp(x, xp, fp)
    return data


def interpolate_pupil(raw, eyes=None):
    """

    :param raw:
    :param eyes:
    :return:
    """
    print("="*40)
    print("Interpolating pupil values")
    if eyes is None:
        eyes = ["L", "R"]
    for eye in eyes:
        # Extract the data from this eye:
        data = np.squeeze(raw.copy().get_data(picks="{}Pupil".format(eye)))
        # Compute the total proportion of data requiring interpolation:
        prop_interp = np.sum(np.isnan(data)) / data.size
        print("Proportion of data requiring interpolation for eye {}: {:2f}%".format(eye, prop_interp * 100))
        # Interpolate the nan:
        data_interp = interp_nan(data)
        # Add back to the mne raw object:
        raw_data = raw.get_data()
        # Extract the index of the channel:
        ch_ind = np.where(np.array(raw.ch_names) == "{}Pupil".format(eye))[0]
        raw_data[ch_ind, :] = data_interp
        # Recreate the raw object:
        raw_new = mne.io.RawArray(raw_data, raw.info, verbose="WARNING")
        raw_new.set_annotations(raw.annotations)
        raw = raw_new
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
    outliers_ind = np.where(data > thresh)
    return outliers_ind


def dilation_speed_rejection(raw, threshold_factor=4, eyes=None):
    print("="*40)
    print("Rejecting samples based on dilation speed")
    if eyes is None:
        eyes = ["L", "R"]

    # Loop through each eye:
    for eye in eyes:
        # Extract the pupil size from this eye:
        data = raw.copy().get_data(picks="{}Pupil".format(eye))
        # Compute the dilation speed:
        dilation_speed = dilation_filter(np.squeeze(data), raw.times, axis=-1)
        # Extract the index of the outliers:
        outliers_ind = mad_outliers_ind(dilation_speed, threshold_factor=threshold_factor, axis=0)
        # Display some information about the proportion of outliers that were found:
        print("For {} eye: ".format(eye))
        print("{:2f}% of rejected samples ({} out of {})".format((outliers_ind[0].shape[0] / data.shape[-1]) *
                                                                 100,
                                                                 outliers_ind[0].shape[0], data.shape[-1]))
        # Replace the outliers by nan.
        data[0, outliers_ind[0]] = np.nan
        # Add back to the mne raw object:
        raw_data = raw.get_data()
        # Extract the index of the channel:
        ch_ind = np.where(np.array(raw.ch_names) == "{}Pupil".format(eye))[0]
        raw_data[ch_ind, :] = data
        # Recreate the raw object:
        raw_new = mne.io.RawArray(raw_data, raw.info, verbose="WARNING")
        raw_new.set_annotations(raw.annotations)
        raw = raw_new
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
        eyes = ["L", "R"]

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
    channels.extend(["".join([eye, description]) for eye in eyes])
    info = mne.create_info(channels, ch_types=["eeg"] * len(channels), sfreq=raw.info["sfreq"])
    raw_new = mne.io.RawArray(data, info, verbose="WARNING")
    raw_new.set_annotations(raw.annotations)
    return raw_new
