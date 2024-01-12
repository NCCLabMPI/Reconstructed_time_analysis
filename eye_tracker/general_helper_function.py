from mne.baseline import rescale
import numpy as np
import pandas as pd
from mne._fiff.pick import _picks_to_idx
from scipy.ndimage import gaussian_filter


def generate_gaze_map(epochs, height, width, sigma=20):
    """
    This function takes in the eyetracker data in the mne epochs object and generates gaze maps. This is highly inspired
    from this code https://github.com/mne-tools/mne-python/blob/main/mne/viz/eyetracking/heatmap.py#L13-L104
    :param epochs:
    :param height:
    :param width:
    :param sigma:
    :return:
    """

    pos_picks = _picks_to_idx(epochs.info, "eyegaze")
    gaze_data = epochs.get_data(picks=pos_picks)
    gaze_ch_loc = np.array([epochs.info["chs"][idx]["loc"] for idx in pos_picks])
    x_data = gaze_data[:, np.where(gaze_ch_loc[:, 4] == -1)[0], :]
    y_data = gaze_data[:, np.where(gaze_ch_loc[:, 4] == 1)[0], :]
    if x_data.shape[1] > 1:  # binocular recording. Average across eyes
        x_data = np.nanmean(x_data, axis=1)  # shape (n_epochs, n_samples)
        y_data = np.nanmean(y_data, axis=1)
    canvas = np.vstack((x_data.flatten(), y_data.flatten()))  # shape (2, n_samples)

    hist, _, _ = np.histogram2d(
        canvas[1, :],
        canvas[0, :],
        bins=(height, width),
        range=[[0, height], [0, width]],
    )

    # Convert density from samples to seconds
    hist /= epochs.info["sfreq"]
    # Smooth the heatmap
    if sigma:
        hist = gaussian_filter(hist, sigma=sigma)

    return hist

def cousineau_morey_correction(data, within_col, between_col, dependent_var):
    """
    This scripts computes the Cousineau Morey correction of data in a dataframe. From each subject and condition,
    the subject mean (across conditions) gets subtracted, and then the grand average gets added (
    :param data:
    :param within_col:
    :param between_col:
    :param dependent_var:
    :return:
    """
    # Compute the subject-specific means:
    subjects_mean = data.groupby([within_col])[dependent_var].mean().reset_index()
    # Compute the grand mean:
    grand_mean = np.mean(subjects_mean[dependent_var].to_numpy())
    # Create a new DataFrame to store the corrected data
    corrected_data = data.copy()

    for subject in data[within_col].unique():
        for between_cond in data[between_col].unique():
            subject_mean = subjects_mean[subjects_mean[within_col] == subject][dependent_var].values
            single_trial_data = data[(data[within_col] == subject) &
                                     (data[between_col] == between_cond)][dependent_var].to_numpy()

            # Apply the Cousineau-Morey correction and store it in the corrected_data DataFrame
            idx = (data[within_col] == subject) & (data[between_col] == between_cond)
            corrected_data.loc[idx, dependent_var] = (single_trial_data - subject_mean) + grand_mean

    return corrected_data


def get_event_ts(data, times):
    """
    This function returns the time stamp of a particular event. This converts time series of zeros and ones to
    time stamps of the beginning of each event.
    :param data: data containing the blinking information. The data should in the format of trials x time with 1 where
    a subject was blinking and 0 otherwise
    :param times: (array) the time vector
    :return: a list of arrays containing the onset of the blinks for each trial
    """
    blinking_onsets = []
    n_blinks_per_trial = []
    data_onset = np.diff(data, axis=-1)
    for trial in range(data.shape[0]):
        blinking_onsets.append(times[np.where(data_onset[trial, :] == 1)[0]])
        n_blinks_per_trial.append(len(blinking_onsets[-1]))
    return blinking_onsets


def baseline_scaling(epochs, correction_method="ratio", baseline=(None, 0), picks=None, n_jobs=1):
    """
    This function performs baseline correction on the data. The default is to compute the mean over the entire baseline
    and dividing each data points in the entire epochs by it. Another option is to substract baseline from each time
    point
    :param epochs: (mne epochs object) epochs on which to perform the baseline correction
    :param correction_method: (string) options to do the baseline correction. Options are:
        mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        Perform baseline correction by
        - subtracting the mean of baseline values ('mean')
        - dividing by the mean of baseline values ('ratio')
        - dividing by the mean of baseline values and taking the log
          ('logratio')
        - subtracting the mean of baseline values followed by dividing by
          the mean of baseline values ('percent')
        - subtracting the mean of baseline values and dividing by the
          standard deviation of baseline values ('zscore')
        - dividing by the mean of baseline values, taking the log, and
          dividing by the standard deviation of log baseline values
          ('zlogratio')
          source: https://github.com/mne-tools/mne-python/blob/main/mne/baseline.py
    :param baseline: (tuple) which bit to take as the baseline
    :param picks: (None or list of int or list of strings) indices or names of the channels on which to perform the
    correction. If none, all channels are used
    :param n_jobs: (int) number of jobs to use to preprocessing the function. Can be ran in parallel
    :return: none, the data are modified in place
    """
    epochs.apply_function(rescale, times=epochs.times, baseline=baseline, mode=correction_method,
                          picks=picks, n_jobs=n_jobs, )

    return None

