from mne.baseline import rescale
import numpy as np
import pandas as pd
from mne._fiff.pick import _picks_to_idx
from scipy.ndimage import gaussian_filter
from mne.stats import permutation_cluster_1samp_test


def max_percentage_index(data, thresh_percent):
    """
    Find the index at which a time series reaches a certain percentage of its peak value.
    :param data: (numpy array) containing the time series
    :param thresh_percent: (float or int) percentage of peak value
    :return ind: index of the first time the percentage of peak value is reached
    """
    if not (0 <= thresh_percent <= 100):
        raise ValueError("Percentage threshold must be between 0 and 100.")

    peak_value = np.max(data)
    threshold_value = peak_value * (thresh_percent / 100)

    # Find the first index where the value is greater than or equal to the threshold
    ind = np.argmax(data >= threshold_value)

    # If the threshold is not reached, return None or raise an exception, depending on your preference.
    if data[ind] < threshold_value:
        return None

    return ind, threshold_value


def cluster_1samp_across_sub(subjects_epochs, conditions, n_permutations=1024, threshold=None, tail=0):
    """
    This function applies the permutation_cluster_1samp_test from MNE, taking in a dictionary containing the epochs
    of each subject. It will then average across trials within the condition of interest (i.e. create evoked) and
    compute the within sample difference between the two conditions. It is a bit overkill to package it into 1 function
    but because this is something we want to do over and over again, makes the code easier to navigate.
    :param subjects_epochs:
    :param conditions:
    :param n_permutations:
    :param threshold:
    :param tail:
    :return:
    """
    evks = {cond: [] for cond in conditions}
    # Loop through each subject:
    for sub in subjects_epochs.keys():
        # Loop through each relevant condition:
        for cond in conditions:
            # Average the data across both eyes:
            data = np.nanmean(subjects_epochs[sub].copy()[cond], axis=1)
            # Remove any trials containing Nan:
            data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
            # Remove any trials containing Nan:
            data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
            evks[cond].append(np.mean(data, axis=0))

    # Convert each condition data to a numpy array:
    evks = {cond: np.array(evks[cond]) for cond in conditions}
    # Compute the evoked difference between task relevance within each subject:
    evks_diff = np.array([evks[conditions[0]][i, :] - evks[conditions[1]][i, :]
                          for i in range(evks[conditions[0]].shape[0])])
    # Perform a cluster based permutation ttest:
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        evks_diff,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=tail,
        adjacency=None,
        out_type="mask",
        verbose=True,
    )
    return evks, evks_diff, T_obs, clusters, cluster_p_values, H0


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
                          picks=picks, n_jobs=n_jobs, verbose="WARNING")

    return None
