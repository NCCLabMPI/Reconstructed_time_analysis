from mne.baseline import rescale
import numpy as np
from scipy.ndimage import gaussian_filter
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import zscore
from math import atan2, degrees


def deg_to_pix(size_deg, distance_cm, screen_size_cm, screen_res):
    """
    This function converts a given degrees of visual angle to pixels
    """
    # Compute conversion factor by calculating how many degrees of visual angle a pixel is:
    deg_per_pixel = np.mean([degrees(atan2(.5 * screen_size_cm[0], distance_cm)) / (.5 * screen_res[0]),
                             degrees(atan2(.5 * screen_size_cm[1], distance_cm)) / (.5 * screen_res[1])])
    # Convert the size in degrees to size in centimeters:
    size_pix = size_deg / deg_per_pixel

    return size_pix


def moving_average(data, window_size, axis=-1, overlapping=False):
    """
    This function performs moving average of multidimensional arrays. Shouthout to
    https://github.com/NGeorgescu/python-moving-average and
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/43200476#43200476 for the inspiration
    :param data: (numpy array) data on which to perform the moving average
    :param window_size: (int) number of samples in the moving average
    :param axis: (int) axis along which to perform the moving average
    :param overlapping: (boolean) whether or not to perform the moving average in an overlapping fashion or not. If
    true, fully overlapping, if false, none overelapping (i.e. moving from window size to window size)
    data = [1, 2, 3, 4, 5, 6]
    overlapping:
    mean(1, 2, 3), mean(2, 3, 4), mean(3, 4, 5)...
    non-overlapping:
    mean(1, 2, 3), mean(4, 5, 6)...
    :return:
    mvavg: (np array) data following moving average. Note that the dimension will have changed compared to the original
    matrix
    """
    if overlapping:
        # Bringing the axis over which to average to first position to have everything happening on first dim thereafter
        data_swap = data.swapaxes(0, axis)
        # Compute cumsum and divide by the window size:
        data_cum_sum = np.cumsum(data_swap, axis=0) / window_size
        # Adding a row of zeros to the first dimension:
        if data_cum_sum.ndim > 1:
            # Add zeros in the first dim:
            data_cum_sum = np.vstack([[0 * data_cum_sum[0]], data_cum_sum])
        else:  # if array is only 1 dim:
            data_cum_sum = np.array([0, *data_cum_sum])
        # Compute the moving average by subtracting the every second row of the data by every other second:
        mvavg = data_cum_sum[window_size:] - data_cum_sum[:-window_size]
        # Bringing back the axes to the original dimension:
        return np.swapaxes(mvavg, 0, axis)
    else:
        # Bringing the axis over which to average to first position to have everything happening on first dim thereafter
        data_swap = data.swapaxes(0, axis)
        # Handling higher dimensions:
        data_dim = data_swap[:int(len(data_swap) / window_size) * window_size]
        # Reshape the data, such that along the 1st dimension, we have the n samples of the independent bins:
        data_reshape = data_dim.reshape(int(len(data_swap) / window_size), window_size, *data_swap.shape[1:])
        # Compute the moving avereage along the 1dim (this is how it should be done based on the reshape above:
        mvavg = data_reshape.mean(axis=1)
        return mvavg.swapaxes(0, axis)


def mvavg(data, window_ms, sfreq):
    """
    This function computes a non-overlapping moving average on mne epochs object.
    :param data: (mne epochs object) epochs to smooth
    :param window_ms: (int) window size in milliseconds
    :param sfreq: (int)
    :return:
    epochs: the smoothed mne epochs object
    """
    n_samples = int(np.floor(window_ms * sfreq / 1000))
    mvavg_data = moving_average(data, n_samples, axis=-1, overlapping=False)
    return mvavg_data


def beh_exclusion(data_df):
    """
    This function performs the exclusion criterion reported in the paper. Packaged in a function such that we can use
    it in various places:
    :param data_df:
    :return:
    """
    trial_orig = list(data_df.index)
    # 1. Remove the trials with wrong visual responses:
    data_df_clean = data_df[data_df["trial_response_vis"] != "fa"]
    # 2. Remove the trials with wrong auditory responses:
    data_df_clean = data_df_clean[data_df_clean["trial_accuracy_aud"] != 0]
    # 3. Remove trials where the visual stimuli were responded second:
    data_df_clean = data_df_clean[data_df_clean["trial_second_button_press"] != 1]
    # 4. Remove trials in which the participants responded to the auditory stimulus in less than 100ms and more than
    # 1260ms:
    data_df_clean = data_df_clean[(data_df_clean["RT_aud"] >= 0.1) & (data_df_clean["RT_aud"] <= 1.260)]
    # Fetch the removed indices:
    trial_final = list(data_df_clean.index)
    rejected_trials = [trial for trial in trial_orig if trial not in trial_final]

    return rejected_trials


def reject_bad_epochs(epochs, baseline_window=None, z_thresh=2, eyes=None, remove_blinks=True, blinks_window=None,
                      remove_nan=False, exlude_beh=True):
    """
    This function rejects epochs based on the zscore of the baseline. For some trials, there may be artifacts
    in the baseline, in which case baseline correction will spread the artifact. Such epochs are discarded.
    :param epochs:
    :param baseline_window:
    :param z_thresh:
    :param eyes:
    :param remove_blinks:
    :param blinks_window:
    :param remove_nan:
    :param exlude_beh:
    return:
        - epochs:
        - inds
    """
    print("=" * 40)
    print("Finding epochs with artifactual baseline")
    if eyes is None:
        eyes = ["left", "right"]
    if baseline_window is None:
        baseline_window = [None, 0]
    if blinks_window is None:
        blinks_window = [0, 0.5]
    # Get the initial number of trials:
    ntrials_orig = len(epochs)

    if exlude_beh:
        inds = beh_exclusion(epochs.metadata.copy().reset_index(drop=True))
        if len(inds) > 0:
            # Drop these epochs:
            epochs.drop(inds, reason="baseline_artifact", verbose="ERROR")
        print("{} out of {} ({:.2f}%) trials were rejected based on behavior.".format(len(inds), ntrials_orig,
                                                                               (len(inds) / ntrials_orig) * 100))
    # Extract the data:
    if z_thresh is not None:
        baseline_data = epochs.copy().crop(tmin=baseline_window[0],
                                           tmax=baseline_window[1]).get_data(picks=["_".join(["pupil", eye])
                                                                                    for eye in eyes])
        # Compute the average across eyes and time:
        baseline_avg = np.mean(np.mean(baseline_data, axis=1), axis=1)
        # Z score:
        baseline_zscore = zscore(baseline_avg, nan_policy='omit')
        # Find the epochs indices that exceed the threshold:
        inds = np.where(np.abs(baseline_zscore) > z_thresh)[0]
        if len(inds) > 0:
            # Drop these epochs:
            epochs.drop(inds, reason="baseline_artifact", verbose="ERROR")
        # Print the proportion of dropped epochs:
        print("{} out of {} ({:.2f}%) trials had artifact in baseline.".format(len(inds), len(baseline_zscore),
                                                                               (len(inds) / len(
                                                                                   baseline_zscore)) * 100))
    if remove_blinks:
        # Extract the blinks channels:
        blink_data = np.squeeze(epochs.copy().crop(tmin=blinks_window[0],
                                                   tmax=blinks_window[1]).get_data(picks=["_".join(["BAD_blink", eye])
                                                                                          for eye in eyes]))
        if blink_data.shape[1] == 2:
            # Combine both eyes data:
            blink_data = np.logical_and(blink_data[:, 0, :], blink_data[:, 1, :]).astype(float)
        # Find the trials in which we have blinks:
        blink_inds = np.where(np.any(blink_data, axis=1))[0]
        if len(blink_inds) > 0:
            # Drop these epochs:
            epochs.drop(blink_inds, reason="blinks", verbose="ERROR")
        # Print the proportion of dropped epochs:
        print("{} out of {} ({:.2f}%) trials had blinks within.".format(len(blink_inds), blink_data.shape[0],
                                                                        (len(blink_inds) /
                                                                            blink_data.shape[0]) * 100,
                                                                        blinks_window))
    if remove_nan:
        data = epochs.get_data(copy=True)
        nan_inds = np.unique(np.where(np.isnan(data))[0])
        if len(nan_inds) > 0:
            # Drop these epochs:
            epochs.drop(nan_inds, reason="NaN", verbose="ERROR")
        # Print the proportion of dropped epochs:
        print("{} out of {} ({:.2f}%) trials had NaN.".format(len(nan_inds), data.shape[0],
                                                                        (len(nan_inds) /
                                                                            data.shape[0]) * 100,
                                                                        blinks_window))
    ntrials_final = len(epochs)

    return epochs, 1 - ntrials_final/ntrials_orig


def max_percentage_index(data, thresh_percent):
    """
    Find the index at which a time series reaches a certain percentage of its peak value.
    :param data: (numpy array) containing the time series
    :param thresh_percent: (float or int) percentage of peak value
    :return ind: index of the first time the percentage of peak value is reached
    """
    if not (0 <= thresh_percent <= 100):
        raise ValueError("Percentage threshold must be between 0 and 100.")

    # If all the data are negative, correct by adding the minimum:
    if np.max(data) < 0:
        data = data + np.abs(np.min(data))
        threshold_value = np.max(data) * (thresh_percent / 100)
    else:
        threshold_value = np.max(data) * (thresh_percent / 100)

    # Find the first index where the value is greater than or equal to the threshold
    ind = np.argmax(data >= threshold_value)

    # If the threshold is not reached, return None or raise an exception, depending on your preference.
    if data[ind] < threshold_value:
        return None

    return ind, threshold_value


def cluster_1samp_across_sub(subjects_epochs, conditions, n_permutations=1024, threshold=None, tail=0,
                             downsample=False):
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
    :param downsample:
    :return:
    """

    evks = {cond: [] for cond in conditions}
    # Loop through each subject:
    for sub in subjects_epochs.keys():
        # Loop through each relevant condition:
        cond_data = {cond: [] for cond in conditions}
        conditions_counts = []
        for cond in conditions:
            # Average the data across both eyes:
            data = np.nanmean(subjects_epochs[sub].copy()[cond], axis=1)
            # Remove any trials containing Nan:
            data = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isnan(data[i, :]))])
            # Remove any trials containing Nan:
            cond_data[cond] = np.array([data[i, :] for i in range(data.shape[0]) if not any(np.isinf(data[i, :]))])
            conditions_counts.append(cond_data[cond].shape[0])
        # Check the counts of each condition:
        if conditions_counts[0] != conditions_counts[1] and downsample:
            min_counts = min(conditions_counts)
            for cond in conditions:
                evks[cond].append(np.mean(cond_data[cond][np.random.choice(cond_data[cond].shape[0], min_counts), :],
                                          axis=0))
        else:
            for cond in conditions:
                evks[cond].append(np.mean(cond_data[cond], axis=0))

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


def generate_gaze_map(epochs, height, width, sigma=20, eyes=None):
    """
    This function takes in the eyetracker data in the mne epochs object and generates gaze maps. This is highly inspired
    from this code https://github.com/mne-tools/mne-python/blob/main/mne/viz/eyetracking/heatmap.py#L13-L104
    :param epochs:
    :param height:
    :param width:
    :param sigma:
    :param eyes:
    :return:
    """
    if eyes is None:
        eyes = ["left", "right"]
    xpicks = []
    ypicks = []
    for eye in eyes:
        xpicks.append("xpos_{}".format(eye))
        ypicks.append("ypos_{}".format(eye))
    x_data = epochs.get_data(picks=xpicks, copy=True)
    y_data = epochs.get_data(picks=ypicks, copy=True)
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
