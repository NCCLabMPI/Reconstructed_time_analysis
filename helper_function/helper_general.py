import math
import matplotlib
from mne.baseline import rescale
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import zscore
from math import atan2, degrees
import mne
from mne_bids import convert_montage_to_mri
from matplotlib import colormaps


def get_cmap_rgb_values(values, cmap=None, center=None):
    """
    This function takes in a list of values and returns a set of RGB values mapping onto a specified color bar. If a
    midpoint is set, the color bar will be normalized accordingly.
    :param values: (list of floats) list of values for which to obtain a color map
    :param cmap: (string) name of the colormap
    :param center: (float) values on which to center the colormap
    return: colors (list of rgb triplets) color for each passed value
    """
    if cmap is None:
        cmap = "RdYlBu_r"
    if center is None:
        center = np.mean([min(values), max(values)])
    # Create the normalization function:
    norm = matplotlib.colors.TwoSlopeNorm(vmin=min(values), vcenter=center, vmax=max(values))
    colormap = colormaps.get_cmap(cmap)
    colors = [colormap(norm(value)) for value in values]

    return colors


def create_super_subject(epochs_dict, targets_col, n_trials=80):
    # Extract all targets:
    targets_id = [list(epochs_dict[sub].metadata[targets_col].unique()) for sub in epochs_dict.keys()]
    targets_id = list(set([item for items in targets_id for item in items]))
    # Loop through each target to get counts per subjects:
    target_counts = {}
    for sub in epochs_dict.keys():
        sub_counts = []
        for target in targets_id:
            sub_counts.append(np.sum(epochs_dict[sub].metadata[targets_col] == target))
        target_counts[sub] = sub_counts
    # Exclude any subjects with less than the set number of trials in each condition:
    valid_subjects = [sub for sub in target_counts.keys() if all([cts >= n_trials for cts in target_counts[sub]])]
    # Delete discarded subjects:
    for sub in epochs_dict.keys():
        if sub not in valid_subjects:
            print("Discarding sub-{}, not enough trials!".format(sub))
            del epochs_dict[sub]

    super_sub_data = []
    super_sub_labels = []
    # Loop through each subject:
    for sub in valid_subjects:
        sub_labels = []
        sub_data = []
        # Loop through each target:
        for target in targets_id:
            # Extract the data:
            data = epochs_dict[sub].copy()[target].get_data(copy=False)
            if data.shape[0] > n_trials:
                data = data[np.random.choice(data.shape[0], n_trials, replace=False), :, :]
            elif data.shape[0] < n_trials:
                raise Exception("sub-{} has less than {} trials, which is not supposed to be possible!")
            sub_data.append(data)
            sub_labels.extend([target] * n_trials)
        # Concatenate the subject data:
        super_sub_data.append(np.concatenate(sub_data, axis=0))
        super_sub_labels.append(sub_labels)
    assert all([lbls == super_sub_labels[0] for lbls in super_sub_labels]), \
        "The labels are misaligned across subjects!"
    # Concatenate the data:
    super_sub_data = np.concatenate(super_sub_data, axis=1)
    super_sub_targets = np.array(super_sub_labels[0])

    return super_sub_data, super_sub_targets


def get_roi_channels(montage, subject, rois, fs_dir, aseg="aparc+aseg", dist=2):
    """
    This function takes in a list of channels and returns only those which are in a particular set of ROIs.
    :param montage: (list of string) list of channels
    :param subject: (list of string) list of channels
    :param rois: (list of string) list of channels
    :param fs_dir:
    :param aseg:
    :param dist:
    """

    # Extract the labels of each channel:
    labels, colors = mne.get_montage_volume_labels(montage, subject, fs_dir, aseg=aseg, dist=dist)

    # Keep each channel found within the list:
    roi_channels = []
    for ch in labels.keys():
        for lbl in labels[ch]:
            if lbl in rois:
                roi_channels.append(ch)
    return list(set(roi_channels))


def extract_first_bout(times, data, threshold, min_duration):
    """

    :param times:
    :param data:
    :param threshold:
    :param min_duration:
    :return:
    """
    # Binarize the data with threshold:
    data_bin = data < threshold
    if np.any(data_bin):
        # Compute the diff:
        data_diff = np.diff(data_bin.astype(float), axis=0)
        # Find onsets and offsets:
        onsets = times[np.where(data_diff == 1)[0]]
        offset_ind = np.where(data_diff == -1)[0]
        for i, onset in enumerate(onsets):
            if i == len(offset_ind):
                offset = times[-1]
            else:
                offset = times[offset_ind[i]]
            if offset - onset > min_duration:
                return onset, offset

        return None, None
    else:
        return None, None


def create_mni_montage(channels, bids_path, fs_dir, fsaverage_dir):
    """
    This function fetches the mni coordinates of a set of channels. Importantly, the channels must
    consist of a string with the subject identifier and the channel identifier separated by a minus,
    like: SF102-G1. This ensures that the channels positions can be fecthed from the right subject
    folder. This is specific to iEEG for which we have different channels in each patient which
    may have the same name.
    :param channels: (list) name of the channels for whom to fetch the MNI coordinates. Must contain
    the subject identifier as well as the channel identifier, like SF102-G1.
    :param bids_path: (mne-bids bidsPATH object) contains all the information to fetch the coordinates.
    :param fs_dir: (string or pathlib path object) path to the free surfer root folder containing the fsaverage
    :param fsaverage_dir: (string or pathlib path object) path to the free surfer root folder containing the fsaverage
    :return: info (mne info object) mne info object with the channels info, including position in MNI space
    """
    from mne_bids import BIDSPath
    from mne.transforms import apply_trans
    import environment_variables as ev
    # First, extract the name of each subject present in the channels list:
    subjects = list(set([channel.split('-')[0] for channel in channels]))
    # Prepare a dictionary:
    mni_coords = {}
    channels_types = {}
    for subject in subjects:
        # Extract this participant's channels:
        subject_channels = [channel.split('-')[1] for channel in channels
                            if channel.split('-')[0] == subject]
        # Create the path to this particular subject:
        subject_path = BIDSPath(root=bids_path.root, subject=subject,
                                session=bids_path.session,
                                datatype=bids_path.datatype,
                                task=bids_path.task)
        # Create the name of the mni file coordinates:
        coordinates_file = 'sub-{}_ses-{}_space-ACPC_electrodes.tsv'.format(subject,
                                                                            subject_path.session)
        channel_file = 'sub-{}_ses-{}_task-{}_channels.tsv'.format(subject, subject_path.session, bids_path.task)
        # Load the coordinates:
        coordinates_df = pd.read_csv(Path(subject_path.directory, coordinates_file), sep='\t')
        channels_df = pd.read_csv(Path(subject_path.directory, channel_file), sep='\t')

        # Get the position:
        position = coordinates_df.loc[coordinates_df['name'].isin(
            subject_channels), ['x', 'y', 'z']].to_numpy()
        # Get the types of the channels:
        subject_channel_type = channels_df.loc[channels_df['name'].isin(subject_channels),
        ['name', 'type']].set_index('name').to_dict()['type']
        subject_channel_type = {"-".join([subject, ch]): subject_channel_type[ch] for ch in subject_channel_type.keys()}
        channels_types.update(subject_channel_type)
        # Create the montage:
        montage = mne.channels.make_dig_montage(ch_pos=dict(zip(["-".join([subject, ch]) for ch in subject_channels],
                                                                position)),
                                                coord_frame="ras")
        # we need to go from scanner RAS back to surface RAS (requires recon-all)
        convert_montage_to_mri(montage, "sub-" + subject, subjects_dir=ev.fs_directory)
        # Add estimated fiducials
        montage.add_estimated_fiducials("sub-" + subject, fs_dir)
        # Fetch the transformation from mri -> mni
        mri_mni_trans = mne.read_talxfm("sub-" + subject, fs_dir)
        # Extract the channel position from the montage:
        ch_pos = montage.get_positions()['ch_pos']
        # Apply affine transformation to each:
        for ind, ch in enumerate(ch_pos.keys()):
            mni_coords[ch] = apply_trans(mri_mni_trans, ch_pos[ch] * 1000) / 1000

    # Create the montage:
    montage = mne.channels.make_dig_montage(ch_pos=mni_coords,
                                            coord_frame='mni_tal')
    # Make sure that the channel types are in lower case:
    channels_types = {ch: channels_types[ch].lower() for ch in channels_types}
    # Project the montage to the surface:
    montage = project_montage_to_surf(montage, channels_types, "fsaverage", fsaverage_dir)
    # Add the MNI fiducials
    montage.add_mni_fiducials(fsaverage_dir)
    # In mne-python, plotting electrodes on the brain requires some additional info about the channels:
    info = mne.create_info(ch_names=channels, ch_types=list(channels_types.values()), sfreq=100)
    # Add the montage:
    info.set_montage(montage)

    return info


def compute_dprime(hits, misses, false_alarms, correct_rejections):
    """
    This function takes in hits, misses, false alarms and correct rejections to compute d'
    :param hits:
    :param misses:
    :param false_alarms:
    :param correct_rejections:
    :return: dprime, c and beta
    """
    z = norm.ppf
    # Calculate hit rate and false alarm rate
    hit_rate = hits / (hits + misses)
    fa_rate = false_alarms / (false_alarms + correct_rejections)

    # Ensure rates are not 0 or 1 to avoid infinite values in Z-transform
    hit_rate = min(0.99, max(0.01, hit_rate))
    fa_rate = min(0.99, max(0.01, fa_rate))

    # Compute the dprime as the difference between the zscores of the hit and false alarms rates:
    dprime = z(hit_rate) - z(fa_rate)
    # Compute the beta:
    beta = math.exp((z(fa_rate) ** 2 - z(hit_rate) ** 2) / 2)

    return dprime, beta


def format_drop_logs(drop_logs_dict):
    """
    This function converts a dictionary of mne python drop logs into a data frame. The dictionary must be of the
    shape "sub": drop log
    :param drop_logs_dict:
    :return:
    """
    trial_rej_reas = [list(set(drop_logs_dict[sub])) for sub in drop_logs_dict.keys()]
    trial_rej_reas = list(set([item for items in trial_rej_reas for item in items if len(item) > 0]))
    drop_logs_df = []
    for sub in drop_logs_dict.keys():
        drop_log_counts = {"sub": sub}
        drop_log_list = [drop[0] for drop in drop_logs_dict[sub] if len(drop) > 0]
        # Loop through each reason to get the counts:
        for reas in trial_rej_reas:
            drop_log_counts[reas] = np.sum(np.array(drop_log_list) == reas) / len(drop_logs_dict[sub])
        drop_log_counts["total"] = len(drop_log_list) / len(drop_logs_dict[sub])
        drop_logs_df.append(pd.DataFrame(drop_log_counts, index=[0]))
    return pd.concat(drop_logs_df).reset_index(drop=True)


def load_beh_data(bids_root, subjects, fn_template, session='1', task='prp', do_trial_exclusion=True):
    """
    This function loads the behavioral data
    :param bids_root:
    :param subjects:
    :param fn_template:
    :param session:
    :param task:
    :param do_trial_exclusion:
    :return:
    """
    # Load the data:
    subjects_data = []
    if do_trial_exclusion:
        prop_rejected = []
        for sub in subjects:
            behavioral_file = Path(bids_root, "sub-" + sub, "ses-" + session, "beh",
                                   fn_template.format(sub, session, task))
            # Load the file:
            subject_data = pd.read_csv(behavioral_file, sep=",")
            # Add the subject ID:
            subject_data["subject"] = sub
            # Apply trial rejection:
            rej_ind = beh_exclusion(subject_data)
            prop_rej = len(rej_ind) / subject_data.shape[0]
            subject_data = subject_data.drop(rej_ind)
            # Append to the rest of the subject
            subjects_data.append(subject_data.reset_index(drop=True))
            # Print the proportion of trials that were discarded:
            print("Subject {} - {:.2f}% trials were discarded".format(sub, prop_rej * 100))
            prop_rejected.append(prop_rej)

        print("The mean proportion of rejected trials is {:.2f}% +- {:.2f}".format(np.mean(prop_rejected) * 100,
                                                                                   np.std(prop_rejected) * 100))
    else:
        for sub in subjects:
            behavioral_file = Path(bids_root, "sub-" + sub, "ses-" + session, "beh",
                                   fn_template.format(sub, session, task))
            # Load the file:
            subject_data = pd.read_csv(behavioral_file, sep=",")
            # Add the subject ID:
            subject_data["subject"] = sub
            # Append to the rest of the subjects:
            subjects_data.append(subject_data.reset_index(drop=True))
    return pd.concat(subjects_data).reset_index(drop=True)


def equate_epochs_events(epochs_list):
    """
    This function equates the events ID of different epochs. In some cases, different epochs may have the same
    events description but different identifier associated to it. Alternatively, some epochs may not have all the
    events within them, leading to the dictionaries being different. In this function, the dictionaries gets equated
    such that all events from all epochs are combined to get each unique event across all and given the same identifier
    Note that it is a bit risky as it touches the trial labels, so make sure you really understand well why that is
    needed before using!
    :param epochs_list: (list of mne epochs object) contains the epochs that have to be equated
    :return:
        - list: lost of mne epochs object. This function operates in place!
    """
    # Extract the events description in each epoch:
    epochs_events = [epo.event_id for epo in epochs_list]
    # Check whether all the epochs have the same events
    if not all([evts == epochs_events[0] for evts in epochs_events]):
        print("WARNING: The epochs do not have the same events! The events dictionaries will be updated!")

        # Extract all the descriptions across all epochs:
        evts_names = [list(epo_evts.keys()) for epo_evts in epochs_events]
        evts_names = list(set([item for items in evts_names for item in items]))  # Flatten the list

        # Create the new events dictionary:
        new_evts = dict(zip(evts_names, list(range(len(evts_names)))))
        trial_conditions_bfre = []
        for i in epochs_list[0].events[:, 2]:
            trial_conditions_bfre.append([key for key in epochs_list[0].event_id.keys()
                                     if epochs_list[0].event_id[key] == i])

        # Loop through each epochs:
        for epo in epochs_list:
            epo_new_events = epo.events.copy()
            epo_new_events[:, 2] = np.array([np.nan] * epo_new_events.shape[0])
            for evts in new_evts.keys():
                if evts in epo.event_id.keys():
                    # Extract the epoch id:
                    evt_id = epo.event_id[evts]
                    evt_ind = np.where(epo.events[:, 2] == evt_id)[0]
                    if len(evt_ind) > 0:
                        epo_new_events[evt_ind, 2] = new_evts[evts]
            # Replace the event dictionary:
            epo.event_id = new_evts
            epo.events = epo_new_events
        trial_conditions_after = []
        for i in epochs_list[0].events[:, 2]:
            trial_conditions_after.append([key for key in epochs_list[0].event_id.keys()
                                          if epochs_list[0].event_id[key] == i])
        assert np.all([trial_conditions_after[i] == cond for i, cond in enumerate(trial_conditions_bfre)]), \
            "The trial description got messed up!!!!"
    return epochs_list


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
    # 1. Remove all trials with no auditory responses:
    data_df_clean = data_df[data_df["RT_aud"].notna()]
    # 2. Remove each trial in which the reaction time exceeds the max reaction time:
    # 2.1. Calculate the max threshold:
    max_win_trials = data_df_clean[(data_df_clean["SOA_lock"] == "offset") &
                                   (data_df_clean["duration"].to_numpy().astype(float) == 1.5) &
                                   (data_df_clean["SOA"].to_numpy().astype(float) == 0.466)]
    max_rt_aud = np.mean(max_win_trials["stim_jit"].to_numpy() + 2 - (1.5 + 0.466))
    # 2.2. Remove all trials with RT higher than that:
    data_df_clean = data_df_clean[data_df_clean["RT_aud"] <= max_rt_aud]
    # 3. Remove trials where the RT is lower than threshold:
    data_df_clean = data_df_clean[data_df_clean["RT_aud"] >= 0.1]
    # 4. Remove trials with false alarm to the visual stimulus:
    data_df_clean = data_df_clean[data_df_clean["trial_response_vis"] != "fa"]
    # 5. Remove incorrect auditory responses:
    data_df_clean = data_df_clean[data_df_clean["trial_accuracy_aud"] == 1]
    # Fetch the removed indices:
    trial_final = list(data_df_clean.index)
    rejected_trials = [trial for trial in trial_orig if trial not in trial_final]

    return rejected_trials


def reject_bad_epochs(epochs, baseline_window=None, z_thresh=2, eyes=None, remove_blinks=True, blinks_window=None,
                      remove_nan=False, exlude_beh=True, events_bound_blinks=True):
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
    :param events_bound_blinks:
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
        # Events bound blinks detects blinks that occur within a specified time window around the events of interest.
        # In our experiment, the critical visual events are the onset and offset of visual stimuli, depending on the
        # tone locking.
        if events_bound_blinks:
            metadata = epochs.metadata.copy().reset_index(drop=True)
            blink_inds = []
            # Find blinks around the onset of the visual events:
            onset_epochs = epochs["onset"]
            trial_inds = list(metadata[metadata["SOA_lock"] == "onset"].index)
            # Extract the data:
            blink_data = np.squeeze(onset_epochs.copy().crop(tmin=blinks_window[0],
                                                             tmax=blinks_window[1]).get_data(
                picks=["_".join(["blink", eye])
                       for eye in eyes]))
            if blink_data.shape[1] == 2:
                # Combine both eyes data:
                blink_data = np.logical_and(blink_data[:, 0, :], blink_data[:, 1, :]).astype(float)
            # Find the trials in which we have blinks:
            blink_inds.extend([trial_inds[i] for i in np.where(np.any(blink_data, axis=1))[0]])

            # For the offset locked trials, this needs to be done separately for each stimulus duration:
            for dur in epochs.metadata["duration"].unique():
                epo = epochs["/".join(["offset", dur])]
                trial_inds = list(metadata[(metadata["SOA_lock"] == "offset") & (metadata["duration"] == dur)].index)
                # Extract the data:
                blink_data = np.squeeze(epo.copy().crop(tmin=blinks_window[0],
                                                        tmax=blinks_window[1]).get_data(
                    picks=["_".join(["blink", eye])
                           for eye in eyes]))
                if blink_data.shape[1] == 2:
                    # Combine both eyes data:
                    blink_data = np.logical_and(blink_data[:, 0, :], blink_data[:, 1, :]).astype(float)
                # Find the trials in which we have blinks:
                blink_inds.extend([trial_inds[i] for i in np.where(np.any(blink_data, axis=1))[0]])

        else:
            # Extract the blinks channels:
            blink_data = np.squeeze(epochs.copy().crop(tmin=blinks_window[0],
                                                       tmax=blinks_window[1]).get_data(picks=["_".join(["blink", eye])
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

    return epochs, 1 - ntrials_final / ntrials_orig


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


def generate_gaze_map(epochs, height, width, sigma=5, eyes=None):
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
