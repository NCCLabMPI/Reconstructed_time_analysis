import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from mne.decoding import (SlidingEstimator, cross_val_multiscore)
from mne.stats.cluster_level import _pval_from_histogram
import environment_variables as ev
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from helper_function.helper_general import extract_first_bout, create_super_subject, get_roi_channels, \
    get_cmap_rgb_values, moving_average, estimate_decoding_duration, compute_ci, compute_pseudotrials
from helper_function.helper_plotter import plot_decoding_accuray
import pandas as pd

# Set the font size:
plt.rcParams.update({'font.size': 14})

# Set list of views:
views = ['lateral', 'medial', 'rostral', 'caudal', 'ventral', 'dorsal']


def decoding(parameters_file, subjects, data_root, session="1", task="dur", analysis_name="decoding",
             task_conditions=None):
    # First, load the parameters:
    if task_conditions is None:
        task_conditions = ["Relevant non-target", "Irrelevant"]
    with open(parameters_file) as json_file:
        param = json.load(json_file)
    save_dir = Path(ev.bids_root, "derivatives", analysis_name, task)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Load all subjects data:
    subjects_epochs = {}
    # Prepare a table to store the onsets and offsets in each roi:
    roi_results = {}
    times = []
    # Loop through each ROI:
    for ii, roi in enumerate(param["rois"]):
        # Create the directory to save the results in:
        roi_name = roi[0].replace("ctx_lh_", "")
        print("=========================================")
        print("ROI")
        print(roi_name)
        roi_results[roi_name] = {}

        # Load each subjects' data:
        for sub in subjects:
            # Create file name:
            epochs_file = Path(data_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                               "ieeg", "epoching",
                               "sub-{}_ses-{}_task-{}_desc-epoching_ieeg-epo.fif".format(sub,
                                                                                         session,
                                                                                         task))
            # Load the file:
            epochs = mne.read_epochs(epochs_file)

            # Issue with the encoding of the channels coordinates frame in previous MNE version
            from mne.io.constants import FIFF
            for d in epochs.info["dig"]:
                d['coord_frame'] = FIFF.FIFFV_COORD_MRI
            # Crop if needed:
            epochs.crop(param["crop"][0], param["crop"][1])
            times = epochs.times
            # Extract the conditions of interest:
            epochs = epochs[param["conditions"]]

            # Extract only the channels in the correct ROI:
            roi_channels = get_roi_channels(epochs.get_montage(), "sub-" + sub,
                                            roi, ev.fs_directory,
                                            aseg="aparc.a2009s+aseg", dist=2)
            if len(roi_channels) == 0:
                print("sub-{} has no electrodes in {}".format(sub, roi_name))
                continue
            # Append to the rest:
            subjects_epochs[sub] = epochs.pick(roi_channels)
        if len(subjects_epochs) == 0:
            continue
        # Create the classifier:
        clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', class_weight='balanced'))
        # Temporal generalization estimator:
        time_res = SlidingEstimator(clf, n_jobs=1, scoring='accuracy',
                                    verbose="ERROR")

        # ==============================================================================================================
        # 1. Create the super subject:
        # ============================
        # Task relevant:
        tr_epochs = {sub: subjects_epochs[sub][task_conditions[0]] for sub in subjects_epochs.keys()}
        # Equate the trial matrices:
        data_tr, labels_tr = create_super_subject(tr_epochs, param["targets_column"],
                                                  n_trials=param["min_ntrials"])
        # Smooth the data.
        window_size = int((param["mvavg_window_ms"] * 0.001) / (times[1] - times[0]))
        data_tr = moving_average(data_tr, window_size, axis=-1, overlapping=False)
        times = moving_average(times, window_size, axis=-1, overlapping=False)
        n_channels_tr = data_tr.shape[1]
        # Compute the pseudotrials:
        if param["pseudotrials"] is not None:
            data_tr, labels_tr = compute_pseudotrials(data_tr, labels_tr, param["pseudotrials"])
        # ============================
        # Task irrelevant:
        ti_epochs = {sub: subjects_epochs[sub][task_conditions[1]] for sub in subjects_epochs.keys()}
        # Equate the trial matrices:
        data_ti, labels_ti = create_super_subject(ti_epochs, param["targets_column"],
                                                  n_trials=param["min_ntrials"])
        data_ti = moving_average(data_ti, window_size, axis=-1, overlapping=False)
        n_channels_ti = data_ti.shape[1]
        # Compute the pseudotrials:
        if param["pseudotrials"] is not None:
            data_ti, labels_ti = compute_pseudotrials(data_ti, labels_ti, param["pseudotrials"])

        assert n_channels_ti == n_channels_tr, ("The number of channels does not match between task relevances "
                                                "conditions!")
        n_channels = n_channels_ti

        # ==============================================================================================================
        # 2. Decoding and estimating durations:
        # ============================
        print("=" * 20)
        print("Decoding in task relevant")
        # Task relevant:
        scores_tr, pvals_tr, ci_tr, onsets_tr, offsets_tr, durations_tr = (
            estimate_decoding_duration(data_tr, labels_tr, times, time_res, n_bootrstrap=param["n_bootsstrap"],
                                       n_perm=param["n_perm"], n_jobs=param["n_jobs"], kfolds=param["kfold"],
                                       alpha=param["alpha"], min_dur_ms=param["dur_threshold"], random_seed=42))
        # ============================
        print("=" * 20)
        print("Decoding in task irrelevant")
        # Task irrelevant:
        scores_ti, pvals_ti, ci_ti, onsets_ti, offsets_ti, durations_ti = (
            estimate_decoding_duration(data_ti, labels_ti, times, time_res, n_bootrstrap=param["n_bootsstrap"],
                                       n_perm=param["n_perm"], n_jobs=param["n_jobs"], kfolds=param["kfold"],
                                       alpha=param["alpha"], min_dur_ms=param["dur_threshold"], random_seed=42))

        # Compute the difference in duration between TR and TI:
        duration_difference = durations_tr - durations_ti
        # Compute the duration difference 95% CI:
        dur_diff_ci = compute_ci(duration_difference, axis=0, interval=0.95)

        # ==============================================================================================================
        # 3. Package the results and save to pickle:
        roi_results[roi_name]["scores_tr"] = scores_tr
        roi_results[roi_name]["pvals_tr"] = pvals_tr
        roi_results[roi_name]["ci_tr"] = ci_tr
        roi_results[roi_name]["onsets_tr"] = onsets_tr
        roi_results[roi_name]["offsets_tr"] = offsets_tr
        roi_results[roi_name]["durations_tr"] = durations_tr
        roi_results[roi_name]["pvals_ti"] = pvals_ti
        roi_results[roi_name]["scores_ti"] = scores_ti
        roi_results[roi_name]["pvals_ti"] = pvals_ti
        roi_results[roi_name]["ci_ti"] = ci_ti
        roi_results[roi_name]["onsets_ti"] = onsets_ti
        roi_results[roi_name]["offsets_ti"] = offsets_ti
        roi_results[roi_name]["durations_ti"] = durations_ti
        roi_results[roi_name]["durations_difference"] = duration_difference
        roi_results[roi_name]["durations_difference_ci"] = dur_diff_ci
        roi_results[roi_name]["n_channels"] = n_channels

        # Save results to file:
        with open(Path(save_dir, 'results-{}.pkl'.format(roi_name)), 'wb') as f:
            pickle.dump(roi_results[roi_name], f)

        # ==============================================================================================================
        # 4. Plot results of this ROI
        if np.all(dur_diff_ci > 0):
            fig_dir = save_dir
        else:
            fig_dir = Path(save_dir, "no_diff")
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
        # Plot decoding accuracy:
        fig, ax = plt.subplots()
        # Task relevant:
        plot_decoding_accuray(times, np.mean(scores_tr, axis=0), ci_tr, smooth_ms=param["smooth_ms"],
                              label=task_conditions[0],
                              color=ev.colors["task_relevance"][task_conditions[0]], ax=ax, ylim=param["ylim"])
        # Task irrelevant:
        plot_decoding_accuray(times, np.mean(scores_ti, axis=0), ci_ti,
                              smooth_ms=param["smooth_ms"], label=task_conditions[1],
                              color=ev.colors["task_relevance"][task_conditions[1]], ax=ax, ylim=param["ylim"])
        ax.legend()
        ax.set_xlim([times[0], times[-1]])
        ax.set_xlabel("Time (sec.)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Decoding over time in {} \n(# channels={})".format(roi_name,
                                                                         roi_results[roi_name]["n_channels"]))
        fig.savefig(Path(fig_dir, "{}_decoding.svg".format(roi_name)),
                    transparent=True, dpi=300)
        fig.savefig(Path(fig_dir, "{}_decoding.png".format(roi_name)),
                    transparent=True, dpi=300)
        plt.close()

    # Save results to file:
    with open(Path(save_dir, 'results-all_roi.pkl'), 'wb') as f:
        pickle.dump(roi_results, f)


if __name__ == "__main__":
    # Set the parameters to use:
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\06-iEEG_decoding_parameters.json")
    # ==================================================================================
    # Decoding analysis of the COGITATE data:
    decoding(parameters, ev.subjects_lists_ecog["dur"], ev.bids_root,
             session="V1", task="Dur", analysis_name="decoding",
             task_conditions=["Relevant non-target", "Irrelevant"])
