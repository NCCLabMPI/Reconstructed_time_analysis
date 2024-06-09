import os
import json
import pickle
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from mne.decoding import SlidingEstimator, cross_val_multiscore
from scipy.ndimage import gaussian_filter1d
import environment_variables as ev
from helper_function.helper_general import create_super_subject, get_roi_channels, \
    decoding_difference_duration, moving_average, compute_pseudotrials

# Set the font size:
plt.rcParams.update({'font.size': 14})

# Set list of views:
views = ['lateral', 'medial', 'rostral', 'caudal', 'ventral', 'dorsal']


def decoding(parameters_file, subjects, data_root, analysis_name="decoding", task_conditions=None,
             subname="all-dur"):
    """
    Perform decoding analysis on iEEG data.

    :param parameters_file: (str) Path to the parameters file in JSON format.
    :param subjects: (list) List of subjects.
    :param data_root: (str) Root directory for data.
    :param analysis_name: (str) Name of the analysis.
    :param task_conditions: (list) List of task conditions.
    :param subname: (str) Name of the subset.
    :return: None
    """
    if task_conditions is None:
        task_conditions = ["Relevant non-target", "Irrelevant", "Target"]

    with open(parameters_file) as json_file:
        param = json.load(json_file)

    save_dir = Path(ev.bids_root, "derivatives", analysis_name, param["task"], subname)
    os.makedirs(save_dir, exist_ok=True)

    subjects_epochs = {}
    roi_results = {}
    times = []

    if param["rois"] == "all":
        labels = mne.read_labels_from_annot("fsaverage", parc='aparc.a2009s', hemi='both', surf_name='pial',
                                            subjects_dir=ev.fs_directory, sort=True)
        roi_names = list(set(lbl.name.replace("-lh", "").replace("-rh", "")
                             for lbl in labels if "unknown" not in lbl.name))
        rois_list = [["ctx_lh_" + roi, "ctx_rh_" + roi] for roi in roi_names]
    else:
        rois_list = param["rois"]

    for ii, roi in enumerate(rois_list):
        roi_name = roi[0].replace("ctx_lh_", "")
        print("=========================================")
        print("ROI")
        print(roi_name)
        roi_results[roi_name] = {}

        for sub in subjects:
            epochs_file = Path(data_root, "derivatives", "preprocessing", f"sub-{sub}", f"ses-{param["session"]}",
                               param["data_type"], param["preprocessing_folder"], param["signal"],
                               f"sub-{sub}_ses-{param["session"]}_task-{param["task"]}_desc-epoching_ieeg-epo.fif")
            epochs = mne.read_epochs(epochs_file)
            times = epochs.times

            # Extract the conditions of interest:
            epochs = epochs[param["conditions"]]

            picks = get_roi_channels(data_root, sub, param["session"], param["atlas"], roi)
            if not picks:
                print(f"sub-{sub} has no electrodes in {roi_name}")
                continue
            subjects_epochs[sub] = epochs.pick(picks)

        if not subjects_epochs:
            continue

        clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', class_weight='balanced'))
        time_res = SlidingEstimator(clf, n_jobs=1, scoring='accuracy', verbose="ERROR")

        # ==============================================================================================================
        # 1. Create the super subject:
        # ============================
        # Task relevant:
        tr_epochs = {sub: subjects_epochs[sub][task_conditions[0]] for sub in subjects_epochs}
        data_tr, labels_tr = create_super_subject(tr_epochs, param["targets_column"], n_trials=param["min_ntrials"])

        window_size = int((param["mvavg_window_ms"] * 0.001) / (times[1] - times[0]))

        if param["smooth_ms"] is not None:
            smooth_samp = int((param["smooth_ms"] * 0.001) / (times[1] - times[0]))
            data_tr = gaussian_filter1d(data_tr, smooth_samp, axis=-1)
        else:
            smooth_samp = 0

        data_tr = moving_average(data_tr, window_size, axis=-1, overlapping=False)
        times = moving_average(times, window_size, axis=-1, overlapping=False)
        n_channels_tr = data_tr.shape[1]

        if param["pseudotrials"] is not None:
            data_tr, labels_tr = compute_pseudotrials(data_tr, labels_tr, param["pseudotrials"])

        # ============================
        # Task irrelevant:
        ti_epochs = {sub: subjects_epochs[sub][task_conditions[1]] for sub in subjects_epochs}
        data_ti, labels_ti = create_super_subject(ti_epochs, param["targets_column"], n_trials=param["min_ntrials"])

        if param["smooth_ms"] is not None:
            data_ti = gaussian_filter1d(data_ti, smooth_samp, axis=-1)
        data_ti = moving_average(data_ti, window_size, axis=-1, overlapping=False)
        n_channels_ti = data_ti.shape[1]

        if param["pseudotrials"] is not None:
            data_ti, labels_ti = compute_pseudotrials(data_ti, labels_ti, param["pseudotrials"])

        assert n_channels_ti == n_channels_tr, "The number of channels does not match between task relevance conditions"

        # ============================
        # Targets:
        target_epochs = {sub: subjects_epochs[sub][task_conditions[2]] for sub in subjects_epochs}
        data_tar, labels_tar = create_super_subject(target_epochs, param["targets_column"],
                                                    n_trials=param["min_ntargets"])

        if param["smooth_ms"] is not None:
            data_tar = gaussian_filter1d(data_tar, smooth_samp, axis=-1)
        data_tar = moving_average(data_tar, window_size, axis=-1, overlapping=False)
        n_channels_tar = data_tar.shape[1]

        if param["pseudotrials"] is not None:
            data_tar, labels_tar = compute_pseudotrials(data_tar, labels_tar, param["pseudotrials"])

        assert n_channels_ti == n_channels_tar, "The number of channels does not match between task relevance conditions"
        n_channels = n_channels_tar

        # ==============================================================================================================
        # 2. Decoding and estimating durations between task relevant and irrelevant:
        # ============================
        print("=" * 20)
        print("Compute decoding difference duration: ")
        scores_tr, scores_ti, decoding_diff, null_dist, pvals_diff, onset, offset, duration = (
            decoding_difference_duration(data_tr, labels_tr, data_ti, labels_ti, times, time_res,
                                         n_perm=param["n_perm"], n_jobs=param["n_jobs"],
                                         kfolds=param["kfold"], alpha=param["alpha"],
                                         min_dur_ms=param["dur_threshold"], random_seed=42, tail=1))

        # Compute the decoding score for target trials:
        print("=" * 20)
        print("Compute decoding in target trials: ")
        scores_targets = np.concatenate([
            cross_val_multiscore(time_res, data_tar, labels_tar, cv=param["kfold"], n_jobs=param["n_jobs"])
            for _ in range(5)
        ], axis=0)

        # ==============================================================================================================
        # 3. Package the results and save to pickle:
        roi_results[roi_name] = {
            "scores_tr": scores_tr,
            "scores_ti": scores_ti,
            "scores_targets": scores_targets,
            "decoding_diff": decoding_diff,
            "null_dist": null_dist,
            "pvals_diff": pvals_diff,
            "onset": onset,
            "offset": offset,
            "duration": duration,
            "n_channels": n_channels
        }

        # Save results to file:
        with open(Path(save_dir, f'results-{roi_name}.pkl'), 'wb') as f:
            pickle.dump(roi_results[roi_name], f)

    # Save results to file:
    with open(Path(save_dir, 'results-all_roi.pkl'), 'wb') as f:
        pickle.dump(roi_results, f)


if __name__ == "__main__":
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\06-iEEG_decoding_parameters_all-dur.json"
    )
    bids_root = r"C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\bids-curate"
    decoding(parameters, ev.subjects_lists_ecog["dur"], bids_root,
             analysis_name="decoding", subname="all-dur",
             task_conditions=["Relevant non-target", "Irrelevant", "Relevant target"])
