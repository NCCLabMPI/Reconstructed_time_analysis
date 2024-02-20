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
from scipy.ndimage import uniform_filter1d
import pickle
from helper_function.helper_general import extract_first_bout, create_super_subject, get_roi_channels, \
    get_cmap_rgb_values
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
    latencies_table = []
    roi_results = {}
    times = []
    # Loop through each ROI:
    for ii, roi in enumerate(param["rois"]):
        # Create the directory to save the results in:
        roi_name = roi[0].replace("ctx_lh_", "")
        roi_results[roi_name] = {
            "scores_ti": None, "scores_tr": None, "scores_diff": None,
            "ci_tr": None, "ci_ti": None, "ci_diff": None,
            "pval_tr": None, "pval_ti": None, "pval_diff": None,
            "diff_onset": None, "diff_offset": None,
            "n_channels": None}

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
            # Extract the conditions of interest:
            epochs = epochs[param["conditions"]]
            epochs.decimate(5)
            times = epochs.times
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
        n_channels_tr = data_tr.shape[1]

        # ============================
        # Task irrelevant:
        ti_epochs = {sub: subjects_epochs[sub][task_conditions[1]] for sub in subjects_epochs.keys()}
        # Equate the trial matrices:
        data_ti, labels_ti = create_super_subject(ti_epochs, param["targets_column"],
                                                  n_trials=param["min_ntrials"])
        n_channels_ti = data_tr.shape[1]

        assert n_channels_ti == n_channels_tr, ("The number of channels does not match between task relevances "
                                                "conditions!")
        roi_results[roi_name]["n_channels"] = n_channels_tr

        # ==============================================================================================================
        # 2. Decoding:
        # ============================
        # Task relevant:
        roi_results[roi_name]["scores_tr"] = np.mean(
            cross_val_multiscore(time_res, data_tr, labels_tr, cv=param["kfold"], n_jobs=param["n_jobs"]),
            axis=0)
        # ============================
        # Task irrelevant:
        roi_results[roi_name]["scores_ti"] = np.mean(
            cross_val_multiscore(time_res, data_ti, labels_ti, cv=param["kfold"], n_jobs=param["n_jobs"]),
            axis=0)

        # Compute the difference in decoding accuracy between the task relevant and irrelevant condition:
        roi_results[roi_name]["scores_diff"] = roi_results[roi_name]["scores_tr"] - roi_results[roi_name]["scores_ti"]

        # ==============================================================================================================
        # 3. Permutation
        # ============================
        # Task relevant:
        scores_perm_tr = []
        for i in range(param["n_perm"]):
            score_ = cross_val_multiscore(time_res, data_tr,
                                          labels_tr[np.random.choice(labels_tr.shape[0], labels_tr.shape[0],
                                                                     replace=False)],
                                          cv=param["kfold"], n_jobs=param["n_jobs"])
            scores_perm_tr.append(np.mean(score_, axis=0))
        scores_perm_tr = np.array(scores_perm_tr)
        # Compute the p value:
        pvals = []
        for t in range(roi_results[roi_name]["scores_tr"].shape[0]):
            pvals.append(_pval_from_histogram([roi_results[roi_name]["scores_tr"][t]], scores_perm_tr[:, t], 1))
        roi_results[roi_name]["pval_tr"] = np.array(pvals)

        # ============================
        # Task irrelevant:
        scores_perm_ti = []
        for i in range(param["n_perm"]):
            score_ = cross_val_multiscore(time_res, data_ti,
                                          labels_ti[np.random.choice(labels_ti.shape[0], labels_ti.shape[0],
                                                                     replace=False)],
                                          cv=param["kfold"], n_jobs=param["n_jobs"])
            scores_perm_ti.append(np.mean(score_, axis=0))
        scores_perm_ti = np.array(scores_perm_ti)

        # Compute the p value:
        pvals = []
        for t in range(roi_results[roi_name]["scores_ti"].shape[0]):
            pvals.append(_pval_from_histogram([roi_results[roi_name]["scores_ti"][t]], scores_perm_tr[:, t], 1))
        roi_results[roi_name]["pval_ti"] = np.array(pvals)

        # Compute the shuffled difference in accuracy:
        scores_perm_diff = np.subtract(scores_perm_tr, scores_perm_ti)

        # Compute the p values for the difference:
        pvals = []
        for t in range(roi_results[roi_name]["scores_diff"].shape[0]):
            pvals.append(_pval_from_histogram([roi_results[roi_name]["scores_diff"][t]], scores_perm_diff[:, t], 1))
        roi_results[roi_name]["pval_diff"] = np.array(pvals)
        # Extract onset and offset of the difference:
        onset, offset = extract_first_bout(times, roi_results[roi_name]["pval_diff"],
                                           param["alpha"],
                                           param["dur_threshold"])
        roi_results[roi_name]["onset_diff"] = onset
        roi_results[roi_name]["offset_diff"] = offset

        # ==============================================================================================================
        # 4. Bootstrap:
        # ============================
        # Compute confidence interval for the time series by bootstrapping the electrodes:
        scores_bootsstrap_tr = []
        scores_bootsstrap_ti = []
        scores_bootsstrap_diff = []
        for i in range(param["n_bootsstrap"]):
            ch_inds = np.random.choice(n_channels_tr, n_channels_tr, replace=True)
            score_bootstrap_tr = np.mean(cross_val_multiscore(time_res, data_tr[:, ch_inds, :], labels_tr,
                                                              cv=param["kfold"], n_jobs=param["n_jobs"]), axis=0)
            score_bootstrap_ti = np.mean(cross_val_multiscore(time_res, data_ti[:, ch_inds, :], labels_ti,
                                                              cv=param["kfold"], n_jobs=param["n_jobs"]), axis=0)
            # Compute the difference:
            score_bootsstrap_diff = score_bootstrap_tr - score_bootstrap_ti
            # Add everything the to the lists:
            scores_bootsstrap_tr.append(score_bootstrap_tr)
            scores_bootsstrap_ti.append(score_bootstrap_ti)
            scores_bootsstrap_diff.append(score_bootsstrap_diff)
        scores_bootsstrap_tr = np.array(scores_bootsstrap_tr)
        scores_bootsstrap_ti = np.array(scores_bootsstrap_ti)
        scores_bootsstrap_diff = np.array(scores_bootsstrap_diff)
        # Compute the 95% CI:
        ci = (((1 - param["ci"]) / 2) * 100, (1 - ((1 - param["ci"]) / 2)) * 100)
        roi_results[roi_name]["ci_tr"] = np.percentile(scores_bootsstrap_tr, ci, axis=0)
        roi_results[roi_name]["ci_ti"] = np.percentile(scores_bootsstrap_ti, ci, axis=0)
        roi_results[roi_name]["ci_diff"] = np.percentile(scores_bootsstrap_diff, ci, axis=0)

        # ==============================================================================================================
        # 6. Plot single ROI results:
        # Plot decoding accuracy:
        fig, ax = plt.subplots()
        # Task relevant:
        plot_decoding_accuray(times, roi_results[roi_name]["scores_tr"], roi_results[roi_name]["ci_tr"],
                              roi_results[roi_name]["pval_tr"], smooth_ms=param["smooth_ms"], label=task_conditions[0],
                              color=ev.colors["task_relevance"][task_conditions[0]], ax=ax, alpha=param["alpha"],
                              pval_height=0.05, ylim=param["ylim"])
        # Task irrelevant:
        plot_decoding_accuray(times, roi_results[roi_name]["scores_ti"], roi_results[roi_name]["ci_ti"],
                              roi_results[roi_name]["pval_ti"], smooth_ms=param["smooth_ms"], label=task_conditions[0],
                              color=ev.colors["task_relevance"][task_conditions[1]], ax=ax, alpha=param["alpha"],
                              pval_height=0.1, ylim=param["ylim"])
        ax.legend()
        ax.set_xlabel("Time (sec.)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Decoding over time in {} \n(# channels={})".format(roi_name,
                                                                         roi_results[roi_name]["n_channels"]))
        fig.savefig(Path(save_dir, "{}_decoding.svg".format(roi_name)),
                    transparent=True, dpi=300)
        fig.savefig(Path(save_dir, "{}_decoding.png".format(roi_name)),
                    transparent=True, dpi=300)
        plt.close()

        # Plot the difference between task relevances:
        fig, ax = plt.subplots()
        plot_decoding_accuray(times, roi_results[roi_name]["scores_diff"], roi_results[roi_name]["ci_diff"],
                              roi_results[roi_name]["pval_diff"], smooth_ms=param["smooth_ms"],
                              color=ev.colors["task_relevance"][task_conditions[0]], ax=ax, alpha=param["alpha"],
                              pval_height=0.05, ylim=None)
        # Add the onset and offsets:
        if roi_results[roi_name]["onset_diff"] is not None:
            ax.axvline(roi_results[roi_name]["onset_diff"], color=ev.colors["soa_onset_locked"]["0"])
            ax.axvline(roi_results[roi_name]["offset_diff"], color=ev.colors["soa_offset_locked"]["0"])
        ax.legend()
        ax.set_xlabel("Time (sec.)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Decoding over time in {} \n(# channels={})".format(roi_name,
                                                                         roi_results[roi_name]["n_channels"]))
        fig.savefig(Path(save_dir, "{}_decoding_diff.svg".format(roi_name)),
                    transparent=True, dpi=300)
        fig.savefig(Path(save_dir, "{}_decoding_diff.png".format(roi_name)),
                    transparent=True, dpi=300)
        plt.close()

    # Save results to file:
    with open(Path(save_dir, 'roi_results.pkl'), 'wb') as f:
        pickle.dump(roi_results, f)
    # Concatenate the table:
    latencies_table = pd.concat(
        [pd.DataFrame({
            "roi": roi,
            "onset": roi_results[roi]["onset_diff"],
            "offset": roi_results[roi]["offset_diff"],
        }, index=[0])
            for roi in roi_results.keys()]
    ).reset_index(drop=True)
    latencies_table.to_csv(Path(Path(ev.bids_root, "derivatives", analysis_name, task),
                                "latency_table.csv"))
    latencies_table = latencies_table[~latencies_table["onset"].isna()]

    # ====================================================================================================
    # Plot the brain:
    # ===============
    # Read the annotations
    annot = mne.read_labels_from_annot("fsaverage", parc='aparc.a2009s', subjects_dir=ev.fs_directory)
    # Onset:
    # Get colors for each label:
    onset_colors = get_cmap_rgb_values(latencies_table["onset"].to_numpy(), cmap="Reds", center=None)
    # Plot the brain:
    Brain = mne.viz.get_brain_class()
    brain = Brain('fsaverage', hemi='lh', surf='inflated',  subjects_dir=ev.fs_directory, size=(800, 600))
    # Loop through each label:
    for roi_i, roi in enumerate(latencies_table["roi"].to_list()):
        # Find the corresponding label:
        lbl = [l for l in annot if l.name == roi + "-lh"]
        brain.add_label(lbl[0], color=onset_colors[roi_i])
    for view in views:
        brain.show_view(view)
        brain.save_image(Path(save_dir, "{}_{}.png".format("onset", view)))
    brain.close()

    # ===============
    # offset:
    # Get colors for each label:
    offset_colors = get_cmap_rgb_values(latencies_table["offset"].to_numpy(), cmap="Blues", center=None)
    # Plot the brain:
    Brain = mne.viz.get_brain_class()
    brain = Brain('fsaverage', hemi='lh', surf='inflated',
                  subjects_dir=ev.fs_directory, size=(800, 600))
    # Loop through each label:
    for roi_i, roi in enumerate(latencies_table["roi"].to_list()):
        # Find the corresponding label:
        lbl = [l for l in annot if l.name == roi + "-lh"]
        brain.add_label(lbl[0], color=offset_colors[roi_i])
    for view in views:
        brain.show_view(view)
        brain.save_image(Path(save_dir, "{}_{}.png".format("offset", view)))
    brain.close()

    # ====================================================================================================
    # Plot time series of difference between the two conditions:
    # ===============
    rois_colors = get_cmap_rgb_values(latencies_table["offset"].to_numpy(), cmap="jet", center=None)
    fig, ax = plt.subplots()
    for i, roi_name in enumerate(list(latencies_table["roi"].unique())):
        # Extract the data of each task:
        ax.plot(times, roi_results[roi_name]["scores_diff"], color=rois_colors[i], label=roi_name)
    ax.set_xlabel("Time (sec.)")
    ax.set_ylabel("Accuracy difference")
    ax.set_title("Difference in decoding accuracy (TR - TI)")
    ax.legend()
    fig.savefig(Path(save_dir, "accuracy_difference_per_roi.svg"),
                transparent=True, dpi=300)
    fig.savefig(Path(save_dir, "accuracy_difference_per_roi.png"),
                transparent=True, dpi=300)
    plt.close()


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
