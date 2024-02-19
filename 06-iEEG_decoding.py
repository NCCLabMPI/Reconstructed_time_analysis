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
from helper_function.helper_general import extract_first_bout, create_super_subject, get_roi_channels, \
    get_cmap_rgb_values
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
    roi_scores = {}
    times = []
    # Loop through each ROI:
    for ii, roi in enumerate(param["rois"]):
        # Create the directory to save the results in:
        roi_name = roi[0].replace("ctx_lh_", "")
        roi_scores[roi_name] = {}
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
        scores_dict = {}
        decoding_latencies = {}
        scores_pval_dict = {}
        scores_ci_dict = {}
        n_channels = 0
        # Loop through each task relevance:
        for tsk in task_conditions:
            # Extract this condition only:
            tsk_epochs = {sub: subjects_epochs[sub][tsk] for sub in subjects_epochs.keys()}
            # Equate the trial matrices:
            data, labels = create_super_subject(tsk_epochs, param["targets_column"],
                                                n_trials=param["min_ntrials"])
            n_channels = data.shape[1]
            # Perform the decoding:
            scores = np.mean(cross_val_multiscore(time_res, data, labels, cv=param["kfold"], n_jobs=param["n_jobs"]),
                             axis=0)
            scores_dict[tsk] = scores

            # Create a null distribution using label shuffle:
            scores_perm = []
            for i in range(param["n_perm"]):
                score_ = cross_val_multiscore(time_res, data,
                                              labels[np.random.choice(labels.shape[0], labels.shape[0],
                                                                      replace=False)],
                                              cv=param["kfold"], n_jobs=param["n_jobs"])
                scores_perm.append(np.mean(score_, axis=0))
            scores_perm = np.array(scores_perm)

            # Compute the p_values:
            pvals = []
            for t in range(scores.shape[0]):
                pvals.append(_pval_from_histogram([scores[t]], scores_perm[:, t], 1))
            scores_pval_dict[tsk] = np.array(pvals)
            decoding_latencies[tsk] = extract_first_bout(times, np.array(pvals),
                                                         param["alpha"],
                                                         param["dur_threshold"])

            # Repeating the same but subsampling the electrodes:
            scores_bootsstrap = []
            for i in range(param["n_bootsstrap"]):
                scores_bootsstrap.append(np.mean(cross_val_multiscore(time_res,
                                                                      data[:, np.random.choice(data.shape[1],
                                                                                               data.shape[1],
                                                                                               replace=True), :],
                                                                      labels,
                                                                      cv=param["kfold"], n_jobs=param["n_jobs"]),
                                                 axis=0))
            # Compute the confidence interval:
            ci = (((1 - param["ci"]) / 2) * 100, ((1 - ((1 - param["ci"]) / 2))) * 100)
            ci_low, ci_up = np.percentile(np.array(scores_bootsstrap), ci, axis=0)
            # Add to the dictionary:
            scores_ci_dict[tsk] = np.array([ci_low, ci_up])

        # Add the score to the results of this ROI:
        roi_scores[roi_name] = scores_dict

        # Plot the results for this ROI:
        fig, ax = plt.subplots()
        for tsk in task_conditions:
            score = uniform_filter1d(scores_dict[tsk], size=10, axis=-1)
            upci = uniform_filter1d(scores_ci_dict[tsk][0], size=10, axis=-1)
            lowci = uniform_filter1d(scores_ci_dict[tsk][1], size=10, axis=-1)
            ax.plot(times, score, color=ev.colors["task_relevance"][tsk], label=tsk)
            ax.fill_between(times, lowci, upci,
                            color=ev.colors["task_relevance"][tsk],
                            alpha=.3)
            if decoding_latencies[tsk][1] is not None:
                ax.axvline(decoding_latencies[tsk][1], color=ev.colors["task_relevance"][tsk])
        ax.set_ylim(param["ylim"])
        ax.legend()
        ax.set_xlabel("Time (sec.)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Decoding over time in {} \n(# channels={})".format(roi_name, n_channels))
        fig.savefig(Path(save_dir, "{}_decoding.svg".format(roi_name)),
                    transparent=True, dpi=300)
        fig.savefig(Path(save_dir, "{}_decoding.png".format(roi_name)),
                    transparent=True, dpi=300)
        plt.close()

        # Compute the difference in onset latency between both conditions:
        if decoding_latencies["Relevant non-target"][0] is not None and decoding_latencies["Irrelevant"][0] is not None:
            onset_diff = decoding_latencies["Relevant non-target"][0] - decoding_latencies["Irrelevant"][0]
            offset_diff = decoding_latencies["Relevant non-target"][1] - decoding_latencies["Irrelevant"][1]
        else:
            onset_diff = 0
            offset_diff = 0
        # Create the latency table for this ROI:
        latencies_table.append(pd.DataFrame({
            "roi": roi_name,
            "onset tr-ti": onset_diff,
            "offset tr-ti": offset_diff,
        }, index=[0]))

    # Concatenate the table:
    latencies_table = pd.concat(latencies_table).reset_index(drop=True)
    latencies_table.to_csv(Path(Path(ev.bids_root, "derivatives", analysis_name, task),
                                "latency_table.csv"))

    # ====================================================================================================
    # Plot the brain:
    # ===============
    # Read the annotations
    annot = mne.read_labels_from_annot("fsaverage", parc='aparc.a2009s', subjects_dir=ev.fs_directory)
    # Onset:
    # Get colors for each label:
    onset_colors = get_cmap_rgb_values(latencies_table["onset tr-ti"].to_numpy(), cmap="Reds", center=None)
    # Plot the brain:
    Brain = mne.viz.get_brain_class()
    brain = Brain('fsaverage', hemi='lh', surf='inflated',
                  subjects_dir=ev.fs_directory, size=(800, 600))
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
    offset_colors = get_cmap_rgb_values(latencies_table["offset tr-ti"].to_numpy(), cmap="Blues", center=None)
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
    # Plot time series of difference between the two conditons:
    # ===============
    # Extract only the ROI where the difference is different from zeros:
    sig_latencies = latencies_table.loc[latencies_table["offset tr-ti"] > 0]
    rois_colors = get_cmap_rgb_values(sig_latencies["offset tr-ti"].to_numpy(), cmap="jet", center=None)
    fig, ax = plt.subplots()
    for i, roi_name in list(sig_latencies["roi"].unique()):
        # Extract the data of each task:
        tr_scores = roi_scores[roi_name]["Relevant non-target"]
        ti_scores = roi_scores[roi_name]["Irrelevant"]
        diff = tr_scores - ti_scores
        ax.plot(times, diff, color=rois_colors[i], label=roi_name)
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
