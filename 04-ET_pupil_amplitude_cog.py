import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from helper_function.helper_general import (baseline_scaling, cluster_1samp_across_sub, equate_epochs_events,
                                            reject_bad_epochs, format_drop_logs)
from helper_function.helper_plotter import plot_ts_ci
import environment_variables as ev

# Set the font size:
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
dpi = 300
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def pupil_amplitude(parameters_file, subjects, bids_root, session="1", task="prp", analysis_name="pupil_amplitude",
                    reject_bad_trials=True):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)
    # Load all subjects data:
    subjects_epochs = {}
    subjects_targets = {}
    # Create the directory to save the results in:
    save_dir = Path(bids_root, "derivatives", analysis_name, task)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Loop through each subject:
    for sub in subjects:
        print("Loading sub-{}".format(sub))
        if isinstance(session, list):
            epochs = []
            for ses in session:
                root = Path(bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + ses,
                            param["data_type"])
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(sub, ses, task,
                                                                           param["data_type"])
                epochs.append(mne.read_epochs(Path(root, file_name)))
            # Equate the epochs events.
            epochs = equate_epochs_events(epochs)
            epochs = mne.concatenate_epochs(epochs, add_offset=True)
        else:
            root = Path(bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                        param["data_type"])
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(sub, session, task,
                                                                       param["data_type"])
            epochs = mne.read_epochs(Path(root, file_name))
        # Crop the epochs:
        epochs.crop(param["crop"][0], param["crop"][1])

        # Extract the eyes:
        eyes = [ch.split("_")[-1] for ch in epochs.ch_names if "pupil" in ch]

        # Extract the relevant channels:
        epochs.pick([f"pupil_{eye}" for eye in eyes])
        print(epochs.info["sfreq"])
        # Downsample:
        epochs.resample(250)
        # Baseline correction:
        baseline_scaling(epochs, correction_method=param["baseline"], baseline=param["baseline_window"])
        subjects_epochs[sub] = epochs.copy()[param["task_relevance"]]
        subjects_targets[sub] = epochs.copy()["target"]

    # Plot the drop logs:
    drop_log_df = format_drop_logs({sub: subjects_epochs[sub].drop_log for sub in subjects_epochs.keys()})

    # Plot the drop log:
    # Extract the columns:
    cols = [col for col in drop_log_df.columns if col != "sub"]
    fig, ax = plt.subplots(figsize=[8.3, 8.3])
    ax.boxplot([drop_log_df[col].to_numpy() for col in cols], labels=cols)
    ax.axhline(param["drop_trials_threshold"], linestyle="--", color="r")
    ax.set_ylabel("Proportion dropped trials")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(Path(save_dir, "drop_log.svg"), transparent=True, dpi=300)
    fig.savefig(Path(save_dir, "drop_log.png"), transparent=True, dpi=300)
    plt.close()

    # Extract the subject that exceed the proportion of dropped trials
    drop_subjects = drop_log_df.loc[drop_log_df["total"] >= param["drop_trials_threshold"], "sub"].to_list()
    for sub in drop_subjects:
        del subjects_epochs[sub]

    # ==================================================================================================================
    # Task relevance comparisons:
    ylim = [0.95, 1.22]
    # =====================================================================================
    # Onset locked task relevance analysis:
    # ===========================================================
    # Create the condition string:
    conditions = param["task_relevance"]
    # Compute cluster based permutation test across subject between task relevant and irrelevant:
    evks, evks_diff, _, clusters, cluster_p_values, _ = (
        cluster_1samp_across_sub(subjects_epochs, conditions,
                                 n_permutations=param["n_permutations"],
                                 threshold=param["threshold"],
                                 tail=1, downsample=False))
    # Plot the results:
    fig, ax = plt.subplots(figsize=[8.3, 11.7 / 3])
    # Task relevant:
    plot_ts_ci(evks[conditions[0]], epochs.times, ev.colors["task_relevance"][param["task_relevance"][0]],
               ax=ax, label=param["task_relevance"][0], sig_thresh=0.05, plot_nonsig_clusters=True)
    # Task irrelevant (plot the cluster only on one to avoid incremental plotting):
    plot_ts_ci(evks[conditions[1]], epochs.times, ev.colors["task_relevance"][param["task_relevance"][1]],
               ax=ax, label=param["task_relevance"][1], clusters=clusters,
               clusters_pval=cluster_p_values, clusters_alpha=0.1, sig_thresh=0.05, plot_nonsig_clusters=True)
    # Compute the targets evoked:
    targets_evoked = np.array([np.mean(subjects_targets[sub]["target"].average().get_data(), axis=0)
                               for sub in subjects_targets.keys()])
    plot_ts_ci(targets_evoked, epochs.times, [0.4, 0.4, 0.4],
               ax=ax, label="target")
    # Decorate the axes:
    ax.set_xlabel("Time (sec.)")
    ax.set_ylabel("Pupil dilation (norm.)")
    ax.set_ylim(ylim)
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend()
    plt.title("Pupil size across durations (N={})".format(len(subjects_epochs)))
    plt.tight_layout()
    fig.savefig(Path(save_dir, "pupil_evoked_titr.svg"), transparent=True, dpi=300)
    fig.savefig(Path(save_dir, "pupil_evoked_titr.png"), transparent=True, dpi=300)
    plt.close()

    # ===========================================================
    # Separately for each trial durations:
    # Prepare a figure for all the durations:
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=[8.3, 11.7])
    for dur_i, dur in enumerate(param["duration"]):
        # Prepare the condition strings:
        conditions = ["/".join([task, dur]) for task in param["task_relevance"]]
        # Run cluster based permutation test:
        evks_dur, evks_diff_dur, _, clusters, cluster_p_values, _ = (
            cluster_1samp_across_sub(subjects_epochs, conditions,
                                     n_permutations=param["n_permutations"],
                                     threshold=param["threshold"],
                                     tail=1, downsample=True))
        # Plot the results:
        # Task relevant:
        plot_ts_ci(evks_dur[conditions[0]], epochs.times,
                   ev.colors["task_relevance"][param["task_relevance"][0]], ax=ax[dur_i],
                   label=param["task_relevance"][0], sig_thresh=0.05 / len(param["duration"]),
                   plot_single_subjects=False, plot_nonsig_clusters=True)
        # Task irrelevant:
        plot_ts_ci(evks_dur[conditions[1]], epochs.times,
                   ev.colors["task_relevance"][param["task_relevance"][1]], ax=ax[dur_i], clusters=clusters,
                   clusters_pval=cluster_p_values, clusters_alpha=0.1,
                   label=param["task_relevance"][1], sig_thresh=0.05 / len(param["duration"]),
                   plot_single_subjects=False, plot_nonsig_clusters=True)
        # Compute the targets evoked:
        targets_evoked = np.array([np.mean(subjects_targets[sub]["/".join([dur])].average().get_data(), axis=0)
                                   for sub in subjects_targets.keys()])
        plot_ts_ci(targets_evoked, epochs.times, [0.4, 0.4, 0.4],
                   ax=ax[dur_i], label="target")

    # Decorate the axes:
    ax[0].set_ylim(ylim)
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set_title("Short")
    ax[1].set_ylabel("Pupil dilation (norm.)")
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set_title("Intermediate")
    ax[2].set_xlabel("Time (sec.)")
    ax[2].set_title("Long")
    ax[2].spines[['right', 'top']].set_visible(False)
    ax[2].legend()
    plt.suptitle("locked pupil size (N={})".format(len(subjects_epochs)))
    plt.tight_layout()
    fig.savefig(Path(save_dir, "pupil_evoked_titr_perdur.svg"), transparent=True, dpi=300)
    fig.savefig(Path(save_dir, "pupil_evoked_titr_perdur.png"), transparent=True, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Set the parameters to use:
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\04-ET_pupil_amplitude_parameters_cog.json")

    # ==================================================================================
    # Cogitate duration analysis:
    task = "Dur"
    pupil_amplitude(parameters, [
         "CE103", "CE106", "CE107", "CE108", "CE110",  "CE112", "CE113", 
         "CE119", "CE120"], ev.cog_bids_root, task=task, session="1",
                    analysis_name="pupil_amplitude", reject_bad_trials=False)
