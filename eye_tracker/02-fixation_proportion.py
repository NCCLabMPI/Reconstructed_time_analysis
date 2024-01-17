import mne
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from eye_tracker.general_helper_function import baseline_scaling
import seaborn as sns
import pandas as pd
import os
import environment_variables as ev
from plotter_functions import soa_boxplot

# Set the font size:
plt.rcParams.update({'font.size': 22})
dpi = 300


def fixation_analysis(parameters_file, subjects):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)

    fixation_proportion = pd.DataFrame()
    # Loop through each subject:
    for sub in subjects:
        print(sub)
        # ===========================================
        # Data loading:
        # Load the epochs:
        root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + param["session"],
                    param["data_type"])
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(sub, param["session"], param["task"],
                                                                      param["data_type"],
                                                                      param["epoch_name"])
        epochs = mne.read_epochs(Path(root, file_name))
        # Extract the relevant conditions:
        epochs = epochs[param["task_relevance"]]
        # Extract the relevant channels:
        epochs.pick(param["picks"])

        # Loop through each relevant condition:
        for task_rel in param["task_relevance"]:
            for duration in param["durations"]:
                for lock in param["locks"]:
                    for soa in param["soas"]:
                        # Extract data locked to the visual stimulus:
                        epochs_vislock = epochs.copy().crop(param["crop"][0], param["crop"][1])
                        # Extract data locked to the audio stimulus:
                        if lock == "offset":
                            t0 = float(soa) + float(duration)
                        else:
                            t0 = float(soa)
                        epochs_audlock = epochs.copy().crop(t0 + param["crop"][0], t0 + param["crop"][1])

                        # Extract the data and average across left and right eye:
                        data_vis_lock = np.mean(
                            epochs_vislock.copy()["/".join([task_rel, duration, lock, soa])].get_data(copy=False),
                            axis=1)
                        data_aud_lock = np.mean(
                            epochs_audlock.copy()["/".join([task_rel, duration, lock, soa])].get_data(copy=False),
                            axis=1)
                        # Compute the fixation proportion:
                        fix_prop_vis_lock = np.mean(data_vis_lock < param["fixdist_thresh_deg"])
                        fix_prop_aud_lock = np.mean(data_aud_lock < param["fixdist_thresh_deg"])

                        # Add to data frame using pd.concat:
                        fixation_proportion = pd.concat([fixation_proportion,
                                                         pd.DataFrame({"sub_id": sub,
                                                                       "task_relevance": task_rel,
                                                                       "duration": float(duration),
                                                                       "SOA_lock": lock,
                                                                       "soa": float(soa),
                                                                       "onset_SOA": float(soa) + float(duration)
                                                                       if lock == "offset" else float(soa),
                                                                       "vislock_fixation": fix_prop_vis_lock,
                                                                       "audiolock_fixation": fix_prop_aud_lock},
                                                                      index=[0])])
    fixation_proportion = fixation_proportion.reset_index(drop=True)
    # Create the save directory:
    save_dir = Path(ev.bids_root, "derivatives", "fixation", "group_level", "data")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Save the peak latencies:
    fixation_proportion.to_csv(Path(save_dir, "fixation_proportion.csv"))

    # =========================================================================
    # Plot the results:
    # Create the save directory:
    save_dir = Path(ev.bids_root, "derivatives", "fixation", "group_level", "figures")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Task relevant:
    fig_tr, ax_tr = soa_boxplot(fixation_proportion[fixation_proportion["task_relevance"] == 'non-target'],
                                "audiolock_fixation",
                                fig_size=[8.3 / 2, 11.7 / 2])
    # Task irrelevant:
    fig_ti, ax_ti = soa_boxplot(fixation_proportion[fixation_proportion["task_relevance"] == 'irrelevant'],
                                "audiolock_fixation",
                                fig_size=[8.3 / 2, 11.7 / 2])
    # Set the y limit to be the same for both plots:
    lims = [[ax_tr[0].get_ylim()[0], ax_ti[0].get_ylim()[0]], [ax_tr[0].get_ylim()[1], ax_ti[0].get_ylim()[1]]]
    max_lims = [min(min(lims)), max(max(lims))]
    ax_tr[0].set_ylim(max_lims)
    ax_ti[0].set_ylim(max_lims)
    # Axes decoration:
    fig_tr.suptitle("Relevant non-target")
    fig_ti.suptitle("Irrelevant non-target")
    fig_tr.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_ti.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_tr.text(0, 0.5, 'Reaction time (sec.)', ha='center', va='center', fontsize=18, rotation=90)
    fig_ti.text(0, 0.5, 'Reaction time (sec.)', ha='center', va='center', fontsize=18, rotation=90)
    fig_tr.savefig(Path(save_dir, "fixation_tr_audiolock.svg"), transparent=True, dpi=dpi)
    fig_tr.savefig(Path(save_dir, "fixation_tr_audiolock.png"), transparent=True, dpi=dpi)
    fig_ti.savefig(Path(save_dir, "fixation_ti_audiolock.svg"), transparent=True, dpi=dpi)
    fig_ti.savefig(Path(save_dir, "fixation_ti_audiolock.png"), transparent=True, dpi=dpi)
    plt.close(fig_tr)
    plt.close(fig_ti)

    # Plot the visual locked data:
    # Task relevant:
    fig_tr, ax_tr = soa_boxplot(fixation_proportion[fixation_proportion["task_relevance"] == 'non-target'],
                                "vislock_fixation",
                                fig_size=[8.3 / 2, 11.7 / 2])
    # Task irrelevant:
    fig_ti, ax_ti = soa_boxplot(fixation_proportion[fixation_proportion["task_relevance"] == 'irrelevant'],
                                "vislock_fixation",
                                fig_size=[8.3 / 2, 11.7 / 2])
    # Set the y limit to be the same for both plots:
    lims = [[ax_tr[0].get_ylim()[0], ax_ti[0].get_ylim()[0]], [ax_tr[0].get_ylim()[1], ax_ti[0].get_ylim()[1]]]
    max_lims = [min(min(lims)), max(max(lims))]
    ax_tr[0].set_ylim(max_lims)
    ax_ti[0].set_ylim(max_lims)
    # Axes decoration:
    fig_tr.suptitle("Relevant non-target")
    fig_ti.suptitle("Irrelevant non-target")
    fig_tr.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_ti.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_tr.text(0, 0.5, 'Reaction time (sec.)', ha='center', va='center', fontsize=18, rotation=90)
    fig_ti.text(0, 0.5, 'Reaction time (sec.)', ha='center', va='center', fontsize=18, rotation=90)
    fig_tr.savefig(Path(save_dir, "fixation_tr_vislock.svg"), transparent=True, dpi=dpi)
    fig_tr.savefig(Path(save_dir, "fixation_tr_vislock.png"), transparent=True, dpi=dpi)
    fig_ti.savefig(Path(save_dir, "fixation_ti_vislock.svg"), transparent=True, dpi=dpi)
    fig_ti.savefig(Path(save_dir, "fixation_ti_vislock.png"), transparent=True, dpi=dpi)
    plt.close(fig_tr)
    plt.close(fig_ti)

    return None


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121"]
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\02-fixation_proportion_parameters.json ")
    fixation_analysis(parameters, subjects_list)
