import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import zscore
from helper_function.helper_plotter import plot_within_subject_boxplot, soa_boxplot
import environment_variables as ev
from helper_function.helper_general import beh_exclusion

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
dpi = 300
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def load_beh_data(root, subjects, session='1', task='prp', do_trial_exclusion=True):
    """
    This function loads the behavioral data
    :param root:
    :param subjects:
    :param session:
    :param task:
    :param do_trial_exclusion:
    :return:
    """
    # Load the data:
    subjects_data = []
    if do_trial_exclusion:
        prop_rejected = []
        for subject in subjects_list:
            if subject == "SX106" or subject == "SX107":
                behavioral_file = Path(root, "sub-" + subject, "ses-" + ses,
                                       "sub-{}_ses-1_run-all_task-prp_events_repetition_1.csv".format(subject))
            else:
                behavioral_file = Path(root, "sub-" + subject, "ses-" + ses, file_name.format(subject))
            # Load the file:
            subject_data = pd.read_csv(behavioral_file, sep=",")
            # Add the subject ID:
            subject_data["subject"] = subject
            # Apply trial rejection:
            rej_ind = beh_exclusion(subject_data)
            prop_rej = len(rej_ind) / subject_data.shape[0]
            subject_data = subject_data.drop(rej_ind)
            # Append to the rest of the subject
            subjects_data.append(subject_data.reset_index(drop=True))
            # Print the proportion of trials that were discarded:
            print("Subject {} - {:.2f}% trials were discarded".format(subject, prop_rej * 100))
            prop_rejected.append(prop_rej)

        print("The mean proportion of rejected trials is {:.2f}% +- {:.2f}".format(np.mean(prop_rejected) * 100,
                                                                           np.std(prop_rejected) * 100))
    else:
        for subject in subjects_list:
            if subject == "SX106" or subject == "SX107":
                behavioral_file = Path(root, "sub-" + subject, "ses-" + ses,
                                       "sub-{}_ses-1_run-all_task-prp_events_repetition_1.csv".format(subject))
            else:
                behavioral_file = Path(root, "sub-" + subject, "ses-" + ses, file_name.format(subject))
            # Load the file:
            subject_data = pd.read_csv(behavioral_file, sep=",")
            # Add the subject ID:
            subject_data["subject"] = subject
            # Append to the rest of the subjects:
            subjects_data.append(subject_data.reset_index(drop=True))
    return pd.concat(subjects_data).reset_index(drop=True)


plot_check_plots = False
# ======================================================================================================================
# Set parameters:
root = r"P:\2023-0357-ReconTime\03_data\raw_data"
ses = "1"
subjects_list = ["SX101", "SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112",
                 "SX113", "SX114", "SX115", "SX116", "SX117", "SX118", "SX119", "SX120", "SX121", "SX123"]
file_name = "sub-{}_ses-1_run-all_task-prp_events.csv"
save_root = Path(ev.bids_root, "derivatives", "figures")
if not os.path.isdir(save_root):
    os.makedirs(save_root)
# Load the data without trial exclusion:
subjects_data_raw = load_beh_data(root, subjects_list, session='1', task='prp', do_trial_exclusion=False)

# Load the data:
subjects_data = load_beh_data(root, subjects_list, session='1', task='prp', do_trial_exclusion=True)

# ========================================================================================================
# Figure quality checks:
# ======================================================
# A: Accuracy:
fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=[8.3, 8.3])
# T1 target detection accuracy per category:
t1_accuracy = []
categories = ["face", "object", "letter", "false-font"]
for subject in subjects_data_raw["sub_id"].unique():
    for category in categories:
        # Extract the data:
        df = subjects_data_raw[(subjects_data_raw["sub_id"] == subject) &
                               (subjects_data_raw["category"] == category.replace("-", "_")) &
                               (subjects_data_raw["task_relevance"] == "target")]
        acc = (df["trial_response_vis"].value_counts().get("hit", 0) / df.shape[0]) * 100
        t1_accuracy.append([subject, category, acc])
t1_accuracy = pd.DataFrame(t1_accuracy, columns=["sub_id", "category", "T1_accuracy"])

# T2 pitch detection accuracy:
t2_accuracy = []
pitches = [1000, 1100]
for subject in subjects_data_raw["sub_id"].unique():
    for pitch in pitches:
        # Extract the data:
        df = subjects_data_raw[(subjects_data_raw["sub_id"] == subject) &
                               (subjects_data_raw["pitch"] == pitch) &
                               (subjects_data_raw["task_relevance"] == "target")]
        acc = (np.nansum(df["trial_accuracy_aud"].to_numpy()) / df.shape[0]) * 100
        t2_accuracy.append([subject, pitch, acc])
t2_accuracy = pd.DataFrame(t2_accuracy, columns=["sub_id", "pitch", "T2_accuracy"])

_, _, _ = plot_within_subject_boxplot(t1_accuracy,
                                      'sub_id', 'category', 'T1_accuracy',
                                      positions=[1, 2, 3, 4], ax=ax1[0], cousineau_correction=False,
                                      title="",
                                      xlabel="Category", ylabel="Accuracy (%)",
                                      xlim=None, width=0.5,
                                      face_colors=[val for val in ev.colors["category"].values()])
_, _, _ = plot_within_subject_boxplot(t2_accuracy,
                                      'sub_id', 'pitch', 'T2_accuracy',
                                      positions=[1, 2], ax=ax1[1], cousineau_correction=False,
                                      title="",
                                      xlabel="Pitch", ylabel="",
                                      xlim=None, width=0.3,
                                      face_colors=[val for val in ev.colors["pitch"].values()])
ax1[0].set_xticklabels(categories)
ax1[1].set_xticklabels(pitches)
ax1[0].spines['right'].set_visible(False)
ax1[1].spines['left'].set_visible(False)
ax1[1].yaxis.set_visible(False)
plt.subplots_adjust(wspace=0)
fig1.suptitle("Task performances")
fig1.savefig(Path(save_root, "t1t2_accuracy.svg"), transparent=True, dpi=dpi)
fig1.savefig(Path(save_root, "t1t2_accuracy.png"), transparent=True, dpi=dpi)
plt.close(fig1)

# ======================================================
# B: Reaction time:
fig2, ax2 = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=True, figsize=[8.3 / 2, 8.3])
_, _, _ = plot_within_subject_boxplot(subjects_data,
                                      'sub_id', 'category', 'RT_vis',
                                      positions=[1, 2, 3, 4], ax=ax2, cousineau_correction=False,
                                      title="",
                                      xlabel="Category", ylabel="Reaction time (sec.)",
                                      xlim=None, width=0.5,
                                      face_colors=[val for val in ev.colors["category"].values()])
ax2.set_xticklabels(categories)
fig2.suptitle("T1 reaction time")
fig2.savefig(Path(save_root, "t1_rt.svg"), transparent=True, dpi=dpi)
fig2.savefig(Path(save_root, "t1_rt.png"), transparent=True, dpi=dpi)
plt.close(fig2)

# ========================================================================================================
# Figure 5:
# ==================================
# Figure 5a:

# Target:
d = 1.5
fig_ta, ax_ta = soa_boxplot(subjects_data[subjects_data["task_relevance"] == 'target'],
                            "RT_aud",
                            fig_size=[8.3 / 3, 11.7 / 2])
# Task relevant:
fig_tr, ax_tr = soa_boxplot(subjects_data[subjects_data["task_relevance"] == 'non-target'],
                            "RT_aud",
                            fig_size=[8.3 / 3, 11.7 / 2])
# Task irrelevant:
fig_ti, ax_ti = soa_boxplot(subjects_data[subjects_data["task_relevance"] == 'irrelevant'],
                            "RT_aud",
                            fig_size=[8.3 / 3, 11.7 / 2])
# Set the y limit to be the same for both plots:
lims = [[ax_tr[0].get_ylim()[0], ax_ti[0].get_ylim()[0]], [ax_tr[0].get_ylim()[1], ax_ti[0].get_ylim()[1]]]
max_lims = [min(min(lims)), max(max(lims))]
ax_tr[0].set_ylim(max_lims)
ax_ti[0].set_ylim(max_lims)
# Axes decoration:
fig_ta.suptitle("Target")
fig_tr.suptitle("Relevant non-target")
fig_ti.suptitle("Irrelevant non-target")
fig_ta.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
fig_tr.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
fig_ti.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
fig_ta.text(0, 0.5, 'Reaction time (sec.)', ha='center', va='center', fontsize=18, rotation=90)
fig_tr.text(0, 0.5, 'Reaction time (sec.)', ha='center', va='center', fontsize=18, rotation=90)
fig_ti.text(0, 0.5, 'Reaction time (sec.)', ha='center', va='center', fontsize=18, rotation=90)
# Save the figures:
fig_ta.savefig(Path(save_root, "figure5a_target.svg"), transparent=True, dpi=dpi)
fig_ta.savefig(Path(save_root, "figure5a_target.png"), transparent=True, dpi=dpi)
fig_tr.savefig(Path(save_root, "figure5a_tr.svg"), transparent=True, dpi=dpi)
fig_tr.savefig(Path(save_root, "figure5a_tr.png"), transparent=True, dpi=dpi)
fig_ti.savefig(Path(save_root, "figure5a_ti.svg"), transparent=True, dpi=dpi)
fig_ti.savefig(Path(save_root, "figure5a_ti.png"), transparent=True, dpi=dpi)
plt.close(fig_ta)
plt.close(fig_tr)
plt.close(fig_ti)

# ==================================
# Figure 5b:
subjects_data["zscore_RT_aud"] = np.zeros(subjects_data.shape[0])
for sub_id in subjects_data["sub_id"].unique():
    subjects_data.loc[subjects_data["sub_id"] == sub_id, "zscore_RT_aud"] = (
        zscore(np.log(subjects_data.loc[subjects_data["sub_id"] == sub_id, "RT_aud"])))
# Task relevant onset:
fig_tr_onset, ax_tr_onset = plt.subplots(nrows=1, ncols=1, figsize=[8.3 / 2, 8.3 / 4])
for soa in list(subjects_data["SOA"].unique()):
    data = subjects_data[(subjects_data["task_relevance"] == 'non-target')
                         & (subjects_data["SOA_lock"] == 'onset')
                         & (subjects_data["SOA"] == soa)]["zscore_RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_tr_onset.ecdf(data, label=str(soa), color=ev.colors["soa_onset_locked"][str(soa)])
# Task irrelevant onset:
fig_ti_onset, ax_ti_onset = plt.subplots(nrows=1, ncols=1, figsize=[8.3 / 2, 8.3 / 4])
for soa in list(subjects_data["SOA"].unique()):
    data = subjects_data[(subjects_data["task_relevance"] == 'irrelevant')
                         & (subjects_data["SOA_lock"] == 'onset')
                         & (subjects_data["SOA"] == soa)]["zscore_RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_ti_onset.ecdf(data, label=str(soa), color=ev.colors["soa_onset_locked"][str(soa)])

# Offset 500:
fig_offset_short, ax_offset_short = plt.subplots(nrows=1, ncols=1, figsize=[8.3 / 3, 8.3 / 4])
for soa in list(subjects_data["SOA"].unique()):
    data = subjects_data[(subjects_data["task_relevance"].isin(['non-target', 'irrelevant']))
                         & (subjects_data["SOA_lock"] == 'offset')
                         & (subjects_data["SOA"] == soa)
                         & (subjects_data["duration"] == 0.5)]["zscore_RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_offset_short.ecdf(data, label=str(soa), color=ev.colors["soa_offset_locked"][str(soa)])

# Offset 1000:
fig_offset_int, ax_offset_int = plt.subplots(nrows=1, ncols=1, figsize=[8.3 / 3, 8.3 / 4])
for soa in list(subjects_data["SOA"].unique()):
    data = subjects_data[(subjects_data["task_relevance"].isin(['non-target', 'irrelevant']))
                         & (subjects_data["SOA_lock"] == 'offset')
                         & (subjects_data["SOA"] == soa)
                         & (subjects_data["duration"] == 1.0)]["zscore_RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_offset_int.ecdf(data, label=str(soa), color=ev.colors["soa_offset_locked"][str(soa)])

# Offset 1500:
fig_offset_long, ax_offset_long = plt.subplots(nrows=1, ncols=1, figsize=[8.3 / 3, 8.3 / 4])
for soa in list(subjects_data["SOA"].unique()):
    data = subjects_data[(subjects_data["task_relevance"].isin(['non-target', 'irrelevant']))
                         & (subjects_data["SOA_lock"] == 'offset')
                         & (subjects_data["SOA"] == soa)
                         & (subjects_data["duration"] == 1.5)]["zscore_RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_offset_long.ecdf(data, label=str(soa), color=ev.colors["soa_offset_locked"][str(soa)])

# Add axes decorations:
ax_tr_onset.set_ylabel("Cumulative Probability")
ax_tr_onset.set_xlabel("Corrected RT (sec.)")
ax_ti_onset.set_ylabel("Cumulative Probability")
ax_ti_onset.set_xlabel("Corrected RT (sec.)")
ax_offset_short.set_ylabel("Cumulative Probability")
ax_offset_short.set_xlabel("Corrected RT (sec.)")
ax_offset_int.set_ylabel("Cumulative Probability")
ax_offset_int.set_xlabel("Corrected RT (sec.)")
ax_offset_long.set_ylabel("Cumulative Probability")
ax_offset_long.set_xlabel("Corrected RT (sec.)")
ax_tr_onset.legend()
# Set the x limits the same across all figures:
xlims = [[ax_tr_onset.get_xlim()[0],
          ax_ti_onset.get_xlim()[0],
          ax_offset_short.get_xlim()[0],
          ax_offset_int.get_xlim()[0],
          ax_offset_long.get_xlim()[0]],
         [ax_tr_onset.get_xlim()[1],
          ax_ti_onset.get_xlim()[1],
          ax_offset_short.get_xlim()[1],
          ax_offset_int.get_xlim()[1],
          ax_offset_long.get_xlim()[1]]
         ]
xlims_new = [min(min(xlims)), max(max(xlims))]
ax_tr_onset.set_xlim(xlims_new)
ax_ti_onset.set_xlim(xlims_new)
ax_offset_short.set_xlim(xlims_new)
ax_offset_int.set_xlim(xlims_new)
ax_offset_long.set_xlim(xlims_new)
# Save the figures:
fig_tr_onset.savefig(Path(save_root, "figure5b_tr_onset.svg"), transparent=True, dpi=dpi)
fig_tr_onset.savefig(Path(save_root, "figure5b_tr_onset.png"), transparent=True, dpi=dpi)
fig_ti_onset.savefig(Path(save_root, "figure5b_ti_onset.svg"), transparent=True, dpi=dpi)
fig_ti_onset.savefig(Path(save_root, "figure5b_ti_onset.png"), transparent=True, dpi=dpi)
fig_offset_short.savefig(Path(save_root, "figure5b_offset_short.svg"), transparent=True, dpi=dpi)
fig_offset_short.savefig(Path(save_root, "figure5b_offset_short.png"), transparent=True, dpi=dpi)
fig_offset_int.savefig(Path(save_root, "figure5b_offset_int.svg"), transparent=True, dpi=dpi)
fig_offset_int.savefig(Path(save_root, "figure5b_offset_int.png"), transparent=True, dpi=dpi)
fig_offset_long.savefig(Path(save_root, "figure5b_offset_long.svg"), transparent=True, dpi=dpi)
fig_offset_long.savefig(Path(save_root, "figure5b_offset_long.png"), transparent=True, dpi=dpi)
plt.close(fig_ta)
plt.close(fig_tr)
plt.close(fig_ti)
