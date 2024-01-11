import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from eye_tracker.general_helper_function import cousineau_morey_correction
from eye_tracker.plotter_functions import plot_within_subject_boxplot
import environment_variables as ev


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


def perform_exclusion(data):
    """
    This function performs the exclusion criterion reported in the paper. Packaged in a function such that we can use
    it in various places:
    :param data:
    :return:
    """
    # 1. Remove the trials with wrong visual responses:
    data = data[data["trial_response_vis"] != "fa"]
    # 2. Remove the trials with wrong auditory responses:
    data = data[data["trial_accuracy_aud"] != 0]
    # 3. Remove trials where the visual stimuli were responded second:
    data = data[data["trial_second_button_press"] != 1]
    # 4. Remove trials in which the participants responded to the auditory stimulus in less than 100ms:
    data = data[data["RT_aud"] >= 0.1]
    # 5. Remove the trials in which the participants took more than 1.260sec to respond:
    data = data[data["RT_aud"] <= 1.260]

    return data


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

    for subject in subjects_list:
        if subject == "SX106" or subject == "SX107":
            behavioral_file = Path(root, "sub-" + subject, "ses-" + ses,
                                   "sub-{}_ses-1_run-all_task-prp_events_repetition_1.csv".format(subject))
        else:
            behavioral_file = Path(root, "sub-" + subject, "ses-" + ses, file_name.format(subject))
        # Load the file:
        subject_data = pd.read_csv(behavioral_file, sep=",")

        # Apply trial rejection:
        n_trial = len(subject_data)
        subject_data = perform_exclusion(subject_data)
        # Add the subject ID:
        subject_data["subject"] = subject
        subjects_data.append(subject_data.reset_index(drop=True))
        # Print the proportion of trials that were discarded:
        print("Subject {} - {}% trials were discarded".format(subject,
                                                              ((n_trial - len(subject_data)) / n_trial) * 100))

    # Concatenate the data:
    return pd.concat(subjects_data).reset_index(drop=True)


# ======================================================================================================================
# Set parameters:
root = r"P:\2023-0357-ReconTime\03_data\raw_data"
ses = "1"
subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112",
                 "SX113", "SX114", "SX115", "SX116", "SX117", "SX118", "SX119", "SX120", "SX121", "SX123"]
file_name = "sub-{}_ses-1_run-all_task-prp_events.csv"
save_root = Path(ev.bids_root, "derivatives", "figures")
if not os.path.isdir(save_root):
    os.makedirs(save_root)
# Load the data:
subjects_data = load_beh_data(root, subjects_list, session='1', task='prp', do_trial_exclusion=True)

# ========================================================================================================
# Figure 5:

# ==================================
# Figure 5a:

# Target:
d = 1.5
fig_ta, ax_ta = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=True, figsize=[8.3/3, 11.7 / 2])
_, _, _ = plot_within_subject_boxplot(subjects_data[(subjects_data["task_relevance"] == 'target')
                                                    & (subjects_data["SOA_lock"] == 'onset')],
                                      'sub_id', 'onset_SOA', 'RT_aud',
                                      positions='onset_SOA', ax=ax_ta[0], cousineau_correction=True,
                                      title="",
                                      xlabel="", ylabel="",
                                      xlim=[-0.1, 0.6], width=0.1,
                                      face_colors=[val for val in ev.colors["soa_onset_locked"].values()])
# Loop through each duration to plot the offset locked SOA separately:
for i, dur in enumerate(sorted(list(subjects_data["duration"].unique()))):
    _, _, _ = plot_within_subject_boxplot(subjects_data[(subjects_data["task_relevance"] == 'target')
                                                        & (subjects_data["SOA_lock"] == 'offset')
                                                        & (subjects_data["duration"] == dur)], 'sub_id',
                                          'onset_SOA',
                                          'RT_aud',
                                          positions='onset_SOA', ax=ax_ta[i + 1], cousineau_correction=True,
                                          title="",
                                          xlabel="", ylabel="",
                                          xlim=[dur-0.1, dur + 0.6], width=0.1,
                                          face_colors=[val for val in ev.colors["soa_offset_locked_{}ms".format(int(dur*1000))].values()])
    ax_ta[i + 1].yaxis.set_visible(False)
# Remove the spines:
for i in [0, 1, 2]:
    ax_ta[i].spines['right'].set_visible(False)
    ax_ta[i + 1].spines['left'].set_visible(False)
    # Add cut axis marks:
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_ta[i].plot([1, 1], [0, 0], transform=ax_ta[i].transAxes, **kwargs)
    ax_ta[i + 1].plot([0, 0], [0, 0], transform=ax_ta[i + 1].transAxes, **kwargs)
    ax_ta[i].plot([1, 1], [1, 1], transform=ax_ta[i].transAxes, **kwargs)
    ax_ta[i + 1].plot([0, 0], [1, 1], transform=ax_ta[i + 1].transAxes, **kwargs)
plt.subplots_adjust(wspace=0.05)

# Task relevant:
fig_tr, ax_tr = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=True, figsize=[8.3/3, 11.7 / 2])
_, _, _ = plot_within_subject_boxplot(subjects_data[(subjects_data["task_relevance"] == 'non-target')
                                                    & (subjects_data["SOA_lock"] == 'onset')], 'sub_id', 'onset_SOA',
                                      'RT_aud',
                                      positions='onset_SOA', ax=ax_tr[0], cousineau_correction=True,
                                      title="",
                                      xlabel="", ylabel="",
                                      xlim=[-0.1, 0.6], width=0.1,
                                      face_colors=[val for val in ev.colors["soa_onset_locked"].values()])
# Loop through each duration to plot the offset locked SOA separately:
for i, dur in enumerate(sorted(list(subjects_data["duration"].unique()))):
    _, _, _ = plot_within_subject_boxplot(subjects_data[(subjects_data["task_relevance"] == 'non-target')
                                                        & (subjects_data["SOA_lock"] == 'offset')
                                                        & (subjects_data["duration"] == dur)], 'sub_id',
                                          'onset_SOA',
                                          'RT_aud',
                                          positions='onset_SOA', ax=ax_tr[i + 1], cousineau_correction=True,
                                          title="",
                                          xlabel="", ylabel="",
                                          xlim=[dur-0.1, dur + 0.6], width=0.1,
                                          face_colors=[val for val in ev.colors["soa_offset_locked_{}ms".format(int(dur*1000))].values()])
    ax_tr[i + 1].yaxis.set_visible(False)
# Remove the spines:
for i in [0, 1, 2]:
    ax_tr[i].spines['right'].set_visible(False)
    ax_tr[i + 1].spines['left'].set_visible(False)
    # Add cut axis marks:
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_tr[i].plot([1, 1], [0, 0], transform=ax_tr[i].transAxes, **kwargs)
    ax_tr[i + 1].plot([0, 0], [0, 0], transform=ax_tr[i + 1].transAxes, **kwargs)
    ax_tr[i].plot([1, 1], [1, 1], transform=ax_tr[i].transAxes, **kwargs)
    ax_tr[i + 1].plot([0, 0], [1, 1], transform=ax_tr[i + 1].transAxes, **kwargs)
plt.subplots_adjust(wspace=0.05)

# Task irrelevant:
fig_ti, ax_ti = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=True, figsize=[8.3/3, 11.7 / 2])
_, _, _ = plot_within_subject_boxplot(subjects_data[(subjects_data["task_relevance"] == 'irrelevant')
                                                    & (subjects_data["SOA_lock"] == 'onset')], 'sub_id', 'onset_SOA',
                                      'RT_aud',
                                      positions='onset_SOA', ax=ax_ti[0], cousineau_correction=True,
                                      title="",
                                      xlabel="", ylabel="",
                                      xlim=[-0.1, 0.6], width=0.1,
                                      face_colors=[val for val in ev.colors["soa_onset_locked"].values()])
# Loop through each duration to plot the offset locked SOA separately:
for i, dur in enumerate(sorted(list(subjects_data["duration"].unique()))):
    _, _, _ = plot_within_subject_boxplot(subjects_data[(subjects_data["task_relevance"] == 'irrelevant')
                                                        & (subjects_data["SOA_lock"] == 'offset')
                                                        & (subjects_data["duration"] == dur)], 'sub_id',
                                          'onset_SOA',
                                          'RT_aud',
                                          positions='onset_SOA', ax=ax_ti[i + 1], cousineau_correction=True,
                                          title="",
                                          xlabel="", ylabel="",
                                          xlim=[dur-0.1, dur + 0.6], width=0.1,
                                          face_colors=[val for val in ev.colors["soa_offset_locked_{}ms".format(int(dur*1000))].values()])
    ax_ti[i + 1].yaxis.set_visible(False)
# Remove the spines:
for i in [0, 1, 2]:
    ax_ti[i].spines['right'].set_visible(False)
    ax_ti[i + 1].spines['left'].set_visible(False)
    # Add cut axis marks:
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_ti[i].plot([1, 1], [0, 0], transform=ax_ti[i].transAxes, **kwargs)
    ax_ti[i + 1].plot([0, 0], [0, 0], transform=ax_ti[i + 1].transAxes, **kwargs)
    ax_ti[i].plot([1, 1], [1, 1], transform=ax_ti[i].transAxes, **kwargs)
    ax_ti[i + 1].plot([0, 0], [1, 1], transform=ax_ti[i + 1].transAxes, **kwargs)
plt.subplots_adjust(wspace=0.05)

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
# Cousineau Morey correction of the data:
subject_data_corrected = cousineau_morey_correction(subjects_data, 'sub_id', 'onset_SOA', 'RT_aud')

# Task relevant onset:
fig_tr_onset, ax_tr_onset = plt.subplots(nrows=1, ncols=1, figsize=[8.3/2, 8.3 / 4])
for soa in list(subject_data_corrected["SOA"].unique()):
    data = subjects_data[(subjects_data["task_relevance"] == 'non-target')
                         & (subjects_data["SOA_lock"] == 'onset')
                         & (subjects_data["SOA"] == soa)]["RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_tr_onset.ecdf(data, label=str(soa), color=ev.colors["soa_onset_locked"][str(soa)])
# Task irrelevant onset:
fig_ti_onset, ax_ti_onset = plt.subplots(nrows=1, ncols=1, figsize=[8.3/2, 8.3 / 4])
for soa in list(subject_data_corrected["SOA"].unique()):
    data = subjects_data[(subjects_data["task_relevance"] == 'irrelevant')
                         & (subjects_data["SOA_lock"] == 'onset')
                         & (subjects_data["SOA"] == soa)]["RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_ti_onset.ecdf(data, label=str(soa), color=ev.colors["soa_onset_locked"][str(soa)])


# Offset 500:
fig_offset_short, ax_offset_short = plt.subplots(nrows=1, ncols=1, figsize=[8.3/3, 8.3 / 4])
for soa in list(subject_data_corrected["SOA"].unique()):
    data = subjects_data[(subjects_data["task_relevance"].isin(['non-target', 'irrelevant']))
                         & (subjects_data["SOA_lock"] == 'offset')
                         & (subjects_data["SOA"] == soa)
                         & (subjects_data["duration"] == 0.5)]["RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_offset_short.ecdf(data, label=str(soa), color=ev.colors["soa_offset_locked_500ms"][str(soa)])

# Offset 1000:
fig_offset_int, ax_offset_int = plt.subplots(nrows=1, ncols=1, figsize=[8.3/3, 8.3 / 4])
for soa in list(subject_data_corrected["SOA"].unique()):
    data = subjects_data[(subjects_data["task_relevance"].isin(['non-target', 'irrelevant']))
                         & (subjects_data["SOA_lock"] == 'offset')
                         & (subjects_data["SOA"] == soa)
                         & (subjects_data["duration"] == 1.0)]["RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_offset_int.ecdf(data, label=str(soa), color=ev.colors["soa_offset_locked_1000ms"][str(soa)])

# Offset 1500:
fig_offset_long, ax_offset_long = plt.subplots(nrows=1, ncols=1, figsize=[8.3/3, 8.3 / 4])
for soa in list(subject_data_corrected["SOA"].unique()):
    data = subjects_data[(subjects_data["task_relevance"].isin(['non-target', 'irrelevant']))
                         & (subjects_data["SOA_lock"] == 'offset')
                         & (subjects_data["SOA"] == soa)
                         & (subjects_data["duration"] == 1.5)]["RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_offset_long.ecdf(data, label=str(soa), color=ev.colors["soa_offset_locked_1500ms"][str(soa)])

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
