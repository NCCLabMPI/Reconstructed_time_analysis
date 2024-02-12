import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from helper_function.helper_plotter import plot_within_subject_boxplot, soa_boxplot
import environment_variables as ev
from helper_function.helper_general import load_beh_data

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
sns.set_style("white")

# ======================================================================================================================
# PRP results:
ses = "1"
task = "prp"
file_name = "sub-{}_ses-{}_run-all_task-{}_events.csv"
save_root = Path(ev.bids_root, "derivatives", "figures")
if not os.path.isdir(save_root):
    os.makedirs(save_root)
# Load the data without trial exclusion:
data_df_raw = load_beh_data(ev.bids_root, ev.subjects_lists_beh[task], file_name, session=ses, task=task,
                            do_trial_exclusion=False)

# Load the data:
data_df = load_beh_data(ev.bids_root, ev.subjects_lists_beh[task], file_name, session=ses, task=task,
                        do_trial_exclusion=True)

# ========================================================================================================
# Figure quality checks:
# ======================================================
# A: Accuracy:
fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=[8.3, 8.3])
# T1 target detection accuracy per category:
t1_accuracy = []
categories = ["face", "object", "letter", "false-font"]
for subject in data_df_raw["sub_id"].unique():
    for category in categories:
        # Extract the data:
        df = data_df_raw[(data_df_raw["sub_id"] == subject) &
                         (data_df_raw["category"] == category.replace("-", "_")) &
                         (data_df_raw["task_relevance"] == "target")]
        acc = (df["trial_response_vis"].value_counts().get("hit", 0) / df.shape[0]) * 100
        t1_accuracy.append([subject, category, acc])
t1_accuracy = pd.DataFrame(t1_accuracy, columns=["sub_id", "category", "T1_accuracy"])

# T2 pitch detection accuracy:
t2_accuracy = []
pitches = [1000, 1100]
for subject in data_df_raw["sub_id"].unique():
    for pitch in pitches:
        # Extract the data:
        df = data_df_raw[(data_df_raw["sub_id"] == subject) &
                         (data_df_raw["pitch"] == pitch) &
                         (data_df_raw["task_relevance"] == "target")]
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
_, _, _ = plot_within_subject_boxplot(data_df,
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
fig_ta, ax_ta = soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
                            "RT_aud",
                            fig_size=[8.3 / 3, 11.7 / 2])
# Task relevant:
fig_tr, ax_tr = soa_boxplot(data_df[data_df["task_relevance"] == 'non-target'],
                            "RT_aud",
                            fig_size=[8.3 / 3, 11.7 / 2])
# Task irrelevant:
fig_ti, ax_ti = soa_boxplot(data_df[data_df["task_relevance"] == 'irrelevant'],
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
data_df["zscore_RT_aud"] = np.zeros(data_df.shape[0])
for sub_id in data_df["sub_id"].unique():
    data_df.loc[data_df["sub_id"] == sub_id, "zscore_RT_aud"] = (
        zscore(np.log(data_df.loc[data_df["sub_id"] == sub_id, "RT_aud"])))
# Task relevant onset:
fig_tr_onset, ax_tr_onset = plt.subplots(nrows=1, ncols=1, figsize=[8.3 / 2, 8.3 / 4])
for soa in list(data_df["SOA"].unique()):
    data = data_df[(data_df["task_relevance"] == 'non-target')
                   & (data_df["SOA_lock"] == 'onset')
                   & (data_df["SOA"] == soa)]["zscore_RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_tr_onset.ecdf(data, label=str(soa), color=ev.colors["soa_onset_locked"][str(soa)])
# Task irrelevant onset:
fig_ti_onset, ax_ti_onset = plt.subplots(nrows=1, ncols=1, figsize=[8.3 / 2, 8.3 / 4])
for soa in list(data_df["SOA"].unique()):
    data = data_df[(data_df["task_relevance"] == 'irrelevant')
                   & (data_df["SOA_lock"] == 'onset')
                   & (data_df["SOA"] == soa)]["zscore_RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_ti_onset.ecdf(data, label=str(soa), color=ev.colors["soa_onset_locked"][str(soa)])

# Offset 500:
fig_offset_short, ax_offset_short = plt.subplots(nrows=1, ncols=1, figsize=[8.3 / 3, 8.3 / 4])
for soa in list(data_df["SOA"].unique()):
    data = data_df[(data_df["task_relevance"].isin(['non-target', 'irrelevant']))
                   & (data_df["SOA_lock"] == 'offset')
                   & (data_df["SOA"] == soa)
                   & (data_df["duration"] == 0.5)]["zscore_RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_offset_short.ecdf(data, label=str(soa), color=ev.colors["soa_offset_locked"][str(soa)])

# Offset 1000:
fig_offset_int, ax_offset_int = plt.subplots(nrows=1, ncols=1, figsize=[8.3 / 3, 8.3 / 4])
for soa in list(data_df["SOA"].unique()):
    data = data_df[(data_df["task_relevance"].isin(['non-target', 'irrelevant']))
                   & (data_df["SOA_lock"] == 'offset')
                   & (data_df["SOA"] == soa)
                   & (data_df["duration"] == 1.0)]["zscore_RT_aud"].to_numpy()
    # Remove the nans:
    data = data[~np.isnan(data)]
    ax_offset_int.ecdf(data, label=str(soa), color=ev.colors["soa_offset_locked"][str(soa)])

# Offset 1500:
fig_offset_long, ax_offset_long = plt.subplots(nrows=1, ncols=1, figsize=[8.3 / 3, 8.3 / 4])
for soa in list(data_df["SOA"].unique()):
    data = data_df[(data_df["task_relevance"].isin(['non-target', 'irrelevant']))
                   & (data_df["SOA_lock"] == 'offset')
                   & (data_df["SOA"] == soa)
                   & (data_df["duration"] == 1.5)]["zscore_RT_aud"].to_numpy()
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
plt.close(fig_tr_onset)
plt.close(fig_ti_onset)
plt.close(fig_offset_short)
plt.close(fig_offset_int)
plt.close(fig_offset_long)

# ======================================================================================================================
# Introspection results:
sessions = ["2", "3"]
task = "introspection"
file_name = "sub-{}_ses-{}_run-all_task-{}_events.csv"
save_root = Path(ev.bids_root, "derivatives", "figures")
if not os.path.isdir(save_root):
    os.makedirs(save_root)
data_df_raw = []
data_df = []
for ses in sessions:
    # Load the data without trial exclusion:
    data_df_raw.append(load_beh_data(ev.bids_root, ev.subjects_lists_beh[task], file_name, session=ses, task=task,
                                     do_trial_exclusion=False))

    # Load the data:
    data_df.append(load_beh_data(ev.bids_root, ev.subjects_lists_beh[task], file_name, session=ses[0], task=task,
                                 do_trial_exclusion=True))
data_df_raw = pd.concat(data_df_raw).reset_index(drop=True)
data_df = pd.concat(data_df).reset_index(drop=True)
# Convert iRT to sec:
data_df["iRT_aud"] = data_df["iRT_aud"] * 0.001
data_df["iRT_vis"] = data_df["iRT_vis"] * 0.001
# Z score RT and iRT separately for each subject:
data_df["zRT_aud"] = np.zeros(data_df.shape[0])
data_df["ziRT_aud"] = np.zeros(data_df.shape[0])
data_df["ziRT_vis"] = np.zeros(data_df.shape[0])
for sub_id in data_df["sub_id"].unique():
    data_df.loc[data_df["sub_id"] == sub_id, "zRT_aud"] = (
        zscore(np.log(data_df.loc[data_df["sub_id"] == sub_id, "RT_aud"])))
    data_df.loc[data_df["sub_id"] == sub_id, "ziRT_aud"] = (
        zscore(np.log(data_df.loc[data_df["sub_id"] == sub_id, "iRT_aud"])))
    data_df.loc[data_df["sub_id"] == sub_id, "ziRT_vis"] = (
        zscore(np.log(data_df.loc[data_df["sub_id"] == sub_id, "iRT_vis"])))

# ========================================================================
# Plot RT and iT:
# Extract the colors:
onset_lock_rt = [ev.colors["soa_onset_locked"][str(soa)] for soa in list(data_df["SOA"].unique())]
offset_lock_rt = [ev.colors["soa_offset_locked"][str(soa)] for soa in list(data_df["SOA"].unique())]
onset_lock_it = [ev.colors["soa_onset_locked_iRT"][str(soa)] for soa in list(data_df["SOA"].unique())]
offset_lock_it = [ev.colors["soa_offset_locked_iRT"][str(soa)] for soa in list(data_df["SOA"].unique())]
d = 1.5

# ========================================
# Target:
# RT:
fig_ta, ax_ta = soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
                            "RT_aud", colors_onset_locked=onset_lock_rt,
                            colors_offset_locked=offset_lock_rt,
                            fig_size=[8.3 / 3, 11.7 / 2])
# IT:
fig_ta, ax_ta = soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
                            "iRT_aud",
                            ax=ax_ta, colors_onset_locked=onset_lock_it,
                            colors_offset_locked=offset_lock_it,
                            fig_size=[8.3 / 3, 11.7 / 2], fig=fig_ta)

# ========================================
# Task relevant:
# RT:
fig_tr, ax_tr = soa_boxplot(data_df[data_df["task_relevance"] == 'non-target'],
                            "RT_aud", colors_onset_locked=onset_lock_rt,
                            colors_offset_locked=offset_lock_rt,
                            fig_size=[8.3 / 3, 11.7 / 2])
# IT:
fig_tr, ax_tr = soa_boxplot(data_df[data_df["task_relevance"] == 'non-target'],
                            "iRT_aud",
                            ax=ax_tr, colors_onset_locked=onset_lock_it,
                            colors_offset_locked=offset_lock_it,
                            fig_size=[8.3 / 3, 11.7 / 2], fig=fig_tr)

# ========================================
# Task irrelevant:
# RT:
fig_ti, ax_ti = soa_boxplot(data_df[data_df["task_relevance"] == 'irrelevant'],
                            "RT_aud",
                            colors_onset_locked=onset_lock_rt,
                            colors_offset_locked=offset_lock_rt,
                            fig_size=[8.3 / 3, 11.7 / 2])
# IT:
fig_ti, ax_ti = soa_boxplot(data_df[data_df["task_relevance"] == 'irrelevant'],
                            "iRT_aud",
                            ax=ax_ti, colors_onset_locked=onset_lock_it,
                            colors_offset_locked=offset_lock_it,
                            fig_size=[8.3 / 3, 11.7 / 2], fig=fig_ti)

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
fig_ta.savefig(Path(save_root, "introspection_target.svg"), transparent=True, dpi=dpi)
fig_ta.savefig(Path(save_root, "introspection_target.png"), transparent=True, dpi=dpi)
fig_tr.savefig(Path(save_root, "introspection_tr.svg"), transparent=True, dpi=dpi)
fig_tr.savefig(Path(save_root, "introspection_tr.png"), transparent=True, dpi=dpi)
fig_ti.savefig(Path(save_root, "introspection_ti.svg"), transparent=True, dpi=dpi)
fig_ti.savefig(Path(save_root, "introspection_ti.png"), transparent=True, dpi=dpi)
plt.close(fig_ta)
plt.close(fig_tr)
plt.close(fig_ti)

# ========================================================================
# Plot regression between iT and RT:
markers = ["v", "^", ">"]
fig_size = [8.3, 8.3]
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=fig_size)

# Loop through onset and offset locked:
for i, lock in enumerate(data_df["SOA_lock"].unique()):
    for ii, soa in enumerate(sorted(list((data_df["SOA"].unique())))):
        sns.regplot(x="ziRT_aud", y="zRT_aud", data=data_df[(data_df["SOA_lock"] == lock) & (data_df["SOA"] == soa)],
                    ax=ax[i], color=ev.colors["soa_{}_locked".format(lock)][str(soa)],
                    scatter_kws={'alpha': 0.5, 's': 7.5}, label=soa, marker=markers[ii])
ax[0].set_title("Onset locked")
ax[0].set_xlabel("")
ax[0].set_ylabel("")
ax[0].legend()
ax[1].set_title("Offset locked")
ax[1].set_xlabel("")
ax[1].set_ylabel("")
ax[1].legend()
fig.supxlabel("z-score iT")
fig.supylabel("z-score RT")
fig.savefig(Path(save_root, "RT-vs-iRT.svg"), transparent=True, dpi=dpi)
fig.savefig(Path(save_root, "RT-vs-iRT.png"), transparent=True, dpi=dpi)
plt.close(fig)
