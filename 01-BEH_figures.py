import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from helper_function.helper_plotter import soa_boxplot
import environment_variables as ev
from helper_function.helper_general import load_beh_data

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

# ======================================================================================================================
# Experiment 1:
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
# ==================================
# Figure 2:
# Figure 2a:
colors_onset = [ev.colors["soa_onset_locked"][str(soa)] for soa in np.sort(data_df["SOA"].unique())]
colors_offset = [ev.colors["soa_offset_locked"][str(soa)] for soa in np.sort(data_df["SOA"].unique())]
# Target:
d = 1.5
# RT vis:
fig_ta, ax_ta = soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
                            "RT_vis",
                            colors_onset_locked=[[0.5, 0.5, 0.5]] * 4, colors_offset_locked=[[0.5, 0.5, 0.5]] * 4,
                            fig_size=[8.3 / 2, 11.7 / 2], avg_line_color=[0.5, 0.5, 0.5], zorder=0,
                            alpha=0.5, label="RT visual", jitter=-0.04)
# RT audio
soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
            "RT_aud", ax=ax_ta,
            colors_onset_locked=colors_onset, colors_offset_locked=colors_offset,
            fig_size=[8.3 / 2, 11.7 / 2], zorder=10000, label="RT audio", jitter=0.04)
for ax in ax_ta:
    ax.axhline(0.64, linestyle="--", color=[0.7, 0.7, 0.7], alpha=0.8)
# Task relevant:
fig_tr, ax_tr = soa_boxplot(data_df[data_df["task_relevance"] == 'non-target'],
                            "RT_aud",
                            colors_onset_locked=colors_onset, colors_offset_locked=colors_offset,
                            fig_size=[8.3 / 2, 11.7 / 2], label="RT audio")
# Task irrelevant:
fig_ti, ax_ti = soa_boxplot(data_df[data_df["task_relevance"] == 'irrelevant'],
                            "RT_aud",
                            colors_onset_locked=colors_onset, colors_offset_locked=colors_offset,
                            fig_size=[8.3 / 2, 11.7 / 2], label="RT audio")
# Set the y limit to be the same for both plots:
lims = [[ax_tr[0].get_ylim()[0], ax_ti[0].get_ylim()[0]], [ax_tr[0].get_ylim()[1], ax_ti[0].get_ylim()[1]]]
max_lims = [min(min(lims)), max(max(lims))]
ax_tr[0].set_ylim(max_lims)
ax_ti[0].set_ylim(max_lims)
# Axes decoration:
fig_ta.suptitle("Target")
fig_tr.suptitle("Relevant non-target")
fig_ti.suptitle("Irrelevant non-target")
fig_ta.text(0.5, 0, 'Time (s)', ha='center', va='center')
fig_tr.text(0.5, 0, 'Time (s)', ha='center', va='center')
fig_ti.text(0.5, 0, 'Time (s)', ha='center', va='center')
fig_ta.text(0, 0.9, 'RT (s)', ha='center', va='center', fontsize=18)
fig_tr.text(0, 0.9, 'RT (s)', ha='center', va='center', fontsize=18)
fig_ti.text(0, 0.9, 'RT (s)', ha='center', va='center', fontsize=18)
# Save the figures:
fig_ta.savefig(Path(save_root, "Experiment1-figure2a_target.svg"), transparent=True, dpi=dpi)
fig_ta.savefig(Path(save_root, "Experiment1-figure2a_target.png"), transparent=True, dpi=dpi)
fig_tr.savefig(Path(save_root, "Experiment1-figure2a_tr.svg"), transparent=True, dpi=dpi)
fig_tr.savefig(Path(save_root, "Experiment1-figure2a_tr.png"), transparent=True, dpi=dpi)
fig_ti.savefig(Path(save_root, "Experiment1-figure2a_ti.svg"), transparent=True, dpi=dpi)
fig_ti.savefig(Path(save_root, "Experiment1-figure2a_ti.png"), transparent=True, dpi=dpi)
plt.close(fig_ta)
plt.close(fig_tr)
plt.close(fig_ti)

# =======================================================================================
# Figure 2b:
figsize = [8.3, 8.3 * (2 / 3)]
tasks = ["target", "non-target", "irrelevant"]
soas = list(np.sort(list(data_df["SOA"].unique())))
durations = list(np.sort(list(data_df["duration"].unique())))
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=figsize, sharex=True, sharey=True)
xlims = []
# ===============================================
# Onset locked:
# Plotting the onset locked  separately for each task relevances:
for i, tsk in enumerate(tasks):
    for soa in soas:
        data = data_df[(data_df["task_relevance"] == tsk)
                       & (data_df["SOA_lock"] == 'onset')
                       & (data_df["SOA"] == soa)]["RT_aud"].to_numpy()
        # Remove the nans:
        data = data[~np.isnan(data)]
        # Compute the 1 and 99 percentile of the distribution to clip the data
        xlims.append(np.percentile(data, [2.5, 97.5]))
        ax[0, i].ecdf(data, label=str(int(soa * 1000)) + " ms", color=ev.colors["soa_onset_locked"][str(soa)])

# ===============================================
# Offset locked:
# For the offset locked, plot separately per task demands and duration:
for dur_i, dur in enumerate(durations):
    for tsk_i, tsk in enumerate(tasks):
        for soa in soas:
            data = data_df[(data_df["task_relevance"] == tsk)
                           & (data_df["SOA_lock"] == 'offset')
                           & (data_df["SOA"] == soa)
                           & (data_df["duration"] == dur)]["RT_aud"].to_numpy()
            # Remove the nans:
            data = data[~np.isnan(data)]
            xlims.append(np.percentile(data, [2.5, 97.5]))
            ax[dur_i + 1, tsk_i].ecdf(data, label=str(int(soa * 1000)) + " ms",
                                      color=ev.colors["soa_offset_locked"][str(soa)])
# Add axis decorations:
fig.supxlabel("RT (s)")
fig.supylabel("CDF")
ax[0, 0].set_xlim([np.min(xlims), np.max(xlims)])
fig.savefig(Path(save_root, "Experiment1-figure2b.svg"), transparent=True, dpi=dpi)
fig.savefig(Path(save_root, "Experiment1-figure2b.png"), transparent=True, dpi=dpi)
plt.close(fig)

# ======================================================================================================================
# Experiment 2:
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

# ==================================
# Figure 2:
# Figure 2a:
onset_lock_rt = [ev.colors["soa_onset_locked"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]
offset_lock_rt = [ev.colors["soa_offset_locked"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]
onset_lock_it = [ev.colors["soa_onset_locked_iRT"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]
offset_lock_it = [ev.colors["soa_offset_locked_iRT"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]
onset_lock_visit = [ev.colors["soa_onset_locked_visiRT"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]
offset_lock_visit = [ev.colors["soa_offset_locked_visiRT"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]

# Target:
d = 1.5
# RT vis:
fig_ta, ax_ta = soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
                            "RT_vis",
                            colors_onset_locked=[[0.5, 0.5, 0.5]] * 4, colors_offset_locked=[[0.5, 0.5, 0.5]] * 4,
                            fig_size=[8.3 / 2, 11.7 / 2], avg_line_color=[0.5, 0.5, 0.5], zorder=0,
                            alpha=0.5, label="RT visual", jitter=-0.04)
# RT audio
soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
            "RT_aud", ax=ax_ta,
            colors_onset_locked=onset_lock_rt, colors_offset_locked=offset_lock_rt,
            fig_size=[8.3 / 2, 11.7 / 2], zorder=10000, label="RT audio", jitter=0.04)
# iRT audio
soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
            "iRT_aud", ax=ax_ta,
            colors_onset_locked=onset_lock_it, colors_offset_locked=offset_lock_it,
            fig_size=[8.3 / 2, 11.7 / 2], zorder=10000, label="iT audio", jitter=-0.04)
# iRT vis
soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
            "iRT_vis", ax=ax_ta,
            colors_onset_locked=onset_lock_visit, colors_offset_locked=offset_lock_visit,
            fig_size=[8.3 / 2, 11.7 / 2], zorder=10000, label="iT visual", jitter=0.04)

for ax in ax_ta:
    ax.axhline(0.64, linestyle="--", color=[0.7, 0.7, 0.7], alpha=0.8)
# Task relevant:
fig_tr, ax_tr = soa_boxplot(data_df[data_df["task_relevance"] == 'non-target'],
                            "RT_aud",
                            colors_onset_locked=colors_onset, colors_offset_locked=colors_offset,
                            fig_size=[8.3 / 2, 11.7 / 2], label="RT audio")
soa_boxplot(data_df[data_df["task_relevance"] == 'non-target'],
            "iRT_aud", ax=ax_tr,
            colors_onset_locked=onset_lock_it, colors_offset_locked=offset_lock_it,
            fig_size=[8.3 / 2, 11.7 / 2], label="iT audio")
soa_boxplot(data_df[data_df["task_relevance"] == 'non-target'],
            "iRT_vis", ax=ax_tr,
            colors_onset_locked=onset_lock_visit, colors_offset_locked=offset_lock_visit,
            fig_size=[8.3 / 2, 11.7 / 2], label="iT visual")
# Task irrelevant:
fig_ti, ax_ti = soa_boxplot(data_df[data_df["task_relevance"] == 'irrelevant'],
                            "RT_aud",
                            colors_onset_locked=colors_onset, colors_offset_locked=colors_offset,
                            fig_size=[8.3 / 2, 11.7 / 2], label="RT audio")
soa_boxplot(data_df[data_df["task_relevance"] == 'irrelevant'],
            "iRT_aud", ax=ax_ti,
            colors_onset_locked=onset_lock_it, colors_offset_locked=offset_lock_it,
            fig_size=[8.3 / 2, 11.7 / 2], label="iT audio")
soa_boxplot(data_df[data_df["task_relevance"] == 'irrelevant'],
            "iRT_vis", ax=ax_ti,
            colors_onset_locked=onset_lock_visit, colors_offset_locked=offset_lock_visit,
            fig_size=[8.3 / 2, 11.7 / 2], label="iT visual")

# Set the y limit to be the same for both plots:
lims = [[ax_tr[0].get_ylim()[0], ax_ti[0].get_ylim()[0]], [ax_tr[0].get_ylim()[1], ax_ti[0].get_ylim()[1]]]
max_lims = [min(min(lims)), max(max(lims))]
ax_tr[0].set_ylim(max_lims)
ax_ti[0].set_ylim(max_lims)
# Axes decoration:
fig_ta.suptitle("Target")
fig_tr.suptitle("Relevant non-target")
fig_ti.suptitle("Irrelevant non-target")
fig_ta.text(0.5, 0, 'Time (s)', ha='center', va='center')
fig_tr.text(0.5, 0, 'Time (s)', ha='center', va='center')
fig_ti.text(0.5, 0, 'Time (s)', ha='center', va='center')
fig_ta.text(0, 0.9, 'RT (s)', ha='center', va='center', fontsize=18)
fig_tr.text(0, 0.9, 'RT (s)', ha='center', va='center', fontsize=18)
fig_ti.text(0, 0.9, 'RT (s)', ha='center', va='center', fontsize=18)
# Save the figures:
fig_ta.savefig(Path(save_root, "Experiment2-figure2a_target.svg"), transparent=True, dpi=dpi)
fig_ta.savefig(Path(save_root, "Experiment2-figure2a_target.png"), transparent=True, dpi=dpi)
fig_tr.savefig(Path(save_root, "Experiment2-figure2a_tr.svg"), transparent=True, dpi=dpi)
fig_tr.savefig(Path(save_root, "Experiment2-figure2a_tr.png"), transparent=True, dpi=dpi)
fig_ti.savefig(Path(save_root, "Experiment2-figure2a_ti.svg"), transparent=True, dpi=dpi)
fig_ti.savefig(Path(save_root, "Experiment2-figure2a_ti.png"), transparent=True, dpi=dpi)
plt.close(fig_ta)
plt.close(fig_tr)
plt.close(fig_ti)

# =======================================================================================
# Figure 2b:
figsize = [8.3, 8.3 * (2 / 3)]
tasks = ["target", "non-target", "irrelevant"]
soas = list(np.sort(list(data_df["SOA"].unique())))
durations = list(np.sort(list(data_df["duration"].unique())))
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=figsize, sharex=True, sharey=True)
xlims = []
# ===============================================
# Onset locked:
# Plotting the onset locked  separately for each task relevances:
for i, tsk in enumerate(tasks):
    for soa in soas:
        if i == 1:
            label_rt = "RT " + str(int(soa * 1000)) + " ms"
            label_irt = None
        elif i == 2:
            label_rt = None
            label_irt = "iT " + str(int(soa * 1000)) + " ms"
        else:
            label_rt = None
            label_irt = None
        # Objective RT:
        data_rt = data_df[(data_df["task_relevance"] == tsk)
                          & (data_df["SOA_lock"] == 'onset')
                          & (data_df["SOA"] == soa)]["RT_aud"].to_numpy()
        # Remove the nans:
        data_rt = data_rt[~np.isnan(data_rt)]
        # Compute the 1 and 99 percentile of the distribution to clip the data
        xlims.append(np.percentile(data_rt, [2.5, 97.5]))
        ax[0, i].ecdf(data_rt, label=label_rt, color=ev.colors["soa_onset_locked"][str(soa)])

        # Introspective RT:
        data_irt = data_df[(data_df["task_relevance"] == tsk)
                           & (data_df["SOA_lock"] == 'onset')
                           & (data_df["SOA"] == soa)]["iRT_aud"].to_numpy()
        # Remove the nans:
        data_irt = data_irt[~np.isnan(data_irt)]
        xlims.append(np.percentile(data_irt, [2.5, 97.5]))
        # Compute the 1 and 99 percentile of the distribution to clip the data
        ax[0, i].ecdf(data_irt, label=label_irt, color=ev.colors["soa_onset_locked_iRT"][str(soa)])

# ===============================================
# Offset locked:
# For the offset locked, plot separately per task demands and duration:
for dur_i, dur in enumerate(durations):
    for tsk_i, tsk in enumerate(tasks):
        for soa in soas:
            if dur_i == 2 and tsk_i == 2:
                label_rt = None
                label_irt = "iT " + str(int(soa * 1000)) + " ms"

            elif dur_i == 2 and tsk_i == 1:
                label_rt = "RT " + str(int(soa * 1000)) + " ms"
                label_irt = None
            else:
                label_rt = None
                label_irt = None
            # Objective RT:
            data_rt = data_df[(data_df["task_relevance"] == tsk)
                              & (data_df["SOA_lock"] == 'offset')
                              & (data_df["SOA"] == soa)
                              & (data_df["duration"] == dur)]["RT_aud"].to_numpy()
            # Remove the nans:
            data_rt = data_rt[~np.isnan(data_rt)]
            xlims.append(np.percentile(data_rt, [2.5, 97.5]))
            ax[dur_i + 1, tsk_i].ecdf(data_rt, label=label_rt,
                                      color=ev.colors["soa_offset_locked"][str(soa)])
            # Introspective RT:
            data_irt = data_df[(data_df["task_relevance"] == tsk)
                               & (data_df["SOA_lock"] == 'offset')
                               & (data_df["SOA"] == soa)
                               & (data_df["duration"] == dur)]["iRT_aud"].to_numpy()
            # Remove the nans:
            data_irt = data_irt[~np.isnan(data_irt)]
            xlims.append(np.percentile(data_irt, [2.5, 97.5]))
            ax[dur_i + 1, tsk_i].ecdf(data_irt, label=label_irt,
                                      color=ev.colors["soa_offset_locked_iRT"][str(soa)])
# Add axis decorations:
fig.supxlabel("RT (s)")
fig.supylabel("CDF")
ax[0, 0].set_xlim([0, np.max(xlims)])
fig.savefig(Path(save_root, "Experiment2-figure2b.svg"), transparent=True, dpi=dpi)
fig.savefig(Path(save_root, "Experiment2-figure2b.png"), transparent=True, dpi=dpi)
plt.close(fig)
