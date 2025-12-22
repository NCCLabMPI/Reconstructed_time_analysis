import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from helper_function.helper_plotter import soa_boxplot
import environment_variables as ev
from helper_function.helper_general import load_beh_data
from matplotlib.gridspec import GridSpec
import pylustrator

pylustrator.start()

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

colors_onset = [ev.colors["soa_onset_locked"][str(soa)] for soa in np.sort(data_df["SOA"].unique())]
colors_offset = [ev.colors["soa_offset_locked"][str(soa)] for soa in np.sort(data_df["SOA"].unique())]


# ==================================
# Figure 3:

# Create figure
fig = plt.figure(figsize=(8.268, 11.693))

# Outer grid: 2 rows (top / bottom), 1 column
# Height ratios control how much vertical space each section takes
outer_gs = GridSpec(
    nrows=2,
    ncols=1,
    height_ratios=[1, 1],  # top ~ half, bottom ~ half
    hspace=0.25
)

# Figure 3a:
onset_lock_rt = [ev.colors["soa_onset_locked"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]
offset_lock_rt = [ev.colors["soa_offset_locked"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]
onset_lock_it = [ev.colors["soa_onset_locked_iRT"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]
offset_lock_it = [ev.colors["soa_offset_locked_iRT"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]
onset_lock_visit = [ev.colors["soa_onset_locked_visiRT"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]
offset_lock_visit = [ev.colors["soa_offset_locked_visiRT"][str(soa)] for soa in list(np.sort(data_df["SOA"].unique()))]

# Target:
d = 1.5
# RT vis:
top_gs = outer_gs[0].subgridspec(
    nrows=1,
    ncols=12,
    wspace=0.3
)

top_axes = []
for i in range(12):
    ax = fig.add_subplot(top_gs[0, i])
    top_axes.append(ax)

soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
            "RT_vis", ax=top_axes[0:4], fig=fig, 
            colors_onset_locked=[[0.5, 0.5, 0.5]] * 4, colors_offset_locked=[[0.5, 0.5, 0.5]] * 4,
            fig_size=[8.3 / 2, 11.7 / 2], avg_line_color=[0.5, 0.5, 0.5], zorder=0,
            alpha=0.4, label="RT visual", jitter=-0.04, linestyle="solid")
# RT audio
soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
            "RT_aud", ax=top_axes[0:4], fig=fig, 
            colors_onset_locked=onset_lock_rt, colors_offset_locked=offset_lock_rt,
            fig_size=[8.3 / 2, 11.7 / 2], zorder=10000, label="RT audio", jitter=0.04, linestyle="solid")
# iRT audio
soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
            "iRT_aud", ax=top_axes[0:4], fig=fig, 
            colors_onset_locked=onset_lock_rt, colors_offset_locked=offset_lock_rt,
            fig_size=[8.3 / 2, 11.7 / 2], zorder=10000, label="iT audio", jitter=-0.04, linestyle="dashed",
            marker='D')
# iRT vis
soa_boxplot(data_df[data_df["task_relevance"] == 'target'],
            "iRT_vis", ax=top_axes[0:4], fig=fig, 
            colors_onset_locked=[[0.5, 0.5, 0.5]] * 4, colors_offset_locked=[[0.5, 0.5, 0.5]] * 4, alpha=0.4,
            fig_size=[8.3 / 2, 11.7 / 2], zorder=10000, label="iT visual", jitter=0.04, linestyle="dashed",
            marker='D')
for a in top_axes[0:4]:
    a.axhline(0.64, linestyle="--", color=[0.7, 0.7, 0.7], alpha=0.8)
# Task relevant:
soa_boxplot(data_df[data_df["task_relevance"] == 'non-target'],
                            "RT_aud", ax=top_axes[4:8], fig=fig,
                            colors_onset_locked=onset_lock_rt, colors_offset_locked=offset_lock_rt,
                            fig_size=[8.3 / 2, 11.7 / 2], label="RT audio", linestyle="solid")
soa_boxplot(data_df[data_df["task_relevance"] == 'non-target'],
            "iRT_aud", ax=top_axes[4:8], fig=fig,
            colors_onset_locked=onset_lock_rt, colors_offset_locked=offset_lock_rt,
            fig_size=[8.3 / 2, 11.7 / 2], label="iT audio", linestyle="dashed",
            marker='D')
soa_boxplot(data_df[data_df["task_relevance"] == 'non-target'],
            "iRT_vis", ax=top_axes[4:8], fig=fig,
            colors_onset_locked=[[0.5, 0.5, 0.5]] * 4, colors_offset_locked=[[0.5, 0.5, 0.5]] * 4,
            fig_size=[8.3 / 2, 11.7 / 2], alpha=0.4, label="iT visual", linestyle="dashed",
            marker='D')
# Task irrelevant:
soa_boxplot(data_df[data_df["task_relevance"] == 'irrelevant'],
                            "RT_aud", ax=top_axes[8:12], fig=fig,
                            colors_onset_locked=colors_onset, colors_offset_locked=colors_offset,
                            fig_size=[8.3 / 2, 11.7 / 2], label="RT audio", linestyle="solid")
soa_boxplot(data_df[data_df["task_relevance"] == 'irrelevant'],
            "iRT_aud", ax=top_axes[8:12], fig=fig,
            colors_onset_locked=onset_lock_rt, colors_offset_locked=offset_lock_rt,
            fig_size=[8.3 / 2, 11.7 / 2], label="iT audio", linestyle="dashed",
            marker='D')
soa_boxplot(data_df[data_df["task_relevance"] == 'irrelevant'],
            "iRT_vis", ax=top_axes[8:12], fig=fig,
            colors_onset_locked=[[0.5, 0.5, 0.5]] * 4, colors_offset_locked=[[0.5, 0.5, 0.5]] * 4,
            fig_size=[8.3 / 2, 11.7 / 2], label="iT visual", linestyle="dashed",
            marker='D', alpha=0.4)
# Set the y limit to be the same for both plots:
lims = [[a.get_ylim()[0], a.get_ylim()[1]] for a in top_axes[4:]]
max_lims = [np.min(np.array(lims)), np.max(np.array(lims))]
[a.set_ylim(max_lims) for a in top_axes[4:]]
lims = [[a.get_ylim()[0], a.get_ylim()[1]] for a in top_axes[0:4]]
max_lims = [np.min(np.array(lims)), np.max(np.array(lims))]
[a.set_ylim(max_lims) for a in top_axes[0:4]]
top_axes[8].set_yticklabels([])

# =======================================================================================
# Figure 2b:
figsize = [8.3, 8.3 * (2 / 3)]
tasks = ["target", "non-target", "irrelevant"]
soas = list(np.sort(list(data_df["SOA"].unique())))
durations = list(np.sort(list(data_df["duration"].unique())))
xlims = []

bottom_gs = outer_gs[1].subgridspec(
    nrows=4,
    ncols=3,
    wspace=0.35,
    hspace=0.4
)

bottom_axes = np.empty([4, 3], dtype=object)
for r in range(4):
    for c in range(3):
        ax = fig.add_subplot(bottom_gs[r, c])
        bottom_axes[r, c] = ax

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
        bottom_axes[0, i].ecdf(data_rt, label=label_rt, color=ev.colors["soa_onset_locked"][str(soa)])

        # Introspective RT:
        data_irt = data_df[(data_df["task_relevance"] == tsk)
                           & (data_df["SOA_lock"] == 'onset')
                           & (data_df["SOA"] == soa)]["iRT_aud"].to_numpy()
        # Remove the nans:
        data_irt = data_irt[~np.isnan(data_irt)]
        xlims.append(np.percentile(data_irt, [2.5, 97.5]))
        # Compute the 1 and 99 percentile of the distribution to clip the data
        bottom_axes[0, i].ecdf(data_irt, label=label_irt, color=ev.colors["soa_onset_locked"][str(soa)], linestyle="dotted")

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
            bottom_axes[dur_i + 1, tsk_i].ecdf(data_rt, label=label_rt,
                                      color=ev.colors["soa_offset_locked"][str(soa)])
            # Introspective RT:
            data_irt = data_df[(data_df["task_relevance"] == tsk)
                               & (data_df["SOA_lock"] == 'offset')
                               & (data_df["SOA"] == soa)
                               & (data_df["duration"] == dur)]["iRT_aud"].to_numpy()
            # Remove the nans:
            data_irt = data_irt[~np.isnan(data_irt)]
            xlims.append(np.percentile(data_irt, [2.5, 97.5]))
            bottom_axes[dur_i + 1, tsk_i].ecdf(data_irt, label=label_irt,
                                      color=ev.colors["soa_offset_locked"][str(soa)], linestyle="dotted")
# Add axis decorations:
fig.supxlabel("RT (s)")
fig.supylabel("CDF")
[ax.set_xlim([0, np.max(xlims)]) for ax in bottom_axes.flatten()]
for i in range(4):
    for ii in range(2):
        bottom_axes[i, ii+1].set_yticklabels([])
for i in range(3):
    for ii in range(3):
        bottom_axes[i, ii].set_xticklabels([])
plt.tight_layout()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).axes[0].set(position=[0.08466, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[1].set(position=[0.1505, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[2].set(position=[0.2164, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[3].set(position=[0.2832, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[4].set(position=[0.4097, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[5].set(position=[0.4755, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[6].set(position=[0.5414, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[7].set(position=[0.6072, 0.5542, 0.05065, 0.3422])
plt.figure(1).axes[8].set(position=[0.6731, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[9].set(position=[0.7389, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[10].set(position=[0.8048, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[11].set(position=[0.8717, 0.5535, 0.05065, 0.3422])
plt.figure(1).axes[12].set(position=[0.08682, 0.3729, 0.2481, 0.06584])
plt.figure(1).axes[13].set(position=[0.4108, 0.3729, 0.2481, 0.06584])
plt.figure(1).axes[14].set(position=[0.6753, 0.3729, 0.2481, 0.06584])
plt.figure(1).axes[15].set(position=[0.08782, 0.2802, 0.2481, 0.06584])
plt.figure(1).axes[16].set(position=[0.4119, 0.2802, 0.2481, 0.06584])
plt.figure(1).axes[17].set(position=[0.6763, 0.2802, 0.2481, 0.06584])
plt.figure(1).axes[18].set(position=[0.08682, 0.1876, 0.2481, 0.06584])
plt.figure(1).axes[19].set(position=[0.4108, 0.1876, 0.2481, 0.06584])
plt.figure(1).axes[20].set(position=[0.6753, 0.1876, 0.2481, 0.06584])
plt.figure(1).axes[21].set(position=[0.08892, 0.09503, 0.2481, 0.06584])
plt.figure(1).axes[22].set(position=[0.4129, 0.09503, 0.2481, 0.06584])
plt.figure(1).axes[23].set(position=[0.6753, 0.09503, 0.2481, 0.06584])
plt.figure(1).texts[0].set(position=(0.5341, 0.0504))
plt.figure(1).texts[1].set(visible=False)
plt.figure(1).text(0.1516, 0.4580, 'T1 Target', transform=plt.figure(1).transFigure, color='#000000ff')  # id=plt.figure(1).texts[2].new
plt.figure(1).text(0.4617, 0.4580, 'T1 Relevant', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[3].new
plt.figure(1).text(0.7272, 0.4580, 'T1 Irrelevant', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[4].new
plt.figure(1).text(0.9289, 0.3099, '0.5s', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[5].new
plt.figure(1).text(0.9289, 0.2173, '1.0s', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[6].new
plt.figure(1).text(0.9289, 0.1247, '1.5s', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[7].new
plt.figure(1).text(0.5032, 0.3532, 'Offset', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[8].new
plt.figure(1).text(0.5032, 0.4416, 'Onset', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[9].new
plt.figure(1).text(0.1537, 0.9042, 'T1 Target', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[10].new
plt.figure(1).text(0.4618, 0.9034, 'T1 Relevant', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[11].new
plt.figure(1).text(0.7273, 0.9019, 'T1 Irrelevant', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[12].new
plt.figure(1).text(0.0021, 0.4468, 'P(RT2<X)', transform=plt.figure(1).transFigure, fontsize=10.)  # id=plt.figure(1).texts[13].new
plt.figure(1).text(0.0212, 0.9042, 'RT(s)', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[14].new
plt.figure(1).text(0.4066, 0.4948, 'Time from T1 onset (s)', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[15].new
plt.figure(1).text(0.0021, 0.9236, 'A', transform=plt.figure(1).transFigure, fontsize=18., weight='bold')  # id=plt.figure(1).texts[16].new
plt.figure(1).text(0.0021, 0.4641, 'B', transform=plt.figure(1).transFigure, fontsize=18., weight='bold')  # id=plt.figure(1).texts[17].new
#% end: automatic generated code from pylustrator
fig.savefig(Path(save_root, "Experiment1-figureSXXX.svg"), transparent=True, dpi=dpi)
plt.close()