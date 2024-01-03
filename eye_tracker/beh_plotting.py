import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from general_helper_function import cousineau_morey_correction
from plotter_functions import plot_within_subject_boxplot

SMALL_SIZE = 36
MEDIUM_SIZE = 38
BIGGER_SIZE = 40

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


# ======================================================================================================================
# Set parameters:
root = r"P:\2023-0357-ReconTime\03_data\raw_data"
ses = "1"
subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112",
                 "SX113", "SX114", "SX115", "SX116", "SX117", "SX118", "SX119", "SX120", "SX121", "SX123"]
file_name = "sub-{}_ses-1_run-all_task-prp_events.csv"

# ======================================================================================================================
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
subjects_data = pd.concat(subjects_data).reset_index(drop=True)
# Drop the targets:
subjects_data = subjects_data[subjects_data["task_relevance"].isin(["non-target", "irrelevant"])].reset_index(drop=True)

# ========================================================================================================
# Plot the results separately for the onset and offset locked trials:

# 1. Onset locked:
onset_locked_data = subjects_data[subjects_data["SOA_lock"] == 'onset']
onset_locked_ti = onset_locked_data[onset_locked_data["task_relevance"] == 'irrelevant']
onset_locked_tr = onset_locked_data[onset_locked_data["task_relevance"] == 'non-target']

# 2. Compute the Cousineau Morey correction:
onset_locked_ti = cousineau_morey_correction(onset_locked_ti,
                                             'sub_id', 'SOA', 'RT_aud')
onset_locked_tr = cousineau_morey_correction(onset_locked_tr,
                                             'sub_id', 'SOA', 'RT_aud')

# 3. Average the data within subject and SOA:
avg_onset_locked_ti = onset_locked_ti.groupby(['sub_id', 'SOA'])['RT_aud'].mean().reset_index()
avg_onset_locked_tr = onset_locked_tr.groupby(['sub_id', 'SOA'])['RT_aud'].mean().reset_index()

# 4. Convert to 2D arrays:
data_ti_2d = np.array([avg_onset_locked_ti[avg_onset_locked_ti['SOA'] == soa]['RT_aud'].to_numpy()
                       for soa in avg_onset_locked_ti["SOA"].unique()]).T
data_tr_2d = np.array([avg_onset_locked_tr[avg_onset_locked_tr['SOA'] == soa]['RT_aud'].to_numpy()
                       for soa in avg_onset_locked_tr["SOA"].unique()]).T

# Compute the mean RT in each condition to color code the boxes:
data_ti_mean = np.median(data_ti_2d, axis=0)
data_tr_mean = np.median(data_tr_2d, axis=0)
min_val = np.min(np.concatenate([data_ti_mean, data_tr_mean]))
max_val = np.max(np.concatenate([data_ti_mean, data_tr_mean]))
# Normalize them by the max and min across:
data_ti_mean = (data_ti_mean - min_val) / (max_val - min_val)
data_tr_mean = (data_tr_mean - min_val) / (max_val - min_val)
color_vals = np.array([data_tr_mean, data_ti_mean])
cmap = matplotlib.colormaps.get_cmap('Reds')

# Boxplot:
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=[14, 14])
_, bplot1, _ = plot_within_subject_boxplot(onset_locked_ti, 'sub_id', 'SOA', 'RT_aud',
                                           positions='SOA', ax=ax[0], cousineau_correction=True, title="Task relevant",
                                           xlabel="SOA (sec)", ylabel="Cousineau Morey Corrected RT (sec)")
_, bplot2, _ = plot_within_subject_boxplot(onset_locked_tr, 'sub_id', 'SOA', 'RT_aud',
                                           positions='SOA', ax=ax[1], cousineau_correction=True, title="Task relevant",
                                           xlabel="SOA (sec)", ylabel="Cousineau Morey Corrected RT (sec)")
plt.xlim([-0.1, 0.55])
ylims = ax[1].get_ylim()
# fill with colors
for i, bplot in enumerate([bplot1, bplot2]):
    for ii, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(cmap(color_vals[i, ii]))
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(r"C:\Users\alexander.lepauvre\Downloads\Figure1.png", dpi=300)

# ===========================================================================================================
# 2. Offset locked:
# In the case of the offset locked, there are no effects of task relevance but there is an effect of duration:
# Extract the offset locked data:
offset_locked_data = subjects_data[subjects_data["SOA_lock"] == 'offset']
offset_locked_data.loc[:, 'SOA'] = offset_locked_data['SOA'] + offset_locked_data['duration']

# Compute the Cousineau Morey correction:
offset_locked_data = cousineau_morey_correction(offset_locked_data, 'sub_id', 'SOA', 'RT_aud')

avg_offset_locked = offset_locked_data.groupby(['sub_id', 'SOA', 'duration'])['RT_aud'].mean().reset_index()

# Sort the data:
avg_offset_locked = avg_offset_locked.sort_values('SOA')

# Boxplot:
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=[21, 14])
bplots = []
data_offset_2d = []
# For the within subject lines, split by durations:
for ind, dur in enumerate(avg_offset_locked['duration'].unique()):
    duration_data = avg_offset_locked[avg_offset_locked["duration"] == dur]
    # Convert to 2D
    data_2d = np.array([duration_data[duration_data['SOA'] == soa]['RT_aud'].to_numpy()
                        for soa in duration_data["SOA"].unique()]).T
    data_offset_2d.append(data_2d)
    bplot = ax[ind].boxplot(data_2d, patch_artist=True, notch=True,
                            positions=duration_data['SOA'].unique(), widths=0.1)
    ax[ind].plot(duration_data['SOA'].unique(), data_2d.T, linewidth=0.4,
                 color=[0.5, 0.5, 0.5], alpha=0.5)
    ax[ind].tick_params(axis='x', labelrotation=45)
    ax[ind].set_title("{} sec ".format(dur))
    bplots.append(bplot)
    ax[ind].set_xlim([min(duration_data['SOA'].unique()) - 0.05, max(duration_data['SOA'].unique()) + 0.05])

# Set the ylims to be equated with the previous plot:
plt.ylim(ylims)
# plt.suptitle("Offset locked Reaction times")
ax[0].set_xlabel("SOA (sec)")
ax[0].set_ylabel("Cousineau Morey Corrected RT (sec)")
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[2].spines['right'].set_visible(False)

# Compute the median values:
data_offset_median = np.median(np.array(data_offset_2d), axis=1)
data_offset_median = (data_offset_median - min_val) / (max_val - min_val)
color_vals = data_offset_median
cmap = matplotlib.colormaps.get_cmap('Reds')
for i, bplot in enumerate(bplots):
    for ii, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(cmap(color_vals[i, ii]))
plt.tight_layout()
plt.savefig(r"C:\Users\alexander.lepauvre\Downloads\Figure2.png", dpi=300)

# ===========================================================================================================
# 2.1. Offset locked for the task relevant separately:
# There is an effect of task relevance, therefore, plotting them separately:
offset_locked_data = subjects_data[(subjects_data["SOA_lock"] == 'offset') &
                                   (subjects_data["task_relevance"] == 'irrelevant')]
# Compute the Cousineau Morey correction:
offset_locked_data = cousineau_morey_correction(offset_locked_data,
                                                'sub_id', 'SOA', 'RT_aud')
avg_offset_locked = offset_locked_data.groupby(['sub_id', 'SOA', 'duration'])['RT_aud'].mean().reset_index()
# To keep things readable, adding the duration to the SOA:
avg_offset_locked['SOA'] = avg_offset_locked['SOA'] + avg_offset_locked['duration']
# Sort the data:
avg_offset_locked = avg_offset_locked.sort_values('SOA')

# Boxplot:
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=[21, 14])
bplots = []
data_offset_2d = []
# For the within subject lines, split by durations:
for ind, dur in enumerate(avg_offset_locked['duration'].unique()):
    duration_data = avg_offset_locked[avg_offset_locked["duration"] == dur]
    # Convert to 2D
    data_2d = np.array([duration_data[duration_data['SOA'] == soa]['RT_aud'].to_numpy()
                        for soa in duration_data["SOA"].unique()]).T
    data_offset_2d.append(data_2d)
    bplot = ax[ind].boxplot(data_2d, patch_artist=True, notch=True,
                            positions=duration_data['SOA'].unique(), widths=0.1)
    ax[ind].plot(duration_data['SOA'].unique(), data_2d.T, linewidth=0.4,
                 color=[0.5, 0.5, 0.5], alpha=0.5)
    ax[ind].tick_params(axis='x', labelrotation=45)
    ax[ind].set_title("{} sec ".format(dur))
    bplots.append(bplot)
    ax[ind].set_xlim([min(duration_data['SOA'].unique()) - 0.05, max(duration_data['SOA'].unique()) + 0.05])
# Set the ylims to be equated with the previous plot:
plt.ylim(ylims)
# plt.suptitle("Offset locked Reaction times")
ax[0].set_xlabel("SOA (sec)")
ax[0].set_ylabel("Reaction Time (sec)")
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[2].spines['right'].set_visible(False)

# Compute the median values:
data_offset_median = np.median(np.array(data_offset_2d), axis=1)
data_offset_median = (data_offset_median - min_val) / (max_val - min_val)
color_vals = data_offset_median
cmap = matplotlib.colormaps.get_cmap('Reds')
for i, bplot in enumerate(bplots):

    for ii, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(cmap(color_vals[i, ii]))
plt.tight_layout()
plt.savefig(r"C:\Users\alexander.lepauvre\Downloads\Figure2_tr.png", dpi=300)

# ===========================================================================================================
# 2.2. Offset locked:
# In the case of the offset locked, there are no effects of task relevance but there is an effect of duration:
# Extract the offset locked data:
offset_locked_data = subjects_data[(subjects_data["SOA_lock"] == 'offset') &
                                   (subjects_data["task_relevance"] == 'non-target')]
# Compute the Cousineau Morey correction:
offset_locked_data = cousineau_morey_correction(offset_locked_data,
                                                'sub_id', 'SOA', 'RT_aud')
avg_offset_locked = offset_locked_data.groupby(['sub_id', 'SOA', 'duration'])['RT_aud'].mean().reset_index()
# To keep things readable, adding the duration to the SOA:
avg_offset_locked['SOA'] = avg_offset_locked['SOA'] + avg_offset_locked['duration']
# Sort the data:
avg_offset_locked = avg_offset_locked.sort_values('SOA')

# Boxplot:
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=[21, 14])
bplots = []
data_offset_2d = []
# For the within subject lines, split by durations:
for ind, dur in enumerate(avg_offset_locked['duration'].unique()):
    duration_data = avg_offset_locked[avg_offset_locked["duration"] == dur]
    # Convert to 2D
    data_2d = np.array([duration_data[duration_data['SOA'] == soa]['RT_aud'].to_numpy()
                        for soa in duration_data["SOA"].unique()]).T
    data_offset_2d.append(data_2d)
    bplot = ax[ind].boxplot(data_2d, patch_artist=True, notch=True,
                            positions=duration_data['SOA'].unique(), widths=0.1)
    ax[ind].plot(duration_data['SOA'].unique(), data_2d.T, linewidth=0.4,
                 color=[0.5, 0.5, 0.5], alpha=0.5)
    ax[ind].tick_params(axis='x', labelrotation=45)
    ax[ind].set_title("{} sec ".format(dur))
    bplots.append(bplot)
    ax[ind].set_xlim([min(duration_data['SOA'].unique()) - 0.05, max(duration_data['SOA'].unique()) + 0.05])
# Set the ylims to be equated with the previous plot:
plt.ylim(ylims)
# plt.suptitle("Offset locked Reaction times")
ax[0].set_xlabel("SOA (sec)")
ax[0].set_ylabel("Reaction Time (sec)")
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[2].spines['right'].set_visible(False)

# Compute the median values:
data_offset_median = np.median(np.array(data_offset_2d), axis=1)
data_offset_median = (data_offset_median - min_val) / (max_val - min_val)
color_vals = data_offset_median
cmap = matplotlib.colormaps.get_cmap('Reds')
for i, bplot in enumerate(bplots):
    for ii, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(cmap(color_vals[i, ii]))
plt.tight_layout()
plt.savefig(r"C:\Users\alexander.lepauvre\Downloads\Figure2_ti.png", dpi=300)
