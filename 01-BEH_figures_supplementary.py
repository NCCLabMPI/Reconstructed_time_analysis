import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, zscore
from helper_function.helper_plotter import plot_within_subject_boxplot
import environment_variables as ev
from helper_function.helper_general import load_beh_data, compute_dprime

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
sns.set_style("white")

# ======================================================================================================================
# Experiment 1:
ses = "1"
task = "prp"
file_name = "sub-{}_ses-{}_run-all_task-{}_events.csv"
save_root = Path(ev.bids_root, "derivatives", "figures", "supplementary")
if not os.path.isdir(save_root):
    os.makedirs(save_root)
# Load the data without trial exclusion:
data_df_raw = load_beh_data(ev.bids_root, ev.subjects_lists_beh[task], file_name, session=ses, task=task,
                            do_trial_exclusion=False)

# Load the data:
data_df = load_beh_data(ev.bids_root, ev.subjects_lists_beh[task], file_name, session=ses, task=task,
                        do_trial_exclusion=True)

# ======================================================
# A: Accuracy:
fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=[8.3 / 4, 8.3 / 4])
# T1 target detection accuracy per category:
t1_dprime = []
categories = ["face", "object", "letter", "false-font"]
for subject in data_df_raw["sub_id"].unique():
    for category in categories:
        # Extract the data:
        df = data_df_raw[(data_df_raw["sub_id"] == subject) &
                         (data_df_raw["category"] == category.replace("-", "_"))]
        hit = df["trial_response_vis"].value_counts().get("hit", 0)
        miss = df["trial_response_vis"].value_counts().get("miss", 0)
        fa = df["trial_response_vis"].value_counts().get("fa", 0)
        cr = df["trial_response_vis"].value_counts().get("cr", 0)
        dprime, _ = compute_dprime(hit, miss, fa, cr)
        t1_dprime.append([subject, category, dprime])
t1_dprime = pd.DataFrame(t1_dprime, columns=["sub_id", "category", "T1_dprime"])

# T2 pitch detection accuracy:
t2_dprime = []
t2_beta = []
pitches = [1000, 1100]
for subject in data_df_raw["sub_id"].unique():
    # Take low pitch as a reference:
    df = data_df_raw[(data_df_raw["sub_id"] == subject)]
    hit = df.loc[(df["pitch"] == pitches[0]) & (df["trial_accuracy_aud"] == 1)].shape[0]
    miss = df.loc[(df["pitch"] == pitches[0]) & (df["trial_accuracy_aud"] != 1)].shape[0]
    fa = df.loc[(df["pitch"] != pitches[0]) & (df["trial_accuracy_aud"] != 1)].shape[0]
    cr = df.loc[(df["pitch"] != pitches[0]) & (df["trial_accuracy_aud"] == 1)].shape[0]
    dprime, beta = compute_dprime(hit, miss, fa, cr)
    t2_dprime.append([subject, dprime])
    t2_beta.append([subject, beta])
t2_dprime = pd.DataFrame(t2_dprime, columns=["sub_id", "dprime"])
t2_beta = pd.DataFrame(t2_beta, columns=["sub_id", "beta"])

# Plot T1 dprimes:
_, _, _ = plot_within_subject_boxplot(t1_dprime,
                                      'sub_id', 'category', 'T1_dprime',
                                      positions=[1, 2, 3, 4], ax=ax1[0], cousineau_correction=False,
                                      title="",
                                      xlabel="Category", ylabel="d'",
                                      xlim=None, width=0.5,
                                      face_colors=[val for val in ev.colors["category"].values()],
                                      xlabel_fontsize=12)
# Plot T2 dprimes:
bplot = ax1[1].boxplot(t2_dprime["dprime"].to_numpy(), patch_artist=True, notch=False,
                       positions=[1], widths=0.3, medianprops=dict(color="black", linewidth=1.5))
for i, patch in enumerate(bplot['boxes']):
    patch.set_facecolor(ev.colors["pitch"]["1000"])
# Plot the beta:
ax2 = ax1[1].twinx()
ax2.sharey(ax1[1])
bplot = ax1[1].boxplot(t2_beta["beta"].to_numpy(), patch_artist=True, notch=False,
                       positions=[2], widths=0.3, medianprops=dict(color="black", linewidth=1.5))
for i, patch in enumerate(bplot['boxes']):
    patch.set_facecolor(ev.colors["pitch"]["1100"])
ax1[1].set_xlabel("Pitch")
ax1[0].set_xticklabels(categories)
ax1[1].set_xticklabels(["d'", r"$\beta$ "])
ax2.set_ylabel(r"$\beta$ ")
ax1[0].spines['right'].set_visible(False)
ax1[1].spines['left'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(length=0)
ax1[1].yaxis.set_visible(False)
plt.subplots_adjust(wspace=0)
fig1.suptitle("Task performances")
fig1.savefig(Path(save_root, "Experiment1-t1t2_accuracy.svg"), transparent=True, dpi=dpi)
fig1.savefig(Path(save_root, "Experiment1-t1t2_accuracy.png"), transparent=True, dpi=dpi)
plt.close(fig1)

# ======================================================
# B: Reaction time:
fig2, ax2 = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=[8.3 /4, 8.3 / 4])
_, _, _ = plot_within_subject_boxplot(data_df,
                                      'sub_id', 'category', 'RT_vis',
                                      positions=[1, 2, 3, 4], ax=ax2[0], cousineau_correction=False,
                                      title="",
                                      xlabel="Category", ylabel="Reaction time (sec.)",
                                      xlim=None, width=0.5,
                                      face_colors=[val for val in ev.colors["category"].values()])
_, _, _ = plot_within_subject_boxplot(data_df,
                                      'sub_id', 'pitch', 'RT_aud',
                                      positions=[1, 2], ax=ax2[1], cousineau_correction=False,
                                      title="",
                                      xlabel="Pitch", ylabel="",
                                      xlim=None, width=0.5,
                                      face_colors=[val for val in ev.colors["pitch"].values()])
ax2[0].spines['right'].set_visible(False)
ax2[1].spines['left'].set_visible(False)
ax2[1].set_xticklabels(pitches)
ax2[0].set_xticklabels(categories)
# Remove the white space between both plots:
plt.subplots_adjust(wspace=0)
fig2.suptitle("Reaction time")
fig2.savefig(Path(save_root, "Experiment1-RT.svg"), transparent=True, dpi=dpi)
fig2.savefig(Path(save_root, "Experiment1-RT.png"), transparent=True, dpi=dpi)
plt.close(fig2)

# ======================================================================================================================
# Experiment 2:
sessions = ["2", "3"]
task = "introspection"
file_name = "sub-{}_ses-{}_run-all_task-{}_events.csv"
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
# ======================================================
# A: Accuracy:
fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=[8.3 /4, 8.3 / 4])
# T1 target detection accuracy per category:
t1_dprime = []
categories = ["face", "object", "letter", "false-font"]
for subject in data_df_raw["sub_id"].unique():
    for category in categories:
        # Extract the data:
        df = data_df_raw[(data_df_raw["sub_id"] == subject) &
                         (data_df_raw["category"] == category.replace("-", "_"))]
        hit = df["trial_response_vis"].value_counts().get("hit", 0)
        miss = df["trial_response_vis"].value_counts().get("miss", 0)
        fa = df["trial_response_vis"].value_counts().get("fa", 0)
        cr = df["trial_response_vis"].value_counts().get("cr", 0)
        dprime, _ = compute_dprime(hit, miss, fa, cr)
        t1_dprime.append([subject, category, dprime])
t1_dprime = pd.DataFrame(t1_dprime, columns=["sub_id", "category", "T1_dprime"])

# T2 pitch detection accuracy:
t2_dprime = []
t2_beta = []
pitches = [1000, 1100]
for subject in data_df_raw["sub_id"].unique():
    # Take low pitch as a reference:
    df = data_df_raw[(data_df_raw["sub_id"] == subject)]
    hit = df.loc[(df["pitch"] == pitches[0]) & (df["trial_accuracy_aud"] == 1)].shape[0]
    miss = df.loc[(df["pitch"] == pitches[0]) & (df["trial_accuracy_aud"] != 1)].shape[0]
    fa = df.loc[(df["pitch"] != pitches[0]) & (df["trial_accuracy_aud"] != 1)].shape[0]
    cr = df.loc[(df["pitch"] != pitches[0]) & (df["trial_accuracy_aud"] == 1)].shape[0]
    dprime, beta = compute_dprime(hit, miss, fa, cr)
    t2_dprime.append([subject, dprime])
    t2_beta.append([subject, beta])
t2_dprime = pd.DataFrame(t2_dprime, columns=["sub_id", "dprime"])
t2_beta = pd.DataFrame(t2_beta, columns=["sub_id", "beta"])

# Plot T1 dprimes:
_, _, _ = plot_within_subject_boxplot(t1_dprime,
                                      'sub_id', 'category', 'T1_dprime',
                                      positions=[1, 2, 3, 4], ax=ax1[0], cousineau_correction=False,
                                      title="",
                                      xlabel="Category", ylabel="d'",
                                      xlim=None, width=0.5,
                                      face_colors=[val for val in ev.colors["category"].values()],
                                      xlabel_fontsize=12)
# Plot T2 dprimes:
bplot = ax1[1].boxplot(t2_dprime["dprime"].to_numpy(), patch_artist=True, notch=False,
                       positions=[1], widths=0.3, medianprops=dict(color="black", linewidth=1.5))
for i, patch in enumerate(bplot['boxes']):
    patch.set_facecolor(ev.colors["pitch"]["1000"])
# Plot the beta:
ax2 = ax1[1].twinx()
ax2.sharey(ax1[1])
bplot = ax1[1].boxplot(t2_beta["beta"].to_numpy(), patch_artist=True, notch=False,
                       positions=[2], widths=0.3, medianprops=dict(color="black", linewidth=1.5))
for i, patch in enumerate(bplot['boxes']):
    patch.set_facecolor(ev.colors["pitch"]["1100"])
ax1[1].set_xlabel("Pitch")
ax1[0].set_xticklabels(categories)
ax1[1].set_xticklabels(["d'", r"$\beta$ "])
ax2.set_ylabel(r"$\beta$ ")
ax1[0].spines['right'].set_visible(False)
ax1[1].spines['left'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(length=0)
ax1[1].yaxis.set_visible(False)
plt.subplots_adjust(wspace=0)
fig1.suptitle("Task performances")
fig1.savefig(Path(save_root, "Experiment2-t1t2_accuracy.svg"), transparent=True, dpi=dpi)
fig1.savefig(Path(save_root, "Experiment2-t1t2_accuracy.png"), transparent=True, dpi=dpi)
plt.close(fig1)

# ======================================================
# B: Reaction time:
fig2, ax2 = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=[8.3/4, 8.3 / 4])
_, _, _ = plot_within_subject_boxplot(data_df,
                                      'sub_id', 'category', 'RT_vis',
                                      positions=[1, 2, 3, 4], ax=ax2[0], cousineau_correction=False,
                                      title="",
                                      xlabel="Category", ylabel="Reaction time (sec.)",
                                      xlim=None, width=0.5,
                                      face_colors=[val for val in ev.colors["category"].values()])
_, _, _ = plot_within_subject_boxplot(data_df,
                                      'sub_id', 'pitch', 'RT_aud',
                                      positions=[1, 2], ax=ax2[1], cousineau_correction=False,
                                      title="",
                                      xlabel="Pitch", ylabel="",
                                      xlim=None, width=0.5,
                                      face_colors=[val for val in ev.colors["pitch"].values()])
ax2[0].spines['right'].set_visible(False)
ax2[1].spines['left'].set_visible(False)
ax2[1].set_xticklabels(pitches)
ax2[0].set_xticklabels(categories)
# Remove the white space between both plots:
plt.subplots_adjust(wspace=0)
fig2.suptitle("Reaction time")
fig2.savefig(Path(save_root, "Experiment2-RT.svg"), transparent=True, dpi=dpi)
fig2.savefig(Path(save_root, "Experiment2-RT.png"), transparent=True, dpi=dpi)
plt.close(fig2)

# =======================================================
# RT2 as a function of iT1:
# Convert iT1 to sec:
data_df.loc[data_df["iRT_vis"] == 0, "iRT_vis"] = 1
# data_df["iRT_vis"] = data_df["iRT_vis"] * 0.001
data_df = data_df.loc[data_df["task_relevance"] != "target"]
# Compute the zscore iRT vis separately for each subject:
data_df["ziRT_vis"] = np.nan
# Loop through each subject
for sub in list(data_df["sub_id"].unique()):
    data_df.loc[data_df["sub_id"] == sub, "ziRT_vis"] = (
        zscore(data_df.loc[data_df["sub_id"] == sub, "iRT_vis"].to_numpy()))

fig3, ax3 = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=[8.3, 11.7])
# Loop through onset and offset:
for i, lock in enumerate(["onset", "offset"]):
    for ii, dur in enumerate(np.sort(data_df["duration"].unique())):
        # Loop through each SOA:
        for iii, soa in enumerate(np.sort(data_df["SOA"].unique())):
            # Extract the relevant data:
            xy = data_df.loc[(data_df["duration"] == dur) & (data_df["SOA_lock"] == lock) &
                             (data_df["SOA"] == soa)]
            # Compute Pearson correlation coefficient:
            results = pearsonr(xy["ziRT_vis"].to_numpy(), xy["RT_aud"].to_numpy() * 1000)
            # Plot the correlation:
            sns.regplot(x="iRT_vis", y="RT_aud", data=xy, scatter=False,
                        label="{}: r={:.2f}, p={:.3f}".format(soa, results.correlation, results.pvalue),
                        ax=ax3[ii, i], color=ev.colors["soa_{}_locked".format(lock)][str(soa)])
        ax3[ii, i].legend(fontsize=9)
        if ii < 2:
            ax3[ii, i].set_xticks([])
            ax3[ii, i].set_xlabel("")
            ax3[ii, i].set_ylabel("")
        else:
            ax3[ii, i].set_xlabel("z-score iT1 (s)")
            ax3[ii, i].set_ylabel("RT2 (s)")
        if i > 0:
            ax3[ii, i].set_ylabel("")
ax3[0, 0].set_title("Onset")
ax3[0, 1].set_title("Offset")
plt.subplots_adjust(hspace=0)
fig3.savefig(Path(save_root, "Experiment2-iT1_RT2_correlation.svg"), transparent=True, dpi=dpi)
fig3.savefig(Path(save_root, "Experiment2-iT1_RT2_correlation.png"), transparent=True, dpi=dpi)
plt.close(fig3)


