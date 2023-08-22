import mne
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from eye_tracker.general_helper_function import baseline_scaling
import seaborn as sns
import pandas as pd
import os
import statsmodels.formula.api as smf


prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']
# First, load the parameters:
bids_root = r"C:\Users\alexander.lepauvre\Documents\PhD\Reconstructed_Time\bids"
visit = "1"
task = "prp"
session = "1"
data_type = "eyetrack"
epoch_name = "visual_onset"
crop = [-0.3, 3.5]
audio_lock_window = [0., 2.0]
picks = ["LPupil", "RPupil"]
task_relevance = ["non-target", "irrelevant"]
durations = ["0.5", "1", "1.5"]
soas = ["0", "0.116", "0.232", "0.466"]
locks = ["onset", "offset"]
subjects = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113", "SX114",
            "SX115", "SX116", "SX118", "SX119", "SX120", "SX121"]
models = {
    "full_model": "peak_latency ~ task_relevance * lock * soa",
    "onset_model": "peak_latency ~ task_relevance * soa",
    "offset_model": "peak_latency ~ task_relevance * soa",
}

# Preallocate the subjects betas:
subjects_betas = []
peak_latencies = pd.DataFrame()
# Loop through each subject:
for sub in subjects:
    print(sub)
    # ===========================================
    # Data loading:
    # Load the epochs:
    root = Path(bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                data_type)
    file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(sub, session, task, data_type,
                                                                  epoch_name)
    epochs = mne.read_epochs(Path(root, file_name))
    # Extract the relevant conditions:
    epochs = epochs[["non-target", "irrelevant"]]
    # Crop if needed:
    epochs.crop(crop[0], crop[1])
    # epochs.decimate(10)
    # Extract the relevant channels:
    epochs.pick(picks)

    # ===========================================
    # Preprocessing
    # Baseline correction:
    baseline_scaling(epochs, correction_method="ratio", baseline=(None, -0.05))
    #  Extract the metadata:
    meta_data = epochs.metadata

    # Loop through each relevant condition:
    conditions = ["task_relevance", "duration", "lock", "soa"]
    for task_rel in task_relevance:
        for duration in durations:
            for lock in locks:
                for soa in soas:
                    # Extract the data:
                    data = epochs.copy()["/".join([task_rel, duration, lock, soa])].get_data()
                    # Average across trials:
                    data = np.mean(data, axis=0)
                    # Average across channels:
                    data = np.mean(data, axis=0)
                    # Extract the peak latency:
                    peak_latency = epochs.times[np.argmax(data)]
                    # The peak latency needs to be adjusted relative to the audio lock and SOA:
                    if lock == "onset":
                        peak_latency = peak_latency - float(soa)
                    else:
                        peak_latency = peak_latency - (float(soa) + float(duration))
                    # Add to data frame using pd.concat:
                    peak_latencies = pd.concat([peak_latencies, pd.DataFrame({"subject": sub,
                                                                              "task_relevance": task_rel,
                                                                              "duration": duration,
                                                                              "lock": lock,
                                                                              "soa": soa,
                                                                              "peak_latency": peak_latency},
                                                                             index=[0])])
peak_latencies = peak_latencies.reset_index(drop=True)
# Create the save directory:
save_dir = Path(bids_root, "derivatives", "pupil_size", "group_level", "data")
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
# Save the peak latencies:
peak_latencies.to_csv(Path(save_dir, "peak_latencies.csv"))

# Plot the peak latency:
for lock in locks:
    fig, ax = plt.subplots(nrows=len(task_relevance), ncols=len(durations), sharex=True, sharey=True)
    for task_i, task_rel in enumerate(task_relevance):
        for dur_i, dur in enumerate(durations):
            # Extract the data of this particular condition:
            df = peak_latencies.loc[(peak_latencies["task_relevance"] == task_rel)
                                    & (peak_latencies["duration"] == dur)
                                    & (peak_latencies["lock"] == lock), ["subject", "soa", "peak_latency"]]
            # Create box plot:
            sns.boxplot(x='soa', y='peak_latency', data=df, ax=ax[task_i, dur_i])
            # Add individual points:
            sns.stripplot(x='soa', y='peak_latency', data=df, jitter=True, color='black', alpha=0.5,
                          ax=ax[task_i, dur_i])
            # Add lines for each subject:
            for subject in df['subject'].unique():
                # Extract the data of this subject:
                sub_data = df[df['subject'] == subject]
                ax[task_i, dur_i].plot(sub_data['soa'], sub_data['peak_latency'], color='grey',
                                       linewidth=0.5, alpha=0.5,)
            ax[task_i, dur_i].set_title("{}, {}sec".format(task_rel, dur, lock))
            ax[task_i, dur_i].set_ylabel("Peak latency (s)")
            ax[task_i, dur_i].set_xlabel("SOA (s)")
            ax[task_i, dur_i].set_xticklabels(soas)
    plt.suptitle("{} locked peak latency".format(lock))
    plt.tight_layout()
    # Create the save directory:
    save_dir = Path(bids_root, "derivatives", "pupil_size", "group_level", "figures")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Save the figure:
    plt.savefig(Path(save_dir, "{}_locked_peak_latency.png".format(lock)))
    plt.close(fig)

# Fit the linear mixed models:
for model in models:
    if model == "full_model":
        data = peak_latencies.copy()
    elif model == "onset_model":
        data = peak_latencies.loc[peak_latencies["lock"] == "onset"].copy()
    elif model == "offset_model":
        data = peak_latencies.loc[peak_latencies["lock"] == "offset"].copy()
    # Fit the model:
    model = smf.ols(formula=models[model], data=data).fit()
    # Print the summary:
    print(model.summary())
    # Save the model:
    save_dir = Path(bids_root, "derivatives", "pupil_size", "group_level", "results")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.save(Path(save_dir, "{}.pickle".format(model)))
