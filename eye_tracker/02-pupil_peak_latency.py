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

# Set the font size:
plt.rcParams.update({'font.size': 22})


def pupil_peak_latency(parameters_file, subjects):

    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)

    peak_latencies = pd.DataFrame()
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
        epochs = epochs[["non-target", "irrelevant"]]
        # Crop if needed:
        epochs.crop(param["crop"][0], param["crop"][1])
        # epochs.decimate(10)
        # Extract the relevant channels:
        epochs.pick(param["picks"])

        # ===========================================
        # Preprocessing
        # Baseline correction:
        baseline_scaling(epochs, correction_method=param["baseline"], baseline=param["baseline_window"])

        # Loop through each relevant condition:
        for task_rel in param["task_relevance"]:
            for duration in param["durations"]:
                for lock in param["locks"]:
                    for soa in param["soas"]:
                        # Extract the data:
                        data = epochs.copy()["/".join([task_rel, duration, lock, soa])].get_data(copy=False)
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
    save_dir = Path(ev.bids_root, "derivatives", "pupil_size", "group_level", "data")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Save the peak latencies:
    peak_latencies.to_csv(Path(save_dir, "peak_latencies.csv"))

    # Plot the peak latency:
    for lock in param["locks"]:
        fig, ax = plt.subplots(nrows=len(param["task_relevance"]), ncols=len(param["durations"]),
                               sharex=True, sharey=True, figsize=(20, 15))
        for task_i, task_rel in enumerate(param["task_relevance"]):
            for dur_i, dur in enumerate(param["durations"]):
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
                ax[task_i, dur_i].set_xticklabels(param["soas"])
        plt.suptitle("{} locked peak latency".format(lock))
        plt.tight_layout()
        # Create the save directory:
        save_dir = Path(ev.bids_root, "derivatives", "pupil_size", "group_level", "figures")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        # Save the figure:
        plt.savefig(Path(save_dir, "{}_locked_peak_latency.png".format(lock)))
        plt.savefig(Path(save_dir, "{}_locked_peak_latency.svg".format(lock)))
        plt.close(fig)


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121"]
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\02-pupil_peak_latency_parameters.json ")
    pupil_peak_latency(parameters, subjects_list)

