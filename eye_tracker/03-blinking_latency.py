import mne
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from eye_tracker.general_helper_function import baseline_scaling
import seaborn as sns
import pandas as pd
import os

# Set the font size:
plt.rcParams.update({'font.size': 22})


def blinking_latency(parameters_file, subjects):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)

    blink_latencies = pd.DataFrame()
    # Loop through each subject:
    for sub in subjects:
        print(sub)
        # ===========================================
        # Data loading:
        # Load the epochs:
        root = Path(param["bids_root"], "derivatives", "preprocessing", "sub-" + sub, "ses-" + param["session"],
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
        if param["baseline"] is not None:
            baseline_scaling(epochs, correction_method=param["baseline"], baseline=param["baseline_window"])

        # Loop through each relevant condition:
        for task_rel in param["task_relevance"]:
            for duration in param["durations"]:
                for lock in param["locks"]:
                    for soa in param["soas"]:
                        # Extract the data:
                        data = epochs.copy()["/".join([task_rel, duration, lock, soa])].get_data()
                        # Reject any  "blink" that was found only in 1 eye:
                        data = np.logical_and(data[:, 0, :], data[:, 1, :])
                        # Find the onset of the blinks in each trial:
                        blink_onsets = []
                        for trial in range(data.shape[0]):
                            blinks_inds = np.where(data[trial, :] == 1)[0]
                            if len(blinks_inds) > 0:
                                blink_onsets.append(epochs.times[blinks_inds[0]])
                        # Average across trials:
                        blink_latency = np.mean(blink_onsets, axis=0)
                        # The peak latency needs to be adjusted relative to the audio lock and SOA:
                        if lock == "onset":
                            blink_latency = blink_latency - float(soa)
                        else:
                            blink_latency = blink_latency - (float(soa) + float(duration))
                        # Add to data frame using pd.concat:
                        blink_latencies = pd.concat([blink_latencies, pd.DataFrame({"subject": sub,
                                                                                    "task_relevance": task_rel,
                                                                                    "duration": duration,
                                                                                    "lock": lock,
                                                                                    "soa": soa,
                                                                                    "blink_latency": blink_latency},
                                                                                   index=[0])])
    blink_latencies = blink_latencies.reset_index(drop=True)
    # Create the save directory:
    save_dir = Path(param["bids_root"], "derivatives", "blinks", "group_level", "data")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Save the peak latencies:
    blink_latencies.to_csv(Path(save_dir, "blink_latencies.csv"))

    # Plot the peak latency:
    for lock in param["locks"]:
        fig, ax = plt.subplots(nrows=len(param["task_relevance"]), ncols=len(param["durations"]),
                               sharex=True, sharey=True, figsize=(20, 15))
        for task_i, task_rel in enumerate(param["task_relevance"]):
            for dur_i, dur in enumerate(param["durations"]):
                # Extract the data of this particular condition:
                df = blink_latencies.loc[(blink_latencies["task_relevance"] == task_rel)
                                         & (blink_latencies["duration"] == dur)
                                         & (blink_latencies["lock"] == lock), ["subject", "soa", "blink_latency"]]
                # Create box plot:
                sns.boxplot(x='soa', y='blink_latency', data=df, ax=ax[task_i, dur_i])
                # Add individual points:
                sns.stripplot(x='soa', y='blink_latency', data=df, jitter=True, color='black', alpha=0.5,
                              ax=ax[task_i, dur_i])
                # Add lines for each subject:
                for subject in df['subject'].unique():
                    # Extract the data of this subject:
                    sub_data = df[df['subject'] == subject]
                    ax[task_i, dur_i].plot(sub_data['soa'], sub_data['blink_latency'], color='grey',
                                           linewidth=0.5, alpha=0.5, )
                ax[task_i, dur_i].set_title("{}, {}sec".format(task_rel, dur, lock))
                ax[task_i, dur_i].set_ylabel("Blink latency (s)")
                ax[task_i, dur_i].set_xlabel("SOA (s)")
                ax[task_i, dur_i].set_xticklabels(param["soas"])
        plt.suptitle("{} locked peak latency".format(lock))
        plt.tight_layout()
        # Create the save directory:
        save_dir = Path(param["bids_root"], "derivatives", "blinks", "group_level", "figures")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        # Save the figure:
        plt.savefig(Path(save_dir, "{}_locked_blinks_latency.png".format(lock)))
        plt.savefig(Path(save_dir, "{}_locked_blinks_latency.svg".format(lock)))
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121"]
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\03-blinking_latency_parameters.json ")
    blinking_latency(parameters, subjects_list)
