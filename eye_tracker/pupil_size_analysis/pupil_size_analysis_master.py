import mne
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from eye_tracker.general_helper_function import baseline_scaling
import os

# First, load the parameters:
bids_root = r"C:\\Users\\alexander.lepauvre\\Documents\\PhD\\Reconstructed_Time\\bids"
visit = "1"
task = "prp"
session = "1"
data_type = "eyetrack"
epoch_name = "visual_onset"
crop = [-0.3, 2.0]
picks = ["LPupil", "RPupil"]
task_relevance = ["non-target", "irrelevant"]
durations = ["0.5", "1", "1.5"]
soas = ["0", "0.116", "0.232", "0.466"]
locks = ["onset", "offset"]
subjects = ["SX102", "SX105", "SX106", "SX107", "SX108", "SX110", "SX111"]

subjects_epochs = []

for sub in subjects:
    # Load the epochs:
    root = Path(bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                data_type)
    file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(sub, session, task, data_type,
                                                                  epoch_name)
    epochs = mne.read_epochs(Path(root, file_name))
    # Crop if needed:
    epochs.crop(crop[0], crop[1])
    # Extract the relevant channels:
    epochs.pick(picks)
    # Baseline correction:
    baseline_scaling(epochs, correction_method="mean", baseline=(None, -0.05))
    subjects_epochs.append(epochs)

    # Plot single subjects data:
    for lock in locks:
        for rel in task_relevance:
            # ===========================================
            fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
            for ind, dur in enumerate(durations):
                # Extract the data from each soa:
                for soa in soas:
                    data = epochs.copy()["/".join([lock, rel, soa, dur])].get_data()
                    avg = np.mean(data, axis=(1, 0))
                    err = sem(np.mean(data, axis=1), axis=0)
                    # Plot the average pupil size in that condition:
                    ax[ind].plot(epochs.times, avg, label=soa)
                    # Add the error around that:
                    ax[ind].fill_between(epochs.times, avg - err, avg + err,
                                         alpha=0.3)
                    ax[ind].set_title("Task: {}, Lock: {}, Dur: {}".format(rel, lock, dur))
            # Save the figure:
            # Save the figure:
            plt.vlines([float(soa) for soa in soas], ymin=ax[0].get_ylim()[0], ymax=ax[0].get_ylim()[1])
            plt.legend()
            plt.ylabel("Pupil size (norm.)")
            plt.xlabel("Time (s)")
            plt.tight_layout()
            # Save the figure:
            save_root = Path(bids_root, "derivatives", "pupil_size", "sub-" + sub,
                             "ses-" + session, data_type)
            if not os.path.isdir(save_root):
                os.makedirs(save_root)
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_ana-{}.png".format(sub, session, task, data_type, epoch_name,
                                                                             "_".join([rel, lock]))
            fig.savefig(Path(save_root, file_name))
            plt.close(fig)


# Plotting the group level analysis:
# Plot single subjects data:
for lock in locks:
    for rel in task_relevance:
        # ===========================================
        fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
        for ind, dur in enumerate(durations):
            # Extract the data from each soa:
            for soa in soas:
                evks = []
                for epo in subjects_epochs:
                    data = epo.copy()["/".join([lock, rel, soa, dur])].get_data()
                    evks.append(np.mean(data, axis=(1, 0)))
                grand_avg = np.mean(np.array(evks), axis=0)
                err = sem(np.array(evks), axis=0)
                # Plot the average pupil size in that condition:
                ax[ind].plot(epochs.times, grand_avg, label=soa)
                # Add the error around that:
                ax[ind].fill_between(epochs.times, grand_avg - err, grand_avg + err,
                                     alpha=0.3)
                ax[ind].set_title("Task: {}, Lock: {}, Dur: {}".format(rel, lock, dur))

        # Save the figure:
        plt.vlines([float(soa) for soa in soas], ax[0].get_ylim())
        plt.legend()
        plt.ylabel("Pupil size (norm.)")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        # Save the figure:
        save_root = Path(bids_root, "derivatives", "pupil_size", "sub-" + "group",
                         "ses-" + session, data_type)
        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_ana-{}.png".format("group", session, task, data_type, epoch_name,
                                                                         "_".join([rel, lock]))
        fig.savefig(Path(save_root, file_name))
        plt.close(fig)
