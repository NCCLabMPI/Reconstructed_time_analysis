import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem
from eye_tracker.plotter_functions import plot_events_latencies
from eye_tracker.general_helper_function import baseline_scaling
import os


# Get matplotlib colors:
prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']
# First, load the parameters:
bids_root = r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\bids"
visit = "1"
experiment = "prp"
session = "1"
data_type = "eyetrack"
epoch_names = ["visual_onset"]
crop = [-0.3, 3.5]
picks = ["LPupil", "RPupil"]
task_relevance = ["non-target", "irrelevant"]
durations = ["0.5", "1", "1.5"]
soas = ["0", "0.116", "0.232", "0.466"]
soas_colors = [
    [179, 156, 77],
    [238, 99, 82],
    [11, 93, 30],
    [29, 132, 181]
]
locks = ["onset", "offset"]


def plot_eyetracker_data(subject):
    # Generate the save root:
    save_root = Path(bids_root, "derivatives", "plots", "sub-" + subject, "ses-" + session, "eyetracker")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for epoch_name in epoch_names:
        # Load the epochs:
        root = Path(bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                    data_type)
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(sub, session, experiment, data_type,
                                                                      epoch_name)
        epochs = mne.read_epochs(Path(root, file_name))
        # Crop if needed:
        epochs.crop(crop[0], crop[1])
        # Baseline correction for the relevant channels:
        baseline_scaling(epochs, correction_method="ratio", baseline=(None, -0.05), picks=["RPupil", "LPupil"])

        # ================================================================
        # Plot the blinking rate:
        # Plot separately for onset and offset locked
        for lock in locks:
            fig = plot_events_latencies(epochs, lock, durations, soas, channels=["Lblink", "Rblink"], audio_lock=False)
            save_root_ = Path(save_root, 'blinks', lock)
            if not os.path.exists(save_root_):
                os.makedirs(save_root_)
            # Save the figure:
            file_name = ("sub-{}_ses-{}_task-{}_{}_desc-{}.png"
                         "").format(sub, session, experiment, 'blinks',
                                    lock)
            fig.savefig(Path(save_root_, file_name))
            plt.close(fig)

            # Plot locked to the audio stimulus:
            fig = plot_events_latencies(epochs, lock, durations, soas, channels=["Lblink", "Rblink"], audio_lock=True)
            save_root_ = Path(save_root, 'blinks', lock)
            if not os.path.exists(save_root_):
                os.makedirs(save_root_)
            # Save the figure:
            file_name = ("sub-{}_ses-{}_task-{}_{}_desc-{}-{}.png"
                         "").format(sub, session, experiment, 'blinks',
                                    lock, 'audlock')
            fig.savefig(Path(save_root_, file_name))
            plt.close(fig)

        # ================================================================
        # Plot the saccades rate:
        # Plot separately for onset and offset locked
        for lock in locks:
            fig = plot_events_latencies(epochs, lock, durations, soas, channels=["Lsaccade", "Rsaccade"],
                                        audio_lock=False)
            save_root_ = Path(save_root, 'saccades', lock)
            if not os.path.exists(save_root_):
                os.makedirs(save_root_)
            # Save the figure:
            file_name = ("sub-{}_ses-{}_task-{}_{}_desc-{}.png"
                         "").format(sub, session, experiment, 'saccades',
                                    lock)
            fig.savefig(Path(save_root_, file_name))
            plt.close(fig)

            # Create the same plots but locked to the audio signal, since it is what's important:
            # Open a figure:
            fig = plot_events_latencies(epochs, lock, durations, soas, channels=["Lsaccade", "Rsaccade"],
                                        audio_lock=True)
            save_root_ = Path(save_root, 'saccades', lock)
            if not os.path.exists(save_root_):
                os.makedirs(save_root_)
            # Save the figure:
            file_name = ("sub-{}_ses-{}_task-{}_{}_desc-{}-{}.png"
                         "").format(sub, session, experiment, 'saccades',
                                    lock, 'audlock')
            fig.savefig(Path(save_root_, file_name))
            plt.close(fig)

        # ================================================================
        # Plot the pupillometry:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        # Extract the left and right eye:
        pupil_epo = epochs.copy().pick(["LPupil", "RPupil"])
        for ind, lock in enumerate(locks):
            for soa in soas:
                pupil_data = np.nanmean(pupil_epo.copy()["/".join([lock, soa])].get_data(), axis=1)
                avg = np.mean(np.squeeze(pupil_data), axis=0)
                err = sem(np.squeeze(pupil_data), axis=0)
                # Plot the average pupil size in that condition:
                ax[ind].plot(epochs.times, avg, label=soa)
                # Add the error around that:
                ax[ind].fill_between(epochs.times, avg - err, avg + err,
                                     alpha=0.3)
            ax[ind].set_title("Lock: {}".format(lock))
        plt.legend()
        plt.ylabel("Pupil size (norm.)")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        # Save the figure:
        file_name = ("sub-{}_ses-{}_task-{}_{}.png"
                     "").format(sub, session, experiment, 'pup')
        fig.savefig(Path(save_root, file_name))
        plt.close(fig)

        # Same but plot separately for the task relevance and durations separately:
        for lock in locks:
            # Open a figure:
            fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(20, 15))
            for task_i, task_rel in enumerate(task_relevance):
                for dur_i, dur in enumerate(durations):
                    # Add the super title:
                    fig.suptitle("Task relevance: {}, Duration: {}".format(task_rel, dur))
                    for soa in soas:
                        data = np.mean(pupil_epo.copy()["/".join([lock, soa, task_rel,
                                                                  dur])].get_data(picks=["LPupil",
                                                                                         "RPupil"]), axis=1)
                        avg = np.mean(np.squeeze(data), axis=0)
                        err = sem(np.squeeze(data), axis=0)
                        # Plot the average pupil size in that condition:
                        ax[task_i, dur_i].plot(epochs.times, avg, label=soa)
                        # Add the error around that:
                        ax[task_i, dur_i].fill_between(epochs.times, avg - err, avg + err,
                                                       alpha=0.3)
                        ax[task_i, dur_i].set_title("{}, {}".format(task_rel, dur))
            plt.legend()
            plt.ylabel("Pupil size (norm.)")
            plt.xlabel("Time (s)")
            plt.tight_layout()
            # Save the figure:
            file_name = ("sub-{}_ses-{}_task-{}_{}_desc-{}.png"
                         "").format(sub, session, experiment, 'pup',
                                    lock)
            fig.savefig(Path(save_root, file_name))
            plt.close(fig)


if __name__ == "__main__":
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121"]
    for sub in subjects_list:
        try:
            print("Plotting subject {}".format(sub))
            plot_eyetracker_data(sub)
        except FileNotFoundError:
            print("Subject {} not found".format(sub))
