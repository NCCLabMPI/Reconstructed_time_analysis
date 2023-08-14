import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem
from eye_tracker.general_helper_function import baseline_scaling
import os
from scipy.ndimage import uniform_filter1d

# Get matplotlib colors:
prop_cycle = plt.rcParams['axes.prop_cycle']
cwheel = prop_cycle.by_key()['color']
# First, load the parameters:
bids_root = r"C:\\Users\\alexander.lepauvre\\Documents\\PhD\\Reconstructed_Time\\bids"
visit = "1"
task = "prp"
session = "1"
data_type = "eyetrack"
epoch_names = ["visual_onset"]
crop = [-0.3, 3.0]
picks = ["LPupil", "RPupil"]
task_relevance = ["non-target", "irrelevant"]
durations = ["0.5", "1", "1.5"]
soas = ["0", "0.116", "0.232", "0.466"]
locks = ["onset", "offset"]


def get_blink_onset(data, times):
    """
    This function returns the onset of the blinks for each trial
    :param data: data containing the blinking information. The data should in the format of trials x time with 1 where
    a subject was blinking and 0 otherwise
    :param times: (array) the time vector
    :return: a list of arrays containing the onset of the blinks for each trial
    """
    blinking_onsets = []
    data_onset = np.diff(data, axis=-1)
    for trial in range(data.shape[0]):
           blinking_onsets.append(times[np.where(data_onset[trial, :] == 1)[0]])
    return blinking_onsets


def plot_eyetracker_data(subject):
    # Generate the save root:
    save_root = Path(bids_root, "derivatives", "plots", "sub-" + subject, "ses-" + session, "eyetracker")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for epoch_name in epoch_names:
        # Load the epochs:
        root = Path(bids_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                    data_type)
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(sub, session, task, data_type,
                                                                      epoch_name)
        epochs = mne.read_epochs(Path(root, file_name))
        # Crop if needed:
        epochs.crop(crop[0], crop[1])
        # Baseline correction for the relevant channels:
        baseline_scaling(epochs, correction_method="ratio", baseline=(None, -0.05), picks=["RPupil", "LPupil"])

        # ================================================================
        # Plot the blinking rate:
        # Extract the left and right eye:
        for lock in locks:
            # Open a figure:
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            blink_data = []
            blink_avg = []
            colors = []
            for ind, soa in enumerate(soas):
                data_l = np.squeeze(epochs.copy()["/".join([lock, soa])].get_data(picks=["Lblink"]))
                data_r = np.squeeze(epochs.copy()["/".join([lock, soa])].get_data(picks=["Rblink"]))
                # Combine both eyes such that if we have a 1 in both arrays, then we have a 1, otherwise a 0:
                blink_data.append(np.logical_and(data_l, data_r).astype(int))
                # Colors:
                colors.append([cwheel[ind]] * data_l.shape[0])
                # Blinking average:
                blink_avg.append(np.average(np.logical_and(data_l, data_r).astype(int), axis=0))
            blinks_onsets = get_blink_onset(np.concatenate(blink_data), epochs.times)
            # Plot the data:
            ax[0].eventplot(blinks_onsets, orientation="horizontal", colors=np.concatenate(colors), linelengths=1,
                            linewidths=8)
            # Smooth the blinking rate a little:
            blink_avg = np.array(blink_avg)
            blink_avg = uniform_filter1d(blink_avg, 10, axis=-1)
            ax[1].plot(epochs.times, blink_avg.T, label=soas)
            ax[0].set_title("Lock: {}".format(lock))
            plt.legend()
            plt.ylabel("Blinking rate")
            plt.xlabel("Time (s)")
            plt.tight_layout()
            # plt.show()
            # Save the figure:
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_ana-{}.png".format(sub, session, task, data_type,
                                                                             epoch_name, "blinking_rate_" + lock)
            fig.savefig(Path(save_root, file_name))
            plt.close(fig)

        # ================================================================
        # Plot the pupillometry:
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        # Extract the left and right eye:
        pupil_epo = epochs.copy().pick(["LPupil", "RPupil"])
        for ind, lock in enumerate(locks):
            for soa in soas:
                data_l = pupil_epo.copy()["/".join([lock, soa])].get_data(picks=["LPupil"])
                data_r = pupil_epo.copy()["/".join([lock, soa])].get_data(picks=["RPupil"])
                avg_l = np.mean(np.squeeze(data_l), axis=0)
                avg_r = np.mean(np.squeeze(data_r), axis=0)
                err_l = sem(np.squeeze(data_l), axis=0)
                err_r = sem(np.squeeze(data_r), axis=0)
                # Plot the average pupil size in that condition:
                ax[ind, 0].plot(epochs.times, avg_l, label=soa)
                # Add the error around that:
                ax[ind, 0].fill_between(epochs.times, avg_l - err_l, avg_l + err_l,
                                        alpha=0.3)
                # Plot the average pupil size in that condition:
                ax[ind, 1].plot(epochs.times, avg_r, label=soa)
                # Add the error around that:
                ax[ind, 1].fill_between(epochs.times, avg_r - err_r, avg_r + err_r,
                                        alpha=0.3)
            ax[ind, 0].set_title("Lock: {}, Eye: {}".format(lock, "Left"))
            ax[ind, 1].set_title("Lock: {}, Eye: {}".format(lock, "Right"))
        plt.legend()
        plt.ylabel("Pupil size (norm.)")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        # Save the figure:
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_ana-{}.png".format(sub, session, task, data_type,
                                                                         epoch_name, "pupil_size")
        fig.savefig(Path(save_root, file_name))
        plt.close(fig)

        # Same but plot separately for the task relevance and durations separately:
        for task_rel in task_relevance:
            for dur in durations:
                # Open a figure:
                fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
                # Add the super title:
                fig.suptitle("Task relevance: {}, Duration: {}".format(task_rel, dur))
                for ind, lock in enumerate(locks):
                    for soa in soas:
                        data_l = pupil_epo.copy()["/".join([lock, soa, task_rel, dur])].get_data(picks=["LPupil"])
                        data_r = pupil_epo.copy()["/".join([lock, soa, task_rel, dur])].get_data(picks=["RPupil"])
                        avg_l = np.mean(np.squeeze(data_l), axis=0)
                        avg_r = np.mean(np.squeeze(data_r), axis=0)
                        err_l = sem(np.squeeze(data_l), axis=0)
                        err_r = sem(np.squeeze(data_r), axis=0)
                        # Plot the average pupil size in that condition:
                        ax[ind, 0].plot(epochs.times, avg_l, label=soa)
                        # Add the error around that:
                        ax[ind, 0].fill_between(epochs.times, avg_l - err_l, avg_l + err_l,
                                                alpha=0.3)
                        # Plot the average pupil size in that condition:
                        ax[ind, 1].plot(epochs.times, avg_r, label=soa)
                        # Add the error around that:
                        ax[ind, 1].fill_between(epochs.times, avg_r - err_r, avg_r + err_r,
                                                alpha=0.3)
                    ax[ind, 0].set_title("Lock: {}, Eye: {}".format(lock, "Left"))
                    ax[ind, 1].set_title("Lock: {}, Eye: {}".format(lock, "Right"))
                plt.legend()
                plt.ylabel("Pupil size (norm.)")
                plt.xlabel("Time (s)")
                plt.tight_layout()
                # Save the figure:
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_ana-{}.png".format(sub, session, task, data_type,
                                                                                 epoch_name, "pupil_size_{}_{}".format(task_rel, dur))
                fig.savefig(Path(save_root, file_name))
                plt.close(fig)


if __name__ == "__main__":
    subjects_list = ["SX102"]
    for sub in subjects_list:
        try:
            print("Plotting subject {}".format(sub))
            plot_eyetracker_data(sub)
        except FileNotFoundError:
            print("Subject {} not found".format(sub))
