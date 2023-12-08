import json
import os
from pathlib import Path
import mne
import environment_variables as ev
from eye_tracker.general_helper_function import baseline_scaling
import matplotlib.pyplot as plt
from plotter_functions import latency_raster_plot


def plot_events_latencies(subjects, parameters_file):
    # First, load the parameters:
    with open(parameters_file) as json_file:
        param = json.load(json_file)

    # Loop through each subject:
    for sub in subjects:
        print(sub)
        # Create the save root:
        save_root = Path(ev.bids_root, "derivatives", param["event_type"], "sub-" + sub, "figures")
        if not os.path.exists(save_root):
            os.makedirs(save_root)
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
        # Extract the relevant channels:
        epochs.pick(param["picks"])

        # ===========================================
        # Preprocessing
        # Baseline correction:
        if param["baseline"] is not None:
            baseline_scaling(epochs, correction_method=param["baseline"], baseline=param["baseline_window"])

        # Plot the data:
        # ================================================================
        # Plot separately for onset and offset locked
        for lock in param["locks"]:
            fig = latency_raster_plot(epochs, lock, param["durations"], param["soas"],
                                      channels=param["picks"], audio_lock=False)
            # Save the figure:
            file_name = ("sub-{}_ses-{}_task-{}_{}_desc-{}.png"
                         "").format(sub, param["session"], param["task"], param["event_type"],
                                    lock)
            fig.savefig(Path(save_root, file_name))
            plt.close(fig)

            # Plot locked to the audio stimulus:
            fig = latency_raster_plot(epochs, lock, param["durations"], param["soas"], channels=param["picks"],
                                      audio_lock=True)
            # Save the figure:
            file_name = ("sub-{}_ses-{}_task-{}_{}_desc-{}-{}.png"
                         "").format(sub, param["session"], param["task"], param["event_type"],
                                    lock, 'audlock')
            fig.savefig(Path(save_root, file_name))
            plt.close(fig)

    return None
