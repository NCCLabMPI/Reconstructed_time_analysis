import mne
from mne.viz.eyetracking import plot_gaze
import json
from pathlib import Path
from eye_tracker.general_helper_function import baseline_scaling
from eye_tracker.preprocessing_helper_function import (extract_eyelink_events, epoch_data,
                                                       load_raw_eyetracker, compute_proportion_bad, add_logfiles_info,
                                                       gaze_to_dva, hershman_blinks_detection, plot_blinks)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import environment_variables as ev
import os
from scipy.stats import zscore

DEBUG = False
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def preprocessing(subject, parameters):
    """
    This function preprocesses the eyetracking data, using several MNE key functionalities for handling the data
    :param subject: (string) name of the subject to process. Note: do not include the sub-!
    :param parameters: (string) parameter json file
    :return: None: saves the epochs to file
    """
    # First, load the parameters:
    with open(parameters) as json_file:
        param = json.load(json_file)
    # Extract the info about the session:
    session = param["session"]
    task = param["task"]
    data_type = param["data_type"]
    preprocessing_steps = param["preprocessing_steps"]

    # =============================================================================================
    # Load the data:
    files_root = Path(ev.bids_root, "sub-" + subject, "ses-" + session, data_type)
    # Load the eyetracker data and associated files:
    logs_list, raws_list, calibs_list, screen_size, screen_distance, screen_res = (
        load_raw_eyetracker(files_root, subject, session, task, ev.raw_root,
                            param["beh_file_name"],
                            param["epochs"]["metadata_column"],
                            param["events_of_interest"][0].replace('*', ''),
                            verbose=False, debug=DEBUG))
    # Concatenate the objects
    raw = mne.concatenate_raws(raws_list)
    if param["plot_blinks"]:
        plot_blinks(raw)
    # Remove the empty calibrations:
    calibs = list(filter(None, calibs_list))
    calibs = [item for items in calibs for item in items]
    # Concatenate the log files:
    log_df = pd.concat(logs_list).reset_index(drop=True)
    # Prepare the proportion of bad data:
    proportion_bad = 0

    # =============================================================================================
    # Loop through the preprocessing steps:
    for step in preprocessing_steps:
        # Extract the parameters of the current step:
        step_param = param[step]
        # Apply the hershman blinks detection algorithm:
        if step == "hershman_blinks":
            raw = hershman_blinks_detection(raw, eyes=step_param["eyes"],
                                            replace_eyelink_blinks=step_param["replace_eyelink_blinks"])
            if param["plot_blinks"]:
                plot_blinks(raw)

        # Detect chunks mark as blinks but which are too long to be blinks:
        if step == "remove_long_blinks":
            # Extract the index of the blinks that are too long:
            long_blinks_ind = np.where((raw.annotations.description == "BAD_blink") &
                                       (raw.annotations.duration > step_param["max_blinks_dur"]))[0]
            print("{} out of {} blinks duration exceeded {}sec and were categorized as bad segments!".format(
                len(long_blinks_ind), np.sum(raw.annotations.description == "BAD_blink"),
                step_param["max_blinks_dur"]))
            # Change the description to BAD:
            if len(long_blinks_ind) > 0:
                raw.annotations.description[long_blinks_ind] = step_param["new_description"]

        # Interpolate the data:
        if step == "interpolate_blinks":
            # Interpolate
            mne.preprocessing.eyetracking.interpolate_blinks(raw, buffer=step_param["buffer"],
                                                             match="BAD_blink",
                                                             interpolate_gaze=step_param["interpolate_gaze"])
            # Show where the data were interpolated:
            if param["plot_blinks"]:
                plot_blinks(raw, blinks_annotations=["blink", param["remove_long_blinks"]["new_description"]])

        # Extract the eyelink events as channels (to keep them after the epoching):
        if step == "extract_eyelink_events":
            print("Extracting the {} from the annotation".format(step_param["events"]))
            # Loop through each event to extract:
            for evt in step_param["events"]:
                raw = extract_eyelink_events(raw, evt, eyes=step_param["eyes"])
            print("A")

        if step == "gaze_to_dva":
            raw = gaze_to_dva(raw, screen_size, screen_res, screen_distance, eyes=step_param["eyes"])

        # Print the proportion of NaN in the data:
        proportion_bad = compute_proportion_bad(raw, desc="BAD_", eyes=["left", "right"])

        if step == "epochs":
            # Convert the annotations to event for epoching:
            print('Creating annotations')
            events_from_annot, event_dict = mne.events_from_annotations(raw, verbose="ERROR",
                                                                        regexp=param["events_of_interest"][0])
            # Epoch the data:
            epochs = epoch_data(raw, events_from_annot, event_dict, **step_param)

            # Add the log file information to the metadata
            if len(param["log_file_columns"]) > 0:
                epochs = add_logfiles_info(epochs, log_df, param["log_file_columns"])

            # Save this epoch to file:
            save_root = Path(ev.bids_root, "derivatives", "preprocessing", "sub-" + subject,
                             "ses-" + session, data_type)
            if not os.path.isdir(save_root):
                os.makedirs(save_root)
            # Generate the file name:
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(subject, session, task, data_type)
            # Save:
            epochs.save(Path(save_root, file_name), overwrite=True, verbose="ERROR")
            epochs.load_data()

            # ==========================================================================================================
            # Checks plots:

            # Depending on whehter or no the events were extracted:
            if "extract_eyelink_events" in preprocessing_steps:
                # Plot the blinks rate:
                fig, ax = plt.subplots(2)
                ax[0].imshow(np.squeeze(epochs.get_data(picks="blink_left")), aspect="auto", origin="lower",
                             extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                ax[1].imshow(np.squeeze(epochs.get_data(picks="blink_right")), aspect="auto", origin="lower",
                             extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                ax[0].set_title("Left eye")
                ax[1].set_title("Right eye")
                ax[1].set_xlabel("Time (s)")
                ax[1].set_ylabel("Trials")
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-blinks.png".format(subject, session, task,
                                                                              data_type)
                plt.savefig(Path(save_root, file_name))
                plt.close()

                # Plot the saccades rate:
                fig, ax = plt.subplots(2)
                ax[0].imshow(np.squeeze(epochs.get_data(picks="saccade_left")), aspect="auto", origin="lower",
                             extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                ax[1].imshow(np.squeeze(epochs.get_data(picks="saccade_right")), aspect="auto", origin="lower",
                             extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                ax[0].set_title("Left eye")
                ax[1].set_title("Right eye")
                ax[1].set_xlabel("Time (s)")
                ax[1].set_ylabel("Trials")
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-saccades.png".format(subject, session, task,
                                                                                data_type)
                plt.savefig(Path(save_root, file_name))
                plt.close()

                # Plot the fixation rate:
                fig, ax = plt.subplots(2)
                ax[0].imshow(np.squeeze(epochs.get_data(picks="fixation_left")), aspect="auto", origin="lower",
                             extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                ax[1].imshow(np.squeeze(epochs.get_data(picks="fixation_right")), aspect="auto", origin="lower",
                             extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                ax[0].set_title("Left eye")
                ax[1].set_title("Right eye")
                ax[1].set_xlabel("Time (s)")
                ax[1].set_ylabel("Trials")
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-fixation.png".format(subject, session, task,
                                                                                data_type)
                plt.savefig(Path(save_root, file_name))
                plt.close()
                for factor in param["plot_factors"]:
                    # Create a figure to plot the histogram:
                    fig, ax = plt.subplots()
                    levels = list(epochs.metadata[factor].unique())
                    # Loop through each level:
                    for i, lvl in enumerate(levels):
                        blink_data = (np.logical_and(np.squeeze(epochs[lvl].get_data(copy=True,
                                                                                     picks=["blink_left"])),
                                                     np.squeeze(epochs[lvl].get_data(copy=True,
                                                                                     picks=["blink_right"]))).
                                      astype(float))
                        if len(blink_data.shape) > 1:
                            blink_counts = np.sum(np.diff(blink_data, axis=1) == 1, axis=1)
                            # Plot the blinks counts as a histogram, adding jitters to each condition to see them
                            # distinctively
                            ax.hist(blink_counts + 0.2 * i, color=colors[i], alpha=0.3, label=lvl, rwidth=0.2)
                    ax.set_xlabel("Blinks counts")
                    ax.set_ylabel("Counts")
                    ax.legend()
                    ax.set_title(factor)
                    file_name = "sub-{}_ses-{}_task-{}_{}_desc-blinks-{}.png".format(subject, session, task,
                                                                                     data_type, factor)
                    # Create a histogram of the blink counts for each level:
                    plt.savefig(Path(save_root, file_name))
                    plt.close()

            # ======================================================
            # Fixation maps:
            plot_gaze(epochs, width=1920, height=1080, show=False)
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-gaze.png".format(subject, session, task,
                                                                        data_type)
            plt.savefig(Path(save_root, file_name))
            plt.close()

            # ======================================================
            # Baseline distributions:
            baseline_data = epochs.copy().crop(tmin=epochs.times[0], tmax=0).get_data(picks=["pupil_left",
                                                                                             "pupil_right"])
            # Compute the average across eyes and time:
            baseline_avg = np.mean(np.mean(baseline_data, axis=1), axis=1)
            # Z score:
            baseline_zscore = zscore(baseline_avg, nan_policy='omit')
            fig, ax = plt.subplots()
            ax.hist(baseline_zscore, bins=50)
            ax.vlines(x=-2, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle="-", color="r", linewidth=2,
                      zorder=10)
            ax.vlines(x=2, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle="-", color="r", linewidth=2,
                      zorder=10)
            ax.set_title("Baseline distributions")
            ax.set_xlabel("z-score")
            ax.set_ylabel("Trials counts")
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-baseline_distribution.png".format(subject, session, task,
                                                                                         data_type)
            plt.savefig(Path(save_root, file_name))
            plt.close()

            # ======================================================
            # Pupil size line plot:
            # Extract the pupil sizes:
            pupil_data = epochs.copy().get_data(picks=["pupil_left", "pupil_right"])
            # Compute the average across eyes and time:
            pupil_avg = np.mean(pupil_data, axis=1)
            fig, ax = plt.subplots()
            ax.plot(epochs.times, pupil_avg.T, alpha=0.5)
            ax.set_title("Pupil size per trial")
            ax.set_xlabel("Time (sec.)")
            ax.set_ylabel("Pupil size (a.u)")
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-pupil_lines.png".format(subject, session, task,
                                                                               data_type)
            plt.savefig(Path(save_root, file_name))
            plt.close()

            # Plot baseline corrected data:
            pupil_epochs = epochs.copy().pick(["pupil_left", "pupil_right"])
            baseline_scaling(pupil_epochs, correction_method="percent", baseline=[None, -0.05])
            pupil_data = pupil_epochs.get_data(copy=True)
            # Compute the average across eyes and time:
            pupil_avg = np.mean(pupil_data, axis=1)
            fig, ax = plt.subplots()
            ax.plot(epochs.times, pupil_avg.T, alpha=0.5)
            ax.set_title("Pupil size per trial")
            ax.set_xlabel("Time (sec.)")
            ax.set_ylabel("Pupil size (%change)")
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-pupil_lines_bascorr.png".format(subject, session, task,
                                                                                       data_type)
            plt.savefig(Path(save_root, file_name))
            plt.close()

            # Loop through each factor:
            for factor in param["plot_factors"]:
                # Create a figure to plot the histogram:
                fig, ax = plt.subplots()
                levels = list(pupil_epochs.metadata[factor].unique())
                # Loop through each level:
                evks = []
                for i, lvl in enumerate(levels):
                    pupil_data = pupil_epochs[lvl].get_data(copy=True)
                    # Compute the average across eyes and time:
                    pupil_avg = np.mean(pupil_data, axis=1)
                    # Plot Single trials:
                    ax.plot(pupil_epochs.times, pupil_avg.T, color=colors[i], alpha=0.3, linewidth=0.2)
                    # Plot evoked:
                    evk = np.mean(pupil_avg, axis=0)
                    ax.plot(pupil_epochs.times, evk, color=colors[i], label=lvl, linewidth=2, zorder=10000)
                    evks.append(evk)
                if len(evks) == 2:
                    ax.plot(pupil_epochs.times, evks[0] - evks[1], color="r", label="diff",
                            linewidth=2)
                ax.set_xlabel("Times (sec.)")
                ax.set_ylabel("Pupil size")
                ax.legend()
                ax.set_title(factor)
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-pupil_{}.png".format(subject, session, task,
                                                                                data_type, factor)
                plt.savefig(Path(save_root, file_name))
                plt.close()

            # Plot a heatmap of the baseline corrected pupil size:
            fig, ax = plt.subplots()
            ax.imshow(pupil_avg, aspect="auto", origin="lower",
                      extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)],
                      vmin=np.nanpercentile(pupil_avg, 5), vmax=np.nanpercentile(pupil_avg, 95))
            ax.set_title("Pupil size")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Trials")
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-pupil_raster.png".format(subject, session, task,
                                                                                data_type)
            plt.savefig(Path(save_root, file_name))
            plt.close()

            # ======================================================
            # Fixation maps:
            # Calibrations:
            for calib_i, calib in enumerate(calibs):
                calib.plot(show=False)
                file_name = "calibration-{}_task-{}_eye-{}.png".format(calib_i, task, calib['eye'])
                plt.savefig(Path(save_root, file_name))
                plt.close()

    return np.mean(proportion_bad)


if __name__ == "__main__":
    # The following subjects have the specified issues:
    # SX101: differences in sampling rate due to experiment program issues
    # SX104: missing files
    # SX117: no eyetracking data
    # ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112", "SX113",
    # "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121"]
    subjects_list = ["SX108", "SX109", "SX110", "SX111", "SX112", "SX113",
                     "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121"]

    parameters_file = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker"
        r"\01-preprocessing_parameters_task-prp.json ")
    # Create a data frame to save the summary of all subjects:
    preprocessing_summary = []
    for sub in subjects_list:
        print("Preprocessing subject {}".format(sub))
        continuous_bad = preprocessing(sub, parameters_file)
        # Append to the summary:
        preprocessing_summary.append(pd.DataFrame({
            "subject": sub,
            "total_bad": continuous_bad,
            "valid_flag": True if np.min(continuous_bad) < 0.5 else False
        }, index=[0]))
    # Concatenate the data frame:
    preprocessing_summary = pd.concat(preprocessing_summary)
    # Save the data frame:
    save_dir = Path(ev.bids_root, "derivatives", "preprocessing")
    preprocessing_summary.to_csv(Path(save_dir, "participants.csv"))
