import mne
from mne.viz.eyetracking import plot_gaze
import json
from pathlib import Path
from helper_function.helper_general import baseline_scaling
from helper_function.helper_preprocessing import (extract_eyelink_events, epoch_data,
                                                  load_raw_eyetracker, compute_proportion_bad, add_logfiles_info,
                                                  gaze_to_dva, hershman_blinks_detection, plot_blinks,
                                                  annotate_nan, reject_bad_epochs, format_summary_table,
                                                  load_cog_eyetracker)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import environment_variables as ev
import os
from scipy.stats import zscore

DEBUG = False
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def preprocessing(subject, parameters, bids_root, session="1", task="prp"):
    """
    This function preprocesses the eyetracking data, using several MNE key functionalities for handling the data
    :param subject: (string) name of the subject to process. Note: do not include the sub-!
    :param parameters: (string) parameter json file
    :param session: (string) session for the data
    :param task: (string) task for the data
    :return: None: saves the epochs to file
    """
    # First, load the parameters:
    with open(parameters) as json_file:
        param = json.load(json_file)
    # Extract the info about the session:
    data_type = param["data_type"]
    preprocessing_steps = param["preprocessing_steps"]

    # =============================================================================================
    # Load the eyetracker data and associated files:
    if "SX" in subject:
        logs_list, raws_list, calibs_list, screen_size, screen_distance, screen_res = (
            load_raw_eyetracker(bids_root, subject, session, task,
                                param["beh_file_name"],
                                param["epochs"]["metadata_column"],
                                param["events_of_interest"][0].replace('*', ''),
                                verbose=False, debug=DEBUG))
        # Concatenate the objects
        raw = mne.concatenate_raws(raws_list)
        # Concatenate the log files:
        log_df = pd.concat(logs_list).reset_index(drop=True)
    else:
        raws_list, calibs_list, screen_size, screen_distance, screen_res = (
            load_cog_eyetracker(bids_root, subject, session, task,
                                verbose=False, debug=DEBUG))
        # Concatenate the objects
        raw = mne.concatenate_raws(raws_list)
        # Concatenate the log files:
        log_df = None

    if param["plot_blinks"]:
        plot_blinks(raw)
    # Remove the empty calibrations:
    calibs = list(filter(None, calibs_list))
    calibs = [item for items in calibs for item in items]
    # Prepare the proportion of bad data:
    proportion_bad = 0
    drop_log = None

    # Determine which eyes were recorded:
    eyes = [ch.split("_")[-1] for ch in raw.ch_names if "pupil" in ch]
    binocular = True if len(eyes) == 2 else False

    # =============================================================================================
    # Loop through the preprocessing steps:
    for step in preprocessing_steps:
        # Extract the parameters of the current step:
        step_param = param[step]

        # Mark nan samples as bad:
        if step == "annotate_nan":
            raw = annotate_nan(raw, eyes=eyes, nan_annotation=step_param["nan_annotation"])

        # Apply the hershman blinks detection algorithm:
        if step == "hershman_blinks":
            raw = hershman_blinks_detection(raw, eyes=eyes,
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
                raw = extract_eyelink_events(raw, evt, eyes=eyes)
            print("A")

        if step == "gaze_to_dva":
            raw = gaze_to_dva(raw, screen_size, screen_res, screen_distance, eyes=eyes)

        # Print the proportion of NaN in the data:
        proportion_bad = compute_proportion_bad(raw, desc="BAD_", eyes=["left", "right"])

        if step == "epochs":
            # Convert the annotations to event for epoching:
            print('Creating annotations')
            events_from_annot, event_dict = mne.events_from_annotations(raw, verbose="ERROR",
                                                                        regexp=param["events_of_interest"][0])
            # Epoch the data:
            epochs = epoch_data(raw, events_from_annot, event_dict, **step_param)
            epochs.load_data()
            # Add the log file information to the metadata
            if len(param["log_file_columns"]) > 0:
                epochs = add_logfiles_info(epochs, log_df, param["log_file_columns"])

            # Remove the bad epochs if needed:
            if "reject_bad_epochs" in preprocessing_steps:
                epochs, n_rej = reject_bad_epochs(epochs,
                                                  baseline_window=param["reject_bad_epochs"]["baseline_window"],
                                                  z_thresh=param["reject_bad_epochs"]["z_thresh"],
                                                  eyes=eyes,
                                                  exlude_beh=param["reject_bad_epochs"]["exlude_beh"])
            # Extract the drop log:
            drop_log = epochs.drop_log
            # Save this epoch to file:
            save_root = Path(bids_root, "derivatives", "preprocessing", "sub-" + subject,
                             "ses-" + session, data_type)
            if not os.path.isdir(save_root):
                os.makedirs(save_root)
            # Generate the file name:
            file_name = "sub-{}_ses-{}_task-{}_{}_desc-epo.fif".format(subject, session, task, data_type)
            # Save:
            epochs.save(Path(save_root, file_name), overwrite=True, verbose="ERROR")

            # ==========================================================================================================
            # Checks plots:

            # Depending on whehter or no the events were extracted:
            if "extract_eyelink_events" in preprocessing_steps:
                # Plot the blinks rate:
                if binocular:
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(np.squeeze(epochs.get_data(picks="blink_right")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[1].imshow(np.squeeze(epochs.get_data(picks="blink_left")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[0].set_title("Left eye")
                    ax[1].set_title("Right eye")
                    ax[1].set_xlabel("Time (s)")
                    ax[1].set_ylabel("Trials")
                else:
                    fig, ax = plt.subplots()
                    ax.imshow(np.squeeze(epochs.get_data(picks=f"blink_{eyes[0]}")),
                              aspect="auto", origin="lower",
                              extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Trials")
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-blinks.png".format(subject, session, task,
                                                                              data_type)
                plt.savefig(Path(save_root, file_name))
                plt.close()

                # Plot the saccades rate:
                if binocular:
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(np.squeeze(epochs.get_data(picks="saccade_right")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[1].imshow(np.squeeze(epochs.get_data(picks="saccade_left")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[0].set_title("Left eye")
                    ax[1].set_title("Right eye")
                    ax[1].set_xlabel("Time (s)")
                    ax[1].set_ylabel("Trials")
                else:
                    fig, ax = plt.subplots()
                    ax.imshow(np.squeeze(epochs.get_data(picks=f"saccade_{eyes[0]}")),
                              aspect="auto", origin="lower",
                              extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Trials")
                file_name = "sub-{}_ses-{}_task-{}_{}_desc-saccades.png".format(subject, session, task,
                                                                                data_type)
                plt.savefig(Path(save_root, file_name))
                plt.close()

                # Plot the fixation rate:
                if binocular:
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(np.squeeze(epochs.get_data(picks="fixation_left")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[1].imshow(np.squeeze(epochs.get_data(picks="fixation_right")), aspect="auto", origin="lower",
                                 extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax[0].set_title("Left eye")
                    ax[1].set_title("Right eye")
                    ax[1].set_xlabel("Time (s)")
                    ax[1].set_ylabel("Trials")
                else:
                    fig, ax = plt.subplots()
                    ax.imshow(np.squeeze(epochs.get_data(picks=f"fixation_{eyes[0]}")),
                              aspect="auto", origin="lower",
                              extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)])
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Trials")
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
                        if binocular:
                            blink_data = (np.logical_and(np.squeeze(epochs[lvl].get_data(copy=True,
                                                                                         picks=["blink_left"])),
                                                         np.squeeze(epochs[lvl].get_data(copy=True,
                                                                                         picks=["blink_right"]))).
                                          astype(float))
                        else:
                            blink_data = np.squeeze(epochs[lvl].get_data(copy=True,
                                                                         picks=f"blink_{eyes[0]}"))
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
            if binocular:
                baseline_data = epochs.copy().crop(tmin=epochs.times[0], tmax=0).get_data(picks=["pupil_left",
                                                                                                 "pupil_right"])
                # Compute the average across eyes and time:
                baseline_avg = np.mean(np.mean(baseline_data, axis=1), axis=1)
            else:
                baseline_avg = np.squeeze(epochs.copy().crop(tmin=epochs.times[0], tmax=0).get_data(
                    picks=f"pupil_{eyes[0]}"))
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
            if binocular:
                pupil_data = epochs.copy().get_data(picks=["pupil_left", "pupil_right"])
                # Compute the average across eyes and time:
                pupil_avg = np.mean(pupil_data, axis=1)
            else:
                pupil_avg = np.squeeze(
                    epochs.copy().get_data(picks=f"pupil_{eyes[0]}"))
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
            if binocular:
                pupil_epochs = epochs.copy().pick(["pupil_left", "pupil_right"])
            else:
                pupil_epochs = epochs.copy().pick(f"pupil_{eyes[0]}")
            baseline_scaling(pupil_epochs, correction_method="percent", baseline=[None, -0.05])
            pupil_data = pupil_epochs.get_data(copy=True)
            # Compute the average across eyes and time:
            if binocular:
                pupil_avg = np.mean(pupil_data, axis=1)
            else:
                pupil_avg = np.squeeze(pupil_data)
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
                try:
                    calib.plot(show=False)
                    file_name = "calibration-{}_task-{}_eye-{}.png".format(calib_i, task, calib['eye'])
                    plt.savefig(Path(save_root, file_name))
                    plt.close()
                except ValueError:
                    print("WARNING: Could not plot the calibration!")
    return proportion_bad, drop_log


if __name__ == "__main__":
    # ==================================================================================
    # COGITATE DATA:
    # Set the parameters:
    parameters_file = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\02-ET_preprocessing_parameters_cog.json")
    task = "Dur"
    preprocessing_summary = {subject: {"drop_logs": None, "proportion_bad": None}
                             for subject in ev.subjects_ecog_eyetrack["dur"]}
    for sub in ev.subjects_ecog_eyetrack["dur"]:
        print("Preprocessing subject {}".format(sub))
        prop_bad, drop_logs = preprocessing(sub, parameters_file, ev.cog_bids_root,
                                            session="1", task=task)
        preprocessing_summary[sub]["proportion_bad"] = np.mean(prop_bad)
        preprocessing_summary[sub]["drop_logs"] = [item[0] if len(item) > 0 else ''
                                                   for item in drop_logs]
    preprocessing_summary = format_summary_table(preprocessing_summary)
    # Save the data frame:
    save_dir = Path(ev.cog_bids_root, "derivatives", "preprocessing")
    preprocessing_summary.to_csv(Path(save_dir, "participants_Dur.csv"))

    # Set the parameters:
    parameters_file = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\02-ET_preprocessing_parameters.json")
    # ==================================================================================
    # Auditory practice preprocessing:
    task = "auditory"
    preprocessing_summary = {subject: {"drop_logs": None, "proportion_bad": None}
                             for subject in ev.subjects_lists_et["prp"]}
    for sub in ev.subjects_lists_et["prp"]:
        print("Preprocessing subject {}".format(sub))
        prop_bad, drop_logs = preprocessing(sub, parameters_file, session="1", task=task)
        preprocessing_summary[sub]["proportion_bad"] = np.mean(prop_bad)
        preprocessing_summary[sub]["drop_logs"] = [item[0] if len(item) > 0 else '' for item in drop_logs]
    preprocessing_summary = format_summary_table(preprocessing_summary)
    # Save the data frame:
    save_dir = Path(ev.bids_root, "derivatives", "preprocessing")
    preprocessing_summary.to_csv(Path(save_dir, "participants_auditory.csv"))

    # ==================================================================================
    # Visual practice preprocessing:
    task = "visual"
    preprocessing_summary = {subject: {"drop_logs": None, "proportion_bad": None}
                             for subject in ev.subjects_lists_et["prp"]}
    for sub in ev.subjects_lists_et["prp"]:
        print("Preprocessing subject {}".format(sub))
        prop_bad, drop_logs = preprocessing(sub, parameters_file, session="1", task=task)
        preprocessing_summary[sub]["proportion_bad"] = np.mean(prop_bad)
        preprocessing_summary[sub]["drop_logs"] = [item[0] if len(item) > 0 else '' for item in drop_logs]
    preprocessing_summary = format_summary_table(preprocessing_summary)
    # Save the data frame:
    save_dir = Path(ev.bids_root, "derivatives", "preprocessing")
    preprocessing_summary.to_csv(Path(save_dir, "participants_visual.csv"))

    # ==================================================================================
    # PRP preprocessing:
    task = "prp"
    preprocessing_summary = {subject: {"drop_logs": None, "proportion_bad": None}
                             for subject in ev.subjects_lists_et[task]}
    for sub in ev.subjects_lists_et[task]:
        print("Preprocessing subject {}".format(sub))
        prop_bad, drop_logs = preprocessing(sub, parameters_file, session="1", task="prp")
        preprocessing_summary[sub]["proportion_bad"] = np.mean(prop_bad)
        preprocessing_summary[sub]["drop_logs"] = [item[0] if len(item) > 0 else '' for item in drop_logs]
    preprocessing_summary = format_summary_table(preprocessing_summary)
    # Save the data frame:
    save_dir = Path(ev.bids_root, "derivatives", "preprocessing")
    preprocessing_summary.to_csv(Path(save_dir, "participants_prp.csv"))

    # ==================================================================================
    # Introspection preprocessing:
    task = "introspection"
    # Session 2:
    preprocessing_summary = {subject: {"drop_logs": None, "proportion_bad": None}
                             for subject in ev.subjects_lists_et[task]}
    for sub in ev.subjects_lists_et[task]:
        print("Preprocessing subject {}".format(sub))
        prop_bad, drop_logs = preprocessing(sub, parameters_file, session="2", task="introspection")
        preprocessing_summary[sub]["proportion_bad"] = np.mean(prop_bad)
        preprocessing_summary[sub]["drop_logs"] = [item[0] if len(item) > 0 else '' for item in drop_logs]
    preprocessing_summary = format_summary_table(preprocessing_summary)
    # Save the data frame:
    save_dir = Path(ev.bids_root, "derivatives", "preprocessing")
    preprocessing_summary.to_csv(Path(save_dir, "participants_introspection_ses-2.csv"))

    # Session 3:
    preprocessing_summary = {subject: {"drop_logs": None, "proportion_bad": None}
                             for subject in ev.subjects_lists_et[task]}
    for sub in ev.subjects_lists_et[task]:
        print("Preprocessing subject {}".format(sub))
        prop_bad, drop_logs = preprocessing(sub, parameters_file, session="3", task="introspection")
        preprocessing_summary[sub]["proportion_bad"] = np.mean(prop_bad)
        preprocessing_summary[sub]["drop_logs"] = [item[0] if len(item) > 0 else '' for item in drop_logs]
    preprocessing_summary = format_summary_table(preprocessing_summary)
    # Save the data frame:
    save_dir = Path(ev.bids_root, "derivatives", "preprocessing")
    preprocessing_summary.to_csv(Path(save_dir, "participants_introspection_ses-3.csv"))
