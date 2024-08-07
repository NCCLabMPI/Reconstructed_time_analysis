import os
import re
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from math import atan2, degrees
from scipy.stats import zscore
from helper_function.based_noise_blinks_detection import based_noise_blinks_detection
from helper_function.helper_general import beh_exclusion

show_checks = False

DFT_SCREEN_DIST_MM = 800

DFT_SCREEN_SIZE_MM = [345,  195]

COG_TRIGGERS = {
    "1": "face_01",
    "2": "face_02",
    "3": "face_03",
    "4": "face_04",
    "5": "face_05",
    "6": "face_06",
    "7": "face_07",
    "8": "face_08",
    "9": "face_09",
    "10": "face_10",
    "11": "face_11",
    "12": "face_12",
    "13": "face_13",
    "14": "face_14",
    "15": "face_15",
    "16": "face_16",
    "17": "face_17",
    "18": "face_18",
    "19": "face_19",
    "20": "face_20",
    "21": "object_01",
    "22": "object_02",
    "23": "object_03",
    "24": "object_04",
    "25": "object_05",
    "26": "object_06",
    "27": "object_07",
    "28": "object_08",
    "29": "object_09",
    "30": "object_10",
    "31": "object_11",
    "32": "object_12",
    "33": "object_13",
    "34": "object_14",
    "35": "object_15",
    "36": "object_16",
    "37": "object_17",
    "38": "object_18",
    "39": "object_19",
    "40": "object_20",
    "41": "letter_01",
    "42": "letter_02",
    "43": "letter_03",
    "44": "letter_04",
    "45": "letter_05",
    "46": "letter_06",
    "47": "letter_07",
    "48": "letter_08",
    "49": "letter_09",
    "50": "letter_10",
    "51": "letter_11",
    "52": "letter_12",
    "53": "letter_13",
    "54": "letter_14",
    "55": "letter_15",
    "56": "letter_16",
    "57": "letter_17",
    "58": "letter_18",
    "59": "letter_19",
    "60": "letter_20",
    "61": "letter_01",
    "62": "letter_02",
    "63": "letter_03",
    "64": "letter_04",
    "65": "letter_05",
    "66": "letter_06",
    "67": "letter_07",
    "68": "letter_08",
    "69": "letter_09",
    "70": "letter_10",
    "71": "letter_11",
    "72": "letter_12",
    "73": "letter_13",
    "74": "letter_14",
    "75": "letter_15",
    "76": "letter_16",
    "77": "letter_17",
    "78": "letter_18",
    "79": "letter_19",
    "80": "letter_20",
    "101": "center",
    "102": "left",
    "103": "right",
    "151": "500ms",
    "152": "1000ms",
    "153": "1500ms",
    "201": "target",
    "202": "non-target",
    "203": "irrelevant"
}


def convert_cog_trig(raw):
    """
    Converts triggers from the COGITATE project to a more user friendly format. During acquisition, triggers
    were sent successively, as we had too many experimental conditions to fit in 8 bits. Instead, we sent successive
    triggers spaced by 50ms to identify all the experimental conditions associated with a trigger. This function
    converts these successive triggers back into a single one at stimulus onset, with strings describing the
    conditions (forward slash separated list)

    Parameters:
    raw (mne.io.Raw): The raw EEG data object containing event annotations.

    Returns:
    mne.io.Raw: The raw EEG data object with updated event descriptions.

    Raises:
    Exception: If the number of triggers in a trial is not exactly four.
    """

    # Extract the event descriptions from the raw data annotations
    evt_dsc = np.array(raw.annotations.description.copy(), dtype="object")

    # Find all the stimulus onset events
    idxs = []
    for val in [str(i + 1) for i in range(80)]:
        idx = np.where(evt_dsc == val)[0]
        if len(idx) > 0:
            idxs.append(idx)

    # Flatten and sort the array of indices
    idxs = np.sort(np.concatenate(idxs, axis=0))

    # Loop through each index to process trial events
    for i, idx in enumerate(idxs):
        # Determine the range of triggers for the current trial
        if i == len(idxs) - 1:
            trial_trig = evt_dsc[idx:]
        else:
            trial_trig = evt_dsc[idx:idxs[i + 1]]

        # Filter for valid triggers based on predefined mapping
        correct_triggers = [trig for trig in trial_trig if trig in COG_TRIGGERS.keys()]

        # Ensure the trial contains exactly four triggers
        if len(correct_triggers) != 4:
            raise Exception("Something went wrong with the triggers!!!")

        # Construct the stimulus identifier list
        stim_id = ["vis_onset",
                   COG_TRIGGERS[correct_triggers[0]].split("_")[0],
                   COG_TRIGGERS[correct_triggers[0]]]
        stim_id.extend([COG_TRIGGERS[trig] for trig in correct_triggers[1:]])
        stim_id = "/".join(stim_id)
        # Update the description of the event with the constructed identifier
        evt_dsc[idx] = stim_id
    raw.annotations.description = evt_dsc

    return raw


def format_summary_table(summary_dict):
    """
    This function converts the summary dictionary into a data frame
    :param summary_dict: (dict) format:
                    {"sub-id":
                        {
                            "drop_logs": list,
                            "proportion_bad": float
                        }
                    }
    :return: df
    """
    # Extract from the drop logs all the different reasons to drop an epoch:
    trial_rej_reas = [list(set(summary_dict[sub]["drop_logs"])) for sub in summary_dict.keys()]
    trial_rej_reas = list(set([item for items in trial_rej_reas for item in items if len(item) > 0]))
    preprocessing_summary_df = []
    # Loop through each subject:
    for sub in summary_dict.keys():
        # Extract this subject drop logs:
        sub_drops = summary_dict[sub]["drop_logs"]
        sub_dict = {
            "subject": sub,
            "proportion_bad": summary_dict[sub]["proportion_bad"],
            **{reas: None for reas in trial_rej_reas}
        }
        # Loop through each reason to get the counts:
        for reas in trial_rej_reas:
            sub_dict[reas] = np.sum(np.array(sub_drops) == reas) / len(sub_drops)
        preprocessing_summary_df.append(pd.DataFrame(sub_dict, index=[0]))
    return pd.concat(preprocessing_summary_df).reset_index(drop=True)


def reject_bad_epochs(epochs, baseline_window=None, z_thresh=2, eyes=None, exlude_beh=True):
    """
    This function rejects epochs based on the zscore of the baseline. For some trials, there may be artifacts
    in the baseline, in which case baseline correction will spread the artifact. Such epochs are discarded.
    :param epochs:
    :param baseline_window:
    :param z_thresh:
    :param eyes:
    :param exlude_beh:
    return:
        - epochs:
        - n rejected trials:
    """
    print("=" * 40)
    print("Rejecting bad epochs")
    if eyes is None:
        eyes = ["left", "right"]
    if baseline_window is None:
        baseline_window = [None, 0]
    # Get the initial number of trials:
    ntrials_orig = len(epochs)

    if exlude_beh:
        print("     Applying behavioral exclusion criterion")
        inds = beh_exclusion(epochs.metadata.copy().reset_index(drop=True))
        if len(inds) > 0:
            # Drop these epochs:
            epochs.drop(inds, reason="bad_beh", verbose="ERROR")
        print("{} out of {} ({:.2f}%) trials were rejected based on behavior.".format(len(inds), ntrials_orig,
                                                                                      (len(inds) / ntrials_orig) * 100))
    # Extract the data:
    if z_thresh is not None:
        print("     Rejecting trials with artifactual baseline: ")
        baseline_data = epochs.copy().crop(tmin=baseline_window[0],
                                           tmax=baseline_window[1]).get_data(picks=["_".join(["pupil", eye])
                                                                                    for eye in eyes])
        # Compute the average across eyes and time:
        baseline_avg = np.mean(np.mean(baseline_data, axis=1), axis=1)
        # Z score:
        baseline_zscore = zscore(baseline_avg, nan_policy='omit')
        # Find the epochs indices that exceed the threshold:
        inds = np.where(np.abs(baseline_zscore) > z_thresh)[0]
        if len(inds) > 0:
            # Drop these epochs:
            epochs.drop(inds, reason="baseline_artifact", verbose="ERROR")
        # Print the proportion of dropped epochs:
        print("{} out of {} ({:.2f}%) trials had artifact in baseline.".format(len(inds), len(baseline_zscore),
                                                                               (len(inds) / len(
                                                                                   baseline_zscore)) * 100))
    ntrials_final = len(epochs)

    return epochs, 1 - ntrials_final / ntrials_orig


def annotate_nan(raw, nan_annotation="BAD_nan", eyes=None):
    """
    This function identitfies nan samples in the data and annotate them as BAD samples for later discarding.
    """
    if eyes is None:
        eyes = ["left", "right"]
    # Loop through each eye:
    for eye in eyes:
        # Extract the data:
        data = np.squeeze(raw.copy().pick("pupil_{}".format(eye)).get_data())
        # Get the nan:
        nan_mask = np.isnan(data).astype(float)
        # Compute the diff to get the onset and offsets:
        nan_diff = np.diff(nan_mask)
        # Extract the onsets and offsets:
        onsets = raw.times[np.where(nan_diff == 1)[0]]
        offsets = raw.times[np.where(nan_diff == -1)[0]]
        # Compute duration:
        if onsets.shape[0] > 0:
            if onsets.shape[0] != offsets.shape[0]:
                if onsets.shape[0] == offsets.shape[0] + 1:
                    offsets = raw.times[-1]

                else:
                    raise Exception("Different numbers of onsets and offsets!")
            duration = offsets - onsets
            # Create annotations accordingly:
            nan_annotations = mne.Annotations(
                onset=onsets,
                duration=duration,
                description=[nan_annotation] * len(onsets),
                ch_names=[('xpos_' + eye, 'ypos_' + eye, 'pupil_' + eye)] * len(onsets),
                orig_time=raw.annotations.orig_time
            )
            # Combine the annotations with the raw:
            raw.set_annotations(raw.annotations + nan_annotations)

    # Return the raw:
    return raw


def load_cog_eyetracker(bids_root, subject, session, task, verbose=False, debug=False):
    """
    This functions loads the eyetracking data using mne python function. In addition, it loads the log files from the
    raw root to extract additional information. For a few subjects, some triggers weren't received by the eyetracker
    and therefore, the logs need to be aligned back to the eyetracker trigger to avoid any mismatch, which is what this
    function does. In addition, loading the calibration results.
    :param bids_root: (Path or string) path to the eyetracking files
    :param subject: (string) name of the subject
    :param session: (string) session
    :param task: (string) task
    :param beh_file_name: (string) template string  for the name of the behavioral log files
    :param annotations_col_names: (list of strings) name to give the columns of the annotation table
    :param event_of_interest: (string) identifier of the events of interest
    :param verbose: (bool) verbose
    :para debug: (bool) debug mode loads only 2 files
    :return:
    """
    # Load all the files:
    raws = []
    calibs = []
    screen_sizes = []
    screen_ress = []
    screen_distances = []
    ctr = 0

    # Create the files roots:
    et_root = Path(bids_root, "sub-" + subject, "ses-" + session, "eyetrack")

    # ===================================================================================
    # Load the eyetracking data and ensure that the log files match:
    for fl in os.listdir(et_root):
        if debug and ctr > 5:  # Load only a subpart of the files for the debugging
            continue
        if fl.endswith('.asc') and fl.split("_task-")[1].split("_eyetrack.asc")[0] == task:
            if verbose:
                print("Loading: " + fl)

            # Extract the run ID:
            run_i = int(re.search(r'run-(\d{2})', fl).group(1))

            try:
                raw = mne.io.read_raw_eyelink(Path(et_root, fl), verbose=verbose)
            except ValueError:
                print("The data in sub-{}, ses-{}, task-{} and run-{} are unreadable".format(subject,
                                                                                             session,
                                                                                             task,
                                                                                             run_i))
                continue

            # Add the triggers:
            raw = convert_cog_trig(raw)
            raws.append(raw)
            calib, screen_dist, screen_size, screen_res = read_calib(Path(et_root, fl))
            calibs.append(calib)
            screen_distances.append(screen_dist)
            screen_sizes.append(screen_size)
            screen_ress.append(screen_res)
            ctr += 1
    # Check that the screen sizes, distances and resolutions are always the same:
    assert len(np.unique(screen_sizes)) == 2, "Found different screen sizes within the same participant!"
    assert len(np.unique(screen_distances)) == 1, "Found different screen distances within the same participant!"
    assert len(np.unique(screen_ress)) == 3, "Found different screen resolutions within the same participant!"
    screen_size = np.unique(screen_sizes)
    screen_distance = np.unique(screen_distances)[0]
    screen_res = np.unique(screen_ress)[1:3] + 1
    return raws, calibs, screen_size, screen_distance, screen_res


def load_raw_eyetracker(bids_root, subject, session, task, beh_file_name,
                        annotations_col_names, event_of_interest, verbose=False, debug=False):
    """
    This functions loads the eyetracking data using mne python function. In addition, it loads the log files from the
    raw root to extract additional information. For a few subjects, some triggers weren't received by the eyetracker
    and therefore, the logs need to be aligned back to the eyetracker trigger to avoid any mismatch, which is what this
    function does. In addition, loading the calibration results.
    :param bids_root: (Path or string) path to the eyetracking files
    :param subject: (string) name of the subject
    :param session: (string) session
    :param task: (string) task
    :param beh_file_name: (string) template string  for the name of the behavioral log files
    :param annotations_col_names: (list of strings) name to give the columns of the annotation table
    :param event_of_interest: (string) identifier of the events of interest
    :param verbose: (bool) verbose
    :para debug: (bool) debug mode loads only 2 files
    :return:
    """
    # Load all the files:
    raws = []
    calibs = []
    logs = []
    screen_sizes = []
    screen_ress = []
    screen_distances = []
    ctr = 0

    # Create the files roots:
    et_root = Path(bids_root, "sub-" + subject, "ses-" + session, "eyetrack")
    beh_root = Path(bids_root, "sub-" + subject, "ses-" + session, "beh")
    # ===================================================================================
    # Load the behavioral log files:
    log_file = pd.read_csv(Path(beh_root, beh_file_name.format(subject, session, "all", task)))

    # ===================================================================================
    # Load the eyetracking data and ensure that the log files match:
    for fl in os.listdir(et_root):
        if debug and ctr > 5:  # Load only a subpart of the files for the debugging
            continue
        if fl.endswith('.asc') and fl.split("_task-")[1].split("_eyetrack.asc")[0] == task:
            if verbose:
                print("Loading: " + fl)

            # Extract the run ID:
            run_i = int(re.search(r'run-(\d{2})', fl).group(1))
            # Extract the corresponding part of the log file:
            if log_file["is_practice"].to_list()[0] == 1:
                run_log = log_file
            else:
                run_log = log_file[log_file["block"] == run_i].reset_index(drop=True)

            try:
                raw = mne.io.read_raw_eyelink(Path(et_root, fl), verbose=verbose)
            except ValueError:
                print("The data in sub-{}, ses-{}, task-{} and run-{} are unreadable".format(subject,
                                                                                             session,
                                                                                             task,
                                                                                             run_i))
                continue

            # Extract the events of interest from the raw annotations:
            evt = [desc.split("/") for desc in raw.annotations.description[
                np.where([event_of_interest in val for val in raw.annotations.description])[0]]]
            # Convert the annotations to a pandas dataframe:
            annotations_df = pd.DataFrame(evt, columns=annotations_col_names)

            # Compare the log files to the annotations to identify and  address any discrepancies:
            if run_log.shape[0] == annotations_df.shape[0]:
                # If the two data frames have the same length, compare the identity column in each respectively.
                # Identity changes on a trial by trial basis:
                assert all(annotations_df["identity"] == run_log["identity"]), \
                    "The events logged in the logs do not match the events in the eyetracker triggers!"
            elif run_log.shape[0] > annotations_df.shape[0]:
                n_miss_triggers = run_log.shape[0] - annotations_df.shape[0]
                start_index = -1
                identities_log_file = run_log["identity"].to_list()
                identities_et = annotations_df["identity"].to_list()
                for i in range(len(identities_log_file) - len(identities_et) + 1):
                    if identities_log_file[i:i + len(identities_et)] == identities_et:
                        start_index = i
                        break
                if start_index >= 0:
                    run_log = run_log.iloc[start_index:start_index + len(identities_et)]
                    print("WARNING: There were {} missing triggers in run-{} of task-{} in ses-{}!".format(
                        n_miss_triggers, run_i, task, session))
                else:
                    raise Exception("The events in the log file do not match the events in the Eyetracking triggers!!!")
            else:
                raise Exception("More triggers than there were events in the log file!!!")
            logs.append(run_log)
            raws.append(raw)
            calib, screen_dist, screen_size, screen_res = read_calib(Path(et_root, fl))
            calibs.append(calib)
            screen_distances.append(screen_dist)
            screen_sizes.append(screen_size)
            screen_ress.append(screen_res)
            ctr += 1
    # Check that the screen sizes, distances and resolutions are always the same:
    assert len(np.unique(screen_sizes)) == 2, "Found different screen sizes within the same participant!"
    assert len(np.unique(screen_distances)) == 1, "Found different screen distances within the same participant!"
    assert len(np.unique(screen_ress)) == 3, "Found different screen resolutions within the same participant!"
    screen_size = np.unique(screen_sizes)
    screen_distance = np.unique(screen_distances)[0]
    screen_res = np.unique(screen_ress)[1:3] + 1
    return logs, raws, calibs, screen_size, screen_distance, screen_res


def read_calib(fname):
    """
    This functions reads the eyelink calib using the mne function read_eyelink_calibration. What is added is the reading
    of the screen physical settings directly from the ascii file.
    :param fname: (string or path object) path to the ascii file that contains the calib
    :return: (list mne calib object or empty list) eyelink calib. In case no calib is found in the file, return
    empty list.
    """
    # Open file
    f = open(fname, 'r')
    fl_txt = f.read().splitlines(True)  # split into lines
    fl_txt = list(filter(None, fl_txt))  # remove emptys
    # Extract screen distance:
    try:
        screen_distance = [txt.strip("\n") for txt in fl_txt if "Screen_distance_mm" in txt][0].split(":")[-1].split(" ")
        # Convert to float:
        screen_distance = np.mean([float(val) for val in screen_distance if val.isdigit()])
    except IndexError:
        print("WARNING: No screen distance value found in the headers, using the default instead!")
        print(f"Default: {DFT_SCREEN_DIST_MM}mm")
        screen_distance = DFT_SCREEN_DIST_MM
    # Extract screen size:
    try:
        screen_size = [txt.strip("\n") for txt in fl_txt if "Screen_size_mm" in txt][0].split(":")[-1].split(" ")
        # Convert to float:
        screen_size = [float(val) * 10 for val in screen_size if val.isdigit()]
    except IndexError:
        print("WARNING: No screen size value found in the headers, using the default instead!")
        print(f"Default: {DFT_SCREEN_SIZE_MM}mm")
        screen_size = DFT_SCREEN_SIZE_MM
    # Extract screen res:
    screen_res = [txt.strip("\n") for txt in fl_txt if "GAZE_COORDS" in txt][0].split("GAZE_COORDS")[-1].split(" ")
    # Convert to float:
    screen_res = [float(val) for val in screen_res if val.replace(".", "").isdigit()]
    # Read in the calib:
    calib = mne.preprocessing.eyetracking.read_eyelink_calibration(fname,
                                                                   screen_size=[screen_size[1], screen_size[0]],
                                                                   screen_distance=screen_distance,
                                                                   screen_resolution=[screen_res[2], screen_res[3]])
    if len(calib) > 2:
        print("A")
    return calib, screen_distance, screen_size, screen_res


def plot_blinks(raw, blinks_annotations=None):
    # Create a copy of the raw object:
    if blinks_annotations is None:
        blinks_annotations = ["BAD_blink"]
    raw_copy = raw.copy()

    # Keep only the blinks annotations:
    raw_copy.annotations.delete(np.where(~np.isin(raw_copy.annotations.description, blinks_annotations))[0])

    # Plot:
    raw_copy.plot(scalings=dict(eyegaze=1e3), block=True)

    return None


def hershman_blinks_detection(raw, eyes=None, replace_eyelink_blinks=True):
    """
    This function applies the blink detection algorithm described here: https://osf.io/jyz43/?view_only= to mne raw
    object
    :param raw:
    :param eyes: eye to use for blink detection. If both are used...
    :param replace_eyelink_blinks:
    :return:
    """
    print("=" * 40)
    print("Applying hershman algorithm for blinks detection!")
    if eyes is None:
        eyes = ["left", "right"]
    # Copy annotations:
    new_annotations = raw.annotations.copy()
    # Remove the eyelink blinks:
    if replace_eyelink_blinks:
        new_annotations.delete(np.where(raw.annotations.description == "BAD_blink")[0])
    # Loop through each eye:
    for eye in eyes:
        # Extract the pupil data of this eye:
        data = np.squeeze(raw.copy().get_data(picks='pupil_' + eye))
        # Indices from the eyelink:
        blinks_inds = np.where(raw.annotations.description == "BAD_blink")[0]
        # Get the blinks time stamps:
        blink_onsets = raw.annotations.onset[blinks_inds]
        blink_durations = raw.annotations.duration[blinks_inds]
        # Set the blinks periods to 0:
        for i, blink_onset in enumerate(blink_onsets):
            # Get the start and end indices:
            start = np.argmin(np.abs(raw.times - blink_onset))
            end = np.argmin(np.abs(raw.times - (blink_onset + blink_durations[i])))
            data[start:end] = 0
        # Replace the nans by 0s:
        data[np.isnan(data)] = 0
        # Apply the hershman algorithm:
        blinks = based_noise_blinks_detection(data, int(raw.info["sfreq"]))

        # Create annotations accordingly:
        blinks_annotations = mne.Annotations(
            onset=blinks["blink_onset"] * 1 / 1000,
            duration=(blinks["blink_offset"] - blinks["blink_onset"]) * 1 / 1000,
            description=["BAD_blink"] * len(blinks["blink_onset"]),
            ch_names=[('xpos_' + eye, 'ypos_' + eye, 'pupil_' + eye)] * len(blinks["blink_onset"]),
            orig_time=raw.annotations.orig_time
        )
        # Add the newly detected blinks:
        new_annotations = new_annotations.__add__(blinks_annotations)
    # Add the new annotations:
    raw.set_annotations(new_annotations)

    return raw


def pix_to_deg(x_pix, y_pix, screen_size_mm, screen_res, screen_dist_mm):
    """
    This function converts the gaze coordinates to degree of visual angle from center of the screen.
    :param x_pix:
    :param y_pix:
    :param screen_size_mm:
    :param screen_dist_mm:
    :return:
    """
    # Calculate deg/px:
    deg_per_pixel = np.mean([degrees(atan2(.5 * screen_size_mm[0], screen_dist_mm)) / (.5 * screen_res[0]),
                             degrees(atan2(.5 * screen_size_mm[1], screen_dist_mm)) / (.5 * screen_res[1])])
    # Convert x and y coordinates from pixels to cm:
    x_deg, y_deg = ((x_pix - screen_res[1] / 2) * deg_per_pixel,
                    (y_pix - screen_res[0] / 2) * deg_per_pixel)
    # Convert to distance from the center of the screen:
    distance_deg = np.sqrt(x_deg ** 2 + y_deg ** 2)
    return distance_deg


def gaze_to_dva(raw, screen_size_mm, screen_res, screen_dist_mm, eyes=None):
    """
    This function converts the gaze measurements from pixel coordinates to degrees of visual angle (dva) from the
    middle of the screen.
    :param raw: (mne raw object) contains the gaze position in x and y pixel coordinates
    :param screen_size_mm: (list) contains the screen size in cm [width, height]
    :param screen_dist_mm: (float) contains the screen distance in cm
    :param eyes: (list of string or None) eyes for which to apply the conversion
    :return:
        - raw: (mne raw object) with the added fixdist_{eye} channel
    """
    print("=" * 40)
    print("Converting the gaze coordinates in distance from the middle of the screen in degrees of visual angle")

    if eyes is None:
        eyes = ["left", "right"]
    # Loop through each eye:
    for eye in eyes:
        # Extract the gaze data of this eye:
        eye_x, eye_y = (np.squeeze(raw.get_data(picks=["xpos_{}".format(eye)])),
                        np.squeeze(raw.get_data(picks=["ypos_{}".format(eye)])))
        # Convert to dva:
        fixation_dist = pix_to_deg(eye_x, eye_y, screen_size_mm, screen_res, screen_dist_mm)
        # Create a raw object for this channel:
        info = mne.create_info(ch_names=["_".join(["fixdist", eye])],
                               ch_types=['eyegaze'],
                               sfreq=raw.info["sfreq"])
        # Add measurement date:
        info.set_meas_date(raw.info['meas_date'])
        # Combine to a mne raw object:
        raw_fix = mne.io.RawArray(np.expand_dims(fixation_dist, axis=0), info, verbose="WARNING")
        # Add channel to the raw object:
        raw.add_channels([raw_fix])
    return raw


def extract_eyelink_events(raw, description="blink", eyes=None):
    """
    This function extracts the eyelink events from the annotation. In the annotation, we have the onset and duration
    for each of the eyelink parser events. These are converted to continuous regressors, with ones where we have the
    event in question and zeros elsewhere. This is for handy processing down the line, where we can reject or regress
    those out.
    :param raw: (mne raw object) raw object containing the annotations and continuous recordings
    :param description: (string) identifier for the event in question (blink, saccades, fixations...). The description
    must match the description found in the raw object annotation
    :param eyes: (list or None) eye to use. By default, set to use both, which will create one channel per eye and per
    event. MONOCULAR NOT IMPLEMENTED
    :return: raw_new (mne raw object) raw object with the added channels encoding the events and their duration
    """
    # Create the new channels, one per eye:
    if eyes is None:
        eyes = ["left", "right"]

    desc_vectors = []
    for eye in eyes:
        # Extract the events
        evts_ind = np.intersect1d(np.where(raw.annotations.description == description)[0],
                                  np.array([ind for ind in range(len(raw.annotations.ch_names))
                                            if len(raw.annotations.ch_names[ind]) > 0
                                            if eye in raw.annotations.ch_names[ind][0]]))
        # Extract the onset and duration of the said event:
        evt_onset = raw.annotations.onset[evts_ind]
        evt_offset = evt_onset + raw.annotations.duration[evts_ind]
        # Convert to samples:
        onset = (evt_onset * raw.info["sfreq"]).astype(int)
        offset = (evt_offset * raw.info["sfreq"]).astype(int)

        # Set the regressor to 1 where the event is happening:
        desc_vector = np.zeros(raw.n_times)
        for i in range(len(onset)):
            desc_vector[onset[i]:offset[i]] = 1
        desc_vectors.append(desc_vector)

    # Add these two channels to the raw data:
    evts_info = mne.create_info(["_".join([description, eye]) for eye in eyes],
                                ch_types=['misc'] * len(eyes),
                                sfreq=raw.info["sfreq"])
    evts_info.set_meas_date(raw.info['meas_date'])
    raw_evts = mne.io.RawArray(np.array(desc_vectors), evts_info, verbose="WARNING")
    raw.add_channels([raw_evts])
    return raw


def add_logfiles_info(epochs, log_file, columns):
    """
    This function is highly specific to the PRP experiment from Micha and Alex, combining the information found in the
    behavioral log file with the epochs data, enabling the use of the same exclusion criterion and so on.
    :param epochs: (mne epochs object) object containing the metadata
    :param log_file: (pd dataframe) log file containing the additional info
    :param columns: (list of strings) name of the columns from the log file to transfer to the metadata
    :return:
    """
    # Check if some epochs were dropped to remove them from the log file too:
    log_file = log_file[[True if len(val) == 0 else False for val in epochs.drop_log]]
    # Make sure the logs and the metadata have the same length:
    assert log_file.shape[0] == epochs.metadata.shape[0], \
        "The log file and the epochs metadata have different number of events!"
    for col in columns:
        # Now, add the relevant info to the metadata:
        if col == "RT_aud":
            epochs.metadata.loc[:, col] = log_file["time_of_resp_aud"].to_numpy() - log_file["aud_stim_time"].to_numpy()
        elif col == "RT_vis":
            epochs.metadata.loc[:, col] = log_file["time_of_resp_vis"].to_numpy() - log_file["vis_stim_time"].to_numpy()
        else:
            epochs.metadata.loc[:, col] = log_file[col]
    return epochs


def compute_proportion_bad(raw, desc="BAD_", eyes=None):
    """
    This function computes the proportion of data that are marked as bad according to the specified description. The
    proportion of data affected by the annotation is returned per eye for binocular recordings.
    :param raw: (mne raw object) contains the eyelink data
    :param desc: (string) string identifier to compute proportion of affected data
    :param eyes: (list) eyes to investigate
    :return: (list) proportion of affected data for each eye
    """
    if eyes is None:
        eyes = ["left", "right"]
    bad_proportions = []
    print("=" * 40)
    print("Proportion of the data marked as {}".format(desc))
    # Loop through each eye
    for eye in eyes:
        # Extract the indices of this eye matching the description:
        bad_annot_ind = np.intersect1d([ind for ind, val in enumerate(raw.annotations.description) if desc in val],
                                       np.array([ind for ind in range(len(raw.annotations.ch_names))
                                                 if len(raw.annotations.ch_names[ind]) > 0
                                                 if eye in raw.annotations.ch_names[ind][0]])
                                       )
        # Compute the sum of the duration:
        if len(bad_annot_ind) > 0:
            bad_dur = np.sum(raw.annotations.duration[bad_annot_ind])
        else:
            bad_dur = 0

        # Compute the proportion:
        bad_proportion = bad_dur / (raw.times[-1] - raw.times[0])
        print("{} eye:       {:2f}%".format(eye, bad_proportion * 100))
        bad_proportions.append(bad_dur / (raw.times[-1] - raw.times[0]))
    return bad_proportions


def create_metadata_from_events(epochs, metadata_column):
    """
    This function parses the events found in the epochs descriptions to create the meta data. The column of the meta
    data are generated based on the metadata column names. The column name must be a list in the same order as the
    strings describing the events. The name of the column must be the name of the overall condition, so say the
    specific column describes the category of the presented stim (faces, objects...), then the column should be called
    category. This will become obsolete here at some point, when the preprocessing is changed to generate the meta data
    directly
    :param epochs: (mne epochs object) epochs for which the meta data will be generated
    :param metadata_column: (list of strings) name of the column of the meta data. Must be in the same order
    as the events description + must be of the same length as the number of word in the events description
    :return: epochs (mne epochs object)
    """

    # Getting the event description of each single trial
    trials_descriptions = [[key for key in epochs.event_id.keys() if epochs.event_id[key] == event]
                           for event in epochs.events[:, 2]]
    trial_descriptions_parsed = [description[0].split(
        "/") for description in trials_descriptions]
    # Making sure that the dimensions of the trials description is consistent across all trials:
    if len(set([len(vals) for vals in trial_descriptions_parsed])) > 1:
        raise ValueError('dimension mismatch in event description!\nThe forward slash separated list found in the '
                         'epochs description has inconsistent length when parsed. Having different number of '
                         'descriptors for different trials is not yet supported. Please make sure that your events '
                         'description are set accordingly')
    if len(metadata_column) != len(trial_descriptions_parsed[0]):
        raise ValueError("The number of meta data columns you have passed doesn't match the number of descriptors for\n"
                         "each trials. Make sure you have matching numbers. In doubt, go and check the events file in\n"
                         "the BIDS directory")
    if len(trial_descriptions_parsed) != len(epochs):
        raise ValueError("Somehow, the number of trials descriptions found in the epochs object doesn't match the "
                         "number of trials in the same epochs. I have no idea how you managed that one champion, so I "
                         "can't really help here")

    # Convert the trials description to a pandas dataframe:
    epochs.metadata = pd.DataFrame.from_records(
        trial_descriptions_parsed, columns=metadata_column)

    return epochs


def epoch_data(raw, events, event_dict, events_of_interest=None, metadata_column=None, tmin=-0.5, tmax=2.0,
               baseline=None, picks="all", reject_by_annotation=False):
    """
    This function epochs the continuous data according to specified events of interest, i.e. not all the events get
    evoked, only those we are interested in!
    :param raw: (mne raw object) contains the data to epochs
    :param events: (array of int) ID of each event
    :param event_dict: (dictionary) description for each event UID
    :param events_of_interest: (list of strings) list of events that we wish to epochs. The name must match strings
    found in the event_dict keys
    :param metadata_column: (list of strings) name of the meta data table columns. The event descriptions must be
    encoded as backward slash separated values. Each string in the event dict key corresponds to a specific parameter
    from the experiment. These are then parsed as a meta data table accordingly
    :param tmin: (float) time from which to epoch (relative to event onset)
    :param tmax: (float) time until which to epoch (relative to event onset)
    :param baseline: (None or tuple) time to use as baseline. If set to None, no baseline correction applied
    :param picks: (list or "all") list of channels to epoch
    :param reject_by_annotation: (boolean) whether or not to reject trials based on annotations
    :return: mne epochs object: the epoched data
    """
    # First, extract the events of interest:
    if events_of_interest is not None:
        select_event_dict = {key: event_dict[key] for key in event_dict if any(substring in key
                                                                               for substring in events_of_interest)}
    else:
        select_event_dict = event_dict
    # Epochs the data accordingly:
    epochs = mne.Epochs(raw, events=events, event_id=select_event_dict, tmin=tmin,
                        tmax=tmax, baseline=baseline, verbose='ERROR', picks=picks,
                        reject_by_annotation=reject_by_annotation)
    # Reject trials whose end falls outside the recording:
    tend = raw.times[-1]
    # Find trials whose end is outside the recording:
    bad_trials = np.where((epochs.events[:, 0] / raw.info["sfreq"]) + tmax > tend)[0]
    # Drop those trials:
    epochs.drop(bad_trials, reason="outside_range")
    # Dropping the bad epochs if there were any:
    epochs.drop_bad()
    # Adding the meta data to the table. The meta data are created by parsing the events strings, as each substring
    # contains specific info about the trial:
    if metadata_column is not None:
        epochs = create_metadata_from_events(epochs, metadata_column)
    return epochs
