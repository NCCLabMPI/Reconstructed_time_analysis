import mne
import numpy as np
from pathlib import Path
import pandas as pd
import os
import subprocess
import warnings

# Eyelink logs' line tags
OTHER = "OTHER"
EMPTY = "EMPTY"
COMMENT = 'COMMENT'
SAMPLE = "SAMPLE"
START = "START"
END = "END"
MSG = "MSG"
EFIX = "EFIX"
ESACC = 'ESACC'
EBLINK = "EBLINK"
START_REC_MARKER = '!MODE RECORD'

DF_REC = "dfRec"
DF_MSG = "dfMsg"
DF_FIXAT = "dfFix"
DF_SACC = "dfSacc"
DF_SACC_EK = f"{DF_SACC}EK"
DF_BLINK = "dfBlink"
DF_SAMPLES = "dfSamples"
EYE = "Eye"
EYELINK = "Eyelink"
EK = "EK"
HERSHMAN = "Hershman"
HERSHMAN_PAD = "HershmanPad"
HAS_ET_DATA = "has_ET_data"

# data columns
ONSET = 'stimOnset'
T_START = 'tStart'
T_END = 'tEnd'
T_SAMPLE = 'tSample'
VPEAK = 'vPeak'
AMP_DEG = 'ampDeg'
BLOCK_COL = "block"
MINIBLOCK_COL = "miniBlock"
ORIENTATION_COL = "stimOrientation"
TRIAL_COL = "trial"
RESP_TIME_COL = "responseTime"
STIM_TYPE_COL = "stimType"
STIM_COL = "stimID"
IS_TASK_RELEVANT_COL = "isTaskRelevant"
STIM_DUR_PLND_SEC = "plndStimulusDur"
STIM_DUR_COL = 'stimDur'
REAL_FIX = "RealFix"
REAL_SACC = "RealSacc"
REAL_PUPIL = "RealPupil"
TRIAL_NUMBER = "trialNumber"

# duration windows

STIM_LOC = "stimulusLocation"
PRE_STIM_DUR = 'PreStim'
FIRST_WINDOW = 'First'
SECOND_WINDOW = 'Second'
THIRD_WINDOW = 'Third'
EPOCH = 'Epoch'
TRIAL = "Trial"
BASELINE_START = "Baseline"

LOC = 'Location'
VIS = 'Visibility'
WORLD_G = 'Game World'
WORLD_R = 'Replay World'
WORLD = 'World'
CATEGORY = 'Category'
IS_LEVEL_ELIMINATED = "IS_LEVEL_ELIMINATED"
SACC_DIRECTION_RAD = "sacc_direction_rad"

SAMPLING_FREQ = 'SamplingFrequency'
WINDOW_START = 'WindowStart'
WINDOW_END = 'WindowEnd'


def ascii_parser(ascii_file):
    """

    :param ascii_file:
    :return:
    """
    f = open(ascii_file, 'r')
    fl_txt = f.read().splitlines(True)  # split into lines
    fl_txt = list(filter(None, fl_txt))  # remove emptys
    fl_txt = np.array(fl_txt)  # concert to np array for simpler indexing
    # Separate lines into samples and messages
    print('Sorting lines')
    n_lines = len(fl_txt)
    line_type = np.array([OTHER] * n_lines, dtype='object')
    i_start_rec = list()

    for iLine in range(n_lines):
        if fl_txt[iLine] == "**\n" or fl_txt[iLine] == "\n":
            line_type[iLine] = EMPTY
        elif fl_txt[iLine].startswith('*') or fl_txt[iLine].startswith('>>>>>'):
            line_type[iLine] = COMMENT
        elif bool(len(fl_txt[iLine][0])) and fl_txt[iLine][0].isdigit():
            fl_txt[iLine] = fl_txt[iLine].replace('.\t', 'NaN\t')
            line_type[iLine] = SAMPLE
        else:  # the type of this line is defined by the first string in the line itself (e.g. START, MSG)
            line_type[iLine] = fl_txt[iLine].split()[0]
        if START in fl_txt[iLine]:
            # from EyeLink Programmers Guide: "The "START" line and several following lines mark the start of
            # recording, and encode the recording conditions for the trial."
            i_start_rec.append(iLine + 1)

    i_start_rec = i_start_rec[0]
    # Import Messages
    print('Parsing events: Stimulus events, fixations, saccades and blinks!')
    events_df = pd.DataFrame()
    i_msg = np.nonzero(line_type == MSG)[0]
    # Messages:
    t_msg = []
    txt_msg = []
    for i in range(len(i_msg)):
        # separate MSG prefix and timestamp from rest of message
        info = fl_txt[i_msg[i]].split()
        # extract info
        t_msg.append(int(info[1]))
        txt_msg.append(' '.join(info[2:]))
    events_df = events_df.append(pd.DataFrame({'time': t_msg, 'event': txt_msg}))
    # Extract the eye:
    eyes_in_file = events_df[events_df["event"].str.contains("RECCFG")].iloc[0, 1].split()[-1]
    # Extract the sfreq:
    sfreq = int(events_df[events_df["event"].str.contains("RECCFG")].iloc[0, 1].split()[2])
    # Fixations:
    i_not_efix = np.nonzero(line_type != EFIX)[0]
    df_fix = pd.read_csv(ascii_file, skiprows=i_not_efix, header=None, delim_whitespace=True, usecols=range(1, 8))
    df_fix.columns = ['eye', T_START, T_END, 'duration', 'xAvg', 'yAvg', 'pupilAvg']
    # Saccades:
    i_not_esacc = np.nonzero(line_type != ESACC)[0]
    df_sacc = pd.read_csv(ascii_file, skiprows=i_not_esacc, header=None, delim_whitespace=True, usecols=range(1, 11))
    df_sacc.columns = ['eye', T_START, T_END, 'duration', 'xStart', 'yStart', 'xEnd', 'yEnd', AMP_DEG, VPEAK]
    # Blinks:
    i_not_eblink = np.nonzero(line_type != EBLINK)[0]
    df_blink = pd.read_csv(ascii_file, skiprows=i_not_eblink, header=None, delim_whitespace=True, usecols=range(1, 5))
    df_blink.columns = ['eye', T_START, T_END, 'duration']

    # Combine all these "events" into an annotation table:
    events_df = pd.DataFrame(pd.DataFrame({
        "onset": events_df["time"].to_numpy(),
        "duration": np.array([0] * events_df.shape[0]),
        "description": events_df["event"].to_numpy()
    }))
    df_fix = pd.DataFrame(pd.DataFrame({
        "onset": df_fix["tStart"].to_numpy(),
        "duration": df_fix["duration"].to_numpy(),
        "description": np.array(["_".join(["fixation", eye]) for eye in df_fix["eye"].to_list()])
    }))
    df_sacc = pd.DataFrame(pd.DataFrame({
        "onset": df_sacc["tStart"].to_numpy(),
        "duration": df_sacc["duration"].to_numpy(),
        "description": np.array(["_".join(["saccade", eye]) for eye in df_sacc["eye"].to_list()])
    }))
    df_blink = pd.DataFrame(pd.DataFrame({
        "onset": df_blink["tStart"].to_numpy(),
        "duration": df_blink["duration"].to_numpy(),
        "description": np.array(["_".join(["blink", eye]) for eye in df_blink["eye"].to_list()])
    }))
    # Concatenate everything into one table:
    annotations_table = pd.concat([events_df, df_fix, df_sacc, df_blink])
    annotations_table = annotations_table.sort_values("onset", ignore_index=True)
    # determine sample columns based on eyes recorded in file
    # eyesInFile = np.unique(df_fix.eye)
    if len(eyes_in_file) == 2:
        print('binocular data detected.')
        cols = [T_SAMPLE, 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
    else:
        eye = eyes_in_file
        print(f"monocular data detected {eye}")
        cols = [T_SAMPLE, f"{eye}X", f"{eye}Y", f"{eye}Pupil"]
    # Import samples
    print('Parsing samples')
    i_not_sample = np.nonzero(np.logical_or(line_type != SAMPLE, np.arange(n_lines) < i_start_rec))[0]
    df_samples = pd.read_csv(ascii_file, skiprows=i_not_sample, header=None, delim_whitespace=True,
                             usecols=range(0, len(cols)))
    df_samples.columns = cols
    # Convert values to numbers
    for eye in ['L', 'R']:
        if eye in eyes_in_file:
            df_samples[f"{eye}X"] = pd.to_numeric(df_samples[f"{eye}X"], errors='coerce')
            df_samples[f"{eye}Y"] = pd.to_numeric(df_samples[f"{eye}Y"], errors='coerce')
            df_samples[f"{eye}Pupil"] = pd.to_numeric(df_samples[f"{eye}Pupil"], errors='coerce')
        else:
            df_samples[f"{eye}X"] = np.nan
            df_samples[f"{eye}Y"] = np.nan
            df_samples[f"{eye}Pupil"] = np.nan
    return df_samples, annotations_table, sfreq


def df2mne(eyelink_df, annotations_df, sfreq):
    """

    :param eyelink_df:
    :param annotations_df:
    :return:
    """
    # Create the info:
    ch_names = ["LX", "LY", "LPupil", "RX", "RY", "RPupil"]
    ch_types = ["eog"] * len(ch_names)
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
    data = eyelink_df[["LX", "LY", "LPupil", "RX", "RY", "RPupil"]].to_numpy()
    raw = mne.io.RawArray(data.T, info)
    # Add the annotations:
    # Creating annotation from onset, duration and description:
    my_annotations = mne.Annotations(onset=(annotations_df["onset"] - eyelink_df["tSample"].to_list()[0]) * 10**-3,
                                     duration=annotations_df["duration"] * 10**-3,
                                     description=annotations_df["description"])

    # Setting the annotation in the raw signal
    raw.set_annotations(my_annotations)
    return raw


def save_to_bids(raw, subject="", session="",  task="", datatype="eyetrack", root=""):
    """

    :param raw:
    :param subject:
    :param session:
    :param task:
    :param datatype:
    :param root:
    :return:
    """
    # Create the save dir:
    save_dir = Path(root, "sub-" + subject, "ses-" + session, datatype)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Create the file name:
    file_name = "sub-{}_ses-{}_task-{}_eyetrack-raw.fif".format(subject, session, task)
    raw.save(Path(save_dir, file_name), overwrite=True)

    return None


def edf2ascii(convert_exe, edf_file_name):
    """
    This function converts an edf file to an ascii file
    :param edf_file_name:
    :param convert_exe:
    :return:
    """
    cmd = convert_exe
    ascfile = Path(edf_file_name.parent, edf_file_name.stem + ".asc")

    # check if an asc file already exists
    if not os.path.isfile(ascfile):
        subprocess.run([cmd, "-p", edf_file_name.parent, edf_file_name])
    else:
        warnings.warn("An Ascii file for " + edf_file_name.stem + " already exists!")
    return ascfile


def ascii2mne_batch(root, subjects, save_root, session="1", convert_exe=""):
    """

    :param subjects:
    :return:
    """
    # Loop through each subject
    for subject in subjects:
        # Get the subject directory:
        subject_dir = Path(root, subject, "ses-" + session)
        # List the files in there:
        subject_files = [fl for fl in os.listdir(subject_dir) if fl.endswith(".edf")]
        for task in tasks:
            task_files = [fl for fl in subject_files if fl.split("_task-")[1].split("_eyetrack.edf")[0] == task]
            # Loop through every file:
            raws = []
            for fl in task_files:
                # Read in EyeLink file
                print('Converting edf to asc')
                asci_file = edf2ascii(convert_exe, Path(subject_dir, fl))
                print('Reading in EyeLink file %s' % fl)
                data_df, msg_df, sfreq = ascii_parser(asci_file)
                raws.append(df2mne(data_df, msg_df, sfreq))
            # Concatenate:
            raw = mne.concatenate_raws(raws)
            save_to_bids(raw, subject=subject.split("-")[1], session=session, task=task.replace("_", ""),
                         datatype="eyetrack", root=save_root)


if __name__ == "__main__":
    subjects_list = ["sub-SX102", "sub-SX103"]
    tasks = ["auditory_and_visual", "visual", "auditory", "prp"]
    data_root = r"C:\Users\alexander.lepauvre\Documents\PhD\Reconstructed_Time\raw_data"
    save_root = r"C:\Users\alexander.lepauvre\Documents\PhD\Reconstructed_Time\bids"
    ascii2mne_batch(data_root, subjects_list, save_root,
                    convert_exe=r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker\edf2asc.exe")
