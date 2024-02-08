import mne
import numpy as np
from pathlib import Path
import pandas as pd
import os
import shutil
import subprocess
import warnings
import environment_variables as ev


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


def ascii2mne_batch(raw_root, subjects, bids_root, tasks, session="1", convert_exe=""):
    """

    :param subjects:
    :return:
    """
    # Loop through each subject
    for subject in subjects:
        # Get the subject directory:
        subject_dir = Path(raw_root, "sub-" + subject, "ses-" + session)
        # List the files in there:
        subject_files = [fl for fl in os.listdir(subject_dir) if fl.endswith(".edf")]
        # Create the save dir:
        if subject == "SX122":
            save_dir = Path(bids_root, "sub-" + "SX116", "ses-" + session, "eyetrack")
        else:
            save_dir = Path(bids_root, "sub-" + subject, "ses-" + session, "eyetrack")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for task in tasks:
            task_files = [fl for fl in subject_files if fl.split("_task-")[1].split("_eyetrack.edf")[0] == task]
            # Loop through every file:
            for fl in task_files:
                # Read in EyeLink file
                print('Converting {} to asc'.format(fl))
                asci_file = edf2ascii(convert_exe, Path(subject_dir, fl))
                # Copy paste the file to the bids directory:
                asci_stem = asci_file.stem
                # Add leading 0 to files which are less than 10, to make sure that the files are loaded in the right
                # order
                if task not in ["auditory", "visual"]:
                    if float(asci_stem.split('run-')[1].split('_task')[0]) <= 9:
                        asci_stem = (asci_stem.split('run-')[0] + "run-0" + asci_stem.split('run-')[1].split('_task')[0] +
                                     '_task' + asci_stem.split('run-')[1].split('_task')[1])
                else:
                    asci_stem = (asci_stem.split('run-')[0] + "run-00" +
                                 '_task' + asci_stem.split('run-')[1].split('_task')[1])
                if subject == "SX122":
                    asci_stem = asci_stem.replace(subject, "SX116")
                print('Copying {} to {}'.format(asci_file, save_dir))
                shutil.copyfile(asci_file, Path(save_dir, asci_stem + asci_file.suffix))


if __name__ == "__main__":
    subjects_list = [
         "SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX111", "SX112", "SX113",
         "SX114", "SX115", "SX116", "SX118", "SX119", "SX120", "SX121", "SX123"
     ]
    tasks_list = ["prp"]
    ascii2mne_batch(ev.raw_root, subjects_list, ev.bids_root, tasks_list,
                    convert_exe=r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker\edf2asc.exe")

    subjects_list = [
        "SX101", "SX105", "SX106", "SX108", "SX109", "SX110", "SX113", "SX114", "SX115", "SX118", "SX122"
    ]
    tasks_list = ["introspection"]
    ascii2mne_batch(ev.raw_root, subjects_list, ev.bids_root, tasks_list, session="2",
                    convert_exe=r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker\edf2asc.exe")

    subjects_list = [
        "SX101", "SX105", "SX106", "SX108", "SX109", "SX110", "SX113", "SX114", "SX115", "SX118", "SX122"
    ]
    tasks_list = ["introspection"]
    ascii2mne_batch(ev.raw_root, subjects_list, ev.bids_root, tasks_list, session="3",
                    convert_exe=r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker\edf2asc.exe")
