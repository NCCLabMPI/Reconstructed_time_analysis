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


def ascii2mne_batch(raw_root, subjects, bids_root, session="1", convert_exe=""):
    """

    :param subjects:
    :return:
    """
    # Loop through each subject
    for subject in subjects:
        # Get the subject directory:
        subject_dir = Path(raw_root, subject, "ses-" + session)
        # List the files in there:
        subject_files = [fl for fl in os.listdir(subject_dir) if fl.endswith(".edf")]
        # Create the save dir:
        save_dir = Path(bids_root, subject, "ses-" + session, "eyetrack")
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
                print('Copying {} to {}'.format(asci_file, save_dir))
                shutil.copyfile(asci_file, Path(save_dir, asci_file.stem + asci_file.suffix))


if __name__ == "__main__":
    subjects_list = [
        "sub-SX102"
    ]
    tasks = ["prp"]
    # data_root = r"C:\Users\alexander.lepauvre\Documents\PhD\Reconstructed_Time\raw_data"
    ascii2mne_batch(ev.raw_root, subjects_list, ev.bids_root,
                    convert_exe=r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker\edf2asc.exe")
