import os
import subprocess
import warnings


def convert_batch(root, subjects, convert_exe, session="1"):
    """
    This function converts a batch of edf files to ascii files. Expects a folder strcuture where
    the parent folder contains subject folders (indicated by subnames) and within each subject
    folder there are edf files. The function will convert all edf files in a folder to ascii
    :param root:
    :param subjects:
    :param convert_exe
    :param session:
    :return:
    """

    # loop through each subject folder
    for subject in subjects:
        # List  the EDF files
        edfs = [fl for fl in os.listdir(root + os.sep + subject + os.sep + "ses-" + session)
                if fl.endswith(".edf")]

        print("Working on subject: " + subject + ". Detected %d files:" % len(edfs))
        print(edfs)

        # loop through the edfs
        for edf in edfs:
            print("Converting " + edf + "...")
            # convert
            convert_file(root + os.sep + subject + os.sep + "ses-" + session, edf, convert_exe)


def convert_file(path, name, convert_exe):
    """
    This function converts an edf file to an ascii file
    :param path:
    :param name:
    :param convert_exe:
    :return:
    """
    cmd = convert_exe
    ascfile = name[:-3] + "asc"

    # check if an asc file already exists
    if not os.path.isfile(path + os.sep + ascfile):
        subprocess.run([cmd, "-p", path, path + os.sep + name])
    else:
        warnings.warn("An Ascii file for " + name + " already exists!")


def main():
    root = r"C:\Users\alexander.lepauvre\Documents\PhD\Reconstructed_Time\data"
    subjects = ["sub-SX105"]
    convert_exe = r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker\edf2asc.exe"

    convert_batch(root, subjects, convert_exe, session="1")


if __name__ == "__main__":
    main()