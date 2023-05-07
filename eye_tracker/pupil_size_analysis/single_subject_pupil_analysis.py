import mne
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from eye_tracker.general_helper_function import baseline_scaling
import os


def get_blinks_matrix(epochs):
    print("A")
    return 1, 2


def get_saccades_matrix(epochs):
    print("A")
    return 1, 2


def single_sub_pupil(subjects, parameters):
    """

    :param subject:
    :param parameters:
    :return:
    """
    # First, load the parameters:
    with open(parameters) as json_file:
        param = json.load(json_file)
    # Load the parameters:
    bids_root = param["bids_root"]
    session = param["session"]
    task = param["task"]
    data_type = param["data_type"]
    analyses_parameters = param["analyses_parameters"]

    # Loop through each analysis:
    for analysis_name in analyses_parameters.keys():
        print("Performing analysis: {}".format(analysis_name))
        analysis_param = analyses_parameters[analysis_name]

        # Load the data:
        root = Path(bids_root, "derivatives", "preprocessing", "sub-" + subjects, "ses-" + session,
                    data_type)
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}-epo.fif".format(subjects, session, task, data_type,
                                                                      analysis_param["epoch_name"])
        epochs = mne.read_epochs(Path(root, file_name))
        # Extract the conditions:
        epochs = epochs[analysis_param["conditions"]]
        # Crop if needed:
        if analysis_param["crop"] is not None:
            epochs.crop(tmin=analysis_param["crop"][0], tmax=analysis_param["crop"][1])

        # Extract the eyelink events channels before moving forward:
        if analysis_param["remove_blinks"]:
            blinks_mat_left, blinks_mat_left = get_blinks_matrix(epochs)
        else:
            blinks_mat_left, blinks_mat_left = None, None
        if analysis_param["remove_saccades"]:
            blinks_mat_left, blinks_mat_left = get_saccades_matrix(epochs)
        else:
            blinks_mat_left, blinks_mat_left = None, None

        # Extract only the channels we need:
        epochs.pick(analysis_param["picks"])
        # Do baseline scaling if necessary:
        if analysis_param["baseline_mode"] is not None:
            baseline_scaling(epochs, correction_method=analysis_param["baseline_mode"],
                             baseline=analysis_param["baseline_window"])

        # Extract each of the condition to compare:
        conds = analysis_param["compared_condition"]
        mean_cond = []
        std_err = []
        con = []
        for cond in conds:
            con.append(cond)
            # Get the data:
            data = epochs.copy()[cond].get_data()
            # Average across both eyes and across trials:
            mean_cond.append(np.mean(data, axis=(1, 0)))
            # Compute the SEM:
            avg_eye = np.mean(data, axis=1)
            std_err.append(sem(avg_eye, axis=0))

        # Plot:
        fig, ax = plt.subplots()
        for ind, data in enumerate(mean_cond):
            # Plot the average pupil size in that condition:
            ax.plot(epochs.times, data, label=con[ind])
            # Add the error around that:
            ax.fill_between(epochs.times, data - std_err[ind], data + std_err[ind],
                            alpha=0.5)
        plt.legend()
        plt.ylabel("Pupil size (norm.)")
        plt.xlabel("Time (s)")
        plt.tight_layout()

        # Save the figure:
        save_root = Path(bids_root, "derivatives", "pupil_size", "sub-" + subjects,
                         "ses-" + session, data_type)
        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        file_name = "sub-{}_ses-{}_task-{}_{}_desc-{}_ana-{}.png".format(subjects, session, task, data_type,
                                                                         analysis_param["epoch_name"], analysis_name)
        fig.savefig(Path(save_root, file_name))
        plt.close(fig)


if __name__ == "__main__":
    subs_list = "SX105"
    parameters_file = r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis\eye_tracker" \
                      r"\pupil_size_analysis\config\no_preprocess_config.json "
    single_sub_pupil(subs_list, parameters_file)
