import mne
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from helper_function.helper_general import baseline_scaling, extract_first_bout
import environment_variables as ev
from pingouin import ttest
from scipy.ndimage import uniform_filter1d
from helper_function.helper_plotter import plot_ts_ci

# Set the font size:
plt.rcParams.update({'font.size': 14})


def univariate_latency(parameters_file, subjects, data_root, session="1", task="dur", analysis_name="ieeg_latency",
                       task_conditions=None):
    # First, load the parameters:
    if task_conditions is None:
        task_conditions = ["Relevant non-target", "Irrelevant"]
    with open(parameters_file) as json_file:
        param = json.load(json_file)

    # Create the directory to save the results in:
    save_dir = Path(ev.bids_root, "derivatives", analysis_name, task)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    times = []
    # Load each subjects' data:
    for sub in subjects:
        # Create file name:
        epochs_file = Path(data_root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + session,
                           "ieeg", "epoching",
                           "sub-{}_ses-{}_task-{}_desc-epoching_ieeg-epo.fif".format(sub,
                                                                                     session,
                                                                                     task))
        # Load the file:
        epochs = mne.read_epochs(epochs_file)
        # Crop if needed:
        epochs.crop(param["crop"][0], param["crop"][1])
        times = epochs.times
        # Extract the conditions of interest:
        epochs = epochs[param["conditions"]]
        # Compute baseline correction:
        baseline_scaling(epochs, correction_method=param["baseline"], baseline=param["baseline_window"])

        # Crop the signal pre and post stimulus onset:
        epochs_pre = epochs.copy().crop(tmin=param["pre_window"][0], tmax=param["pre_window"][0])
        epochs_post = epochs.copy().crop(tmin=param["post_window"][0], tmax=param["post_window"][0])
        # Loop through all the channels:
        channels_results = {}
        for ch in epochs_pre.ch_names:
            # Extrac the data:
            data_pre = np.squeeze(epochs_pre.get_data(picks=ch))
            data_post = np.squeeze(epochs_post.get_data(picks=ch))
            results = ttest(data_pre, data_post, paired=True, r=0.707, alternative='two-sided').round(3)
            channels_results[ch] = results['BF10'].item()

        # Extract all the onset responsive channels:
        responsive_channels = [ch for ch in channels_results.keys() if float(channels_results[ch]) >= param["bayes_thresh"]]
        if len(responsive_channels) == 0:
            continue
        # Pick only those channels:
        epochs.pick(responsive_channels)

        # Extract task relevant and task irrelevant conditions.
        tr_epochs = epochs["Relevant non-target"]
        ti_epochs = epochs["Irrelevant"]

        # Loop through each channel:
        ch_latencies = {}
        for ch in tr_epochs.ch_names:
            # Extract the data:
            data_tr = uniform_filter1d(np.squeeze(tr_epochs.get_data(picks=ch)), size=20, axis=-1)
            data_ti = uniform_filter1d(np.squeeze(ti_epochs.get_data(picks=ch)), size=20, axis=-1)
            # Compute a sliding mann whitney u:
            res = mannwhitneyu(data_tr, data_ti, alternative="two-sided", axis=0)
            # Isolate bouts that are significant for n ms or more:
            onset, offset = extract_first_bout(times, np.array(res.pvalue),
                                               param["alpha"],
                                               param["dur_threshold"])
            if onset is not None:
                # Store the channel latency
                ch_latencies["-".join([sub, ch])] = [onset, offset]            # Plot the channel results:
                fig, ax = plt.subplots()
                plot_ts_ci(data_tr, epochs.times, ev.colors["task_relevance"]["Relevant non-target"],
                           plot_ci=True, ax=ax, label="Relevant non-target")
                plot_ts_ci(data_ti, epochs.times, ev.colors["task_relevance"]["Irrelevant"],
                           plot_ci=True, ax=ax, label="Irrelevant")
                ax.axvline(onset, color=ev.colors["soa"]["0.0"])
                ax.axvline(offset, color=ev.colors["soa_offset_locked"]["0.0"])
                ax.legend()
                ax.set_ylabel("HGP (norm.)")
                ax.set_xlabel("Time (sec.)")
                fig.savefig(Path(save_dir, "sub-{}_ch-{}_trti.svg".format(sub, ch.replace("*", ""))),
                            transparent=True, dpi=300)
                fig.savefig(Path(save_dir, "sub-{}_ch-{}_trti.png".format(sub, ch.replace("*", ""))),
                            transparent=True, dpi=300)
                plt.close()



if __name__ == "__main__":
    # Set the parameters to use:
    parameters = (
        r"C:\Users\alexander.lepauvre\Documents\GitHub\Reconstructed_time_analysis"
        r"\07-iEEG_univariate_latency_parameters.json")
    # ==================================================================================
    # Decoding analysis of the COGITATE data:
    univariate_latency(parameters, ev.subjects_lists_ecog["dur"], ev.bids_root,
                       session="V1", task="Dur", analysis_name="ieeg_latenyc",
                       task_conditions=["Relevant non-target", "Irrelevant"])
