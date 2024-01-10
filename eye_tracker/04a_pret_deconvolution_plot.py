import numpy as np
import pandas as pd
from scipy.io import savemat
import environment_variables as ev
from pathlib import Path
import os
import mne
import matplotlib.pyplot as plt
from plotter_functions import plot_within_subject_boxplot


def plot_pret_latencies(data_dir, session="1", task="prp", conditions_mapping=None):
    """

    :param data_dir:
    :param output_dir:
    :param session:
    :param task:
    :param conditions_filter:
    :param factors:
    :return:
    """
    if conditions_mapping is None:
        conditions_mapping = {
            "SOA": {
                "SOA1": 0.0,
                "SOA2": 0.116,
                "SOA3": 0.232,
                "SOA4": 0.466
            },
            "duration": {
                "duration1": 0.5,
                "duration2": 1.0,
                "duration3": 1.5
            }
        }
    # Read the results:
    results = pd.read_csv(Path(data_dir, "ses-{}_task-{}_desc-deconvolution_res.csv".format(session, task)))
    # Convert the conditions back:
    for col in conditions_mapping.keys():
        results[col] = results[col].replace(conditions_mapping[col])
    # Plot the results:
    face_colors_audio = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ]
    face_color_vis = [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ]
    # ===========================================================================
    # Onset locked:
    fig, ax_tr = plt.subplots(nrows=1, ncols=1)
    plot_within_subject_boxplot(results[(results["lock"] == "onset") & (results["task"] == "nontarget")],
                                "subject", "SOA",
                                "audOnset", positions="SOA",
                                ax=ax_tr, cousineau_correction=False, title="", xlabel="", ylabel="", xlim=None,
                                width=0.1, face_colors=face_colors_audio, edge_colors=None, xlabel_fontsize=9)
    plot_within_subject_boxplot(results[(results["lock"] == "onset") & (results["task"] == "nontarget")],
                                "subject", "SOA",
                                "visOnset", positions="SOA",
                                ax=ax_tr, cousineau_correction=False, title="", xlabel="", ylabel="", xlim=None,
                                width=0.1, face_colors=face_color_vis, edge_colors=None, xlabel_fontsize=9)
    ax_tr.set_xlim([-0.1, 0.5])
    ax_tr.set_xlabel("SOA (sec.)")
    ax_tr.set_ylabel("Event latency (sec.)")

    fig, ax_ti = plt.subplots(nrows=1, ncols=1)
    plot_within_subject_boxplot(results[(results["lock"] == "onset") & (results["task"] == "irrelevant")],
                                "subject", "SOA",
                                "audOnset", positions="SOA",
                                ax=ax_ti, cousineau_correction=False, title="", xlabel="", ylabel="", xlim=None,
                                width=0.1, face_colors=face_colors_audio, edge_colors=None, xlabel_fontsize=9)
    plot_within_subject_boxplot(results[(results["lock"] == "onset") & (results["task"] == "irrelevant")],
                                "subject", "SOA",
                                "visOnset", positions="SOA",
                                ax=ax_ti, cousineau_correction=False, title="", xlabel="", ylabel="", xlim=None,
                                width=0.1, face_colors=face_color_vis, edge_colors=None, xlabel_fontsize=9)
    ax_ti.set_xlim([-0.1, 0.5])
    ax_ti.set_xlabel("SOA (sec.)")
    ax_ti.set_ylabel("Event latency (sec.)")
    # Uniformize the ylims:
    ylims = [ax_tr.get_ylim()[0], ax_tr.get_ylim()[1], ax_ti.get_ylim()[0], ax_ti.get_ylim()[1]]
    ylims_new = [min(ylims), max(ylims)]
    ax_tr.set_ylim(ylims_new)
    ax_ti.set_ylim(ylims_new)

    # ===========================================================================
    # Offset locked:
    results_offset = results[results["lock"] == "onset"]
    # Add the duration to the SOA:
    results_offset["onset_SOA"] = results_offset["SOA"].to_numpy() + results_offset["duration"].to_numpy()
    fig, ax_tr = plt.subplots(nrows=1, ncols=1)
    for dur in list(results_offset["duration"].unique()):
        plot_within_subject_boxplot(results_offset[results_offset["duration"] == dur],
                                    "subject", "onset_SOA",
                                    "audOnset", positions="onset_SOA",
                                    ax=ax_tr, cousineau_correction=False, title="", xlabel="", ylabel="", xlim=None,
                                    width=0.1, face_colors=face_colors_audio, edge_colors=None, xlabel_fontsize=9)
        plot_within_subject_boxplot(results_offset[results_offset["duration"] == dur],
                                    "subject", "onset_SOA",
                                    "visOffset", positions="onset_SOA",
                                    ax=ax_tr, cousineau_correction=False, title="", xlabel="", ylabel="", xlim=None,
                                    width=0.1, face_colors=face_color_vis, edge_colors=None, xlabel_fontsize=9)
    ax_tr.set_xlim([0.4, 2.1])
    ax_tr.set_xlabel("SOA (sec.)")
    ax_tr.set_ylabel("Event latency (sec.)")
    plt.show()


if __name__ == "__main__":
    mapping = {
        "SOA": {
            "SOA1": 0.0,
            "SOA2": 0.116,
            "SOA3": 0.232,
            "SOA4": 0.466
        },
        "duration": {
            "duration1": 0.5,
            "duration2": 1.0,
            "duration3": 1.5
        }
    }
    subjects_list = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113",
                     "SX114", "SX115", "SX116"]  # "SX118", "SX119", "SX120", "SX121"]
    plot_pret_latencies(Path(ev.bids_root, "derivatives", "pret"), session="1", task="prp")
