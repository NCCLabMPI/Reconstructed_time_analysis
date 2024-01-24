import numpy as np
import pandas as pd
from scipy.io import savemat
import environment_variables as ev
from pathlib import Path
import os
import mne
import matplotlib.pyplot as plt
from plotter_functions import plot_within_subject_boxplot, soa_boxplot

# Set the font size:
plt.rcParams.update({'font.size': 14})
dpi = 300


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
    # Create the save directory:
    save_dir = Path(ev.bids_root, "derivatives", "pret")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
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
    # Convert to seconds:
    results["tau-audOnset"] = results["tau-audOnset"] / 1000
    results["tau-visOnset"] = results["tau-visOnset"] / 1000
    results["tau-visOffset"] = results["tau-visOffset"] / 1000
    # Add the onset soa column:
    results.loc[:, "onset_SOA"] = results["SOA"]
    results.loc[results["lock"] == "offset", "onset_SOA"] = (results.loc[results["lock"] == "offset", "onset_SOA"] +
                                                             results.loc[results["lock"] == "offset", "duration"])

    # Plot the tau to the auditory stimuli:
    fig_aud, ax_aud = soa_boxplot(results,
                                  "tau-audOnset",
                                  fig_size=[8.3 / 2, 11.7 / 2], lock_column="lock",
                                  subject_column="subject",
                                  between_column="onset_SOA", ax=None, fig=None)
    # Plot the tau to the visual stimuli onset:
    fig_visOnset, ax_visOnset = soa_boxplot(results,
                                            "tau-visOnset",
                                            fig_size=[8.3 / 2, 11.7 / 2], lock_column="lock",
                                            subject_column="subject",
                                            between_column="onset_SOA", ax=None, fig=None,
                                            colors_onset_locked=[ev.colors["visOnset"][soa]
                                                                 for soa in ev.colors["visOnset"].keys()],
                                            colors_offset_locked=[ev.colors["visOnset"][soa]
                                                                  for soa in ev.colors["visOnset"].keys()]
                                            )
    # Plot the tau to the visual stimuli offset:
    fig_visOffset, ax_visOffset = soa_boxplot(results,
                                              "tau-visOffset",
                                              fig_size=[8.3 / 2, 11.7 / 2], lock_column="lock",
                                              subject_column="subject",
                                              between_column="onset_SOA", ax=None, fig=None,
                                              colors_onset_locked=[ev.colors["visOffset"][soa]
                                                                   for soa in ev.colors["visOffset"].keys()],
                                              colors_offset_locked=[ev.colors["visOffset"][soa]
                                                                    for soa in ev.colors["visOffset"].keys()])
    ylims = ax_aud[0].get_ylim() + ax_visOnset[0].get_ylim() + ax_visOffset[0].get_ylim()
    ylims = [min(ylims), max(ylims)]
    ax_aud[0].set_ylim(ylims)
    ax_visOnset[0].set_ylim(ylims)
    ax_visOffset[0].set_ylim(ylims)
    # Axes decoration:
    fig_aud.suptitle("T2 pupil response")
    fig_visOnset.suptitle("T1 pupil response")
    fig_visOffset.suptitle("Offset pupil response")
    fig_aud.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_visOnset.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_visOffset.text(0.5, 0, 'Time (sec.)', ha='center', va='center')
    fig_aud.text(0, 0.5, r'$\tau_{\mathrm{audio}}$', ha='center', va='center', fontsize=18, rotation=90)
    fig_visOnset.text(0, 0.5, r'$\tau_{\mathrm{Onset}}$', ha='center', va='center', fontsize=18, rotation=90)
    fig_visOffset.text(0, 0.5, r'$\tau_{\mathrm{Offset}}$', ha='center', va='center', fontsize=18, rotation=90)
    fig_aud.savefig(Path(save_dir, "pupil_response_audio.svg"), transparent=True, dpi=dpi)
    fig_aud.savefig(Path(save_dir, "pupil_response_audio.png"), transparent=True, dpi=dpi)
    fig_visOnset.savefig(Path(save_dir, "pupil_response_visOnset.svg"), transparent=True, dpi=dpi)
    fig_visOnset.savefig(Path(save_dir, "pupil_response_visOnset.png"), transparent=True, dpi=dpi)
    fig_visOffset.savefig(Path(save_dir, "pupil_response_visOffset.svg"), transparent=True, dpi=dpi)
    fig_visOffset.savefig(Path(save_dir, "pupil_response_visOffset.png"), transparent=True, dpi=dpi)
    plt.close(fig_aud)
    plt.close(fig_visOnset)
    plt.close(fig_visOffset)


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
    plot_pret_latencies(Path(ev.bids_root, "derivatives", "pret"), session="1", task="prp")
