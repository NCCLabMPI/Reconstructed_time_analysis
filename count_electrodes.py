import mne
from mne_bids import BIDSPath
import environment_variables as ev
import pandas as pd
from pathlib import Path


subjects_list = ev.subjects_lists_ecog["dur"]

cts = {"seeg": 0, "ecog": 0}

# Get their path:
for sub in subjects_list:
    path = BIDSPath(root=r"C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\bids-curate",
                    subject=sub, session="1", task="Dur", datatype="ieeg")

    # Load the channels.tsv:
    channels = pd.read_csv(Path(path.directory, f"sub-{sub}_ses-1_task-Dur_channels.tsv"), sep="\t")

    # Get each channel types:
    cts["seeg"] += channels[channels["type"] == "SEEG"].shape[0]
    cts["ecog"] += channels[channels["type"] == "ECOG"].shape[0]

print(cts)

# Count the electrodes remaining after preprocessing:
cts = {"seeg": 0, "ecog": 0}
for sub in subjects_list:
    epochs = mne.read_epochs(fr"C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\bids-curate\derivatives\preprocessing\sub-{sub}\ses-1\ieeg\epoching\high_gamma\sub-{sub}_ses-1_task-Dur_desc-epoching_ieeg-epo.fif", verbose='ERROR')
    epochs.drop_channels(epochs.info['bads'])
    ch_types = epochs.get_channel_types()
    # Get each channel types:
    cts["seeg"] += len([ch for ch in ch_types if ch == 'seeg'])
    cts["ecog"] += len([ch for ch in ch_types if ch == 'ecog'])

print(cts)