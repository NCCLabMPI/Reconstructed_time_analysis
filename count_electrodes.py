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



