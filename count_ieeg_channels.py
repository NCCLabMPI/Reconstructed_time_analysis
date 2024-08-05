import os
import pandas as pd
import environment_variables as ev

def count_channels_in_bids(bids_root, subjects):
    channel_counts = []
    total_ecog_count = 0
    total_seeg_count = 0
    for subject in subjects:
        subject_path = os.path.join(bids_root, 'sub-' + subject)
        for root, _, files in os.walk(subject_path):
            for file in files:
                if file.endswith('_channels.tsv'):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, sep='\t')

                    ecog_count = df[df['type'] == 'ECOG'].shape[0]
                    seeg_count = df[df['type'] == 'SEEG'].shape[0]

                    channel_counts.append({
                        'subject': subject,
                        'ecog_count': ecog_count,
                        'seeg_count': seeg_count
                    })
                    total_ecog_count += ecog_count
                    total_seeg_count += seeg_count

    return channel_counts, total_ecog_count, total_seeg_count

# Usage example
bids_root = r'C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\bids-curate'
channel_counts, total_ecog_count, total_seeg_count = count_channels_in_bids(bids_root, ev.subjects_lists_ecog['dur'])

# Printing the results
for count in channel_counts:
    print(f"Subject: {count['subject']}, ECoG Channels: {count['ecog_count']}, SEEG Channels: {count['seeg_count']}")
print(f"Total: ECoG Channels: {total_ecog_count}, SEEG Channels: {total_seeg_count}")
