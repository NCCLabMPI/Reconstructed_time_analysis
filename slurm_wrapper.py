import os
import subprocess
import mne
import environment_variables as ev


def decoding_batch(configs):
    """
    This function runs jobs for each config found in the current config folder!
    :return:
    """
    labels = mne.read_labels_from_annot("fsaverage", parc='aparc.a2009s', hemi='both', surf_name='pial',
                                        subjects_dir=ev.fs_directory, sort=True)
    roi_names = list(set(lbl.name.replace("-lh", "").replace("-rh", "")
                         for lbl in labels if "unknown" not in lbl.name))
    # Launching a job for each:
    for config in configs:
        for roi in roi_names:
            # Run the rsa analysis script using the customized config file
            run_command = f'sbatch SLURM_decoding.sh --config="{config}" --roi="{roi}"'
            print(run_command)
            subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    # Getting the current dir
    pwd = os.getcwd()
    decoding_batch([
         "./decoding_pseudotrials.json",
         "./decoding_pseudotrials_5ms.json",
         "./decoding_no_pseudo.json",
         "./decoding_no_pseudo_5ms.json",
         "./decoding_pseudotrials_acc.json",
         "./decoding_pseudotrials_5ms_acc.json",
         "./decoding_no_pseudo_acc.json",
         "./decoding_no_pseudo_5ms_acc.json"
    ])
