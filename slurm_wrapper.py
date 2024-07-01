import os
import subprocess


def decoding_batch(configs):
    """
    This function runs jobs for each config found in the current config folder!
    :return:
    """
    # Launching a job for each:
    for config in configs:
        # Run the rsa analysis script using the customized config file
        run_command = f'sbatch SLURM_decoding.sh --config="{config}"'
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
