#!/bin/bash
#SBATCH --partition=xnat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=20000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=alex.lepauvre@ae.mpg.de
#SBATCH --time 48:00:00
#SBATCH --output=./slurm-%A_%a.out
#SBATCH --job-name=MVPA

config=""
roi=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config=*)
      config="${1#*=}"
    --roi=*)
      roi="${1#*=}"
  esac
  shift
  echo ${config}
done

cd /hpc/users/alexander.lepauvre/sw/github/Reconstructed_time_analysis

module purge; module load Anaconda3/2020.11; source /hpc/shared/EasyBuild/apps/Anaconda3/2020.11/bin/activate; conda activate /home/alexander.lepauvre/miniconda3/envs/mne

export PYTHONPATH=$PYTHONPATH:/hpc/users/alexander.lepauvre/sw/github/ECoG

python ./06-iEEG_decoding.py --config "${config}" --roi "${roi}"
