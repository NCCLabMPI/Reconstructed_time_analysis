#!/bin/bash

#SBATCH --job-name=pret_deconvolution
#SBATCH --partition=octopus
#SBATCH --cpus-per-task=40
#SBATCH --output=%x_%j.out
#SBATCH --chdir=/mnt/beegfs/users/alexander.lepauvre/Reconstructed_time_analysis/eye_tracker
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alex.lepauvre@ae.mpg.de

module load matlab

srun matlab -nodesktop < O5b_pret_deconvolution.m
