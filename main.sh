#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=4-06:00:00
#SBATCH -o /home/mila/l/lena-nehale.ezzine/scratch/cyclic_peptide/slurm-%j.out

micromamba activate dem

exec python dem/train.py experiment=torchani_idem 