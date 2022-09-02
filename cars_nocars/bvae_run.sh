#!/bin/bash
#SBATCH --partition=main
#SBATCH -t 40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G

# your code goes below
module load python/3.9.9
source ../py39/bin/activate
python3.8 bvae_code.py