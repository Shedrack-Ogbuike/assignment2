#!/bin/bash
#SBATCH --job-name=enel645_a2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/work/TALC/ensf617_2026w/slurm_%j.out

source ~/venvs/enel645_a2/bin/activate

echo "Running on: $(hostname)"
nvidia-smi

python /work/TALC/ensf617_2026w/src/train_multimodal.py

