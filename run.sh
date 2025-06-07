#!/bin/bash

#SBATCH --job-name=movie
#SBATCH --output=%x_%j_%t.out
#SBATCH --error=%x_%j_%t.out
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=A6000:4
#SBATCH --time=0-23:59:59

source /home/nv059721/.bashrc
module load cuda/12.4.0
conda activate Movie_Rec

export NCCL_P2P_DISABLE=1
#srun echo "${SLURM_ARRAY_TASK_ID}" && nvidia-smi &

for i in {1..1}; do
	srun --ntasks=1 --gpus-per-task=A6000:4 --cpus-per-task=16 --output=$i.out --error=$i.err bash -c "nvidia-smi >> $i.out && torchrun mf_DDP.py" &

done

wait
