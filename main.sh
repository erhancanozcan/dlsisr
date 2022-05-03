#!/bin/bash -l

# Specify project
#$ -P dl523

# Request appropriate time (default 12 hours; gpu jobs time limit - 2 days (48 hours), cpu jobs - 30 days (720 hours) )
#$ -l h_rt=24:00:00

# CPUs
#$ -pe omp 2

# GPU usage
#$ -l gpus=1

#$ -l gpu_c=3.7

# Join output and error streams into one file
#$ -j y

# When to email
#$ -m bea

#load appropriate envornment
module load python3/3.8.10
module load pytorch
module load cuda

#execute the program
python main.py

