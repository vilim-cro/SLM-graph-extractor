#!/bin/sh
#BSUB -q gpuv100                      # Submit to the V100 GPU queue
#BSUB -J M7bInference                 # Job name
#BSUB -n 4                         # Number of CPU cores
#BSUB -gpu "num=1:mode=exclusive_process" # Request 1 GPU in exclusive mode
#BSUB -R "select[gpu32gb]"            # Ensure GPU has 32GB memory
#BSUB -R "select[sxm2]"               # Ensure the GPU is Tesla V100-SXM2
#BSUB -R "rusage[mem=40GB]"           # Request 35GB memory
#BSUB -W 10:00                        # Wall time (hh:mm)
#BSUB -B                              # Notify when the job begins
#BSUB -N                              # Notify when the job ends
#BSUB -o gpuv100_%J.out                   # Output log
#BSUB -e gpuv100_%J.err                   # Error log

# Check GPU status before running the program
nvidia-smi

module load python3/3.10.14           # Load Python
cd $BLACKHOLE                         # Navigate to working directory
source DL/bin/activate              # Activate virtual environment
module load cuda/12.4                 # Load appropriate CUDA version

# Set environment variable for memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your program
#python3 unmistral7b.py
python3 gemma7b.py

# Check GPU status after running the program
nvidia-smi
