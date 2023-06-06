#!/bin/sh


#SBATCH --account=urbangroup         # Replace ACCOUNT with your group account name
#SBATCH --job-name=SOFE     # The job name.
#SBATCH -c 18                      # The number of cpu cores to use
#SBATCH -t 0-12:00                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=6gb         # The memory the job will use per cpu core

module load anaconda
module load QE/7.2
# module load cuda11.1/toolkit

# export LD_LIBRARY_PATH="/burg/home/msa2187/.local/lib/python3.9/site-packages/nvidia/cublas/lib":$LD_LIBRARY_PATH
export CONDA_ROOT="/burg/opt/anaconda3-2022.05/"
. $CONDA_ROOT/etc/profile.d/conda.sh
conda activate sofe

export PYTHONPATH=$PYTHONPATH:/burg/opt/anaconda3-2022.05/lib/python3.9/site-packages:/burg/urbangroup/users/msa2187/sofe_env/bin/pip:/burg/urbangroup/users/msa2187/sofe_env/lib/python3.9/site-packages
export ASE_ESPRESSO_COMMAND=/burg/opt/QE/7.2/bin/pw.x -in PREFIX.pwi > PREFIX.pwo
export ESPRESSO_PSEUDO=/burg/urbangroup/users/msa2187/pseudo

#Command to execute Python program
/burg/urbangroup/users/msa2187/sofe_env/bin/python sputter.py

#End of script