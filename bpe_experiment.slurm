## batch job to run experiments
#!/bin/bash
#SBATCH --job-name=bpe_exp
#SBATCH --partition=kill-shared
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G ## max amount of memory per node you require
#SBATCH --error=err_bpe_exp-%A_%a.err ## %A - filled with jobid
#SBATCH --output=out_bpe_exp-%A_%a.out ## %A - filled with jobid 
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=vchan26@hawaii.edu


module purge
module load lang/Anaconda3
source ~/.bash_profile
source activate eceassignment1

python ece496b_basics/train_bpe.py
