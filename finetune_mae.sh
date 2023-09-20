#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH --mem=16G
#SBATCH -n 4
#SBATCH -J finetune-mae-on-mscoco 
#SBATCH -o finetune-mae-on-mscoco-%j.out
#SBATCH -e finetune-mae-on-mscoco-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiayu_zheng@brown.edu

# sometimes you need to specify python3 otherwise the system will use python2 as python
# python3 finetune_on_mscoco.py
torchrun finetune_on_mscoco.py
