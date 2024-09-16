#!/bin/bash

#PBS -u brijendra
#PBS -N trapti
#PBS -q gpu
#PBS -l select=2:ncpus=20:ngpus=1
#PBS -o out.log
#PBS -j oe
#PBS -V

module load compilers/intel/parallel_studio_xe_2018_update3_cluster_edition
#source /apps/intel/parallel_stdio_xe_2018_update3_cluster_edition
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1
cd $PBS_O_WORKDIR



python3 /nfsroot/data/home/brijendra/trapti/asd_NonASD.py


exit;



