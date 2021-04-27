#!/bin/bash
# request resources:
#PBS -N slit-simulate
#PBS -l nodes=1:ppn=1
#PBS -l walltime=72:00:00
#PBS -l mem=16gb
#PBS -q himem

if [ $PBS_O_WORKDIR ]; then
    cd $PBS_O_WORKDIR
fi

if [[ ! -d result ]]; then
    mkdir result
fi

if [[ ! -d tcc ]]; then
    mkdir tcc
fi

./clean
python3 simulate.py
mv *.png result
python3 analyse.py
mv *.pdf result
