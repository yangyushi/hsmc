#!/bin/bash
# request resources:
#PBS -N slit-simulate
#PBS -l nodes=1:ppn=1
#PBS -l walltime=72:00:00
#PBS -l mem=16gb
#PBS -q himem

capture_err() {
    exit_code=$1
    if [[ $exit_code -ne 0 ]]; then
        exit 1
    fi
}

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
python3 simulate.py; capture_err $?
mv *.png result
python3 analyse.py; capture_err $?
python3 plot.py; capture_err $?
mv *.pdf result
rm -rf tcc_bulk
