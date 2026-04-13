#!/bin/bash
# request resources:
#PBS -N slit-simulate
#PBS -l nodes=1:ppn=1
#PBS -l walltime=240:00:00

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

if [[ ! -d figure ]]; then
    mkdir figure
fi

if [[ ! -d tcc ]]; then
    mkdir tcc
fi

python3 simulate.py; capture_err $?
python3 bulk.py; capture_err $?
python3 analysis.py; capture_err $?
python3 plot.py; capture_err $?
