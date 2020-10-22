#!/bin/bash
# bashe script for calling the command 
# `run.sh 2`
# with repeated run using varying params

# prune quant pq
set -e
device="$1"  # parse 1st arg, use CPU or GPU

if [ "${device}" = "GPU" ]; then
    for comp in prune quant pq; do
        if [ "${comp}" = "quant" ]; then
            echo "++++> predcition only for network "${comp}
            ./run.sh 3 ${comp} 1
        else
            echo "++++> predcition only for network "${comp}
            for i in 0.{0..99..10}; do
                ./run.sh 3 ${comp} $i
            done
            ./run.sh 3 ${comp} 1.0
        fi
    done
fi 

if [ "${device}" = "CPU" ]; then
    echo "++++> predcition only for network prune"
    for i in 0.{0..99..10}; do
        ./run.sh 4 $i
    done
    ./run.sh 4 ${comp} 1.0
fi