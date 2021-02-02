#!/bin/bash
# bash scripts for calling the following python scripts:
# main.py : gmm model training & prediction 
# pred_only.py : load pretrained gmm model and predict 
# Author: Jian Yang


set -e
stage="$1" # parse first argument 

# train + prediction
if [ $stage -eq 0 ]; then
    python3 main.py \
        --train-txt ./data/train/feat/ivectors_enroll_mfcc/ivector.txt \
        --eval-txt ./data/eval/feat/ivectors_enroll_mfcc/ivector.txt  \
        --logging-dir logs/ \
        --model-dir pretrained/
fi

# load pretrained and do prediction only
if [ $stage -eq 1 ]; then
    python3 pred_only.py \
        --eval-txt ./data/eval/feat/ivectors_enroll_mfcc/ivector.txt  \
        --logging-dir ./logs/ \
        --model-genuine ./pretrained/gmm-ivector-2021-02-02_00_02_35-genuine.pkl \
        --model-spoof ./pretrained/gmm-ivector-2021-02-02_00_02_35-spoof.pkl
fi