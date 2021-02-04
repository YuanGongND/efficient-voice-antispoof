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
        --train-txt ./data/train/feat/xvectors_enroll_mfcc/xvector.txt \
        --eval-txt ./data/eval/feat/xvectors_enroll_mfcc/xvector.txt  \
        --logging-dir logs/ \
        --model-dir pretrained/
fi

# load pretrained and do prediction only
if [ $stage -eq 1 ]; then
    python3 pred_only.py \
        --eval-txt ./data/eval/feat/xvectors_enroll_mfcc/xvector.txt  \
        --logging-dir ./logs/ \
        --model-genuine ./pretrained/gmm-xvector-2021-02-04_17_52_14-genuine.pkl \
        --model-spoof ./pretrained/gmm-xvector-2021-02-04_17_52_14-spoof.pkl
fi