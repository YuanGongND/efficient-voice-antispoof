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
        --train-dir ../feat_cqcc/train \
        --train-utt2label ../data/ASVspoof2017/protocol_V2/train.txt.utt2label \
        --eval-dir ../feat_cqcc/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir logs/ \
        --model-dir pretrained/
fi

# load pretrained and do prediction only
if [ $stage -eq 1 ]; then
    python3 pred_only.py \
        --eval-dir ../feat_cqcc/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir logs/ \
        --model-genuine pretrained/gmm-cqcc-2021-01-04_23_19_57-genuine.pkl \
        --model-spoof pretrained/gmm-cqcc-2021-01-04_23_19_57-spoof.pkl
fi