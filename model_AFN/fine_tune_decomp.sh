#!/bin/bash
# bash script for fine tune decomposition
# decomped model saved in snapshots/attention

set -e

for rank in 2 4; do
    for type in tucker cp; do
    echo "Decomp with rank = $rank, type = $type"
        CUDA_VISIBLE_DEVICES='0,1' python fine_tune_decomp.py \
            --train-dir ../feat_aligned/ASVspoof2017/train \
            --train-utt2label ../data/ASVspoof2017/protocol_V2/train.txt.utt2label \
            --validation-dir ../feat_aligned/ASVspoof2017/dev \
            --validation-utt2label ../data/ASVspoof2017/protocol_V2/dev.txt.utt2label \
            --logging-dir snapshots/attention/ \
            --epochs 30 \
            --log-interval 50 \
            --seed 1 \
            --rank $rank \
            --type $type
    done
done

