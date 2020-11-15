#!/bin/bash
# bash script for auto run of reduced input

set -e

# uncomment below for GPU machine run
cmd="CUDA_VISIBLE_DEVICES='0,1' python"
# uncomment below for Jetson run
# cmd="CUDA_VISIBLE_DEVICES='0' python3"
# uncomment below for Pi4 run
# cmd="python3"

for size in 512 256 128; do
    eval $cmd predict_only.py \
        --eval-dir ../feat_257_$size/ASVspoof2017/eval_1k_sample \
        --eval-utt2label ../feat_257_$size/ASVspoof2017/labels/eval_1k_sample_$size.utt2label \
        --logging-dir snapshots/predict_only/ \
        --test-batch-size 1 \
        --input-dim $size
done

eval $cmd predict_only.py \
    --eval-dir ../feat_aligned/ASVspoof2017/eval_1k_sample \
    --eval-utt2label ../feat_aligned/ASVspoof2017/labels/eval_1k_sample.utt2label \
    --logging-dir snapshots/predict_only/ \
    --test-batch-size 1 \