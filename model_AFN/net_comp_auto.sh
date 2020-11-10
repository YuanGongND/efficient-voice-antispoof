#!/bin/bash
# bash script for auto run of network compression of various args

set -e
# add new compression methods here: decomp, quant, prune
comp_methods=('decomp' 'quant' 'prune')
dataset=$1  # parse 1st arg, use 'sample' or 'whole'
device=$2  # parse 2nd arg, use 'GPU' or 'CPU'
# uncomment below for GPU machine run
cmd="CUDA_VISIBLE_DEVICES='0,1' python"
# uncomment below for Jetson run
# cmd="CUDA_VISIBLE_DEVICES='0' python3"
# uncomment below for Pi4 run
# cmd="python3"

# predict only with network compress choices on 1k_sample set
if [ "${dataset}" = "sample" ]; then
    if [ "${device}" = "GPU" ]; then
        for comp in "${comp_methods[@]}"; do
            echo $comp
            if [ "$comp" = "decomp" ]; then 
                for rank in 2 4; do
                    for type in tucker cp; do
                        eval $cmd predict_only.py \
                            --eval-dir ../feat_aligned/ASVspoof2017/eval_1k_sample \
                            --eval-utt2label ../feat_aligned/ASVspoof2017/labels/eval_1k_sample.utt2label \
                            --logging-dir snapshots/predict_only/ \
                            --test-batch-size 1 \
                            --compress $comp \
                            --decomp-rank $rank \
                            --decomp-type $type
                    done
                done
            fi
            # TODO: check the error when use qmethod = static
            if [ "$comp" = "quant" ]; then 
                for qmethod in dynamic; do
                    eval $cmd predict_only.py \
                        --eval-dir ../feat_aligned/ASVspoof2017/eval_1k_sample \
                        --eval-utt2label ../feat_aligned/ASVspoof2017/labels/eval_1k_sample.utt2label \
                        --logging-dir snapshots/predict_only/ \
                        --test-batch-size 1 \
                        --compress $comp \
                        --quant-method $qmethod
                done
            fi
            if [ "$comp" = "prune" ]; then
                for pmethod in L1Unstructured RandomUnstructured; do
                    for i in 0.{0..99..10}; do
                        eval $cmd predict_only.py \
                            --eval-dir ../feat_aligned/ASVspoof2017/eval_1k_sample \
                            --eval-utt2label ../feat_aligned/ASVspoof2017/labels/eval_1k_sample.utt2label \
                            --logging-dir snapshots/predict_only/ \
                            --test-batch-size 1 \
                            --compress $comp \
                            --prune-pct $i \
                            --prune-method $pmethod
                    done
                    eval $cmd predict_only.py \
                        --eval-dir ../feat_aligned/ASVspoof2017/eval_1k_sample \
                        --eval-utt2label ../feat_aligned/ASVspoof2017/labels/eval_1k_sample.utt2label \
                        --logging-dir snapshots/predict_only/ \
                        --test-batch-size 1 \
                        --compress $comp \
                        --prune-pct 1.0 \
                        --prune-method $pmethod
                done
            fi
        done
    fi

    if [ "${device}" = "CPU" ]; then
        for comp in "${comp_methods[@]}"; do
            echo $comp
            #TODO
            # eval $cmd
        done
    fi

fi
