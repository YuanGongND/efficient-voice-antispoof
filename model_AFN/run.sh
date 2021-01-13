#!/bin/bash
# bash scripts for calling the following python scripts:
# main.py : for model training 
# predict_only.py : for model forward passing
# Author: Jian Yang

# Uncomment below for GPU desktop run
cmd="CUDA_VISIBLE_DEVICES='0,1' python3"
# Uncomment below for Jetson run
# cmd="CUDA_VISIBLE_DEVICES='0' python3"
# Uncomment below for Pi4 run
# cmd="python3"

set -e
stage="$1" # parse 1st arg: choice of exp

# training, saving & picking the best model 
# run baseline AFN with train+validation+eval, repeat 5 rounds
if [ $stage -eq 0 ]; then
    for i in 1 2 3 4 5; do
        echo "AFN baseline training round: "$i
        eval $cmd main.py \
            --train-dir ../feat_aligned/train \
            --train-utt2label ../data/ASVspoof2017/protocol_V2/train.utt2label \
            --validation-dir ../feat_aligned/dev \
            --validation-utt2label ../data/ASVspoof2017/protocol_V2/dev.utt2label \
            --eval-dir ../feat_aligned/eval \
            --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.utt2label \
            --logging-dir ./logs/AFN4-train/ \
            --epochs 30 \
            --log-interval 50
    done
fi

# training, saving & picking the best model 
# run deformed AFN with train+validation+eval, each repeats 5 rounds
if [ $stage -eq 1 ]; then
    for win in 64 128 256 512; do
        for seg in s e h l hl lh; do
            for i in 1 2 3 4 5; do
                echo "Deformed AFN-"${win}"_"${seg}" training round: "$i
                eval $cmd main.py \
                    --train-dir ../feat_257_${win}/seg_${seg}/train \
                    --train-utt2label ../data/ASVspoof2017/protocol_V2/train.utt2label \
                    --validation-dir ../feat_257_${win}/seg_${seg}/dev \
                    --validation-utt2label ../data/ASVspoof2017/protocol_V2/dev.utt2label \
                    --eval-dir ../feat_257_${win}/seg_${seg}/eval \
                    --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.utt2label \
                    --logging-dir ./logs/AFN4De-train/ \
                    --epochs 30 \
                    --log-interval 50 \
                    --seg ${seg} \
                    --seg-win ${win}
            done
        done
    done
fi

# prediction only on the baseline AFN model (CPU)
if [ $stage -eq 10 ]; then
    eval $cmd base_pred.py \
        --eval-dir ../feat_aligned/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
        --model-path ./logs/AFN4-train/AFN4-1091-orig-model_best.pth \
        --test-batch-size 1 \
        --logging-dir ./logs/AFN4-pred-1k \
        --no-cuda
fi
# prediction only on the baseline AFN model (GPU)
if [ $stage -eq 11 ]; then
    eval $cmd base_pred.py \
        --eval-dir ../feat_aligned/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
        --model-path ./logs/AFN4-train/AFN4-1091-orig-model_best.pth \
        --test-batch-size 1 \
        --logging-dir ./logs/AFN4-pred-1k
fi

# prediction only on the AFN model with PRUNING (CPU)
if [ $stage -eq 20 ]; then
    for pmethod in L1Unstructured RandomUnstructured; do
        for pct in 0.{0..99..10} 1.0; do
            eval $cmd prune_pred.py \
                --eval-dir ../feat_aligned/eval \
                --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                --model-path ./logs/AFN4-train/AFN4-1091-orig-model_best.pth \
                --test-batch-size 1 \
                --logging-dir ./logs/AFN4-pred-1k \
                --prune-method ${pmethod} \
                --prune-pct ${pct} \
                --no-cuda
        done
    done
fi
# prediction only on the AFN model with PRUNING (GPU)
if [ $stage -eq 21 ]; then
    for pmethod in L1Unstructured RandomUnstructured; do
        for pct in 0.{0..99..10} 1.0; do
            eval $cmd prune_pred.py \
                --eval-dir ../feat_aligned/eval \
                --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                --model-path ./logs/AFN4-train/AFN4-1091-orig-model_best.pth \
                --test-batch-size 1 \
                --logging-dir ./logs/AFN4-pred-1k \
                --prune-method ${pmethod} \
                --prune-pct ${pct}
        done
    done
fi

# prediction only on the AFN model with QUANTIZATION (CPU) (not feasible on GPU)
# ToDo: https://pytorch.org/docs/stable/quantization.html
# ToDo: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.htmlT
if [ $stage -eq 30 ]; then
    for qmethod in dynamic; do
        eval $cmd quant_pred.py \
            --eval-dir ../feat_aligned/eval \
            --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
            --model-path ./logs/AFN4-train/AFN4-1091-orig-model_best.pth \
            --test-batch-size 1 \
            --logging-dir ./logs/AFN4-pred-1k \
            --quant-method ${qmethod}
    done
fi

# fine tuning the DECOMPOSED AFN model (this step includes training, thus using GPU)
if [ $stage -eq 40 ]; then
    for rank in 2 4; do
        for type in tucker cp; do
            echo "Decomp with rank = $rank, type = $type"
            CUDA_VISIBLE_DEVICES='0,1' python3 fine_tune_decomp.py \
                --train-dir ../feat_aligned/train \
                --train-utt2label ../data/ASVspoof2017/protocol_V2/train.utt2label \
                --validation-dir ../feat_aligned/dev \
                --validation-utt2label ../data/ASVspoof2017/protocol_V2/dev.utt2label \
                --model-path ./logs/AFN4-train/AFN4-1091-orig-model_best.pth \
                --logging-dir ./logs/AFN4-train/ \
                --epochs 30 \
                --log-interval 50 \
                --rank ${rank} \
                --decomp-type ${type}
        done
    done
fi
# prediction only on the AFN model with DECOMPOSITION (CPU)
if [ $stage -eq 41 ]; then
    for rank in 2 4; do
        for type in tucker cp; do
            eval $cmd decomp_pred.py \
                --eval-dir ../feat_aligned/eval \
                --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                --model-path ./logs/AFN4-train/fine_decomp-AFN4-1091-orig-rank_${rank}-${type}-model_best.pth \
                --test-batch-size 1 \
                --logging-dir ./logs/AFN4-pred-1k \
                --decomp-rank ${rank} \
                --decomp-type ${type} \
                --no-cuda
        done
    done
fi
# prediction only on the AFN model with DECOMPOSITION (GPU)
if [ $stage -eq 42 ]; then
    for rank in 2 4; do
        for type in tucker cp; do
            eval $cmd decomp_pred.py \
                --eval-dir ../feat_aligned/eval \
                --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                --model-path ./logs/AFN4-train/fine_decomp-AFN4-1091-orig-rank_${rank}-${type}-model_best.pth \
                --test-batch-size 1 \
                --logging-dir ./logs/AFN4-pred-1k \
                --decomp-rank ${rank} \
                --decomp-type ${type}
        done
    done
fi


# prediction only on the deformed AFN model (CPU)
if [ $stage -eq 100 ]; then
    for win in 64 128 256 512; do
        for seg in s e h l hl lh; do
            eval $cmd base_pred.py \
                --eval-dir ../feat_257_${win}/seg_${seg}/eval \
                --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                --model-path ./logs/AFN4De-train/AFN4De-${win}-${seg}-model_best.pth \
                --test-batch-size 1 \
                --logging-dir ./logs/AFN4De-pred-1k \
                --no-cuda \
                --seg ${seg} \
                --seg-win ${win}
        done
    done
fi
# prediction only on the deformed AFN model (GPU)
if [ $stage -eq 101 ]; then
    for win in 64 128 256 512; do
        for seg in s e h l hl lh; do
            eval $cmd base_pred.py \
                --eval-dir ../feat_257_${win}/seg_${seg}/eval \
                --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                --model-path ./logs/AFN4De-train/AFN4De-${win}-${seg}-model_best.pth \
                --test-batch-size 1 \
                --logging-dir ./logs/AFN4De-pred-1k \
                --seg ${seg} \
                --seg-win ${win}
        done
    done
fi

# prediction only on the deformed AFN model with PRUNING (CPU)
if [ $stage -eq 200 ]; then
    for win in 64 128 256 512; do
        for seg in s e h l hl lh; do
            for pmethod in L1Unstructured RandomUnstructured; do
                for pct in 0.{0..99..10} 1.0; do
                    eval $cmd prune_pred.py \
                        --eval-dir ../feat_257_${win}/seg_${seg}/eval \
                        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                        --model-path ./logs/AFN4De-train/AFN4De-${win}-${seg}-model_best.pth \
                        --test-batch-size 1 \
                        --logging-dir ./logs/AFN4De-pred-1k \
                        --seg ${seg} \
                        --seg-win ${win} \
                        --prune-method ${pmethod} \
                        --prune-pct ${pct} \
                        --no-cuda
                done
            done
        done
    done
fi
# prediction only on the deformed AFN model with PRUNING (GPU)
if [ $stage -eq 201 ]; then
    for win in 64 128 256 512; do
        for seg in s e h l hl lh; do
            for pmethod in L1Unstructured RandomUnstructured; do
                for pct in 0.5; do
                    eval $cmd prune_pred.py \
                        --eval-dir ../feat_257_${win}/seg_${seg}/eval \
                        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                        --model-path ./logs/AFN4De-train/AFN4De-${win}-${seg}-model_best.pth \
                        --test-batch-size 1 \
                        --logging-dir ./logs/AFN4De-pred-1k \
                        --seg ${seg} \
                        --seg-win ${win} \
                        --prune-method ${pmethod} \
                        --prune-pct ${pct}
                done
            done
        done
    done
fi

# prediction only on the deformed AFN model with QUANTIZATION (CPU) (not feasible on GPU)
if [ $stage -eq 300 ]; then
    for win in 64 128 256 512; do
        for seg in s e h l hl lh; do
            for qmethod in dynamic static; do
                eval $cmd quant_pred.py \
                    --eval-dir ../feat_257_${win}/seg_${seg}/eval \
                    --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                    --model-path ./logs/AFN4De-train/AFN4De-${win}-${seg}-model_best.pth \
                    --test-batch-size 1 \
                    --logging-dir ./logs/AFN4De-pred-1k \
                    --seg ${seg} \
                    --seg-win ${win} \
                    --quant-method ${qmethod}
            done
        done
    done
fi

# fine tuning the DECOMPOSED deformed AFN model (this step includes training, thus using GPU)
if [ $stage -eq 400 ]; then
    for win in 64 128 256 512; do
        for seg in s e h l hl lh; do
            for rank in 2 4; do
                for type in tucker cp; do
                    echo "Decomp with rank = $rank, type = $type"
                    CUDA_VISIBLE_DEVICES='0,1' python3 fine_tune_decomp.py \
                        --train-dir ../feat_257_${win}/seg_${seg}/train \
                        --train-utt2label ../data/ASVspoof2017/protocol_V2/train.utt2label \
                        --validation-dir ../feat_257_${win}/seg_${seg}/dev \
                        --validation-utt2label ../data/ASVspoof2017/protocol_V2/dev.utt2label \
                        --model-path ./logs/AFN4De-train/AFN4De-${win}-${seg}-model_best.pth \
                        --logging-dir ./logs/AFN4De-train/ \
                        --epochs 30 \
                        --log-interval 50 \
                        --seg ${seg} \
                        --seg-win ${win} \
                        --rank ${rank} \
                        --decomp-type ${type}
                done
            done
        done
    done
fi
# prediction only on the deformed AFN model with DECOMPOSITION (CPU)
if [ $stage -eq 401 ]; then
    for win in 64 128 256 512; do
        for seg in s e h l hl lh; do
            for rank in 2 4; do
                for type in tucker cp; do
                    eval $cmd decomp_pred.py \
                        --eval-dir ../feat_257_${win}/seg_${seg}/eval \
                        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                        --model-path ./logs/AFN4De-train/fine_decomp-AFN4De-${win}-${seg}-rank_${rank}-${type}-model_best.pth \
                        --test-batch-size 1 \
                        --logging-dir ./logs/AFN4-pred-1k \
                        --seg ${seg} \
                        --seg-win ${win} \
                        --decomp-rank ${rank} \
                        --decomp-type ${type} \
                        --no-cuda
                done
            done
        done
    done
fi
# prediction only on the deformed AFN model with DECOMPOSITION (GPU)
if [ $stage -eq 402 ]; then
    for win in 64 128 256 512; do
        for seg in s e h l hl lh; do
            for rank in 2 4; do
                for type in tucker cp; do
                    eval $cmd decomp_pred.py \
                        --eval-dir ../feat_257_${win}/seg_${seg}/eval \
                        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval_1k.utt2label \
                        --model-path ./logs/AFN4De-train/fine_decomp-AFN4De-${win}-${seg}-rank_${rank}-${type}-model_best.pth \
                        --test-batch-size 1 \
                        --logging-dir ./logs/AFN4-pred-1k \
                        --seg ${seg} \
                        --seg-win ${win} \
                        --decomp-rank ${rank} \
                        --decomp-type ${type}
                done
            done
        done
    done
fi