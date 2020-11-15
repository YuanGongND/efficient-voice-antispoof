#!/bin/bash
# bash scripts for calling the following python scripts:
# main.py : for model training 
# feature_plot.py : for visualizing attention heatmaps 
# predict_only.py : for model forward passing 

set -e
stage="$1" # parse first argument 
compress="$2"  # parse 2nd arg
percentage="$3"  # parse 3rd arg

# run train+validation+eval
if [ $stage -eq 0 ]; then
    CUDA_VISIBLE_DEVICES='0,1' python main.py \
	    --train-dir ../feat_aligned/ASVspoof2017/train \
        --train-utt2label ../data/ASVspoof2017/protocol_V2/train.txt.utt2label \
        --validation-dir ../feat_aligned/ASVspoof2017/dev \
        --validation-utt2label ../data/ASVspoof2017/protocol_V2/dev.txt.utt2label \
        --eval-dir ../feat_aligned/ASVspoof2017/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir snapshots/attention/ \
        --epochs 30 \
        --log-interval 50 \
        --seed 1
fi

# run train+validation+eval with half size feature 257*512
if [ $stage -eq 512 ]; then
for i in 1 2 3 4 5; do
    echo "round "$i
    CUDA_VISIBLE_DEVICES='0,1' python main.py \
	    --train-dir ../feat_257_512/ASVspoof2017/train \
        --train-utt2label ../data/ASVspoof2017/protocol_V2/train.txt.utt2label \
        --validation-dir ../feat_257_512/ASVspoof2017/dev \
        --validation-utt2label ../data/ASVspoof2017/protocol_V2/dev.txt.utt2label \
        --eval-dir ../feat_257_512/ASVspoof2017/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir snapshots/attention/ \
        --epochs 30 \
        --log-interval 50 \
        --seed 1
done
fi

# run train+validation+eval with half size feature 257*256
if [ $stage -eq 256 ]; then
    CUDA_VISIBLE_DEVICES='0,1' python main.py \
	    --train-dir ../feat_257_256/ASVspoof2017/train \
        --train-utt2label ../data/ASVspoof2017/protocol_V2/train.txt.utt2label \
        --validation-dir ../feat_257_256/ASVspoof2017/dev \
        --validation-utt2label ../data/ASVspoof2017/protocol_V2/dev.txt.utt2label \
        --eval-dir ../feat_257_256/ASVspoof2017/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir snapshots/attention/ \
        --epochs 30 \
        --log-interval 50 \
        --seed 1
fi

# run train+validation+eval with half size feature 257*128
if [ $stage -eq 128 ]; then
for i in 1 2 3 4 5; do
    echo "round "$i
    CUDA_VISIBLE_DEVICES='0,1' python main.py \
	    --train-dir ../feat_257_128/ASVspoof2017/train \
        --train-utt2label ../data/ASVspoof2017/protocol_V2/train.txt.utt2label \
        --validation-dir ../feat_257_128/ASVspoof2017/dev \
        --validation-utt2label ../data/ASVspoof2017/protocol_V2/dev.txt.utt2label \
        --eval-dir ../feat_257_128/ASVspoof2017/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir snapshots/attention/ \
        --epochs 30 \
        --log-interval 50 \
        --seed 1
done
fi

# run prediction using pretrained model and plot features
if [ $stage -eq 1 ]; then
    CUDA_VISIBLE_DEVICES='0,1' python feature_plot.py \
        --eval-dir ../feat_aligned/ASVspoof2017/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir snapshots/feat_plot/ \
        --test-batch-size 1 \
	    --plot-dir ../plot/attention/
fi

# run prediction on every dataset and save eer scoring
# rename --scoreing-txt and --lable-txt if change model
# NO network compression applied
if [ $stage -eq 2 ]; then
    CUDA_VISIBLE_DEVICES='0,1' python predict_only.py \
        --eval-dir ../feat_aligned/ASVspoof2017/train \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/train.txt.utt2label \
        --logging-dir snapshots/predict_only/ \
        --test-batch-size 4 \
        --scoring-txt snapshots/scoring/train_att4_1rep_pred.txt \
        --label-txt snapshots/scoring/train_att4_1rep_label.txt

    CUDA_VISIBLE_DEVICES='0,1' python predict_only.py \
        --eval-dir ../feat_aligned/ASVspoof2017/dev \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/dev.txt.utt2label \
        --logging-dir snapshots/predict_only/ \
        --test-batch-size 4 \
	    --scoring-txt snapshots/scoring/dev_att4_1rep_pred.txt \
	    --label-txt snapshots/scoring/dev_att4_1rep_label.txt

    CUDA_VISIBLE_DEVICES='0,1' python predict_only.py \
        --eval-dir ../feat_aligned/ASVspoof2017/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir snapshots/predict_only/ \
        --test-batch-size 4 \
	    --scoring-txt snapshots/scoring/eval_att4_1rep_pred.txt \
	    --label-txt snapshots/scoring/eval_att4_1rep_label.txt
fi

# similar to stage = 2, add extra args to support auto run for network compression
if [ $stage -eq 3 ]; then
    CUDA_VISIBLE_DEVICES='0,1' python predict_only.py \
        --eval-dir ../feat_aligned/ASVspoof2017/train \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/train.txt.utt2label \
        --logging-dir snapshots/predict_only/ \
        --test-batch-size 1 \
        --compress $compress \
        --prune-pct $percentage

    CUDA_VISIBLE_DEVICES='0,1' python predict_only.py \
        --eval-dir ../feat_aligned/ASVspoof2017/dev \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/dev.txt.utt2label \
        --logging-dir snapshots/predict_only/ \
        --test-batch-size 1 \
        --compress $compress \
        --prune-pct $percentage

    CUDA_VISIBLE_DEVICES='0,1' python predict_only.py \
        --eval-dir ../feat_aligned/ASVspoof2017/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir snapshots/predict_only/ \
        --test-batch-size 1 \
        --compress $compress \
        --prune-pct $percentage
fi

# similar to stage = 3, only use CPU
if [ $stage -eq 4 ]; then
    CUDA_VISIBLE_DEVICES='0,1' python predict_only.py \
        --eval-dir ../feat_aligned/ASVspoof2017/train \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/train.txt.utt2label \
        --logging-dir snapshots/predict_only/ \
        --test-batch-size 1 \
        --compress $compress \
        --prune-pct $percentage \
        --no-cuda

    CUDA_VISIBLE_DEVICES='0,1' python predict_only.py \
        --eval-dir ../feat_aligned/ASVspoof2017/dev \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/dev.txt.utt2label \
        --logging-dir snapshots/predict_only/ \
        --test-batch-size 1 \
        --compress $compress \
        --prune-pct $percentage \
        --no-cuda

    CUDA_VISIBLE_DEVICES='0,1' python predict_only.py \
        --eval-dir ../feat_aligned/ASVspoof2017/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir snapshots/predict_only/ \
        --test-batch-size 1 \
        --compress $compress \
        --prune-pct $percentage \
        --no-cuda
fi