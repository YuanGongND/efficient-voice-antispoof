#!/bin/bash
# bash scripts for calling the following python scripts:
# main.py : for model training 
# feature_plot.py : for visualizing attention heatmaps 
# predict_only.py : for model forward passing 

stage="$1" # parse first argument 

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

