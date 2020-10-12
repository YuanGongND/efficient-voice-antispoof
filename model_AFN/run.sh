#!/bin/bash
# bash scripts for calling the following python scripts:
# main.py : for model training 
# feature_plot.py : for visualizing attention heatmaps 
# predict_only.py : for model forward passing 

stage="$1" # parse first argument 

if [ $stage -eq 0 ]; then
    CUDA_VISIBLE_DEVICES='0,1' python main.py \
	    --train-dir ../../feat/ASVspoof2017/train \
        --train-utt2label ../../data/ASVspoof2017/protocol_V2/train.txt.utt2label \
        --validation-dir ../../feat/ASVspoof2017/dev \
        --validation-utt2label ../../data/ASVspoof2017/protocol_V2/dev.txt.utt2label \
        --eval-dir ../../feat/ASVspoof2017/eval \
        --eval-utt2label ../data/ASVspoof2017/protocol_V2/eval.txt.utt2label \
        --logging-dir snapshots/attention/ --epochs 30 --log-interval 50
fi

# TODO: not yet adapt to the feature plot args, DON'T run stage=1
if [ $stage -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=`free-gpu` python feature_plot.py \
        --eval-scp src/data_reader/spec/new_color_map2.scp \
        --eval-utt2label src/data_reader/spec/utt2label/eval_utt2label \
        --logging-dir snapshots/predict_only/ --test-batch-size 1 \
	--plot-dir src/data_reader/plot/attention/new_colormap/shit/
fi

# TODO: not yet adapt to the predict args, DON'T run stage=2
if [ $stage -eq 2 ]; then
    CUDA_VISIBLE_DEVICES=`free-gpu` python predict_only.py \
        --eval-scp src/data_reader/spec/train_spec_cmvn_tensor.scp \
        --eval-utt2label src/data_reader/spec/utt2label/train_utt2label \
        --logging-dir snapshots/predict_only/ --test-batch-size 4 \
	--scoring-txt snapshots/scoring/train_attention8_pred.txt \
	--label-txt snapshots/scoring/train_attention8_label.txt

    CUDA_VISIBLE_DEVICES=`free-gpu` python predict_only.py \
        --eval-scp src/data_reader/spec/dev_spec_cmvn_tensor.scp \
        --eval-utt2label src/data_reader/spec/utt2label/dev_utt2label \
        --logging-dir snapshots/predict_only/ --test-batch-size 4 \
	--scoring-txt snapshots/scoring/dev_attention8_pred.txt \
	--label-txt snapshots/scoring/dev_attention8_label.txt

    CUDA_VISIBLE_DEVICES=`free-gpu` python predict_only.py \
        --eval-scp src/data_reader/spec/eval_spec_cmvn_tensor.scp \
        --eval-utt2label src/data_reader/spec/utt2label/eval_utt2label \
        --logging-dir snapshots/predict_only/ --test-batch-size 4 \
	--scoring-txt snapshots/scoring/eval_attention8_pred.txt \
	--label-txt snapshots/scoring/eval_attention8_label.txt
fi

