# Uncomment below for GPU desktop run
cmd="CUDA_VISIBLE_DEVICES='0,1' python3"
# Uncomment below for Jetson run
# cmd="CUDA_VISIBLE_DEVICES='0' python3"
# Uncomment below for Pi4 run
# cmd="python3"


set -e 
stage="$1" # parse 1st arg: choice of exp


# xvector
if [ $stage -eq 0 ]; then
    CUDA_VISIBLE_DEVICES='0,1' python3 main_NN.py \
            --train-file ../data/train/feat/xvectors_enroll_mfcc/xvector.txt \
            --train-utt2label ../../../data/ASVspoof2017/protocol_V2/train.utt2label \
            --validation-file ../data/dev/feat/xvectors_enroll_mfcc/xvector.txt \
            --validation-utt2label ../../../data/ASVspoof2017/protocol_V2/dev.utt2label \
            --eval-file ../data/eval/feat/xvectors_enroll_mfcc/xvector.txt \
            --eval-utt2label ../../../data/ASVspoof2017/protocol_V2/eval.utt2label \
            --dim 512 \
            --logging-dir ./logs/ \
            --epochs 30 \
            --log-interval 50
fi

# xvector
if [ $stage -eq 1 ]; then
    CUDA_VISIBLE_DEVICES='0,1' python3 main_NN.py \
            --train-file ../data/train/feat/xvectors_enroll_mfcc/xvector.txt \
            --train-utt2label ../../../data/ASVspoof2017/protocol_V2/train.utt2label \
            --validation-file ../data/dev/feat/xvectors_enroll_mfcc/xvector.txt \
            --validation-utt2label ../../../data/ASVspoof2017/protocol_V2/dev.utt2label \
            --eval-file ../data/eval/feat/xvectors_enroll_mfcc/xvector.txt \
            --eval-utt2label ../../../data/ASVspoof2017/protocol_V2/eval.utt2label \
            --dim 512 \
            --logging-dir ./logs/ \
            --epochs 30 \
            --log-interval 50
fi

# pred_only CPU
if [ $stage -eq 10 ]; then
    eval $cmd pred_only_NN.py \
        --eval-file ../data/eval/feat/xvectors_enroll_mfcc/xvector.txt \
        --eval-utt2label ../../../data/ASVspoof2017/protocol_V2/eval.utt2label \
        --model-path ./pretrained/MLP_512-model_best.pth \
        --dim 512 \
        --logging-dir ./logs/ \
        --no-cuda 
fi


# pred_only GPU
if [ $stage -eq 11 ]; then
    eval $cmd pred_only_NN.py \
        --eval-file ../data/eval/feat/xvectors_enroll_mfcc/xvector.txt \
        --eval-utt2label ../../../data/ASVspoof2017/protocol_V2/eval.utt2label \
        --model-path ./pretrained/MLP_512-model_best.pth \
        --dim 512 \
        --logging-dir ./logs/ 
fi