#!/bin/bash
# torchaudio extract logspec,  features for:
# ASVspoof2017 train, dev, eval, train_dev
# Dependency: extract_logspec.py
# Author: Jian Yang

set -e
data_dir=`pwd`/../data/ASVspoof2017
label_dir=`pwd`/../data/ASVspoof2017/protocol_V2
feat_dir=`pwd`/../feat/ASVspoof2017
seg_feat_dir=`pwd`/..
stage=$1 # parse first arg, use 0 or 1

function check_sorted {
    file=$1
    sort -k1,1 -u <$file >$file.tmp
    if ! cmp -s $file $file.tmp; then
    echo "$0: file $1 is not in sorted order or not unique, sorting it"
    mv $file.tmp $file
    else
    rm $file.tmp
    fi
}

if [ $stage -eq 0 ]; then
    # extract logspec with cmvn 
    for name in train dev eval; do
        check_sorted $label_dir/${name}.txt
        echo "start feature extraction for "${name}
        python extract_logspec.py \
            --data-dir $data_dir/${name} \
            --label-file $label_dir/${name}.txt \
            --feat-dir $feat_dir/${name} \
            --no-cuda \
            --logging-dir snapshots/feat_extraction/
    done

fi

if [ $stage -eq 1 ]; then
    # extract segmented logspec with several options of win_size and method 
    for name in train dev eval; do
        check_sorted $label_dir/${name}.txt
        for win_size in 64 128 256 512; do
            for method in h l hl lh; do
                echo "start feature extraction for "${name}"_"${win_size}"_"${method}
                python extract_logspec.py \
                    --data-dir $data_dir/${name} \
                    --label-file $label_dir/${name}.txt \
                    --feat-dir $seg_feat_dir/feat_257_${win_size}/seg_${method}/${name} \
                    --logging-dir snapshots/feat_extraction/ \
                    --no-cuda \
                    --segment \
                    --seg-win ${win_size} \
                    --seg-method ${method}
            done
        done
        
    done

fi