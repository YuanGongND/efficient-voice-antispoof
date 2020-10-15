#!/bin/bash
# torchaudio extract logspec,  features for:
# ASVspoof2017 train, dev, eval, train_dev
# Jian Yang

set -e
data_dir=`pwd`/../data/ASVspoof2017
label_dir=`pwd`/../data/ASVspoof2017/protocol_V2
feat_dir=`pwd`/../feat_aligned/ASVspoof2017
stage=0

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
    for name in train dev eval train_dev; do
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