#!/bin/bash
# This script implements a basic ivector extration
# which will be used as input for later classifiers

. ./cmd.sh
. ./path.sh
set -e

if [ $# != 1 ] ; then 
    echo "USAGE: $0 wav_path" 
    echo " e.g.: $0 ./wav" 
    exit 1;
fi 

if [ -d "./data" ];then
    rm -rf ./data
fi

#wavdir=../../data/ASVspoof2017
wavdir=$1
nnet_dir=`pwd`/exp/xvector_nnet_1a

. parse_options.sh || exit 1;

KALDI_ROOT=/home/ndmobilecomp/kaldi
DataPre=1
FIXDATA=1
FeatureForMfcc=1
VAD=1
EXTRACT=1
ToTXT=1

for name in train dev eval; do
    datadir=`pwd`/data/$name
    logdir=$datadir/log
    featdir=$datadir/feat

    if [ $DataPre -eq 1 ]; then
        echo ==========================================
        echo "get utt2spk, DataPre start on" `date`
        echo ==========================================

        python prep_data.py --wav-dir $wavdir/$name/ \
                            --label-file $wavdir/protocol_V2/$name.utt2label \
                            --out-dir $datadir
        utils/utt2spk_to_spk2utt.pl $datadir/utt2spk > $datadir/spk2utt || exit 1
        utils/spk2utt_to_utt2spk.pl $datadir/spk2utt > $datadir/utt2spk || exit 1

        echo ===== data preparation finished successfully `date`==========
    fi

    if [ $FIXDATA -eq 1 ]; then
        echo ==========================================
        echo "sorted spk2utt ... : fix_data_dir start on" `date`
        echo ==========================================
        utils/fix_data_dir.sh $datadir
        echo ====== fix_data_dir $name finished successfully `date` ==========
    fi

    if [ $FeatureForMfcc -eq 1 ]; then
        echo ==========================================
        echo "FeatureForSpeaker start on" `date`
        echo ========================================== 
        # Extract speaker features MFCC.
        steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 1 --cmd "$train_cmd" \
        $datadir $logdir/make_enrollmfcc $featdir/mfcc
        utils/fix_data_dir.sh $datadir
        echo ==== FeatureForSpeaker $name test successfully `date` ===========
    fi

    if [ $VAD -eq 1 ];then
        echo ==========================================
        echo "generate vad file in data/train, VAD start on" `date`
        echo ==========================================
        # Compute VAD decisions. These will be shared across both sets of features.
        sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
        $datadir $logdir/make_enrollvad $featdir/vad

        utils/fix_data_dir.sh $datadir
        
        echo ========== $name VAD test successfully `date` ===============
    fi

    if [ $EXTRACT -eq 1 ]; then
        echo ==========================================
        echo "EXTRACT start on" `date`
        echo ==========================================
        # Extract the xVectors
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd" --nj 1 \
        $nnet_dir $datadir $featdir/xvectors_enroll_mfcc
        
        echo ========= EXTRACT just for testing `date`=============
    fi

    if [ $ToTXT -eq 1 ]; then
        echo ==========================================
        echo "ToTXT start on" `date`
        echo ==========================================
        # transform the ark file to txt file
        $KALDI_ROOT/src/bin/copy-vector ark:$datadir/feat/xvectors_enroll_mfcc/xvector.1.ark ark,t:- >$datadir/feat/xvectors_enroll_mfcc/xvector.txt
        echo ========= ToTXT $name success `date`=============
    fi
done