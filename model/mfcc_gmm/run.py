#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[COPYRIGHT info: TBA]

@File: run.py
@Author: Jian Yang
@Affiliation: University of Notre Dame
@Last Updated: 07/12/2020

python implementation of the baseline experiment for efficient voice antispoof based
on Mel-frequency cepstral coefficients (MFCC) features + Gaussian Mixture Models (GMMs)

The main function here uses the ASVspoof 2017 dataset, details can be found at:
"""

import os
import numpy as np
import pandas as pd
import librosa
from python_speech_features import mfcc
from utils import get_parent_dir


def main():

    # import matplotlib.pyplot as plt
    # filepath = os.path.dirname(os.path.abspath(__file__))
    # audio_path = os.path.join(filepath, "D18_1000001.wav")
    # x, fs = librosa.core.load(audio_path, sr=None)

    # frame_length = 0.025 # 20ms
    # frame_hop = 0.01 # 10ms
    # n_MFCC = 19 # number of cepstral coefficients excluding 0'th coefficient [default 19]
    # fl=0.002
    # fh=0.125

    # fea = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n_MFCC+1, n_fft=512)


    # print(len(x))
    # print(fs)
    # print(fea.shape)

    ################################
    #        PRE-PROCESSING        #
    ################################
    # set path
    cur_filepath = os.path.abspath(__file__)
    path_to_data = '/home/jerry/Documents/Projects/ASVspoof2017/data'  # debug only
    # path_to_data = os.path.join(get_parent_dir(cur_filepath, level=2), 'data')
    path_to_prtcl = os.path.join(path_to_data, 'protocol_V2')

    if not os.path.isdir(path_to_data):
        raise FileNotFoundError("Invalid data path: {}".format(path_to_data))

    train_prtcl_file = os.path.join(path_to_prtcl, 'ASVspoof2017_V2_train.trn.txt')
    dev_prtcl_file = os.path.join(path_to_prtcl, 'ASVspoof2017_V2_dev.trl.txt')
    eval_prtcl_file = os.path.join(path_to_prtcl, 'ASVspoof2017_V2_eval.trl.txt')

    # read inputs
    prtcl_cols = ["filename", "label", "speaker", "phrase", "env", "plb_dvc", "rec_dvc"]
    train_prtcl_df = pd.read_csv(
        train_prtcl_file, sep=" ", header=None, names=prtcl_cols
    )
    dev_prtcl_df = pd.read_csv(
        dev_prtcl_file, sep=" ", header=None, names=prtcl_cols
    )
    eval_prtcl_df = pd.read_csv(
        eval_prtcl_file, sep=" ", header=None, names=prtcl_cols
    )


    ################################
    #      FEATURE EXTRACTION      #
    ################################
    # feature extraction for GENUINE

    # feature extraction for SPOOF

    ################################
    #           TRAINING           #
    ################################
    # GMM training for GENUINE

    # GMM training for SPOOF

    ################################
    #           EVAL/TEST          #
    ################################
    # feature extraction for eval/test data

    # GMM fit & EER scoring
    print(0)


if __name__ == "__main__":
    main()
