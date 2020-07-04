#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[COPYRIGHT info: TBA]

@File: nsgcqwin.py
@Author: Jian Yang
@Affiliation: University of Notre Dame
@Last Updated: 07/02/2020

Python implementation of CQT: Constant-Q/Variable-Q transform,
derived from MATLAB code by XXXX.
Original Matlab code avaialable at:
https://github.com/azraelkuan/asvspoof2017/tree/master/baseline/CQCC_v1.0/CQT_toolbox_2013

--
Original matlab code copyright follows:

"""
from typing import List
import numpy as np
import math
from nsgcqwin import nsgcqwin
from nsgtf_real import nsgtf_real


GAMMA = 0


def cqt(
    x: np.array,
    B: int,
    fs: float,
    fmin: float,
    fmax: float,
    gamma: float = GAMMA
) -> List:

    # window design
    g, shift, M = nsgcqwin(fmin=fmin, fmax=fmax, bins=B, sr=fs, Ls=len(x), gamma=gamma)
    fbas = fs * np.cumsum(shift[1:]) / len(x)
    fbas = fbas[0:int(len(M)/2-1)]

    # compute coefficients
    bins = int(len(M) / 2 - 1)
    # TODO: add options for rasterize: 1) full; 2) piecewise
    # use full rasterize here as default
    M[1:bins+1] = M[bins]
    M[bins+2:] = M[1:bins+1][::-1]

    # TODO: add options for normalize as Matlab code does
    # use sine for normalizr here as default
    normFacVec = 2 * M[0: bins+2] / len(x)
    normFacVec = np.concatenate(
        (normFacVec, normFacVec[1:-1][::-1]),
        axis=0
    )

    g = np.array(
        [g[x] * normFacVec[x] for x in range(2 * bins + 2)]
    )

    c = nsgtf_real(f=x, g=g, shift=shift, M=M, phasemode='global')

    cDC = np.transpose(c[0])
    cNyq = np.transpose(c[bins+1])
    c = c[1:(bins+1)]
    # TODO: c is an np array of np array, better flat to matrix
    c = np.array([[i for i in x] for x in c])

    return c, cDC, cNyq


# the following main func is used for test run only
if __name__ == "__main__":
    import os
    import librosa
    filepath = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(filepath, "D18_1000001.wav")
    x, fs = librosa.core.load(audio_path, sr=None)

    print("length of raw audio data: " + str(len(x)))
    print("raw sampling rate: " + str(fs))

    # params
    B = 96  # num of bins per octave
    fs = fs  # sampling freq
    fmax = fs / 2  # highest freq to be analyzed default Nyquist

    octa = math.ceil(math.log2(fmax / 20))
    fmin = fmax / (2 ** octa)  # lowest freq to be analyzed, default ~20Hz to fulfill a octave

    d = 16  # num of uniform samples in the 1st octave
    cf = 19  # num of cepstral coef excluding 0'th coef

    gamma = 228.7 * (2 ** (1 / B) - 2 ** (-1 / B))

    c, cDC, cNyq = cqt(
        x=x,
        B=B,
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        gamma=gamma
    )
