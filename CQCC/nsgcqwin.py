#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[COPYRIGHT info: TBA]

@File: nsgcqwin.py
@Author: Jian Yang
@Affiliation: University of Notre Dame
@Last Updated: 07/02/2020

Python implementation of NSGCQWIN: Constant-Q/Variable-Q dictionary generator,
derived from MATLAB code by NUHAG, University of Vienna, Austria.
Original Matlab code avaialable at:
https://github.com/azraelkuan/asvspoof2017/tree/master/baseline/CQCC_v1.0/CQT_toolbox_2013

--
Original matlab code copyright follows:

AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.
"""
from typing import List
import numpy as np
import math

import utils


BWFAC = 1
MIN_WIN = 4
FRACTIONAL = False


def nsgcqwin(
    fmin: float,
    fmax: float,
    bins: int,
    sr: int,
    Ls: int,
    gamma: float,
    min_win: int = MIN_WIN
) -> List:

    nf = sr / 2
    if fmax > nf:
        fmax = nf

    fftres = sr / Ls
    b = math.floor(bins * math.log2(fmax / fmin))
    fbas = fmin * 2 ** (np.arange(b) / bins)

    Q = 2 ** (1 / bins) - 2 ** (-1 / bins)
    cqtbw = Q * fbas + gamma
    np.transpose(cqtbw)

    tmp_arr = (fbas + cqtbw / 2 > nf)
    tmpIdx = np.argwhere(tmp_arr == True)
    if len(tmpIdx) > 0:
        fbas = fbas[0:tmpIdx[0][0]]
        cqtbw = cqtbw[0:tmpIdx[0][0]]

    tmp_arr = (fbas - cqtbw / 2 < 0)
    tmpIdx = np.argwhere(tmp_arr == True)
    if len(tmpIdx) > 0:
        fbas = fbas[tmpIdx[-1][0]+1:]
        cqtbw = cqtbw[tmpIdx[-1][0]+1:]
        print("fmin set to " + str(fftres * math.floor(fbas[0]/fftres) + "Hz!!"))

    Lfbas = len(fbas)
    fbas = np.insert(fbas, 0, 0)
    fbas = np.append(fbas, nf)
    orig_fbas = fbas[1:-1]
    fbas = np.concatenate((fbas, sr - orig_fbas[::-1]), axis=0)

    bw = np.concatenate(
        (
            np.array([2 * fmin]),
            cqtbw,
            np.array([fbas[Lfbas+2] - fbas[Lfbas]]),
            cqtbw[::-1]
        ),
        axis=0
    )
    bw = bw / fftres
    fbas = fbas / fftres

    posit = np.zeros(len(fbas))
    posit[0:Lfbas+2] = np.floor(fbas[0:Lfbas+2])
    posit[Lfbas+2:] = np.ceil(fbas[Lfbas+2:])

    shift = np.concatenate(
        (np.array([np.mod(-posit[-1], Ls)]), np.diff(posit)),
        axis=0
    )

    # TODO: add fractional as input args.
    if FRACTIONAL:
        corr_shift = fbas - posit
        M = np.ceil(bw + 1)
    else:
        bw = np.round(bw)
        M = bw

    for i in range(2 * (Lfbas + 1)):
        if bw[i] < min_win:
            bw[i] = min_win
            M[i] = bw[i]

    if FRACTIONAL:
        # TODO: fill this according to Matlab code
        raise ValueError("the case \"FRACTIONAL = 1\" has not yet been implemented!")
    else:
        # TODO: set hannwin as input options among multiple windown functions
        g = np.array([utils.hannwin(x) for x in bw])

    M = BWFAC * np.ceil(M / BWFAC)  # TODO: add BWFAC as an input arg

    # set up Tukey window for 0- and Nyquist freq
    for j in [0, Lfbas+1]:
        if M[j] > M[j + 1]:
            g[j] = np.ones((int(M[j]),), dtype=float)
            idx_l = int(np.floor(M[j] / 2) - np.floor(M[j+1] / 2))
            idx_r = int(np.floor(M[j] / 2) + np.ceil(M[j+1] / 2))
            g[j][idx_l:idx_r] = np.array([utils.hannwin(M[j+1])])
            g[j] = g[j] / np.sqrt(M[j])

    return g, shift, M


# The following main function is used for test run only
if __name__ == "__main__":
    B = 96
    gamma = 228.7 * (2 ** (1 / B) - 2 ** (-1 / B))

    g, shift, M = nsgcqwin(
        fmin=15.625,
        fmax=8000,
        bins=B,
        sr=16000,
        Ls=64244,
        gamma=gamma
    )

    print(g.shape)
    print(shift.shape)
    print(M.shape)
