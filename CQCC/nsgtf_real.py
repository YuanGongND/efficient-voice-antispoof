#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[COPYRIGHT info: TBA]

@File: nsgtf_real.py
@Author: Jian Yang
@Affiliation: University of Notre Dame
@Last Updated: 07/02/2020

Python implementation of ,
derived from MATLAB code by XXXX.
Original Matlab code avaialable at:
https://github.com/azraelkuan/asvspoof2017/tree/master/baseline/CQCC_v1.0/CQT_toolbox_2013

--
Original matlab code copyright follows:

"""

import numpy as np
from itertools import chain
from utils import chkM


def nsgtf_real(
    f: np.array,
    g: np.array,
    shift: np.array,
    M: np.array,
    phasemode: str = 'global'
) -> np.array:
    Ls = f.shape[0]
    CH = 1  # TODO: add support for multi-channel

    if CH > Ls:
        # TODO: fill this according to Matlab code
        pass

    N = len(shift)
    M = chkM(M, g)

    f = np.fft.fft(f)

    posit = np.cumsum(shift)-shift[0]

    fill = np.sum(shift) - Ls
    if fill > 0:
        f = np.append(f, np.zeros((int(fill), ), dtype=float))

    Lg = np.array([len(x) for x in g])
    tmp_arr = posit - np.floor(Lg/2) <= (Ls + fill) / 2
    find_N = np.argwhere(tmp_arr == True)
    if len(find_N) > 0:
        N = find_N[-1][0] + 1

    c = np.empty(shape=[N,], dtype=object)

    for ii in range(N):
        idx_iter = chain(
            range(int(np.ceil(Lg[ii]/2)), Lg[ii]),
            range(0, int(np.ceil(Lg[ii]/2)))
        )
        idx = list(idx_iter)
        win_range = (np.mod(
            posit[ii] + np.array(range(int(-np.floor(Lg[ii]/2)), int(np.ceil(Lg[ii]/2)))),
            Ls+fill
        )).astype(int)

        if M[ii] < Lg[ii]:
            col = int(np.ceil(Lg[ii]/M[ii]))
            temp = np.zeros((int(col*M[ii]), ), dtype=complex)
            idx_tmp = list(
                chain(
                    # end-floor(Lg(ii)/2)+1:end,
                    range(len(temp)-int(np.floor(Lg[ii]/2)), len(temp)),
                    # 1:ceil(Lg(ii)/2)
                    range(0, int(np.ceil(Lg[ii]/2)))
                )
            )
            temp[idx_tmp] = f[win_range] * g[ii][idx]
            # TODO: may have bugs for the following two lines
            temp = np.reshape(temp, (M[ii], col))
            c[ii] = np.squeeze(np.fft.ifft(np.sum(temp, 1)))
        else:
            temp = np.zeros((int(M[ii]), ), dtype=complex)
            idx_tmp = list(
                chain(
                    # end-floor(Lg(ii)/2)+1:end,
                    range(len(temp)-int(np.floor(Lg[ii]/2)), len(temp)),
                    # 1:ceil(Lg(ii)/2)
                    range(0, int(np.ceil(Lg[ii]/2)))
                )
            )
            temp[idx_tmp] = np.multiply(f[win_range], g[ii][idx])

            if phasemode == 'global':
                fsNewBins = M[ii]
                fkBins = posit[ii]
                displace = fkBins - np.floor(fkBins/fsNewBins) * fsNewBins
                temp = np.roll(temp, int(displace))

            c[ii] = np.fft.ifft(temp)

    return c
