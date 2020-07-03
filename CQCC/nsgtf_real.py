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
from math import ceil
from itertools import chain

from utils import chkM
from fft import fftp, ifftp


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

    c = np.empty(shape=[N,])

    for ii in range(N):
        idx_iter = chain(
            range(int(np.ceil(Lg[ii]/2)), Lg[ii]),
            range(0, int(np.ceil(Lg[ii]/2)))
        )
        idx = list(idx_iter)
        win_range = np.mod(
            posit[ii] + np.array(range(int(-np.floor(Lg[ii]/2)-1), int(np.ceil(Lg[ii]/2)-1))),
            Ls+fill
        ) + 1

        if M[ii] < Lg[ii]:
            col = int(np.ceil(Lg[ii]/M[ii]))
            temp = np.zeros(int(col*M[ii]), )
            idx_tmp = [
                end-floor(Lg(ii)/2)+1:end,
                1:ceil(Lg(ii)/2)
            ]
            temp[   [end-floor(Lg(ii)/2)+1:end,1:ceil(Lg(ii)/2)],:    ]
        else:

    print(0)

    '''
for ii = 1:N
    idx = [ceil(Lg(ii)/2)+1:Lg(ii),1:ceil(Lg(ii)/2)];
    win_range = mod(posit(ii)+(-floor(Lg(ii)/2):ceil(Lg(ii)/2)-1),...
        Ls+fill)+1;
    
    if M(ii) < Lg(ii) % if the number of frequency channels is too small,
        % aliasing is introduced (non-painless case)
        col = ceil(Lg(ii)/M(ii));
        temp = zeros(col*M(ii),CH);
        
        temp([end-floor(Lg(ii)/2)+1:end,1:ceil(Lg(ii)/2)],:) = ...
            bsxfun(@times,f(win_range,:),g{ii}(idx));
        temp = reshape(temp,M(ii),col,CH);
        
        c{ii} = squeeze(ifft(sum(temp,2)));
        % Using c = cellfun(@(x) squeeze(ifft(x)),c,'UniformOutput',0);
        % outside the loop instead does not provide speedup; instead it is
        % slower in most cases.
    else
        temp = zeros(M(ii),CH);
        temp([end-floor(Lg(ii)/2)+1:end,1:ceil(Lg(ii)/2)],:) = ...
            bsxfun(@times,f(win_range,:),g{ii}(idx));
        
        if strcmp(phasemode,'global')
            %apply frequency mapping function (see cqt)
            fsNewBins = M(ii);
            fkBins = posit(ii);
            displace = fkBins - floor(fkBins/fsNewBins) * fsNewBins;
            temp = circshift(temp, displace);
        end
        
        c{ii} = ifft(temp);
%         c{ii} = c{ii}.* ( 2* M(ii)/Lg(ii) ); %energy normalization
    end
end
    '''
    print(0)
