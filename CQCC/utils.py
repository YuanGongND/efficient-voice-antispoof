#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[COPYRIGHT info: TBA]

@File: nsgcqwin.py
@Author: Jian Yang
@Affiliation: University of Notre Dame
@Last Updated: 07/01/2020

Python implementation of utility functions for NSGCQWIN,
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
import numpy as np


def hannwin(l):
    r = np.arange(l, dtype=float)
    r *= np.pi*2./l
    r = np.cos(r)
    r += 1.
    r *= 0.5
    return r
