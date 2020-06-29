# -*- coding: utf-8 -*-
# @Time    : 2/26/20 6:27 PM
# @Author  : Yuan Gong
# @Affiliation  : University of Notre Dame
# @Email   : yuangongfdu@gmail.com 
# @File    : compute_cqcc.py

import os
import librosa
import constants
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile

def deltas(x, hlen):
    wi

"""
Input parameters:
x : input signal
fs : sampling frequency
B : number of bins per octave [default = 96]
fmax : highest frequency to be analyzed [default = Nyquist frequency]
fmin : lowest frequency to be analyzed [default = ~20Hz to fullfill an integer number of octave]
d : number of uniform samples in the first octave [default 16]
cf : number of cepstral coefficients excluding 0'th coefficient [default 19]
ZsdD : any sensible combination of the following  [default ZsdD]:
'Z'  include 0'th order cepstral coefficient
's'  include static coefficients (c)
'd'  include delta coefficients (dc/dt)
'D'  include delta-delta coefficients (d^2c/dt^2)

Output parameters:
CQcc : constant Q cepstral coefficients (nCoeff x nFea)
LogP_absCQT : log power magnitude spectrum of constant Q trasform
TimeVec : time at the centre of each frame [sec]
FreqVec : center frequencies of analysis filters [Hz]
Ures_LogP_absCQT : uniform resampling of LogP_absCQT
Ures_FreqVec : uniform resampling of FreqVec [Hz]
"""
def cqcc(x, fs, B=96, fmax=None, fmin=None, d=16, cf=19, ZsdD='ZsdD'):
    if fmax == None:
        fmax = fs / 2
    if fmin == None:
        oct = math.ceil(math.log2(fmax / 20))
        fmin = fmax / (2 ** oct)

    gamma = 228.7 * (2 ** (1 / B) - 2 ** (-1 / B))

    Xcq = librosa.cqt(x, sr=sr, fmin=fmin, hop_length=128, n_bins=96 * 9, bins_per_octave=96)
    absCQT = abs(Xcq)
    FreqVec = [fmin * (2 ** (x / B)) for x in list(range(absCQT.shape[0]))]

    kl = (B * math.log2(1 + 1 / d))

    LogP_absCQT = np.log(np.square(absCQT) + 2.22e-16)
    print(FreqVec[0])
    print(1/(fmin * (2 ** (kl / B) - 1)))
    tt = librosa.core.resample(LogP_absCQT[0,:], 1/(fmin * (2 ** (kl / B) - 1)), FreqVec[0])
    Ures_LogP_absCQT = [librosa.core.resample(LogP_absCQT[i,:], FreqVec[i], 1/(fmin * (2 ** (kl / B) - 1))) for i in range(len(FreqVec))]

    #Xcq = cqt(x, B, fs=fs, fmin=fmin, fmax, 'rasterize', 'full', 'gamma', gamma)
    Xcq_len = len(x)

    # absCQT = abs(Xcq);
    # TimeVec = (1:size(absCQT, 2)) * Xcq_xlen / size(absCQT, 2) / fs;
    # FreqVec = fmin * (2. ^ ((0:size(absCQT, 1) - 1) / B));
    # LogP_absCQT = log(absCQT. ^ 2 + eps);

if __name__ == '__main__':
    audio_path = os.path.join(constants.DUMMY_PATH, 'D18_1000001.wav')

y, sr = librosa.core.load(audio_path, sr=None)
#sr, y = wavfile.read(audio_path)
print(len(y))
print(sr)
plt.plot(y)
#
cc = librosa.cqt(y, sr=sr, fmin=15.6250, hop_length=128, n_bins=96 * 9, bins_per_octave=96)
B=96
d=16
cf=19
ZsdD='ZsdD'
fs=sr
fmax=None
fmin=None
x = y
if fmax == None:
    fmax = fs / 2
if fmin == None:
    oct = math.ceil(math.log2(fmax / 20))
    fmin = fmax / (2 ** oct)

gamma = 228.7 * (2 ** (1 / B) - 2 ** (-1 / B))

Xcq = librosa.cqt(x, sr=sr, fmin=fmin, hop_length=128, n_bins=96 * 9, bins_per_octave=96)
absCQT = abs(Xcq)
FreqVec = [fmin * (2 ** (x / B)) for x in list(range(absCQT.shape[0]))]

kl = (B * math.log2(1 + 1 / d))

LogP_absCQT = np.log(np.square(absCQT) + 2.22e-16)
print(FreqVec[0])
print(1 / (fmin * (2 ** (kl / B) - 1)))
test_abs = np.asfortranarray(LogP_absCQT[0, :])
tt = librosa.core.resample(test_abs, 1 / (fmin * (2 ** (kl / B) - 1)), FreqVec[0])
np.transpose(LogP_absCQT)
Ures_LogP_absCQT = [librosa.core.resample(LogP_absCQT[:, i], 1 / (fmin * (2 ** (kl / B) - 1)), FreqVec[i]) for i in
                    range(len(FreqVec))]

# Xcq = cqt(x, B, fs=fs, fmin=fmin, fmax, 'rasterize', 'full', 'gamma', gamma)
Xcq_len = len(x)