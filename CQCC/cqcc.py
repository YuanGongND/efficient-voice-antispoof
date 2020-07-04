import os
import numpy as np
import matplotlib.pyplot as plt

from cqt import cqt
from scipy.fftpack import dct
from scipy.signal import lfilter


def resample(X, Tx, Fs, P, Q, METHOD):
    pass


def deltas(x, hlen):
    win = np.array(range(-hlen, hlen+1))[::-1]
    xx = 0
    D = lfilter(win, 1, xx, axis=1)  # TODO: check this step
    D = D[:, hlen*2:]
    D = D / (2 * np.sum(np.square(range(hlen))))


def cqcc(
    x: np.array,
    fs: float,
    B: int = 96,
    fmax: float = None,
    fmin: float = None,
    d: int = 16,
    cf: int = 19,
    ZsdD: str = 'ZsdD'  # TODO: change this to input options
) -> np.array:

    if not fmax:
        fmax = fs / 2
    if not fmin:
        octa = np.ceil(np.log2(fmax / 20))
        fmin = fmax / (2 ** octa)

    gamma = 228.7 * (2 ** (1 / B) - 2 ** (-1 / B))

    c, cDC, cNyq = cqt(
        x=x,
        B=B,
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        gamma=gamma
    )

    # Log power spectrum
    absCQT = np.abs(c)
    # TimeVec = np.array(range(absCQT.shape[1])) * len(x) / absCQT.shape[1] / fs
    FreqVec = fmin * 2 ** (np.array(range(absCQT.shape[0])) / B)
    LogP_absCQT = np.log(np.square(absCQT) + np.spacing(1))

    # uniform resampling
    kl = B * np.log2(1 + 1 / d)
    Ures_LogP_absCQT, Ures_FreqVec = resample(
        LogP_absCQT,
        FreqVec,
        1/(fmin*(2**(kl/B)-1)),
        1,
        1,
        'spline'
    )  # TODO: implement this

    CQceptrum = dct(Ures_LogP_absCQT, norm='ortho')

    # TODO: deal with possible combos of ZsdD, here we assume all ZsdD are selected
    scoeff = 1
    CQceptrum_temp = CQceptrum[scoeff-1:cf+1, :]
    f_d = 1
    CQcc = np.concatenate(
        (
            CQceptrum_temp,
            deltas(CQceptrum_temp, f_d),
            deltas(deltas(CQceptrum_temp, f_d), f_d)
        ),
        axis=0
    )
    return CQcc


if __name__ == "__main__":
    import librosa
    filepath = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(filepath, "D18_1000001.wav")
    x, fs = librosa.core.load(audio_path, sr=None)

    print(len(x))
    print(fs)
    plt.plot(x)
    # plt.show()

    CQcc = cqcc(x=x, fs=fs)
