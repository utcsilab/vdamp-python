from __future__ import division
import numpy as np
import pywt
import matplotlib.pyplot as plt
from utils import fftnc, ifftnc
from wavelets import Wavelet


def imshowgray(im, vmin=None, vmax=None):
    plt.imshow(im, cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)


def wavMask(dims, scale):
    sx, sy = dims
    res = np.ones(dims)
    NM = np.round(np.log2(dims))
    for n in range(int(np.min(NM) - scale + 2) // 2):
        res[:int(np.round(2 ** (NM[0] - n))), :int(np.round(2 ** (NM[1] - n)))] = \
            res[:int(np.round(2 ** (NM[0] - n))), :int(np.round(2 ** (NM[1] - n)))] / 2
    return res


def imshowWAV(Wim, scale=1):
    plt.imshow(np.abs(Wim) * wavMask(Wim.shape, scale), cmap=plt.get_cmap('gray'))


def coeffs2Wimg(a, coeffs):
    h, v, d = coeffs
    return np.vstack((np.hstack((a, h)), np.hstack((v, d))))


def unstack_coeffs(Wim):
    L1, L2 = np.hsplit(Wim, 2)
    a, v = np.vsplit(L1, 2)
    h, d = np.vsplit(L2, 2)
    return a, [h, v, d]


def Wimg2coeffs(Wim, scales):
    a, c = unstack_coeffs(Wim)
    coeffs = [c]
    for i in range(scales - 1):
        a, c = unstack_coeffs(a)
        coeffs.insert(0, c)
    coeffs.insert(0, a)
    return coeffs


def dwt2(im, scales):
    coeffs = pywt.wavedec2(im, wavelet='haar', mode='per', level=scales)
    Wim, rest = coeffs[0], coeffs[1:]
    for scales_r in rest:
        Wim = coeffs2Wimg(Wim, scales_r)
    return Wim


def idwt2(Wim, scales):
    coeffs = Wimg2coeffs(Wim, scales)
    return pywt.waverec2(coeffs, wavelet='haar', mode='per')

def pyramid(bands):
    # IN: BANDS
    # OUT:WAV_IMAGE (to be displayed)

    scales = np.size(bands)
    s = scales - 1  # bc zero indx
    Wim = bands[s].a

    for x in np.arange(scales):

        indx = s-x

        if np.size(Wim) == 1:
            wx = 1
            wy = 1
        else:
            wx, wy = np.shape(Wim)
            hx, hy = np.shape(bands[indx].h)
            vx, vy = np.shape(bands[indx].v)

        new = np.zeros((2 * wx, 2 * wy), dtype=complex)


        new[:wx, :wy] = Wim
        new[wx:, :wy] = bands[indx].v
        new[:wx, wy:] = bands[indx].h
        new[wx:, wy:] = bands[indx].d

        Wim = new
        # Wim = np.vstack((np.hstack((Wim, bands[indx].h)), np.hstack((bands[indx].v, bands[indx].d))))
    return Wim


def pyramidInv(Wim, scales):
    bands = []  # np.empty((scales,1))

    for s in (np.arange(scales - 1)): # iterate through first three scales
        sr, sc = np.shape(Wim)  # this should be updated with each scale
        h = Wim[:(sr // 2), (sc // 2):]
        v = Wim[sr // 2:, :(sc // 2)]
        d = Wim[sr // 2:, (sc // 2):]
        bands.append(Wavelet(h, v, d))
        Wim = Wim[:(sr // 2), :(sc // 2)]

    # finalize last scale
    sr, sc = np.shape(Wim)
    a = Wim[:(sr // 2), :(sc // 2)]
    h = Wim[:(sr // 2), (sc // 2):]
    v = Wim[sr // 2:, :(sc // 2)]
    d = Wim[sr // 2:, (sc // 2):]
    bands.append(Wavelet(h, v, d, a))

    return bands


def HaarSpec(scales, sample):
    spec = np.zeros((sample, scales, 2))
    filtLen = 1

    for s in np.arange(scales):
        filtLen = filtLen * 2

        filt = np.zeros(sample)
        filt[0:filtLen] = np.power(2, -(s+1)/2)
        spec[:, s, 0] = np.abs(fftnc(filt)) ** 2

        filt = np.zeros(sample)
        filt[0:(filtLen // 2)] = (-1) * np.power(2, -(s+1)/2)
        filt[(filtLen // 2):filtLen] = np.power(2, -(s+1)/2)

        spec[:, s, 1] = np.abs(fftnc(filt)) ** 2
    return spec
