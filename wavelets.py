from __future__ import division
import numpy as np
import pywt


class Wavelet():
    def __init__(self, h, v, d, a=None):
        self.a = a
        self.h = h
        self.v = v
        self.d = d

    def updateVals(self, h, v, d, a=None):
        self.a = a
        self.h = h
        self.v = v
        self.d = d


# IMAGE TO BANDS

def multiscaleDecomp(im, scales):
    # OUT : bands
    # bands is an array of 4 pointers to instances of a wave/scale/band object,
    # with attributes W.a, W., W.v and W.d for A, horizontal, vertical and diagnoal

    bands = []  # np.empty((scales,1))
    a = im

    # insted of a for loop you could also do pywt.wavedec2(data, 'haar', mode='per', level=4, axes=(-2, -1))
    for s in np.arange(scales):
        coeffs = pywt.dwt2(a, 'haar')
        a, (h, v, d) = coeffs
        bands.append(Wavelet(h, v, d))

    bands[s].a = a

    return bands


# BANDS TO IMAGE

def multiscaleRecon(bands):
    # Inverse Haar wavelet transform
    # IN: array pointing to 4 (# scales) wavelet bands
    # OUT: image
    scales = np.size(bands)

    s = scales - 1
    coeffs = (bands[s].a, (bands[s].h, bands[s].v, bands[s].d))
    x = pywt.idwt2(coeffs, 'haar')

    for i in range(scales-1)[::-1]:
        coeffs = (x, (bands[i].h, bands[i].v, bands[i].d))
        x = pywt.idwt2(coeffs, 'haar')
    im = x
    # coeffs = [bands[s].a, [bands[s].h, bands[s].v, bands[s].d]]
    #
    # s = s - 1
    #
    # for i in np.arange(scales - 1):
    #     item = [bands[s - i].h, bands[s - i].v, bands[s - i].d]
    #     coeffs.append(item)
    #     # coeffs.append(bands[s - i].h)
    #     # coeffs.append(bands[s - i].v)
    #     # coeffs.append(bands[s - i].d)
    #
    # im = pywt.waverec(coeffs, wavelet='haar')

    return im


