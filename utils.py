from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def ifftnc(k):
    x = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(k)))
    x = np.sqrt(np.size(k)) * x
    return x


def fftnc(k):
    k = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(k)))
    k = k / (np.sqrt(np.size(k)))

    return k


def genPDF(imSize, p, pctg, distType, radius, disp):
    # p - power of polynomial, pctg - sampling factor, radius - radius of fully sampled center
    # assume only 2D image (no 1D pdf)
    # default distType=2, radius = 0, disp =0
    minval = 0
    maxval = 1
    val = 0.5
    # may want to define this outside of this fxn
    # if (imSize ==1)
    # imSize = ones(2) # imsize is vector [x,y], initialize to be [1,1]
    sx = imSize(1)
    sy = imSize(2)
    PCTG = np.floor(pctg * sx * sy)

    r = 1  # Is this the right base value?
    if sx != 1 and sy != 1:
        x, y = np.meshgrid(np.linspace(-1, 1, sy), np.linspace(-1, 1, sx))

        # should also check for distType, but we're assuming L2
        r = np.sqrt(x ** 2 + y ** 2)  # distance to every point on

    idx = np.where(r < radius)
    pdf = (1 - r) ** p

    if np.floor(np.sum(pdf)) > PCTG:
        print('increase p')

    while 1:
        val = minval / 2 + maxval / 2
        pdf = (1 - r) ** p + val
        # IF all are greater than 0
        np.clip(pdf, 0, 1)  # cap out values at 1
        pdf[idx] = 1  # inside radius = 1
        N = np.floor(np.sum(pdf))
        if N > PCTG:
            maxval = val
        if N < PCTG:
            minval = val
        if N == PCTG:
            break

    if disp:
        plt.imshow(pdf)

    return pdf

class history():
    def __init__(self, maxIter, CthrShape,  *args):

        self.timer = np.zeros(maxIter)
        self.Cthr = np.zeros(CthrShape, dtype=complex)
        self.x_mse = np.zeros(maxIter)

        # todo add the optional vars when saveHist = 1
