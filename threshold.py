from __future__ import division
import numpy as np
from wavelets import *
import copy


def multiscaleSURESoft(C, var):
    # Soft thresholding of a wavelet representation
    # IN: bands - set of 4 (# scales) wavelet objects
    # var - (TAU)estimated MSE of bands. TAU has the same structure as bands
    # lambda - sparse weighting (how do we find this?)

    # OUT: bands after soft thresholding, err - estimated MSE of bands
    bands = copy.deepcopy(C)
    scales = np.size(bands)
    df = copy.deepcopy(bands)

    err = []

    for s in np.arange(scales):
        err_a = None
        if bands[s].a is not None:
            bands[s].a, df[s].a = SUREsoft(bands[s].a, var[s].a)
            e = np.multiply(df[s].a, var[s].a)
            err_a = np.average(e) / 2

        bands[s].h, df[s].h = SUREsoft(bands[s].h, var[s].h)
        e = np.multiply(df[s].h, var[s].h)
        err_h = np.average(e) / 2

        bands[s].v, df[s].v = SUREsoft(bands[s].v, var[s].v)
        e = np.multiply(df[s].v, var[s].v)
        err_v = np.average(e) / 2

        bands[s].d, df[s].d = SUREsoft(bands[s].d, var[s].d)
        e = np.multiply(df[s].d, var[s].d)
        err_d = np.average(e) / 2
        err.append(Wavelet(err_h, err_v, err_d, err_a))

    return bands, err, df


def multiscaleComplexSoft(bands, var, lam):
    scales = np.size(bands)
    df = copy.deepcopy(bands)
    err = []
    for s in np.arange(scales):
        err_a = None
        if bands[s].a is not None:
            bands[s].a, df[s].a = complexSoft(bands[s].a, var[s].a * lam[s].a)
            e = df[s].a * var[s].a
            err_a = np.average(e) / 2

        bands[s].h, df[s].h = complexSoft(bands[s].h, var[s].h * lam[s].h)

        e = df[s].h * var[s].h
        err_h = np.average(e) / 2

        bands[s].v, df[s].v = complexSoft(bands[s].v, var[s].v * lam[s].v)
        e = df[s].v * var[s].v
        err_v = np.average(e) / 2

        bands[s].d, df[s].d = complexSoft(bands[s].d, var[s].d * lam[s].d)
        e = df[s].d * var[s].d
        err_d = np.average(e) / 2
        err.append(Wavelet(err_h, err_v, err_d, err_a))
    return bands, err, df


def SUREsoft(z, var):
    # finds threshold by SURE
    # then calls complexSoft threshold function
    # [ SURE portion to be written]
    # OUT: gz - thresholded z, df - number of degrees of freedom

    V = copy.deepcopy(var)

    z2 = z.flatten()
    z2sort = - np.sort(-(abs(z2)))
    zindx = (abs(z2)).argsort()[::-1]
    if np.size(V) <= 1:
        V0 = np.array([V])
        V = V0 * np.ones(np.size(z))
    V = V.flatten()
    V = V[zindx]
    z0 = np.ones(np.size(z2sort))
    lam = z2sort

    SURE_inf = np.flipud(np.cumsum(np.flipud(z2sort ** 2)))
    SURE_sup = np.cumsum(z0) * (lam ** 2) - lam * np.cumsum(V / z2sort) + 2 * np.cumsum(V)
    SURE = SURE_inf + SURE_sup - np.sum(V)
    idxmin = np.argmin(SURE)
    gz, df = complexSoft(z, lam[idxmin])

    return gz, df

def complexSoft(z, lam):
    # IN: z - noisy vector, lam - threshold
    # OUT: gz - thresholded z, df - deg free

    mag = np.abs(z)
    x = (lam / mag)
    gdual = x.copy()
    gdual[np.isnan(gdual)] = 1  # nan comes from dividing by zero (in line 101), we take that to be infinity
    gdual[gdual > 1] = 1
    gz = z * (1 - gdual)
    df = 2 - (2 - (gdual < 1)) * gdual

    # mag = np.abs(z)
    # gdual = (lam / mag)
    # gdualmin = gdual.copy()
    # gdualmin[np.isnan(gdualmin)] = 1  # nan comes from dividing by zero (in line 101), we take that to be infinity
    # gdual[np.isnan(gdual)] = 1
    # gdualmin[gdualmin > 1] = 1
    # gz = z * (1 - gdualmin)
    # df = 2 - (2 - (gdual < 1)) * gdualmin

    return gz, df