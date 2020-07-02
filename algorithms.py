from __future__ import division
import numpy as np
import time
import matplotlib.pyplot as plt
from waveletUtils import *
from utils import *
from wavelets import *
from threshold import *
from algorithms import *

# todo ************** BE SUPER CAREFUL ABOUT MATRIX MULT. vs. EL-by-El MULT ****************
def vdamp(dcoil, mask, prob_map, var0, x0, opts):

    nx, ny = np.shape(dcoil)

    maxIter = opts.get('maxIter', 50)
    maxTime = opts.get('maxTime', 60)
    verbose = opts.get('verbose', 0)
    scales = opts.get('scales', 4)
    saveHist = opts.get('saveHist', 0)
    sureFlag = opts.get('SURE', 1)
    denoiserDiv = opts.get('denoiserDiv', 1)
    stopDelta = opts.get('stopDelta', 0)

    if sureFlag == 0:
        sparse_weight = opts.get('lam', 0.1)

    tau = []
    lam = []
    err = []
    C_tilde = []
    alpha = []

    for s in np.arange(scales):
        tau.append(Wavelet(0, 0, 0))
        err.append(Wavelet(0, 0, 0))
        C_tilde.append(Wavelet(0, 0, 0))
        alpha.append(Wavelet(0, 0, 0))

        if sureFlag == 0:
            lam.append(Wavelet(0, 0, 0))
    tau[scales-1].a = 0
    err[scales-1].a = 0
    C_tilde[scales-1].a = 0
    alpha[scales-1].a = 0
    if sureFlag == 0:
        lam[scales-1].a = sparse_weight/100
    specX = HaarSpec(scales, nx)
    specY = HaarSpec(scales, ny)

    W0 = multiscaleDecomp(x0, scales)

    inv_p = prob_map ** (np.float(-1))
    inv_p_m1 = inv_p - 1

    m2 = inv_p * mask * dcoil

    r = ifftnc(m2)
    C = multiscaleDecomp(r, scales)

    y_cov = mask * (inv_p_m1 * inv_p * np.abs(dcoil)**2 + inv_p*var0)

    for s in np.arange(scales):
        tau[s].h = np.matmul(np.matmul(np.transpose(specX[:, s, 1]), y_cov), specY[:, s, 0])
        tau[s].v = np.matmul(np.matmul(np.transpose(specX[:, s, 0]), y_cov), specY[:, s, 1])
        tau[s].d = np.matmul(np.matmul(np.transpose(specX[:, s, 1]), y_cov), specY[:, s, 1])
    tau[scales-1].a = np.matmul(np.matmul(np.transpose(specX[:, s, 0]), y_cov), specY[:, s, 0])

    # todo if saveHist (in matlab, VDAMP line 154 - 167

    hist = history(maxIter, (nx, ny, maxIter))
    time_init = time.time()

    for iter in np.arange(maxIter):

        # todo check if verbose = true, print out VDAMP line 179

        if sureFlag:
            [C_thr_new, err, df] = multiscaleSURESoft(C, tau)
        else:
            [C_thr_new, err, df] = multiscaleComplexSoft(C, tau, lam)

        hist.Cthr[:, :, iter] = pyramid(C_thr_new)

        if stopDelta > 0 and iter > 1:
            C_thr_py = pyramid(C_thr)
            C_thr_new_py = pyramid(C_thr_new)
            A = np.linalg.norm(C_thr_py.flatten() - C_thr_new_py.flatten())
            y = np.linalg.norm(C_thr_new_py.flatten())

            dk = A / y
            if dk < stopDelta:
                print('Stopping delta reached')
                break


        C_thr = C_thr_new
        hist.timer[iter] = time.time() - time_init
        # if hist.timer[iter] > maxTime:
        #     break
        for s in np.arange(scales):
            if C_thr[s].a is not None:
                # a
                alpha[s].a = np.mean(df[s].a) / 2
                C_tilde[s].a = (C_thr[s].a - alpha[s].a * C[s].a)

                if denoiserDiv == 1:
                    # todo need to flatten each portion of the tuple individually.
                    num = np.matmul(C[s].a.flatten().transpose(), C_tilde[s].a.flatten())
                    denom = np.linalg.norm(C_tilde[s].a.flatten()) ** 2

                    # Cdiv = np.divide(num[denom > 0], denom[denom > 0])
                    # Cdiv[denom == 0] = 0
                    Cdiv = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
                else:
                    Cdiv = 1 / (1 - alpha[s].a)

                C_tilde[s].a = C_tilde[s].a * Cdiv
            # h
            alpha[s].h = np.mean(df[s].h) / 2
            C_tilde[s].h = (C_thr[s].h - alpha[s].h * C[s].h)
            if denoiserDiv == 1:
                num = np.matmul(C[s].h.flatten().transpose(), C_tilde[s].h.flatten())
                denom = np.linalg.norm(C_tilde[s].h.flatten()) ** 2
                # Cdiv = num[denom > 0] / denom[denom > 0]
                # Cdiv[denom == 0] = 0
                Cdiv = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)

            else:
                Cdiv = 1 / (1 - alpha[s].h)

            C_tilde[s].h = C_tilde[s].h * Cdiv

            # v
            alpha[s].v = np.mean(df[s].v) / 2
            C_tilde[s].v = (C_thr[s].v - alpha[s].v * C[s].v)
            if denoiserDiv == 1:
                num = np.matmul(C[s].v.flatten().transpose(), C_tilde[s].v.flatten())
                denom = np.linalg.norm(C_tilde[s].v.flatten()) ** 2
                # Cdiv = num[denom > 0] / denom[denom > 0]
                # Cdiv[denom == 0] = 0
                Cdiv = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)

            else:
                Cdiv = 1 / (1 - alpha[s].v)

            C_tilde[s].v = C_tilde[s].v * Cdiv

            # d
            alpha[s].d = np.mean(df[s].d) / 2
            C_tilde[s].d = (C_thr[s].d - alpha[s].d * C[s].d)
            if denoiserDiv == 1:
                num = np.matmul(C[s].d.flatten().transpose(), C_tilde[s].d.flatten())
                denom = np.linalg.norm(C_tilde[s].d.flatten()) ** 2
                # Cdiv = num[denom > 0] / denom[denom > 0]
                # Cdiv[denom == 0] = 0
                Cdiv = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)


            else:
                Cdiv = 1 / (1 - alpha[s].d)

            C_tilde[s].d = C_tilde[s].d * Cdiv

        r_tilde = multiscaleRecon(C_tilde)
        z = mask * (dcoil - fftnc(r_tilde))
        r = r_tilde + ifftnc(inv_p * z)
        C = multiscaleDecomp(r, scales)

        y_cov = mask * (inv_p_m1 * inv_p * abs(z) ** 2 + inv_p * var0)

        for s in np.arange(scales):
            tau[s].h = np.matmul(np.matmul(specX[:, s, 1].transpose(), y_cov), specY[:, s, 0])
            tau[s].v = np.matmul(np.matmul(specX[:, s, 0].transpose(), y_cov), specY[:, s, 1])
            tau[s].d = np.matmul(np.matmul(specX[:, s, 1].transpose(), y_cov), specY[:, s, 1])

        tau[scales-1].a = np.matmul(np.matmul(specX[:, s, 0].transpose(), y_cov), specY[:, s, 0])

    for i in np.arange(np.size(hist.timer)):

        xk_tilde = multiscaleRecon(pyramidInv(hist.Cthr[:, :, i], opts.get('scales')))
        z2 = mask * (dcoil - fftnc(xk_tilde))
        xk = ifftnc(fftnc(xk_tilde) + z2)

        # todo if saveHist ______ add components
    x_hat = xk
    return x_hat, hist