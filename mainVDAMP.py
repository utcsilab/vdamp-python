from __future__ import division

import scipy.io
import cmath
import matplotlib.pyplot as plt
from algorithms import *
from utils import *



# ## Using Shepp Logan Image
# data = scipy.io.loadmat('dcoilMine.mat')
# dcoil = np.asarray(data['dcoil'])
# mask = np.asarray(data['mask'])
# prob_map = np.asarray(data['prob_map'])
# var0 = np.asarray(data['var0'])
# x0 = np.asarray(data['x0'])
# scales = 4
# opts = {'maxIter': 30, 'maxTime': 100, 'scales': scales, 'SURE': 1, 'denoiserDiv': 1}
#
#
# result = scipy.io.loadmat('xhatMine.mat') #result from matlab (Charles Miller Code) for comparison
# xhat_result = result['x_hat']
#
#
# [xhat, hist] = vdamp(dcoil, mask, prob_map, var0, x0, opts)
#
# fig1, axs = plt.subplots(3, 1, constrained_layout = True)
# axs[0].imshow(abs(xhat))
# axs[0].set_title('result from python vdamp')
# axs[1].imshow(abs(xhat_result))
# axs[1].set_title('result from matlab vdamp')
# axs[2].imshow(abs(x0))
# axs[2].set_title('original/true image')
#
#
# fig2 = plt.figure()
# plt.imshow(abs(xhat) - abs(x0))
# fig2.suptitle('difference between original (x0) and reconstructed img')
# plt.show()




## Using BRAIN IMAGE (from compressed sensing homework)

data = np.load('brain.npz')
im, mask_unif, mask_vardens, pdf_unif, pdf_vardens = \
    data['im'], data['mask_unif'], data['mask_vardens'], data['pdf_unif'], data['pdf_vardens']

scales = 4
x0 = im
y0 = fftnc(x0)
mask = mask_vardens
prob_map = pdf_vardens
SNR = 40
var0 = (np.mean(np.abs(y0.flatten())**2)) / (10 ** (SNR/10))
sx, sy = np.shape(x0)
noise = np.sqrt(var0/2) * np.random.randn(sx, sy) +  np.sqrt(var0/2) * np.random.randn(sx, sy) * 1j
im_data = mask * (fftnc(x0) + noise)
opts = {'maxIter': 100, 'maxTime': 100, 'scales': scales, 'SURE': 1, 'denoiserDiv': 1}

[xhat, hist] = vdamp(im_data, mask, prob_map, var0, x0, opts)


fig1 = plt.figure()
plt.imshow(abs(xhat))
fig1.suptitle('result from vdamp')

fig2 = plt.figure()
plt.imshow(abs(x0))
fig2.suptitle('original/true image')

fig3 = plt.figure()
plt.imshow(abs(ifftnc(im_data)))
fig3.suptitle('data collected')


fig4 = plt.figure()
plt.imshow(abs(xhat) - abs(x0))
fig4.suptitle('difference between original (x0) and reconstructed img')
plt.show()

print('displaying wavelet')
w0 = multiscaleDecomp(x0, scales)
im_wav = pyramid(w0)
iters = np.array([1, opts['maxIter']//2, opts['maxIter']-1])

for i in range(np.size(iters)):
    it = iters[i]
    C = multiscaleDecomp(hist.Cthr[:, :, it], scales)
    im_C = pyramid(C)
    diff = abs(im_C - im_wav)
    fig1, axs = plt.subplots(3, 1, constrained_layout = True)
    axs[0].imshow(abs(im_C))
    axs[0].set_title('thresholded wavelet')
    axs[1].imshow(abs(im_wav))
    axs[1].set_title('original/true wavelet decomp')
    axs[2].imshow(diff)
    axs[2].set_title('diff')