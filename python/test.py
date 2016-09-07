import scipy
import pywt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from PIL import Image

x = []
f = open('../fft/image1.png_hor.txt', 'r')
for line in f:
	x.append(float(line))

wavelet = pywt.Wavelet('bior2.8')
#levels  = int( math.floor( math.log(image.shape[0], 2) ) )
WaveletCoeffs = pywt.wavedec( x, wavelet, level=None)

#threshold = noiseSigma*math.sqrt(2*math.log(image.size, 2))
threshold = 100*math.sqrt(2*math.log(len(x),2))
NewWaveletCoeffs = map (lambda x: pywt.threshold(x,threshold), WaveletCoeffs)
result = pywt.waverec( NewWaveletCoeffs, wavelet)

plt.figure(1)
plt.subplot(211)
plt.plot(x)
plt.subplot(212)
plt.plot(result)
plt.show()





'''
wavelet = pywt.Wavelet('haar')
levels = Integer(floor(log2(len(x)))
WaveletCoeffs = pywt.wavedec2(x, wavelet, level=levels)

noiseSigma = 16.0
threshold = noiseSigma*sqrt(2*log2(len(x)))
NewWaveletCoeffs = map (lambda x: pywt.thresholding.soft(x,threshold), WaveletCoeffs)
result = pywt.waverec2( NewWaveletCoeffs, wavelet)

plt.plot(result)
plt.show()
'''

'''
cA, cD = pywt.dwt(x, 'haar')
result = pywt.idwt(cA, cD, 'haar')

plt.figure(1)
plt.subplot(211)
plt.plot(x)
plt.subplot(212)
plt.plot(result)
plt.show()
'''