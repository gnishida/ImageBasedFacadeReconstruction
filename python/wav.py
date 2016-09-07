import scipy
import pywt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from PIL import Image

image = mpimg.imread('lena.jpg').astype(float)[:,:,0]
#plt.imshow(image, cmap = plt.get_cmap('gray'))
#plt.show()
noiseSigma = 16.0
image += np.random.normal(0, noiseSigma, size=image.shape)
plt.imshow(image, cmap = plt.get_cmap('gray'))
plt.show()

print pywt.wavelist('bior')
#wavelet = pywt.Wavelet('haar')
wavelet = pywt.Wavelet('bior2.8')
#wavelet = pywt.Wavelet('sym15')
#wavelet = pywt.Wavelet('coif2')
levels  = int( math.floor( math.log(image.shape[0], 2) ) )

#WaveletCoeffs = pywt.wavedec2( image, wavelet, level=levels)
WaveletCoeffs = pywt.wavedec2( image, wavelet, level=None)

threshold = noiseSigma*math.sqrt(2*math.log(image.size, 2))
NewWaveletCoeffs = map (lambda x: pywt.threshold(x,threshold), WaveletCoeffs)
NewImage = pywt.waverec2( NewWaveletCoeffs, wavelet)

plt.imshow(NewImage, cmap = plt.get_cmap('gray'))
plt.show()
