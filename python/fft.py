import matplotlib.pyplot as plt
import numpy as np

x = []
f = open('../fft/image19.png_hor.txt', 'r')
for line in f:
	x.append(float(line))

x = np.array(x)
y = np.fft.fft(x)

z = np.fft.ifft(y[0:100])

plt.figure(1)
plt.subplot(311)
plt.plot(y)
plt.subplot(312)
plt.plot(x)
plt.subplot(313)
plt.plot(z)
plt.show()

'''
cA, cD = pywt.dwt(x, 'db1')
result = pywt.idwt(cA, cD, 'db1')

plt.figure(1)
plt.subplot(211)
plt.plot(x)
plt.subplot(212)
plt.plot(result)
plt.show()
'''