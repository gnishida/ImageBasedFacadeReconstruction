import pywt
import matplotlib.pyplot as plt
import numpy as np

x = []
f = open('../fft/image1.png_ver.txt', 'r')
for line in f:
	x.append(float(line))

x = np.array(x)
z = np.polyfit(xrange(len(x)), x, 30)
p = np.poly1d(z)

result = []
for i in xrange(len(x)):
	result.append(p(i))

plt.figure(1)
plt.subplot(211)
plt.plot(x)
plt.subplot(212)
plt.plot(result)
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