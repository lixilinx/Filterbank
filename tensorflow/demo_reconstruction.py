import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from fb_dft import Analysis, Synthesis

my_dft_fb = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'fb_design', 'my_dft_fb.mat'))['fb']
ana = Analysis(my_dft_fb)
syn = Synthesis(my_dft_fb)

x = np.zeros([1, 1600])
x[:,0] = 1
x[:,10] = -1/2
x[:,100] = 1/3
x[:,1000] = -1/4

X, ana_bfr = ana(x) # analysis
y, syn_bfr = syn(X) # synthesis
plt.plot(x[0])
plt.plot(y[0]) # output should be impuses as well
plt.legend(['original', 'reconstructed'])
plt.title('Nearly PR')
plt.show()