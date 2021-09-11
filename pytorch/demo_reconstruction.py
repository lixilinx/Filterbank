import os
import matplotlib.pyplot as plt
import scipy.io
import torch
torch.set_default_tensor_type('torch.FloatTensor')
from fb_dft import Analysis, Synthesis

my_dft_fb = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'fb_design', 'my_dft_fb.mat'))['fb']
ana = Analysis(my_dft_fb)
syn = Synthesis(my_dft_fb)

x = torch.zeros([1, 1600]) # impulses as input
x[:,0] = 1
x[:,10] = -1/2
x[:,100] = 1/3
x[:,1000] = -1/4

X, ana_bfr = ana(x) # analysis
y, syn_bfr = syn(X) # synthesis
plt.plot(x[0].cpu().numpy())
plt.plot(y[0].cpu().numpy()) # output should be the same impuses
plt.legend(['original', 'reconstructed']) 
plt.title('Nearly PR')
plt.show()