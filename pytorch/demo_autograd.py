import os
import scipy.io
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from fb_dft import Analysis, Synthesis

my_dft_fb = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'fb_design', 'my_dft_fb.mat'))['fb']
ana = Analysis(my_dft_fb)
syn = Synthesis(my_dft_fb)
ana_bfr, syn_bfr = None, None

x = torch.randn(10, 160000)
x[:, -16000:] = 0.0
x.requires_grad = True

E = 0.0
for i in range(10):
    X, ana_bfr = ana(x[:,i*16000:(i+1)*16000], ana_bfr)
    y, syn_bfr = syn(X, syn_bfr)
    E += torch.sum(y*y)
dE_dx = torch.autograd.grad(E, x)[0]
print('Should have dE_dx/2=x. The relative abs error is: {}'.format((torch.norm(dE_dx/2-x)/torch.norm(x)).item()))