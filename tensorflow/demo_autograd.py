import os
import scipy.io
import tensorflow as tf
from fb_dft import Analysis, Synthesis

my_dft_fb = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'fb_design', 'my_dft_fb.mat'))['fb']
ana = Analysis(my_dft_fb)
syn = Synthesis(my_dft_fb)

#@tf.function
def f():
    x = tf.concat([tf.random.normal([10, 9*16000]), tf.zeros([10, 1*16000])], axis=-1);
    E = 0.0
    ana_bfr, syn_bfr = None, None
    with tf.GradientTape() as g:
        g.watch(x)
        for i in range(10):
            X, ana_bfr = ana(x[:, 16000*i:16000*(i+1)], ana_bfr)
            y, syn_bfr = syn(X, syn_bfr)
            E += tf.reduce_sum(y*y)
    dE_dx = g.gradient(E, x)
    return tf.norm(dE_dx/2 - x)/tf.norm(x)
    
print('Should have dE_dx/2=x. The relative abs error is: {}'.format(f()))