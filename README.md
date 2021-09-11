# Filterbank
Filterbank in Tensorflow and Pytorch

### Background

Short-time Fourier transform (STFT) is still widely used today for time-frequency analysis. However, filter banks are far more flexible and performant with little extra computational complexity.

##### Types of filter banks and their prototype filter designs 

Filter banks with efficient implementations always use periodic modulation sequences and FFT like fast algorithms. [PSMFB](https://sites.google.com/site/lixilinx/home/psmfb) is a flexible design tool for such types. A discrete Fourier transform (DFT) modulated prototype design example is included here for demo purpose.

##### View of overlapping chunks of the time domain signal

This operation is required for analysis. It corresponds to **numpy.lib.stride_tricks.as_strided** in Numpy,  **tf.signal.frame** in Tensorflow, and **torch.nn.functional.unfold** in Pytorch.   

##### Overlap-and-add

This operation is required for synthesis. It corresponds to **tf.signal.overlap_and_add** in Tensorflow, and **torch.nn.functional.fold** in Pytorch. 
Seems that numpy/scipy does not provide a native vectorized overlap-and-add implementation. 

### Implementations in this package

Currently, I have implemented the DFT modulated filter bank for real-valued signals. Implementations for complex-valued signals or other modulations like DCT-IV should be similar. 

Both functional and class (tf.keras.layers.Layer for Tensorflow and torch.nn.Module for Pytorch) implementations are provided.   

Regarding the prototype filter design tool [PSMFB](https://sites.google.com/site/lixilinx/home/psmfb), it is necessary to choose the filter length to be a multiple of FFT length (for DFT modulation) so that no zero padding is needed in vectorized implementation. The analysis-synthesis delay can be adjusted with step size down to one sampling period.  Clearly, it cannot be smaller than (hop_size - 1) in any causal implementation. Also, circular shifting is required if (delay + 1) is not a multiple of FFT length.     




