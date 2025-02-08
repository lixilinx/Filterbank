# Filterbank in Tensorflow and Pytorch

### Background

Short-time Fourier transform (STFT) is still widely used for time-frequency analysis. However, filter banks are far more flexible and performant with little extra computational complexity.

##### Types of filter banks and their prototype filter designs 

Filter banks with efficient implementations always use periodic modulation sequences and FFT like fast algorithms. There are plenty of design examples [here](https://github.com/lixilinx/FilterbanksBestPractices). A discrete Fourier transform (DFT) modulated prototype design example is included here for demo purpose.

##### View of overlapping chunks of the time domain signal

This operation is required for analysis. It corresponds to **numpy.lib.stride_tricks.as_strided** in Numpy,  **tf.signal.frame** in Tensorflow, and **torch.nn.functional.unfold** in Pytorch.   

##### Overlap-and-add

This operation is required for synthesis. It corresponds to **tf.signal.overlap_and_add** in Tensorflow, and **torch.nn.functional.fold** in Pytorch. 
Seems that numpy/scipy does not provide a native vectorized overlap-and-add implementation. 

### Implementations in this package

I have implemented the DFT modulated filter bank for real-valued signals. The filter length is assumed to be a multiple of FFT length so that no zero padding is needed for vectorized implementation.  Implementations for complex-valued signals and/or other modulations like DCT-IV should be similar.



