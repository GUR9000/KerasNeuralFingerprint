Keras-based implementation of neural fingerprints
============================================================================================================

A convolutional neural network operating on molecular graphs (SMILES) of arbitrary size for chemical property prediction (e.g. solubility).


## Requirements:

Python, Numpy -- preferrably using [Anaconda](https://www.continuum.io/downloads)

Either [Theano](http://deeplearning.net/software/theano/install.html) or [Tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)

[RDkit](http://www.rdkit.org/docs/Install.html) -- the easiest way to install it when using Anaconda is "conda install -c https://conda.anaconda.org/rdkit rdkit"


---------------------------------------

Paper describing the method: [Convolutional Networks on Graphs for Learning Molecular Fingerprints](http://arxiv.org/pdf/1509.09292.pdf)

The original implementation using numpy/autograd can be found at (https://github.com/HIPS/neural-fingerprint)
