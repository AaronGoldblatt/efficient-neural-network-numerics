# Efficient Neural-Network Numerics and SVD Compression

Experiments with alternative convolution and matrix-multiplication implementations, numerical precision, and low-rank neural-network compression.

## Implementations

- Convolution: im2col, Winograd, and FFT methods in `src.conv2d`.
- Matrix multiplication: SVD and logarithmic-domain methods in `src.matmul`.
- Reconstruction-error comparisons across numerical precisions.
- Rank sweeps measuring approximation accuracy and runtime.
- Low-rank SVD layers for MNIST digit recognition.

## Experiments

- `1-numerical_precision.ipynb`: precision and reconstruction error.
- `2-svd_rank.ipynb`: rank, accuracy, and runtime tradeoffs.
- `mnist.py`: compression experiments on a digit-recognition network.

A saved experiment recorded 98.32% MNIST test accuracy at a 2.40× compression ratio for the selected fully connected weight matrices. This ratio is not whole-model or on-disk compression. Recorded runtime includes testing and is not single-inference latency.

## Attribution

Built on the ML-HW-SYS starter implementation and experimental framework. Existing source credits and license notices are retained.
