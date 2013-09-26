FloatingPyramids
================

FloatingPyramids performs dynamic range reduction on images using Laplacian
pyramids.

The algorithms used are described in the papers [Local Laplacian Filters:
Edge-aware Image Processing with a Laplacian
Pyramid](http://people.csail.mit.edu/sparis/publi/2011/siggraph/) and [Fast and
Robust Pyramid-based Image
Processing](http://dspace.mit.edu/handle/1721.1/67030).

Dependencies
------------
FloatingPyramids is written in C++ and CUDA and requires [CUDA 5.0 or
later](https://developer.nvidia.com/cuda-toolkit), [OpenCV](http://opencv.org/),
and a Linux-compatible build environment.

Installation
------------
To compile FloatingPyramids, edit the `Makefile` and adjust the variables for
your environment (most importantly, choose `CUDA_ARCH` to match the compute
capability of your GPU).  Then run `make`; the resulting executable will be
named `bin/pyramids`.

Usage
-----
    pyramids <input_image> <output_prefix>

Note that `<input_image>` currently must have dimensions that are powers of 2.
The monochrome output is written to the file `<output_prefix>_final.png`, and
an experimental color output is written to `colorout.png`.
