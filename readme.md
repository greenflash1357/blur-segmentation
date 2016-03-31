# A Convolutional Neural Network for Blur Segmentation

The code under this repository provides an implementation of a neural network for blur segmentation in images. 
It is written in [Julia](www.julialang.org) and uses the [Mocha](https://github.com/pluskid/Mocha.jl) framwork for deep learning.

For more details on the network or the method please refer to the [paper](paper.pdf).

## Dependencies

You will need a working installation of the following Julia packages:

* [Mocha](https://github.com/pluskid/Mocha.jl)
* [Images](https://github.com/timholy/Images.jl)
* [PyPlot](https://github.com/stevengj/PyPlot.jl)
* [HDF5](https://github.com/JuliaLang/HDF5.jl)
* [JLD](https://github.com/JuliaLang/JLD.jl)
* [Grid](https://github.com/timholy/Grid.jl)
* [ImageMagick](https://github.com/JuliaIO/ImageMagick.jl)
* [Colors](https://github.com/JuliaGraphics/Colors.jl)
* [BinDeps](https://github.com/JuliaLang/BinDeps.jl)

## Quick start

Run `demo.jl` to do a segmentation for the example image `image.jpg`, using the provided network parameters in `snapshot.jld`.

The file `segmentation.jl` provides a function `blursegmentation(net, image)` that will return the probabilty map and the segmentation, given a network and an image.
For convenience `create_segmentation_net()` will create the required network for you. All you need to do is to load the network parameters from a snapshot of a network with the same topology.

## Training

To train the network yourself, you will need trainings data. Please refer to the readme in the `dataset` folder for further instructions on how to get the required data.

After that you can simply run `train.jl`.
The training process will take some time and the intermediate training state will be saved to a `snapshots` folder. We highly recommend to use the GPU backend for training. If you want to use other options, you can change the solver parameters.

## Evaluation

To evaluate a network, run `eval.jl`. It will compute segmentations for all images in `dataset/evalset` and compare them to the ground truth using Intersection-over-union.