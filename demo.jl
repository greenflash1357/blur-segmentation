using PyPlot

ENV["MOCHA_USE_CUDA"] = "true"
using Mocha

include("utils/plotting.jl") # plotting tools

include("segmentation.jl")

backend = DefaultBackend()
init(backend)
net = create_segmentation_net(backend)
load_snapshot(net, "snapshot.jld")

im = PyPlot.imread("image.jpg")

probmap, segmentation = blursegmentation(net, im)

figure()
subplot(1,2,1)
imshow(im, interpolation="none")
title("Original Image")
subplot(1,2,2)
plot_segmentation(im, segmentation)
title("Estimated Segmentation")

destroy(net)
shutdown(backend)