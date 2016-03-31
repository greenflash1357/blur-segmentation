using PyPlot

ENV["MOCHA_USE_CUDA"] = "true"
using Mocha

include("utils/plotting.jl") # plotting tools
include("utils/io.jl") # I/O tools

include("segmentation.jl")

backend = DefaultBackend()
init(backend)
net = create_segmentation_net(backend)
load_snapshot(net, "snapshot.jld")

acc = []

imagedir = "dataset/evalset/image"
gtdir = "dataset/evalset/gt"

if !isdir("results")
	mkdir("results")
end

for (imfile,gtfile) in zip(readdir(imagedir), readdir(gtdir))

	# load image
	println("Processing $(imfile)...")
	im = PyPlot.imread(joinpath(imagedir, imfile))
	gt = PyPlot.imread(joinpath(gtdir, gtfile))

	# do segmentation
	tic()
	probmap, segmentation = blursegmentation(net, im)
	toc()

	# show/save result
	intersection = (segmentation.>0) & (gt.>0)
	union = (segmentation.>0) | (gt.>0)
	v = sum(intersection) / sum(union)
	V = @sprintf("%.4f%%", v*100)
	println("Intersection over Union: $(V)")
	push!(acc, v)

	figure()
	subplot(1,2,1)
	plot_segmentation(im, gt)
	title("Ground Truth")
	subplot(1,2,2)
	plot_segmentation(im, segmentation)
	title("Estimated Segmentation")

	path = joinpath("results", split(imfile,".")[1]*".jld")
	save(path,"segmentation",segmentation, "prob",probmap, "IU",v)
	save(joinpath("results", gtfile), grayim((segmentation)'))

end

println(@sprintf("Average Accuracy: %.2f", mean(acc)))

destroy(net)
shutdown(backend)