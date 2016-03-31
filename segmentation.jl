using Images
using PyPlot

ENV["MOCHA_USE_CUDA"] = "true"
using Mocha

include("utils/images.jl") # image processing tools

include("network.jl")


function create_segmentation_net(backend)
    common_layers = create_common_layers()
    data_layer = MemoryDataLayer(name="data", tops=[:data], batch_size=64, 
                                 data=Array[zeros(Float32,31,31,3,64)])
    softmax_layer = SoftmaxLayer(name="class", tops=[:prob], bottoms=[:ip2])
    mem_out = MemoryOutputLayer(name="output", bottoms=[:prob])
    net = Net("segmentation-net", backend, [data_layer, common_layers..., softmax_layer, mem_out])
    return net
end


function blursegmentation(net, im)

    println("Starting segmentation...")

    if eltype(im) <: Integer
        im = im2double(im)
    end
    im = map(Float32, im)
    colordims = size(im,3)
    rows,cols = size(im,1,2)
    pad = padarray(im,[15,15,0],[15,15,0],"reflect")
    N = rows*cols
    batchsize = get_layer(net,"data").batch_size

    # loop over all patches 
    n = 0
    while n+batchsize <= N
        for idx = n+1:n+batchsize
            r,c = ind2sub((rows,cols), idx)
            patch = permutedims(pad[r:r+30,c:c+30,:],[2,1,3])
            get_layer(net,"data").data[1][:,:,:,idx-n] = patch
        end
        forward(net)
        n += batchsize
    end
    if n < N
        for idx = n+1:N
            r,c = ind2sub((rows,cols), idx)
            patch = permutedims(pad[r:r+30,c:c+30,:],[2,1,3])
            get_layer(net,"data").data[1][:,:,:,idx-n] = patch
        end
        forward(net)
    end

    # get ouput
    output = get_layer_state(net, "output").outputs[1]
    reset_outputs(get_layer_state(net, "output"))
    output = cat(2,output...)
    probs = output[2,:];
    vals,idxs = findmax(output,1)
    predictions = [Float32(ind2sub((2,batchsize),idx)[1]-1) for idx in idxs]

    segmentation = reshape(predictions[1:prod(size(im,1,2))],size(im,1,2))
    probability = reshape(probs[1:prod(size(im,1,2))],size(im,1,2))

    println("Done.")

    return probability, segmentation
end