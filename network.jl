ENV["MOCHA_USE_CUDA"] = "true"
using Mocha

function create_common_layers()

	conv1_layer  = ConvolutionLayer(name="conv1", n_filter=96, kernel=(7,7), bottoms=[:data], tops=[:conv1], neuron=Neurons.ReLU())
	pool1_layer  = PoolingLayer(name="pool1", kernel=(2,2), stride=(2,2), bottoms=[:conv1], tops=[:pool1])
	conv2_layer = ConvolutionLayer(name="conv2", n_filter=256, kernel=(5,5), bottoms=[:pool1], tops=[:conv2], neuron=Neurons.ReLU())
	pool2_layer = PoolingLayer(name="pool2", kernel=(2,2), stride=(2,2), bottoms=[:conv2], tops=[:pool2])
	fc1_layer   = InnerProductLayer(name="ip1", output_dim=1024, neuron=Neurons.ReLU(), bottoms=[:pool2], tops=[:ip1])
	fc2_layer   = InnerProductLayer(name="ip2", output_dim=2, bottoms=[:ip1], tops=[:ip2])

	common_layers = [conv1_layer, pool1_layer, conv2_layer, pool2_layer, fc1_layer, fc2_layer]
	
	return common_layers
end
