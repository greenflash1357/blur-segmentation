ENV["MOCHA_USE_CUDA"] = "true"
using Mocha

include("utils/io.jl") # I/O tools

include("network.jl")


data_layer  = AsyncHDF5DataLayer(name="train-data", source="train.txt", batch_size=64, shuffle=false)
common_layers = create_common_layers()
drop_fc1_layer = DropoutLayer(name="drop1", bottoms=[:ip1], ratio=0.5)
loss_layer  = SoftmaxLossLayer(name="loss", bottoms=[:ip2,:label])

backend = DefaultBackend()
init(backend)

net = Net("train-net", backend, [data_layer, common_layers..., drop_fc1_layer, loss_layer])

exp_dir = "snapshots"

method = SGD()
params = make_solver_parameters(method, max_iter=1000000, regu_coef=0.0005,
                                mom_policy=MomPolicy.Fixed(0.9),
                                lr_policy=LRPolicy.Fixed(0.0001),
                                #lr_policy=LRPolicy.Inv(0.005, 0.0001, 0.75),
                                load_from=exp_dir)
solver = Solver(method, params)

setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=500)

# report training progress
add_coffee_break(solver, TrainingSummary(), every_n_iter=500)

# save snapshots
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=50000)

# show performance on test data
data_layer_test = HDF5DataLayer(name="test-data", source="test.txt", batch_size=100)
acc_layer = AccuracyLayer(name="accuracy", bottoms=[:ip2, :label])
test_net = Net("test-net", backend, [data_layer_test, common_layers..., acc_layer])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=5000)
# show performance on trainingsdata
acc_layer_2 = AccuracyLayer(name="trainings-accuracy",bottoms=[:ip2,:label])
train_acc_net = Net("train-acc-net", backend, [data_layer, common_layers..., acc_layer_2])
add_coffee_break(solver, ValidationPerformance(train_acc_net), every_n_iter=25000)

@time solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(backend)

extract_filters(exp_dir)
