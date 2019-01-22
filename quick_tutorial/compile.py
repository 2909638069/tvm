#! /usr/bin/python3
import nnvm.testing
import tvm
import nnvm.compiler

# define a net
batch_size = 1

image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

num_class = 1000
out_shape = (batch_size, num_class)

net, params = nnvm.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size,
    image_shape=image_shape)
print(net.debug_str())


# compile
target = tvm.target.arm_cpu('rasp3b')
with nnvm.compiler.build_config():
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)


# save
with open("deploy.json", "w") as fo:
    fo.write(graph.json())

lib.export_library("deploy.so", tvm.contrib.cc.create_shared, cc="/usr/bin/arm-linux-gnueabihf-g++")
lib.export_library("deploy.tar")

with open("deploy.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))

