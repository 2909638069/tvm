# inference on pi
# create module and load parameters
# load an image
# run the module


#! /usr/bin/python3

import numpy as np
from tvm.contrib import graph_runtime
import tvm

ctx = tvm.cpu()
graph = open("deploy.json").read()
lib = tvm.module.load("./deploy.tar")
params = bytearray(open("deploy.params", "rb").read())
module = graph_runtime.create(graph, lib, ctx)
module.load_params(params)

batch_size = 1
image_shape = (1, 28, 28)
data_shape = (batch_size,) + image_shape
num_class = 10
out_shape = (batch_size, num_class)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")


module = graph_runtime.create(graph, lib, ctx)
module.set_input("data", data)
module.load_params(params)
module.run()

out = module.get_output(0, tvm.ndarray.empty(out_shape))
out.asnumpy()
print(out.asnumpy().flatten()[1:10])
print('OK')
