#! /usr/bin/python3

import numpy as np
from tvm.contrib import graph_runtime
import tvm
import time

graph = open("deploy.json").read()
lib = tvm.module.load("./deploy.tar")
params = bytearray(open("deploy.params", "rb").read())

ctx = tvm.cpu()
module = graph_runtime.create(graph, lib, ctx)
module.load_params(params)
batch_size = 1
begining = time.time()
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

num_class = 1000
out_shape = (batch_size, num_class)

data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

module.set_input("data", data)

module.run()

out = module.get_output(0, tvm.ndarray.empty(out_shape))
out.asnumpy()
print(time.time() - begining)
print(out.asnumpy().flatten()[1:10])
print(1)
