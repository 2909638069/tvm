#! /usr/bin/python3
import tvm
from tvm.contrib import graph_runtime
from PIL import Image
import numpy as np
import time

# load grap, lib and params.
# then create runtime module and load params.
ctx = tvm.cpu()
graph = open("deploy.json").read()
lib = tvm.module.load("./deploy.tar")
params = bytearray(open("deploy.params", "rb").read())
module = graph_runtime.create(graph, lib, ctx)
module.load_params(params)

# reuse the synset
synset_name = 'synset.txt'
with open(synset_name) as fi:
    synset = eval(fi.read())

# timer! the whole forward pass needs 0.5s ~ 1s on rasp3b
def transform_image(image):
    image = np.array(image) - np.array([123., 117.,104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image
begin = time.time()
img_name = 'cat.png'
image = Image.open(img_name).resize((224, 224))
x = transform_image(image)
dtype = 'float32'
x = x.astype(dtype)
module.set_input("data", x)
module.run()
tvm_output = module.get_output(0)
top1 = np.argmax(tvm_output.asnumpy()[0])
print('TVM prediction top-1:', top1, synset[top1])
print(time.time() - begin, 'seconds')
