#! /usr/bin/python3
import mxnet as mx
import nnvm
import tvm
import numpy as np
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from PIL import Image
from matplotlib import pyplot as plt
import nnvm.compiler

# download, load and transform an image
def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image
img_name = 'cat.png'
download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_name)
image = Image.open(img_name).resize((224, 224))
plt.imshow(image)
plt.show()
x = transform_image(image)
print('x', x.shape)

# download, load synset
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'synset.txt'
download(synset_url, synset_name)

with open(synset_name) as f:
    synset = eval(f.read())

# load model from mxnet and convert to static graph and HybridBlock for nnvm
block = get_model('resnet18_v1', pretrained=True)
sym, params = nnvm.frontend.from_mxnet(block)
sym = nnvm.sym.softmax(sym) # we want a probability so add a softmax operator

# compile by nnvm
target = tvm.target.arm_cpu('rasp3b')
shape_dict = {'data': x.shape}
with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

# save
with open("deploy.json", "w") as fo:
    fo.write(graph.json())
lib.export_library("deploy.so", tvm.contrib.cc.create_shared,
    cc="/usr/bin/arm-linux-gnueabihf-g++")
lib.export_library("deploy.tar")
with open("deploy.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
