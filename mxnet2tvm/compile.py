# MNIST
# training, saving and compiling on hp

# https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/save_load_params.html
# 

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import numpy as np

# Use GPU if one exists, else use CPU
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# MNIST images are 28x28. Total pixels in input layer is 28x28 = 784
num_inputs = 784
# Clasify the images into one of the 10 digits
num_outputs = 10
# 64 images in a batch
batch_size = 64

# Load the training data
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True).transform_first(transforms.ToTensor()),
                                   batch_size, shuffle=True)

# Build a simple convolutional network
def build_lenet(net):
    with net.name_scope():
        # First convolution
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Second convolution
        net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Flatten the output before the fully connected layers
        net.add(gluon.nn.Flatten())
        # First fully connected layers with 512 neurons
        net.add(gluon.nn.Dense(512, activation="relu"))
        # Second fully connected layer with as many neurons as the number of classes
        net.add(gluon.nn.Dense(num_outputs))

        return net

# Train a given model using MNIST data
def train_model(model):
    # Initialize the parameters with Xavier initializer
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    # Use cross entropy loss
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    # Use Adam optimizer
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': .001})

    # Train for one epoch
    for epoch in range(1):
        # Iterate through the images and labels in the training data
        for batch_num, (data, label) in enumerate(train_data):
            # get the images and labels
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Ask autograd to record the forward pass
            with autograd.record():
                # Run the forward pass
                output = model(data)
                # Compute the loss
                loss = softmax_cross_entropy(output, label)
            # Compute gradients
            loss.backward()
            # Update parameters
            trainer.step(data.shape[0])

            # Print loss once in a while
            if batch_num % 50 == 0:
                curr_loss = nd.mean(loss).asscalar()
                print("Epoch: %d; Batch %d; Loss %f" % (epoch, batch_num, curr_loss))


net = build_lenet(gluon.nn.Sequential())
train_model(net)

file_name = "net.params"
net.save_parameters(file_name)


new_net = build_lenet(gluon.nn.Sequential())
new_net.load_parameters(file_name, ctx=ctx)

import matplotlib.pyplot as plt

def verify_loaded_model(net):
    """Run inference using ten random images.
    Print both input and output of the model"""

    def transform(data, label):
        return data.astype(np.float32)/255, label.astype(np.float32)

    # Load ten random images from the test dataset
    sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                  10, shuffle=True)

    for data, label in sample_data:

        # Display the images
        img = nd.transpose(data, (1,0,2,3))
        img = nd.reshape(img, (28,10*28,1))
        imtiles = nd.tile(img, (1,1,3))
        plt.imshow(imtiles.asnumpy())
        plt.show()

        # Display the predictions
        data = nd.transpose(data, (0, 3, 1, 2))
        out = net(data.as_in_context(ctx))
        predictions = nd.argmax(out, axis=1)
        print('Model predictions: ', predictions.asnumpy())

        break

verify_loaded_model(new_net)


net = build_lenet(gluon.nn.HybridSequential())
net.hybridize()
train_model(net)

# saving

net.export("lenet", epoch=1)

deserialized_net = gluon.nn.SymbolBlock.imports("lenet-symbol.json", ['data'], "lenet-0001.params", ctx=ctx)
verify_loaded_model(deserialized_net)

# compiling
import nnvm
sym, params = nnvm.frontend.from_mxnet(deserialized_net)

import tvm
target = tvm.target.arm_cpu('rasp3b')
data_shape = (1, 1, 28, 28)
with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(
        sym, target, shape={"data": data_shape}, params=params)

with open("deploy.json", "w") as fo:
    fo.write(graph.json())

lib.export_library("deploy.so", tvm.contrib.cc.create_shared,
    cc="/usr/bin/arm-linux-gnueabihf-g++")
lib.export_library("deploy.tar")

with open("deploy.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
print('OK')
