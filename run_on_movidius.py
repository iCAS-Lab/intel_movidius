try: from openvino.inference_engine import IECore, IENetwork
except ImportError: print('Make sure you activated setupvars.sh!')
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf

# Prepare the dataset
############################################################################################
# Load the iris dataset
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Set up the core -> Finding your device and loading the model
############################################################################################
ie = IECore()
device_list = ie.available_devices

# Load any network from file
model_xml = "keras_mnist_model.xml"
model_bin = "keras_mnist_model.bin"
net = IENetwork(model=model_xml, weights=model_bin)


# create some kind of blob
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

# Input Shape
print('Input Shape: ' + str(net.inputs[input_blob].shape))

# Run the model on the device
##############################################################################################
# load model to device
exec_net = ie.load_network(network=net, device_name='MYRIAD')

# execute the model and read the output
acc = 0
for i in range(len(X_test)):
    res = exec_net.infer(inputs={input_blob: X_test[i]})
    res = res[out_blob]
    # The result is the softmax output corresponding to the probability distribution in one hot encoding
    # Except for the pytorch model - here we don't do the softmax. The highest value corresponds to the class.
    acc += 1 if (np.argmax(res)==y_test[i]) else 0
print("Test accuracy:",acc/len(X_test))
