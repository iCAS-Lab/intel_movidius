"""
This is the file docstring.
"""
################################################################################
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
################################################################################
# Clean up settings
REMOVE_MODEL = True
REMOVE_DATA = True
TRAIN_NEW_MODEL = True
USE_GPUS = False
N_GPUS = 4
################################################################################
# GLOBAL VARS
DS_NAME = 'mnist'
OUTPUT_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 5
TRAIN_PERC = 60 # Train percent
VAL_PERC = 10   # Validation percent
TEST_PERC = 30  # Not Needed
TRAIN = 'test+train[:{}%]'.format(str(TRAIN_PERC))
VAL = 'test+train[{}%:{}%]'.format(str(TRAIN_PERC),str(TRAIN_PERC+VAL_PERC))
TEST = 'test+train[{}%:]'.format(str(TRAIN_PERC+VAL_PERC))
################################################################################
# Paths
MODEL_OUT_DIR = os.path.join(os.path.abspath('..'), 'models')
WORKING_DIR = os.path.join(os.path.abspath('..'),'data')
# Output filenames
MODEL_NAME = 'lenet.h5'
# Print the dirs
print('LOG --> MODEL_OUT_DIR: '+str(MODEL_OUT_DIR))
print('LOG --> DATASET_DIR: '+str(WORKING_DIR))
# Check that dirs exist if not create
if not os.path.exists(MODEL_OUT_DIR):
    os.mkdir(MODEL_OUT_DIR)
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
# Cleanup last run
if os.path.exists(os.path.join(MODEL_OUT_DIR, MODEL_NAME)) and REMOVE_MODEL:
    os.remove(os.path.join(MODEL_OUT_DIR, MODEL_NAME))
if os.path.exists(os.path.join(WORKING_DIR, 'x_test.npz')) and REMOVE_DATA:
    os.remove(os.path.join(WORKING_DIR, 'x_test.npz'))
if os.path.exists(os.path.join(WORKING_DIR, 'y_test.npz')) and REMOVE_DATA:
    os.remove(os.path.join(WORKING_DIR, 'y_test.npz'))
if os.path.exists(os.path.join(WORKING_DIR, 'x_norm.npz')) and REMOVE_DATA:
    os.remove(os.path.join(WORKING_DIR, 'x_norm.npz'))
################################################################################
# Distribute on multiple GPUS
if USE_GPUS:
    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(device_type)
    devices_names = [d.name.split('e:')[1] for d in devices]
    print(devices_names)
    strategy = tf.distribute.MirroredStrategy(devices=devices_names[:N_GPUS])
else:
    device_type = 'CPU'
    devices = tf.config.experimental.list_physical_devices(device_type)
    device_names = [d.name.split('e:')[1] for d in devices]
    strategy = tf.distribute.OneDeviceStrategy(device_names[0])
################################################################################
# Helper Functions
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def preprocess_ds(a_ds, info, a_split, eval_flag=False):
    """Normalize images, shuffle, and prep datasets."""
    ds = a_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if not eval_flag:
        ds = ds.shuffle(info.splits[a_split].num_examples)
    # ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def convert_to_numpy(ds):
    """Converts the tensors to numpy arrays"""
    global OUTPUT_CLASSES
    images = []
    labels = []
    ds_len = len(ds)
    print('LOG --> Processing dataset of length {}...'.format(ds_len))
    for image, label in tfds.as_numpy(ds):
        images.append(np.asarray(image))
        labels.append(int(label))
    images = np.asarray(images)
    labels = to_categorical(np.asarray(labels), OUTPUT_CLASSES)
    print(images.shape)
    print(labels.shape)
    return images, labels

def save_data_as_npz(images, labels, suffix, split):
    """Saves data as seperate image and label files."""
    global WORKING_DIR
    if split:
        np.savez_compressed(os.path.join(WORKING_DIR, 'x_{}'.format(suffix)),
                            images[::split])
    else:
        np.savez_compressed(os.path.join(WORKING_DIR, 'x_{}'.format(suffix)),
                            images)
        np.savez_compressed(os.path.join(WORKING_DIR, 'y_{}'.format(suffix)),
                            labels)
    return images, labels
################################################################################
# Dataset Import
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    DS_NAME,
    as_supervised=True,
    split=[TRAIN, VAL, TEST],
    shuffle_files=True,
    with_info=True
)

ds_train = preprocess_ds(ds_train, ds_info, TRAIN, False)
ds_val = preprocess_ds(ds_val, ds_info, VAL, False)
ds_test = preprocess_ds(ds_test, ds_info, TEST, True)
x_train, y_train = convert_to_numpy(ds_train)
x_test, y_test = convert_to_numpy(ds_test)
x_val, y_val = convert_to_numpy(ds_val)

# Save test set
save_data_as_npz(x_test, y_test, 'test', None)
save_data_as_npz(x_train, y_train, 'norm', 10)

img_shape = x_train.shape[1:]
################################################################################
# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_OUT_DIR, MODEL_NAME),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=0,
    mode='min'
)
callbacks = [checkpoint]
################################################################################
# Model Construction
print('LOG --> Input Shape: '+str(img_shape))
with strategy.scope():
    # Input Layer
    inputs = tf.keras.Input(img_shape)
    # Begin Network
    layer = tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), activation='relu')(inputs)
    layer = tf.keras.layers.AveragePooling2D((2,2))(layer)
    layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu')(layer)
    layer = tf.keras.layers.AveragePooling2D((2,2))(layer)
    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dense(units=120, activation='relu')(layer)
    layer = tf.keras.layers.Dense(units=84, activation='relu')(layer)
    # Output Layer
    outputs = tf.keras.layers.Dense(units=OUTPUT_CLASSES, activation='softmax')(layer)

    # Instantiate the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

# Show the model summary
model.summary()

if TRAIN_NEW_MODEL:
    # Train the model
    model.fit(
        x_train, y_train,
        epochs=EPOCHS, 
        verbose=2,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        batch_size=BATCH_SIZE
    )
else:
    model.load_weights(os.path.join(MODEL_OUT_DIR, MODEL_NAME))

# Evaluate the model on the test data
inference_start = time.time()
loss, acc = model.evaluate(
    x_test,
    y_test,
    verbose=2
)
inference_end = time.time()
total_inference_time = inference_end - inference_start
print('INFERENCE PERFORMED ON {} IMAGES IN BATCHES OF {}'.format(len(x_test), BATCH_SIZE))
print('EVALUATION LATENCY: {}'.format(total_inference_time))
print('EVALUATION LOSS: {}, EVALUATION ACC: {}'.format(loss,acc))
################################################################################
# Convert H5 file
saved_model = tf.keras.models.load_model('/root/models/lenet.h5')
tf.saved_model.save(saved_model,'lenet')
################################################################################
# Create graph file
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
#path of the directory where you want to save your model
frozen_out_path = './'
# name of the .pb file
frozen_graph_filename = "inference_graph"
# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)

input("LOG --> Waiting for graph to be written")
################################################################################
import time
from pathlib import Path

import cv2
#import matplotlib.pyplot as plt
import numpy as np
#from IPython.display import Markdown
from openvino.inference_engine import IECore

# The paths of the source and converted models
model_path = Path("/root/models/inference_graph.pb")
ir_path = Path(model_path).with_suffix(".xml")

# Construct the command for Model Optimizer
mo_command = f"""python3 /opt/intel/openvino_2021.4.752/deployment_tools/model_optimizer/mo.py
                 --input_model "{model_path}"
                 --input_shape "{list(img_shape)}"
                 --data_type FP16
                 --output_dir "{model_path.parent}"
                 """
mo_command = " ".join(mo_command.split())
print(mo_command)
input("Waiting for mo to run...")

ie = IECore()
net = ie.read_network(model=ir_path, weights=ir_path.with_suffix(".bin"))
exec_net = ie.load_network(network=net, device_name="MYRIAD")

input_key = list(exec_net.input_info)[0]
output_key = list(exec_net.outputs.keys())[0]
network_input_shape = exec_net.input_info[input_key].tensor_desc.dims

# grab a sample
print(y_test[0])
sample = np.expand_dims(np.squeeze(x_test[0]), 0)
result = exec_net.infer(inputs={input_key: sample})[output_key]

print(result)
