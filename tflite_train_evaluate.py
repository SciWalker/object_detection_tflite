import numpy as np
import os
import io

import PIL
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

  #full_path = r'D:\Projects\Dataset\detection\id\id\ukraine\front\41_6151511_198991.jpg'

#with tf.io.gfile.GFile(full_path, 'rb') as fid:
#  encoded_jpg = fid.read()
# encoded_jpg_io = io.BytesIO(encoded_jpg)
#image = PIL.Image.open(encoded_jpg_io)

#if image.format != 'JPEG':
#  print(image.format)
#  raise ValueError('Image format not JPEG')
#else:
#  print('JPEG')
model = object_detector.create(train_data, epochs=20, model_spec=spec, batch_size=2, train_whole_model=True, validation_data=validation_data)

model.evaluate_tflite('../../trained_models/model.tflite', test_data)