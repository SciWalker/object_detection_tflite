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

full_path = r'D:\Projects\Dataset\detection\id\id\ukraine\front\41_6151511_198991.jpg'
# full_path = r'D:\Projects\Dataset\detection\id\dl\russia\front\22_6517563_364725.jpg'

with tf.io.gfile.GFile(full_path, 'rb') as fid:
	encoded_jpg = fid.read()
encoded_jpg_io = io.BytesIO(encoded_jpg)
image = PIL.Image.open(encoded_jpg_io)

if image.format != 'JPEG':
	print(image.format)
	raise ValueError('Image format not JPEG')
else:
	print('JPEG')
	
# spec = model_spec.get('efficientdet_lite1')
# spec.steps_per_execution = 1
# train_data, validation_data, test_data = object_detector.DataLoader.from_csv(filename=r'D:\Projects\tflite\training\tflite-train.csv')
# train_data = object_detector.DataLoader.from_pascal_voc('', r'D:\Projects\cnn_detection\workspace\id_detection\annotations\9_tflite\train', label_map={1: "id", 2: "passport"})
# test_data = object_detector.DataLoader.from_pascal_voc('', r'D:\Projects\cnn_detection\workspace\id_detection\annotations\9_tflite\test', label_map={1: "id", 2: "passport"})
# validation_data = object_detector.DataLoader.from_pascal_voc('', r'D:\Projects\cnn_detection\workspace\id_detection\annotations\9_tflite\val', label_map={1: "id", 2: "passport"})

# model = object_detector.create(train_data, epochs=1, model_spec=spec, batch_size=1, train_whole_model=True, validation_data=validation_data)
# model.evaluate(test_data)
# model.export(export_dir=r'D:\Projects\tflite\training\models')
# model.evaluate_tflite('model.tflite', test_data)
