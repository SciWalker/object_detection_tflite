from cProfile import label
from distutils.sysconfig import customize_compiler
import tensorflow as tf
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import DataLoader
import json
#Import the same libs that TFLiteModelMaker interally uses
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train_lib

#Setup variables
batch_size = 2 #or whatever batch size you want
epochs = 50
checkpoint_dir = "temp/" #whatever your checkpoint directory is

class CustomDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        #super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.train_image_path = ''
        self.validation_image_path = ''
        self.test_image_path=''
        self.label_map = {}

#Create whichever object detector's spec you want
spec = object_detector.EfficientDetLite1Spec(
    model_name='efficientdet-lite1',
    hparams='', #enable grad_checkpoint=True if you want
    model_dir=checkpoint_dir, 
    epochs=epochs, 
    batch_size=batch_size,
    steps_per_execution=1, 
    moving_average_decay=0,
    var_freeze_expr='(efficientnet|fpn_cells|resample_p6)',
    tflite_max_detections=50, 
    strategy=None, 
    tpu=None, 
    gcp_project=None,
    tpu_zone=None, 
    use_xla=False, 
    profile=False, 
    debug=False, 
    tf_random_seed=2314,
    verbose=1
)
#convert the key of dictionary to int
def convert_dict_key_to_int(dict):
    new_dict={}
    for key,value in dict.items():
        new_dict[int(key)]=value
    return new_dict
#read from json file
def read_json(filename,batch_num):
    with open(filename, 'r') as f:
        json_read=json.load(f)
    return json_read['train_images_path'][batch_num],json_read['validation_images_path'],json_read['test_images_path'],json_read['label_map']
#Load you datasets
#train_data, validation_data, test_data = object_detector.DataLoader.from_csv('train_1.csv')
data_obj=CustomDataLoader()
data_obj.train_image_path, data_obj.validation_image_path, data_obj.test_image_path,data_obj.label_map =read_json('../../inputs/inputs.json','batch_1')
label_map_decoded=convert_dict_key_to_int(data_obj.label_map)
print(data_obj.train_image_path)
#label_map_decoded=json.load(data_obj.label_map,object_hook=lambda d: {int(k): v for k, v in d.items()})
train_data = object_detector.DataLoader.from_pascal_voc('',data_obj.train_image_path, label_map_decoded)
test_data = object_detector.DataLoader.from_pascal_voc('',data_obj.test_image_path,label_map_decoded)
validation_data = object_detector.DataLoader.from_pascal_voc('',data_obj.validation_image_path, label_map_decoded)

#Create the object detector 
detector = object_detector.create(train_data, 
                                model_spec=spec, 
                                batch_size=batch_size, 
                                train_whole_model=True, 
                                validation_data=validation_data,
                                epochs = epochs,
                                do_train = False
                                )
"""
From here on we use internal/"private" functions of the API,
you can tell because the methods's names begin with an underscore
"""

#Convert the datasets for training
train_ds, steps_per_epoch, _ = detector._get_dataset_and_steps(train_data, batch_size, is_training = True)
validation_ds, validation_steps, val_json_file = detector._get_dataset_and_steps(validation_data, batch_size, is_training = False)

#Get the interal keras model    
model = detector.create_model()

#Copy what the API internally does as setup
config = spec.config
config.update(
    dict(
        steps_per_epoch=steps_per_epoch,
        eval_samples=batch_size * validation_steps,
        val_json_file=val_json_file,
        batch_size=batch_size
    )
)
train.setup_model(model, config) #This is the model.compile call basically
model.summary()

"""
Here we restore the weights
"""

#Load the weights from the latest checkpoint
try:
  latest = tf.train.latest_checkpoint(checkpoint_dir) #example: "/content/drive/My Drive/Colab Notebooks/checkpoints_heavy/ckpt-35"
  completed_epochs = int(latest.split("/")[-1].split("-")[1]) #the epoch the training was at when the training was last interrupted
  model.load_weights(latest)
  print("Checkpoint found {}".format(latest))
except Exception as e:
  completed_epochs=1
  print("Checkpoint not found: ", e)

#Train the model 
model.fit(
    train_ds,
    epochs=epochs,
    initial_epoch=completed_epochs, 
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_ds,
    validation_steps=validation_steps,
    callbacks=train_lib.get_callbacks(config.as_dict(), validation_ds) #This is for saving checkpoints at the end of every epoch
)

#Save/export the trained model
#Tip: for integer quantization you simply have to NOT SPECIFY 
#the quantization_config parameter of the detector.export method
export_dir =r'D:\Projects\iot\GreenHouse\PlantReg_v2\TensorFlow\trained_models' #save the tflite wherever you want
quant_config = QuantizationConfig.for_float16() #or whatever quantization you want
detector.model = model #inject our trained model into the object detector
detector.export(export_dir = export_dir, tflite_filename='model.tflite', quantization_config = quant_config)