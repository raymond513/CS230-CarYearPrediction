from keras.preprocessing import image
import os
import argparse
import sys
from pprint import pprint
import shutil as sh
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD, RMSprop

def main(args):
    pprint(args)
   

    # load json and create model
    json_file = open(args.model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(args.model_name + '.h5')
    print('Loaded model from disk')


   	test_image = image.load_img('../../car_ims/val/1991/015759.jpg', target_size=(224,224))
	test_image = image.img_to_array(test_image)
	print(loaded_model.predict(test_image))

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--predict_data', type=str,
        help='Dir with data to predict classes for'
        , default= './car_ims/tst')
        
    parser.add_argument('--model_name', type=str,
        help='Saved model name location', default='./vgg16/tst/vgg16_lr001_dr7_decaye-4_finalModel')
        
    parser.add_argument('--lr_decay', type=float,
        help='Learning rate decay.', default=1e-3)
    
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate.', default=0.0003)
    
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))



"""

if __name__ == "__main__":
	# load json and create model
    json_file = open(args.model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(args.model_name + '.h5')
    print('Loaded model from disk')

   	test_image = image.load_img('../../car_ims/val/1991/015759.jpg', target_size=(224,224))
	test_image = image.img_to_array(test_image)
	print(loaded_model.predict(test_image))


   
	test_image = image.load_img('../../car_ims/val/1991/015759.jpg', target_size=(224,224))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image,axis=0)
	result = classifier.predict(test_image)
	training_set.class_indices
	print(result)
	"""
	"""