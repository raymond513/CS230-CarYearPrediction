'''
Created on 10 lut 2018

@author: mgdak
'''

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
from sklearn.metrics import classification_report, confusion_matrix


def main(args):
    pprint(args)
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        args.predict_data, 
        target_size=(224, 224), 
        batch_size=1, 
        class_mode='categorical')
    
    filenames = test_generator.filenames
    nb_samples = len(filenames)

    print("nb_samples %.2f%%" % (nb_samples))

    # load json and create model
    json_file = open(args.model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(args.model_name + '.h5')
    print('Loaded model from disk')
    pprint(len(test_generator.classes))
    print(type(test_generator.classes[8040]))
    pprint("HELLOOOOOOOOOOOOOOOOOOOOOO") 
    #y_true=loaded_model.evaluate_generator.classes
    #pprint(y_true)
     
    # evaluate loaded model on test data
    """
    sgd = SGD(lr=args.learning_rate, decay=args.lr_decay, momentum=0.9, nesterov=True)
    loaded_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    score = loaded_model.predict_generator(test_generator, steps = nb_samples)
    for i in range(0, nb_samples):
        idx = np.argmax(score[i])
        pprint(filenames[i])
        pprint("prediction idx, value %.2f%% %.2f%%" %(idx, score[i][idx]))
        pprint(test_generator.classes[i])
    pprint(loaded_model.metrics_names)
    predictionIndices = np.argmax(score,axis=1)
    print(predictionIndices)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, predictionIndices))
    labels = (test_generator.class_indices)
    labels2 = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predictionIndices]
    print (confusion_matrix(predictionIndices,labels))
    """
    test_generator.reset()
    Y_pred = loaded_model.predict_generator(test_generator)
    classes = test_generator.classes[test_generator.index_array]
    y_pred = np.argmax(Y_pred, axis=-1)
    sum(y_pred==classes)/10000
    print(confusion_matrix(test_generator.classes[test_generator.index_array],y_pred))

    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

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
