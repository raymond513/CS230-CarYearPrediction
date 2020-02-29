import numpy as np
from keras.preprocessing import image

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


    """
	test_image = image.load_img('../../car_ims/val/1991/015759.jpg', target_size=(224,224))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image,axis=0)
	result = classifier.predict(test_image)
	training_set.class_indices
	print(result)
	"""