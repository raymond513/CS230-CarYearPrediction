import numpy as np
from keras.preprocessing import image

if __name__ == "__main__":
	test_image = image.load_img('../../car_ims/val/1991/015759.jpg', target_size=(224,224))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image,axis=0)
	result = classifier.predict(test_image)
	training_set.class_indices
	print(result)