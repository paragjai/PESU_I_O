import cv2
import os 
import numpy as np
class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# preprocessors is a list of SimplePreprocessor objects.
		self.preprocessors = preprocessors

		if self.preprocessors is None:
				self.preprocessors = []

	# preprocess all the images, puts them in a numpy array, puts the corresponding labels in another numpy array and returns both.
	def load(self, imagePaths, verbose=-1):
		data = []
		labels = []

		for (i, imagePath) in enumerate(imagePaths):
			# example imagePath : /dataset_name/class/image_name.jpg
			image = cv2.imread(imagePath)
			# print("image.shape", image.shape)
			label = imagePath.split(os.path.sep)[-2]

			if self.preprocessors is not None:
				# Apply each of the preprocessors to each of the image
				for p in self.preprocessors:
					image = p.preprocess(image)
					# print("preprocess image shape", image.shape)
			# append the preprocessed image to the data list
			data.append(image)
			# append the corresponding label to the labels list
			labels.append(label)

			# We would want to print after every verbose number of images are preprocessed. Just a convention. Not a must.
			# It is good to know the progress after some fixed number of iterations so that we know that the code is running.
			if verbose > 0 and i > 0 and (i + 1)%verbose == 0:
				print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))

		return (np.array(data), np.array(labels))



