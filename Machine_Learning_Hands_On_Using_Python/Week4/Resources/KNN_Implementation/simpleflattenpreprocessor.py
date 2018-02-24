import numpy as np

class SimpleFlattenPreprocessor:
	# Convert an image of shape (m,n,p) to a vector of shape (m*n*p,).
	def __init__(self):
		pass

	def preprocess(self, image):
		# Check the resources to see what flatten does.
		return image.flatten()