from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse # For command line arguments
import os # We learnt about this module!!!!
from simplepreprocessor import SimplePreprocessor # We created this module.
from simpledatasetloader import SimpleDatasetLoader # We created this module too!!!
from simpleflattenpreprocessor import SimpleFlattenPreprocessor # This one too!!!
ap = argparse.ArgumentParser()

#If we do not have a default value we set required to be True. Example path: ./datasets/animals
ap.add_argument("-d", "--dataset", help="path to input dataset", type=str, required=True)

# Since we have a default value, we do not need requrired=True
ap.add_argument("-k", "--neighbors", help="# of nearest neighbors for classification", type=int, default=1)

ap.add_argument("-c", "--width", help="width to resize the image", type=int, default=32)

ap.add_argument("-r", "--height", help="height to resize the image", type=int, default=32)

# Convert this to a dictionary
cmd_dict = vars(ap.parse_args())


# This is where we will use the os module which we learnt about. Hard-work wasn't for waste after all..:)
print("[INFO] loading images...")
dataset_path = cmd_dict['dataset']
imagePaths = []
validExtensions = ['jpg', 'jpeg', 'png', 'bmp']
for pathName, folderNames, fileNames in os.walk(dataset_path):
	for fileName in fileNames:
		if fileName.split(".")[-1] in validExtensions:
			imagePaths.append(pathName+"/"+fileName)

# print("imagePaths:",imagePaths)

new_width = cmd_dict['width']
new_height = cmd_dict['height']


sp = SimplePreprocessor(new_width, new_height)
sfp = SimpleFlattenPreprocessor()	
sdl = SimpleDatasetLoader(preprocessors = [sp,sfp]) # It is an ordered sequence. Order matters. First we resize then flatten.

# After every 500 iterations we would want to see the progress.
(data, labels) = sdl.load(imagePaths, verbose=500)
#print("data.shape", data.shape)
#print("Example string labels",labels[0:5])

# Information about the memory consumption of the image.
print("[INFO] feature matrix : {:.3f}MB".format(data.nbytes/(1024*1000.0))) # 3 digits after the decimal
# Map the string labels (class name) to integers.
le = LabelEncoder()
labels = le.fit_transform(labels) # le.classes_ attribute will have the corresponding string labels.
#print("Example integer labels", labels[0:5])


# partition the data into training and testing. 
# Generally, 75 percent is kept for training and 25 percent for testing.
# Since it is the Vanilla implementation, training and testing is done on the images directly. In practice, we extract features from the images
# and the training and testing data consists of featureVectors.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state = 42)
#print("trainX", trainX.shape)
#print("trainY", trainY.shape)
print("[INFO] evaluating k-NN classifier...")

# Just because we are using a library directly for knn, doesn't mean that you need not learn how the algorithm works.
# We use it because the implementation of this will be quite efficient.
model = KNeighborsClassifier(n_neighbors = cmd_dict["neighbors"])
model.fit(trainX, trainY)

# We are trying to predict on test data.
# Refer resources folder to learn about recall, precision and F1 as a performance measure. Support is same as frequency.
y_cap = model.predict(testX) # Predicted labels
y = testY # Actual labels
print(classification_report(y, y_cap, target_names = le.classes_))
