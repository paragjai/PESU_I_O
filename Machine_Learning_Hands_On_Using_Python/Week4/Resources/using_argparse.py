# argparse is a module in python to accept command line arguments
import argparse

# Step1: Always instantiate an argument parser object
ap = argparse.ArgumentParser()

# Step2: Decide what arguments to be passed
# Suppose we are coding for k-nearest neighbors and we want the following to be passed as the argument:
# 1. path to the image dataset named 'animals'
# 2. Preprocessing each image in the dataset : height and width to which we want each image has to be resized to.
# 3. k value for knn.

# Step3: Amongst these how many of them are optional? How many of them are must? If optional, what is the default value? What is the datatype of the passed argument?
# 1. must, dtype:string
# 2. is optional. It will have a default value of height=32, width=32, dtype:tuple/list
# 3. must, dtype:int

# Step4: Add arguments you expect user to pass through command line window
ap.add_argument("-d", "--dataset", help="path to the image dataset", required=True, type=str)
ap.add_argument("-p", "--preprocess", help="height followed by width", default=[32, 32], nargs='+', type=int) # nargs ='+', and type=int makes ap.preprocess as a list #https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument
ap.add_argument("-k", "--kValue", help="k value for nearest neighbor", required=True, type=int)

# Step5: Convert the command line arguments as key-value pairs/dictionary.
cmdDict = vars(ap.parse_args())

print("----------Passed arguments printed as dictionary----------")
print(cmdDict)

# Step6: Run the code
# In general, we would run it as 
#	python using_argparse.py
# But since argument 1 and argument 3 are must to be passed, just using the above command will generate error.
# Command to be used to execute this file:
#	pass the path to be : "./datasets/animals"
#	pass the preprocessing dimensions as : 28 28
#	pass the k value as : 3
#	Cmd : python2 using_argparse.py -d ./datasets/animals -p 28 28 -k 3

# Note that:
# 	irrespective of type of the datatype expected for the argument they are all passed the same way
#	list elements are separated by spaces

