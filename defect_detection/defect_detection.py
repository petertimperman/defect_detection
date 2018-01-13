"""
	First attempt at using keras and CUDA backed Tesnor flow to
	create a cnn to detect feature in steel 
	Test file are in the ../images directory 
	postive: raw positve images 
	aug_training_positive: postive images with some manipluated versions
	aug_training_files: .txt with coordinates of defect 
	 
"""

#Import statements 
import numpy
import keras 
import os 
from scipy import misc #For reading in images 




#Training images preproccessing 
data_path = '/home/rave/defect_workspace/images/'
training_dict = dict() 
test_dict = dict()
#Import images
for test_file in os.listdir(data_path+'aug_training_files'):
	test_file = data_path+'aug_training_files/'+test_file
	test_name= os.path.splitext(os.path.basename(test_file))[0]
	test_data=numpy.loadtxt(test_file)
	test_dict[test_name] = test_data
test_names=test_dict.keys()
for raw_image in os.listdir(data_path+'aug_training_positive'):
	raw_image = data_path+'aug_training_positive/'+raw_image
	image_name= os.path.splitext(os.path.basename(raw_image))[0]
	if image_name in test_names: 
		image_matrix=misc.imread(raw_image)
		training_dict[image_name] = image_matrix
	else:
		print ("Could not find data file corresponding to: ", image_name)





	



#Train network 




