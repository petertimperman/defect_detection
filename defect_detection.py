"""
        First attempt at using keras and CUDA backed Tesnor flow to
        create a cnn to detect feature in steel 
        
        Using 10 images we will train the newtork against one loop.
        This is a proof of concept only.

        Test file are in the ../images directory 
        postive: raw positve images 
        aug_training_positive: postive images with some manipluated versions
        aug_training_files: .txt with coordinates of defect 
         
"""

#Import statements 
import numpy 
import sys 
import os 
from scipy import misc #For reading in images 
import pdb
from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras 


#error= open("error.txt", "w")
#sys.stderr = error
#Training images preproccessing 
data_path = '/home/ptimperman/defect_workspace/images/'
loop_files = dict() #Dictionary that maps image name to corordinates of defects  
train_data =list() #Dict that maps image name to image
#Import images
for test_file in os.listdir(data_path+'aug_training_files'):
        test_file = data_path+'aug_training_files/'+test_file
        test_name= os.path.splitext(os.path.basename(test_file))[0]
        test_data=numpy.loadtxt(test_file)
        loop_files[test_name] = test_data
train_names=loop_files.keys() #Filter out images with out data 
for raw_image in os.listdir(data_path+'aug_training_positive'):
        raw_image = data_path+'aug_training_positive/'+raw_image
        image_name= os.path.splitext(os.path.basename(raw_image))[0]
        if image_name in train_names: 
                image_matrix=misc.imread(raw_image)
                #pdb.set_trace()
                #print(image_matrix.shape)
                train_data.append( (image_matrix, loop_files[image_name][1,]))#Load only the first loop 
        else:
                print ("Could not find data file corresponding to: ", image_name)
#Convert into list of
x_set =  list()
y_set = list()
for data_pair in train_data:
        y_set.append(data_pair[1])
        x_set.append(data_pair[0])
        #print(data_pair[0].shape)
        #print(data_pair[1].shape)

#Set up model
x_train = numpy.array(x_set[0:10]).reshape((10,1024,1024,1)) 
x_test = numpy.array(x_set[11:21]).reshape((10,1024,1024,1))
#y_train = numpy.array(y_train[0:10]).reshape((10,4,1,1)) 
y_train= numpy.array(y_set[0:10])
y_test = numpy.array(y_set[11:21])

number_of_kernels = 40
nuerons_in_layer = 40 
epochs = 20 

model = Sequential()
model.add(Conv2D(number_of_kernels, kernel_size=(5,5), strides=(1,1), activation ='relu', data_format='channels_last',  input_shape=(1024,1024,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(nuerons_in_layer, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=.01), metrics=['accuracy'])
print("Model compiled.")
#pdb.set_trace()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
model.fit(x_train ,y_train,batch_size=2, epochs= epochs ,verbose = 1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print(score)
with open('single_loop_detection log.txt', 'a') as log_file:
        log_file.write("Number of kernels: " + str(number_of_kernels)+"\n")
        log_file.write("Nuerons in each layer: "+ str(nuerons_in_layer)+"\n")
        log_file.write("Epochs: " + str(epochs)+"\n")
        log_file.write("Score: " + str(score[1])+"\n") 



#Train network 




