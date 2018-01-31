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
import os 
from scipy import misc #For reading in images 
import pdb
from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras 
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
x_train =  list()
y_train = list()
for data_pair in train_data:
        y_train.append(data_pair[1])
        x_train.append(data_pair[0])
        #print(data_pair[0].shape)
        #print(data_pair[1].shape)

#Set up model
x_train = numpy.array(x_train[0:10]).reshape((10,1024,1024,1)) 

#y_train = numpy.array(y_train[0:10]).reshape((10,4,1,1)) 
y_train= numpy.array(y_train[0:10])
model = Sequential()
model.add(Conv2D(32, kernel_size=(265,265), strides=(1,1), activation ='relu', data_format='channels_last',  input_shape=(1024,1024,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=.01), metrics=['accuracy'])
print("Model compiled.")
#pdb.set_trace()
print(x_train.shape)
print(y_train.shape)
model.fit(x_train ,y_train,batch_size=2, epochs=10, verbose = 1, validation_data=(numpy.array(x_train[11:21]), numpy.array(y_train[11:21])))
score = model.evaluate(x_train[11:21], y_train[11:21], verbose=1)
print(score[0])
print(score[1])

        



#Train network 




