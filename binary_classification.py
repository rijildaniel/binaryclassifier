"""
    Topic: Binary classification of images
"""

"""
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip 
    Download the sample dataset:
    Train dataset path: /tmp/cats_and_dogs_filtered/cats_and_dogs_filtered/train
    Validation Dataset path: /tmp/cats_and_dogs_filtered/cats_and_dogs_filtered/validation
"""


#This is part is optional-----------------------------
"""
import os
import zipfile

file = input("Enter the zip file to extract: ")
zf = zipfile.ZipFile(file, 'r')
zf.extractall(file)
zf.close()
"""
#-------------------------------------------------------

#You can start from here
#Import the dataset for training the model........................................
path1 = str(input("Enter the path to directory containing image dataset(Folder name must be named after image name): "))
train_dst = os.path.join(path1)
groups = os.listdir(train_dst)
if(len(groups) !=2):
  exit(-1)

#For checking the model performance and increase it.......................................
dec = input("Do you have validation data?(y/n): ")
if(dec == 'y' or dec == 'Y'):
  path2 = input("Enter the path to directory containing validation image dataset(Folder name must be named after image name): ")
  validation_dst = os.path.join(path2)
  if(len(os.listdir(validation_dst)) !=2):
    exit(-1)

print("Total Training Dataset: ")
print(groups[0] + " = " + str(len(os.listdir(train_dst + "/" + groups[0]))))
print(groups[1] + " = " + str(len(os.listdir(train_dst + "/" + groups[1]))))
print("Total Validation Dataset:")
print(groups[0] + " = " + str(len(os.listdir(validation_dst + "/" + groups[0]))))
print(groups[1] + " = " + str(len(os.listdir(validation_dst + "/" + groups[1]))))
print()

import tensorflow as tf

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512,activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                   ])

model.summary()

#RMSprop is efficient gradient descent algorithm
from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001),metrics=['acc'])

#Preprocessing of images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255
trainDataGen = ImageDataGenerator(rescale=1/255)
validationDataGen = ImageDataGenerator(rescale=1/255)
train_generator = trainDataGen.flow_from_directory(
                    path1,
                    target_size=(300,300),
                    batch_size=128,
                    class_mode='binary'    # Since we use binary_crossentropy loss, we need binary labels
                  )

validation_generator = validationDataGen.flow_from_directory(
                    path2,
                    target_size=(300,300),
                    batch_size=32,
                    class_mode='binary'    # Since we use binary_crossentropy loss, we need binary labels
                  )

history = model.fit_generator(train_generator,
                              steps_per_epoch=8,
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator,
                              validation_steps=8
                             )


import numpy as np
from keras.preprocessing import image
file = input("Enter an image to test: ")
x = image.load_img(file,target_size=(300,300))   #loading image of size 300x300
x = image.img_to_array(x)                   #loading
x = np.expand_dims(x,axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)

if classes[0]>0.5:
   print(file + ' is ' + groups[0] + ' image')
else:
    print(file + ' is ' + groups[1] + ' image')