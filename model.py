# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:50:42 2020

@author: hp
"""
import math
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

compute_step_epoch=lambda x:int(math.ceil(1.* x/32))#calculation of epoches step


cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=(64, 64, 3)))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Adding a second convolutional layer(to increase a success)
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])#loss='categorical_crossentropy'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('data_set/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# Creating the Test set
test_set = test_datagen.flow_from_directory('data_set/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

steps_of_epochs_training=compute_step_epoch(1452)
steps_of_epoch_test=compute_step_epoch(363)
# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit_generator(training_set,
                  steps_per_epoch = steps_of_epochs_training,#no. of images in our training set
                  epochs = 25,
                  validation_data =test_set,
                  validation_steps = steps_of_epoch_test)



model_json = cnn.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
cnn.save_weights('model-bw.h5')
print('Weights saved')