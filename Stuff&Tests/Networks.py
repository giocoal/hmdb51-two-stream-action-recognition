import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import os

path = Path.cwd()
print(path)

train_dataset = keras.preprocessing.image_dataset_from_directory(os.path.join(path, 'data\\test\\train'),
                                                            image_size=(224,224),
                                                            color_mode='rgb',
                                                            batch_size=16,
                                                            label_mode='categorical',
                                                            seed=1)

val_dataset = keras.preprocessing.image_dataset_from_directory(os.path.join(path, 'data\\test\\test'),
                                                            image_size=(224,224),
                                                            color_mode='rgb',
                                                            batch_size= 16,
                                                            label_mode='categorical',
                                                            seed=1)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomHeight(0.2),
    layers.RandomWidth(0.2),
    layers.Resizing(224,224)
])



num_classes = 3

model_spat = keras.models.Sequential()

model_spat.add(keras.layers.Conv2D(96, (7,7), strides = 2, input_shape=(224, 224, 3), activation = "relu"))
model_spat.add(keras.layers.BatchNormalization())
model_spat.add(keras.layers.MaxPooling2D((3,3), strides=2, padding="same"))

model_spat.add(keras.layers.ZeroPadding2D(padding = (1,1)))
model_spat.add(keras.layers.Conv2D(256, (5,5), strides = 2, activation='relu'))
model_spat.add(keras.layers.BatchNormalization())
model_spat.add(keras.layers.MaxPooling2D((3,3), strides=2, padding="same"))
          
model_spat.add(keras.layers.ZeroPadding2D(padding = (1,1)))
model_spat.add(keras.layers.Conv2D(512, (3,3), strides = 1, activation='relu'))

model_spat.add(keras.layers.ZeroPadding2D(padding = (1,1)))
model_spat.add(keras.layers.Conv2D(512, (3,3), strides = 1, activation='relu'))

model_spat.add(keras.layers.ZeroPadding2D(padding = (1,1)))
model_spat.add(keras.layers.Conv2D(512, (3,3), strides = 1, activation='relu'))
model_spat.add(keras.layers.MaxPooling2D((3,3), strides=2, padding="same"))

model_spat.add(keras.layers.Flatten())

model_spat.add(keras.layers.Dense(4096, activation='relu'))
model_spat.add(keras.layers.Dropout(0.5))      #valore dropout 0.5 oppure 0.9? paper li usa entrambi 

model_spat.add(keras.layers.Dense(2048, activation='relu'))
model_spat.add(keras.layers.Dropout(0.5))

model_spat.add(keras.layers.Softmax())

model_spat.add(keras.layers.Dense(num_classes, activation="softmax"))




num_classes = 3

model_mot= keras.models.Sequential()

data_augmentation

model_mot.add(keras.layers.Conv2D(96, (7,7), strides = 2, input_shape=(224, 224, 3), activation = "relu"))
model_mot.add(keras.layers.BatchNormalization())
model_mot.add(keras.layers.MaxPooling2D((3,3), strides=2, padding="same"))

model_mot.add(keras.layers.ZeroPadding2D(padding = (1,1)))
model_mot.add(keras.layers.Conv2D(256, (5,5), strides = 2, activation='relu'))
model_mot.add(keras.layers.BatchNormalization())
model_mot.add(keras.layers.MaxPooling2D((3,3), strides=2, padding="same"))
          
model_mot.add(keras.layers.ZeroPadding2D(padding = (1,1)))
model_mot.add(keras.layers.Conv2D(512, (3,3), strides = 1, activation='relu'))

model_mot.add(keras.layers.ZeroPadding2D(padding = (1,1)))
model_mot.add(keras.layers.Conv2D(512, (3,3), strides = 1, activation='relu'))

model_mot.add(keras.layers.ZeroPadding2D(padding = (1,1)))
model_mot.add(keras.layers.Conv2D(512, (3,3), strides = 1, activation='relu'))
model_mot.add(keras.layers.MaxPooling2D((3,3), strides=2, padding="same"))

model_mot.add(keras.layers.Flatten())

model_mot.add(keras.layers.Dense(4096, activation='relu'))
model_mot.add(keras.layers.Dropout(0.5))

model_mot.add(keras.layers.Dense(2048, activation='relu'))
model_mot.add(keras.layers.Dropout(0.5))

model_mot.add(keras.layers.Softmax())

model_mot.add(keras.layers.Dense(num_classes, activation="softmax"))



model_spat.compile(loss=keras.losses.categorical_crossentropy, metrics=['accuracy'], 
              optimizer=keras.optimizers.RMSprop(learning_rate=0.001))




history = model_spat.fit(train_dataset,  epochs=2, 
                    validation_data=val_dataset)