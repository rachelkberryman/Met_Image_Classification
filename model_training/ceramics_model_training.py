import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from model_training.functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

# Getting images
# defining global variable path
image_path = "./data/individual_objects_data/images/"
cultures = ['China', 'Japan', 'American', 'Chinese']
ceramics_csv_path = "./data/CSVs/ceramics_data.csv"

# loading ceramics images
image_files = load_images(image_path)
# loading and processing labels data
ceramics_data, ceramic_image_indexes, ceramic_image_files = process_image_labels(ceramics_csv_path,
                                                                                 image_files,
                                                                                 cultures)

# making a dictionary to access the correct images later
images_dict = {k: v for k, v in zip(ceramic_image_indexes, ceramic_image_files)}

# randomly shuffling the indexes
# needed because many of the items from different countries have indexes that are close to oneanother.
random.shuffle(ceramic_image_indexes)

# Reading and processing images
# defining new image size
# 3 = number of channels, since many of the images are RGB.
n_rows = 150
n_cols = 150
n_channels = 3

# Mapping the labels to a numeric value
label_map = {
    "China": 0,
    "Chinese": 0,
    "Japan": 1,
    "American": 2
}

# processing images and splitting into X and Y
# Also pre-processing the target variable, so that it's one-hot encoded.
x, y = process_images(ceramics_data,
                      ceramic_image_indexes,
                      label_map,
                      images_dict,
                      n_rows,
                      n_cols)

# holding out 1/3 or data as test + validation
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)

# splitting test set into test and validation
# 12% of original data is holdout test.
# 18% of original data is validation (used for model training).
val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.4, random_state=42)

# getting the length of each one, to use for the model later.
n_train = train_x.shape[0]
n_val = val_x.shape[0]

# setting a batch size.
batch_size = 32

# setting up the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# generating additional datasets, with flips, added blur, etc.
# keeping rotation, width and height shifts relatively small, since most of
# the images are very clear.
train_data_gen = ImageDataGenerator(
    rescale=1./255, # rescaling b/w 0 & 1
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False
)

# for test data, only doing rescaling.
test_data_gen = ImageDataGenerator(rescale=1./255)

# setting up the generators
training_datagen = train_data_gen.flow(train_x, train_y.todense(), batch_size=batch_size)
testing_datagen = test_data_gen.flow(val_x, val_y.todense(), batch_size=batch_size)

## Starting training!! :D
history = model.fit_generator(training_datagen,
                              steps_per_epoch=n_train//batch_size,
                              epochs=100,
                              validation_data=testing_datagen,
                              validation_steps=n_val//batch_size)

# getting label of actuals, taking the one-hot-encoded location
actuals = [x.argmax() for x in test_y]
# getting predictions
predictions = []
for i, test_label in enumerate(test_y.todense()):
    predictions.append(model.predict(test_x[i].reshape(1, 150, 150, 3)).argmax())

# saving actuals and predictions, to be able to work with later.
predictions_dict = {"actuals": actuals, "predictions":predictions}
predictions_df = pd.DataFrame(predictions_dict, index=range(len(predictions)))
predictions_df.to_csv("./data/CSVs/test_predictions_model_2_100_episodes.csv")

