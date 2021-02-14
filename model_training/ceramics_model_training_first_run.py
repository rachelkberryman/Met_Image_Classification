import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
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


def load_images(path):
    # Put files into lists and return them as one list of size 4
    image_files = sorted([os.path.join(path, file)
         for file in os.listdir(path) if file.endswith('.jpg')])

    return image_files

image_files = load_images(image_path)

# loading the cermaic images data
data = pd.read_csv("./data/CSVs/ceramics_data.csv", index_col=0, lineterminator='\n')
# taking only the china, japan & US examples
culture_indexes = data['culture'].isin(['China', 'Japan', 'American', 'Chinese'])
ceramics_data = data.loc[culture_indexes]
ceramics_list = ceramics_data.loc[:, 'objectID'].values
# selecting only the images of ceramics
ceramic_image_indexes = []
ceramic_image_files = []
for file in image_files:
    start = file.find("id_") + len("id_")
    end = file.find(".jpg")
    substring = file[start:end]
    if int(substring) in ceramics_list:
        ceramic_image_files.append(file)
        ceramic_image_indexes.append(int(substring))
    else:
        continue

# making a dictionary to access the correct images later
images_dict = {k: v for k, v in zip(ceramic_image_indexes, ceramic_image_files)}
# randomly shuffling the indexes
# needed because many of the items from different countries have indexes that are close to oneanother.
random.shuffle(ceramic_image_indexes)
# splitting out a test set
# total len is 5,987. 4800 includes a roughly 20% test set.
train_indexes = ceramic_image_indexes[:4800]
test_indexes = ceramic_image_indexes[4800:]

# Reading and processing images
# defining new image size
# 3 = number of channels, since many of the images are RGB.
n_rows = 150
n_cols = 150
n_channels = 3

label_map = {
    "China": 0,
    "Chinese": 0,
    "Japan": 1,
    "American": 2
}


def load_and_preprocess(indexes, images_dict):
    """
    Args:
        indexes (list): List of which indexes to select from dict and pre-process
        images_dict (list): dict of format {image_id: image_file}
    Returns:
        x (array): training features, array of pixels for each image
        y (array): corresponding labels.
    """
    x = []
    y = []

    for index in indexes:
        image = images_dict[index]
        # using openCV for the pre-processing.
        x.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (n_rows, n_cols), interpolation=cv2.INTER_CUBIC))
        label = ceramics_data.loc[ceramics_data['objectID']==index]['culture'].values[0]
        # converting label to an int
        label = label_map[label]
        y.append(label)

    x, y = np.array(x), np.array(y)

    return x, y

x, y = load_and_preprocess(ceramic_image_indexes, images_dict)

encoder = OneHotEncoder()
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))
# have to one-hot-encode the Y-values to get the Ys in the right shape

# test_x, test_y = load_and_preprocess(test_indexes, images_dict)
train_x, test_x, train_y, test_y = train_test_split(x, y_one_hot, test_size=0.2, random_state=42)


# getting the length of each one, to use for the model later.
n_train = len(train_y)
n_test = len(test_x)

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

# Generating additional datasets, with flips, added blur, etc.
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

# training the generators
training_datagen = train_data_gen.flow(train_x, train_y.todense(), batch_size=batch_size)
testing_datagen = test_data_gen.flow(test_x, test_y.todense(), batch_size=batch_size)

## Starting training!! :D
history = model.fit_generator(training_datagen,
                              steps_per_epoch=n_train//batch_size,
                              epochs=64,
                              validation_data=testing_datagen,
                              validation_steps=n_test//batch_size)

# getting label of actuals
actuals = [x.argmax() for x in test_y]
# getting predictions
predictions = []
for index, test_label in enumerate(test_y.todense()):
    predictions.append(model.predict(test_x[index].reshape(1, 150, 150, 3)).argmax())

# saving actuals and predictions, to be able to work with later.
predictions_dict = {"actuals": actuals, "predictions":predictions}
predictions_df = pd.DataFrame(predictions_dict, index=range(len(predictions)))
predictions_df.to_csv("./data/CSVs/test_predictions_model_1.csv")
actuals
predictions


# plotting prediction accuracy

# TODO: retrain without image generation
