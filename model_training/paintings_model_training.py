import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
paintings_csv_path = "./data/CSVs/paintings_data.csv"

# loading ceramics images
image_files = load_images(image_path)
# loading and processing labels data
paintings_data, paintings_image_indexes, paintings_image_files = process_image_labels(paintings_csv_path,
                                                                                      image_files)

# making a dictionary to access the correct images later
images_dict = {k: v for k, v in zip(paintings_image_indexes, paintings_image_files)}

# randomly shuffling the indexes
# needed because many of the items from different countries have indexes that are close to oneanother.
random.shuffle(paintings_image_indexes)

# Reading and processing images
# defining new image size
# 3 = number of channels, since many of the images are RGB.
n_rows = 150
n_cols = 150
n_channels = 3

# Mapping the labels to a numeric value
label_map = {
    "American": 0,
    "Japan": 1,
    "China": 2,
    "India": 3
}

# processing images and splitting into X and Y
# Also pre-processing the target variable, so that it's one-hot encoded.
x, y, encoder = process_images(paintings_data,
                               paintings_image_indexes,
                               label_map,
                               images_dict,
                               n_rows,
                               n_cols)
# saving the item ID to be able to access the image later
y_with_index = pd.DataFrame(index=np.array(paintings_image_indexes), data=y.todense())
# holding out 1/3 or data as test + validation
train_x, test_x, train_y, test_y = train_test_split(x, y_with_index, test_size=0.3, random_state=42)

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
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(4, activation='softmax'))
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
training_datagen = train_data_gen.flow(train_x, train_y.values, batch_size=batch_size)
val_datagen = test_data_gen.flow(val_x, val_y.values, batch_size=batch_size)

# final holdout set, batch size is smaller to only use true data
testing_datagen = test_data_gen.flow(test_x, test_y.values, batch_size=1, shuffle=False)

## Starting training!! :D
history = model.fit_generator(training_datagen,
                              steps_per_epoch=n_train//batch_size,
                              epochs=100,
                              validation_data=val_datagen,
                              validation_steps=n_val//batch_size)

# getting label of actuals, taking the one-hot-encoded location
# CHECK: Is argmax the best way to do this??
actuals = [x.argmax() for x in test_y]
# getting predictions
predictions = []
for i, test_label in enumerate(test_y.todense()):
    predictions.append(model.predict(test_x[i].reshape(1, 150, 150, 3)).argmax())

# saving actuals and predictions, to be able to work with later.
predictions_dict = {"actuals": actuals, "predictions": predictions}
predictions_df = pd.DataFrame(predictions_dict, index=range(len(predictions)))
predictions_df.to_csv("./data/CSVs/test_predictions_paintings_model_100_episodes.csv")

# Predictions step 2:
model_predictions = model.predict_generator(testing_datagen, steps=len(test_y.todense()))
test_predictions = [i.argmax() for i in model_predictions]
test_predictions
actuals


test_y.todense()[0]

train_actuals = [x.argmax() for x in train_y]
# getting predictions
train_predictions = []
for i, train_label in enumerate(train_y.todense()):
    train_predictions.append(model.predict(train_x[i].reshape(1, 150, 150, 3)).argmax())

# saving actuals and predictions, to be able to work with later.
train_predictions_dict = {"actuals": train_actuals, "predictions": train_predictions}
train_predictions_df = pd.DataFrame(train_predictions_dict, index=range(len(train_predictions)))

# saving model, image_gens
model.save("./models/keras_model_1")

# pickling encoder
encoder
output = open('./models/encoder.pkl', 'wb')
pickle.dump(encoder, output)
output.close()

# plot loss during training
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# plot accuracy during training
ax.set_title('Accuracy')
ax.plot(history.history['accuracy'], label='Train Accuracy', color='dimgrey')
ax.plot(history.history['val_accuracy'], label='Test Accuracy', color='deepskyblue')
ax.legend()
fig.savefig(f'./data/plots/training_plots_1.png', bbox_inches='tight')
plt.close(fig)

