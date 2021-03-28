import random
import urllib
import requests
import pickle
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
N_ROWS = 150
N_COLS = 150

# loading model and label encoder
model = models.load_model("./models/keras_model_1")
encoder_file = open('./models/encoder.pkl', 'rb')
encoder = pickle.load(encoder_file)

# import data with links and labels for images
random_data = pd.read_csv("./data/CSVs/paintings_random_image_data.csv", index_col=0)

# select a random one
image_data = random_data.iloc[random.randint(1, random_data.shape[0])]

# load image
image_file = urllib.request.urlretrieve(image_data['primaryImageSmall'])[0]
image = cv2.resize(cv2.imread(image_file, cv2.IMREAD_COLOR), (N_ROWS, N_COLS), interpolation=cv2.INTER_CUBIC)
x = np.array(image)

# getting label
label_map = {"America": 0,
             "Japan": 1,
             "China": 2,
             "India": 3}

label = label_map[image_data['culture']]
# applying encoder to the label
y = encoder.transform(np.array(label).reshape(-1, 1))
# apply re-scaling
test_data_gen = ImageDataGenerator(rescale=1./255)
testing_datagen = test_data_gen.flow(x.reshape(1, 150, 150, 3), y.todense(), batch_size=1, shuffle=False)

# generate prediction
prediction = model.predict_generator(testing_datagen, steps=len(y.todense())).argmax()

# return actual label and prediction
prediction_label = list(label_map.keys())[prediction]

