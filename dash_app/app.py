import dash
import base64
import random
import urllib
import pandas as pd
import requests
import model
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import flask
import glob
import os
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

import plotly.express as px
import pandas as pd

N_ROWS = 150
N_COLS = 150

# loading model and label encoder
model = models.load_model("./models/keras_model_1")
encoder_file = open('./models/encoder.pkl', 'rb')
encoder = pickle.load(encoder_file)

# Dash component wrappers
def Row(children=None, **kwargs):
    return html.Div(children, className="row", **kwargs)

app = dash.Dash(__name__)
image_url = 'https://images.metmuseum.org/CRDImages/as/web-large/DP153551.jpg'
# image_filename = './data/individual_objects_data/images/image_object_id_10426.jpg'

# app.layout = html.Div([
#     html.Img(src='data:image/png;base64,{}'.format("image".decode()))
# ])
app.layout = html.Div(className='container', children=[
    Row(html.H1("Metropolitan Museum of Art Paintings Classifier")),
    Row(html.Button("Random Image", id='button', n_clicks=0)),
    Row(id='random-image', style={'display': 'none'}),
    Row(id='image-culture', style={'display': 'none'}),
    Row(id="label-explanation"),
    Row(html.Img(id='image', src='image')),
    Row(id="prediction")
])

@app.callback(
    [Output('random-image', 'children'),
     Output('image-culture', 'children')],
     Input('button', 'n_clicks')
)
def generate_random_image(n_clicks):
    random_data = pd.read_csv('./dash_app/paintings_random_image_data.csv', index_col=0)
    image_data = random_data.iloc[random.randint(1, random_data.shape[0])]
    random_image = image_data['primaryImageSmall']
    image_culture = image_data['culture']
    return random_image, image_culture


@app.callback(
    Output('label-explanation', 'children'),
     Input('image-culture', 'children')
)
def generate_random_image(image_culture):
    return f"Correct image culture is {image_culture}"


@app.callback(Output("image", "src"),
     [Input('random-image', 'children')])
def plot_updated_image(random_image):
    image_file = urllib.request.urlretrieve(random_image)[0]
    encoded_image = base64.b64encode(open(image_file, 'rb').read())
    image = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return image

@app.callback(Output("prediction", "children"),
     [Input('random-image', 'children'),
      Input('image-culture', 'children')])
def generate_prediction(image_data, image_culture):
    image_file = urllib.request.urlretrieve(image_data)[0]
    image = cv2.resize(cv2.imread(image_file, cv2.IMREAD_COLOR), (N_ROWS, N_COLS), interpolation=cv2.INTER_CUBIC)
    x = np.array(image)

    # getting label
    label_map = {"America": 0,
                 "Japan": 1,
                 "China": 2,
                 "India": 3}

    label = label_map[image_culture]
    # applying encoder to the label
    y = encoder.transform(np.array(label).reshape(-1, 1))
    # apply re-scaling
    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    testing_datagen = test_data_gen.flow(x.reshape(1, 150, 150, 3), y.todense(), batch_size=1, shuffle=False)

    # generate prediction
    prediction = model.predict_generator(testing_datagen, steps=len(y.todense())).argmax()

    # return actual label and prediction
    prediction_label = list(label_map.keys())[prediction]
    return prediction_label


if __name__ == '__main__':
    app.run_server(debug=True)