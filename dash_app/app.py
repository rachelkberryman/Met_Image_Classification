import base64
import random
import urllib
import model
import dash
import pickle
import cv2
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from keras import models
from keras.preprocessing.image import ImageDataGenerator

# setting dimensions of image for inputting into model
N_ROWS = 150
N_COLS = 150

# loading model and label encoder
model = models.load_model("./models/keras_model_1")
encoder_file = open('./models/encoder.pkl', 'rb')
encoder = pickle.load(encoder_file)

# initiating dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = html.Div(className='container', children=[
    dbc.Row(dbc.Col(html.H1("Metropolitan Museum of Art Paintings Classifier"), style={'textAlign': 'center'})),
    dbc.Row(html.H1(" ")),
    dbc.Row(dbc.Col(html.Font(
        "Click the button below to generate a random photo from the Metroploitan Museum of Art's collection, and to predict its culture."),
                    style={'textAlign': 'center', 'backgroundColor': 'lightgrey'}, width=4), justify='center'),
    dbc.Row(html.H1(" ")),
    dbc.Row(dbc.Col(dbc.Button("Generate Random Image", id='button', n_clicks=0, color='danger', size='lg'),
                    style={'textAlign': 'center'})),
    dbc.Row(id='random-image', style={'display': 'none'}),
    dbc.Row(id='image-culture', style={'display': 'none'}),
    dbc.Row(html.H1(" ")),
    dbc.Row(html.H1(" ")),
    dbc.Row(dbc.Col(id="label-explanation", style={'textAlign': 'center'})),
    dbc.Row(html.H1(" ")),
    dbc.Row(dbc.Col(html.Img(id='image', src='image'), width=6), justify='center'),
    dbc.Row(html.H1(" ")),
    dbc.Row(html.H1(" ")),
    dbc.Row(dbc.Col(id="prediction", style={'textAlign': 'center'})),
    dbc.Row(html.H1(" ")),
    dbc.Row(html.H1(" ")),
    dbc.Row(html.H1(" ")),
    dbc.Row(html.H1(" ")),
    dbc.Row(dbc.Col(html.Img(id='logo', src='logo', style={'height': '85%', 'width': '7%'})), justify='start')
], style={'align-items': 'center', 'justify-content': 'center'})


@app.callback(
    [Output('random-image', 'children'),
     Output('image-culture', 'children')],
    Input('button', 'n_clicks')
)
def generate_random_image(n_clicks):
    # loading CSV of data (list of links to painting images online)
    random_data = pd.read_csv('./dash_app/paintings_random_image_data.csv', index_col=0)

    # selecting a random image from the data
    image_data = random_data.iloc[random.randint(1, random_data.shape[0])]

    # getting image data and string of culture name
    random_image = image_data['primaryImageSmall']
    image_culture = image_data['culture']
    return random_image, image_culture


@app.callback(
    Output('label-explanation', 'children'),
    Input('image-culture', 'children')
)
def generate_random_image(image_culture):
    # printing image culture name
    to_return = html.P(children=[
        html.Span("Correct image culture is: "),
        html.Strong(image_culture)
    ], style={'fontSize': '18px'})
    return to_return


@app.callback(Output("image", "src"),
              [Input('random-image', 'children')])
def plot_updated_image(random_image):
    # processing image to display it
    image_file = urllib.request.urlretrieve(random_image)[0]
    encoded_image = base64.b64encode(open(image_file, 'rb').read())
    image = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return image


@app.callback(Output("prediction", "children"),
              [Input('random-image', 'children'),
               Input('image-culture', 'children')])
def generate_prediction(image_data, image_culture):
    # loading image data
    image_file = urllib.request.urlretrieve(image_data)[0]

    # resizing & pre-processing image data for the model
    image = cv2.resize(cv2.imread(image_file, cv2.IMREAD_COLOR), (N_ROWS, N_COLS), interpolation=cv2.INTER_CUBIC)

    # converting image data to an array
    x = np.array(image)

    # getting label
    label_map = {"America": 0,
                 "Japan": 1,
                 "China": 2,
                 "India": 3}

    # getting integer label using label map
    label = label_map[image_culture]

    # applying encoder to the label
    y = encoder.transform(np.array(label).reshape(-1, 1))

    # apply re-scaling
    test_data_gen = ImageDataGenerator(rescale=1. / 255)

    # applying generator to resize the image
    testing_datagen = test_data_gen.flow(x.reshape(1, 150, 150, 3), y.todense(), batch_size=1, shuffle=False)

    # generate prediction
    prediction = model.predict_generator(testing_datagen, steps=len(y.todense())).argmax()

    # return actual label and prediction
    prediction_label = list(label_map.keys())[prediction]

    # generating color. If correct, shows prediction green. If wrong, shows it red.
    if prediction_label == image_culture:
        color = "green"
    else:
        color = 'red'

    to_return = html.P(children=[
        html.Span("The model's prediction for this painting is: "),
        html.Strong(prediction_label, style={'color': color})
    ], style={'fontSize': '18px'})
    return to_return


@app.callback(Output("logo", "src"),
              [Input('random-image', 'children')])
def plot_logo(random_image):
    # adding logo to the bottom of the app
    encoded_image = base64.b64encode(open('./dash_app/assets/logo.png', 'rb').read())
    image = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return image


if __name__ == '__main__':
    app.run_server(debug=True)
