# Met Image Classification
Image classification of the art archives from the Metropolitan Museum of Art in New York.

## About the Project
The goal of this project is to apply machine learning to the treasure trove of freely available data from the Metropolitan Museum of Art in New York City. "The Met", as the museum is commonly called, has a [free web API](https://metmuseum.github.io). The dataset contains metadata about the works of art (year created, country of origin, artist, etc). Many of the datapoints also contain a link to an image of the item.

A sampling of the images from this dataset is shown below.
![artworks](./assets/ALL_items_grid_4.png)

Using the data for paintings contained in the dataset, the goal of the project was to predict from which of 4 possible countries a painting came: **China, Japan, America** or **India**, using only the image pixels as input data (i.e., not including any metadata as input).

Once the model was trained, the next step in the project was to create an interactive front-end of this classifier, using Plotly Dash. A screenshot of this front-end application is shown below.
![app](./assets/app.png)

### Running the Code
To download the dataset used in training the classifier, run `python3 data_acquisition/api_access.py`.

To train the paintings classifier, run `python3 model_training/paintings_model_training.py`.

To start up the Dash app locally, run `python3 dash_app/app.py`.

### The Paintings Classifier Web App
You can check out the final app [here](https://met-paintings-classifier.herokuapp.com).
The code for the app, built in Plotly Dash, is found in the folder `./app/`

### More Info
For more information about this project, see the medium posts about it: 

1. [End-to-End Image Recognition With Open Source Data — Part 1: Data Acquisition & Model Training](https://data4help.medium.com/end-to-end-image-recognition-with-open-source-data-part-1-data-acquisition-model-training-fe9f4be9b915?sk=6bf42693c15f57b69ef0d3bf50f70a5f)
2. [End-to-End Image Recognition With Open Source Data — Part 2: Model Deployment with Plotly Dash and Heroku](https://data4help.medium.com/end-to-end-image-recognition-with-open-source-data-part-2-model-deployment-with-plotly-dash-and-3c8608b99faa)
