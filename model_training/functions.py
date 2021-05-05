import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_images(path):
    # Put files into lists and return them as one list of size 4
    image_files = sorted([os.path.join(path, file)
         for file in os.listdir(path) if file.endswith('.jpg')])

    return image_files


def process_image_labels(csv_path, image_files):
    """
    Loads and processes CSV with labels for particular
    art type, labeling them by culture.
    Args:
        csv_path (str): path to the CSV with culture labels and metadata.
        image_files (list): List of all image files, to load & filter
    """
    # loads and processes CSV with labels.
    # loading the cermaic images data
    items_data = pd.read_csv(csv_path, index_col=0, lineterminator='\n')
    # getting object IDs
    items_list = items_data.loc[:, 'objectID'].values
    # selecting only the images of ceramics
    item_image_indexes = []
    item_image_files = []
    for file in image_files:
        start = file.find("id_") + len("id_")
        end = file.find(".jpg")
        substring = file[start:end]
        if int(substring) in items_list:
            item_image_files.append(file)
            item_image_indexes.append(int(substring))
        else:
            continue

    return items_data, item_image_indexes, item_image_files


def process_images(items_data, indexes, label_map, images_dict, n_rows, n_cols):
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
        label = items_data.loc[items_data['objectID'] == index]['culture'].values[0]
        # converting label to an int
        label = label_map[label]
        y.append(label)

    x, y = np.array(x), np.array(y)
    # encoding the target variable.
    # NNs require
    encoder = OneHotEncoder()
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1))
    # have to one-hot-encode the Y-values to get the Ys in the right shape

    return x, y_one_hot, encoder

