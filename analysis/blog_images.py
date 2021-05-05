import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def load_images(path):
    # Put files into lists and return them as one list of size 4
    image_files = sorted([os.path.join(path, file)
         for file in os.listdir(path) if file.endswith('.jpg')])

    return image_files

# getting the painting metadata
paintings = pd.read_csv("./data/CSVs/paintings_data.csv", index_col=0)

# loading the image arrays
image_path = "./data/individual_objects_data/images/"

image_files = load_images(image_path)
# splitting into the 4 different cultures
cultures = ["American", "China", "Japan", "India"]

# initiating dict to hold the data for the different cultures
cultures_dict = {}
for culture in cultures:
    # getting just the data for that culture
    culture_data = paintings.loc[paintings['culture']==culture]
    items_list = culture_data.loc[:, 'objectID'].values
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
    cultures_dict[culture] = item_image_files

# generating the images
for culture, data in cultures_dict.items():
    image_arrays = []
    for image in data:
        # using openCV for the pre-processing.
        image_arrays.append(cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), (200,200)))
    # making the plots
    fig = plt.figure(figsize=(30, 10))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 5),
                     axes_pad=0,
                     )
    plt.axis('off')
    sample = np.random.choice(np.arange(len(image_arrays)), 10)
    example_images = []
    for i in sample:
        example_images.append(image_arrays[i])
    for ax, im in zip(grid, example_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(f'./data/plots/{culture}_paintings_grid.png', bbox_inches='tight')
    plt.close(fig)


