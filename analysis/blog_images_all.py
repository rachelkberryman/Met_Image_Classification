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
all_data = pd.read_csv("./data/CSVs/all_data.csv", index_col=0, lineterminator='\n')

# loading the image arrays
image_path = "./data/individual_objects_data/images/"

image_files = load_images(image_path)

all_data = all_data[all_data['primaryImageSmall']!='']

# getting list of all items
items_list = all_data.loc[:, 'objectID'].values
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

image_arrays = []
for image in item_image_files[:30000]:
    # using openCV for the pre-processing.
    image_arrays.append(cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), (200,200)))

# making the plots
fig = plt.figure(figsize=(50, 40))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(5, 10),
                 axes_pad=0,
                 )
plt.axis('off')
sample = np.random.choice(np.arange(len(image_arrays)), 50)
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

fig.savefig(f'./data/plots/ALL_items_grid_4.png', bbox_inches='tight')
plt.close(fig)


