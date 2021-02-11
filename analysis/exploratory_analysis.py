import json
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

"""
Analyzing data from the Metropolitan Museum of Art in New York City. 
"""
# Downloading each of the JSON files
path = Path("/Users/rachelberryman/Desktop/git/Met_Image_Classification/data/individual_objects_data/")
files = []
for file in tqdm(path.iterdir()):
    if file.is_dir():
        continue
    else:
        try:
            with open(file, 'r', encoding='ASCII') as fi:
                data = json.load(fi)
                files.append(data)
        except UnicodeDecodeError:
            print(f'Error file name{file}')
            continue

data = pd.DataFrame(files)
# saving as CSV
data.to_csv('./data/CSVs/first_go_data.csv')
data.head()
# currently have about 130,000 rows.
data.shape
data = pd.read_csv("./data/CSVs/first_go_data.csv", index_col=0, lineterminator='\n')

# checking how many should have images.
# answer is 73,722
count = 0
for x in data.loc[:, 'primaryImageSmall'].values:
    if x == "":
        continue
    else:
        count += 1
count

# sorting data to only include data with images
images_data = data[data['primaryImageSmall']!='']
images_data.shape

# sorting by only paintings
images_data.classification.unique()
paintings = images_data[images_data['classification']=='Paintings']
# there are about 4,000 paintings in this first go.
paintings.shape

# checking how many countries we have paintings from each country
# vast majority from the US. Others are france and england, very imbalanced
paintings.country.value_counts()

# doing the same checks, but for ceramics
ceramics = images_data[images_data['classification']=='Ceramics']
ceramics.shape
# ceramics seem to be more balanced.
ceramics.country.value_counts()
ceramics.department.value_counts()
ceramics.culture.value_counts().head(15)
ceramics = ceramics[ceramics['country'].isin(countries_list)]
ceramics.to_csv("./data/CSVs/ceramics_data.csv")

# doing the same checks, but for ceramics
jewelry = images_data[images_data['classification']=='Jewelry']
jewelry.shape
# ceramics seem to be more balanced.
jewelry.country.value_counts()

# seeing which class of art has the highest number of images
images_data.classification.value_counts().head(25)
drawings = images_data[images_data['classification']=='Drawings']
drawings.country.value_counts()

prints = images_data[images_data['classification']=='Prints']
prints.country.value_counts()
