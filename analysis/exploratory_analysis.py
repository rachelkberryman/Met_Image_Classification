import json
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

"""
Analyzing data from the Metropolitan Museum of Art in New York City. 
"""
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
data = pd.read_csv('./data/CSVs/all_data.csv', lineterminator='\n')
data.head()
# currently have about 130,000 rows.
data.shape

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
x = images_data.classification.value_counts().head(100)
paintings = images_data[images_data['classification']=='Paintings']
# there are about 4,000 paintings in this first go.
paintings.shape


# checking how many countries we have paintings from each country
# vast majority from the US. Others are france and england, very imbalanced
painting_cultures = paintings.culture.value_counts()
# selecting paintings from America, China, Japan
countries_list = ['American', 'Japan', 'China']
paintings_selected = paintings[paintings['culture'].isin(countries_list)]

# adding paintings from India
india_paintings = paintings[paintings['culture'].str.contains('India')]
india_paintings.loc[:, 'culture'] = "India"
paintings_final = pd.concat([paintings_selected, india_paintings])

paintings_final.to_csv("./data/CSVs/paintings_data.csv")
paintings_final.culture.value_counts()


paintings_final = pd.read_csv("./data/CSVs/paintings_data.csv", index_col=0)
american_paintings = paintings_final[paintings_final['culture']=='American'].objectID
ceramics = ceramics[ceramics['culture'].isin(countries_list)]




# doing the same checks, but for ceramics
ceramics = images_data[(images_data['classification']=='Ceramics')|
                       (images_data['classification']=='Ceramics-Vessels')|
                       (images_data['classification']=='Ceramics-Containers')]
ceramics.shape
# ceramics seem to be more balanced.
ceramics.country.value_counts()
ceramics.department.value_counts()
ceramics.culture.value_counts().head(15)
ceramics = ceramics[ceramics['country'].isin(countries_list)]
ceramics.to_csv("./data/CSVs/ceramics_data.csv")
paintings.culture.value_counts().head(15)
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

# generating dataset to randomly sample images from
paintings = pd.read_csv("./data/CSVs/paintings_data.csv")
paintings.columns
paintings_for_random_images = paintings.loc[:, ['culture', 'primaryImageSmall']]
paintings_for_random_images.head()
paintings_for_random_images.replace({"American": "America"}, inplace=True)
paintings_for_random_images.head()
paintings_for_random_images.to_csv("./data/CSVs/paintings_random_image_data.csv")

