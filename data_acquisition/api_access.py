import json
import requests
import urllib.request

# getting all of the objects
all_objects_request = requests.get("https://collectionapi.metmuseum.org/public/collection/v1/objects")
all_objects = all_objects_request.json()

# saving the objects JSON
with open('./data/all_objects_data/all_object_IDs.json', 'w') as fi:
    json.dump(all_objects, fi)

# for each object, getting its data
object_ids_list = all_objects['objectIDs']
data = []
for i in object_ids_list:
    # getting request response for each object
    individual_object_request = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{i}")
    individual_object_data = individual_object_request.json()
    # saving to list to add to dataframe later
    data.append(individual_object_data)
    # saving request response for each object
    with open(f'./data/individual_objects_data/data_object_id_{i}.json', 'w') as fi:
        json.dump(individual_object_data, fi)

    # not all items have an image.
    # checking if the item has an image. If so, saving it. If not, adding the object ID to a list.
    no_image_items = []
    image_items = []
    if bool(individual_object_data['primaryImageSmall']):
        # some image links contain a space and are not readable.
        if " " in individual_object_data['primaryImageSmall']:
            no_image_items.append(i)
            continue
        else:
            # for readable links, retrieving the image and saving it.
            urllib.request.urlretrieve(individual_object_data['primaryImageSmall'],
                                       f"./data/individual_objects_data/images/image_object_id_{i}.jpg")
            image_items.append(i)
    else:
        no_image_items.append(i)









