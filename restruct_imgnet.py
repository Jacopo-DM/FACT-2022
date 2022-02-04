from xml.dom import minidom
import os
import shutil
import pandas as pd
from tqdm import tqdm

"""
This script restructures the val folder of the ILSVC ImageNet dataset
to the same structure as the train folder.
"""


df = pd.read_csv('scouter/data/imagenet/LOC_val_solution.csv')

df['PredictionString'] = df['PredictionString'].apply(lambda x: x.split(" ")[0])
df = df.sort_values(by=['ImageId'])

dir = os.fsencode("scouter/data/imagenet/ILSVRC/Data/CLS-LOC/val")

for file in tqdm(os.listdir(dir)):

     filename = os.fsdecode(file)

     if "JPEG" not in filename:
         continue
     name = filename[:-5]


     row = df.loc[df['ImageId'] == name]

     folder_name = row['PredictionString'].values[0]

     isExist = os.path.exists("scouter/data/imagenet/ILSVRC/Data/CLS-LOC/val/" + folder_name)

     if not isExist:
         os.makedirs("scouter/data/imagenet/ILSVRC/Data/CLS-LOC/val/" + folder_name)


     shutil.move("scouter/data/imagenet/ILSVRC/Data/CLS-LOC/val/" + filename, "scouter/data/imagenet/ILSVRC/Data/CLS-LOC/val/" + folder_name + "/" + filename)


