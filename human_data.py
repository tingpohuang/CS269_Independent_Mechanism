# import required libraries
import numpy as gfg
import matplotlib.image as img
import pandas as pd
from tqdm import tqdm

# read an image
imageMat = img.imread('./data/fairface/train/1.jpg')

# if image is colored (RGB)
if(imageMat.shape[2] == 3):

    # reshape it from 3D matrice to 2D matrice
    imageMat_reshape = imageMat.reshape(1,
                                        -1)
    print("Reshaping to 2D array:",
          imageMat_reshape.shape)

# if image is grayscale
else:
    # remain as it is
    imageMat_reshape = imageMat

# converting it to dataframe.
mat_df = pd.DataFrame(imageMat_reshape)

# 86745
for i in tqdm(range(2, 10000)):
    fileName = './data/fairface/train/{}.jpg'.format(i)
    imageMat = img.imread(fileName)
    imageMat_reshape = imageMat.reshape(-1)
    mat_df.loc[len(mat_df.index)] = imageMat_reshape.tolist()


print(mat_df.shape)
# exporting dataframe to CSV file.
mat_df.to_csv('race.csv',
              header=None,
              index=None)
