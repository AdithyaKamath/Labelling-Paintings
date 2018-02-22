import pandas as pd 
import numpy as np
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

path = "data/newtrain/"

path_train = "data/artist_train/"
path_test = "data/artist_test/"
data = pd.read_csv("data/all_data_info.csv")
X = data.groupby('artist').filter(lambda x: len(x) >= 300)
x = X['new_filename']
y = X['artist']

labels = []

i1 = 0
for i in range(x.shape[0]):
    image_name = path + str(x.iloc[i1])
    try:
        img = Image.open(image_name)
        labels.append(y.iloc[i1])
        i1 += 1
    except Exception as err:
        print("Image not found: " + str(image_name))
        print(err)
        x = x.drop(x.index[i1])
        y = y.drop(x.index[i1])

le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels) 
no_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes = no_classes) 

x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.2, stratify = labels , random_state = 2)

print("Saving train files")
for i in range(x_train.shape[0]):
    image_name = path + str(x.iloc[i])
    image_save = path_train + str(x.iloc[i])
    img = Image.open(image_name)
    img.save(image_save)
print("Saving test files")
for i in range(x_test.shape[0]):
    image_name = path + str(x.iloc[i])
    image_save = path_test + str(x.iloc[i])
    img = Image.open(image_name)
    img.save(image_save)

print("Saving labels")
np.save('labels_train.npy',labels_train)
np.save('labels_test.npy',labels_test)
x_train = np.array(x_train)
x_test = np.array(x_test)
np.save('x_train.npy',x_train)
np.save('x_test.npy',x_test)
