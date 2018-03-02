import pandas as pd 
import numpy as np
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.utils import to_categorical

path = "data/newtrain/"
store_path = 'time/'
data = pd.read_csv("data/all_data_info.csv")
data_time = data[data['date'].notnull()]
df_time = data_time[pd.to_numeric(data_time['date'], errors='coerce').notnull()]
df_time['date'] = pd.to_numeric(data_time['date'], errors='coerce', downcast='integer')
df_time['date'] = df_time['date'].astype('int')

print(df_style.style.value_counts(), df_style.shape[0])

x = df_style.new_filename
y = df_style.style
labels = []

i1 = 0
for i in range(x.shape[0]):
    image_name = path + str(x.iloc[i1])
    try:
        img = image.load_img(image_name)
        labels.append(y.iloc[i1])
        mean += image.img_to_array(img)
        i1 += 1
    except Exception as err:
        print("Image not found: " + str(image_name))
        print(err)
        x = x.drop(x.index[i1])
        y = y.drop(y.index[i1])
    if i % 2000 == 0:
        print(i)
mean1 = mean/x.shape[0]
print(mean1[0][0])

xnp = np.array(x)
ynp = np.array(y)

x_train, x_test, labels_train, labels_test = train_test_split(xnp, ynp, test_size=0.2, stratify = ynp , random_state = 2)
print(x_train.shape, x_test.shape, labels_train.shape, labels_test.shape)
x_cv, x_test, labels_cv, labels_test = train_test_split(x_test, labels_test, test_size=0.5, stratify = labels_test , random_state = 2)
print(x_cv.shape, x_test.shape, labels_cv.shape, labels_test.shape)

np.save(store_path + 'x_train.npy', x_train)
np.save(store_path + 'x_test.npy', x_train)
np.save(store_path + 'x_cv.npy', x_cv)
np.save(store_path + 'y_train.npy', labels_train)
np.save(store_path + 'y_test.npy', labels_test)
np.save(store_path + 'y_cv.npy', labels_cv)
np.save(store_path + 'mean.npy',mean1)