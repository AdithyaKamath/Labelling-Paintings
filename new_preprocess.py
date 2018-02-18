import pandas as pd 
from keras.preprocessing import image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

path = "C:/Users/lenovo/Documents/CS/6Sem/CG/newtrain/"
path_save = "C:/Users/lenovo/Documents/CS/6Sem/CG/"

data = pd.read_csv("C:/Users/lenovo/Documents/CS/6Sem/CG/all_data_info.csv")

X = data.groupby('artist').filter(lambda x: len(x) >= 300)

#labels = X['artist']

print("Size of data: " + str(X.shape[0]))
images = []
y = []

count = 0

for row in X.itertuples():
    image_name = path + str(row[12])  # 11 --> filename
    #print(row[12])
    try:
        img = image.img_to_array(image.load_img(image_name, target_size = (224,224)))
        images.append([image_name,image.img_to_array(img)])
        y.append(row[1])
        #x[count] = img
        count+=1
    
    except:
        print("Unable to find: " + image_name)
    
    if count % 100 == 0:
        print(count)

print("Transforming data")
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y) 

y = np.array(y)
x = np.array(images)
images = []

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, stratify = y, shuffle = True)

x = []
y = []

print("Saving Train files")
np.save(path_save + 'Temp/x_train.npy',x_train)
np.save(path_save + 'Temp/y_train.npy',y_train)
print("Saving Test files")
np.save(path_save + 'Temp/x_test.npy',x_test)
np.save(path_save + 'Temp/y_test.npy',y_test)



