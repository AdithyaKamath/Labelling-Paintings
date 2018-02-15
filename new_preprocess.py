import pandas as pd 
from keras.preprocessing import image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

path = ""

data = pd.read_csv("alldata.csv")

X = data.groupby('artist').filter(lambda x: len(x) >= 300)

#labels = X['artist']

x = np.array()
y = np.array()

for row in X.itertuples():
    image_name = path + row['new_filename']
    try:
        img = image.load_img(image_name, target_size = (224,224))
        y.append(row['artist'])
        x.append(res)
    
    except:
        print("Unable to find: " + row['new_filename'])

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, stratify = y, shuffle = True)

np.save(path+'Temp/x_train.npy',x_train)
np.save(path+'Temp/y_train.npy',y_train)

np.save(path+'Temp/x_test.npy',x_test)
np.save(path+'Temp/y_test.npy',y_test)



