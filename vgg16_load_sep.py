import pandas as pd 
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.layers import Dense, GlobalAveragePooling2D,Input,Flatten,Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping

path = "data/newtrain/"
data = pd.read_csv("data/all_data_info.csv")
X = data.groupby('artist').filter(lambda x: len(x) >= 300)
x = X['new_filename']
y = X['artist']
print("Size of data: " + str(X.shape[0]))

print(x.shape, y.shape)
i1 = 0
for i in range(x.shape[0]):
    image_name = path + str(x.iloc[i1])
    try:
       img = image.load_img(image_name, target_size = (224,224))
       i1 += 1
    except:
        print("Image not found: " + str(image_name))
        x = x.drop(x.index[i1])
        y = y.drop(x.index[i1])
print(x.shape, y.shape)

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y) 
no_classes = len(np.unique(y))
print("Number of classes: " + str(no_classes))
y = to_categorical(y, num_classes = no_classes)

inter_model = VGG16(weights='imagenet', include_top=False)
     

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify = y, random_state = 2)

images = np.zeros((x_test.shape[0],224,224,3))
labels_test = []
print("Loading CV data\n")
try:
    print("Loading from file")
    features_test = np.load("features_test.npy")
    labels_test = np.load("labels_test.npy")
    print("Loading from file successful")
except:
    print("Unable to find file")
    for idx, new_filename in enumerate(x_test):
        try:
            image_name = path + str(new_filename)
            img = image.img_to_array(image.load_img(image_name, target_size = (224,224)))
            #print(new_filename,y_test[idx])
            x = image.img_to_array(img)
            images[idx] = img
            labels_test.append(y_test[idx])
            if idx % 1000 == 0:
                print(idx)
        except:
            print("Image not found: " + str(image_name))
    print("Generating Features for test")
    images = preprocess_input(images)
    labels_test = np.array(labels_test)
    features_test = inter_model.predict(images)
    print("Feature generation complete")

    print("Saving features")
    np.save("features_test.npy", features_test)
    np.save("labels_test.npy", labels_test)

print(features_test.shape, labels_test.shape)
print(type(features_test), type(labels_test))
images = np.zeros((x_train.shape[0],224,224,3))
labels_train = []
print("Loading train data\n")
try:
    print("Loading from file")
    features_train = np.load("features_train.npy")
    labels_train = np.load("labels_train.npy")
    print("Loading from file successful")
except:
    print("Unable to find file")
    for idx, new_filename in enumerate(x_train):
        try:
            image_name = path + str(new_filename)
            img = image.img_to_array(image.load_img(image_name, target_size = (224,224)))
            #print(new_filename,y_test[idx])
            x = image.img_to_array(img)
            images[idx] = img
            labels_train.append(y_train[idx])
            if idx % 3000 == 0:
                print(idx)
        except:
            print("Image not found: " + str(image_name))

    print("Generating Features for train")
    images = preprocess_input(images)
    labels_train = np.array(labels_train)
    features_train = inter_model.predict(images)
    print("Feature generation complete")

    print("Saving features")
    np.save("features_train.npy", features_train)
    np.save("labels_train.npy", labels_train)

images =[]
print(features_train.shape, labels_train.shape)
print(type(features_train), type(labels_train))

input_layer = Input(shape=features_train.shape[1:])
f1=Flatten()(input_layer)
y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.05))(f1)
y=Dropout(0.5)(y)
y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.05))(y)
y=Dropout(0.5)(y)
#y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(y)
#y=Dropout(0.5)(y)
predictions = Dense(no_classes, activation='softmax',kernel_initializer='glorot_normal')(y)
model = Model(inputs=input_layer, outputs=predictions)
rms=RMSprop(lr=0.0001)
model.compile(optimizer=rms, loss='categorical_crossentropy',metrics=['accuracy'])

print("Training Now")
model.fit(x = features_train, y = labels_train, nb_epoch = 50, validation_data = [features_test, labels_test],callbacks= [EarlyStopping(patience=4)])

print("Training Complete")

print("Saving Model")
model.save('vgg16_artist_pred.h5')