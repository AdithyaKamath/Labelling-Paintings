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

inter_model = VGG16(weights='imagenet', include_top=False)

path = "data/newtrain/"
data = pd.read_csv("data/all_data_info.csv")
X = data.groupby('artist').filter(lambda x: len(x) >= 300)
x = X['new_filename']
y = X['artist']
print("Size of data: " + str(X.shape[0]))

images = np.zeros((x.shape[0]-2,224,224,3))
labels = []

print(x.shape, y.shape)

flag = 1
try:
    print("Loading from file")
    features_test = np.load("features_test1.npy")
    labels_test = np.load("labels_test1.npy")
    features_train = np.load("features_train1.npy")
    labels_train = np.load("labels_train1.npy")
    flag = 0
except:
    print("Files not found")
    i1 = 0
    for i in range(x.shape[0]):
        image_name = path + str(x.iloc[i1])
        try:
            img = image.img_to_array(image.load_img(image_name, target_size = (224,224)))
            images[i1] = img
            labels.append(y.iloc[i1])
            i1 += 1
        except Exception as err:
            print("Image not found: " + str(image_name))
            print(err)
            x = x.drop(x.index[i1])
            y = y.drop(x.index[i1])
    print("Generating Features")
    print(images.shape)
    images = preprocess_input(images)
    labels = np.array(labels)
    features = inter_model.predict(images)
    print("Feature generation complete")

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y) 
no_classes = len(np.unique(y))

if flag:
    print(x.shape, y.shape)
    print(features.shape, labels.shape)
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels) 
    no_classes = len(np.unique(labels))
    print("Number of classes: " + str(no_classes))
    labels = to_categorical(y, num_classes = no_classes)  

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, stratify = labels , random_state = 2)
    print("Saving features")
    np.save("features_test1.npy", features_test)
    np.save("labels_test1.npy", labels_test)
    np.save("features_train1.npy", features_train)
    np.save("labels_train1.npy", labels_train)

images =[]
print(features_train.shape, labels_train.shape)
print(features_test.shape, labels_test.shape)

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
model.compile(optimizer= rms, loss='categorical_crossentropy',metrics=['accuracy', 'top_k_categorical_accuracy'])

print("Training Now")
model.fit(x = features_train, y = labels_train, epochs = 50, validation_data = [features_test, labels_test],callbacks= [EarlyStopping(patience=4)])

print("Training Complete")

print("Saving Model")
model.save('vgg16_artist_pred_new.h5')