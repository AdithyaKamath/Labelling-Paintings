# 75.81
import pandas as pd 
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, GlobalAveragePooling2D,Input,Flatten,Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


path = "data/newtrain/"
store_path = 'artist/'
x_train = np.load(store_path + "x_train.npy")
x_test = np.load(store_path + "x_cv.npy")
y_train = np.load(store_path + "y_train.npy")
y_test = np.load(store_path + "y_cv.npy")
mean = np.load(store_path + "mean.npy")


print("Size of train: " + str(x_train.shape))
print("Size of CV: " + str(x_test.shape))

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train) 
y_test = le.transform(y_test)
no_classes = len(np.unique(y_train))
print("Number of classes: " + str(no_classes))
y_train = to_categorical(y_train, num_classes = no_classes)
y_test = to_categorical(y_test, num_classes = no_classes)

images_train = np.zeros((x_train.shape[0],224,224,3))
print("Loading train images")
for i in range(x_train.shape[0]):
    image_name = path + str(x_train[i])
    img = image.img_to_array(image.load_img(image_name))
    images_train[i] = img

print(images_train.shape)
images_train = preprocess_input(images_train)


images_test = np.zeros((x_test.shape[0],224,224,3))

print("Loading test images")
for i in range(x_test.shape[0]):
    image_name = path + str(x_test[i])
    img = image.img_to_array(image.load_img(image_name))
    images_test[i] = img

print(images_test.shape)
images_test = preprocess_input(images_test)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),strides=(2,2),activation='relu', input_shape=(224,224,3), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,kernel_size=3,strides=(2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(4*no_classes, activation='linear'))
model.add(Dense(no_classes, activation='softmax'))
model.summary()
adam = Adam(lr=0.0001,decay = 0.01)
model.compile(optimizer= adam, loss='categorical_crossentropy',metrics=['accuracy', 'top_k_categorical_accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=4,min_lr=0.000001)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=7, verbose=1, mode='auto')

train_datagen = ImageDataGenerator(horizontal_flip = True)
test_datagen = ImageDataGenerator()

print("Training Now")
model.fit(x = images_train,y = y_train, epochs = 30, validation_data = [images_test, y_test],callbacks= [early,reduce_lr])


print("Training Complete")

print("Saving Model")
model.save('models/vgg16_artist.h5')