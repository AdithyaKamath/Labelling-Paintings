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
from keras.layers.advanced_activations import PReLU
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

flag = 1
save = 1

inter_model = VGG16(weights='imagenet', include_top=False)

path = "data/newtrain/"
store_path = 'style/'
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


if(flag == 0):
    images = np.zeros((x_train.shape[0],224,224,3))

    print("Loading train images")
    for i in range(x_train.shape[0]):
        image_name = path + str(x_train[i])
        img = image.img_to_array(image.load_img(image_name))
        images[i] = img
    #sd = np.std(images)
    #images -= mean
    #images /= sd
    print("Generating Features for train")
    print(images.shape)
    images = preprocess_input(images)
    features_train = inter_model.predict(images)
    print("Feature generation complete")

    images = np.zeros((x_test.shape[0],224,224,3))

    print("Loading test images")
    for i in range(x_test.shape[0]):
        image_name = path + str(x_test[i])
        img = image.img_to_array(image.load_img(image_name))
        images[i] = img
    #sd = np.std(images)
    #images -= mean
    #images /= sd
    print("Generating Features for test")
    print(images.shape)
    images = preprocess_input(images)
    features_test = inter_model.predict(images)
    print("Feature generation complete")
    images =[]

    if save:
        np.save(store_path + "features_vgg16_train.npy", features_train)
        np.save(store_path + "features_vgg16_test.npy", features_test)
else:
    print("Loading features from files")
    features_train = np.load(store_path + "features_vgg16_train.npy")
    features_test = np.load(store_path + "features_vgg16_test.npy")
    print("Finished loading from file")

input_layer = Input(shape=features_train.shape[1:])
f1=Flatten()(input_layer)
y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.05))(f1)
y=Dropout(0.5)(y)
y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.05))(y)
y = Dropout(0.5)(y)
#y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(y)
#y=Dropout(0.5)(y)
predictions = Dense(no_classes, activation='softmax',kernel_initializer='glorot_normal')(y)
model = Model(inputs=input_layer, outputs=predictions)
rms=RMSprop(lr=0.0001)#, decay = 0.01)

model.compile(optimizer= rms, loss='categorical_crossentropy',metrics=['accuracy', 'top_k_categorical_accuracy'])

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4,min_lr=0.000001)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode='auto')

print("Training Now")
model.fit(x = features_train, y = y_train, epochs = 50, validation_data = [features_test, y_test],callbacks= [early])#,reduce_lr])

print("Training Complete")

print("Saving Model")
model.save('models/vgg16_style.h5')