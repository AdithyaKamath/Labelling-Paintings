import pandas as pd 
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, GlobalAveragePooling2D,Input,Flatten,Dropout
from keras import regularizers
from keras.models import load_model
from keras.callbacks import EarlyStopping

inter_model = ResNet50(weights='imagenet', include_top=False, input_shape = (224,224,3))
inter_model.summary()

path = "data/newtrain/"
x_train = np.load("x_train.npy")
x_test = np.load("x_cv.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_cv.npy")


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

for layer in inter_model.layers[:43]:
    layer.trainable = False

f1=Flatten()(inter_model.output)
y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.05))(f1)
y=Dropout(0.5)(y)
#y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.05))(y)
#y=Dropout(0.5)(y)
#y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(y)
#y=Dropout(0.5)(y)
predictions = Dense(no_classes, activation='softmax',kernel_initializer='glorot_normal')(y)
model = Model(inputs=inter_model.input, outputs=predictions)
adam = Adam(lr=0.0001,decay = 0.01)
model.compile(optimizer= adam, loss='categorical_crossentropy',metrics=['accuracy', 'top_k_categorical_accuracy'])
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=1, mode='auto')


print("Training Now")
model.fit(x = images_train, y = y_train, epochs = 50, validation_data = [images_test, y_test],callbacks= [EarlyStopping(patience=5)])


print("Training Complete")

print("Saving Model")
model.save('resnet_artist_pred_new.h5')