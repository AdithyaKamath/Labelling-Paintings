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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

path = "C:/Users/lenovo/Documents/CS/6Sem/CG/newtrain/"

x_train = np.load("C:/Users/lenovo/Documents/CS/6Sem/CG/Labelling-Paintings/Data/x_train.npy")
x_test = np.load("C:/Users/lenovo/Documents/CS/6Sem/CG/Labelling-Paintings/Data/x_test.npy")
y_train = np.load("C:/Users/lenovo/Documents/CS/6Sem/CG/Labelling-Paintings/Data/y_train.npy")
y_test = np.load("C:/Users/lenovo/Documents/CS/6Sem/CG/Labelling-Paintings/Data/y_test.npy")

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train) 
y_test = le.transform(y_test)
no_classes = len(np.unique(y_train))
print("Number of classes: " + str(no_classes))
y_train = to_categorical(y_train, num_classes = no_classes)
y_test = to_categorical(y_test, num_classes = no_classes)

inter_model = VGG16(weights='imagenet', include_top=False)

def generator1(data,y, im_height = 224, im_width = 224, train = True):
    while 1:
        for i in range(len(data)//32):
            labels = []
            images = np.zeros((32,224,224,3))
            for idx, image_path in enumerate(data[32*i:32*(i+1)]):
                img = image.img_to_array(image.load_img(path + str(image_path)))
                images[idx] = img
                if(train):
                    labels.append(y[idx])

            images = preprocess_input(images)
            features = inter_model.predict(images)
            if(train):
                labels = np.array(labels)
                yield features, labels
            else:
                yield features

input_layer = Input(shape=(7,7,512))
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
model.compile(optimizer=rms, loss='categorical_crossentropy',metrics=['accuracy', 'top_k_categorical_accuracy'])

# fit_generator doesn't work if this isn't executed. No idea why
labels = []
images = np.zeros((32,224,224,3))
train = 1
i = 0
for idx, image_path in enumerate(x_train[32*i:32*(i+1)]):
    img = image.img_to_array(image.load_img(path + str(image_path)))
    images[idx] = img
    if(train):
        labels.append(y[idx])

images = preprocess_input(images)
features = inter_model.predict(images)

model.fit_generator(generator1(x_train,y_train),steps_per_epoch=x_train.shape[0]//32, epochs = 50, validation_data = generator1(x_test,y_test),validation_steps = y_test.shape[0]//32,callbacks= [EarlyStopping(patience=4)])