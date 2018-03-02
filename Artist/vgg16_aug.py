import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras import regularizers
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping



path = "data/newtrain/"
path_train = "data/artist_train/"
path_test = "data/artist_test/"
# data = pd.read_csv("data/all_data_info.csv")
# X = data.groupby('artist').filter(lambda x: len(x) >= 300)
# x = X['new_filename']
# y = X['artist']
# print("Size of data: " + str(X.shape[0]))



labels_train = np.load('labels_train.npy') 
labels_test = np.load('labels_test.npy')
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')

print(labels_train.shape, labels_test.shape)
print(x_train.shape, x_test.shape)

print("Loading train images")
images_train = np.zeros((x_train.shape[0],224,224,3))
for i in range(x_train.shape[0]):
    image_name = path + str(x_train[i])
    img = image.img_to_array(image.load_img(image_name, target_size = (224,224)))
    images_train[i] = img

print("Loading test images")
images_test = np.zeros((x_test.shape[0],224,224,3))
for i in range(x_test.shape[0]):
    image_name = path + str(x_test[i])
    img = image.img_to_array(image.load_img(image_name, target_size = (224,224)))
    images_test[i] = img


no_classes = 57
print("Number of classes: " + str(no_classes))
print(np.unique(labels_train))


model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
for layer in model.layers:
    layer.trainable = False

#Adding custom Layers 
l = model.output
l = Flatten()(l)
l = Dense(1024, activation="relu")(l)
l = Dropout(0.5)(l)
l = Dense(1024, activation="relu")(l)
predictions = Dense(no_classes, activation="softmax")(l)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=['accuracy', 'top_k_categorical_accuracy'])
# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

# input_layer = Input(shape=features_train.shape[1:])
# f1=Flatten()(input_layer)
# y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.05))(f1)
# y=Dropout(0.5)(y)
# y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.05))(y)
# y=Dropout(0.5)(y)
# #y = Dense(1024, activation='relu',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(y)
# #y=Dropout(0.5)(y)
# predictions = Dense(no_classes, activation='softmax',kernel_initializer='glorot_normal')(y)
# model = Model(inputs=input_layer, outputs=predictions)
# rms=RMSprop(lr=0.0001)
# model.compile(optimizer= rms, loss='categorical_crossentropy',metrics=['accuracy', 'top_k_categorical_accuracy'])
#train_generator = train_datagen.flow(images_train, labels_train)

#validation_generator = test_datagen.flow(images_test, labels_test)

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model 
model_final.fit_generator(train_datagen.flow(images_train, labels_train), samples_per_epoch = x_train.shape[0], epochs = 50, validation_data = test_datagen.flow(images_test, labels_test), nb_val_samples = x_test.shape[0], callbacks = [checkpoint, early])
