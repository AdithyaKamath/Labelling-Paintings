import numpy as np

from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.utils import to_categorical

path = "G:/Academics/6th Sem/CG/"

x_train = np.load(path + 'Temp/x_train.npy')
x_test = np.load(path + 'Temp/x_test.npy')

y_train = np.load(path + 'Temp/y_train.npy')
y_test = np.load(path + 'Temp/y_test.npy')


x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

no_classes = max(y_train + y_test) + 1

y_train = to_categorical(y_train, num_classes = no_classes)
y_test = to_categorical(y_test, num_classes = no_classes)


	
model = VGG16(weights='imagenet', include_top=False)

					
inter_model=model
#inter_model=Model(inputs=model.input,outputs=model.get_layer('block4_pool').output)

features=inter_model.predict(x_train,batch_size=20)
ftest=inter_model.predict(x_test,batch_size=20)

np.save(path + "Temp/vgg_features_train.npy", features)
np.save(path + "Temp/vgg_features_test.npy", ftest)