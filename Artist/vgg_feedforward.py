import numpy as np
import tensorflow as tf



from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input,Flatten,Dropout
from keras import regularizers



path = "G:/Academics/6th Sem/CG/"



features=np.load(path + "Temp/vgg_features_train.npy")
ftest=np.load(path + "Temp/vgg_features_test.npy")

y_train = np.load(path + 'Temp/y_train.npy')
y_test = np.load(path + 'Temp/y_test.npy')

no_classes = max(y_train + y_test) + 1


input_layer=Input(shape=features.shape[1:])
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


model.fit(features, y_train,
          epochs=50,
          batch_size=100,shuffle=True,validation_data=(ftest,y_test))

score = model_new.evaluate(ftest, y_test, batch_size=30)
print(score)

