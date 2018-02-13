import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

path = "G:/Academics/6th Sem/CG/"



x_train = np.load(path + 'Temp/x_train.npy')
y_train = np.load(path + 'Temp/y_train.npy')

x_test = np.load(path + 'Temp/x_test.npy')
y_test = np.load(path + 'Temp/y_test.npy')

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))


clf = svm.SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(accuracy_score(y_pred, y_test))

