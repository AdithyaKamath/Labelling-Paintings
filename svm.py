import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

path = "G:/Academics/6th Sem/CG/"



x = np.load(path + 'Temp/x_train300.npy')
y = np.load(path + 'Temp/y_train300.npy')

x = np.reshape(x, (x.shape[0], -1))



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, stratify = y, shuffle = True)

clf = svm.SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(accuracy_score(y_pred, y_test))

