import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC



features=np.load("vgg_features_train.npy")
ftest=np.load("vgg_features_test.npy")

y_train=np.load('y_train.npy')
y_test=np.load('y_test.npy')

#y_train=np.argmax(y_train,axis=1)

#(y_train,y_test)=np.split(y_train,2)


features=np.reshape(features,(features.shape[0],-1))
ftest=np.reshape(ftest,(ftest.shape[0],-1))

#clf=svm.SVC(C=0.5,decision_function_shape='ovo',kernel='linear')


ti=0
tj=0
tm=0.

for i in range(10):

	pca = PCA(n_components=i+2)



	pca.fit(features)

	fnew=pca.transform(features)
	ftnew=pca.transform(ftest)

	for j in range(100):	
		bag=BaggingClassifier(KNeighborsClassifier(),
		n_estimators=j+2,
		max_samples=1.0)


		bag.fit(fnew, y_train)
		s=bag.predict(ftnew)
		t=accuracy_score(s, y_test)
		#print(t)
		
		if(tm<t):
			tm=t
			ti=i
			tj=j
	
	print('n_components = ',ti+2,'    n_estimators = ',tj+2,'    accuracy = ',tm)
'''

bdt_discrete = AdaBoostClassifier(
DecisionTreeClassifier(max_depth=2),
n_estimators=100,
learning_rate=1.5,
algorithm="SAMME")

bdt_discrete.fit(fnew, y_train)
s=bdt_discrete.predict(ftnew)
t=accuracy_score(s, y_test)
print(t)

print('n_components = ',t1+1,'  n_classifiers = ',t2+1,'  accuracy = ',large)
'''
'''
	clf = RandomForestClassifier(n_estimators=)
	clf=clf.fit(fnew,y_train)

	s=clf.predict(ftnew)
	print(accuracy_score(s, y_test))
'''

'''
clf=svm.SVC(C=0.5,decision_function_shape='ovo',kernel='linear')


bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=100,
    learning_rate=1,
    algorithm="SAMME")

bdt_discrete.fit(features, y_train)

for i in bdt_discrete.staged_predict(ftest):
	print(accuracy_score(i, y_test))




#clf.fit(features,y_train)

#x_pred=clf.predict(ftest)
'''

	