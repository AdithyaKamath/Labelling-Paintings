import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression


clf4=AdaBoostClassifier(
	DecisionTreeClassifier(max_depth=2),
	n_estimators=100,
	learning_rate=1.5,
	algorithm="SAMME")
clf1=BaggingClassifier(KNeighborsClassifier(),
		n_estimators=8,
		max_samples=1.0)
clf2=GradientBoostingClassifier()
clf3=svm.SVC(kernel='linear')

		
metaclf=GradientBoostingClassifier()
#svm.SVC(C=0.5,decision_function_shape='ovo',kernel='linear')



def classify(clf,f1,f2,f3,text):
	 

	clf.fit(f1, y1)
	
	print(text)
	pred1=clf.predict(f2)
	print('Test 1: ',accuracy_score(clf.predict(f2), y2))
	pred2=clf.predict(f3)
	print('Test 2: ',accuracy_score(clf.predict(f3), y3))
	
	return [pred1,pred2]





f1=np.load('f1.npy')
f2=np.load('f2.npy')
f3=np.load('f3.npy')

y1=np.load('y1.npy')
y2=np.load('y2.npy')
y3=np.load('y3.npy')

f1=np.reshape(f1,(f1.shape[0],-1))
f2=np.reshape(f2,(f2.shape[0],-1))
f3=np.reshape(f3,(f3.shape[0],-1))

pca = PCA(n_components=9)

pca.fit(f1)

f1_ada=pca.transform(f1)
f2_ada=pca.transform(f2)
f3_ada=pca.transform(f3)


pca = PCA(n_components=5)

pca.fit(f1)

f1_bag=pca.transform(f1)
f2_bag=pca.transform(f2)
f3_bag=pca.transform(f3)

AdaPred=classify(clf1,f1_ada,f2_ada,f3_ada,'Adaboost')
BagPred=classify(clf2,f1_ada,f2_ada,f3_ada,'Bagging')
GradPred=classify(clf3,f1_ada,f2_ada,f3_ada,'Gradient')
SVMPred=classify(clf4,f1_ada,f2_ada,f3_ada,'Gradient')






meta_train=np.stack((AdaPred[0],BagPred[0],GradPred[0],SVMPred[0]),axis=-1)
meta_test=np.stack((AdaPred[1],BagPred[1],GradPred[1],SVMPred[1]),axis=-1)

meta_test=np.concatenate((meta_train,meta_test),axis=0)
y3=np.concatenate((y2,y3),axis=0)


'''
pca = PCA(n_components=1)

pca.fit(meta_train)
pca.transform(meta_train)
meta_test=pca.transform(meta_test)
metatest=np.int32(meta_test)

#clf=metaclf
#clf.fit(meta_train, y2)
#final_pred=clf.predict(meta_test)

final_pred=np.ceil(meta_test)
'''
'''
for i in range(final_pred.shape[0]):
	if(final_pred[i]<0):
		final_pred[i]=0
	elif(final_pred[i]>2):
		final_pred[i]=2
	elif(final_pred[i]==-0):
		final_pred[i]=0


print(final_pred)
print('Final SVM')
print(accuracy_score(final_pred, y3))
'''
pTrue=[]
yTrue=[]

pFalse=[]
yFalse=[]

for i in range(meta_test.shape[0]):
	if((meta_test[i]==meta_test[i,0]).all()):
		pTrue.append(meta_test[i,0])
		yTrue.append(y3[i])
	else:
		pFalse.append(meta_test[i,0])
		yFalse.append(y3[i])

print(accuracy_score(pTrue, yTrue))
#print(accuracy_score(pFalse, yFalse))
