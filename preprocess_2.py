import csv
import numpy as np
import pickle
import itertools

from sklearn import preprocessing
from sklearn.model_selection import train_test_split



path = "G:/Academics/6th Sem/CG/"

image_labels = []

image_labels_final = []




with open(path+"Dataset/train_info.csv",'r',encoding = "utf8") as f:
	for l in  csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
			image_labels.append([cell.encode('utf8') for cell in l])
		

del(image_labels[0])


image_labels = sorted(image_labels, key = lambda col:col[1])



for key, group in itertools.groupby(image_labels, lambda col: col[1]):
	temp = list(group)
	if(len(temp)>=300):
		image_labels_final.extend(temp)


	

		

with open(path+'Temp/training_data.txt',"rb") as fp:
	images = pickle.load(fp)

artist_data = [(c2, n2)
for (c1, c2, c3, c4, c5, c6), (n1, n2)
in itertools.product(image_labels, images)
if c1.decode('utf-8') == n1]


[y, x] = zip(*artist_data)


le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y) 


x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, stratify = y, shuffle = True)



np.save(path+'Temp/x_train.npy',x_train)
np.save(path+'Temp/y_train.npy',y_train)

np.save(path+'Temp/x_test.npy',x_test)
np.save(path+'Temp/y_test.npy',y_test)


