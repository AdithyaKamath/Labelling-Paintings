import csv
import numpy as np
import pickle
import itertools
from sklearn import preprocessing


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


	

		

with open(path+'Temp/train300.txt',"rb") as fp:
	images = pickle.load(fp)

artist_data = [(c2, n2)
for (c1, c2, c3, c4, c5, c6), (n1, n2)
in itertools.product(image_labels, images)
if c1.decode('utf-8') == n1]


[y_train, x_train] = zip(*artist_data)


le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train) 


x_train = np.array(x_train)
y_train = np.array(y_train)

np.save(path+'Temp/x_train300.npy',x_train)
np.save(path+'Temp/y_train300.npy',y_train)



