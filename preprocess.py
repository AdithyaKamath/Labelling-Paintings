import glob
from keras.preprocessing import image
import numpy as np
import pickle
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
path = "G:/Academics/6th Sem/CG/"


images = []

#count = 0
for i in [1]:
	for filename in glob.glob(path+'Dataset/train_'+str(i)+'/*.jpg'):
		#if count>3000:
		#	break

		image_name = filename.split("\\")[1]
		try:
			img = image.load_img(filename, target_size = (224,224))
			images.append([image_name,image.img_to_array(img)])
			count = count + 1
		except:
			pass


	
with open(path+'Temp/train300.txt',"wb") as fp:
	pickle.dump(images,fp)