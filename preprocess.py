import glob
from keras.preprocessing import image
import numpy as np
import pickle
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
path = "G:/Academics/6th Sem/CG/"


images = []

#count = 0

for filename in glob.glob(path+'Dataset/train/*.jpg'):
	#if count>3000:
	#	break

	image_name = filename.split("\\")[1]
	try:
		img = image.load_img(filename, target_size = (224,224))
		images.append([image_name,image.img_to_array(img)])
	#	count = count + 1
	except:
		pass

#count = 0

for filename in glob.glob(path+'Dataset/test/*.jpg'):
	#if count>3000:
	#	break

	image_name = filename.split("\\")[1]
	try:
		img = image.load_img(filename, target_size = (224,224))
		images.append([image_name,image.img_to_array(img)])
	#	count = count + 1
	except:
		pass
	
with open(path+'Temp/training_data.txt',"wb") as fp:
	pickle.dump(images,fp)