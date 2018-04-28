import glob
import numpy as np
import pickle
import os
from PIL import Image
from tqdm import tqdm
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
path = "data/train"

all_files = [x for x in os.listdir(path)
                 if os.path.isfile( os.path.join(path, x) )]

count = 0


for image_filename in tqdm(all_files):
	try:
		img = Image.open( os.path.join(path, image_filename) )
		img.resize((224, 224))\
			.convert('RGB')\
			.save( os.path.join("data/train/", image_filename) )
		count = count + 1
	except Exception as e:
		print("Unable to process {}".format(image_filename))
		print(e)



# images = []

# #count = 0

# for filename in glob.glob(path+'Dataset/train/*.jpg'):
# 	#if count>3000:
# 	#	break

# 	image_name = filename.split("\\")[1]
# 	try:
# 		img = image.load_img(filename, target_size = (224,224))
# 		images.append([image_name,image.img_to_array(img)])
# 	#	count = count + 1
# 	except:
# 		pass

# #count = 0

# for filename in glob.glob(path+'Dataset/test/*.jpg'):
# 	#if count>3000:
# 	#	break

# 	image_name = filename.split("\\")[1]
# 	try:
# 		img = image.load_img(filename, target_size = (224,224))
# 		images.append([image_name,image.img_to_array(img)])
# 	#	count = count + 1
# 	except:
# 		pass
	
# with open(path+'Temp/training_data.txt',"wb") as fp:
# 	pickle.dump(images,fp)