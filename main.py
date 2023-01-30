# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

from PIL import Image
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import os


def pos_define(width_plate,height_plate,width_back,height_back):
	w=random.choice(range(0,width_back))
	h=random.choice(range(0,height_back))
	while (width_back-w)<width_plate:
		w = random.choice(range(0, width_back))

	while (height_back-h)<height_plate:
		h = random.choice(range(0, height_back))

	return w,h

img=Image.open('./data/plates/P01.jpg')
img_data=list(img.getdata())
transform_image=np.zeros([255,255])
transform_image[:]=-1
rotation_list=[0,30,45,60,90,120,135,150,180,210,225,240,270,300,315,330]
rotation=random.choice(rotation_list)
rotated_img=img.rotate(45,expand=True)

a=1;b=0.5;c=0
d=0;e=1;f=0
g=0;h=0;k=1

img = cv2.imread('./data/plates/P01.jpg')
M = np.float32([[1, 0.5, 0],
             	[0, 1  , 0],
            	[0, 0  , 1]])
cols,rows,dim=img.shape

scaled_list=[25,50,75,100,125,150,175,200]
scale_percent=random.choice(scaled_list)
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
plt.imshow(img)
plt.show()
plt.imshow(resized_img)
plt.show()

shear_img=cv2.warpPerspective(img,M,(int(rows*1.5),int(cols*1.5)))

plt.axis('off')
plt.imshow(shear_img)
plt.show()


img_num=3000
plate_names=os.listdir('./data/plates')
num_plates=len(plate_names)
back_names=os.listdir('./data/backgrounds')
num_back=len(back_names)
root='./data'




for p in plate_names:
	for b in back_names:
		plate_img=cv2.imread(os.path.join(root,'plates',p))
		back_img=cv2.imread(os.path.join(root,'backgrounds',b))

		p_name,_=os.path.splitext(p)
		b_name, _ = os.path.splitext(b)

		w_plate,h_plate=plate_img.shape[1],plate_img.shape[0]
		w_back,h_back=back_img.shape[1],back_img.shape[0]

		w_overlay,h_overlay=pos_define(w_plate,h_plate,w_back,h_back)

		overlay_img=back_img
		w_new=(w_overlay+w_plate)
		h_new=(h_overlay+h_plate)
		hh= h_new-h_overlay
		ww=w_new-w_overlay
		t=overlay_img[w_overlay:w_new,h_overlay:h_new,:]

		overlay_img[w_overlay:w_new,h_overlay:h_new,:]=plate_img

		plt.imshow(overlay_img)
		plt.show()
		cv2.imwrite(os.path.join(root,'overlayed',p_name+'_'+b_name+'.jpg'),overlay_img)








