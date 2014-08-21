
import os
import sys
import util
import scipy.misc
import scipy.io
import numpy as np
import random
import string
import pdb
import cv2




Folder = "/database/ChengCheng/Light-norm/light-zjz-norm/PIC/dst"
save_Folder = "/database/ChengCheng/Light-norm/light-zjz-norm/PIC/label_dst"



flist = os.listdir(Folder)

for label, fname in enumerate(flist):
	str = fname.split("_")
	name = str[1]
	path_out = os.path.join(save_Folder, name)
	img = cv2.imread(os.path.join(Folder, fname),0)
	if not os.path.exists(path_out):
		os.makedirs(path_out)
	cv2.imwrite(os.path.join(path_out, fname),img)


	#imgFullPath.rfind("/")