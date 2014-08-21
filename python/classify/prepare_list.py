# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:31:26 2013

@author: hanwei
modified by lyq, 2014.01.10
"""

from multiprocessing import Process
from multiprocessing import Pool

import os
import sys
import util
import scipy.misc
import scipy.io
import numpy as np
import random
import string
import cv2

SIZE = 128
CHANNELS = 1 
NUM_PER_PATCH = 1024
Mean_Img_Num = 5000
NUM_IMG = 30

 
 
def prepare_list(Folder, outFolder):

    subFolderList = os.listdir(outFolder)
    list.sort(subFolderList)
    for index, fname in enumerate(subFolderList):
        #print fname
        os.remove(os.path.join(outFolder, fname))
        print "removed   " + fname
        #f=open(name,"rb")
        #fileList = open(os.path.join(outFolder, fname),'rb').readlines()

        #if len(fileList) < NUM_IMG:
             

    allLabels = []
   
    for i, FolderName in enumerate(Folder):
        #print FolderName
        subFolderList = os.listdir(FolderName)
        list.sort(subFolderList)
        #print subFolderList
        for index, labelname in enumerate(subFolderList):
            #print os.path.join(Folder, fname)
            #imgnames = os.listdir(os.path.join(Folder, fname))
            #print imgnames
            if labelname not in allLabels:
                allLabels.append(labelname)
            #print FolderName
            
            pic_fname = os.listdir(os.path.join(FolderName,labelname))
            #print 
            if len(pic_fname) != 0:
                f=open(os.path.join(outFolder,"%s.npy" % labelname),"ab")
            
                for idx, picname in enumerate(pic_fname):
                    pic_list = os.path.join(FolderName,labelname,picname)
                    tmp = pic_list + "\n"
                    f.writelines(tmp)
                #flist.append(pic_list)
                f.close()

def del_People(outFolder):

    subFolderList = os.listdir(outFolder)
    list.sort(subFolderList)
    for index, fname in enumerate(subFolderList):
        #f=open(name,"rb")
        fileList = open(os.path.join(outFolder, fname),'rb').readlines()

        if len(fileList) < NUM_IMG:
            os.remove(os.path.join(outFolder, fname))   


if __name__ == '__main__':

    fname = []

    list_dir = "/home/chengcheng/pair"


    folder1 = "/database_001/img/face/Alignment_160/CAS_PEA"
    fname.append(folder1)

    prepare_list(fname, list_dir)

    del_People(list_dir)


    
