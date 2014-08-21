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


Test_Num = 500

 
 
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
    #subFolderList = os.listdir(Folder)
   
    for i, fname in enumerate(Folder):
        subFolderList = os.listdir(fname)
        for idx, img_name in enumerate(subFolderList):
            str1 = img_name.split('_')
            if str1[0] not in allLabels:
                allLabels.append(str1[0])


    for i, fname in enumerate(Folder):
        subFolderList = os.listdir(fname)
        for idx, img_name in enumerate(subFolderList):
            str1 = img_name.split('_')

            f=open(os.path.join(outFolder,"%s.npy" % str1[0]),"ab")
            pic_path = os.path.join(fname,img_name)
            tmp = pic_path + "\n"
            f.writelines(tmp)
            f.close()


def del_People(outFolder):

    subFolderList = os.listdir(outFolder)
    list.sort(subFolderList)
    for index, fname in enumerate(subFolderList):
        #f=open(name,"rb")
        fileList = open(os.path.join(outFolder, fname),'rb').readlines()

        if len(fileList) < NUM_IMG:
            os.remove(os.path.join(outFolder, fname))  

def prepare_for_caffe(tmp,list_dir): 


    allLabels = []
    path_list = []
    flist = [fname for fname in os.listdir(tmp) if fname.endswith('.npy')]

    print len(flist)

    data_by_category = [[] for i in range(len(flist))]

    for idx , fname in enumerate(flist):
        data_by_category[idx] = open(os.path.join(tmp, fname),'rb').readlines()

        if len(data_by_category[idx]) != 0:
            length = len(fname)
            label_name = fname[0:length-4]
            allLabels.append(label_name)
            path_list += data_by_category[idx]
        else:
            print idx
            print fname

    random.shuffle(path_list)

    # for training

    f=open(os.path.join(list_dir,"training.npy"),"wb")

    for i in range(0,len(path_list) - Test_Num):
        str1 = path_list[i].strip()
        tmp = str1.split('/')
        length = len(tmp)
        str1 = tmp[length -1]
        tmp = str1.split('_')

        idx = allLabels.index(tmp[0])
        data = path_list[i].strip() + ':' + str(idx) + '\n'
        f.writelines(data)

    f.close


    # for test

    f=open(os.path.join(list_dir,"test.npy"),"wb")
    for i in range(len(path_list) - Test_Num + 1,len(path_list)):
        str1 = path_list[i].strip()
        tmp = str1.split('/')
        length = len(tmp)
        str1 = tmp[length -1]
        tmp = str1.split('_')

        idx = allLabels.index(tmp[0])

        data = path_list[i].strip() + ':' + str(idx) + '\n'
        f.writelines(data)

    f.close

    


if __name__ == '__main__':

    fname = []

    tmp = "/home/chengcheng/make_data/tmp"

    list_dir = '/home/chengcheng/make_data'


    folder1 = "/database_002/img/clothing/data_processed"
    fname.append(folder1)

    # folder1 = "/database_001/img/face/Alignment_160/Star"
    # fname.append(folder1)

    # folder1 = "/database_001/img/face/Alignment_160/Movie"
    # fname.append(folder1)

    # folder1 = "/database_001/img/face/Alignment_160/array"
    # fname.append(folder1)

    # folder1 = "/database_001/img/face/Alignment_160/ID_Card"
    # fname.append(folder1)

    # folder1 = "/database_001/img/face/Alignment_160/multi_pie"
    # fname.append(folder1)

    # folder1 = "/database_001/img/face/Alignment_160/PolyU"
    # fname.append(folder1)

    # folder1 = "/database_001/img/face/Alignment_160/sel_cqbz"
    # fname.append(folder1)

    # folder1 = "/database_001/img/face/Alignment_160/Star"
    # fname.append(folder1)

    prepare_list(fname, tmp)

    #del_People(tmp)

    prepare_for_caffe(tmp,list_dir)


    
