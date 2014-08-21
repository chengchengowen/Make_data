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

SIZE = 234
CHANNELS = 3 
NUM_PER_PATCH = 1024
Mean_Img_Num = 5000
NUM_IMG = 80

 
 
def getBatch(meta, allLabels):
    global SIZE, CHANNELS
    print " going to load %d images " % len(meta)
    data = np.zeros((SIZE*SIZE*CHANNELS, len(meta)), dtype=np.uint8)
    labels = np.zeros((len(meta)), dtype=np.single)
    imgnames = []
    


    for i in range(0,len(meta)):
        str1 = meta[i].strip()
        tmp = str1.split(':')
        labels[i] = tmp[1]
        str2 = tmp[0]
        tmp = str2.split('/')
        length = len(tmp)
        imgnames.append(tmp[length - 1])
        try:
            data[:,i] = load_img_data(str2)
                  
        except IOError as e:
            print meta[i].strip()
            exit(1)
    return data, labels, imgnames
        

def makeBatches(allImgMeta, out_dir, batchSize, startIdx = 0):
    numImg = len(allImgMeta)
    numBatches = numImg / batchSize # the last batch keep the remainder
    if numImg % batchSize != 0:
        numBatches += 1

    print 'Going to make %d baches' % numBatches
    for idx_batch in range(numBatches):
        #        if idx_batch < numBatches - 2:
        #            continue
        print "### Making the %dth batch ###" % idx_batch
        b_start = batchSize * idx_batch
        b_end = batchSize * (idx_batch + 1)
        if idx_batch == numBatches - 1:
            b_start = numImg - batchSize
            b_end = numImg
        batchMeta = allImgMeta[b_start:b_end]
        data, labels, imgnames = getBatch(batchMeta)

        out_fname = os.path.join(out_dir, "data_batch_%04d" % (idx_batch+startIdx))
        print "saving to %s" % out_fname
        util.pickle(out_fname, {'data':data, 'labels':labels, 'images':imgnames})
            



 
def prepareTrain(train_list, outFolder,startIdx):
    global NUM_PER_PATCH
    fileList = open(train_list,'rb').readlines()
    random.shuffle(fileList)

    if len(fileList) < Mean_Img_Num:
         num_mean_img = len(fileList)
    else:
         num_mean_img = Mean_Img_Num

    data = np.zeros((SIZE*SIZE*CHANNELS, num_mean_img), dtype=np.uint8)

    for i in range(0,num_mean_img):
        str1 = fileList[i].strip();
        tmp = str1.split(':')
        data[:,i] = load_img_data(tmp[0])


    

    tmp = data.reshape(data.shape[1],data.shape[0])
    dataSum = np.sum(tmp, axis=0, dtype = np.float64)

    globalCount = tmp.shape[0]
    meanImg = dataSum / globalCount
    util.pickle(outFolder+"/meanImg", meanImg)
    
    allLabels = []
    allImgMeta = []
    for line in fileList:
        str1 = line.strip()
        tmp = str1.split(':')
        length = len(tmp)
        label = tmp[1]
        #print label
        if label not in allLabels:
            allLabels.append(label)            

    print "####### Got %d classes ######" % len(allLabels)
    meta = {}
    meta['data_mean'] = meanImg
    meta['label_names'] = allLabels
    util.pickle( os.path.join(outFolder, "batches.meta"), meta)




    numImg = len(fileList)
    numBatches = numImg / NUM_PER_PATCH # the last batch keep the remainder
    if numImg % NUM_PER_PATCH != 0:
        numBatches += 1

    print 'Going to make %d baches' % numBatches
    for idx_batch in range(numBatches):
        #        if idx_batch < numBatches - 2:
        #            continue
        print "### Making the %dth batch ###" % idx_batch
        b_start = NUM_PER_PATCH * idx_batch
        b_end = NUM_PER_PATCH * (idx_batch + 1)
        if idx_batch == numBatches - 1:
            b_start = numImg - NUM_PER_PATCH
            b_end = numImg
        batchMeta = fileList[b_start:b_end]

        data, labels, imgnames = getBatch(batchMeta,allLabels)
        out_fname = os.path.join(outFolder, "data_batch_%04d" % (startIdx+idx_batch))
        print "saving to %s" % out_fname
        util.pickle(out_fname, {'data':data, 'labels':labels, 'images':imgnames})

    #fileList.close()


def processTest(test_list, out_dir, startIdx):
    global NUM_PER_PATCH
    meta = util.unpickle(os.path.join(out_dir, "batches.meta"))
    allLabels = meta['label_names']


    fileList = open(test_list,'rb').readlines()
    random.shuffle(fileList)

    print "####### Got %d classes ######" % len(allLabels)
    print "####### Got %d images ######" % len(fileList)

    numImg = len(fileList)
    numBatches = numImg / NUM_PER_PATCH # the last batch keep the remainder
    if numImg % NUM_PER_PATCH != 0:
        numBatches += 1

    print 'Going to make %d baches' % numBatches
    for idx_batch in range(numBatches):
        #        if idx_batch < numBatches - 2:
        #            continue
        print "### Making the %dth batch ###" % idx_batch
        b_start = NUM_PER_PATCH * idx_batch
        b_end = NUM_PER_PATCH * (idx_batch + 1)
        if idx_batch == numBatches - 1:
            b_start = numImg - NUM_PER_PATCH
            b_end = numImg
        batchMeta = fileList[b_start:b_end]

        data, labels, imgnames = getBatch(batchMeta,allLabels)
        out_fname = os.path.join(out_dir, "data_batch_%04d" % (idx_batch+startIdx))
        print "saving to %s" % out_fname
        util.pickle(out_fname, {'data':data, 'labels':labels, 'images':imgnames})

    #fileList.close()

def del_People(outFolder):

    subFolderList = os.listdir(outFolder)
    list.sort(subFolderList)
    for index, fname in enumerate(subFolderList):
        #f=open(name,"rb")
        fileList = open(os.path.join(outFolder, fname),'rb').readlines()

        if len(fileList) < NUM_IMG:
            os.remove(os.path.join(outFolder, fname))   

def load_img_data(fileName):
    global SIZE, CHANNELS
    img = scipy.misc.imread(fileName)
    size = img.shape[:2]
    ratio = np.float32(SIZE)/min(size)
    resizedImg = scipy.misc.imresize(img, (int(round((size[0]*ratio))),int(round((size[1]*ratio)))))
    if len(resizedImg.shape)==2:
        # this is an grey img
        tmp = np.zeros((resizedImg.shape[0], resizedImg.shape[1], CHANNELS), np.uint8)
        for ch in range(CHANNELS):
            tmp[:,:,ch] = resizedImg
        resizedImg = tmp
        #    print resizedImg.shape
    
    try:
        croppedImg = resizedImg
        if resizedImg.shape[0] == SIZE:
            if resizedImg.shape[1] == SIZE:
                croppedImg = resizedImg
            else:
                offset1low = (resizedImg.shape[1]-SIZE)/2
                offset1high = -((resizedImg.shape[1]-SIZE)/2)
                if resizedImg.shape[1] % 2 == 1:
                    offset1high -= 1
                croppedImg = resizedImg[:, offset1low:offset1high, :]
        else:
            offset0low = (resizedImg.shape[0]-SIZE)/2
            offset0high = -((resizedImg.shape[0]-SIZE)/2)
            if resizedImg.shape[0] % 2 == 1:
                offset0high -= 1
            croppedImg = resizedImg[offset0low:offset0high, :, :]
            #if croppedImg.shape[0] != 256 or croppedImg.shape[1] != 256 or croppedImg.shape[2] != 3:
            # print fileName, croppedImg.shape
        vec = np.array([],dtype=np.uint8);
        for ch in range(CHANNELS):
            vectmp = np.reshape(croppedImg[:,:,ch], SIZE*SIZE)
            vec = np.concatenate((vec, vectmp))
        return vec
    except IndexError as e:
    
        print resizedImg.shape, fileName
        sys.exit(1)        
        #print croppedImg.shape

def prepare_Train(training_list, out_dir):

    img_list = open(os.path.join(training_list,fname),'rb').readlines()

    random.shuffle(img_list)

    if len(img_list) < Mean_Img_Num:
         num_mean_img = len(img_list)
    else:
         num_mean_img = Mean_Img_Num

    data = np.zeros((SIZE*SIZE*CHANNELS, num_mean_img), dtype=np.uint8)

    for i in range(0,num_mean_img):
        str1 = img_list[i].strip()
        tmp = str1.split(':')
        data[:,i] = load_img_data(tmp[0])


    

    tmp = data.reshape(data.shape[1],data.shape[0])
    dataSum = np.sum(tmp, axis=0, dtype = np.float64)

    globalCount = tmp.shape[0]
    meanImg = dataSum / globalCount
    util.pickle(out_dir+"/meanImg", meanImg)
    
    allLabels = []
    allImgMeta = []
    for line in img_list:
        str = line.strip()
        tmp = str.split(':')
        label = tmp[1]
        #print label
        if label not in allLabels:
            allLabels.append(label)            

    print "####### Got %d classes ######" % len(allLabels)
    meta = {}
    meta['data_mean'] = meanImg
    meta['label_names'] = allLabels
    util.pickle( os.path.join(out_dir, "batches.meta"), meta)




    numImg = len(img_list)
    numBatches = numImg / NUM_PER_PATCH # the last batch keep the remainder
    if numImg % NUM_PER_PATCH != 0:
        numBatches += 1

    print 'Going to make %d baches' % numBatches
    for idx_batch in range(numBatches):
        #        if idx_batch < numBatches - 2:
        #            continue
        print "### Making the %dth batch ###" % idx_batch
        b_start = NUM_PER_PATCH * idx_batch
        b_end = NUM_PER_PATCH * (idx_batch + 1)
        if idx_batch == numBatches - 1:
            b_start = numImg - NUM_PER_PATCH
            b_end = numImg
        batchMeta = img_list[b_start:b_end]

        data, labels, imgnames = getBatch(batchMeta,allLabels)
        out_fname = os.path.join(out_dir, "data_batch_%04d" % startIdx + idx_batch)
        print "saving to %s" % out_fname
        util.pickle(out_fname, {'data':data, 'labels':labels, 'images':imgnames})


if __name__ == '__main__':

    #img_path = '/database_002/Alignment_Face/Array_angle/P00012/S01_P00012_C01001_I00_A0103.jpg'

    #load_img_data(img_path)


    list_dir = "/home/chengcheng/make_data/training.npy"
    startIdx= 0
    outFolder = "/database_002/batches/clothing"

    prepareTrain(list_dir,outFolder,startIdx)


    list_dir = "/home/chengcheng/make_data/test.npy"
    startIdx= 8000
    outFolder = "/database_002/batches/clothing"

    prepareTrain(list_dir,outFolder,startIdx)
    
