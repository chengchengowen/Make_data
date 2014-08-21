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

SIZE = 64
CHANNELS = 1 
NUM_PER_PATCH = 1024

def getMeanImg(raw_dir):
    global SIZE, CHANNELS
    flist = os.listdir(raw_dir)
    list.sort(flist)
    globalSum = np.zeros(SIZE*SIZE*CHANNELS, dtype=np.float64)
    globalCount = 0
    raw_info = []
    for label, fname in enumerate(flist):
        print "Reading", fname
        data = util.unpickle(raw_dir + '/' + fname)
        dataSum = np.sum(data, axis=0, dtype = np.float64)
        globalSum += dataSum
        globalCount += data.shape[0]
        raw_info.append((fname, label, data.shape[0])) #(name of the label, label value, number of images)
    meanImg = globalSum / globalCount
    return meanImg, raw_info

def getMeta(traindir):
    dirnames = os.listdir(traindir)
    print 'Found %d classes under %s' % (len(dirnames), traindir)
    meta = {}
    meta['label_names'] = dirnames
    return meta
    

def readAndResize(fileName):
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
        #    print croppedImg.shape

    

def getBatch(meta):
    global SIZE, CHANNELS
    print " going to load %d images " % len(meta)
    data = np.zeros((SIZE*SIZE*CHANNELS, len(meta)), dtype=np.uint8)
    labels = np.zeros((len(meta)), dtype=np.single)
    for i, entry in enumerate(meta):
        labels[i] = entry[1]
        try:
            data[:,i] = readAndResize(entry[0])
        except IOError as e:
            print entry[0]
            exit(1)
    return data, labels


def processValidation(labelfname, validir, out_dir, startIdx):
    global NUM_PER_PATCH
    #labelfname = '/data1/LSVRC2012/ILSVRC2010_validation_ground_truth.txt'
    #validir = "/data1/LSVRC2012/val"    
    #out_dir = "/data2/ILSVRC2010/train_batchesd"
    labels = list(np.loadtxt(labelfname)-1)

    flist = os.listdir(validir)
    list.sort(flist)
    flist = [os.path.join(validir, fname) for fname in flist]

    makeBatches(zip(flist, labels), out_dir, NUM_PER_PATCH, startIdx), 


def processTest(test_dir, out_dir, startIdx):
    global NUM_PER_PATCH

    out_file = out_dir + "/imglist"
    trsimgMeta, trsLabels = util.unpickle(out_file)



    allImgMeta = []
    allLabels = []
    subdirnames = os.listdir(test_dir)
    list.sort(subdirnames)
    for classLabel, subdir in enumerate(trsLabels):
        for num,test_sub_dir in enumerate(subdirnames):
            # print classLabel
            # print test_sub_dir
            # print subdir
            if test_sub_dir == subdir:
                imgnames = os.listdir(os.path.join(test_dir, subdir))
                fullnames = [os.path.join(test_dir, subdir, name) for name in imgnames]
                meta = zip(fullnames, [classLabel] * len(fullnames))
                allImgMeta += meta

        # #allLabels.append(subdir)
        # 
        # 
        # # name, label pair
        # #meta = zip(fullnames, [string.atoi(subdir)] * len(fullnames))
        # 
        # 
    print "####### Got %d classes ######" % len(subdirnames)
    print "####### Got %d images ######" % len(allImgMeta)
    print "shuffling..."
    random.shuffle(allImgMeta)
    #return allImgMeta, allLabels


    #allImgMeta, testLabels = collectAndShuffle(test_dir)

    makeBatches(allImgMeta, out_dir, NUM_PER_PATCH, startIdx)

  
def fixLabel(labels):
    synnet_meta_file = '/data2/ILSVRC2010/meta.mat'
    synnet_meta = scipy.io.loadmat(synnet_meta_file)
    synnet_meta = synnet_meta['synsets'][:1000]
    ILSVRC_ID = [item[0][0][0][0] for item in synnet_meta]
    ImageNet_ID = [item[0][1][0] for item in synnet_meta]

    train_dir = "/data1/LSVRC2010/train"
    subdirnames = os.listdir(train_dir)
    list.sort(subdirnames)
    realID = [subdirnames.index(id) for id in ImageNet_ID]

    for i in range(len(labels)):
        labels[i] = realID[int(round(labels[i]))]

    

def collectAndShuffle(train_dir):
    allImgMeta = []
    allLabels = []
    subdirnames = os.listdir(train_dir)
    list.sort(subdirnames)
    for classLabel, subdir in enumerate(subdirnames):
        allLabels.append(subdir)
        imgnames = os.listdir(os.path.join(train_dir, subdir))
        fullnames = [os.path.join(train_dir, subdir, name) for name in imgnames]
        # name, label pair
        #meta = zip(fullnames, [string.atoi(subdir)] * len(fullnames))
        meta = zip(fullnames, [classLabel] * len(fullnames))
        allImgMeta += meta
    print "####### Got %d classes ######" % len(allLabels)
    print "####### Got %d images ######" % len(allImgMeta)
    print "shuffling..."
    random.shuffle(allImgMeta)
    return allImgMeta, allLabels
        

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
        data, labels = getBatch(batchMeta)
        #labels1 = labels//2
        #labels2 = labels//3
        #labels3 = labels//4

        out_fname = os.path.join(out_dir, "data_batch_%04d" % (idx_batch+startIdx))
        print "saving to %s" % out_fname
        util.pickle(out_fname, {'data':data, 'labels':labels})
            

def prepareTrain(train_dir, out_dir, meanImg_dir):
    global NUM_PER_PATCH
    #train_dir = "/data1/LSVRC2010/train"
    #out_dir = "/data2/ILSVRC2010/train_batches"
    
    meanImg = util.unpickle(meanImg_dir + '/meanImg')
    allImgMeta, allLabels = collectAndShuffle(train_dir)
    meta = {}
    meta['data_mean'] = meanImg
    meta['label_names'] = allLabels
    util.pickle( os.path.join(out_dir, "batches.meta"), meta)

    makeBatches(allImgMeta, out_dir, NUM_PER_PATCH)
    out_file = out_dir + "/imglist"
    util.pickle(out_file, [allImgMeta, allLabels])

#lyq add
def collectOneClass(classFolder):
    global SIZE, CHANNELS
    filelist = os.listdir(classFolder)
    list.sort(filelist)
    dataCol = np.zeros((len(filelist), SIZE*SIZE*CHANNELS), dtype = np.uint8)
    for index, filename in enumerate(filelist):
        dataCol[index,:] = readAndResize(os.path.join(classFolder, filename))
    return dataCol

def collectImgByClass(inFolder, outFolder):
    subFolderList = os.listdir(inFolder)
    list.sort(subFolderList)
    for index, foldername in enumerate(subFolderList):
        print foldername
        dataCol = collectOneClass(os.path.join(inFolder, foldername))
        util.pickle(outFolder +"/"+ foldername, dataCol)

def doGetMeanImg(reszFolder, meanImgFolder):
    meanImg, info = getMeanImg(reszFolder)
    util.pickle(meanImgFolder+"/meanImg", meanImg)

if __name__ == '__main__':
    trainFolder = "/home/chengcheng/Handwriting/NAKAYOSI-img"
    reszFolder = "/database/trainBatches/HWDB/classify/resize"
    meanImgFolder = "/database/trainBatches/HWDB/classify/resize"
    outFolder = "/database/trainBatches/HWDB/classify/on-line_Japanese"
    #labelfVali = "/data1/LSVRC2012/ILSVRC2010_validation_ground_truth.txt"
    #valiFolder = "/data1/LSVRC2012/val"
    #startIdxVali = 8000
    testFolder = "/home/chengcheng/Handwriting/KUCHIBUE-img"
    startIdxTest = 5000
    collectImgByClass(trainFolder, reszFolder)
    doGetMeanImg(reszFolder, meanImgFolder)
    #prepareTrain(trainFolder, outFolder, meanImgFolder)
    #processValidation(labelfVali, valiFolder, outFolder, startIdxVali)
    #fixValidationLabel()
    processTest(testFolder, outFolder, startIdxTest)
    
