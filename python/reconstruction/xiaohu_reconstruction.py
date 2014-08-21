# -*- coding: utf-8 -*-
"""
Modified by shao,
20140404
process dataset when intial images and ground truth are in different folders

Created on 2014.02.13

@author: lyq
process the images into batches for reconstruction
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

SIZE = 160
SIZE_STD = 160
DATA_CHANNELS = 1 
STD_CHANNELS  = 1
NUM_PER_BATCH = 512

def getMeanImg(raw_dir):
    global SIZE, DATA_CHANNELS
    flist = os.listdir(raw_dir)
    list.sort(flist)
    globalSum = np.zeros(SIZE*SIZE*DATA_CHANNELS, dtype=np.float64)
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
    

def readAndResize(fileName, szAfter, chnAfter):
    if chnAfter == 1:
        img = scipy.misc.imread(fileName, 1)
    elif chnAfter == 3:
        img = scipy.misc.imread(fileName)
    
    size = img.shape[:2]
    ratio = np.float32(szAfter)/min(size)
    resizedImg = scipy.misc.imresize(img, (int(round((size[0]*ratio))),int(round((size[1]*ratio)))))
    if len(resizedImg.shape)==2:
        # this is an grey img
        tmp = np.zeros((resizedImg.shape[0], resizedImg.shape[1], chnAfter), np.uint8)
        for ch in range(chnAfter):
            tmp[:,:,ch] = resizedImg
        resizedImg = tmp
        #    print resizedImg.shape
    
    try:
        croppedImg = resizedImg
        if resizedImg.shape[0] == szAfter:
            if resizedImg.shape[1] == szAfter:
                croppedImg = resizedImg
            else:
                offset1low = (resizedImg.shape[1]-szAfter)/2
                offset1high = -((resizedImg.shape[1]-szAfter)/2)
                if resizedImg.shape[1] % 2 == 1:
                    offset1high -= 1
                croppedImg = resizedImg[:, offset1low:offset1high, :]
        else:
            offset0low = (resizedImg.shape[0]-szAfter)/2
            offset0high = -((resizedImg.shape[0]-szAfter)/2)
            if resizedImg.shape[0] % 2 == 1:
                offset0high -= 1
            croppedImg = resizedImg[offset0low:offset0high, :, :]
            #if croppedImg.shape[0] != 256 or croppedImg.shape[1] != 256 or croppedImg.shape[2] != 3:
            # print fileName, croppedImg.shape
        vec = np.array([],dtype=np.uint8);
        for ch in range(chnAfter):
            vectmp = np.reshape(croppedImg[:,:,ch], szAfter*szAfter)
            vec = np.concatenate((vec, vectmp))
        return vec
    except IndexError as e:
    
        print resizedImg.shape, fileName
        sys.exit(1)        
        #    print croppedImg.shape

    
def getBatch(meta):
    global SIZE, SIZE_STD, DATA_CHANNELS
    print " going to load %d images " % len(meta)
    data = np.zeros((SIZE*SIZE*DATA_CHANNELS, len(meta)), dtype=np.uint8)
    dataStd = np.zeros((SIZE_STD*SIZE_STD*STD_CHANNELS, len(meta)), dtype=np.uint8)
    # added by shao, 20140410
    imgNames = []
    for i, entry in enumerate(meta):
        try:
            data[:,i] = readAndResize(entry[0], SIZE, DATA_CHANNELS)
            dataStd[:, i] = readAndResize(entry[1], SIZE_STD, STD_CHANNELS)
            # added by shao, 20140410
            imgFullPath = entry[0]
            ind = imgFullPath.rfind("/")
            imgNames.append(imgFullPath[ind+1:])
            
        except IOError as e:
            print entry[0]
            exit(1)
    return data, dataStd, imgNames


def processValidation(labelfname, validir, out_dir, startIdx):
    global NUM_PER_BATCH
    #labelfname = '/data1/LSVRC2012/ILSVRC2010_validation_ground_truth.txt'
    #validir = "/data1/LSVRC2012/val"
    #out_dir = "/data2/ILSVRC2010/train_batchesd"
    labels = list(np.loadtxt(labelfname)-1)

    flist = os.listdir(validir)
    list.sort(flist)
    flist = [os.path.join(validir, fname) for fname in flist]

    makeBatches(zip(flist, labels), out_dir, NUM_PER_BATCH, startIdx), 


def processTest(folderCls, imgStdCls, startIdx, out_dir):
    global NUM_PER_BATCH

    allImgMeta = collectAndShuffle(folderCls, imgStdCls)
    makeBatches(allImgMeta, out_dir, NUM_PER_BATCH, startIdx)

  
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

    
def collectAndShuffle(folderCls, imgStdCls):
    allImgMeta = []
    allLabels = []
    for ind in range(len(folderCls)):
        subfolder = folderCls[ind]
        imgnames = os.listdir(subfolder)
        fullnames = [os.path.join(subfolder, name) for name in imgnames]
        # modified by shao, 20140404
        # nameStd = os.path.join(subfolder, imgStdCls[ind][0])
        nameStd = os.path.join(imgStdCls[ind])

        # name, std image pair
        meta = zip(fullnames, [nameStd] * len(fullnames))
        #print meta #for test
        allImgMeta += meta
    print "####### Got %d classes ######" % len(folderCls)
    print "####### Got %d images ######" % len(allImgMeta)
    print "shuffling..."
    random.shuffle(allImgMeta)
    return allImgMeta
        

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
        # modified by shao, 20140410
        data, dataStd, imgNames = getBatch(batchMeta)

        out_fname = os.path.join(out_dir, "data_batch_%04d" % (idx_batch+startIdx))
        print "saving to %s" % out_fname
        # modified by shao, 20140410
        util.pickle(out_fname, {'data':data, 'data_std':dataStd, 'img_name': imgNames})
            

def prepareTrain(folderCls, imgStdCls, meanImg_dir, out_dir):
    global NUM_PER_BATCH
    
    meanImg = util.unpickle(meanImg_dir + '/meanImg')
    meanImgStd = util.unpickle(meanImg_dir + '/meanImgStd')
    allImgMeta = collectAndShuffle(folderCls, imgStdCls)
    meta = {}
    meta['data_mean'] = meanImg
    meta['data_mean_std'] = meanImgStd
    util.pickle( os.path.join(out_dir, "batches.meta"), meta)

    makeBatches(allImgMeta, out_dir, NUM_PER_BATCH)
    out_file = out_dir + "/imglist"
    util.pickle(out_file, allImgMeta)

#lyq add
def collectOneClass(classFolder):
    global SIZE, DATA_CHANNELS
    filelist = os.listdir(classFolder)
    list.sort(filelist)
    dataCol = np.zeros((len(filelist), SIZE*SIZE*DATA_CHANNELS), dtype = np.uint8)
    for index, filename in enumerate(filelist):
        dataCol[index,:] = readAndResize(os.path.join(classFolder, filename), SIZE, DATA_CHANNELS)
    return dataCol

def collectImgByClass(folderCls, outFolder):
    for index, foldername in enumerate(folderCls):
        dataCol = collectOneClass(foldername)
        ind = foldername.rfind("/")
        folder = foldername[ind+1:]
        util.pickle(outFolder +"/"+ folder, dataCol)

def doGetMeanImg(reszFolder, meanImgFolder):
    meanImg, info = getMeanImg(reszFolder)
    util.pickle(meanImgFolder+"/meanImg", meanImg)
    
def parseStdImgList(listStd, folderImg, folderStdImg):
    fileStdList = open(listStd, 'r')
    folderCls = []
    imgStdCls = []
    for line in fileStdList:
        posSuf = line.rfind('.')
        nmImg  = line[0 : posSuf]
        cell   = nmImg.split('_')
        imgCls = os.path.join(folderImg, cell[0])
        if imgCls not in folderCls:
            folderCls.append(imgCls)
            arr = os.path.join(folderStdImg, line.strip())
            imgStdCls.append(arr)
        else:
            ind = folderCls.index(imgCls)
            arr = os.path.join(folderStdImg,line.strip())
            imgStdCls[ind].append(arr)
    return folderCls, imgStdCls
    
def getMeanImgStd(imgStdCls, meanImgFolder):
    global SIZE_STD, STD_CHANNELS
    globalSum = np.zeros(SIZE_STD*SIZE_STD*STD_CHANNELS, dtype=np.float64)
    globalCount = 0
    data = np.zeros((1, SIZE_STD*SIZE_STD*STD_CHANNELS), dtype = np.uint8)
    for ind in range(len(imgStdCls)):
        fullname = imgStdCls[ind]
        print "Reading", fullname
        data[0,:] = readAndResize(fullname, SIZE_STD, STD_CHANNELS)
        dataSum = np.sum(data, axis=0, dtype = np.float64)
        globalSum += dataSum
        globalCount += data.shape[0]  
        
    meanImg = globalSum / globalCount
    util.pickle(meanImgFolder+"/meanImgStd", meanImg)

if __name__ == '__main__':
    stdImgListTrain = "/database/shaoxiaohu/MultiPIE_pose30/train.txt"
    stdImgListTest = "/database/shaoxiaohu/MultiPIE_pose30/test.txt"
    trainFolder = "/database/shaoxiaohu/MultiPIE_pose30/images"
    stdImgFolder = "/database/shaoxiaohu/MultiPIE_pose30/stdImages"
    reszFolder = "/database/shaoxiaohu/MultiPIE_pose30/images_96/train_resize_96"
    meanImgFolder = "/database/shaoxiaohu/MultiPIE_pose30/images_96"
    outFolder = "/database/shaoxiaohu/MultiPIE_pose30/images_96/train_batches_96" 
    
    startIdxTest = 8000
    folderCls, imgStdCls = parseStdImgList(stdImgListTrain, trainFolder, stdImgFolder)
    collectImgByClass(folderCls, reszFolder)
    doGetMeanImg(reszFolder, meanImgFolder)
    getMeanImgStd(imgStdCls, meanImgFolder)
    prepareTrain(folderCls, imgStdCls, meanImgFolder, outFolder)
    folderClsTest, imgStdClsTest = parseStdImgList(stdImgListTest, trainFolder, stdImgFolder)
    processTest(folderClsTest, imgStdClsTest, startIdxTest, outFolder)
