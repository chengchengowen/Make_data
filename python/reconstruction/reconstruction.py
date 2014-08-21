# -*- coding: utf-8 -*-
"""
Created on 2014.02.13

@author: lyq
process the images into batches for reconstruction

modify by Cheng Cheng on 2014.03.03
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
import pdb
import cv2

SIZE = 160
SIZE_STD = 64
CHANNELS = 1 

NUM_PER_BATCH = 1024

def getMeanImg(raw_dir):
    global SIZE, CHANNELS
    print SIZE
    flist = os.listdir(raw_dir)
    #print flist
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
    print globalCount
    print globalSum
    return meanImg, raw_info

def getMeta(traindir):
    dirnames = os.listdir(traindir)
    print 'Found %d classes under %s' % (len(dirnames), traindir)
    meta = {}
    meta['label_names'] = dirnames
    return meta
    

def readAndResize(fileName, szAfter, chnAfter):
    #print fileName
    img = scipy.misc.imread(fileName,1)
    #pdb.set_trace()
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
    global SIZE, SIZE_STD, CHANNELS
    print " going to load %d images " % len(meta)
    data = np.zeros((SIZE*SIZE*CHANNELS, len(meta)), dtype=np.uint8)
    dataStd = np.zeros((SIZE_STD*SIZE_STD*CHANNELS, len(meta)), dtype=np.uint8)
    # added by shao, 20140410
    imgNames = []
    for i, entry in enumerate(meta):
        try:
            data[:,i] = readAndResize(entry[0], SIZE, CHANNELS)
            dataStd[:, i] = readAndResize(entry[1], SIZE_STD, CHANNELS)
            # added by shao, 20140410
            imgFullPath = entry[0]
            ind = imgFullPath.rfind("/")
            imgNames.append(imgFullPath[ind+1:])
            # img = cv2.imread(entry[0],0)
            # imgstd = cv2.imread(entry[1],0)

            # cv2.imwrite("img.png", img)
            # cv2.imwrite("imgstd.png", imgstd)

            #cv2.imshow("img",img)
            #cv2.imshow("imgstd",imgstd)
            #cv2.WaitKey(0)


            
        except IOError as e:
            print entry[0]
            exit(1)
    return data, dataStd, imgNames

def datainformation(data_dir):
    subdirnames = os.listdir(data_dir)
    list.sort(subdirnames)
    for classLabel, subdir in enumerate(subdirnames):
        imgnames = os.listdir(os.path.join(data_dir, subdir))
        fullnames = [os.path.join(data_dir, subdir, name) for name in imgnames]
        print "####### Got %d images ######" % len(imgnames)
        raw_input()

def collectAndShuffle(train_dir, stdImgfolder):
    #str = "str"
    #print str
    allImgMeta = []
    allLabels = []
    subdirnames = os.listdir(train_dir)
    list.sort(subdirnames)
    for classLabel, subdir in enumerate(subdirnames):
        allLabels.append(stdImgfolder + '/' + subdir + '.png')
        imgnames = os.listdir(os.path.join(train_dir, subdir))
        fullnames = [os.path.join(train_dir, subdir, name) for name in imgnames]
        stm_name = stdImgfolder + '/' + subdir + '.png'

        # name, label pair
        meta = zip(fullnames, [stm_name] * len(fullnames), [subdir] * len(fullnames))
        #print meta

        
        #raw_input()
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
        # modified by shao, 20140410
        data, dataStd, imgNames = getBatch(batchMeta)

        out_fname = os.path.join(out_dir, "data_batch_%04d" % (idx_batch+startIdx))
        print "saving to %s" % out_fname
        # modified by shao, 20140410
        util.pickle(out_fname, {'data':data, 'data_std':dataStd, 'img_name': imgNames})
    
     

def prepareTraining(tr_folder, meanImg_dir, out_dir):
    global NUM_PER_BATCH
    
    meanImg = util.unpickle(meanImg_dir + '/meanImg')
    meanImgStd = util.unpickle(meanImg_dir + '/meanImgStd')
    allImgMeta = collectAndShuffle(tr_folder)
    meta = {}
    meta['data_mean'] = meanImg
    meta['data_mean_std'] = meanImgStd
    util.pickle( os.path.join(out_dir, "batches.meta"), meta)

    makeBatches(allImgMeta, out_dir, NUM_PER_BATCH)
    out_file = out_dir + "/imglist"
    util.pickle(out_file, allImgMeta)

#lyq add
def collectOneClass(classFolder,image_size,image_channels):
    #global SIZE, CHANNELS
    #print classFolder
    filelist = os.listdir(classFolder)
    list.sort(filelist)
    dataCol = np.zeros((len(filelist), image_size*image_size*image_channels), dtype = np.uint8)
    for index, filename in enumerate(filelist):
        dataCol[index,:] = readAndResize(os.path.join(classFolder, filename), image_size, image_channels)
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
    
def parseStdImgList(Tr_folder_Img):
    Tr_List = os.listdir(Tr_folder_Img)

    #fileStdList = open(Tr_folder_Img, 'r')
    #fileStdList = listStd
    
    folderCls = []
    imgStdCls = []
    for line in Tr_List:
        print line
        posSuf = line.rfind('.')
        nmImg  = line[0 : posSuf]
        cell   = nmImg.split('_')
        imgCls = Tr_folder_Img + '/' + cell[0]
        if imgCls not in folderCls:
            folderCls.append(imgCls)
            arr = []
            arr.append(line.strip())
            imgStdCls.append(arr)
        else:
            ind = folderCls.index(imgCls)
            imgStdCls[ind].append(line.strip())
    return folderCls


def collectImgByName(InputFolder, outFolder):
    flist = os.listdir(InputFolder)
    flist = [os.path.join(InputFolder, fname) for fname in flist]
    for  foldername in flist:
        print foldername
        dataCol = collectOneClass(foldername,SIZE,CHANNELS)
        ind = foldername.rfind("/")
        folder = foldername[ind+1:]
        util.pickle(outFolder +"/"+ folder, dataCol)

def doGetMeanImgStd(stdImgfolder, meanImgFolder):
    global SIZE_STD, CHANNELS , SIZE
    #print SIZE_STD
    dataCol = collectOneClass(stdImgfolder,SIZE_STD,CHANNELS)
    dataSum = np.sum(dataCol, axis=0, dtype = np.float64)
    globalSum = dataSum
    globalCount = dataCol.shape[0]
    meanImg = globalSum / globalCount 
    #print globalSum
    #meanImg, info=getMeanImg(stdImgfolder)    
    util.pickle(meanImgFolder+"/meanImgStd", meanImg)
    #resizedImg = np.imresize(meanImg,(SIZE_STD,SIZE_STD))  

def prepareTrain(train_dir, stdImgfolder, out_dir, meanImg_dir, startIdx):
    global NUM_PER_BATCH


    meanImg = util.unpickle(meanImg_dir + '/meanImg')
    meanImgStd = util.unpickle(meanImg_dir + '/meanImgStd')
    allImgMeta, allLabels = collectAndShuffle(train_dir, stdImgfolder)
    meta = {}
    meta['data_mean'] = meanImg
    meta['data_mean_std'] = meanImgStd
    util.pickle( os.path.join(out_dir, "batches.meta"), meta)

    makeBatches(allImgMeta, out_dir, NUM_PER_BATCH, startIdx)
    out_file = out_dir + "/imglist"
    util.pickle(out_file, allImgMeta)

def prepareTest(Test_dir, stdImgfolder, out_dir, meanImg_dir, startIdx):
    global NUM_PER_PATCH
  
    allImgMeta, allLabels = collectAndShuffle(Test_dir, stdImgfolder)
  
    makeBatches(allImgMeta, out_dir, NUM_PER_BATCH, startIdx)
    out_file = out_dir + "/imglist"
    util.pickle(out_file, allImgMeta)


def labels_info(trainFolder, testFolder):
    trn_list = os.listdir(trainFolder)
    tst_list = os.listdir(testFolder)

    for i in range(len(tst_list)):
        trn_list.append(tst_list[i])

    return trn_list
    #flist = tst_list + tst_list

def Pre_For_Reconstruction():

    global NUM_PER_BATCH
    trainFolder = "/database/align_reconstruction/tmp/training"
    testFolder = "/database/align_reconstruction/tmp/test"


    reszFolder = "/database/align_reconstruction/tmp/resize"
    meanImg_dir = "/database/align_reconstruction/tmp/resize"
    stdImgfolder = "/database/align_reconstruction/array_std/zjz"
    out_dir = "/database/align_reconstruction/batches"
    #collectImgByName(trainFolder,reszFolder)
    #doGetMeanImg(reszFolder,meanImg_dir)

    #doGetMeanImgStd(stdImgfolder, meanImg_dir)
    startIdx=0
    prepareTrain(trainFolder, stdImgfolder, out_dir, meanImg_dir, startIdx)

    
    startIdx = 3000
    prepareTest(testFolder, stdImgfolder, out_dir, meanImg_dir, startIdx)






if __name__ == '__main__':

    #trainFolder = "/database/Array/exp_no_oclu/Train"
    #Pre_For_Classification()
    #Pre_For_Classification()
    Pre_For_Reconstruction()
    #datainformation(trainFolder)
    #collectImgByName(trainFolder,reszFolder)
    #doGetMeanImg(reszFolder, meanImgFolder)
    #doGetMeanImgStd(stdImgfolder, meanImgFolder)
    #prepareTraining(trainFolder, meanImgFolder, outFolder)
    #collectAndShuffle(trainFolder)


    #stdImgListTrain = os.listdir(trainFolder)
    #num = len(stdImgListTrain)
    #folderCls = parseStdImgList(trainFolder)

    
    
    #rint num
    #collectImgByClass(folderCls, reszFolder)
    #
    #
    #
    #folderClsTest, imgStdClsTest = parseStdImgList(stdImgListTest, trainFolder)
    #processTest(folderClsTest, imgStdClsTest, startIdxTest, outFolder)
    
