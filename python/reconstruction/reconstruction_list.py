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

SIZE = 160
SIZE_STD = 32
CHANNELS = 1 
NUM_PER_PATCH = 1024
Mean_Img_Num = 5000

 

def readAndResize(fileName, global_size, global_cha):
    #global SIZE, CHANNELS
    img = scipy.misc.imread(fileName)
    size = img.shape[:2]
    ratio = np.float32(global_size)/min(size)
    resizedImg = scipy.misc.imresize(img, (int(round((size[0]*ratio))),int(round((size[1]*ratio)))))
    if len(resizedImg.shape)==2:
        # this is an grey img
        tmp = np.zeros((resizedImg.shape[0], resizedImg.shape[1], global_cha), np.uint8)
        for ch in range(global_cha):
            tmp[:,:,ch] = resizedImg
        resizedImg = tmp
        #    print resizedImg.shape
    
    try:
        croppedImg = resizedImg
        if resizedImg.shape[0] == global_size:
            if resizedImg.shape[1] == global_size:
                croppedImg = resizedImg
            else:
                offset1low = (resizedImg.shape[1]-global_size)/2
                offset1high = -((resizedImg.shape[1]-global_size)/2)
                if resizedImg.shape[1] % 2 == 1:
                    offset1high -= 1
                croppedImg = resizedImg[:, offset1low:offset1high, :]
        else:
            offset0low = (resizedImg.shape[0]-global_size)/2
            offset0high = -((resizedImg.shape[0]-global_size)/2)
            if resizedImg.shape[0] % 2 == 1:
                offset0high -= 1
            croppedImg = resizedImg[offset0low:offset0high, :, :]
            #if croppedImg.shape[0] != 256 or croppedImg.shape[1] != 256 or croppedImg.shape[2] != 3:
            # print fileName, croppedImg.shape
        vec = np.array([],dtype=np.uint8);
        for ch in range(global_cha):
            vectmp = np.reshape(croppedImg[:,:,ch], global_size*global_size)
            vec = np.concatenate((vec, vectmp))
        return vec
    except IndexError as e:
    
        print resizedImg.shape, fileName
        sys.exit(1)        
        #    print croppedImg.shape

    

def getBatch(meta, allLabels):
    global SIZE, CHANNELS
    print " going to load %d images " % len(meta)
    data = np.zeros((SIZE*SIZE*CHANNELS, len(meta)), dtype=np.uint8)
    labels = np.zeros((len(meta)), dtype=np.single)
    imgnames = []
    
    # for line in meta:
    #     str = line.strip()

    #     label[] = tmp[length - 2]
    #     if label not in allLabels:
    #         allLabels.append(label) 



    for i in range(0,len(meta)):
        str = meta[i].strip()
        tmp = str.split('/')
        length = len(tmp)
        labels[i] = allLabels.index(tmp[length - 2])
        imgnames.append(tmp[length - 1])
        try:
            data[:,i] = readAndResize(meta[i].strip())
                  
        except IOError as e:
            print meta[i].strip()
            exit(1)
    return data, labels, imgnames

 
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
        data, labels, imgnames = getBatch(batchMeta)
        #labels1 = labels//2
        #labels2 = labels//3
        #labels3 = labels//4

        out_fname = os.path.join(out_dir, "data_batch_%04d" % (idx_batch+startIdx))
        print "saving to %s" % out_fname
        util.pickle(out_fname, {'data':data, 'labels':labels, 'images':imgnames})
            




#lyq add
def collectOneClass(classFolder):
    global SIZE, CHANNELS
    filelist = os.listdir(classFolder)
    list_len = len(filelist)
    if list_len > 300:
        del filelist[300:list_len]
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
        #if index > 10000:
        dataCol = collectOneClass(os.path.join(inFolder, foldername))
        util.pickle(outFolder +"/"+ foldername, dataCol)

 
def prepareTrain(train_list, outFolder):
    global NUM_PER_PATCH
    fileList = open(train_list,'rb').readlines()
    random.shuffle(fileList)

    if len(fileList) < Mean_Img_Num:
         num_mean_img = len(fileList)
    else:
         num_mean_img = Mean_Img_Num

    data = np.zeros((SIZE*SIZE*CHANNELS, num_mean_img), dtype=np.uint8)

    for i in range(0,num_mean_img):
        data[:,i] = readAndResize(fileList[i].strip())


    

    tmp = data.reshape(data.shape[1],data.shape[0])
    dataSum = np.sum(tmp, axis=0, dtype = np.float64)

    globalCount = tmp.shape[0]
    meanImg = dataSum / globalCount
    util.pickle(outFolder+"/meanImg", meanImg)
    
    allLabels = []
    allImgMeta = []
    for line in fileList:
        str = line.strip()
        tmp = str.split('/')
        length = len(tmp)
        label = tmp[length - 2]
        #print label
        if label not in allLabels:
            allLabels.append(label)            

    print "####### Got %d classes ######" % len(allLabels)
    meta = {}
    meta['data_mean'] = meanImg
    meta['label_names'] = allLabels
    util.pickle( os.path.join(out_dir, "batches.meta"), meta)




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
        out_fname = os.path.join(out_dir, "data_batch_%04d" % idx_batch)
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

def getBatch(meta, allLabels):
    global SIZE, CHANNELS
    print " going to load %d images " % len(meta)
    data = np.zeros((SIZE*SIZE*CHANNELS, len(meta)), dtype=np.uint8)
    labels = np.zeros((len(meta)), dtype=np.single)
    imgnames = []

    for i in range(0,len(meta)):
        str = meta[i].strip()
        tmp = str.split('/')
        length = len(tmp)
        labels[i] = allLabels.index(tmp[length - 2])
        imgnames.append(tmp[length - 1])
        try:
            data[:,i] = readAndResize(meta[i].strip())
                  
        except IOError as e:
            print meta[i].strip()
            exit(1)
    return data, labels, imgnames

def get_list_batch(meta):
    global SIZE, SIZE_STD, CHANNELS
    print " going to load %d images " % len(meta)
    data = np.zeros((SIZE*SIZE*CHANNELS, len(meta)), dtype=np.uint8)
    dataStd = np.zeros((SIZE_STD*SIZE_STD*CHANNELS, len(meta)), dtype=np.uint8)
    imgnames = []

    for i , entry in enumerate(meta):
        str = entry[0]
        tmp = str.split('/')
        length = len(tmp)
        #labels[i] = allLabels.index(tmp[length - 2])
        imgnames.append(tmp[length - 1])
        try:
            data[:,i] = readAndResize(entry[0],SIZE,CHANNELS)
            dataStd[:,i] = readAndResize(entry[1],SIZE_STD,CHANNELS)
                  
        except IOError as e:
            print meta[i].strip()
            exit(1)
    return data, dataStd, imgnames

def make_list_batches(allImgMeta, out_dir, batchSize, startIdx = 0):
    numImg = len(allImgMeta)
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
        batchMeta = allImgMeta[b_start:b_end]
        data, dataStd, imgnames = get_list_batch(batchMeta)

        out_fname = os.path.join(out_dir, "data_batch_%04d" % (idx_batch+startIdx))
        print "saving to %s" % out_fname
        util.pickle(out_fname, {'data':data, 'data_std':dataStd, 'name':imgnames})



def process_for_reconstruction(img_list, std_file, out_dir):
    #subFolderList = os.listdir(std_file)

    meta = {}

    meanImg = readAndResize('/database/test.jpg',SIZE, CHANNELS)
    meanImgstd = readAndResize('/database/test.jpg', SIZE_STD, CHANNELS)

    meta['data_mean'] = meanImg
    meta['data_mean_std'] = meanImgstd
    util.pickle(os.path.join(out_dir,"batches.meta"), meta)
    std_img = [fname for fname in os.listdir(std_file) if fname.endswith('.png')]
    #print len(subFolderList)
    #std_img.sort()


    fileList = open(img_list,'rb').readlines()

    random.shuffle(std_img)
    trainingMeta = []
    testMeta = []
    train_test_ratio = 0.8*len(std_img)
    #print train_test_ratio
    for i , std_name in enumerate(std_img):
        tmp = std_name.split('.')
        #length = len(std_name)
        label = tmp[0]
        num = 0
        meta = []
        for line in fileList:
            #nPos = line.index(label)
            # str = line.strip()
            # tmp = str.split('/')
            # length = len(tmp)
            # line_label = tmp[length - 2]
            if label in  line:
                meta.append(line.strip())
                num = num + 1

        if i < train_test_ratio:
            trainingMeta += zip(meta,[os.path.join(std_file,std_name)] * len(meta))
        else:
            testMeta += zip(meta,os.path.join(std_file,std_name) * len(meta))
        
        str = "i = %d, name = %s, trainmeta = %d, testmeta = %d" %(i,std_name,len(trainingMeta), len(testMeta))
        print str
        if num == 0:
            del std_img[i]
            # str = "i = %d, name = %s, trainmeta = %d" %(i,std_name,len(trainingMeta))
            # print str

    util.pickle(os.path.join(out_dir,"trainingMeta.meta"), trainingMeta)
    util.pickle(os.path.join(out_dir,"testMeta.meta"), testMeta)
    #for training
    print "prepare for training"
    random.shuffle(trainingMeta)
    make_list_batches(trainingMeta,out_dir,NUM_PER_PATCH)



    #for test
    print "prepare for test"
    random.shuffle(testMeta)
    make_list_batches(trainingMeta,out_dir,NUM_PER_PATCH,8000)




    #print len(allImgMeta)
def prepare_for_rec(out_dir):

    meta = {}

    meanImg = readAndResize('/database/test.jpg',SIZE, CHANNELS)
    meanImgstd = readAndResize('/database/test.jpg', SIZE_STD, CHANNELS)

    meta['data_mean'] = meanImg
    meta['data_mean_std'] = meanImgstd
    util.pickle(os.path.join(out_dir,"batches.meta"), meta)
    

    trainingMeta = util.unpickle(os.path.join(out_dir, "trainingMeta.meta"))

    testMeta = util.unpickle(os.path.join(out_dir, "testMeta.meta"))

    print "prepare for training"
    random.shuffle(trainingMeta)
    make_list_batches(trainingMeta,out_dir,NUM_PER_PATCH)



    #for test
    print "prepare for test"
    random.shuffle(testMeta)
    make_list_batches(trainingMeta,out_dir,NUM_PER_PATCH,8000)

if __name__ == '__main__':

    startIdxTest = 8000

    train_list = "/database/ChengCheng/list/reconstruction.txt"
    #test_list = "/home/chengcheng/Matlab/list/all/test.txt"

    out_dir = "/database/ChengCheng/trainBatches/reconstruction"
    std_file = "/database/ChengCheng/STD"
    #prepareTrain(train_list,out_dir)

    #processTest(train_list, out_dir, startIdxTest)
    #repare_for_rec(out_dir)
    process_for_reconstruction(train_list, std_file, out_dir)



    
