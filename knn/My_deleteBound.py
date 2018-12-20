# -*- coding: utf-8 -*-
import time
from os import listdir
from math import log
from numpy import *
from numpy import linalg
from operator import itemgetter
from My_Dis import computeSim

def doDelete(i):
    trainFiles = 'docVector/wordTFIDF/wordTFIDFMapTrainSample'+str(i)
    middleResultFile = 'docVector/middle/middleResult'+str(i)
    deleteBoundFile = 'docVector/deleteBound/deleteBoundTFIDFMapTrain'+str(i)
    deleteBoundWriter = open(deleteBoundFile, 'w')
    oldcount=0
    newcount=0
    deleteBoundMap={}

    middleResultDocMap = {}

    for line in open(middleResultFile).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        middleResultMap = {}
        m = len(lineSplitBlock) - 1
        for k in range(1, m, 2):  # 在每个文档向量中提取(word, tfidf)存入字典
            middleResultMap[lineSplitBlock[k]] = lineSplitBlock[k + 1]

        temp_key = lineSplitBlock[0]
        middleResultDocMap[temp_key] = middleResultMap

    for line in open(trainFiles).readlines():
        oldcount=oldcount+1
        lineSplitBlock = line.strip('\n').split(' ')
        trainWordMap = {}
        m = len(lineSplitBlock) - 1
        for k in range(2, m, 2):  # 在每个文档向量中提取(word, tfidf)存入字典
            trainWordMap[lineSplitBlock[k]] = lineSplitBlock[k + 1]

        similarity=computeSim(trainWordMap, middleResultDocMap[lineSplitBlock[0]])
        print("similarity"+str(similarity)+"")
        if similarity>0.1:
            newcount=newcount+1
            deleteBoundMap[lineSplitBlock[0]+"_"+lineSplitBlock[1]]=trainWordMap
            deleteBoundWriter.write('%s %s ' % (lineSplitBlock[0],lineSplitBlock[1]));
            for word in trainWordMap:
                deleteBoundWriter.write('%s %s ' % (word, trainWordMap[word]));
            deleteBoundWriter.write('\n')
    print("oldcount"+str(oldcount))
    print("newcount"+str(newcount))

    # 遍历每一个测试样例计算与所有训练样本的距离，做分类


if __name__ == '__main__':
    for i in range(10):
        doDelete(i)