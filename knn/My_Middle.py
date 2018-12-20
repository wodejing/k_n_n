# -*- coding: utf-8 -*-
import time
from os import listdir
from math import log
from numpy import *
from numpy import linalg
from operator import itemgetter

def doAdd(oldMap,newMap):
    for word in newMap.keys():
        if word in oldMap.keys():
            oldMap[word] = float(newMap[word]) + float(oldMap[word])
        else:
            oldMap[word]=float(newMap[word])
    return oldMap

#得到每个类的类中心向量
def doMiddle(i):
    trainFiles = 'docVector/wordTFIDF/wordTFIDFMapTrainSample'+str(i)
    middleResultFile = 'docVector/middle/middleResult'+str(i)
    middleResultWriter = open(middleResultFile, 'w')

    trainDocWordMap = {}  # 字典<key, value> key=cate_doc, value={{word1,tfidf1}, {word2, tfidf2},...}
    countMap={}

    for line in open(trainFiles).readlines():
        trainWordMap = {}
        lineSplitBlock = line.strip('\n').split(' ')
        m = len(lineSplitBlock) - 1
        for i in range(2, m, 2):  # 在每个文档向量中提取(word, tfidf)存入字典
            trainWordMap[lineSplitBlock[i]] = lineSplitBlock[i + 1]
        temp_key=lineSplitBlock[0]

        if temp_key in trainDocWordMap.keys():
            trainDocWordMap[temp_key] = doAdd(trainDocWordMap[temp_key], trainWordMap)
        else:
            trainDocWordMap[temp_key]=trainWordMap
        countMap[temp_key] = countMap.get(temp_key, 0.0) + 1.0

    for key in trainDocWordMap.keys():
        print("key: "+key+"  countMap[key]:"+str(countMap[key]))
        middleResultWriter.write('%s ' % (key))
        for word in trainDocWordMap[key]:
            trainDocWordMap[key][word]=float(trainDocWordMap[key][word])/(1.0*countMap[key])
            print(word+" "+str(trainDocWordMap[key][word]))
            middleResultWriter.write('%s %f ' % (word, trainDocWordMap[key][word]))
        middleResultWriter.write('\n')

if __name__ == '__main__':
    for i in range(10):
        doMiddle(i)
