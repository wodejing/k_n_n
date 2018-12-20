# -*- coding: utf-8 -*-
import time
from os import listdir
from math import log
from numpy import *
from numpy import linalg
from operator import itemgetter

########################################################
## 生成训练集和测试集的文档向量
## @param indexOfSample 迭代的序号
## @param trainSamplePercent 训练集合和测试集合划分百分比
########################################################
# def computeTFMultiIDF(indexOfSample, trainSamplePercent,IDFPerWord):
#
#     # IDFPerWord = {}  # <word, IDF值> 从文件中读入后的数据保存在此字典结构中
#     # for line in open('IDFPerWord').readlines():
#     #     (word, IDF) = line.strip('\n').split(' ')
#     #     IDFPerWord[word] = IDF
#
#     fileDir = 'processedSampleOnlySpecial_2'
#     trainFileDir = "docVector/wordTFIDF/" + 'wordTFIDFMapTrainSample' + str(indexOfSample)
#     testFileDir = "docVector/wordTFIDF/" + 'wordTFIDFMapTestSample' + str(indexOfSample)
#
#     tsTrainWriter = open(trainFileDir, 'w')
#     tsTestWriter = open(testFileDir, 'w')
#
#     sumcount=0
#     traincount=0
#     cateList = listdir(fileDir)
#     for i in range(len(cateList)):
#         sumcount+=1
#         sampleDir = fileDir + '/' + cateList[i]
#         sampleList = listdir(sampleDir)
#
#         testBeginIndex = indexOfSample * (len(sampleList) * (1 - trainSamplePercent))
#         testEndIndex = (indexOfSample + 1) * (len(sampleList) * (1 - trainSamplePercent))
#
#         for j in range(len(sampleList)):
#
#             TFPerDocMap = {}  # <word, 文档doc下该word的出现次数>
#             sumPerDoc = 0  # 记录文档doc下的单词总数
#             sample = sampleDir + '/' + sampleList[j]
#             for line in open(sample).readlines():
#                 sumPerDoc += 1
#                 word = line.strip('\n')
#                 TFPerDocMap[word] = TFPerDocMap.get(word, 0) + 1
#
#             if (j >= testBeginIndex) and (j <= testEndIndex):
#                 tsWriter = tsTestWriter
#
#             else:
#                 tsWriter = tsTrainWriter
#                 traincount+=1
#
#             tsWriter.write('%s %s ' % (cateList[i], sampleList[j]))  # 写入类别cate，文档doc
#
#
#             for word, count in TFPerDocMap.items():
#                 TF = float(count) / float(sumPerDoc)
#                 tsWriter.write('%s %f ' % (word, TF * float(IDFPerWord[word])))  # 继续写入类别cate下文档doc下的所有单词及它的TF-IDF值
#             tsWriter.write('\n')
#         print('just finished %d round ' % i)
#     print('count: %d   traincount %d' % (sumcount, traincount))
#
#
#         # if i==0: break
#     tsTrainWriter.close()
#     tsTestWriter.close()
#     tsWriter.close()

def computeTFMultiIDF(indexOfSample, trainSamplePercent,IDFPerWord):

    # IDFPerWord = {}  # <word, IDF值> 从文件中读入后的数据保存在此字典结构中
    # for line in open('IDFPerWord').readlines():
    #     (word, IDF) = line.strip('\n').split(' ')
    #     IDFPerWord[word] = IDF
    fileDir = 'processedSampleOnlySpecial_2'
    trainFileDir = "docVector/wordTFIDF/" + 'wordTFIDFMapTrainSample' + str(indexOfSample)
    testFileDir = "docVector/wordTFIDF/" + 'wordTFIDFMapTestSample' + str(indexOfSample)

    tsTrainWriter = open(trainFileDir, 'w')
    tsTestWriter = open(testFileDir, 'w')

    sumcount=0
    traincount=0
    cateList = listdir(fileDir)
    for i in range(len(cateList)):
        sampleDir = fileDir + '/' + cateList[i]
        sampleList = listdir(sampleDir)

        testBegin = (math.modf(float(indexOfSample) * (1 - trainSamplePercent)))[0]
        testEnd = (math.modf((float(indexOfSample+1)) * (1 - trainSamplePercent)))[0]

        testBeginIndex = len(sampleList)*testBegin
        testEndIndex = len(sampleList) * testEnd
        print("testBeginIndex:"+str(testBeginIndex)+"  testEndIndex:"+str(testEndIndex))
        temp=len(sampleList)*trainSamplePercent
        print(sampleDir+": "+str(temp) +"  "+str(len(sampleList)-temp) )


        for j in range(len(sampleList)):

            TFPerDocMap = {}  # <word, 文档doc下该word的出现次数>
            sumPerDoc = 0  # 记录文档doc下的单词总数
            sample = sampleDir + '/' + sampleList[j]
            for line in open(sample).readlines():
                sumPerDoc += 1
                word = line.strip('\n')
                TFPerDocMap[word] = TFPerDocMap.get(word, 0) + 1

            sumcount+=1
            if testBeginIndex > testEndIndex:
                if j>testBeginIndex or j<testEndIndex:
                    tsWriter = tsTestWriter
                else:
                    tsWriter = tsTrainWriter
                    traincount += 1
            else:
                if (j >= testBeginIndex) and (j <= testEndIndex):
                    tsWriter = tsTestWriter

                else:
                    tsWriter = tsTrainWriter
                    traincount += 1

            tsWriter.write('%s %s ' % (cateList[i], sampleList[j]))  # 写入类别cate，文档doc


            for word, count in TFPerDocMap.items():
                TF = float(count) / float(sumPerDoc)
                tsWriter.write('%s %f ' % (word, TF * float(IDFPerWord[word])))  # 继续写入类别cate下文档doc下的所有单词及它的TF-IDF值
            tsWriter.write('\n')
        # print('just finished %d round ' % i)
    print('sumcount: %d   traincount %d' % (sumcount, traincount))


        # if i==0: break
    tsTrainWriter.close()
    tsTestWriter.close()
    tsWriter.close()



if __name__ == '__main__':
    IDFPerWord = {}  # <word, IDF值> 从文件中读入后的数据保存在此字典结构中
    for line in open('docVector/IDFPerWord').readlines():
        (word, IDF) = line.strip('\n').split(' ')
        IDFPerWord[word] = IDF

    for i in range(10):
        computeTFMultiIDF(i, 0.7, IDFPerWord)
