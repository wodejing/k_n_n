# -*- coding: utf-8 -*-
import time
from os import listdir
from math import log
from numpy import *
from numpy import linalg
from operator import itemgetter
from My_Dis import computeSim

def changeFileToMap(file):
    docWordMap = {}  # 字典<key, value> key=cate_doc, value={{word1,tfidf1}, {word2, tfidf2},...}

    for line in open(file).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        wordMap = {}
        m = len(lineSplitBlock) - 1
        for i in range(2, m, 2):  # 在每个文档向量中提取(word, tfidf)存入字典
            wordMap[lineSplitBlock[i]] = lineSplitBlock[i + 1]

        temp_key = lineSplitBlock[0] + '_' + lineSplitBlock[1]  # 在每个文档向量中提取类目cate，文档doc，
        docWordMap[temp_key] = wordMap
    return docWordMap

def doProcess(n,k):
    itemList = listdir('originSample')
    kNNResultFile = 'docVector/KNNClassifyResult'
    KNNResultWriter = open(kNNResultFile, 'w')
    # trainDocWordMap,testDocWordMap = computeTFMultiIDF(0, 0.9)  # 字典<key, value> key=cate_doc, value={{word1,tfidf1}, {word2, tfidf2},...}

    sumPMap={}  # 预测属于该类,实际属于该类
    sumRMap={}  # 预测属于该类，实际上不属于该类
    sumFMap={}  # 预测不属于该类，实际上属于该类


    for i in range(n):
        trainFiles = 'docVector/wordTFIDF/wordTFIDFMapTrainSample' + str(i)
        testFiles = 'docVector/wordTFIDF/wordTFIDFMapTestSample' + str(i)
        trainDocWordMap = changeFileToMap(trainFiles)  # 字典<key, value> key=cate_doc, value={{word1,tfidf1}, {word2, tfidf2},...}
        testDocWordMap = changeFileToMap(testFiles)
        print("第"+str(i)+"次")
        KNNResultWriter.write('%d\n' % (i))
        count = 0  # 处理的总分类数
        countAMap = {}  # 预测属于该类,实际属于该类
        countBMap = {}  # 预测属于该类，实际上不属于该类
        countCMap = {}  # 预测不属于该类，实际上属于该类
        for item in testDocWordMap.items():
            classifyRight = item[0].split('_')[0]
            classifyResult = KNNComputeCate(item[0], item[1], trainDocWordMap,k)  # 调用KNNComputeCate做分类
            count += 1
            print('this is %d round' % count)

            if classifyRight == classifyResult:
                countAMap[classifyResult] = countAMap.get(classifyResult, 0) + 1
            else:
                countBMap[classifyResult] = countBMap.get(classifyResult, 0) + 1
                countCMap[classifyRight] = countCMap.get(classifyRight, 0) + 1
        for item in countAMap:
            a = countAMap.get(item, 0)
            b = countBMap.get(item, 0)
            c = countCMap.get(item, 0)
            P = float(a) / float(a + b)
            R = float(a) / float(a + c)
            F = P * R * 2 / (P + R)

            KNNResultWriter.write('%s P: %f R: %f F: %f\n' % (item,P, R, F))
            sumPMap[item] = P + sumPMap.get(item,0)
            sumRMap[item] = R + sumRMap.get(item, 0)
            sumFMap[item] = F + sumFMap.get(item, 0)

    sumP = 0.0
    sumR = 0.0
    sumF = 0.0
    KNNResultWriter.write("10 average\n")
    for item in itemList:
        P = sumPMap.get(item, 0) / (1.0 * n)
        R = sumRMap.get(item, 0) / (1.0 * n)
        F = sumFMap.get(item, 0) / (1.0 * n)
        sumP = sumP + P
        sumR = sumR + R
        sumF = sumF + F
        KNNResultWriter.write('%s P: %f R: %f F: %f\n' % (item,P, R, F))
        print(item + " P:" + str(P) + " R:" + str(R) + " F:" + str(F))
    length=len(itemList)
    KNNResultWriter.write(" average\n")
    KNNResultWriter.write('P: %f R: %f F: %f\n' % (sumP/(1.0*length), sumR/(1.0*length), sumF/(1.0*length)))
    print("avage " " P:" + str(sumP/(1.0*length)) + " R:" + str(sumR/(1.0*length)) + " F:" + str(sumF/(1.0*length)))
    KNNResultWriter.close()
    return sumP/(1.0*length)




#########################################################
## @param cate_Doc 测试集<类别_文档>
## @param testDic 测试集{{word, TFIDF}}
## @param trainMap 训练集<类_文件名，<word, TFIDF>>
## @return sortedCateSimMap[0][0] 返回与测试文档向量距离和最小的类
#########################################################
def KNNComputeCate(cate_Doc, testDic, trainMap,k):
    simMap = {}  # <类目_文件名,距离> 后面需要将该HashMap按照value排序
    for item in trainMap.items():
        similarity = computeSim(testDic, item[1])  # 调用computeSim()
        simMap[item[0]] = similarity
        print("similarity:"+str(similarity))

    sortedSimMap = sorted(simMap.items(), key=itemgetter(1), reverse=True)  # <类目_文件名,距离> 按照value排序

    cateSimMap = {}  # <类，距离和>
    for i in range(k):
        print("前10个sim: "+str(sortedSimMap[i][1]) )
        cate = sortedSimMap[i][0].split('_')[0]
        cateSimMap[cate] = cateSimMap.get(cate, 0) + sortedSimMap[i][1]

    sortedCateSimMap = sorted(cateSimMap.items(), key=itemgetter(1), reverse=True)

    return sortedCateSimMap[0][0]

def find_k():
    n=1
    PValue = 'docVector/PValue'
    PValueWriter = open(PValue, 'w')
    for k in range(4,10,1):
        result = doProcess(n, k)
        PValueWriter.write('%d %f\n' % (k, result))
    PValueWriter.close()


if __name__ == '__main__':
    # find_k()
    n=10
    k=8
    start = time.clock()
    doProcess(n, k)
    end = time.clock()
    print('runtime: ' + str(end - start))








