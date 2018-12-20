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

def changeFileToMapList(file):
    docWordMap = {}
    docList = []
    lastKind = ""
    for line in open(file).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        returnMap = {}
        m = len(lineSplitBlock) - 1
        for i in range(2, m, 2):  # 在每个文档向量中提取(word, tfidf)存入字典
            returnMap[lineSplitBlock[i]] = lineSplitBlock[i + 1]

        temp_key = lineSplitBlock[0]
        if temp_key != lastKind:
            if lastKind != "":
                docWordMap[lastKind] = docList
                docList = []
        docList.append(returnMap)
        lastKind = lineSplitBlock[0]
    docWordMap[lastKind] = docList
    return docWordMap

def doProcess(n,k,gsimDiff):
    itemList = listdir('originSample')
    kNNResultFile = 'docVector/KNNGimClassifyResult'
    KNNResultWriter = open(kNNResultFile, 'w')
    sumPMap = {}  # 预测属于该类,实际属于该类
    sumRMap = {}  # 预测属于该类，实际上不属于该类
    sumFMap = {}  # 预测不属于该类，实际上属于该类
    for i in range(n):
        trainFiles = 'docVector/optiBound/optiBoundTFIDFMapTrain' + str(i)
        testFiles = 'docVector/wordTFIDF/wordTFIDFMapTestSample' + str(i)

        trainDocWordMap = changeFileToMapList(trainFiles)
        testDocWordMap = changeFileToMap(testFiles)
        print("第" + str(i) + "次")
        KNNResultWriter.write('%d\n' % (i))

        count = 0  # 处理的总分类数
        countAMap = {}  # 预测属于该类,实际属于该类
        countBMap = {}  # 预测属于该类，实际上不属于该类
        countCMap = {}  # 预测不属于该类，实际上属于该类
        rightCount=0

        for item in testDocWordMap:
            gsimMap = getGsim(testDocWordMap[item], trainDocWordMap)  # 得到组相似度平均值
            # for item in gsimMap:
            #     print(item[0]+" "+str(item[1]))
            count += 1
            print('this is %d round' % count)

            for kind in gsimMap:
                print(kind[0]+"  "+str(kind[1]))


            gsimResult = []  # 存储组相似度相差不大于0.15的分类的结果
            old = 0.0
            for kind in gsimMap:
                if len(gsimResult) == 0:
                    gsimResult.append(kind[0])
                    old = kind[1]
                else:
                    if old - kind[1] < gsimDiff:
                        gsimResult.append(kind[0])
                        old = kind[1]
                    else:
                        break

            classifyRight = item.split('_')[0]
            print("len(gsimResult):"+str(len(gsimResult)))
            # KNNResultWriter.write('%d\n' % (len(gsimResult)))
            if len(gsimResult) == 1:
                classifyResult = gsimResult[0]
            else:
                classifyResult = getResult(testDocWordMap[item], gsimResult,i,k)  # 得到testDocWordMap[item]的在gsimResult的分类结果

            if classifyRight == classifyResult:
                countAMap[classifyResult] = countAMap.get(classifyResult, 0) + 1
                rightCount+=1
            else:
                countBMap[classifyResult] = countBMap.get(classifyResult, 0) + 1
                countCMap[classifyRight] = countCMap.get(classifyRight, 0) + 1
        # print("count:"+str(count)+"  rightCount:"+str(rightCount))

        for item in itemList:
            a = countAMap.get(item,0)
            b = countBMap.get(item,0)
            c = countCMap.get(item,0)
            P = float(a) / float(a + b)
            R = float(a) / float(a + c)
            F = P * R * 2 / (P + R)

            KNNResultWriter.write('%s P: %f R: %f F: %f\n' % (item, P, R, F))
            sumPMap[item] = sumPMap.get(item, 0) + P
            sumRMap[item] = sumRMap.get(item, 0) + R
            sumFMap[item] = sumFMap.get(item, 0) + F
        accuracy = float(rightCount) / float(count)
        print('rightCount : %d , count : %d , accuracy : %.6f' % (rightCount, count, accuracy))
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
        KNNResultWriter.write('%s P: %f R: %f F: %f\n' % (item, P, R, F))
        print(item + " P:" + str(P) + " R:" + str(R) + " F:" + str(F))
    length = len(itemList)
    KNNResultWriter.write("average\n")
    KNNResultWriter.write('P: %f R: %f F: %f\n' % (sumP / (1.0 * length), sumR / (1.0 * length), sumF / (1.0 * length)))
    print("avage " " P:" + str(sumP / (1.0 * length)) + " R:" + str(sumR / (1.0 * length)) + " F:" + str(sumF / (1.0 * length)))

    # accuracy = float(rightCount) / float(count)
    # print('rightCount : %d , count : %d , accuracy : %.6f' % (rightCount, count, accuracy))
    KNNResultWriter.close()
    return sumP / (1.0 * length)

def doProcess_m(n,k,gsimDiff,m):
    itemList = listdir('originSample')
    sumPMap = {}  # 预测属于该类,实际属于该类
    sumRMap = {}  # 预测属于该类，实际上不属于该类
    sumFMap = {}  # 预测不属于该类，实际上属于该类
    for i in range(n):
        trainFiles = 'docVector/optiBoundm/optiBoundTFIDFMapTrain' + str(m)
        testFiles = 'docVector/wordTFIDF/wordTFIDFMapTestSample' + str(i)
        print("m:"+str(m))

        trainDocWordMap = changeFileToMapList(trainFiles)
        testDocWordMap = changeFileToMap(testFiles)
        print("第" + str(i) + "次")

        count = 0  # 处理的总分类数
        countAMap = {}  # 预测属于该类,实际属于该类
        countBMap = {}  # 预测属于该类，实际上不属于该类
        countCMap = {}  # 预测不属于该类，实际上属于该类
        rightCount=0

        for item in testDocWordMap:
            gsimMap = getGsim(testDocWordMap[item], trainDocWordMap)  # 得到组相似度平均值
            # for item in gsimMap:
            #     print(item[0]+" "+str(item[1]))
            count += 1
            print('this is %d round' % count)

            for kind in gsimMap:
                print(kind[0]+"  "+str(kind[1]))


            gsimResult = []  # 存储组相似度相差不大于0.15的分类的结果
            old = 0.0
            for kind in gsimMap:
                if len(gsimResult) == 0:
                    gsimResult.append(kind[0])
                    old = kind[1]
                else:
                    if old - kind[1] < gsimDiff:
                        gsimResult.append(kind[0])
                        old = kind[1]
                    else:
                        break

            classifyRight = item.split('_')[0]
            print("len(gsimResult):"+str(len(gsimResult)))
            # KNNResultWriter.write('%d\n' % (len(gsimResult)))
            if len(gsimResult) == 1:
                classifyResult = gsimResult[0]
            else:
                classifyResult = getResult(testDocWordMap[item], gsimResult,i,k)  # 得到testDocWordMap[item]的在gsimResult的分类结果

            if classifyRight == classifyResult:
                countAMap[classifyResult] = countAMap.get(classifyResult, 0) + 1
                rightCount+=1
            else:
                countBMap[classifyResult] = countBMap.get(classifyResult, 0) + 1
                countCMap[classifyRight] = countCMap.get(classifyRight, 0) + 1
        # print("count:"+str(count)+"  rightCount:"+str(rightCount))

        for item in itemList:
            a = countAMap.get(item,0)
            b = countBMap.get(item,0)
            c = countCMap.get(item,0)
            P = float(a) / float(a + b)
            R = float(a) / float(a + c)
            F = P * R * 2 / (P + R)
            sumPMap[item] = sumPMap.get(item, 0) + P
            sumRMap[item] = sumRMap.get(item, 0) + R
            sumFMap[item] = sumFMap.get(item, 0) + F
        accuracy = float(rightCount) / float(count)
        print('rightCount : %d , count : %d , accuracy : %.6f' % (rightCount, count, accuracy))
    sumP = 0.0
    sumR = 0.0
    sumF = 0.0

    for item in itemList:
        P = sumPMap.get(item, 0) / (1.0 * n)
        R = sumRMap.get(item, 0) / (1.0 * n)
        F = sumFMap.get(item, 0) / (1.0 * n)
        sumP = sumP + P
        sumR = sumR + R
        sumF = sumF + F

        print(item + " P:" + str(P) + " R:" + str(R) + " F:" + str(F))
    length = len(itemList)

    print("avage " " P:" + str(sumP / (1.0 * length)) + " R:" + str(sumR / (1.0 * length)) + " F:" + str(sumF / (1.0 * length)))
    return sumP / (1.0 * length)


def getResult(testWordMap,gsimResult,i,k):
    trainFiles = 'docVector/deleteBound/deleteBoundTFIDFMapTrain'+str(i)

    trainDocWordMap = {}
    docList = []
    lastKind = ""

    #得到每个类别的list[{} {}]
    for line in open(trainFiles).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        returnMap = {}
        m = len(lineSplitBlock) - 1
        for i in range(2, m, 2):  # 在每个文档向量中提取(word, tfidf)存入字典
            returnMap[lineSplitBlock[i]] = lineSplitBlock[i + 1]

        temp_key = lineSplitBlock[0]
        if temp_key != lastKind:
            if lastKind != "":
                # print(lastKind)
                trainDocWordMap[lastKind] = docList
                docList = []
        docList.append(returnMap)
        lastKind = lineSplitBlock[0]
    # print(lastKind)
    trainDocWordMap[lastKind] = docList

    returnMap={}
    i=0
    for item in trainDocWordMap:
        if item in gsimResult:
            for wordList in trainDocWordMap[item]:
                sim=computeSim(testWordMap,wordList)
                i=i+1
                returnMap[item+"_"+str(i)]=sim
    sortedSimMap = sorted(returnMap.items(), key=itemgetter(1), reverse=True)  # <类目_文件名,距离> 按照value排序

    size=len(sortedSimMap)
    print("size:"+str(size))
    # k = 7
    cateSimMap = {}  # <类，距离和>
    for i in range(k):
        cate = sortedSimMap[i][0].split('_')[0]
        # print("cate: "+cate)
        cateSimMap[cate] = cateSimMap.get(cate, 0) + sortedSimMap[i][1]
        # print("cate"+ cate+"  cateSimMap[cate]"+str(cateSimMap[cate]))

    sortedCateSimMap = sorted(cateSimMap.items(), key=itemgetter(1), reverse=True)
    print("sortedCateSimMap[0][0] "+sortedCateSimMap[0][0])
    return sortedCateSimMap[0][0]


#########################################################
## @param cate_Doc 测试集<类别_文档>
## @param testDic 测试集{{word, TFIDF}}
## @param trainMap 训练集{类，[{} {}]}
## @return sortedCateSimMap[0][0] 返回与测试文档向量距离和最小的类
#########################################################
def getGsim(testDic, trainMap):
    simMap = {}  # <类目_文件名,距离> 后面需要将该HashMap按照value排序
    for kind in trainMap:
        sum=0.0
        for item in trainMap[kind]:
            similarity = computeSim(testDic, item)  # 调用computeSim()
            sum=sum+similarity
        simMap[kind] = sum/(1.0*len(trainMap[kind]))

    sortedSimMap = sorted(simMap.items(), key=itemgetter(1), reverse=True)  # <类目_文件名,距离> 按照value排序
    return sortedSimMap

def find_k():
    n=1
    gsimDiff=0.01
    PValue = 'docVector/PValue1_k'
    PValueWriter = open(PValue, 'w')
    for k in range(4,10,1):
        result=doProcess(n, k,gsimDiff)
        PValueWriter.write('%d %f\n' % (k, result))
    PValueWriter.close()

def find_gim():
    n=1
    k = 8
    PValue = 'docVector/PValue1_Gim'
    PValueWriter = open(PValue, 'w')
    for gsimDiff in range(0, 40, 5):
        result = doProcess(n, k, float(gsimDiff/2000))
        PValueWriter.write('%f %f\n' % (float(gsimDiff/2000), result))
    PValueWriter.close()

def find_m():
    # from My_OptimizedData import getOpti_m
    # for m in range(5, 30, 5):
    #     getOpti_m(0, m)
    k = 7
    gsimDiff = 0.005
    PValue = 'docVector/PValue1_m'
    PValueWriter = open(PValue, 'w')
    for m in range(1,8,1):
        result = doProcess_m(1, k, gsimDiff,m)
        PValueWriter.write('%d %f\n' % (m*5, result))
    PValueWriter.close()

if __name__ == '__main__':

    # find_k()
    # find_gim()
    # find_m()

    # start = time.clock()

    # from My_Middle import doMiddle
    # for i in range(10):
    #     doMiddle(i)
    #
    # from My_deleteBound import doDelete
    # for i in range(10):
    #     doDelete(i)
    #
    # from My_OptimizedData import getOpti
    # for i in range(10):
    #     getOpti(i)
    #
    n=10
    k=8
    gsimDiff=0.005
    doProcess(n,k,gsimDiff)
    # end = time.clock()
    # print('runtime: ' + str(end - start))



