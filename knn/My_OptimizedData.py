# -*- coding: UTF-8 -*-
from numpy import *

from My_Dis import computeSim

# 载入数据测试数据集
# { 类别，[{word,TF*IDF},{}...] }
def loadDataMap(inFile):
    returnDocMap = {}
    docList=[]
    lastKind=""
    for line in open(inFile).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        returnMap = {}
        m = len(lineSplitBlock) - 1
        for i in range(2, m, 2):  # 在每个文档向量中提取(word, tfidf)存入字典
            returnMap[lineSplitBlock[i]] = lineSplitBlock[i + 1]

        temp_key = lineSplitBlock[0]
        if temp_key!=lastKind:
            if lastKind != "":
                # print(lastKind)
                returnDocMap[lastKind] = docList
                docList = []
        docList.append(returnMap)
        lastKind = temp_key
    returnDocMap[lastKind] = docList
    return returnDocMap  #{ 类别，[{word,TF*IDF},{}...] }


# 初始化k个质心，随机获取
def initCentroids(dataSet, k):
    # print(dataSet)
    # list=np.random.sample(dataSet, k)
    import random
    list=random.sample(dataSet, k)
    return list  # 从dataSet中随机获取k个数据项返回



# 对每个属于dataSet的item，计算item与centroidList中k个质心的余弦距离，找出距离最大的，
# 并将item加入相应的簇类中dataList=[{},{}...]
#return {簇类，[{} {} {}]}
def maxDistance(dataList, centroidList):
    clusterDict = {}  # 用dict来保存簇类结果
    for item in dataList:
        # print(item)
        flag = 0  # 簇分类标记，记录与相应簇距离最近的那个簇
        maxSim = 0  # 初始化为最大值

        for i in range(len(centroidList)):
            simDistance = computeSim(item, centroidList[i])  # 计算相应的余弦距离
            # from My_Dis import computeOs
            # distance =computeOs(item,centroidList[i])
            if simDistance > maxSim:
                maxSim = simDistance
                flag = i  # 循环结束时，flag保存的是与当前item距离最近的那个簇标记
        # print("flag:"+str(flag))
        if flag not in clusterDict.keys():  # 簇标记不存在，进行初始化
            clusterDict[flag] = []
        # print flag, item
        clusterDict[flag].append(item)  # 加入相应的类别中
    # print("maxDistance:"+str(len(clusterDict)))

    return clusterDict  # 返回新的聚类结果

# 得到k个质心
#clusterDict {簇，[{key:word;key:word } {}]}
#return [{} {}]
def getCentroids(clusterDict):
    centroidList = list()
    for key in clusterDict.keys():
        sumDic = {}
        length=len(clusterDict[key])
        for temp in clusterDict[key]:
            for key in temp.keys():
                if key in sumDic.keys():
                    sumDic[key]=float(sumDic[key])+float(temp[key])
                else:
                    sumDic[key]=float(temp[key])
        for key in sumDic.keys():
            sumDic[key]=float(sumDic[key])/(1.0*length)
        centroidList.append(sumDic)


    return centroidList

# 计算簇集合间的均方误差
# 将簇类中各个向量与质心的距离进行累加求和
#clusterDict：{簇，[{} {}]}
#centroidList:质心[{} {} {}]
# def getVar(clusterDict, centroidList):
#     sum = 0.0
#     for key in clusterDict.keys():
#         distance = 0.0
#         for item in clusterDict[key]:
#             from My_Dis import computeOs
#             distance += computeOs(centroidList[key], item)
#         sum += distance
#     return sum



def getOpti(i):

    inFile = "docVector/deleteBound/deleteBoundTFIDFMapTrain"+str(i)  # 数据集文件
    dataKindMap = loadDataMap(inFile)  # 载入数据集{ 类别，[ { {word,TF*IDF},{word,TF*IDF} },... ] }
    optiBoundFile = 'docVector/optiBound/optiBoundTFIDFMapTrain'+str(i)
    optiBoundWriter = open(optiBoundFile, 'w')

    for kind in dataKindMap:
        centroidList = initCentroids(dataKindMap[kind], 20)  # 初始化质心，设置k=4  [{} {} {}]
        print("centroidList1:"+str(len(centroidList)))
        clusterDict = maxDistance(dataKindMap[kind], centroidList)  # 第一次聚类迭代#return {簇类，[{} {} {}]}
        print("centroidList2:" + str(len(centroidList)))
        print("clusterDict:" + str(len(clusterDict)))


        print('***** 第1次迭代 *****')
        print('簇类')
        # for key in clusterDict.keys():
        #     print(key, ' --> ', clusterDict[key])
        # print('k个均值向量: ', centroidList)

        k = 2
        while k <= 20:  # 当迭代次数大于20时，迭代结束
            centroidList = getCentroids(clusterDict)  # 获得新的质心
            # print("centroidList:"+str(k) + str(len(centroidList)))
            clusterDict = maxDistance(dataKindMap[kind], centroidList)  # 新的聚类结果
            print('***** 第%d次迭代 *****' % k)
            print
            print('簇类')
            print
            k += 1
        flag=0
        print("centroidList:" + str(len(centroidList)))
        for doc in centroidList:
            optiBoundWriter.write('%s %d ' % (kind,flag))
            for word in doc:
                optiBoundWriter.write('%s %s ' % (word, doc[word]))
            optiBoundWriter.write('\n')
            flag+=1

def getOpti_m(i,m):

    inFile = "docVector/deleteBound/deleteBoundTFIDFMapTrain"+str(i)  # 数据集文件
    dataKindMap = loadDataMap(inFile)  # 载入数据集{ 类别，[ { {word,TF*IDF},{word,TF*IDF} },... ] }
    optiBoundFile = 'docVector/optiBoundm/optiBoundTFIDFMapTrain'+str(int(m/5))
    optiBoundWriter = open(optiBoundFile, 'w')

    for kind in dataKindMap:
        centroidList = initCentroids(dataKindMap[kind], m)
        clusterDict = maxDistance(dataKindMap[kind], centroidList)  # 第一次聚类迭代#return {簇类，[{} {} {}]}

        print('***** 第1次迭代 *****')
        print('簇类')
        # for key in clusterDict.keys():
        #     print(key, ' --> ', clusterDict[key])
        # print('k个均值向量: ', centroidList)

        k = 2
        while k <= 20:  # 当迭代次数大于20时，迭代结束
            centroidList = getCentroids(clusterDict)  # 获得新的质心
            clusterDict = maxDistance(dataKindMap[kind], centroidList)  # 新的聚类结果

            print('***** 第%d次迭代 *****' % k)
            print
            print('簇类')
            # for key in clusterDict.keys():
            #     print(key, ' --> ', clusterDict[key])
            # print('k个均值向量: ', centroidList)
            print
            k += 1

        flag=0
        print("centroidList1:"+str(len(centroidList)))
        for doc in centroidList:
            optiBoundWriter.write('%s %d ' % (kind,flag))
            for word in doc:
                optiBoundWriter.write('%s %s ' % (word, doc[word]))
            optiBoundWriter.write('\n')
            flag+=1

if __name__ == '__main__':
    for i in range(10):
        getOpti(i)


    # for m in range(5, 40, 5):
    #     getOpti_m(0, m)

