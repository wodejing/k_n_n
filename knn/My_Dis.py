from numpy import *
import numpy as np
#################################################
## @param testDic 一维测试文档向量<<word, tfidf>>
## @param trainDic 一维训练文档向量<<word, tfidf
## @return 返回余弦相似度
# def computeSim(testDic, trainDic):
#     testList = []  # 测试向量与训练向量共有的词在测试向量中的tfidf值
#     trainList = []  # # 测试向量与训练向量共有的词在训练向量中的tfidf值
#
#     for word, weight in testDic.items():
#         if word in trainDic:
#             testList.append(float(weight))  # float()将字符型数据转换成数值型数据，参与下面运算
#             trainList.append(float(trainDic[word]))
#
#     testVect = mat(testList)  # 列表转矩阵，便于下面向量相乘运算和使用Numpy模块的范式函数计算
#     trainVect = mat(trainList)
#     num = float(testVect * trainVect.T)
#     denom = linalg.norm(testVect) * linalg.norm(trainVect)
#     # print 'denom:%f' % denom
#     return float(num) / (1.0 + float(denom))

def computeSim(testDic, trainDic):
    testList = []  # 测试向量与训练向量共有的词在测试向量中的tfidf值
    trainList = []  # # 测试向量与训练向量共有的词在训练向量中的tfidf值

    for word, weight in testDic.items():
        if word in trainDic:
            testList.append(float(weight))  # float()将字符型数据转换成数值型数据，参与下面运算
            trainList.append(float(trainDic[word]))
        else:
            testList.append(float(weight))  # float()将字符型数据转换成数值型数据，参与下面运算
            trainList.append(0.0)
    for word,weight in trainDic.items():
        if word not in testDic:
            testList.append(0.0)  # float()将字符型数据转换成数值型数据，参与下面运算
            trainList.append(float(trainDic[word]))

    # testVect = mat(testList)  # 列表转矩阵，便于下面向量相乘运算和使用Numpy模块的范式函数计算
    # trainVect = mat(trainList)
    # num = float(testVect * trainVect.T)
    # denom = linalg.norm(testVect) * linalg.norm(trainVect)
    # # print 'denom:%f' % denom
    # return float(num) / (float(denom))
    myx = np.array(testList)
    myy = np.array(trainList)
    cos1 = np.sum(myx * myy)
    cos21 = np.sqrt(sum(myx * myx))
    cos22 = np.sqrt(sum(myy * myy))
    return cos1 / float(cos21 * cos22)


def computeOs(testDic, trainDic):
    testList = []  # 测试向量与训练向量共有的词在测试向量中的tfidf值
    trainList = []  # # 测试向量与训练向量共有的词在训练向量中的tfidf值

    for word, weight in testDic.items():
        if word in trainDic:
            testList.append(float(weight))  # float()将字符型数据转换成数值型数据，参与下面运算
            trainList.append(float(trainDic[word]))

    testVect = mat(testList)  # 列表转矩阵，便于下面向量相乘运算和使用Numpy模块的范式函数计算
    trainVect = mat(trainList)

    dis=sqrt(sum(power(testVect - trainVect, 2)))

    # print 'denom:%f' % denom
    return dis