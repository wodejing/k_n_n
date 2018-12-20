# -*- coding: utf-8 -*-
from numpy import *
from os import listdir,mkdir,path
import re
from nltk.corpus import stopwords
import nltk
import operator
nltk.download('stopwords')

#去除停留词，选取出现次数大于4的单词

##############################################################
## 1. 创建新文件夹，存放预处理后的文本数据，一行一个单词
##############################################################
def createFiles():
    srcFilesList = listdir('originSample')  # 文本存放的目录，本次实验选取的是其中的七个文本
    print(srcFilesList)
    for i in range(len(srcFilesList)):
        if i<0: continue
        dataFilesDir = 'originSample/' + srcFilesList[i]  # 选取的7个文件夹每个的路径
        print(dataFilesDir)
        dataFilesList = listdir(dataFilesDir)        #dataFilesDir存储的是一类文本，dataFilesList得到该分类下的文档
        targetDir = 'processedSample_includeNotSpecial/' + srcFilesList[i] # 20个新文件夹每个的路径
        if path.exists(targetDir)==False:
            mkdir(targetDir)
        else:
            print('%s exists' % targetDir)
        for j in range(len(dataFilesList)):
            createProcessFile(srcFilesList[i],dataFilesList[j]) # 调用createProcessFile()在新文档中处理文本
            print('%s %s' % (srcFilesList[i],dataFilesList[j]))
##############################################################
## 2. 建立目标文件夹，生成目标文件
## @param srcFilesName 某组新闻文件夹的文件名，比如alt.atheism
## @param dataFilesName 文件夹下某个数据文件的文件名
## @param dataList 数据文件按行读取后的字符串列表
##############################################################
def createProcessFile(srcFilesName,dataFilesName):
    srcFile = 'originSample/' + srcFilesName + '/' + dataFilesName
    targetFile= 'processedSample_includeNotSpecial/' + srcFilesName\
                + '/' + dataFilesName
    fw = open(targetFile,'w')
    dataList = open(srcFile,encoding='gb18030', errors='ignore').readlines()
    for line in dataList:
        resLine = lineProcess(line) # 调用lineProcess()处理每行文本
        for word in resLine:
            fw.write('%s\n' % word) #一行一个单词
    fw.close()
##############################################################
##3. 对每行字符串进行处理，主要是去除非字母字符，转换大写为小写，去除停用词
## @param line 待处理的一行字符串
## @return words 按非字母分隔后的单词所组成的列表
##############################################################
def lineProcess(line):
    stopwords = nltk.corpus.stopwords.words('english') #去停用词
    porter = nltk.PorterStemmer()  #词干分析
    splitter = re.compile('[^a-zA-Z]')  #去除非字母字符，形成分隔
    words = [porter.stem(word.lower()) for word in splitter.split(str(line))\
             if len(word)>0 and\
             word.lower() not in stopwords]
    return words


########################################################
## 统计每个词的总的出现次数
## @param strDir
## @param wordMap
## return newWordMap 返回字典，<key, value>结构，按key排序，value都大于4，即都是出现次数大于4的词
#########################################################
def countWords():
    wordMap = {}
    newWordMap = {}
    fileDir = 'processedSample_includeNotSpecial'
    sampleFilesList = listdir(fileDir)
    for i in range(len(sampleFilesList)):
        sampleFilesDir = fileDir + '/' + sampleFilesList[i]
        sampleList = listdir(sampleFilesDir)
        for j in range(len(sampleList)):
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            for line in open(sampleDir).readlines():
                word = line.strip('\n')
                wordMap[word] = wordMap.get(word, 0.0) + 1.0
    #只返回出现次数大于4的单词
    for key, value in wordMap.items():
        if value > 3:
            newWordMap[key] = value
    sortedNewWordMap = sorted(newWordMap.items())
    print('wordMap size : %d' % len(wordMap))
    print('newWordMap size : %d' % len(sortedNewWordMap))
    return sortedNewWordMap

#####################################################
##特征词选取,选取总出现次数大于4
####################################################
def filterSpecialWords():
    fileDir = 'processedSample_includeNotSpecial'
    wordMapDict = {}
    sortedWordMap = countWords()
    for i in range(len(sortedWordMap)):
        wordMapDict[sortedWordMap[i][0]]=sortedWordMap[i][0]
    sampleDir = listdir(fileDir)
    for i in range(len(sampleDir)):
        targetDir = 'processedSampleOnlySpecial_2' + '/' + sampleDir[i]
        srcDir = 'processedSample_includeNotSpecial' + '/' + sampleDir[i]
        if path.exists(targetDir) == False:
            mkdir(targetDir)
        sample = listdir(srcDir)
        for j in range(len(sample)):
            targetSampleFile = targetDir + '/' + sample[j]
            fr=open(targetSampleFile,'w')
            srcSampleFile = srcDir + '/' + sample[j]
            for line in open(srcSampleFile).readlines():
                word = line.strip('\n')
                if word in wordMapDict.keys():
                    fr.write('%s\n' % word)
            fr.close()
if __name__ == '__main__':
    createFiles()
    filterSpecialWords()