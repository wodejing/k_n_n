import time
from os import listdir
from math import log
from numpy import *


###################################################
## 计算所有单词的IDF值
#返回值为{word,IDF; word,IDF}
###################################################
def computeIDF():
    N=7718 #选取文本的总数
    fileDir = 'processedSampleOnlySpecial_2'
    wordDocMap = {}  # <word, count> ，存入单词word出现的文档的总数
    IDFPerWordMap = {}  # <word, IDF值> ，存入单词word的IDF

    sum=0
    cateList = listdir(fileDir)
    for i in range(len(cateList)):
        sampleDir = fileDir + '/' + cateList[i]
        sampleList = listdir(sampleDir)
        for j in range(len(sampleList)):
            wordMap = {}
            sample = sampleDir + '/' + sampleList[j]
            for line in open(sample).readlines():
                word = line.strip('\n')
                if wordMap.get(word,0)==0:
                    if word == "subject":
                        sum+=1
                    wordDocMap[word] = wordDocMap.get(word, 0.0) + 1.0  # 单词word出现过的文档数
                    wordMap[word] =1
                else:
                    continue

        print('just finished %d round ' % i)
    print("sum:"+str(sum))
    max=0
    count=0
    tempword=""
    for word in wordDocMap:
        # countDoc = len(wordDocMap[word])  # 统计set中的文档个数
        if wordDocMap[word]>max:
            max=wordDocMap[word]
            tempword=word
        count+=wordDocMap[word]

        IDF = log((N) / (wordDocMap[word]+1)) / (1.0*log(10))
        IDFPerWordMap[word] = IDF

    print("max:"+str(max))
    print("count:" + str(count))
    print("word:"+tempword)
    return IDFPerWordMap


###################################################
## 将IDF值写入文件保存
###################################################
if __name__ == '__main__':
    # start = time.clock()
    IDFPerWordMap = computeIDF()
    # end = time.clock()
    # print
    # 'runtime: ' + str(end - start)
    fw = open('docVector/IDFPerWord', 'w')
    for word, IDF in IDFPerWordMap.items():
        fw.write('%s %.6f\n' % (word, IDF))
    fw.close()