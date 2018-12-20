import matplotlib.pyplot as plt
from pylab import *  # 支持中文
if __name__ == '__main__':
    result1=[]
    result2 = []
    PValue1 = 'docVector/PValue'
    PValue2 = 'docVector/PValue1_k'
    for line in open(PValue1).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        result1.append(float(lineSplitBlock[1]))

    for line in open(PValue2).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        result2.append(float(lineSplitBlock[1]))

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    names = ['4', '5', '6', '7', '8','9']
    x = range(len(names))
    plt.ylim(0.76, 0.86,0.01)
    print(str(result1))
    plt.plot(x, result1, marker='o', mec='r', mfc='w', label=u'Knn')
    plt.plot(x, result2, marker='*', ms=10, label=u'改进knn')
    plt.legend()  # 让图例生效
    plt.xticks(x, names, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"k值")  # X轴标签
    plt.ylabel(u"准确率")  # Y轴标签
    # plt.title("A simple plot")  # 标题
    plt.show()