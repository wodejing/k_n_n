import matplotlib.pyplot as plt
from pylab import *  # 支持中文
if __name__ == '__main__':
    result=[]
    PValue = 'docVector/PValue1_m'
    for line in open(PValue).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        result.append(float(lineSplitBlock[1]))

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    names = ['5', '10', '15', '20', '25','30','35']
    x = range(len(names))
    plt.ylim(0.76, 0.86,0.01)
    print(str(result))
    plt.plot(x, result, marker='o', mec='r', mfc='w', label=u'改进Knn')
    plt.legend()  # 让图例生效
    plt.xticks(x, names, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"聚类数")  # X轴标签
    plt.ylabel(u"准确率")  # Y轴标签
    # plt.title("A simple plot")  # 标题
    plt.show()