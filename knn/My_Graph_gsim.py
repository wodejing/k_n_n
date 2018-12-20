import matplotlib.pyplot as plt
from pylab import *  # 支持中文
if __name__ == '__main__':
    result=[]
    PValue = 'docVector/PValue1_Gim'
    for line in open(PValue).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        print(str(len(lineSplitBlock)))
        result.append(float(lineSplitBlock[1]))

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    names = ['0.0000', '0.0025', '0.0050', '0.0075', '0.0100','0.0125','0.015','0.0175']
    x = range(len(names))
    plt.ylim(0.76, 0.86,0.01)
    print(str(result))
    plt.plot(x, result, marker='o', mec='r', mfc='w', label=u'改进Knn')
    plt.legend()  # 让图例生效
    plt.xticks(x, names, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"α值")  # X轴标签
    plt.ylabel(u"准确率")  # Y轴标签
    # plt.title("A simple plot")  # 标题
    plt.show()