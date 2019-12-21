import numpy as np
from numpy import *
import numpy as np
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


def load_data(path):
    data = []
    with open(path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        i = 0
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            data.append(row)
            # i += 1
            # if i == 10000:
            #     break

    data = [[int(x) for x in row] for row in data]  # 将数据从string形式转换为float形式

    data = np.array(data)  # 将list数组转化成array数组便于查看数据结构
    # data = data.transpose()
    print(data.shape)  # 利用.shape查看结构。

    return data

def graph(H, r):
    x = []
    i = 0
    while i< 24.25:
        x.append(i)
        i += 0.25
    i = 0
    while i < r:
        y = []
        j = 0
        while j < 97:
            y.append(int(H[i, j]))
            j += 1
        plt.plot(x, y)
        plt.grid(True)  ##增加格点
        i += 1

    plt.show()



if __name__ == '__main__':
    path = "data_csv/20111120_REAL.csv"
    V = load_data(path)

    model = NMF(n_components=2, init='random', random_state=0)

    model.fit(V)

    print(model.components_)


    print(model.reconstruction_err_)
    print(model.n_iter_)
