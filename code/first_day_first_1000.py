from numpy import *
import numpy as np
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from geopy.distance import geodesic
import random
from math import *
from matplotlib import style
from sklearn.cluster import KMeans
import pandas as pd


def load_data():
    data = []
    for k in range(0, 10):
        if k == 6:
            continue
        with open("data_csv/2011112" + str(k) + "_REAL.csv") as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            i = 0
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                data.append(row)
                # i += 1
                # if i == 1000:
                #     break

    data = [[int(x) for x in row] for row in data]  # 将数据从string形式转换为float形式

    data = np.array(data)  # 将list数组转化成array数组便于查看数据结构
    # data = data.transpose()
    print(data.shape)  # 利用.shape查看结构。

    np.save("total_9_data.npy", data)

    return data


def initiate(data, r):
    row_num = data.shape[0]
    col_num = data.shape[1]
    W = mat(np.random.rand(row_num, r))
    H = mat(np.random.rand(r, col_num))
    # W = mat(ones((row_num, r)))
    # H = mat(ones((r, col_num)))
    return W, H


def iterate(V, W, H, r):
    row_num = V.shape[0]
    col_num = V.shape[1]
    first_numerator = W.transpose() * V
    first_denominator = W.transpose() * W * H
    for i in range(0, r):
        for j in range(0, col_num):
            H[i, j] = H[i, j] * first_numerator[i, j] / first_denominator[i, j]

    second_numerator = V * H.transpose()
    second_denominator = W * H * H.transpose()
    for i in range(0, row_num):
        for j in range(0, r):
            if second_denominator[i, j] != 0:
                W[i, j] = W[i, j] * second_numerator[i, j] / second_denominator[i, j]
            else:
                W[i, j] = W[i, j] * second_numerator[i, j] / 0.000000001
    return W, H


def judge(V, W, H, previous):
    Error = V - W * H
    MSE = 0
    row_num = V.shape[0]
    col_num = V.shape[1]
    for i in range(0, row_num):
        for j in range(0, col_num):
            MSE += pow(Error[i, j], 2)
    RMSE = sqrt(MSE)
    rate = (RMSE - previous) / previous
    print(RMSE, previous, rate)

    if rate < 0.001 and -0.001 < rate:
        return True, RMSE
    return False, RMSE


def read():
    W = np.load("20111128_W.npy")
    H = np.load("20111128_H.npy")
    return W, H


def print_file(W, H):
    np.save("20111128_W.npy", W)
    np.save("20111128_H.npy", H)


def graph(H, r):
    x = []
    i = 0
    while i < 24:
        x.append(i)
        i += 0.25
    i = 0
    # 循环写法
    # while i < r:
    #     y = []
    #     j = 0
    #     while j < 96:
    #         y.append(float(H[i, j]))
    #         j += 1
    #     plt.plot(x, y)
    #     plt.grid(True)  ##增加格点
    #     i += 1
    y1 = [float(H[0, j]) for j in range(0, 96)]
    y2 = [float(H[1, j]) for j in range(0, 96)]
    y3 = [float(H[2, j]) for j in range(0, 96)]
    plt.plot
    plt.plot(x, y1, color="blue", label="r1")
    plt.plot(x, y2, color="red", label="r2")
    plt.plot(x, y3, color="green", label="r3")
    plt.grid(True)

    plt.show()


def center_point(num, luduan, longi_lati):
    point_temp_1 = luduan[num][0]
    point_temp_2 = luduan[num][1]

    flag1 = 0
    flag2 = 0
    for i in longi_lati:
        if i[0] == point_temp_1:
            lon1 = i[1]
            lat1 = i[2]
            flag1 = 1
        elif i[0] == point_temp_2:
            lon2 = i[1]
            lat2 = i[2]
            flag2 = 1
        elif flag1 == 1 and flag2 == 1:
            break
    if flag1 == 0 or flag2 == 0:
        return 0, 0

    lon = (lon1 + lon2) * math.pi / 360
    lat = (lat1 + lat2) * math.pi / 360
    return lon, lat


def distance_physics(num1, num2, longi_lati, luduan):
    (lon1, lat1) = center_point(num1, luduan, longi_lati)
    (lon2, lat2) = center_point(num2, luduan, longi_lati)

    dis_phy = geodesic((lon1, lat1), (lon2, lat2)).km

    return dis_phy


def distance_logistics(num1, num2, W):
    # 对于9个的大W
    dis_log = 0
    for i in range(0, 9):
        temp = (W[num1][0] - W[num2][0]) ** 2 + (W[num1][1] - W[num2][1]) ** 2 + (W[num1][2] - W[num2][2]) ** 2
        dis_log += temp
        num1 += 1000
        num2 += 1000
    dis_log = sqrt(dis_log)
    return dis_log


def distance(num1, num2, longi_lati, luduan, W, alpha=0.1, tau=1):
    dis_phy = distance_physics(num1, num2, longi_lati, luduan)
    dis_log = distance_logistics(num1, num2, W)
    dis = alpha * dis_phy + (1 - alpha) * dis_log / tau

    return dis


def k_center_package(k, W):
    data = []

    # 获取V（1000*96）
    # with open("data_csv/2011112" + "0" + "_REAL.csv") as csvfile:
    #     csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    #     i = 0
    #     for row in csv_reader:  # 将csv 文件中的数据保存到data中
    #         data.append(row)
    #         i += 1
    #         if i == 1000:
    #             break
    #
    # data = [[int(x) for x in row] for row in data]  # 将数据从string形式转换为float形式
    #
    # V = np.array(data)  # 将list数组转化成array数组便于查看数据结构
    # print(V.shape)  # 利用.shape查看结构。

    #非负矩阵分解
    # i = 0
    # previous = 0.00001
    # (W, H) = initiate(V, r)
    # while True:
    #     (W, H) = iterate(V, W, H, r)
    #     (bool, previous) = judge(V, W, H, previous)
    #     i += 1
    #     print(i)
    #     if i == 1000 or bool:
    #         break
    # print("W___________")
    # print(W)
    # print("H____________")
    # print(H)
    # np.save("first_day_1000_W.npy", W)
    # np.save("first_day_1000_H.npy", H)

    # W = np.load("mean_W.npy")
    # H = np.load("first_day_1000_H.npy")

    model = KMeans(n_clusters=k, max_iter=1000)  # 分为k类，并发数4
    model.fit(W)  # 开始聚类

    # 简单打印结果
    r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    print(r)  # 详细输出原始数据及其类别
    p = pd.Series(model.labels_)
    result_y = []
    for i in range(0, 127049):
        result_y.append(p[i])

    cluster = [[] for i in range(0, k)]
    for j in range(0, len(result_y)):
        cluster[result_y[j]].append(j)

    cluster = array(cluster)
    result_y = array(result_y)
    # print(cluster)cl


    return cluster, result_y






def k_center(W, k):
    row_num = 1000
    center = []
    cluster = []
    for i in range(0, k):
        n = random.randint(0, row_num)
        center.append(n)
        cluster.append([n])

    times = 0
    while True:
        (center, old_cluster) = k_center_interate(W, k, center, cluster)
        if center == 0 or times == 2:
            break
        cluster = []
        for i in range(0, k):
            cluster.append([center[i]])
        times += 1
        print(times)
        print(cluster)

    result_y = [0 for i in W]
    for i in range(0, k):
        for j in range(0, len(old_cluster[i])):
            result_y[old_cluster[i][j]] = i

    result = np.array(result_y)
    np.save("k_center_cluster.npy")
    np.save("k_center_result_y.npy")

    return cluster, result_y


def k_center_interate(W, k, center, cluster):
    row_num = 1000
    new_dist = [0 for i in range(0, 1000)]
    longi_lati = np.load("longi_lati.npy")
    luduan = np.load("totoal_9_luduan.npy")

    for i in range(0, row_num):
        print("第", i, "次迭代")

        d = []
        if i not in center:
            for j in range(0, k):
                d.append(distance(i, center[j], longi_lati, luduan, W))
                print(j)

            num = d.index(min(d))
            # print(num)
            for g in cluster[num]:
                new_dist[g] += distance(i, g, longi_lati, luduan, W)
                new_dist[i] += distance(i, g, longi_lati, luduan, W)
            cluster[num].append(i)
            # print(cluster)
        else:
            continue

    new_center = []
    for i in range(0, k):
        min_dis = float("inf")
        pos = -1
        # print("第", i, "个簇")
        for j in cluster[i]:
            if new_dist[j] < min_dis:
                # print(new_dist[j])
                min_dis = new_dist[j]
                pos = j
        new_center.append(pos)

    for i in center:
        if i not in new_center:
            return new_center, cluster

    return 0, cluster


def paint_k_center(ypred, W):
    y_class = array(ypred)
    ax = plt.subplot(111, projection='3d')
    x_array = np.array(W, dtype=float)
    y_array = np.array(y_class, dtype=int)
    a = x_array[np.where(y_array == 0)]
    b = x_array[np.where(y_array == 1)]
    c = x_array[np.where(y_array == 2)]
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], c='r', label='a')
    ax.scatter(b[:, 0], b[:, 1], b[:, 2], c='b', label='b')
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], c='g', label='c')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.legend(loc='upper left')
    ax.view_init(35, 300)
    plt.show()


def history_score(V, test, day):
    row_num = test.shape[0]
    col_num = test.shape[1]

    set = [row_num * i for i in range(0, 9)]
    P_history = []

    for i in range(0, row_num):
        P_history.append([])



        for t in range(0, col_num):
            v_it = [V[i + k][t] for k in set]
            # 算标准差
            sum_sigma = 0
            for j in v_it:
                sum_sigma += (j - mean(v_it)) ** 2
            sigma = sum_sigma / len(v_it)
            h1 = 5 * sigma / 9
            if  h1 == 0.0:
                h1 = 0.0000001
            sum_gaussian = 0
            for g in range(0, 9):
                x = (test[i][t] - V[i + g * row_num][t]) / h1
                gaussian = np.exp(-(x - 0) ** 2 / (2 * 1 ** 2)) / (math.sqrt(2 * math.pi) * 1)
                sum_gaussian += gaussian

            p = sum_gaussian / (9 * h1)
            P_history[i].append(p)
        if i % 100 == 0:
            print("历史", i)

    P_history = np.array(P_history)
    np.save("P_history_total_2011112" + str(day) + ".npy", P_history)

    return P_history


def neighbor_score(test, cluster, day):
    row_num = test.shape[0]
    col_num = test.shape[1]
    P_neighbor = np.array(zeros((row_num, col_num)))
    # M = np.array(zeros((row_num, col_num)))

    # for i in range(0, row_num):
    #     for j in range(0, col_num):
    #         sum = 0
    #         for k in range(0, 9):
    #             sum += V[i + k * row_num][j]
    #         mean = sum / 9.0
    #         M[i][j] = mean

    for i in range(0, len(cluster)):
        print("簇", i)
        for j in range(0, len(cluster[i])):
            print("簇内第", j, "个")
            for t in range(0, col_num):
                v_it = [test[cluster[i][m]][t] for m in range(0, len(cluster[i]))]
                sum = 0
                for q in v_it:
                    sum += q
                mean = sum / len(v_it)

                # 算标准差
                sum_sigma = 0
                for f in cluster[i]:
                    sum_sigma += (test[f][t] - mean) ** 2
                sigma = sum_sigma / len(v_it)
                h2 = 5 * sigma / len(cluster[i])

                sum_gaussian = 0

                for g in range(0, len(cluster[i])):
                    if h2 == 0.0:
                        h2 = 0.0000000001
                    x = (test[cluster[i][j]][t] - test[cluster[i][g]][t]) / h2
                    gaussian = np.exp(-(x - 0) ** 2 / (2 * 1 ** 2)) / (math.sqrt(2 * math.pi) * 1)
                    sum_gaussian += gaussian

                p = sum_gaussian / (len(cluster[i]) * h2)
                P_neighbor[cluster[i][j]][t] = p

    P_neighbor = np.array(P_neighbor)
    np.save("P_neighbor_total_2011112" + str(day) + ".npy", P_neighbor)

    return P_neighbor


def train_data(V, train, cluster, day, beta=0.5):
    # for i in range(0, 10):
    #     if i == 6:
    #         continue
    #     train = np.load("data_2011112" + str(i) + ".npy")



    P_history = history_score(V, train, day)
    P_neighbor = neighbor_score(train, cluster, day)

    # P_history = np.load("P_history_total_20111120.npy")
    # P_neighbor = np.load("P_neighbor_total_20111120.npy")


    score = beta * P_history + (1 - beta) * P_neighbor
    np.save("score_2011112" + str(day) + ".npy", score)

    return score


def find_min(score):
    min_list = []
    for i in range(0, 100):
        min_list.append([])
        (row, col) = np.divmod(np.argmin(score), np.shape(score)[1])
        s = score[row][col]
        min_list[i].append(row)
        min_list[i].append(col)
        min_list[i].append(s)
        score[row][col] = float("inf")

    return min_list


def get_first_1000(day):
    data = []

    with open("data_csv/2011112" + str(day) + "_REAL.csv") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        i = 0
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            data.append(row)
            i += 1
            if i == 1000:
                break

    data = [[int(x) for x in row] for row in data]  # 将数据从string形式转换为float形式

    data = np.array(data)  # 将list数组转化成array数组便于查看数据结构
    print(data.shape)  # 利用.shape查看结构。
    # np.save("data_2011112"+str(i) + "_1000.npy", data)

    return data


def get_day(day):
    data = []

    with open("data_csv/2011112" + str(day) + "_REAL.csv") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        i = 0
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            data.append(row)

    data = [[int(x) for x in row] for row in data]  # 将数据从string形式转换为float形式

    data = np.array(data)  # 将list数组转化成array数组便于查看数据结构
    print(data.shape)  # 利用.shape查看结构。
    # np.save("data_2011112"+str(i) + "_1000.npy", data)

    return data


def mean_V():
    data = []
    for k in range(0, 10):
        print(k)
        if k == 6:
            continue
        with open("data_csv/2011112" + str(k) + "_REAL.csv") as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            i = 0
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                data.append(row)

    data = [[int(x) for x in row] for row in data]  # 将数据从string形式转换为float形式


    V = np.zeros((127049, 96))

    for i in range(0, 127049):
        print(i)
        for j in range(0, 96):
            sum = 0
            for k in range(0, 9):
                sum += data[i + k * 127049][j]
            mean = sum / 9.0
            V[i][j] = mean

    V = np.array(V)  # 将list数组转化成array数组便于查看数据结构
    print(V.shape)  # 利用.shape查看结构。

    np.save("mean_V.npy", V)

    return V


def visualize(min_list):
    luduan = np.load("luduan.npy")
    longi_lati = np.load("longi_lati.npy")
    for i in range(0, len(min_list)):

        (lon, lat) = center_point(min_list[i][0], luduan, longi_lati)
        lon = lon * 180 / math.pi
        lat = lat * 180 / math.pi

        min_list[i].append(lon)
        min_list[i].append(lat)

    return min_list


def real_min_list(score):
    min_list = []
    luduan = np.load("luduan.npy")
    longi_lati = np.load("longi_lati.npy")
    i = 0
    while True:
        if i == 0:
            (row, col) = np.divmod(np.argmin(score), np.shape(score)[1])
            s = score[row][col]
            (lon, lat) = center_point(row, luduan, longi_lati)
            lon = lon * 180 / math.pi
            lat = lat * 180 / math.pi
            min_list.append([])
            min_list[i].append(row)
            min_list[i].append(col)
            min_list[i].append(s)
            min_list[i].append(lon)
            min_list[i].append(lat)
            score[row][col] = float("inf")
            i += 1
            continue

        (row, col) = np.divmod(np.argmin(score), np.shape(score)[1])
        (lon, lat) = center_point(row, luduan, longi_lati)
        if lon == 0 and lat == 0:
            continue
        lon = lon * 180 / math.pi
        lat = lat * 180 / math.pi


        for j in range(0, i):
            flag = 0
            time_scale = [min_list[j][1] + delta for delta in [-2, -1, 0, 1, 2]]
            lon_scale_min = min_list[j][3] * 0.999
            lon_scale_max = min_list[j][3] * 1.001
            lat_scale_min = min_list[j][4] * 0.999
            lat_scale_max = min_list[j][4] * 1.001


            if col in time_scale and lon_scale_min < lon and lon < lon_scale_max and lat_scale_min < lat and lat < lat_scale_max:
                print(j, row, col, s, lon, lat)
                score[row][col] = float("inf")
                flag = 1
                break
        if flag == 1:
            continue


        print("-------------------这是真正的：", i, row, col, s, lon, lat)

        s = score[row][col]
        min_list.append([])
        min_list[i].append(row)
        min_list[i].append(col)
        min_list[i].append(s)
        min_list[i].append(lon)
        min_list[i].append(lat)
        score[row][col] = float("inf")
        i += 1
        if i == 24:
            break

    return min_list





if __name__ == '__main__':
    r = 3
    style.use('fivethirtyeight')

    # mean_V()


    # V = np.load("mean_V.npy")
    # print(V.shape)
    # i = 0
    # previous = 0.00001
    # (W, H) = initiate(V, r)
    # while True:
    #     (W, H) = iterate(V, W, H, r)
    #     (bool, previous) = judge(V, W, H, previous)
    #     i += 1
    #     print(i)
    #     if i == 1000 or bool:
    #         break
    # print("W___________")
    # print(W)
    # print("H____________")
    # print(H)
    # np.save("mean_W.npy", W)
    # np.save("mean_H.npy", H)

    W = np.load("mean_W.npy")
    H = np.load("mean_H.npy")
    print(W.shape)

    # graph(H, r)

    k = 

    (cluster, result_y) = k_center_package(k, W)
    np.save("mean_cluster_1000.npy", cluster)
    np.save("mean_result_y_1000.npy", result_y)

    cluster = np.load("mean_cluster.npy")
    result_y = np.load("mean_result_y.npy")


    V = np.load("total_9_data.npy")

    # 改第n天
    day = 7
    train = get_day(day)
    score = train_data(V, train, cluster, day, beta=0.5)
    # min_list = find_min(score)
    # result = visualize(min_list)

    result = real_min_list(score)
    np.save("result_2011112" + str(day) + ".npy", result)
    print(result)









