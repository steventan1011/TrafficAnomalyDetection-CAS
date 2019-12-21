from folium import plugins
import folium
import numpy as np
import os


if __name__ == '__main__':
    day = 0

    result = np.load("result_2011112" + str(day) + "_h24.npy")

    lati_lon = []
    for i in range(0, len(result)):
        lati_lon.append([])
        lati_lon[i].append(result[i][4])
        lati_lon[i].append(result[i][3])

    lati_sum = 0
    lon_sum = 0
    for i in range(0, len(lati_lon)):
        lati_sum += lati_lon[i][0]
        lon_sum += lati_lon[i][1]

    lati_mean = lati_sum / len(lati_lon)
    lon_mean = lon_sum / len(lati_lon)

    print(lati_mean, lon_mean)
    m = folium.Map([lati_mean, lon_mean], zoom_start=10)  #中心区域的确定

    location = []
    for i in range(0, len(lati_lon)):
        location.append([lati_lon[i][0], lati_lon[i][1]])

    for i in range(0, len(lati_lon)):
        temp = str(int(result[i][0]))+ "\t"+ str(int(result[i][1])) + ":00" + "  \n" + "[" + str(location[i][1]) + ", " +  str(location[i][0]) + "]"
        folium.Marker([location[i][0], location[i][1]], popup=temp).add_to(m)  # 在地图上设置一个标志符号
        folium.Marker()

    # route = folium.PolyLine(    #polyline方法为将坐标用线段形式连接起来
    #     location,    #将坐标点连接起来
    #     weight=3,  #线的大小为3
    #     color='orange',  #线的颜色为橙色
    #     opacity=0.8    #线的透明度
    # ).add_to(m)    #将这条线添加到刚才的区域m内

    m.save(os.path.join(r'C:/Users/steve/desktop', 'Heatmap_2011112' + str(day) + '.html'))  #将结果以HTML形式保存到桌面上