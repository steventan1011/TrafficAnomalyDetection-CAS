import json
from math import *
import os
from geopy.distance import geodesic

def load_data(path):
    files = os.listdir(os.getcwd()+os.sep+path)

    data = {}
    j = 0
    for file in files:
        i = int(file)
        data[str(i)] = {}
        for line in open(path + file, encoding="utf-8"):
            temp = line.split(",")
            data[temp[2]][temp[0]] = {}
            data[temp[2]][temp[0]]["sample_number"] = temp[0]
            data[temp[2]][temp[0]]["mark"] = temp[1]
            data[temp[2]][temp[0]]["taxi_number"] = temp[2]
            data[temp[2]][temp[0]]["sample_time"] = int(temp[3])
            data[temp[2]][temp[0]]["longitude"] = float(temp[4])
            data[temp[2]][temp[0]]["latitude"] = float(temp[5])
            data[temp[2]][temp[0]]["road_left"] = int(temp[6])
            data[temp[2]][temp[0]]["road_right"] = int(temp[7])
            data[temp[2]][temp[0]]["speed"] = int(temp[8])
            data[temp[2]][temp[0]]["angle"] = int(temp[9])
            data[temp[2]][temp[0]]["state"] = temp[10]
        print("第", j, "次", "第", i, "篇")
        with open("data_json/20111123/" + str(i) + ".json", "w", encoding='utf-8') as f:
            d = {}
            d["data"] = data[str(i)]
            f.write(json.dumps(d, indent=4))

        j += 1
    return data

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data_dict = f.read()
        data_dict = json.loads(data_dict)["data"]
    return data_dict


def judge(data):
    sample = []
    for j in data:
        sample.append(data[j]["sample_number"])
    new = 0
    flag_minor = 0
    for k in range(1, len(sample)):
        if flag_minor == 0:
            i = str(sample[k-1])
            data[i]["flag"] = 1
        else:
            i = str(sample[new])
        j = str(sample[k])
        data[j]["flag"] = 1

        lon1 = (pi / 180) * data[i]["longitude"]
        lon2 = (pi / 180) * data[j]["longitude"]
        lat1 = (pi / 180) * data[i]["latitude"]
        lat2 = (pi / 180) * data[j]["latitude"]
        R = 6378.1
        d = acos(sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(lon2-lon1))*R

        if "flag" in data[i]:
            if data[i]["flag"] == 0:
                continue

        if data[i]["state"] == "3" or data[i]["state"] == "4":
            data[i]["flag"] = 0
        elif data[i]["speed"] > 90:
            data[i]["flag"] = 0
        elif d > 2:
            data[j]["flag"] = 0
            print(j)
            print(d)
            if flag_minor == 0:
                flag_minor = 1
                new = k-1

        else:
            data[i]["flag"] = 1
            flag_minor = 0

    return data



if __name__ == '__main__':
    # path = "data/20111123/"
    # data = load_data(path)
    path = "data_json/20111123/13301104001.json"
    data = read_json(path)
    data = judge(data)



    new_data = {}
    for i in data:
        if data[i]["flag"] == 0:
            print(data[i])
        elif data[i]["flag"] == 1:
            new_data[i] = data[i]






