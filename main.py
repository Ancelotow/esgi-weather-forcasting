import os
from sklearn import linear_model

DAYS = [365, 1, 0]

def createDataset():
    data = []
    with open("temperature.txt") as f:
        for line in f.readlines():
            temperature = list(map(float, line.strip().split()))
            data.append(temperature[0])
    print(data)
    dataset = []
    for t in range(max(DAYS), len(data)):
        line = []
        for d in DAYS:
            line.append(data[t - d])
        dataset.append(line)
    return dataset


if __name__ == '__main__':
    print(createDataset())


