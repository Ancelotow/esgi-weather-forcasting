import numpy as np
from sklearn import linear_model

DAYS = [365, 14, 7, 3, 2, 1, 0]


def create_dataset():
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


def forcasting(dataset):
    inputs = []
    outputs = []
    for data in dataset:
        line_input = []
        for i in range(len(data) - 1):
            line_input.append(data[i])
        inputs.append(line_input)
        outputs.append(data[len(data) - 1])

    reg = linear_model.LinearRegression()
    reg.fit(inputs, outputs)

    error = 0.0
    for i in range(len(inputs)):
        prediction = reg.predict([inputs[i]])
        error += abs(prediction - outputs[i])
    error = error / len(inputs)

    return reg, error



if __name__ == '__main__':
    dataset = create_dataset()
    reg, error = forcasting(dataset)
    print(f'a={reg.coef_} b=†{reg.intercept_}')
    print(f'error={error}')


