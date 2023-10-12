import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

DAYS = [365, 14, 7, 3, 2, 1, 0]


def get_data():
    data = []
    with open("temperature.txt") as f:
        for line in f.readlines():
            temperature = list(map(float, line.strip().split()))
            data.append(temperature[0])
    return data


def create_dataset(data):
    dataset = []
    for t in range(max(DAYS), len(data)):
        line = []
        for d in DAYS:
            line.append(data[t - d])
        dataset.append(line)
    return dataset


def linear_regression(dataset):
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
    predictions = []
    for i in range(len(inputs)):
        prediction = reg.predict([inputs[i]])
        predictions.append(prediction)
        error += abs(prediction - outputs[i])
    error = error / len(inputs)
    return reg, predictions, error


def show_diag(data, predictions):
    plt.plot(data[365:])
    plt.plot(predictions)
    plt.show()


if __name__ == '__main__':
    data = get_data()
    dataset = create_dataset(data)
    reg, predictions, error = linear_regression(dataset)
    print(f'a={reg.coef_} b=†{reg.intercept_}')
    print(f'error={error}')


    forecast = [14] * len(predictions)
    pred = predictions[-1]
    for i in range(365 * 5):
        data.append(pred[0])
        line = []
        for d in DAYS:
            line.append(data[-d])
        p = reg.predict([line[:-1]])
        forecast.append(p)

    plt.plot(data[365:])
    plt.plot(predictions)
    plt.plot(forecast)
    plt.show()




