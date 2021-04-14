from FeatureStandardizer import FeatureStandardizer
import random
import numpy as np
import pickle


def main():
    #agaricus_data()
    #breast_data()
    simple_data()
    simple2_data()


def agaricus_data():
    feature_info = [
        ('edible', 'cat', 'output'),
        ('cap-shape', 'cat'),
        ('cap-surface', 'cat'),
        ('cap-color', 'cat'),
        ('bruises', 'cat'),
        ('odor', 'cat'),
        ('gill-attachment', 'cat'),
        ('gill-spacing', 'cat'),
        ('gill-size', 'cat'),
        ('gill-color', 'cat'),
        ('stalk-shape', 'cat'),
        ('stalk-root', 'cat'),
        ('stalk-surface-above-ring', 'cat'),
        ('stalk-surface-below-ring', 'cat'),
        ('stalk-color-above-ring', 'cat'),
        ('stalk-color-below-ring', 'cat'),
        ('veil-type', 'cat'),
        ('veil-color', 'cat'),
        ('ring-number', 'cat'),
        ('ring-type', 'cat'),
        ('spore-print-color', 'cat'),
        ('population', 'cat'),
        ('habitat', 'cat'),
    ]
    fs = FeatureStandardizer(feature_info, 'data/agaricus-lepiota.data')
    fs.process('data/Agaricus.pkl')


def simple_data():
    num = 500
    X = np.zeros((num, 2))
    y = np.zeros(num)
    for i in range(num):
        X[i][0] = random.randrange(100)
        X[i][1] = random.randrange(100)
        if X[i][0] + X[i][1] > 100:
            y[i] = 1
        else:
            y[i] = 0
    data = {
        'X': X,
        'y': y
    }
    with open('data/Simple.pkl', 'wb') as outfile:
        outfile.write(pickle.dumps(data))


def simple2_data():
    num = 500
    X = np.zeros((num, 2))
    y = np.zeros(num)
    for i in range(num):
        X[i][0] = random.randrange(100)
        X[i][1] = random.randrange(100)
        if X[i][0] + X[i][1] < 50:
            y[i] = 0
        elif X[i][0] + X[i][1] < 100:
            y[i] = 1
        elif X[i][0] + X[i][1] < 150:
            y[i] = 0
        else:
            y[i] = 1
    data = {
        'X': X,
        'y': y
    }
    with open('data/Simple2.pkl', 'wb') as outfile:
        outfile.write(pickle.dumps(data))


def breast_data():
    feature_info = [
        ('y', 'cat', 'output'),
        ('x1', 'float'),
        ('x2', 'float'),
    ]
    train_data = np.genfromtxt('data/BreastTissueRaw.txt', dtype=['i8', 'S5', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'], delimiter='\t', skip_header=True)
    num_rows = train_data.shape[0]
    num_cols = 9
    num_classes = 6
    x_class = {}
    c = 0
    X = np.zeros((num_rows, num_cols))
    Y = -np.ones((num_rows, num_classes))
    for i in range(num_rows):
        for j in range(num_cols):
            X[i, j] = train_data[i][j + 2]
            if not train_data[i][1] in x_class:
               x_class[train_data[i][1]] = c
               c += 1
            p = x_class[train_data[i][1]]
            Y[i, p] = 1
    write_file('data/BreastTissue.pkl', X, Y)



if __name__ == '__main__':
    main()
