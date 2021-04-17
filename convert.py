from FeatureStandardizer import FeatureStandardizer
import random
import numpy as np
import pickle


def main():
    ionosphere_data()
    biodeg_data()
    krvskp_data()
    mushroom_data()
    sick_data()
    spambase_data()
    tictactoe_data()
    vote_data()
    simple_data()
    simple2_data()
    unbalanced_data(0.25)


def biodeg_data():
    print('Create biodeg data')
    fs = FeatureStandardizer('data/biodeg.struct', 'data/biodeg.csv')
    fs.process('data/Biodeg.pkl')


def mushroom_data():
    print('Create mushroom data')
    fs = FeatureStandardizer('data/agaricus-lepiota.struct', 'data/agaricus-lepiota.data')
    fs.process('data/Mushroom.pkl')


def krvskp_data():
    print('Create KR vs KP data')
    fs = FeatureStandardizer('data/kr-vs-kp.struct', 'data/kr-vs-kp.data')
    fs.process('data/Krvskp.pkl')


def spambase_data():
    print('Create spam base')
    fs = FeatureStandardizer('data/spambase.struct', 'data/spambase.data')
    fs.process('data/Spam.pkl')


def tictactoe_data():
    print('Create tic tac toe')
    fs = FeatureStandardizer('data/tic-tac-toe.struct', 'data/tic-tac-toe.data')
    fs.process('data/Tictactoe.pkl')


def vote_data():
    print('Create vote')
    fs = FeatureStandardizer('data/house-votes-84.struct', 'data/house-votes-84.data')
    fs.process('data/Vote.pkl')


def sick_data():
    print('Create sick')
    fs = FeatureStandardizer('data/ann-train.struct', 'data/ann-train.data')
    fs.process('data/Sick.pkl')


def ionosphere_data():
    print('Create ionosphere')
    fs = FeatureStandardizer('data/ionosphere.struct', 'data/ionosphere.data')
    fs.process('data/Ionosphere.pkl')


def simple_data():
    print('Create simple data')
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
    print("Num features before: 2")
    print("Num classes: 1")
    print("Num features after: 2")
    print("Num items: {}".format(X.shape[0]))
    print("Num pos: {}".format(sum(y)))
    print("Num neg: {}".format(sum(1-y)))
    print("")
    with open('data/Simple.pkl', 'wb') as outfile:
        outfile.write(pickle.dumps(data))


def simple2_data():
    print('Create simple2')
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
    print("Num features before: 2")
    print("Num classes: 1")
    print("Num features after: 2")
    print("Num items: {}".format(X.shape[0]))
    print("Num pos: {}".format(sum(y)))
    print("Num neg: {}".format(sum(1-y)))
    print("")
    with open('data/Simple2.pkl', 'wb') as outfile:
        outfile.write(pickle.dumps(data))


def unbalanced_data(pos_percent):
    print('Create unbalanced data')
    num = 500
    features = np.zeros(2)
    X = np.zeros((num, 2))
    y = np.zeros(num)
    i = 0
    while i < num:
        features[0] = random.randrange(100)
        features[1] = random.randrange(100)
        if features[0] + features[1] < 50:
            label = 0
        elif features[0] + features[1] < 100:
            label = 1
        elif features[0] + features[1] < 150:
            label = 0
        else:
            label = 1
        if label == 0 or random.random() < pos_percent:
            X[i][0] = features[0]
            X[i][1] = features[1]
            y[i] = label
            i += 1
    data = {
        'X': X,
        'y': y
    }
    print("Num features before: 2")
    print("Num classes: 1")
    print("Num features after: 2")
    print("Num items: {}".format(X.shape[0]))
    print("Num pos: {}".format(sum(y)))
    print("Num neg: {}".format(sum(1-y)))
    print("")
    with open('data/Unbalanced.pkl', 'wb') as outfile:
        outfile.write(pickle.dumps(data))


if __name__ == '__main__':
    main()
