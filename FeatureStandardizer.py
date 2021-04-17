import numpy as np
import pickle


class FeatureStandardizer:

    def __init__(self, struct_file_name, data_file_name):
        self.feature_info = []
        self.struct_file_name = struct_file_name
        self.data_file_name = data_file_name
        self.output_column = -1
        self.output_positive = ''

    def process(self, save_file_name):
        self.__read_struct()
        print("Num features before: {}".format(len(self.feature_info)-1))
        train_data = np.genfromtxt(self.data_file_name, dtype=['S100']*len(self.feature_info), delimiter=',', skip_header=False)
        num_rows = len(train_data)
        num_cols = len(train_data[0])

        cats = self.__get_categories(train_data)
        num_expanded_cols = 0
        num_classes = 0
        for k in range(num_cols):
            if k == self.output_column:
                if len(cats[k]) == 2:
                    num_classes += 1
                else:
                    num_classes += len(cats[k])
            else:
                if len(cats[k]) == 2:
                    num_expanded_cols += 1
                else:
                    num_expanded_cols += len(cats[k])
        X = np.zeros((num_rows, num_expanded_cols))
        print("Num classes: {}".format(num_classes))
        if num_classes == 1:
            y = np.zeros(num_rows)
        else:
            y = np.zeros((num_rows, num_classes))

        for i in range(num_rows):
            j = 0
            for k in range(num_cols):
                if k == self.output_column:
                    m = cats[k][train_data[i][k]]
                    if num_classes == 1:
                        y[i] = int(self.output_positive == train_data[i][k].decode('ascii'))
                    else:
                        y[i, m] = 1
                else:
                    if self.feature_info[k][1] != 'cat':
                        X[i, j] = train_data[i][k]
                        j += 1
                    else:
                        m = cats[k][train_data[i][k]]
                        if len(cats[k]) == 2:
                            X[i, j] = m
                            j += 1
                        else:
                            X[i, j + m] = 1
                            j += len(cats[k])
        print("Num features after: {}".format(X.shape[1]))
        print("Num items: {}".format(X.shape[0]))
        print("Num pos: {}".format(sum(y)))
        print("Num neg: {}".format(sum(1-y)))
        print("")
        self.__write_file(save_file_name, X, y)

    def __read_struct(self):
        with open(self.struct_file_name, "r") as infile:
            i = 0
            for line in infile:
                items = line.strip("\n").split("\t")
                self.feature_info.append((items[0], items[1]))
                if len(items) == 3:
                    self.output_column = i
                    self.output_positive = items[2]
                i += 1

    def __get_categories(self, train_data):
        cat_count = []
        for j in range(len(train_data[0])):
            if self.feature_info[j][1] == 'cat':
                items = {}
                k = 0
                for i in range(len(train_data)):
                    if train_data[i][j] not in items:
                        items[train_data[i][j]] = k
                        k += 1
                cat_count.append(items)
            else:
                cat_count.append({0})
        return cat_count

    def __write_file(self, name, X, y):
        data = {
            'X': X,
            'y': y
        }
        with open(name, 'wb') as outfile:
            outfile.write(pickle.dumps(data))
