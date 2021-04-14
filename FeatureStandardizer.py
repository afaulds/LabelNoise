import numpy as np
import pickle


class FeatureStandardizer:


    def __init__(self, feature_info, file_name):
        self.feature_info = feature_info
        self.file_name = file_name


    def process(self, save_file_name):
        train_data = np.genfromtxt(self.file_name, dtype=['S5']*len(self.feature_info), delimiter=',', skip_header=False)
        output_column = self.__get_output_column()
        num_rows = len(train_data)
        num_cols = len(train_data[0])

        cats = self.__get_categories(train_data)
        num_expanded_cols = 0
        num_classes = 0
        for k in range(num_cols):
            if k == output_column:
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
        Y = np.zeros((num_rows, num_classes))

        for i in range(num_rows):
            j = 0
            for k in range(num_cols):
                if k == output_column:
                    m = cats[k][train_data[i][k]]
                    if num_classes == 1:
                        Y[i, 0] = m
                    else:
                        Y[i, m] = 1
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
        self.__write_file('data/Agaricus.pkl', X, Y)


    def __get_output_column(self):
        for i in range(len(self.feature_info)):
            if len(self.feature_info[i]) == 3:
                return i
        return -1


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


    def __write_file(self, name, X, Y):
        data = {
            'X': X,
            'Y': Y
        }
        with open(name, 'wb') as outfile:
            outfile.write(pickle.dumps(data))
