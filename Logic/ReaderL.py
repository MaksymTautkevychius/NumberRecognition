import numpy as np

class ReaderL:
    def __init__(self):
        pass
    @staticmethod
    def read_file(filename):
        data_list, labels = [], []
        with open(filename, mode='r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) != 785:
                    continue
                label = int(parts[0])
                pixels = [int(x) / 255.0 for x in parts[1:]]
                data_list.append(pixels)
                labels.append(label)
        return data_list, labels



train_data_list, actual_numbers = ReaderL.read_file('mnist_train.csv')


