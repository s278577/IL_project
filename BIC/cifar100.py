import pickle
import numpy as np
import os

class Cifar100:
    def __init__(self):
        with open('cifar100ds/train','rb') as f:
            self.train = pickle.load(f, encoding='latin1')
        with open('cifar100ds/test','rb') as f:
            self.test = pickle.load(f, encoding='latin1')
        self.train_data = self.train['data']
        self.train_labels = self.train['fine_labels']
        self.test_data = self.test['data']
        self.test_labels = self.test['fine_labels']
        self.train_groups, self.val_groups, self.test_groups = self.initialize()
        self.batch_num = 5

    def initialize(self):
        train_groups = [[],[],[],[],[],[],[],[],[],[]]
        for train_data, train_label in zip(self.train_data, self.train_labels):
            # print(train_data.shape)
            train_data_r = train_data[:1024].reshape(32, 32)
            train_data_g = train_data[1024:2048].reshape(32, 32)
            train_data_b = train_data[2048:].reshape(32, 32)
            train_data = np.dstack((train_data_r, train_data_g, train_data_b))
            if train_label < 10:
                train_groups[0].append((train_data,train_label))
            elif 10 <= train_label < 20:
                train_groups[1].append((train_data,train_label))
            elif 20 <= train_label < 30:
                train_groups[2].append((train_data,train_label))
            elif 30 <= train_label < 40:
                train_groups[3].append((train_data,train_label))
            elif 40 <= train_label < 50:
                train_groups[4].append((train_data,train_label))

            elif 50 <= train_label < 60:
                train_groups[5].append((train_data,train_label))
            elif 60 <= train_label < 70:
                train_groups[6].append((train_data,train_label))
            elif 70 <= train_label < 80:
                train_groups[7].append((train_data,train_label))
            elif 80 <= train_label < 90:
                train_groups[8].append((train_data,train_label))
            elif 90 <= train_label < 100:
                train_groups[9].append((train_data,train_label))
       

        val_groups = [[],[],[],[],[],[],[],[],[],[]]
        for i, train_group in enumerate(train_groups):
            val_groups[i] = train_groups[i][9000:]
            train_groups[i] = train_groups[i][:9000]
        

        test_groups =[[],[],[],[],[],[],[],[],[],[]]
        for test_data, test_label in zip(self.test_data, self.test_labels):
            test_data_r = test_data[:1024].reshape(32, 32)
            test_data_g = test_data[1024:2048].reshape(32, 32)
            test_data_b = test_data[2048:].reshape(32, 32)
            test_data = np.dstack((test_data_r, test_data_g, test_data_b))
            if test_label < 10:
                train_groups[0].append((train_data,test_label))
            elif 10 <= test_label < 20:
                train_groups[1].append((train_data,test_label))
            elif 20 <= test_label < 30:
                train_groups[2].append((train_data,test_label))
            elif 30 <= test_label < 40:
                train_groups[3].append((train_data,test_label))
            elif 40 <= test_label < 50:
                train_groups[4].append((train_data,test_label))

            elif 50 <= test_label < 60:
                train_groups[5].append((train_data,test_label))
            elif 60 <= test_label < 70:
                train_groups[6].append((train_data,test_label))
            elif 70 <= test_label < 80:
                train_groups[7].append((train_data,test_label))
            elif 80 <= test_label < 90:
                train_groups[8].append((train_data,test_label))
            elif 90 <= test_label < 100:
                train_groups[9].append((train_data,test_label))
       

        return train_groups, val_groups, test_groups

    def getNextClasses(self, i):
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

if __name__ == "__main__":
    cifar = Cifar100()
    print(len(cifar.train_groups[0]))