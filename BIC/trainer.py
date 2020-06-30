import torch
import torchvision
from torchvision.models import vgg16
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from torch.optim.lr_scheduler import LambdaLR, StepLR
import seaborn as sns
import torch
import torchvision
from torchvision.models import vgg16
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from torch.optim.lr_scheduler import LambdaLR, StepLR

import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import pickle
from dataset import BatchData
from model import PreResNet, BiasLayer
from cifar100 import Cifar100
from exemplar import Exemplar
from copy import deepcopy


class Trainer:
    def __init__(self, total_cls):
        self.total_cls = total_cls
        self.seen_cls = 0
        self.dataset = Cifar100()
        self.model = PreResNet(32,total_cls).cuda()
        print(self.model)
        self.model = nn.DataParallel(self.model)
        self.bias_layer1 = BiasLayer().cuda()
        self.bias_layer2 = BiasLayer().cuda()
        self.bias_layer3 = BiasLayer().cuda()
        self.bias_layer4 = BiasLayer().cuda()
        self.bias_layer5 = BiasLayer().cuda()
        self.bias_layer6 = BiasLayer().cuda()
        self.bias_layer7 = BiasLayer().cuda()
        self.bias_layer8 = BiasLayer().cuda()
        self.bias_layer9 = BiasLayer().cuda()
        self.bias_layer10 = BiasLayer().cuda()
        self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5,\
            self.bias_layer6, self.bias_layer7, self.bias_layer8, self.bias_layer9, self.bias_layer10]
        self.input_transform= Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32,padding=4),
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

        self.input_transform_eval= Compose([
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)



    def test(self, testdata):
        print("test data number : ",len(testdata))
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        t=self.total_cls
        matrix = new_confusion_matrix(lenx=t, leny=t)
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
            update_confusion_matrix(matrix, pred, label)
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        self.model.train()
        show_confusion_matrix(matrix)
        print("---------------------------------------------")
        return acc


    def eval(self, criterion, evaldata):
        self.model.eval()
        losses = []
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(evaldata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p, label)
            losses.append(loss.item())
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        print("Validation Loss: {}".format(np.mean(losses)))
        print("Validation Acc: {}".format(100*correct/(correct+wrong)))
        self.model.train()
        return



    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, batch_size, epoches, lr, max_size):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        exemplar = Exemplar(max_size, total_cls)

        previous_model = None

        dataset = self.dataset
        test_xs = []
        test_ys = []
        train_xs = []
        train_ys = []

        test_accs = []
        for inc_i in range(dataset.batch_num):
            print(f"Incremental num : {inc_i}")
            train, val, test = dataset.getNextClasses(inc_i)
            print(len(train), len(val), len(test))
            train_x, train_y = zip(*train)
            val_x, val_y = zip(*val)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)
            test_ys.extend(test_y)

            train_xs, train_ys = exemplar.get_exemplar_train()
            train_xs.extend(train_x)
            train_xs.extend(val_x)
            train_ys.extend(train_y)
            train_ys.extend(val_y)


            train_data = DataLoader(BatchData(train_xs, train_ys, self.input_transform),
                        batch_size=batch_size, shuffle=True, drop_last=True)
            val_data = DataLoader(BatchData(val_x, val_y, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)
            test_data = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)
            scheduler = StepLR(optimizer, step_size=70, gamma=0.1)


            # bias_optimizer = optim.SGD(self.bias_layers[inc_i].parameters(), lr=lr, momentum=0.9)
            bias_optimizer = optim.Adam(self.bias_layers[inc_i].parameters(), lr=0.001)
            # bias_scheduler = StepLR(bias_optimizer, step_size=70, gamma=0.1)
            exemplar.update(total_cls//dataset.batch_num, (train_x, train_y), (val_x, val_y))

            self.seen_cls = exemplar.get_cur_cls()
            print("seen cls number : ", self.seen_cls)
            val_xs, val_ys = exemplar.get_exemplar_val()
            val_bias_data = DataLoader(BatchData(val_xs, val_ys, self.input_transform),
                        batch_size=100, shuffle=True, drop_last=False)
            test_acc = []


            for epoch in range(epoches):
                print("---"*50)
                print("Epoch", epoch)
                scheduler.step()
                cur_lr = self.get_lr(optimizer)
                print("Current Learning Rate : ", cur_lr)
                self.model.train()
                if inc_i > 0:
                    self.stage1_distill(train_data, criterion, optimizer)
                else:
                    self.stage1(train_data, criterion, optimizer)
                acc = self.test(test_data)
            if inc_i > 0:
                for epoch in range(epoches):
                    # bias_scheduler.step()
                    self.stage2(val_bias_data, criterion, bias_optimizer)
                    if epoch % 50 == 0:
                        acc = self.test(test_data)
                        test_acc.append(acc)
            for i, layer in enumerate(self.bias_layers):
                layer.printParam(i)
            self.previous_model = deepcopy(self.model)
            acc = self.test(test_data)
            test_acc.append(acc)
            test_accs.append(max(test_acc))
            print(test_accs)


    def bias_forward(self, input):
        in1 = input[:, :10]
        in2 = input[:, 10:20]
        in3 = input[:, 20:30]
        in4 = input[:, 30:40]
        in5 = input[:, 40:50]
        in6 = input[:, 50:60]
        in7 = input[:, 60:70]
        in8 = input[:, 70:80]
        in9 = input[:, 80:90]
        in10=input[:,90:100]
        out1 = self.bias_layer1(in1)
        out2 = self.bias_layer2(in2)
        out3 = self.bias_layer3(in3)
        out4 = self.bias_layer4(in4)
        out5 = self.bias_layer5(in5)
        out6 = self.bias_layer1(in6)
        out7 = self.bias_layer2(in7)
        out8 = self.bias_layer3(in8)
        out9 = self.bias_layer4(in9)
        out10 = self.bias_layer5(in10)
        return torch.cat([out1, out2, out3, out4, out5,out6,out7,out8,out9,out10], dim = 1)



    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage1_distill(self, train_data, criterion, optimizer):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        alpha = (self.seen_cls - 20)/ self.seen_cls
        print("classification proportion 1-alpha = ", 1-alpha)
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            with torch.no_grad():
                pre_p = self.previous_model(image)
                pre_p = self.bias_forward(pre_p)
                pre_p = F.softmax(pre_p[:,:self.seen_cls-20]/T, dim=1)
            logp = F.log_softmax(p[:,:self.seen_cls-20]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], label)
            loss = loss_soft_target * T * T + (1-alpha) * loss_hard_target
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))


    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage2(self, val_bias_data, criterion, optimizer):
        print("Evaluating ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(val_bias_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage2 loss :", np.mean(losses))

def update_confusion_matrix(matrix, preds, datas):
    for pred, data in zip(preds,datas):
        matrix[data.item(),pred.item()] = matrix[data.item(),pred.item()]+1

def new_confusion_matrix(lenx=100, leny=100):
    matrix = np.zeros((leny, lenx))
    return matrix

def show_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(15,9))
    ax = sns.heatmap(matrix, linewidth=0.2,cmap='Reds')
    plt.savefig(f"cm_{matrix.shape[0]}.png") #Store the pic locally
    plt.show()
