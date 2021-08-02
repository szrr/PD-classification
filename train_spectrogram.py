import numpy as np
import math
from PIL import Image
from scipy import signal
import os
import librosa
from spectroinitial import trainspecload
from spectroinitial import testspecload
from spectroinitial import spectrogramInit
from AlexNetmodel import AlexNet
import torch
import torch.nn as nn
import torchvision
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

datapath_train = 'F:\Parkinson speech\dataset\KCL train'
datapath_test = 'F:\Parkinson speech\dataset\KCL test'
minitrain = 'F:\Parkinson speech\dataset\minitrain'
minitest = 'F:\Parkinson speech\dataset\minitest'
spectro_train = 'F:\Parkinson speech\dataset\sepctrotrain'
spectro_test = 'F:\Parkinson speech\dataset\sepctrotest'

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    #device = torch.device('cuda:0')
    #spectrogramInit(datapath_train, spectro_train)
    #spectrogramInit(datapath_test, spectro_test)
    x_train, y_train = trainspecload(spectro_train)#.to(device)
    x_test, y_test = trainspecload(spectro_test)#.to(device)
    #y_test = y_test.reshape(-1)
    #y_train = y_train.reshape(-1)
    print("dataloading finish")
    print("x_train shape:", x_train.shape,"y_train shape:" , y_train.shape[0])
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_train = x_train.unsqueeze(1)
    y_train = y_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)
    y_test = y_test.unsqueeze(1)
    print(x_train.shape, y_train.shape)
    #torch.Size([636, 1, 51529]) torch.Size([636, 1])
    batch_size = 16
    LR = 0.001
    train_dataset = MyDataset(x_train, y_train)  # 创建训练数据集
    test_dataset = MyDataset(x_test, y_test)  # 创建测试数据集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 创建训练数据集加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = AlexNet()
    momentum_sgd = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    cost = CrossEntropyLoss()
    epoch = 20

    for _epoch in range(epoch):
        avg_loss = 0
        count = 0
        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 2))
            momentum_sgd.zero_grad()
            y_tr = train_label.squeeze(1)
            #print("Train_x:", train_x, "Train label:",y_tr)
            predict_y = model(train_x.float())
            loss = cost(predict_y, y_tr.long())
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            momentum_sgd.step()
            #avg_loss += loss
            #count += 1
        #avg_loss /= count
        #scheduler.step(avg_loss)
        correct = 0
        _sum = 0

        print("testing epoch ", _epoch)
        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            # print([predict_y])
            predict_ys = np.argmax(predict_y, axis=-1)
            print(predict_ys)
            label_np = test_label.numpy()
            label_np = label_np.reshape(label_np.size)
            label_np = torch.from_numpy(label_np)
            print(label_np)
            _ = predict_ys == label_np
            print(_)
            correct += np.sum(_.numpy())
            print(correct)
            _sum += _.shape[0]
            print(_sum)

        print('accuracy: {:.2f}'.format(correct / _sum))
        torch.save(model, 'F:\Parkinson speech\\1DCNN\onedimcnn_{:.2f}.pkl'.format(correct / _sum))
