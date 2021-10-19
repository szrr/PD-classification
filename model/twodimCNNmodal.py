import torch
import torch.nn as nn
from collections import OrderedDict


class twodimcnn1(nn.Module):
    """
    Input - 1x128x128
    Output - 2
    """

    def __init__(self):
        super(twodimcnn1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
        #     nn.ReLU(),
        #     #nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        #     nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 128, 512),
            nn.Dropout(0.3),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(64, 3))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        # y = self.conv5(y)

        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class twodimcnn1_spec(nn.Module):
    """
    Input - 1x128x128
    Output - 2
    """
    def __init__(self):
        super(twodimcnn1_spec, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
        #     nn.ReLU(),
        #     #nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        #     nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.fc1 = nn.Sequential(
            nn.Linear(2*2*128, 512),
            nn.Dropout(0.5),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(64, 3))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        #y = self.conv5(y)

        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class aggregation_twodimcnn(nn.Module):
    """
    Input - 1x128x128
    Output - 2
    """
    def __init__(self):
        super(aggregation_twodimcnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(256*3, 256),
            nn.Dropout(0.3),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(64, 2))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        f1 = self.conv3(y)
        f2 = self.conv4(f1)
        f3 = self.conv5(f2)

        f1 = self.gap(f1)
        f2 = self.gap(f2)
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        f3 = f3.view(f3.size(0), -1)
        y = torch.cat([f1, f3, f2], dim=-1)

        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class twodimcnn(nn.Module):
    """
    Input - 1x128x128
    Output - 2
    """
    def __init__(self):
        super(twodimcnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            )
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.3),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(64, 2))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        #y = self.conv5(y)

        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class twodcnn_severity(nn.Module):
    """
    Input - 1x128x128
    Output - 3
    """
    def __init__(self):
        super(twodcnn_severity, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            )
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.3),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(64, 3))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)

        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class aggregation_twodcnn_severity(nn.Module):
    """
    Input - 1x128x128
    Output - 3
    """
    def __init__(self):
        super(aggregation_twodcnn_severity, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(256*3, 256),
            nn.Dropout(0.3),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(64, 3))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        f1 = self.conv3(y)
        f2 = self.conv4(f1)
        f3 = self.conv5(f2)

        f1 = self.gap(f1)
        f2 = self.gap(f2)
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        f3 = f3.view(f3.size(0), -1)
        y = torch.cat([f1, f3, f2], dim=-1)

        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y