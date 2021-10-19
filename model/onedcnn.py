import torch
from torch.nn import Module
from torch import nn

class aggregation_onedcnn(nn.Module):
    """
    Input - 1 * 65536
    Output - 2
    """
    def __init__(self):
        super(aggregation_onedcnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, stride=4),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            # nn.MaxPool1d(4)
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.gap = nn.AdaptiveAvgPool1d(1)

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
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        f1 = self.conv6(y)
        f2 = self.conv7(f1)
        f3 = self.conv8(f2)
        gapf1 = self.gap(f1)
        gapf2 = self.gap(f2)
        f3 = f3.view(f3.size(0), -1)
        gapf1 = gapf1.view(gapf1.size(0), -1)
        gapf2 = gapf2.view(gapf2.size(0), -1)

        y = torch.cat([gapf1, f3, gapf2], dim=-1)

        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class onedcnn(nn.Module):
    """
    Input - 1 * 65536
    Output - 2
    """
    def __init__(self):
        super(onedcnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, stride=4),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            # nn.MaxPool1d(4)
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))

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
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.conv7(y)
        y = self.conv8(y)

        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class onedcnn_severity(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, stride=4),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            # nn.MaxPool1d(4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))

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
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.conv7(y)
        y = self.conv8(y)

        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class aggregation_onedcnn_severity(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, stride=4),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            # nn.MaxPool1d(4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 3, 256),
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
        y = self.conv5(y)
        f1 = self.conv6(y)
        f2 = self.conv7(f1)
        f3 = self.conv8(f2)
        gapf1 = self.gap(f1)
        gapf2 = self.gap(f2)
        f3 = f3.view(f3.size(0), -1)
        gapf1 = gapf1.view(gapf1.size(0), -1)
        gapf2 = gapf2.view(gapf2.size(0), -1)

        y = torch.cat([gapf1, f3, gapf2], dim=-1)

        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y