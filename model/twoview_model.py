import torch
from torch.nn import Module
from torch import nn
from model import twodimcnn1

class wave_view_model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=5, padding=2, stride=4), nn.ReLU(), nn.Dropout(p=0.1))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv4 = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv6 = nn.Sequential(nn.Conv1d(128, 256, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv7 = nn.Sequential(nn.Conv1d(256, 256, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv8 = nn.Sequential(nn.Conv1d(256, 256, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.conv7(y)
        y = self.conv8(y)
        return y

class spec_view_model1(nn.Module):
    def __init__(self):
        super(spec_view_model1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        return y

class twoview_model(nn.Module):
    def __init__(self):
        super(twoview_model1, self).__init__()
        self.wave_feature_extraction = wave_view_model1()
        self.spec_feature_extraction = spec_view_model1()
        self.dp1 = nn.Dropout(0.3)
        # self.fc1 = nn.Linear(256*2,128)
        # self.fc2 = nn.Linear(128, 10)
        # self.fc3 = nn.Linear(10, 3)
        self.fc1 = nn.Linear(256 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, waveform, spectrogram):
        wave_feature = self.wave_feature_extraction(waveform)
        spec_feature = self.spec_feature_extraction(spectrogram)
        wave_feature = wave_feature.view(wave_feature.size(0), -1)
        spec_feature = spec_feature.view(spec_feature.size(0), -1)

        com=torch.cat([wave_feature,spec_feature],dim=-1)
        y = self.fc1(com)
        y = self.dp1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class twoview_severity(nn.Module):
    def __init__(self):
        super(twoview_severity, self).__init__()
        self.wave_feature_extraction = wave_view_model1()
        self.spec_feature_extraction = spec_view_model1()
        self.dp1 = nn.Dropout(0.3)
        # self.fc1 = nn.Linear(256*2,128)
        # self.fc2 = nn.Linear(128, 10)
        # self.fc3 = nn.Linear(10, 3)
        self.fc1 = nn.Linear(256 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, waveform, spectrogram):
        wave_feature = self.wave_feature_extraction(waveform)
        spec_feature = self.spec_feature_extraction(spectrogram)
        wave_feature = wave_feature.view(wave_feature.size(0), -1)
        spec_feature = spec_feature.view(spec_feature.size(0), -1)

        com=torch.cat([wave_feature,spec_feature],dim=-1)
        y = self.fc1(com)
        y = self.dp1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class wave_view_model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=5, padding=2, stride=4), nn.ReLU(), nn.Dropout(p=0.1))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv4 = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv6 = nn.Sequential(nn.Conv1d(128, 256, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv7 = nn.Sequential(nn.Conv1d(256, 256, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.conv8 = nn.Sequential(nn.Conv1d(256, 256, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.gap = nn.AdaptiveAvgPool1d(1)

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
        return y

class spec_view_model2(nn.Module):
    def __init__(self):
        super(spec_view_model2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

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
        return y

class twoview_aggregation_model(nn.Module):
    def __init__(self):
        super(twoview_aggregation_model, self).__init__()
        self.wave_feature_extraction = wave_view_model2()
        self.spec_feature_extraction = spec_view_model2()
        self.dp1 = nn.Dropout(0.3)
        # self.fc1 = nn.Linear(256*6,128)
        # self.fc2 = nn.Linear(128, 10)
        # self.fc3 = nn.Linear(10, 3)
        self.fc1 = nn.Linear(256 * 6, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, waveform, spectrogram):
        wave_feature = self.wave_feature_extraction(waveform)
        spec_feature = self.spec_feature_extraction(spectrogram)
        wave_feature = wave_feature.view(wave_feature.size(0), -1)
        spec_feature = spec_feature.view(spec_feature.size(0), -1)

        com=torch.cat([wave_feature,spec_feature],dim=-1)
        y = self.fc1(com)
        y = self.dp1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

class aggregation_twoview_severity(nn.Module):
    def __init__(self):
        super(aggregation_twoview_severity, self).__init__()
        self.wave_feature_extraction = wave_view_model2()
        self.spec_feature_extraction = spec_view_model2()
        self.dp1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 6, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, waveform, spectrogram):
        wave_feature = self.wave_feature_extraction(waveform)
        spec_feature = self.spec_feature_extraction(spectrogram)
        wave_feature = wave_feature.view(wave_feature.size(0), -1)
        spec_feature = spec_feature.view(spec_feature.size(0), -1)

        com=torch.cat([wave_feature,spec_feature],dim=-1)
        y = self.fc1(com)
        y = self.dp1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y