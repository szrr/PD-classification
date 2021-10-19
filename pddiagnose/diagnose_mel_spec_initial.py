import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from scipy import signal
import os
import librosa

#采样率
fs = 44100
#窗口大小为0.025s
framelength = 0.025
#NFFT点数=0.025*fs
framesize = int(framelength * fs)
#窗口长度
seg_len = 65536
#重叠部分长度
seg_gap = int(seg_len*0.5)

def normalize(s):
# RMS normalization
	new_s = s/np.sqrt(np.sum(np.square(np.abs(s)))/len(s))
	return new_s

def string2num(y):
    y1 = []
    for i in y:
        if (i == 'h'):
            y1.append(0)
        else:
            y1.append(1)
    y1 = np.float32(np.array(y1))
    return y1

def train_mel_spec_init(datapath, fold):
    print("train spectrogram initialing……")
    x_train = []
    y_train = []
    fnames = os.listdir(datapath)
    count = 1
    for name in fnames:
        print("Data num:", count)
        data, fs = librosa.load(datapath + '/' + name, sr=44100)
        data = normalize(data)
        #归一化
        data = data * 1.0 / max(data)
        start = 0
        con = 1
        while(con):
            if(start + seg_len < len(data)):
                # 提取mel特征
                mel_spect = librosa.feature.melspectrogram(data[start:start+seg_len], sr=fs, n_fft=1024, hop_length=512)
                # 转化为log形式
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                #是否添加norm
                x_train.append(mel_arr)
                y_train.append(name[0])
                start += seg_gap
            elif(start + seg_len == len(data)):
                # 提取mel特征
                mel_spect = librosa.feature.melspectrogram(data[start:start+seg_len], sr=fs, n_fft=1024, hop_length=512)
                # 转化为log形式
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                # 是否添加norm
                x_train.append(mel_arr)
                y_train.append(name[0])
                con = 0
            else:
                # 提取mel特征
                mel_spect = librosa.feature.melspectrogram(data[len(data)-seg_len:len(data)], sr=fs, n_fft=1024, hop_length=512)
                # 转化为log形式
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                # 是否添加norm
                #print(mel_arr.shape)
                x_train.append(mel_arr)
                y_train.append(name[0])
                con = 0
        count += 1
    x_ = np.float32(x_train)
    np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newspecxtrain.npy", x_)
    y_ = string2num(y_train)
    np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newspecytrain.npy", y_)
    print("train mel_spectrogram initial finish")
    return x_, y_

def test_mel_spec_init(datapath, fold):
    print("test spectrogram initialing……")
    x_train = []
    y_train = []
    data_seg = []
    data_seg.append(0)
    sample_name = []
    fnames = os.listdir(datapath)
    count = 1
    for name in fnames:
        sample_name.append(name[0:4])
        print("Data num:", count)
        data, fs = librosa.load(datapath + '/' + name, sr=44100)
        # data = normalize(data)
        # 归一化
        data = data * 1.0 / max(data)
        start = 0
        con = 1
        while(con):
            if(start + seg_len < len(data)):
                # 提取mel特征
                mel_spect = librosa.feature.melspectrogram(data[start:start+seg_len], sr=fs, n_fft=1024, hop_length=512)
                # 转化为log形式
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                #是否添加norm
                x_train.append(mel_arr)
                y_train.append(name[0])
                start += seg_gap
            elif(start + seg_len == len(data)):
                # 提取mel特征
                mel_spect = librosa.feature.melspectrogram(data[start:start+seg_len], sr=fs, n_fft=1024, hop_length=512)
                # 转化为log形式
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                # 是否添加norm
                x_train.append(mel_arr)
                y_train.append(name[0])
                con = 0
            else:
                # 提取mel特征
                mel_spect = librosa.feature.melspectrogram(data[len(data)-seg_len:len(data)], sr=fs, n_fft=1024, hop_length=512)
                # 转化为log形式
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                # 是否添加norm
                x_train.append(mel_arr)
                y_train.append(name[0])
                con = 0
        count += 1
        print("Data segment:", (np.array(x_train)).shape[0])
        data_seg.append((np.array(x_train)).shape[0])
    x_ = np.float32(x_train)
    np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newspecxtest.npy", x_)
    y_ = string2num(y_train)
    np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newspecytest.npy", y_)
    data_seg = np.array(data_seg)
    np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newspecdataseg.npy", data_seg)
    sample_name = np.array(sample_name)
    np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newspecsamplename.npy", sample_name)
    print("test mel_spectrogram initial finish")
    return x_, y_, data_seg, sample_name