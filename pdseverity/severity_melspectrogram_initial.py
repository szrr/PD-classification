import librosa.display
import numpy as np
import os
import librosa

fs = 44100
framelength = 0.025
framesize = int(framelength * fs)
seg_len = 65536
seg_gap = int(65536 * 0.5)

def normalize(s):
# RMS normalization
	new_s = s/np.sqrt(np.sum(np.square(np.abs(s)))/len(s))
	return new_s

def relable(y):
	y = np.array(y)
	for i in range(y.shape[0]):
		if (int(y[i]) == 3):
			y[i] = 2
	return y

def train_severity_mel_spec_init(datapath, fold):
    print("train spectrogram initialing……")
    x_train = []
    y_train = []
    fnames = os.listdir(datapath)
    count = 1
    for name in fnames:
        print("Data num:", count)
        data, fs = librosa.load(datapath + '/' + name, sr=44100)
        data = normalize(data)
        data = data * 1.0 / max(data)
        start = 0
        con = 1
        while(con):
            if(start + seg_len < len(data)):
                mel_spect = librosa.feature.melspectrogram(data[start:start+seg_len], sr=fs, n_fft=framesize)
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                x_train.append(mel_arr)
                y_train.append(name[-5])
                start += seg_gap
            elif(start + seg_len == len(data)):
                mel_spect = librosa.feature.melspectrogram(data[start:start+seg_len], sr=fs, n_fft=framesize)
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                x_train.append(mel_arr)
                y_train.append(name[-5])
                con = 0
            else:
                mel_spect = librosa.feature.melspectrogram(data[len(data)-seg_len:len(data)], sr=fs, n_fft=framesize)
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                x_train.append(mel_arr)
                y_train.append(name[-5])
                con = 0
        count += 1
    x_ = np.float32(x_train)
    np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\specxtrain.npy", x_)
    y_ = relable(y_train)
    y_ = np.float32(y_)
    np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\specytrain.npy", y_)
    print("train mel_spectrogram initial finish")
    return x_, y_

def test_severity_mel_spec_init(datapath, fold):
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
        data = normalize(data)
        data = data * 1.0 / max(data)
        start = 0
        con = 1
        while(con):
            if(start + seg_len < len(data)):
                mel_spect = librosa.feature.melspectrogram(data[start:start+seg_len], sr=fs, n_fft=framesize)
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                x_train.append(mel_arr)
                y_train.append(name[-5])
                start += seg_gap
            elif(start + seg_len == len(data)):
                mel_spect = librosa.feature.melspectrogram(data[start:start+seg_len], sr=fs, n_fft=framesize)
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                x_train.append(mel_arr)
                y_train.append(name[-5])
                con = 0
            else:
                mel_spect = librosa.feature.melspectrogram(data[len(data)-seg_len:len(data)], sr=fs, n_fft=framesize)
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                mel_spect += 80.0
                mel_arr = np.array(mel_spect)
                x_train.append(mel_arr)
                y_train.append(name[-5])
                con = 0
        count += 1
        print("Data segment:", (np.array(x_train)).shape[0])
        data_seg.append((np.array(x_train)).shape[0])
    x_ = np.float32(x_train)
    np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\specxtest.npy", x_)
    y_ = relable(y_train)
    y_ = np.float32(y_)
    np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\specytest.npy", y_)
    data_seg = np.array(data_seg)
    np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\specdataseg.npy", data_seg)
    sample_name = np.array(sample_name)
    np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\specsamplename.npy", sample_name)
    print("test mel_spectrogram initial finish")
    return x_, y_, data_seg, sample_name