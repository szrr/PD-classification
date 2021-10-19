import numpy as np
import os
import librosa

seg_len = 65536 #split segment length
seg_ov = int(seg_len*0.5) # 50% overlap

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

def traindataload(path, fold):
	print("dataloading……")
	x_train = []
	y_train = []
	fnames = os.listdir(path)
	count = 1
	for name in fnames:
		print("Data num:",count)
		sig, fs = librosa.load(path + '/' + name, sr=44100)
		#normalize signal
		data = normalize(sig)
		if (len(data) < seg_len):
			pad_len = int(seg_len - len(data))
			pad_rem = int(pad_len % 2)
			pad_len /= 2
			signal = np.pad(data, (int(pad_len), int(pad_len + pad_rem)), 'constant', constant_values=0)
		elif (len(data) > seg_len):
			signal = []
			end = seg_len
			st = 0
			while (end < len(data)):
				signal.append(data[st:end])
				st = st + seg_ov
				end = st + seg_len
			signal = np.array(signal, dtype='float32')
			if (end >= len(data)):
				num_zeros = int(end - len(data))
				if (num_zeros > 0):
					n1 = np.array(data[st:end], dtype='float32')
					n2 = np.zeros([num_zeros], dtype='float32')
					s = np.concatenate([n1, n2], 0)
				else:
					s = np.array(data[int(st):int(end)])
			signal = np.vstack([signal, s])
		else:
			signal = data

		if (signal.ndim > 1):
			for i in range(signal.shape[0]):
				x_train.append(signal[i])
				y_train.append(name[0])
		else:
			x_train.append(signal)
			y_train.append(name[0])
		count += 1
		print("Data segment:",len(x_train))
	x_ = np.float32(x_train)
	np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newxtrain.npy", x_)
	y_ = string2num(y_train)
	np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newytrain.npy", y_)
	print("dataloading finish")
	return x_, y_

def testdataload(path, fold):
	print("dataloading……")
	x_train = []
	y_train = []
	data_seg = []
	data_seg.append(0)
	sample_name = []
	fnames = os.listdir(path)
	count = 1
	for name in fnames:
		sample_name.append(name[0:4])
		print("Data num:",count)
		sig, fs = librosa.load(path + '/' + name, sr=44100)
		# normalize signal
		data = normalize(sig)
		if (len(data) < seg_len):
			pad_len = int(seg_len - len(data))
			pad_rem = int(pad_len % 2)
			pad_len /= 2
			signal = np.pad(data, (int(pad_len), int(pad_len + pad_rem)), 'constant', constant_values=0)
		elif (len(data) > seg_len):
			signal = []
			end = seg_len
			st = 0
			while (end < len(data)):
				signal.append(data[st:end])
				st = st + seg_ov
				end = st + seg_len
			signal = np.array(signal, dtype='float32')
			if (end >= len(data)):
				num_zeros = int(end - len(data))
				if (num_zeros > 0):
					n1 = np.array(data[st:end], dtype='float32')
					n2 = np.zeros([num_zeros], dtype='float32')
					s = np.concatenate([n1, n2], 0)
				else:
					s = np.array(data[int(st):int(end)])
			signal = np.vstack([signal, s])
		else:
			signal = data

		if (signal.ndim > 1):
			for i in range(signal.shape[0]):
				x_train.append(signal[i])
				y_train.append(name[0])
		else:
			x_train.append(signal)
			y_train.append(name[0])
		count += 1
		print("Data segment:",len(x_train))
		data_seg.append(len(x_train))
	x_ = np.float32(x_train)
	np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newxtest.npy", x_)
	y_ = string2num(y_train)
	np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newytest.npy", y_)
	data_seg = np.array(data_seg)
	np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newdataseg.npy", data_seg)
	sample_name = np.array(sample_name)
	np.save("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(fold) + "\\newsamplename.npy", sample_name)
	print("dataloading finish")

	return x_, y_, data_seg, sample_name
