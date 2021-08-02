import numpy as np
import os
import librosa


classes = ['h','p'] #  2 classes
dict = {classes[0]:42,classes[1]:31}

seg_len = 16000 # signal split length (in samples) in time domain
seg_ov = int(seg_len*0.8) # 20% overlap

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

def traindataload(path):
	print("dataloading……")
	x_train = []
	y_train = []
	fnames = os.listdir(path)
	count = 1
	for name in fnames:
		print("Data num:",count)
		sig, fs = librosa.load(path + '/' + name, sr=11025) #可以降采样
		#normalize signal
		data = normalize(sig)#不normalize会好吗？
		#data = sig
		if (len(data) < seg_len):  #data长度小于seg_len
			pad_len = int(seg_len - len(data))
			pad_rem = int(pad_len % 2)
			pad_len /= 2
			signal = np.pad(data, (int(pad_len), int(pad_len + pad_rem)), 'constant', constant_values=0)
		elif (len(data) > seg_len):
			signal = []
			end = seg_len  #16000
			st = 0
			while (end < len(data)):
				signal.append(data[st:end])
				st = st + seg_ov  #seg_ov = 70% * 16000
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
	np.save(r'F:\Parkinson speech\dataset\samplextrain.npy', x_)
	y_ = string2num(y_train)
	np.save(r'F:\Parkinson speech\dataset\sampleytrain.npy', y_)
	print("dataloading finish")
	return x_, y_

def testdataload(path):
	print("dataloading……")
	x_train = []
	y_train = []
	fnames = os.listdir(path)
	count = 1
	for name in fnames:
		print("Data num:",count)
		sig, fs = librosa.load(path + '/' + name, sr=11025) #可以降采样
		# normalize signal
		data = normalize(sig)
		if (len(data) < seg_len):  #data长度小于16000
			pad_len = int(seg_len - len(data))
			pad_rem = int(pad_len % 2)
			pad_len /= 2
			signal = np.pad(data, (int(pad_len), int(pad_len + pad_rem)), 'constant', constant_values=0)
		elif (len(data) > seg_len):
			signal = []
			end = seg_len  #16000
			st = 0
			while (end < len(data)):
				signal.append(data[st:end])
				st = st + seg_ov  #seg_ov = 70% * 16000
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
	np.save(r'F:\Parkinson speech\dataset\samplextest.npy', x_)
	y_ = string2num(y_train)
	np.save(r'F:\Parkinson speech\dataset\sampleytest.npy', y_)
	print("dataloading finish")
	return x_, y_

