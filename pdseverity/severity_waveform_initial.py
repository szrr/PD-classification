import numpy as np
import os
import librosa

seg_len = 65536 # signal split length (in samples) in time domain
seg_ov = int(seg_len*0.5) # 50% overlap

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

def train_wave_initial(path, fold):
	#print("dataloading……")
	x_train = []
	y_train = []
	fnames = os.listdir(path)
	count = 1
	for name in fnames:
		print("Data num:",count)
		print(name, "label", name[-5])
		sig, fs = librosa.load(path + '/' + name, sr=44100) #可以降采样
		#normalize signal
		data = normalize(sig)
		#data = sig
		if (len(data) < seg_len):  #data长度小于seg_len
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
				st = st + seg_ov  #seg_ov = 30% * 16000
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
				y_train.append(name[-5])
		else:
			x_train.append(signal)
			y_train.append(name[-5])
		count += 1
		print("Data segment:",len(x_train))
	x_ = np.float32(x_train)
	np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\xtrain.npy", x_)
	y_ = relable(y_train)
	y_ = np.float32(y_)
	np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\ytrain.npy", y_)
	#print("dataloading finish")
	return x_, y_

def test_wave_initial(path, fold):
	#print("dataloading……")
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
		sig, fs = librosa.load(path + '/' + name, sr=44100) #可以降采样
		# normalize signal
		data = normalize(sig)
		#data = sig
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
				st = st + seg_ov  #seg_ov = 30% * 16000
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
				y_train.append(name[-5])
		else:
			x_train.append(signal)
			y_train.append(name[-5])
		count += 1
		print("Data segment:",len(x_train))
		data_seg.append(len(x_train))
	x_ = np.float32(x_train)
	np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\xtest.npy", x_)
	y_ = relable(y_train)
	y_ = np.float32(y_)
	np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\ytest.npy", y_)
	data_seg = np.array(data_seg)
	np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\dataseg.npy", data_seg)
	sample_name = np.array(sample_name)
	np.save("F:\Parkinson speech\dataset\\multi class five fold\\" + str(fold) + "\\samplename.npy", sample_name)
	print("dataloading finish")

	return x_, y_, data_seg, sample_name
