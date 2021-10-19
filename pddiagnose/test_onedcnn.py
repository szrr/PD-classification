import prettytable as pt
from torch.nn import CrossEntropyLoss
from model.onedcnn import *
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pddiagnose.diagnose_waveform_initial import *

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

#get fold(x) 'train' and 'test' data address
def getaddr(x):
    return "F:\Parkinson speech\dataset\\five fold cross validation\\" + str(x) + "\\train",\
           "F:\Parkinson speech\dataset\\five fold cross validation\\" + str(x) + "\\test"

#
def get_waveform_addr(x):
    x_train = np.load("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(x) + "\\newxtrain.npy")
    y_train = np.load("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(x) + "\\newytrain.npy")
    x_test = np.load("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(x) + "\\newxtest.npy")
    y_test = np.load("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(x) + "\\newytest.npy")
    seg = np.load("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(x) + "\\newdataseg.npy")
    sample_name = np.load("F:\Parkinson speech\dataset\\five fold cross validation\\" + str(x) + "\\newspecsamplename.npy")
    return x_train, y_train, x_test, y_test, seg, sample_name

if __name__ == '__main__':
    fold = 1
    #initial audio segments and relevant information of fold(x)
    train_addr, test_addr = getaddr(fold)
    x_train, y_train = traindataload(train_addr, fold)
    x_test, y_test, seg, sample_name = testdataload(test_addr, fold) #seg: the number of segments per sample

    #load initialled audio segments and relevant information of fold(x)
    x_train, y_train, x_test, y_test, seg, sample_name = get_waveform_addr(fold)
    print("dataloading finish")

    device = torch.device('cuda:0')
    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    x_test = torch.from_numpy(x_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    #dimensional adjustment of data
    x_train = x_train.unsqueeze(1)
    y_train = y_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    batch_size = 64
    epoch = 100

    train_dataset = MyDataset(x_train, y_train)  # create training dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = onedcnn().to(device)
    #model = aggregation_onedcnn().to(device)

    adam = optim.Adam(model.parameters(), lr=0.001)
    cost = CrossEntropyLoss()

    for _epoch in range(epoch):
        avg_loss = 0
        count = 0
        for idx, (train_x, train_label) in enumerate(train_loader):
            adam.zero_grad()
            y_tr = train_label.squeeze(1)
            predict_y = model(train_x.float())
            loss = cost(predict_y, y_tr.long())
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            adam.step()
            count += 1
        #sample accuracy
        sample_correct = 0
        sample_sum = seg.shape[0] - 1
        #segment accuracy
        seg_correct = 0
        seg_sum = seg[-1]
        #single sample segment accuracy
        sample_seg_acc = []
        print("testing epoch: ", _epoch + 1)

        for idx in range(seg.shape[0] - 1):
            corr_seg_num = 0
            for i in range(seg[idx], seg[idx + 1]):
                predict_y = model((x_test[i:i + 1, :, :]).float()).detach()
                predict_ys = np.argmax(predict_y.cpu(), axis=-1)
                label_np = (y_test[i:i + 1, :]).cpu().numpy()
                label_np = label_np.reshape(label_np.size)
                label_np = torch.from_numpy(label_np)
                corr = predict_ys == label_np
                corr = np.sum(corr.numpy())
                corr_seg_num += corr
            seg_correct += corr_seg_num
            seg_acc = corr_seg_num / (seg[idx + 1] - seg[idx])
            seg_acc = round(seg_acc, 4)
            sample_seg_acc.append(seg_acc)
            if ((corr_seg_num * 2) > (seg[idx + 1] - seg[idx])):
                sample_correct += 1
        sample_seg_acc = np.array(sample_seg_acc)
        seg_acc_table = pt.PrettyTable()
        seg_acc_table.field_names = sample_name
        seg_acc_table.add_row(sample_seg_acc)
        print(seg_acc_table)
        print("Seg correct num:" + str(seg_correct) + "  Seg num:" + str(seg_sum))
        print('Seg correct acc: {:.8f}'.format(seg_correct / seg_sum))
        print("Sample correct num:" + str(sample_correct) + "  Sample num:" + str(sample_sum))
        print('Sample accuracy: {:.4f}'.format(sample_correct / sample_sum))
        #torch.save(model, 'F:\Parkinson speech\param\\1DCNN diagnose\\1\\onedimcnn{:.8f}.pkl'.format(seg_correct/seg_sum))
