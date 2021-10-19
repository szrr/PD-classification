import prettytable as pt
from torch.nn import CrossEntropyLoss
from model.onedcnn import *
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pdseverity.severity_waveform_initial import *


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

def get_initial_addr(x):
    return "F:\Parkinson speech\dataset\\multi class five fold\\" + str(x) + "\\train",\
           "F:\Parkinson speech\dataset\\multi class five fold\\" + str(x) + "\\test"

def get_data(x):
    x_train = np.load("F:\Parkinson speech\dataset\\multi class five fold\\" + str(x) + "\\xtrain.npy")
    y_train = np.load("F:\Parkinson speech\dataset\\multi class five fold\\" + str(x) + "\\ytrain.npy")
    x_test = np.load("F:\Parkinson speech\dataset\\multi class five fold\\" + str(x) + "\\xtest.npy")
    y_test = np.load("F:\Parkinson speech\dataset\\multi class five fold\\" + str(x) + "\\ytest.npy")
    seg = np.load("F:\Parkinson speech\dataset\\multi class five fold\\" + str(x) + "\\dataseg.npy")
    sample_name = np.load("F:\Parkinson speech\dataset\\multi class five fold\\" + str(x) + "\\samplename.npy")
    return x_train, y_train, x_test, y_test, seg, sample_name

if __name__ == '__main__':
    fold = 1
    # initial audio segments and relevant information of fold(x)
    train_addr, test_addr = get_initial_addr(fold)
    x_train, y_train = train_wave_initial(train_addr, fold)
    x_test, y_test, seg, sample_name = test_wave_initial(test_addr, fold)

    x_train, y_train, x_test, y_test, seg, sample_name = get_data(fold)
    print("dataloading finish")

    device = torch.device('cuda:0')
    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    x_test = torch.from_numpy(x_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)
    x_train = x_train.unsqueeze(1)
    y_train = y_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    batch_size = 64
    epoch = 100

    train_dataset = MyDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = onedcnn_severity().to(device)
    #model = aggregation_onedcnn_severity().to(device)

    adam = optim.Adam(model.parameters(), lr=0.001)
    #weight_CE = torch.FloatTensor([1, 1, 2]).to(device)
    cost = CrossEntropyLoss()
    #cost = CrossEntropyLoss(weight=weight_CE)

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
            seg0_num = 0
            seg1_num = 0
            seg2_num = 0
            for i in range(seg[idx], seg[idx + 1]):
                predict_y = model((x_test[i:i + 1, :, :]).float()).detach()
                predict_ys = np.argmax(predict_y.cpu(), axis=-1)
                label_np = (y_test[i:i + 1, :]).cpu().numpy()
                label_np = label_np.reshape(label_np.size)
                label_np = torch.from_numpy(label_np)
                corr = predict_ys == label_np
                corr = np.sum(corr.numpy())
                corr_seg_num += corr
                seg0 = predict_ys == 0
                seg1 = predict_ys == 1
                seg2 = predict_ys == 2
                seg0 = np.sum(seg0.numpy())
                seg1 = np.sum(seg1.numpy())
                seg2 = np.sum(seg2.numpy())
                seg0_num += seg0
                seg1_num += seg1
                seg2_num += seg2
            seg_correct += corr_seg_num
            seg_acc = corr_seg_num / (seg[idx + 1] - seg[idx])
            seg_acc = round(seg_acc, 4)
            sample_seg_acc.append(seg_acc)
            if ((corr_seg_num >= seg0_num) and (corr_seg_num >= seg1_num) and (corr_seg_num >= seg2_num)):
                sample_correct += 1
        sample_seg_acc = np.array(sample_seg_acc)
        seg_acc_table = pt.PrettyTable()
        seg_acc_table.field_names = sample_name
        seg_acc_table.add_row(sample_seg_acc)
        print(seg_acc_table)
        print("Seg correct num:" + str(seg_correct) + "  Seg num:" + str(seg_sum))
        print('Seg correct acc: {:.8f}'.format(seg_correct/seg_sum))
        print("Sample correct num:" + str(sample_correct) + "  Sample num:" + str(sample_sum))
        print('Sample accuracy: {:.4f}'.format(sample_correct / sample_sum))
        #torch.save(model, 'F:\Parkinson speech\param\\1DCNN severity\\1\\new_1dcnn_model2{:.8f}.pkl'.format(seg_correct/seg_sum))
