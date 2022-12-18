# Import需要的套件
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt

# Read image 利用 OpenCV(cv2) 读入照片并存放在 numpy array 中
def readfile(path):
    # label 是一个 boolean variable, 代表需不需要回传 y 值
    image_dir = sorted(os.listdir(path))  # os.listdir(path)将path路径下的文件名以列表形式读出
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))  # os.path.join(path, file) 路径名合并
        x[i, :, :] = cv2.resize(img, (128, 128))
    return x

# 读取test image的文件名保存为列表返回
def readtestimage(path):
    image_dir = sorted(os.listdir(path))
    imglist = []
    for i, file in enumerate(image_dir):
        imglist.append(file)
    return imglist

# 分别将 training set、testing set 用 readfile 函式读进来
workspace_dir = '../data/BitmojiDataset/'
print("Reading data")
print("...")
train_x = readfile(os.path.join(workspace_dir, "trainimages"))
train_y = pd.read_csv('../data/train.csv').values[:, 1].tolist()
# 将label映射为0或1
for i, iter in enumerate(train_y):
    if iter == -1:
        train_y[i] = 0
train_y = np.array(train_y, dtype=np.uint8)
test_x = readfile(os.path.join(workspace_dir, "testimages"))
print("Reading data complicated")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 配置一下训练设备

''' Dataset '''
# training 时做 data augmentation
# transforms.Compose 将图像操作串联起来
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 随机将图片水平翻转
    transforms.RandomRotation(10),  # 随机旋转图片 (-15,15)
    transforms.RandomResizedCrop(128), # 随机裁剪图片后Resize到（128，128）大小
    transforms.ToTensor(),  # 将图片转成 Tensor, 并把数值normalize到[0,1](data normalization)
])
# testing 时不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:  # 如果没有标签那么只返回X
            return X

''' Model '''


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 维度 [3, 128, 128]
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return self.fc(out)

''' Training '''
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    trainSet = ImgDataset(train_features, train_labels, train_transform)
    testSet = ImgDataset(test_features, test_labels, test_transform)
    train_iter = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(testSet, batch_size=batch_size, shuffle=False)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # loss使用的是交叉熵损失
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            train_pred = net(X.to(DEVICE))
            batch_loss = loss(train_pred, y.to(DEVICE))
            batch_loss.backward()
            optimizer.step()
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == y.numpy())
            train_loss += batch_loss.item()
        train_ls.append(train_loss / len(train_features))

        net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                val_pred = net(X.to(DEVICE))
                batch_loss = loss(val_pred, y.to(DEVICE))

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == y.numpy())
                val_loss += batch_loss.item()
            test_ls.append(val_loss / len(test_features))

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epochs, time.time() - epoch_start_time, \
               train_acc / len(train_features), train_loss / len(train_features), val_acc / len(test_features),
               val_loss / len(test_features)))

    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate([X_train, X_part], 0)
            y_train = np.concatenate([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = Classifier().to(DEVICE)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'折{i + 1}，训练 loss：{float(train_ls[-1]):f}, '
              f'验证 loss：{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

''' K-折交叉验证 '''
# k, num_epochs, lr, weight_decay, batch_size = 5, 120, 0.001, 1e-4, 32
# train_l, valid_l = k_fold(k, train_x, train_y, num_epochs, lr,
#                           weight_decay, batch_size)
# print(f'{k}-折验证: 平均训练 acc: {float(train_l):f}, '
#       f'平均验证 acc: {float(valid_l):f}')

''' 选择调优的超参数训练所有的训练集得到bestModel '''
def train2(num_epochs, lr, weight_decay, batch_size):
    train_val_set = ImgDataset(train_x, train_y, train_transform)
    train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)
    model_best = Classifier().to(DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_best.parameters(), lr=lr, weight_decay = weight_decay)
    num_epoch = num_epochs
    Loss_list = [] # 记录每一个epoch的损失
    Accuracy_list = [] # 记录每一个epoch的正确率
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0

        model_best.train()
        for i, data in enumerate(train_val_loader):
            optimizer.zero_grad()
            train_pred = model_best(data[0].to(DEVICE))
            batch_loss = loss(train_pred, data[1].to(DEVICE))
            batch_loss.backward()
            optimizer.step()
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))
        Accuracy_list.append(train_acc / train_val_set.__len__())
        Loss_list.append(train_loss / train_val_set.__len__())
    torch.save(model_best, 'weights/CNN2.pth')  # 保存整个model的状态
    return Accuracy_list, Loss_list

''' Testing '''
def test(model):
    test_set = ImgDataset(test_x, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model.eval()
    prediction = [] # 记录预测结果
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data.to(DEVICE))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)
    # 将结果映射为1或-1
    for i, iter in enumerate(prediction):
        if iter == 0:
            prediction[i] = -1
    image_id = readtestimage(os.path.join(workspace_dir, "testimages"))
    ans = pd.DataFrame({'image_id': image_id, 'is_male': prediction})
    ans.to_csv('submission.csv', index=False, header=True)
    print("Testing complicated")


''' 定义提取特征图的工具类 '''
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layer):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layer = extracted_layer

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layer:
                outputs.append(x)
        return outputs

# 特征图打印输出
def Feature_visual(outputs):
    feature_imgs = []
    for i in range(len(outputs)):
        out = outputs[i].data.cpu().squeeze().numpy()
        feature_img1 = out[0, :, :].squeeze()  # 选择第一个特征图进行可视化
        feature_img2 = out[1, :, :].squeeze()
        feature_img3 = out[2, :, :].squeeze()
        feature_img4 = out[3, :, :].squeeze()
        feature_img1 = np.asarray(feature_img1 * 255, dtype=np.uint8)
        feature_img2 = np.asarray(feature_img2 * 255, dtype=np.uint8)
        feature_img3 = np.asarray(feature_img3 * 255, dtype=np.uint8)
        feature_img4 = np.asarray(feature_img4 * 255, dtype=np.uint8)
        feature_imgs.append(feature_img1)
        feature_imgs.append(feature_img2)
        feature_imgs.append(feature_img3)
        feature_imgs.append(feature_img4)
    plt.figure(figsize=(14, 14), dpi=100)
    plt.subplot(2, 2, 1)
    plt.imshow(feature_imgs[0], cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(feature_imgs[1], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(feature_imgs[2], cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(feature_imgs[4], cmap='gray')

# 提取特征图函数
def eval(model):
    test_set = ImgDataset(test_x, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            extract_feature = FeatureExtractor(model,None)(data.to(DEVICE))
            Feature_visual(extract_feature)
            break

# 训练曲线打印输出
def plotTrainaccAndLoss(Loss_list, Accuracy_list):
    plt.plot(np.arange(len(Loss_list)), Loss_list,label="train loss")
    plt.plot(np.arange(len(Accuracy_list)), Accuracy_list, label="train acc")
    plt.legend() #显示图例
    plt.xlabel('epoches')
    plt.title('Train acc&loss')
    plt.show()

if __name__ == '__main__':
    train2(num_epochs=120, lr=0.001, weight_decay=1e-4, batch_size=32)
    model = torch.load('./weights/CNN2.pth', map_location=torch.device('cpu'))
    test(model)



