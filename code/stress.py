import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import MyDataset
import torch.optim as optim
from model import Encoder_Decoder_stress
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# パラメータ
batchsize = 10
lr = 0.001
epochs = 100


# datasetをloadする
pre_geometry_dataset = np.load('../data/dataset/pre_geometry.npy')
geometry_dataset = np.load('../data/dataset/geometry.npy')
contact_dataset = np.load('../data/dataset/contact.npy')
stress_dataset = np.load('../data/dataset/stress.npy')

# dataを[1280,3,32,32],labelを[1280,1,32,32]に変形
data = np.stack([pre_geometry_dataset, geometry_dataset, contact_dataset], axis=1)
label = np.reshape(stress_dataset, [stress_dataset.shape[0], -1, stress_dataset.shape[1], stress_dataset.shape[2]])

# 9:1にバリデーションする
tr_data, va_data, tr_label, va_label = train_test_split(data, label, test_size=0.1, random_state=1, shuffle=True)

# resultに保存する
np.save('result/stress/tr_data', tr_data)
np.save('result/stress/tr_label', tr_label)
np.save('result/stress/va_data', va_data)
np.save('result/stress/va_label', va_label)

# 学習データをイテレータにする
dataset = MyDataset(tr_data, tr_label)
trainloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)

# テストデータをイテレータにする
dataset = MyDataset(va_data, va_label)
validloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)


# 訓練用関数
def train(model, device, criterion, optimizer, trainloader):
    model.train()
    running_loss = 0

    for data, label in trainloader:
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_loss += loss.item()

    return running_loss/len(trainloader)


# 評価用関数
def valid(model, device, criterion, validloader):
    model.eval()
    running_loss = 0

    for data, label in validloader:
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, label)
            running_loss += loss.item()

    return running_loss/len(validloader)


# モデル，評価関数，最適化関数を呼び出す
model = Encoder_Decoder_stress(inputDim=3, outputDim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

# 指定したエポック数だけ学習指せる
tr_loss = []
va_loss = []

for epoch in range(1, 1+epochs):
    loss = train(model, device, criterion, optimizer, trainloader)
    tr_loss.append(loss)

    loss = valid(model, device, criterion, validloader)
    va_loss.append(loss)

    print(str(epoch)+'epoch通過')

else:
    torch.save(model, 'result/stress/model.pth')


# lossの推移をグラフにする
x = [i for i in range(epochs)]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, tr_loss, label='tr_loss')
ax.plot(x, va_loss, label='va_loss')
ax.set_xlabel('epoch')
ax.set_ylabel('MSE_loss')
ax.legend(loc='upper right')
fig.savefig('result/stress/loss.png')
plt.show()