import sys
# sys.path.append('/home/smjo/DDG/')
import glob
import os
from Loader import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import Model
from Loss import Total_Loss

# deivce
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

loss = Total_Loss(device)


SleepEDF_path ='/DataCommon2/smjo/Sleep/SleepEDF/100Hz_EEG_WO30/*'
Data_Path = sorted(glob.glob(SleepEDF_path))[:5]
Train = Data_Path[:4]
Test = Data_Path[-1]

train = []
for i in Train:
    train.extend(glob.glob(i + "/*"))

train, val = train_test_split(train, test_size=0.2, random_state= 777)
epochs = 150
learning_rate = 0.0005
batch_size = 128


train_dataset = Sleepedf_dataset(train)
val_dataset = Sleepedf_dataset(val)


trainLoader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 4)
valLoader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers = 4)
model = Model.Base().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)



print('-----learning-------')
print(len(trainLoader))
loss_tr = []
loss_val = []
acc_tr = []
acc_val = []
for epoch in range(epochs):
    model.train()
    loss_ep = 0  # add batch loss in epoch
    acc_ep = 0
    for batch in tqdm(trainLoader):
        optimizer.zero_grad()
        data = batch['x'].to(device)
        y = batch['y'].to(device)
        s = batch['s'].to(device)
        optimizer.zero_grad()
        loss_batch, acc_batch, _, _ = loss.forward(data, y, s, model, train=True)
        optimizer.step()

        acc_batch = acc_batch / batch['x'].shape[0]  # acc/batch
        loss_ep += loss_batch.item()
        acc_ep += acc_batch

    loss_tr.append((loss_ep) / (len(trainLoader)))
    acc_tr.append((acc_ep) / (len(trainLoader)))

    loss_ep_val = 0
    acc_ep_val = 0

    with torch.no_grad():
        model.eval()
        for batch_idx, batch in tqdm(enumerate(valLoader)):
            data = batch['x'].to(device)
            y = batch['y'].to(device)
            s = batch['s'].to(device)
            loss_batch, acc_batch, _, _= loss.forward(data, y, s, model, train=False)

            acc_batch = acc_batch / batch['x'].shape[0]  # acc/batch

            loss_ep_val += loss_batch.item()
            acc_ep_val += acc_batch

        loss_val.append((loss_ep_val) / len(valLoader))
        acc_val.append((acc_ep_val) / len(valLoader))
    print("epoch : ", epoch, "  train loss : ", loss_tr[epoch], 'train acc : ', acc_tr[epoch], "    val loss : ",
          loss_val[epoch], 'val acc : ', acc_val[epoch])

    ad = '/DataCommon2/smjo/ANN/' + str(acc_val[epoch])[:5] + '_' + str(epoch)
    if not os.path.exists(ad):
        os.makedirs(ad)
    torch.save(model, ad + '/Model' + str(epoch) + '.pt')

# save result
col = ['loss_tr', 'loss_val', 'acc_tr', 'acc_val']
data = np.array([loss_tr,
                 loss_val,
                 acc_tr,
                 acc_val])
print(data.shape)
data = np.transpose(data)
df = pd.DataFrame(data=data, columns=col)
# df.to_excel('/DataCommon2/smjo/SSL/6data/SleepEDF20/2s/result1/'+'SleepEDF.xlsx', index = False)

plt.plot(range(epochs), loss_tr, color='red')
plt.plot(range(epochs), loss_val, color='blue')
plt.title('Model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# plt.figure(figsize =(15, 10))
plt.plot(range(epochs), acc_tr, color='red')
plt.plot(range(epochs), acc_val, color='blue')
plt.title('Model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')