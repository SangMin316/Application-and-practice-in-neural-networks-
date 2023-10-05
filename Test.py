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
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import Model
from Loss import Total_Loss

# deivce
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

loss = Total_Loss(device)


SleepEDF_path ='/DataCommon2/smjo/Sleep/SleepEDF/100Hz_EEG_WO30/*'
Data_Path = sorted(glob.glob(SleepEDF_path))[:5]
Test = Data_Path[-1]
test = []
test.extend(glob.glob(Test + "/*"))
batch_size = 128
test_dataset = Sleepedf_dataset(test)

testLoader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers = 4)
model = torch.load('/DataCommon2/smjo/ANN/0.861_90/Model90.pt')


print('-----learning-------')
print(len(testLoader))
loss_ep_test = 0
acc_ep_test = 0
label_list = []
predicted_list = []

with torch.no_grad():
    model.eval()
    for batch in tqdm(testLoader):
        data = batch['x'].to(device)
        y = batch['y'].to(device)
        s = batch['s'].to(device)
        loss_batch, acc_batch,label, predicted = loss.forward(data, y, s, model, train=True)

        acc_batch = acc_batch / batch['x'].shape[0]  # acc/batch
        loss_ep_test += loss_batch.item()
        acc_ep_test += acc_batch
        label_list.extend(label.cpu().detach().numpy())
        predicted_list.extend(predicted.cpu().detach().numpy())

print('acc:', acc_ep_test / len(testLoader))




conf_matrix = confusion_matrix(label_list, predicted_list)
print(conf_matrix)
print('micro F1:',f1_score(label_list, predicted_list, average='micro'))
print('macro F1:',f1_score(label_list, predicted_list, average='macro'))
print('weighted F1:',f1_score(label_list, predicted_list, average='weighted'))
print('kappa_score',cohen_kappa_score(label_list,predicted_list))
# print('recall',recall_score(label_list, predicted_list))
# print('precision',precision_score((label_list, predicted_list)))
target_names = ['W', 'N1', 'N2','N3',"R"]
print(classification_report(label_list,predicted_list,target_names= target_names, digits= 5))
categories = ['Wake', 'N1', 'N2', 'N3', 'REM']



title = 'Confusion Matrix in Test (SleepEDF20)'
cmap = plt.cm.Blues

A = np.zeros((5,5))
for i in range(conf_matrix.shape[0]):
  print(sum(conf_matrix[i,:]))
  A[i,:] = conf_matrix[i,:]/float(sum(conf_matrix[i,:]))

conf_matrix = A

plt.figure(figsize=(7, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)  # , cmap=plt.cm.Greens
# plt.title(title, size=15, fontweight="bold")
# plt.colorbar(fraction=0.05, pad=0.05).set_label(label = 'accuracy %',size=13)
plt.colorbar(fraction=0.05, pad=0.05)

tick_marks = np.arange(5,5)
plt.yticks(np.arange(5), ('W','N1','N2','N3','R'))
plt.xticks(np.arange(5), ('W','N1','N2','N3','R'))
plt.xlabel('Predicted Labels',size=13)
plt.ylabel('True Labels', size=13)

fmt = '.2%'
thresh = 1
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 ha="center", va="center", color="black" if conf_matrix[i, j] > thresh else "black")  #horizontalalignment
plt.savefig('SleepEDF_conv2s_confusion matrix.png',bbox_inches = 'tight', pad_inches=0)
plt.show()