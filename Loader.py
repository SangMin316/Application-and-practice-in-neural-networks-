import sys
sys.path.append('/home/smjo/DDG/')
import torch
import numpy as np
from torch.utils.data import Dataset

# Sleep Stage Classification
class Sleepedf_dataset(Dataset):
    def __init__(self, files):
        self.files = files

    # unify stage 3,4
    def one_hot_encoding(self,y):
        if y == '1':
          y = torch.tensor([0,1,0,0,0]).type(torch.float32)
        elif y == '2':
          y = torch.tensor([0,0,1,0,0]).type(torch.float32)
        elif y == '3':
          y = torch.tensor([0,0,0,1,0]).type(torch.float32)
        elif y == '4':
          y = torch.tensor([0,0,0,1,0]).type(torch.float32)
        elif y == 'R':
          y = torch.tensor([0,0,0,0,1]).type(torch.float32)
        else: # wake, move
          y = torch.tensor([1,0,0,0,0]).type(torch.float32)
        return y

    def one_hot_subject_encoding(self,s):
        s = int(s)
        z = torch.zeros(20)
        z[s] = 1
        return z

    def __getitem__(self, index):
        sample = np.load(self.files[index])
        y = self.one_hot_encoding(sample['y'])
        s = self.one_hot_subject_encoding(self.files[index].split('/')[-2])
        return {'x': torch.tensor(sample['x']).type(torch.float32),
                'y': y,
                's': s.type(torch.float32)}

    def __len__(self):
        return len(self.files)