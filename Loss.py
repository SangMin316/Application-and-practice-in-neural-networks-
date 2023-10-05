import torch
from torch import nn

class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X,device):
        L2_distances = torch.cdist(X, X) ** 2
        over = self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(device)
        return torch.exp(-L2_distances[None, ...] / over[:, None, None]).sum(dim=0)



class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y,device):
        K = self.kernel(torch.vstack([X, Y]),device)
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY



class Total_Loss(torch.nn.modules.loss._Loss):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.mmd_loss = MMDLoss()

    def MMD(self,z,s):
        all_Z = []
        for i in range(4):
            label = torch.zeros(z.shape[0], 20)
            label[:, i] = 1
            label = label.type(torch.bool)
            label_index = s[label]
            label_index = torch.nonzero(label_index).to(self.device)
            subject_z = z[label_index]

            all_Z.append(torch.squeeze(subject_z))

        mmdloss = []
        for i in range(len(all_Z)):
            for j in range(len(all_Z)):
                if j >> i:
                    mmd = self.mmd_loss.forward(all_Z[i],all_Z[j],self.device)
                    mmdloss.append(mmd)
                else:
                    continue
        return  sum(mmdloss)/len(mmdloss)

    def forward(self, x, y, s, encoder, train):
        z, pred = encoder.forward(x)
        loss1 = self.criterion(pred, y)

        if train:
            loss2 = self.MMD(z,s)
            loss = loss1 + loss2

        else:
            loss = loss1
        loss.requires_grad_(True)

        if train:
            loss.backward(retain_graph=True)
        _, label = torch.max(y, 1)
        _, predicted = torch.max(pred, 1)

        acc = (predicted == label).sum().item()

        return loss, acc, label, predicted