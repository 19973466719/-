flag = 'predict' 

seed =2022

# the path of result
model_path = f'../runs/result in {flag} {seed}'  

print(f'the result save in {flag}{seed}.csv')

import sys
import time
from datetime import datetime
from pathlib import Path
from glob import glob 

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm_notebook as tqdm
def output(model: nn.Module, test_loader, device, show=True):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    seqs = []
    with torch.no_grad():
        for idx, (*x, y) in tqdm(test_loader):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)

            y_hat = model(*x)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
            seqs.append(idx)
            
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    seqs = np.concatenate(seqs).reshape(-1)

    return seqs,targets,outputs
from dataset import MyDataset
from model import DeepDTAF
class DatasetV11:
    def __init__(self,*a,**b):
        self.dataset = MyDataset(*a,**b)
        
    def __getitem__(self, index):
        return self.dataset.seq_path[index].name.split('.')[0], self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
max_seq_len = 1000

max_pkt_len = 63

max_smi_len = 150
# seed = 7

# gpu
device = torch.device("cuda:0")
i = glob(model_path)
assert len(i)==1
path = Path(i[0])
SHOW_PROCESS_BAR = False
data_path = '../data/'


torch.manual_seed(seed)
np.random.seed(seed)
batch_size = 16
n_epoch = 20
interrupt = None
save_best_epoch = 13 

assert 0<save_best_epoch<n_epoch

loss_function = nn.MSELoss()

model = DeepDTAF()
model.load_state_dict(torch.load(path / 'best_model.pt',map_location=device))
model.to(device);
data_loaders = {phase_name:
                    DataLoader(DatasetV11(data_path, phase_name,
                                         max_seq_len, max_pkt_len, max_smi_len, pkt_window=None, pkt_stride=None),
                               batch_size=batch_size,
                               pin_memory=True,
                               num_workers=8,
                               shuffle=True)
                for phase_name in ['training', 'validation', 'test']}

for p in ['training', 'validation', 'test']:
    print(f'{flag}{seed}_{p}.csv')
    t,o,n = output(model,data_loaders[p],device)
    a = pd.DataFrame()
    a=a.assign(pdbid=t,predicted=n,real=o,set=p)
    a.to_csv(f'{flag}{seed}_{p}.csv')