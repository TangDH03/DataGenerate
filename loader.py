from torch.utils.data import Dataset,DataLoader
from skimage.io import imread
import numpy as np
import torch
class datas(Dataset):
    def __init__(self,txt_file="./train.txt",root_dir="./datas/"):
        self.train_file = txt_file
        self.dic = []
        self.root_dir = root_dir
        f = open(self.train_file,"r")
        for line in f:
            vals = line.split(" ")
            tup = (vals[0].replace('\n',''),vals[1].replace('\n',''))
            self.dic.append(tup)
    def __len__(self):
        return len(self.dic)
    def __getitem__(self, index):
        data = imread(self.root_dir+self.dic[index][0],True)
        label = imread(self.root_dir+self.dic[index][1],True)
        data = data.astype(np.float32)
        label = label.astype(np.float32)
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        data = data.view(-1,512,512)
        label = label.view(-1,512,512)
        return data,label

