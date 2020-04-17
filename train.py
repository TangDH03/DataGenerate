from loader import datas
from torch.utils.data import  DataLoader
from net import AUTOMAP
from torch import nn,optim,min,max
from cv2 import imwrite
import numpy as np
def main():
    dataset = datas()
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)
    net = AUTOMAP([1,128,128],[1,128,128])
    criterion = nn.MSELoss()
    optimzer = optim.RMSprop(net.parameters(),lr=0.00002)
    iter = 0
    proces = 0
    while iter<50:
        for img,label in dataloader:
            optimzer.zero_grad()
            predict = net.forward(img)
            predict = (predict-min(predict)) / (max(predict)-min(predict))
            loss = criterion(predict,label)
            print(loss.data.item())
            loss.backward()
            optimzer.step()
            if proces%10==0:
                imwrite("./result/%d.png" % proces, predict[0].detach().squeeze().numpy())
            proces+=1
        iter+=1

main()