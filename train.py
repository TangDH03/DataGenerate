from loader import datas
from torch.utils.data import  DataLoader
from net import AUTOMAP
from torch import nn,optim
from cv2 import imwrite
def main():
    dataset = datas()
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)
    net = AUTOMAP([512,512],[128,128])
    criterion = nn.MSELoss()
    optimzer = optim.RMSprop(net.parameters())
    iter = 0
    for img,label in dataloader:
        predict = net.forward(img)
        imwrite("./result/%d.png"%iter,predict)
        loss = criterion(predict,label)
        print(loss.data.item())
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        iter+=1

main()