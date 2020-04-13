import os
import cv2
from cv2 import imwrite
from skimage.io import imread
from skimage.transform import radon,iradon,resize
dir_name = './data/deep_lesion/raw/'
has_processed = 0
import numpy as np
def generate(name):
    global has_processed
    if name.split('.')[-1]!='png':
        return
    matrix = imread(name,True)
    # mask = np.full((512,512),32678,dtype=np.uint16)
    # matrix = matrix-mask
    matrix = matrix.astype(np.int32)
    matrix = matrix-32678
    matrix = np.where(matrix>=-1000,matrix,-1000)
    #matrix.resize((256,256))
    imwrite("s%4d.pmg"%has_processed, matrix)
    img = imread(str(has_processed) + ".png",True)
    theta = np.linspace(0.,180,256,endpoint=False)
    radon_img1 = radon(img,theta,circle=True)
    radon_img = 255*(radon_img1/(2*256*256))
    imwrite("r%4d.png"%has_processed,radon_img)
    #ra_img = imread(str(has_processed)+"ra.png",True)
    #recover = iradon(radon_img1,theta,circle=True)
    #cv2.imwrite(str(has_processed)+"reco.png",recover)

    has_processed+=1

def process(name):
    dirs = []
    if os.path.isdir(name):
        dirs = os.listdir(name)
        for dir in dirs:
            if os.path.isdir(name+dir+'/'):
                process(name+dir+'/')
            else:
                generate(name+dir)
    else:
        generate(name)

process(dir_name)

