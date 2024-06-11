import os
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from INeuralNetwork import INeuralNetwork
from PIL import Image

n=INeuralNetwork()


def preprocess(f):
    image=Image.open(f)
    image=image.resize((50,50))
    #print(image.size)
    a=np.array(image)/255.0
    #print(a.shape)
    a=a.reshape(50*50*3)
    return a

train="Project 1/Train/"
types=["Butterfly","Grasshopper","Ladybug","Dragonfly","Mosquito"]

for i in range(len(types)):
    directory=train+types[i]
    c = 0
    for filename in os.listdir(directory):
        f= os.path.join(directory,filename)
        print(i, f, c)
        try:
            if os.path.isfile(f):
                img=preprocess(f)
                label=i
        except:
            print('Skipped')
            continue
    
        target=np.zeros(len(types))
        target[label]=1.0

       

        n.train(img,target)

        c += 1
        if c == 100:
            break

torch.save(n.state_dict(),"Ins.pth")
