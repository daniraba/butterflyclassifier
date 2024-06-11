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
    directory=train+types[i] # Iterates through the types
    c = 0
    for filename in os.listdir(directory): # Path to the folder
        f= os.path.join(directory,filename)
        print(i, f, c) # Label, image, count
        try:
            if os.path.isfile(f): # Checks whether it is a file
                img=preprocess(f)
                label=i
        except: # Skip if image is not RGB
            print('Skipped')
            continue
    
        target=np.zeros(len(types))
        target[label]=1.0

       

        n.train(img,target) # Training

        c += 1
        if c == 100: # Training the first 100 files per insect
            break

torch.save(n.state_dict(),"Ins.pth")
