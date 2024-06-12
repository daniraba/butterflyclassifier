import torch 
import torch.nn as nn
import numpy as np
from PIL import Image

device = torch.device('cpu')

FINAL_WIDTH = 50
FINAL_HEIGHT = 50

RESIZE_HEIGHT = 100

LAYER_1 = 1000
LAYER_2 = 500

def preprocess(f):
    image=Image.open(f)
    size=image.size
    width=size[0]
    height=size[1]
    ratio=width/height
    new_height=RESIZE_HEIGHT
    new_width=int(new_height*ratio)
    image=image.resize((new_width,new_height))
    size=image.size
    center_x=size[0]/2
    center_y=size[1]/2
    target_width=FINAL_WIDTH
    target_height=FINAL_HEIGHT
    left=int(center_x-(target_width/2))
    right=left + target_width
    top=int(center_y-(target_height/2))
    bottom=top+target_height
    image=image.crop((left,top,right,bottom))
    a=np.array(image)/255.0
    a=a.reshape(FINAL_HEIGHT*FINAL_WIDTH*3)
    return a

class INeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.model=nn.Sequential(
            nn.Linear(FINAL_HEIGHT * FINAL_WIDTH * 3, LAYER_1),
            nn.Sigmoid(),
            nn.Linear(LAYER_1,LAYER_2),
            nn.Sigmoid(),
            nn.Linear(LAYER_2,4),
            nn.Sigmoid()
        )

        self.loss_function=nn.MSELoss()
        self.optimiser=torch.optim.Adam(self.parameters(),lr=0.0001)

        self.to(device)

    def forward(self,inputs):
        inputs=torch.FloatTensor(inputs).to(device)
        return self.model(inputs)
    
    def train(self,inputs,targets):
        targets=torch.FloatTensor(targets).to(device)
        outputs=self.forward(inputs)

        loss=self.loss_function(outputs,targets)#Error
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


        
    