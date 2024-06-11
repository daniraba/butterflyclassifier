import os
import torch
import numpy as np
from PIL import Image
from INeuralNetwork import INeuralNetwork

n=INeuralNetwork()
n.load_state_dict(torch.load("Ins.pth"))



def preprocess(f):
    image=Image.open(f)
    image=image.resize((50,50))
    #print(image.size)
    a=np.array(image)/255.0
    #print(a.shape)
    a=a.reshape(50*50*3)
    return a

test="Test/"
types=["Butterfly Test","Grasshopper Test","Ladybug Test","Dragonfly Test","Mosquito Test"]

correct = 0
total = 0
for i in range(len(types)):
    directory=test+types[i]
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
        output=n.forward(img).detach().numpy()

        #print(output)
        
        total += 1
        guess=np.argmax(output)
       
        if guess==label:
            correct+=1

print("Accuracy:",correct/total)
