import os # Operating system
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from INeuralNetwork import INeuralNetwork, preprocess
from PIL import Image

n=INeuralNetwork()


train_dir="Train/"
types=["MALACHITE","MONARCH","PAINTED LADY"]
num_classes = len(types)
file_lists = []

# Check if file is bad
def is_bad_file(f):
    try:
        image = preprocess(f)
        return False
    except:
        return True

for i in range(num_classes):
    dir_path = os.path.join(train_dir, types[i])
    files = os.listdir(dir_path)
    remove_list = []
    for file in files:
        if is_bad_file(dir_path + '/' + file):
            remove_list.append(file)

    for r in remove_list:
        files.remove(r)

    file_lists.append(files)

for i in range(num_classes):
    print(len(file_lists[i]))

stop_at = 100
epochs = 3

for epoch in range(epochs):
    for i in range(stop_at):
        for label in range(num_classes):
            dir = train_dir + types[label] + '/'
            file_list = file_lists[label]
            file_name = file_list[i]
            
            f = dir + file_name
            print(i, stop_at, f)

            img = preprocess(f)
            
            target = np.zeros(num_classes)
            target[label] = 1.0

            n.train(img, target)
    

torch.save(n.state_dict(),"Ins.pth")
print("End:", datetime.now())
