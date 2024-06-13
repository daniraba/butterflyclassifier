import os # Operating system
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from INeuralNetwork import INeuralNetwork, preprocess
from PIL import Image

n=INeuralNetwork()
n.load_state_dict(torch.load('Ins.pth'))


test_dir="Test/"
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
    dir_path = os.path.join(test_dir, types[i])
    files = os.listdir(dir_path)
    remove_list = []
    for file in files:
        if is_bad_file(dir_path + '/' + file):
            remove_list.append(file)

    for r in remove_list:
        files.remove(r)

    file_lists.append(files)



total = 0
correct = 0

total_labels = [0, 0, 0]
total_correct = [0, 0, 0]

for i in range(10):
    for label in range(num_classes):
        dir = test_dir + types[label] + '/'
        file_list = file_lists[label]
        file_name = file_list[i]
        
        f = dir + file_name
        print(i, 100, f)

        img = preprocess(f)

        output = n.forward(img).detach().numpy() # Getting the output of the network

        guess = np.argmax(output) # Getting the highest value from the results of the output

        total += 1 # Seen an image

        total_labels[label] += 1 # Of that class

        if guess == label: # Getting it correct
            correct += 1
            total_correct[label] += 1

print("Accuracy:", correct / total)
print(total_labels)
print(total_correct)


        

torch.save(n.state_dict(),"Ins.pth")
print("End:", datetime.now())
