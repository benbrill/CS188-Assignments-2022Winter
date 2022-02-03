import os
import numpy as np
import cv2
import re
import pickle

with open('categories_tinyplaces.txt') as f:
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]
categories = [re.sub(r'\s[0-9]+', '', line) for line in lines]
category_nums = [re.findall(r'[0-9]+', line)[0] for line in lines]

def load_train_labels():
    with open(f'train.txt') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    lines = [re.sub(r'\s[0-9]+', '', line) for line in lines]
    labs = []
    for cat in categories:
        count = 0
        for line in lines:
            if cat in line:
                labs.append(line)
                count += 1
            if count == 500:
                break
    labs = ["images/" + lab for lab in labs]
    return labs

def load_val_labels():
    with open('val.txt') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    labs = []
    for num in category_nums:
        count = 0
        for line in lines:
            if num in line[-2:]:
                labs.append(line[:-3])
                count += 1
            if count == 50:
                break
    labs = ["images/" + lab for lab in labs]
    return labs

def create_image_data(data, num_labels):
    if data == "train":
        labels = load_train_labels()
    elif data == "val":
        labels = load_val_labels()
    image_array = []
    for img in labels:
        img = cv2.imread(img)
        img = cv2.resize(img, (32, 32))
        r = img[:,:,0] #Slicing to get R data
        g = img[:,:,1] #Slicing to get G data
        b = img[:,:,2] #Slicing to get B data
        image_array.append(np.array([[r] + [g] + [b]], np.uint8))
    labels = np.repeat(np.arange(num_labels), len(labels) / num_labels)
    d = {'data': np.array(image_array), 'label': labels}
    return d

if "__main__" == __name__:
    train_data = create_image_data("train", 2)
    val_data = create_image_data("val", 2)
    pickle.dump(train_data, open("tinyplaces-train.p", "wb"))
    pickle.dump(val_data, open("tinyplaces-val.p", "wb"))
    train_data = create_image_data("train", 20)
    val_data = create_image_data("val", 20)
    pickle.dump(train_data, open("tinyplaces-train-multiclass.p", "wb"))
    pickle.dump(val_data, open("tinyplaces-val-multiclass.p", "wb"))

