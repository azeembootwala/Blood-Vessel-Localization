# This python code will pick the US images from the directory and run different pre-processing operations
# For this purpose we need a special python generator function to load the images from disk to memory sequentially
# We have two functions here , which have intrinsically the same structure
# traindatagen picks up training images and labels and throws them into our CNN_vessel.py file
# testdatagen picks up validation images and throws them into our inference.py file along with their names as its needed to save the images
import numpy as np
import cv2
import os
import pandas
import scipy.io
from sklearn.utils import shuffle

def traindatagen(batch_size=16):
    i=0
    j=0
    os.chdir("/directory for trained images/") # Directory where training images are being stored
    my_list = os.listdir()
    max_iter = int(round(len(my_list),-2)) # We only want the iterations to be run till 110400 so we round it
    X = np.zeros((batch_size,512,512,1))
    Y = np.zeros((batch_size,4))
    while True:
        if i == batch_size:
            i=0
            yield X,Y # the yield keyword makes it the generator function
        os.chdir("/diirectory for trained images/")
        img = cv2.imread(my_list[j])
        img = img.mean(axis=2) # We take the mean across the second axis (python indexing starts from 0 so 0,1,2 ) to convert the images size from 512x512x3 into 512x512x1
        img = img/255 # We normalize the images by dividing by 255
        X[i,:,:,0] = img
        os.chdir("/directory where ground truth bounding box is present/")
        txt_file = my_list[j].replace(".png",".txt") # we take the associated labels in the corresponding .txt files
        for line in open(txt_file):
            row = line.split(" ")
            Y[i,0]=row[4]
            Y[i,1]=row[5]
            Y[i,2]=row[6]
            Y[i,3]=row[7]
        if j==int(max_iter):
            j=-1
            my_list = shuffle(my_list)
        i+=1
        j+=1


def testdatagen(batch_size=16):
    i=0
    j=0
    listname = []
    os.chdir("/directory for validation images/") # Directory where images are being stored
    my_list = os.listdir()
    #max_iter = int(round(len(my_list)-2))
    max_iter = (len(my_list)//batch_size)*batch_size
    X = np.zeros((batch_size,512,512,1))
    Y = np.zeros((batch_size,4))
    while True:
        if i == batch_size:
            i=0
            name_list = listname
            listname=[]
            yield X,Y,name_list
        os.chdir("/directoy for validation images/")
        img = cv2.imread(my_list[j])
        listname.append(my_list[j])
        img = img.mean(axis=2) # We take the mean across the second axis (python indexing starts from 0 so 0,1,2 ) to convert the images size from 512x512x3 into 512x512x1
        img = img/255
        X[i,:,:,0] = img
        os.chdir("/directory for ground truth data on validation images/")
        txt_file = my_list[j].replace(".png",".txt")
        for line in open(txt_file):
            row = line.split(" ")
            Y[i,0]=row[4]
            Y[i,1]=row[5]
            Y[i,2]=row[6]
            Y[i,3]=row[7]
        if j ==int(max_iter):
            j = -1
            my_list =shuffle(my_list)
        i+=1
        j+=1
