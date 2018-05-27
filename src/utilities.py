# This script contains weight initialization functions and intersection over union fuction for model evaluation
# They are simply being called into CNN_vessel file
# Author :- Azeem Bootwala
# Date :- Januaray, 2018


import numpy as np


def IOU(Y,T):
    area_Y = (Y[2]-Y[0])*(Y[3]-Y[1])
    area_T = (T[2]-T[0])*(T[3]-T[1])
    area_intersect = (np.minimum(Y[2],T[2])-np.maximum(Y[0],T[0]))*(np.minimum(Y[3],T[3])-np.maximum(Y[1],T[1]))
    area_union = area_Y + area_T -area_intersect
    return area_intersect/area_union


def init_filter(shape, pool_size):
    W = np.random.randn(*shape)/np.sqrt((shape[-1]*np.prod(shape[:2])) + np.prod(shape[:-3])/np.prod(pool_size))
    b = np.zeros(shape[-1])
    return W.astype(np.float32),b.astype(np.float32)

def init_weights_bias(M1, M2):
    W = np.random.randn(M1, M2)/np.sqrt(M1+M2)
    b = np.zeros(M2)
    return W.astype(np.float32),b.astype(np.float32)

def y2ind(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N,K))
    for i in range(0,N):
        ind[i,y[i]]=1
    return ind

def error_rate(a,b):
    return np.mean(a!=b)
