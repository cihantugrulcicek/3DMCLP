#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
basepath = 'C:\\Users\\Abc\\Google Drive\\Academic\\07.Codes\\Python Codes\\Maximal_Covering\\DMCLP\\Instances'


# In[19]:


def read(basepath,filename):
    with open(basepath+filename,'r') as f:
        name = filename[:-4].split('_')
        nbUser = int(name[2])
        nbPeriod = int(name[3])
        trend = name[4]

        p = 0.0

        Y,W,D = {},{},{}
        for i in range(nbUser):
            Y[i] = {}
            D[i] = {}

        line = f.readline()
        while line:
            if 'cost' in line:
                arr = line.split(',')
                p = float(arr[1])
            elif 'User' in line:
                pass
            else:
                arr = line.split(',')
                user = int(arr[0])
                period = int(arr[1])
                w = float(arr[2])
                x = int(arr[3])
                y = int(arr[4])
                d = int(arr[5])
                Y[user][period] = [x,y]
                D[user][period] = d
                W[user] = w

            line = f.readline()
        f.close()
        
        return Y,W,D,p

