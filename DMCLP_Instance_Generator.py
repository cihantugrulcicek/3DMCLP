#!/usr/bin/env python
# coding: utf-8

# In[1]:

import DMCLP_utils as utils
import numpy as np
import os


# In[2]:

def generate(Q,upsilon_w,upsilon_d,granularity=10,nbGrid=4,nbPeriod=5,
             w_bar=0.5,delta_w = 0.2,d_bar=90,delta_d=0.2,delta_t=0.2,delta_p=0.2,
             seed=0):
    
    if seed:
        np.random.seed(seed=seed)
    else:
        np.random.seed(seed=777)
    
    # generate grids
    G = {}
    xx = np.linspace(min(Q['x']),max(Q['x']),int(np.sqrt(nbGrid))+1)
    yy = np.linspace(min(Q['y']),max(Q['y']),int(np.sqrt(nbGrid))+1)
    count = 0
    for y in range(1,len(yy)):
        for x in range(1,len(xx)):
            G[count] = {'x':[xx[x-1],xx[x]],'y':[yy[y-1],yy[y]]}
            count += 1
    
    # pick a random point within each grid
    Y,W,D = dict(),dict(),dict()
    for g,grid in G.items():
        Y[g] = {t: [np.random.randint(min(grid['x']),max(grid['x'])),np.random.randint(min(grid['y']),max(grid['y']))] for t in range(nbPeriod)}

    # assign w values within each grid
    W = {}
    for g,grid in G.items():
        w_total = 0
        for x in np.linspace(min(grid['x']),max(grid['x']),granularity):
            for y in np.linspace(min(grid['y']),max(grid['y']),granularity):
                for t in np.linspace(0,nbPeriod,granularity):
                    w_total += utils.w_func([x,y],w_bar,delta_w,upsilon_w,t,delta_t)
        W[g] = np.round(w_total/(granularity**3),decimals=4)
    
    # assign d values within each grid
    D = {}
    for g,grid in G.items():
        D_dummy = {}
        for t in range(nbPeriod):
            d_total = 0
            for x in np.linspace(min(grid['x']),max(grid['x']),granularity):
                for y in np.linspace(min(grid['y']),max(grid['y']),granularity):
                    d_total += utils.d_func([x,y],d_bar,delta_d,upsilon_d,t,delta_t)
            D_dummy[t] = int(d_total/(granularity**2))
        D[g] = D_dummy
    
    # compute relocation penalty p
    p_sum = 0
    for x in np.linspace(min(Q['x']),max(Q['x']),granularity):
        for y in np.linspace(min(Q['y']),max(Q['y']),granularity):
            for t in np.linspace(0,nbPeriod,granularity):
                p_sum += utils.w_func([x,y],w_bar,delta_w,upsilon_w,t,delta_t)
    
    S_area = (max(Q['x'])-min(Q['x']))*(max(Q['y'])-min(Q['y']))
    p = (1 + delta_p)*p_sum/(nbPeriod*S_area)
            
    return Y,W,D,p