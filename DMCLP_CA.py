#!/usr/bin/env python
# coding: utf-8

# In[1]:

import DMCLP_utils as utils
import numpy as np
import copy
import timeit
import operator

# In[2]:

def get_y_star(T,h,upsilon_w,upsilon_d,args=()):
    
    # generate points on S
    points = [[x,y] 
              for x in np.linspace(min(args['Q']['x']),max(args['Q']['x']),args['CA_grid_division'])
              for y in np.linspace(min(args['Q']['y']),max(args['Q']['y']),args['CA_grid_division']) 
             ]

    # create a dictionary to store the UAV projections
    y_star = dict()
    
    # compute optimal CA locations for each period
    for t,t_val in enumerate(T):
        count = 0
        varphi = dict()
        tau_t = 0
        if t == len(T)-1:
            tau_t = t_val + 0.5*(T[t]-T[t-1])
        else:
            tau_t = (T[t] + T[t+1])*0.5
        for p,point in enumerate(points):
            w_y = utils.w_func(point,args['w_bar'],args['delta_w'],upsilon_w,tau_t,args['delta_t'])
            d_y = utils.d_func(point,args['d_bar'],args['delta_d'],upsilon_d,tau_t,args['delta_t'])
            varphi[p] = (np.pi*w_y*d_y**3)/(3*(d_y-args['L_min'])*utils.L_bar(h,args['F'],args['B'],args['eta'])**2)

        # determine the y point whose objective value is the highest
        best_y = max(varphi.items(), key=operator.itemgetter(1))[0]
        
        # set period projection to the best y
        y_star[t] = points[best_y]
        
    return y_star

def get_h_star(T,y_star,upsilon_w,upsilon_d,args=()):
    
    # generate points on h
    h_cand = np.linspace(min(args['Q']['h']),max(args['Q']['h']),args['CA_h_division'])
    
    # create a dictionary to store UAV altitudes
    h_star = dict()
    
    # compute the best altitude for each period
    for t,t_val in enumerate(T):
        varphi_h = dict()
        tau_t = 0
        if t == len(T)-1:
            tau_t = t_val + 0.5*(T[t]-T[t-1])
        else:
            tau_t = (T[t] + T[t+1])*0.5
        for h,h_val in enumerate(h_cand):
            w_y = utils.w_func(y_star[t],args['w_bar'],args['delta_w'],upsilon_w,tau_t,args['delta_t'])
            d_y = utils.d_func(y_star[t],args['d_bar'],args['delta_d'],upsilon_d,tau_t,args['delta_t'])
            varphi_h[h] = (np.pi*w_y*d_y**3)/(3*(d_y - args['L_min'])*utils.L_bar(h,args['F'],args['B'],args['eta'])**2)

        # determine the altitude whose objective value is the highest
        best_h = max(varphi_h.items(), key=operator.itemgetter(1))[0]
        
        # set period altitude to the best h
        h_star[t] = h_cand[best_h]
        
    return h_star

def solve_initial(T,upsilon_w,upsilon_d,args=()):
    # fix the altitude
    h = min(args['Q']['h'])
    
    # get y_star for each period
    y_star = get_y_star(T,h,upsilon_w,upsilon_d,args)
    
    # get h_star for each period
    h_star = get_h_star(T,y_star,upsilon_w,upsilon_d,args)
    
    # concatenate y_star and h_star
    X = {t: [y_star[t][0],y_star[t][1],h_star[t]] for t,t_val in enumerate(T)}
    
    return X

def select_periods(X,prob):
    per1 = 0
    per2 = 1
    if np.random.rand() <= prob:
        while True:
            per1 = np.random.choice(list(X.keys()))
            per2 = np.random.choice(list(X.keys()))
            if per1 != per2:
                break
    else:
        dist_max = -np.Inf
        for t1,x1 in X.items():
            for t2,x2 in X.items():
                if t1 != t2 and t2 > t1:
                    distance = np.linalg.norm(np.subtract(x1,x2))
                    if distance > dist_max:
                        dist_max = distance
                        per1 = t1
                        per2 = t2

    return per1,per2

def get_new_cor(x1,x2,step_size,Q):
    
    if x1[0] == x2[0] and x1[1] == x2[1]:
        theta_horizontal = np.arctan((x1[1]-x2[1])/(x1[0]-x2[0]))*180/np.pi
        theta_overall = np.arctan((x1[2]-x2[2])/np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2))*180/np.pi
    else:
        theta_horizontal = np.arctan((x1[1]-x2[1])/(x1[0]-x2[0]))*180/np.pi
        theta_overall = np.arctan((x1[2]-x2[2])/np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2))*180/np.pi
        
    x_change = step_size*np.cos(theta_overall)*np.cos(theta_horizontal)
    y_change = step_size*np.cos(theta_overall)*np.sin(theta_horizontal)
    h_change = step_size*np.sin(theta_overall)
    
    x1_new = x1[0] - x_change
    if x1_new < min(Q['x']):
        x1_new = min(Q['x'])
    elif x1_new > max(Q['x']):
        x1_new = max(Q['x'])
    
    y1_new = x1[1] - y_change
    if y1_new < min(Q['y']):
        y1_new = min(Q['y'])
    elif y1_new > max(Q['y']):
        y1_new = max(Q['y'])
    
    h1_new = x1[2] - h_change
    if h1_new < min(Q['h']):
        h1_new = min(Q['h'])
    elif h1_new > max(Q['h']):
        h1_new = max(Q['h'])
    
    x2_new = x1[0] + x_change
    if x2_new < min(Q['x']):
        x2_new = min(Q['x'])
    elif x2_new > max(Q['x']):
        x2_new = max(Q['x'])
    
    y2_new = x1[1] + y_change
    if y2_new < min(Q['y']):
        y2_new = min(Q['y'])
    elif y2_new > max(Q['y']):
        y2_new = max(Q['y'])
    
    h2_new = x1[2] + h_change
    if h2_new < min(Q['h']):
        h2_new = min(Q['h'])
    elif h2_new > max(Q['h']):
        h2_new = max(Q['h'])
    
    return {'x1':[x1_new,y1_new,h1_new],'x2':[x2_new,y2_new,h2_new]}

def solve(I,T,Y,W,D,p,upsilon_w,upsilon_d,args=()):
    # initialize replication counter and empty list to store replication results
    rep_no = 0
    rep_objs = dict()
    rep_Xs = dict()
    
    start = timeit.default_timer()

    while rep_no < args['CA_number_of_replications']:
        
        # initialize iteration counters
        k = 0
        i = 0
        rho = {k: args['CA_rho']}
        count = 0

        # determine initial solution
        X = {count: solve_initial(T,upsilon_w,upsilon_d,args)}
        obj = {count: utils.omega(I,T,X[count],Y,W,D,p,args)}
        
        while k < args['CA_K']:
            if count != 0:
                if i == args['CA_l']:
                    rho[count] = rho[count-1]*args['CA_varrho']
                    i = 0
                else:
                    rho[count] = rho[count-1]

            per1,per2 = select_periods(X[count],args['CA_P'])
            
            # determine new coordinates for selected locations
            x1_new,x2_new = None,None
            if X[count][per1][2] > X[count][per2][2]:
                new_cors = get_new_cor(X[count][per1],X[count][per2],rho[count],args['Q'])
                x1_new,x2_new = new_cors['x1'],new_cors['x2']
            else:
                new_cors = get_new_cor(X[count][per2],X[count][per1],rho[count],args['Q'])
                x2_new,x1_new = new_cors['x1'],new_cors['x2']

            # create a dummy solution to test if new solution is better
            X_new = copy.deepcopy(X[count])
            X_new[per1] = x1_new
            X_new[per2] = x2_new

            obj_new = utils.omega(I,T,X_new,Y,W,D,p,args)

            if obj_new > obj[count]:
                obj[count+1] = obj_new
                X[count+1] = copy.deepcopy(X_new)
                k = 0
            else:
                obj[count+1] = obj[count]
                X[count+1] = copy.deepcopy(X[count])
                k += 1

            i += 1

            count += 1

        rep_objs[rep_no] = obj[len(obj)-1]
        rep_Xs[rep_no] = X[len(X)-1]
        
        rep_no += 1
        
    best_rep = max(rep_objs.items(),key=operator.itemgetter(1))[0]
            
    finish = timeit.default_timer()

    cpu = np.round(finish-start,decimals=2)
        
    # return the replication that yields the best objective function value
    return rep_Xs[best_rep],cpu