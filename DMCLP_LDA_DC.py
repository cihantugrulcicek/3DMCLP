#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import copy
import operator
import DMCLP_utils as utils


# In[ ]:


def get_random_location(T,Q):
    X = {}
    for t in T:
        X[t] = [np.random.randint(min(Q['x']),max(Q['x'])),
                np.random.randint(min(Q['y']),max(Q['y'])),
                np.random.randint(min(Q['h']),max(Q['h']))]
    return X
    
def get_distance(x1,x2):
    return np.linalg.norm(np.subtract(x1,x2))

def get_gradient(I,T,X,Y,p,omega5,gradient_type,args=()):
    # create an empty list of gradients and set all elements to zero
    g_x = []
    for t in T:
        g_x.append([0.0,0.0,0.0])
        
    # compute gradient with respect to g(X) function if type = 1
    if gradient_type == 1:
        for t in T:
            if t == 0:
                for k in range(3):
                    if X[t] == X[t+1]:
                        g_x[t][k] += 0.0
                    else:
                        g_x[t][k] += -p*((X[t][k] - X[t+1][k])/get_distance(X[t],X[t+1]))

                    for i in I[t]:
                        if k == 2:
                            g_x[t][k] += (omega5[i][t]*X[t][k])/((np.log(10)*((X[t][0] - Y[i][t][0])**2 + (X[t][1] - Y[i][t][1])**2 + (X[t][2])**2)))
                        else:
                            g_x[t][k] += (omega5[i][t]*(X[t][k] - Y[i][t][k]))/((np.log(10)*((X[t][0] - Y[i][t][0])**2 + (X[t][1] - Y[i][t][1])**2 + (X[t][2])**2)))


            elif t == len(T)-1:
                for k in range(3):
                    if X[t] == X[t-1]:
                        g_x[t][k] += 0.0
                    else:
                        g_x[t][k] += -p*((X[t][k] - X[t-1][k])/get_distance(X[t],X[t-1]))

                    for i in I[t]:
                        if k == 2:
                            g_x[t][k] += (omega5[i][t]*X[t][k])/((np.log(10)*((X[t][0] - Y[i][t][0])**2 + (X[t][1] - Y[i][t][1])**2 + (X[t][2])**2)))
                        else:
                            g_x[t][k] += (omega5[i][t]*(X[t][k] - Y[i][t][k]))/((np.log(10)*((X[t][0] - Y[i][t][0])**2 + (X[t][1] - Y[i][t][1])**2 + (X[t][2])**2)))


            else:
                for k in range(3):
                    if X[t] == X[t-1] and X[t] == X[t+1]:
                        g_x[t][k] += 0.0
                    elif X[t] == X[t-1]:
                        g_x[t][k] += p*((X[t+1][k] - X[t][k])/get_distance(X[t],X[t+1]))
                    elif X[t] == X[t+1]:
                        g_x[t][k] += -p*((X[t][k] - X[t-1][k])/get_distance(X[t],X[t-1]))
                    else:
                        g_x[t][k] += -p*(((X[t][k] - X[t-1][k])/get_distance(X[t],X[t-1])) - ((X[t+1][k] - X[t][k])/get_distance(X[t],X[t+1])))

                    for i in I[t]:
                        if k == 2:
                            g_x[t][k] += (omega5[i][t]*X[t][k])/((np.log(10)*((X[t][0] - Y[i][t][0])**2 + (X[t][1] - Y[i][t][1])**2 + (X[t][2])**2)))
                        else:
                            g_x[t][k] += (omega5[i][t]*(X[t][k]-Y[i][t][k]))/((np.log(10)*((X[t][0] - Y[i][t][0])**2 + (X[t][1] - Y[i][t][1])**2 + (X[t][2])**2)))
    else:
        for t in T:
            for i in I[t]:
                for k in range(3):
                    if k == 2:
                        g_x[t][k] += (omega5[i][t]*X[t][k])/((np.log(10)*((X[t][0] - Y[i][t][0])**2 + (X[t][1] - Y[i][t][1])**2 + (X[t][2])**2)))
                    else:
                        g_x[t][k] += (omega5[i][t]*(X[t][k]-Y[i][t][k]))/((np.log(10)*((X[t][0] - Y[i][t][0])**2 + (X[t][1] - Y[i][t][1])**2 + (X[t][2])**2)))

    return g_x

def omega_prime_p2(I,T,X,Y,p,omega5,args=()):
    relocation = utils.g(T,X)
    coverage = 0
    for t in T:
        for i in I:
            coverage += omega5[i][t]*np.log10(np.sqrt((X[t][0] - Y[i][t][0])**2 + (X[t][1] - Y[i][t][1])**2 + (X[t][2])**2))
            
    return -p*relocation + coverage
    
def solve(I,T,Y,p,duals,args=()):
    
    # create empty lists to track objective function values and locations
    obj_best = 0
    
    # calculate auxiliary coefficients omega_5
    omega3 = dict()
    for i in I:
        omega3[i] = {t: duals['lambda'][i][t] - duals['vartheta'][i][t] - duals['delta'][i][t] for t in T}
    
    omega5 = dict()
    for i in I:
        omega5[i] = {t: 10*args['eta']*omega3[i][t] for t in T}
    
    O = sum([(args['F'] + args['B'])*omega3[i][t] for i in I for t in T])
    
    # divide user set into two subset with negative and positive coefficients
    I_plus,I_minus = dict(),dict()
    for t in T:
        I_plus[t] = [i for i in I if omega5[i][t] >= 0]
        I_minus[t] = [i for i in I if i not in I_plus[t]]
    
    objs,locs = dict(),dict()
    
    for rep_no in range(args['DC_number_of_replications']):
        # randomly initialize X
        X = get_random_location(T,args['Q'])

        # initialize best objective and best X
        obj_best = O + omega_prime_p2(I,T,X,Y,p,omega5,args)
        X_best = copy.deepcopy(X)

        # initialize the iteration counter
        iter_count = 0

        while True:

            # save current X
            X_old = copy.deepcopy(X)
            
            # find the gradient of current solution for the I_plus set
            gradient = get_gradient(I_plus,T,X,Y,p,omega5,1,args)
            
            obj = O + omega_prime_p2(I,T,X,Y,p,omega5,args)
            
            # update locations by the gradient with step_size
            for t in T:
                for k in range(3):
                    X[t][k] -= np.round((args['DC_gradient_step_size']/(iter_count+1))*gradient[t][k],decimals=6)
                    if k == 2:
                        X[t][k] = max(min(args['Q']['h']),min(max(args['Q']['h']),X[t][k]))
                        
            # save X and objective function value after gradient operation
            obj_new1 = O + omega_prime_p2(I,T,X,Y,p,omega5,args)
            
            if obj_new1 > obj_best:
                obj_best = obj_new1
                X_best = copy.deepcopy(X)

            # find the gradient of current solution for the I_minus set
            gradient = get_gradient(I_minus,T,X,Y,p,omega5,2,args)

            # update locations by the gradient with step_size
            for t in T:
                for k in range(3):
                    X[t][k] -= np.round((args['DC_gradient_step_size']/(iter_count+1))*gradient[t][k],decimals=6)
                    if k == 2:
                        X[t][k] = max(min(args['Q']['h']),min(max(args['Q']['h']),X[t][k]))
                    
            # save X and objective function value after gradient operation
            obj_new2 = O + omega_prime_p2(I,T,X,Y,p,omega5,args)
            
            if obj_new2 > obj_best:
                obj_best = obj_new2
                X_best = copy.deepcopy(X)

            # determine the norm of locations found after first and second gradient operation
            diff = sum([np.linalg.norm(np.subtract(x1,x2)) for x1,x2 in zip(X_old.values(),X.values())])
            
            # if the norm value is less than a threshold or iteration limit is reached, terminate
            if diff <= args['DC_epsilon_gradient'] or iter_count == args['DC_iter_max']:
                break

            # otherwise proceed to the next iteration
            else:
                iter_count += 1
                
        objs[rep_no] = obj_best
        locs[rep_no] = copy.deepcopy(X_best)
    
    iter_best = max(objs.items(), key=operator.itemgetter(1))[0]
    
    return locs[iter_best]