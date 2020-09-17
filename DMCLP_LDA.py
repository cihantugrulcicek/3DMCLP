#!/usr/bin/env python
# coding: utf-8

# In[1]:

import timeit
import numpy as np
import copy
import DMCLP_LDA_DC as DC
import DMCLP_utils as utils

# In[2]:

def initialize_duals(I,T):
    dual_lambda,dual_vartheta,dual_delta = dict(),dict(),dict()
    for i in I:
        dual_lambda[i] = {t: 1 for t in T}
        dual_vartheta[i] = {t: 1 for t in T}
        dual_delta[i] = {t: 1 for t in T}
        
    duals = {'lambda':dual_lambda,'vartheta':dual_vartheta,'delta':dual_delta}
    
    return duals

def solve_p1(I,T,W,D,duals,args):
    # create ampty dictionaries to store Z and S
    Z,S = {},{}
    
    # compute omega1 and omega2 for each (i,t) pair and set Z and S variables
    # according to the Table 1
    for i in I:
        Z_dummy,S_dummy = {},{}
        for t in T:
            nu = (W[i]*D[i][t])/(D[i][t] - args['L_min'])
            kappa = 1/(D[i][t] - args['L_min'])
            omega1 = nu - args['bigM']*(duals['vartheta'][i][t] + duals['delta'][i][t])
            omega2 = -kappa - duals['lambda'][i][t] + duals['vartheta'][i][t]
            if omega1 >= 0 and omega2 >= 0:
                Z_dummy[t] = 1
                S_dummy[t] = args['bigM']
            elif omega1 >= 0 and omega2 < 0:
                Z_dummy[t] = 1
                S_dummy[t] = 0
            elif omega1 < 0 and omega2 >= 0:
                if omega1 + args['bigM']*omega2 >= 0:
                    Z_dummy[t] = 1
                    S_dummy[t] = args['bigM']
                else:
                    Z_dummy[t] = 0
                    S_dummy[t] = 0
            else:
                Z_dummy[t] = 0
                S_dummy[t] = 0                
            
        Z[i] = Z_dummy
        S[i] = S_dummy

    return Z,S

def omega_LR(I,T,X,Z,S,Y,W,D,p,duals,args=()):
    relocation = utils.g(T,X)
    coverage = 0
    for i in I:
        for t in T:
            nu = (W[i]*D[i][t])/(D[i][t] - args['L_min'])
            kappa = 1/(D[i][t] - args['L_min'])
            omega1 = nu - args['bigM']*(duals['vartheta'][i][t] + duals['delta'][i][t])
            omega2 = -kappa - duals['lambda'][i][t] + duals['vartheta'][i][t]
            omega3 = duals['lambda'][i][t] - duals['vartheta'][i][t] - duals['delta'][i][t]
            omega4 = args['bigM']*(duals['vartheta'][i][t] + duals['delta'][i][t]) + (duals['delta'][i][t]*D[i][t])
            coverage += omega1*Z[i][t] + omega2*S[i][t] + omega3*utils.L(X[t],Y[i][t],args) + omega4
    
    return -p*relocation + coverage

def get_step_size(I,T,Y,D,X,Z,S,LB,UB,beta,args=()):
    sum_norm = 0
    for t in T:
        for i in I:
            pl = utils.L(X[t],Y[i][t],args)
            sum_norm += (pl - S[i][t])**2
            + (S[i][t] + args['bigM']*(1 - Z[i][t]) - pl)**2 
            + (D[i][t] + args['bigM']*(1 - Z[i][t]) - pl)**2
    
    step_size = beta * ((UB - LB)/sum_norm)
    
    return step_size

def update_duals(I,T,Y,D,X,Z,S,duals,LB,UB,beta,args=()):
    step_size = get_step_size(I,T,Y,D,X,Z,S,LB,UB,beta,args)
    for i in I:
        for t in T:
            pl = utils.L(X[t],Y[i][t],args)
            duals['lambda'][i][t]   = max(0,duals['lambda'][i][t] - step_size * (pl-S[i][t]))
            duals['vartheta'][i][t] = max(0,duals['vartheta'][i][t] - step_size * (S[i][t] + args['bigM']*(1 - Z[i][t]) - pl))
            duals['delta'][i][t]    = max(0,duals['delta'][i][t] - step_size * (D[i][t] + args['bigM']*(1 - Z[i][t]) - pl))
            
    return duals

def solve(I,T,Y,W,D,p,args=()):
    
    # set upper and lower bound values and create empty dictionaries to track the progress
    UB = sum([sum([1 for t in T]) for i in I])
    LB = -np.Inf
    UB_track, LB_track, gap_track, X_track = dict(),dict(),dict(),dict()
    best_iter = 0 # iteration at which the best LB is found
    beta = 2.0 # step_size
    beta_change = 0 # counter to track number of iterations without an improvement in UB
    
    # initialize dual variables
    duals = initialize_duals(I,T)
    
    # initialize iteration counter
    iter_count=0
    
    # start clock time
    start = timeit.default_timer()
    
    # iterate the algorithm until either maximum iteration number or maximum CPU time is reached
    while (iter_count < args['LDA_iter_max']) or (timeit.default_timer() - start >= args['LDA_CPU_max']):
        
        signal = 0
        
        # solve the first sub-problem
        Z,S = solve_p1(I,T,W,D,duals,args)
        
        # solve the second sub-problem
        X = DC.solve(I,T,Y,p,duals,args)
        
        # add X to the list
        X_track[iter_count] = X
        
        # compute the upper bound value for current iteration by omega_LR function
        ub_iteration = omega_LR(I,T,X,Z,S,Y,W,D,p,duals,args)
        
        # if a tighter upper bound is found, update UB
        # and reset the counter for the step size beta
        if ub_iteration < UB and ub_iteration > 0 and ub_iteration > LB: 
            UB = ub_iteration
            beta_change = 0
        # otherwise increase the counter for the step size beta
        else:
            beta_change += 1
            # if the counter for the step size reaches the number of iteration with no improvement, then 
            # divide beta by 2 and reset the counter
            if beta_change == args['LDA_halving_iter_max']:
                beta = beta/2
                beta_change = 0
            
        # save current UB to track list
        UB_track[iter_count] = UB
        
        # retrieve the lower bound value for current iteration
        lb_iteration = utils.omega(I,T,X,Y,W,D,p,args)
        
        # if a tighter lower bound is found, update the LB and save the iteration to best_iter
        if lb_iteration > LB: 
            LB = lb_iteration
            best_iter = iter_count
            signal = 1
        
        # save current LB to track list
        LB_track[iter_count] = LB
        
        # save duality gap to track list
        gap_track[iter_count] = np.round((UB-LB)/UB,decimals=6)
        
        # if the optimality gap is less than epsilon
        if (UB-LB)/UB <= args['LDA_epsilon_acc']:
            # terminate the algorithm
            break
        
        # otherwise
        else:
            # update dual variables
            duals = update_duals(I,T,Y,D,X,Z,S,duals,LB,ub_iteration,beta,args)
            
            # proceed to the next iteration
            iter_count += 1
        
        if args['verbose']:
            if iter_count-1 == 0:
                print('{:^10} {:^10} {:^10} {:^10} {:^10}'.format('Iteration','UB','LB','Gap(%)','beta'))
                print('{:>10} {:>10.2f} {:>10.4f} {:>10.4f} {:>10.4f}'.format(iter_count-1,UB_track[iter_count-1],LB_track[iter_count-1],gap_track[iter_count-1],beta))
            elif signal == 1:
                #print('{:>10} {:>10.2f} {:>10.2f} {:>10.4f} {:>10.4f}'.format(iter_count,UB,LB,gap_track[iter_count-1],beta))
                print('{:>10} {:>10.2f} {:>10.4f} {:>10.4f} {:>10.4f}'.format('*{}'.format(iter_count-1),ub_iteration,LB_track[iter_count-1],gap_track[iter_count-1],beta))
            elif iter_count % 5 == 0:
                #print('{:>10} {:>10.2f} {:>10.2f} {:>10.4f} {:>10.4f}'.format(iter_count,UB,LB,gap_track[iter_count-1],beta))
                print('{:>10} {:>10.2f} {:>10.4f} {:>10.4f} {:>10.4f}'.format(iter_count-1,ub_iteration,LB_track[iter_count-1],gap_track[iter_count-1],beta))
        
    finish = timeit.default_timer()
    cpu = np.round(finish-start,decimals=2)
                
    return X_track[best_iter],cpu