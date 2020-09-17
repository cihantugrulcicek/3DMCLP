#!/usr/bin/env python
# coding: utf-8

import gurobipy as GRB
import numpy as np

def solve(I,T,Y,W,D,p,args=()):
    
    nu,kappa = dict(),dict()
    for i in I:
        nu[i] = {t: (W[i]*D[i][t])/(D[i][t] - args['L_min']) for t in T}
        kappa[i] = {t: 1/(D[i][t] - args['L_min']) for t in T}
        
    try:
        
        with GRB.Env(empty=True) as env:
            if not args['show_console']:
                env.setParam('OutputFlag', 0)
            env.start()
            with GRB.Model('DMCLP_Gurobi',env=env) as m:
                x = m.addVars(T,3,lb=0,name='x')
                z = m.addVars(I,T,vtype=GRB.BINARY,name='z')
                s = m.addVars(I,T,lb=0,name='s')

                total_reloc = m.addVar(lb=0,name='total_reloc')
                reloc = m.addVars(T[1:],lb=0,name='reloc') # uav relocation variables
                reloc_aux = m.addVars(T[1:],3,lb=-GRB.INFINITY,name='reloc_aux') # auxiliary variables to store differences on each axis for uav relocation

                radius_user = m.addVars(I,T,lb=0,name='radius_user')
                radius_user_aux = m.addVars(I,T,2,lb=-GRB.INFINITY,name='radius_user_aux')
                dist_user = m.addVars(I,T,lb=0,name='dist_user')
                dist_user_aux = m.addVars(I,T,3,lb=-GRB.INFINITY,name='dist_user_aux')
                log_dist_user = m.addVars(I,T,lb=-GRB.INFINITY,name='log_dist_user')

                L = m.addVars(I,T,lb=-GRB.INFINITY,name='pl')
                prob_los = m.addVars(I,T,lb=0,ub=1,name='prob_los')
                pl_aux1 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux1')
                pl_aux2 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux2')
                pl_aux3 = m.addVars(I,T,lb=0,name='pl_aux3')
                pl_aux4 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux4')
                pl_aux5 = m.addVars(I,T,lb=0,name='pl_aux5')
                pl_aux6 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux6')
                pl_aux7 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux7')
                pl_aux8 = m.addVars(I,T,lb=0,name='pl_aux8')
                pl_aux9 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux9')
                pl_aux10 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux10')
                pl_aux11 = m.addVars(I,T,lb=0,name='pl_aux11')
                pl_aux12 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux12')
                pl_aux13 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux13')
                pl_aux14 = m.addVars(I,T,lb=0,name='pl_aux14')
                pl_aux15 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux15')
                pl_aux16 = m.addVars(I,T,lb=0,name='pl_aux16')
                pl_aux17 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux17')
                pl_aux18 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux18')
                pl_aux19 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux19')
                pl_aux20 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux20')
                pl_aux21 = m.addVars(I,T,lb=0,name='pl_aux21')
                pl_aux22 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux22')
                pl_aux23 = m.addVars(I,T,lb=0,name='pl_aux23')
                pl_aux24 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux24')
                pl_aux25 = m.addVars(I,T,lb=-GRB.INFINITY,name='pl_aux25')

                # set objective function
                obj = (-p*total_reloc + quicksum(nu[i][t]*z[i,t] - kappa[i][t]*s[i,t] for i in I for t in T))
                m.setObjective(obj,GRB.MAXIMIZE)

                # constraints on service region
                m.addConstrs( (x[t,0] <= 1500 for t in T), name='service_area_x')
                m.addConstrs( (x[t,1] <= 1500 for t in T), name='service_area_y')
                m.addConstrs( (x[t,2] <= 500 for t in T), name='service_area_h_up')
                m.addConstrs( (x[t,2] >= 50 for t in T), name='service_area_h_lo')

                # constraints for computing relocations
                m.addConstr( ( quicksum(reloc[t] for t in T[1:]) == total_reloc), name='total_reloc')
                m.addConstrs( ( quicksum(reloc_aux[t,k]*reloc_aux[t,k] for k in range(3)) <= reloc[t]*reloc[t] for t in T[1:]), name='reloc')
                m.addConstrs( ( reloc_aux[t,k] == x[t,k] - x[t-1,k] for t in T[1:] for k in range(3)), name='reloc_aux')

                # constraints for loss functions
                m.addConstrs( ( L[i,t] - D[i][t] <= args['bigM']*(1-z[i,t]) for i in I for t in T), name='cons5')
                m.addConstrs( ( s[i,t] <= args['bigM']*z[i,t] for i in I for t in T), name='cons7')
                m.addConstrs( ( s[i,t] <= L[i,t] for i in I for t in T), name='cons8')
                m.addConstrs( ( L[i,t] - args['bigM']*(1-z[i,t]) <= s[i,t] for i in I for t in T), name='cons9')

                m.addConstrs( ( L[i,t] == args['F'] + 10*args['eta']*log_dist_user[i,t] + args['B']*prob_los[i,t] for i in I for t in T), name='pl')

                # constraints for log10(||x_t - y_it||) 
                m.addConstrs( ( quicksum( dist_user_aux[i,t,k]*dist_user_aux[i,t,k] for k in range(3) ) <= dist_user[i,t]*dist_user[i,t] for i in I for t in T ), name='dist_user')
                m.addConstrs( ( dist_user_aux[i,t,k] == x[t,k] - Y[i][t][k] for i in I for t in T for k in range(2) ), name='dist_user_xy')
                m.addConstrs( ( dist_user_aux[i,t,2] == x[t,2]  for i in I for t in T ), name='dist_user_h')
                for i in I:
                    for t in T:
                        m.addGenConstrLogA(dist_user[i,t], log_dist_user[i,t], 10.0, "log_dist_user[{},{}]".format(i,t), "FuncPieces=-1 FuncPieceError=1e-5")

                # constraints for prob_los = 1/(1+alpha*exp(-beta(theta-alpha)))
                for i in I:
                    for t in T:
                        m.addConstr(pl_aux1[i,t] - pl_aux2[i,t] == 1, name='pl_aux1[{},{}]'.format(i,t))
                        m.addConstr(pl_aux1[i,t] == 0.25*pl_aux3[i,t], name='pl_aux2[{},{}]'.format(i,t))
                        m.addConstr(pl_aux3[i,t] == pl_aux4[i,t]*pl_aux4[i,t], name='pl_aux3[{},{}]'.format(i,t))
                        m.addConstr(pl_aux4[i,t] == prob_los[i,t] + pl_aux7[i,t], name='pl_aux4[{},{}]'.format(i,t))
                        m.addConstr(pl_aux2[i,t] == 0.25*pl_aux5[i,t], name='pl_aux5[{},{}]'.format(i,t))
                        m.addConstr(pl_aux5[i,t] == pl_aux6[i,t]*pl_aux6[i,t], name='pl_aux6[{},{}]'.format(i,t))
                        m.addConstr(pl_aux6[i,t] == prob_los[i,t] - pl_aux7[i,t], name='pl_aux7[{},{}]'.format(i,t))

                        m.addConstr(pl_aux7[i,t] == 1 + args['alpha']*pl_aux8[i,t], name='pl_aux8[{},{}]'.format(i,t))

                        m.addGenConstrExp(pl_aux9[i,t], pl_aux8[i,t], name='pl_aux9[{},{}]'.format(i,t))
                        m.addConstr(pl_aux9[i,t] == -args['beta']*(180/np.pi*pl_aux10[i,t] - args['alpha']), name='pl_aux10[{},{}]'.format(i,t))

                        m.addConstr(pl_aux12[i,t] - pl_aux13[i,t] == x[t,2], name='pl_aux11[{},{}]'.format(i,t))
                        m.addConstr(pl_aux12[i,t] == 0.25*pl_aux14[i,t], name='pl_aux12[{},{}]'.format(i,t))
                        m.addConstr(pl_aux14[i,t] == pl_aux15[i,t]*pl_aux15[i,t], name='pl_aux13[{},{}]'.format(i,t))
                        m.addConstr(pl_aux15[i,t] == pl_aux11[i,t] + radius_user[i,t], name='pl_aux14[{},{}]'.format(i,t))
                        m.addConstr(pl_aux13[i,t] == 0.25*pl_aux16[i,t], name='pl_aux15[{},{}]'.format(i,t))
                        m.addConstr(pl_aux16[i,t] == pl_aux17[i,t]*pl_aux17[i,t], name='pl_aux16[{},{}]'.format(i,t))
                        m.addConstr(pl_aux17[i,t] == pl_aux11[i,t] - radius_user[i,t], name='pl_aux17[{},{}]'.format(i,t))

                        for k in range(2):
                            m.addConstr( radius_user_aux[i,t,k] == x[t,k] - Y[i][t][k], name='radius_user_xy[{},{},{}]'.format(i,t,k))
                        m.addConstr( radius_user[i,t]*radius_user[i,t] >= quicksum( radius_user_aux[i,t,k]*radius_user_aux[i,t,k] for k in range(2) ), name='radius_user[{},{}]'.format(i,t))

                        m.addConstr(pl_aux18[i,t] - pl_aux19[i,t] == pl_aux20[i,t], name='pl_aux18[{},{}]'.format(i,t))
                        m.addConstr(pl_aux18[i,t] == 0.25*pl_aux21[i,t], name='pl_aux19[{},{}]'.format(i,t))
                        m.addConstr(pl_aux21[i,t] == pl_aux22[i,t]*pl_aux22[i,t], name='pl_aux20[{},{}]'.format(i,t))
                        m.addConstr(pl_aux22[i,t] == pl_aux10[i,t] + pl_aux25[i,t], name='pl_aux21[{},{}]'.format(i,t))
                        m.addConstr(pl_aux19[i,t] == 0.25*pl_aux23[i,t], name='pl_aux22[{},{}]'.format(i,t))
                        m.addConstr(pl_aux23[i,t] == pl_aux24[i,t]*pl_aux24[i,t], name='pl_aux23[{},{}]'.format(i,t))
                        m.addConstr(pl_aux24[i,t] == pl_aux10[i,t] - pl_aux25[i,t], name='pl_aux24[{},{}]'.format(i,t))
                        m.addGenConstrCos(pl_aux11[i,t], pl_aux25[i,t], name='pl_aux25[{},{}]'.format(i,t))
                        m.addGenConstrSin(pl_aux11[i,t], pl_aux20[i,t], name='pl_aux26[{},{}]'.format(i,t))


                if args['write_file']:
                    m.write('file.lp')

                m.Params.TimeLimit = args['time_limit']

                m.Params.FeasibilityTol = 1e-9

                m.Params.NonConvex = 2

                m.optimize()

                objval = 0
                X = dict()
                cpu = 0
                    
                if m.status in [2,7,8,9,10,13]:
                    if args['write_solution']: 
                        m.write('solution.sol')

                    X = {t: [x[t,0].x, x[t,1].x, x[t,2].x] for t in T}
                    return X,m.Runtime
                else: 
                    if args['compute_IIS']:
                        m.computeIIS()
                        m.write('model.ilp')
                    X = {t: [0,0,50] for t in T}
                    return X,cpu
            
    except GRB.GurobiError as e:
        #print('Error code ' + str(e.errno) + ": " + str(e))
        X = {t: [0,0,50] for t in T}
        return X,0

    except AttributeError:
        #print('Encountered an attribute error')
        X = {t: [0,0,50] for t in T}
        return X,0
