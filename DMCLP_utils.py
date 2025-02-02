#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


def g(T,X):
    return sum([np.linalg.norm(np.subtract(X[t],X[t-1])) for t in T[1:]])

def L(x,y,args=()):
    radius = np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    dist = np.sqrt(radius**2+x[2]**2)
    log_dist = np.log10(dist)
    prob_los = 1
    if radius != 0:
        theta = 180/np.pi*np.arctan(x[2]/radius)
        prob_los = 1/(1+args['alpha']*np.exp(-args['beta']*(theta-args['alpha'])))
    pl = args['F'] + 10*args['eta']*log_dist + args['B']*prob_los 
    return pl

def omega(I,T,X,Y,W,D,p,args):
    relocation = g(T,X)
    coverage = 0
    for i in I:
        for t in T:
            mu = max(0,(D[i][t] - L(X[t],Y[i][t],args))/(D[i][t] - args['L_min']))
            coverage += W[i]*mu
            
    return -p*relocation + coverage

def upsilon(trend='inc'):
    if trend == 'inc':
        return lambda t,delta_t: 1 + np.log(1 + delta_t*t)
    elif trend == 'dec':
        return lambda t,delta_t: np.exp(-delta_t*t)
    else:
        return lambda t,delta_t: 1 + delta_t*np.random.uniform(-1,1)*t
    
def w_func(y,w_bar,delta_w,t_func,t,delta_t):
    return w_bar*(1 + delta_w*np.cos(np.pi*np.linalg.norm(y)))*t_func(t,delta_t)

def d_func(y,d_bar,delta_d,t_func,t,delta_t):
    return d_bar*(1 + delta_d*np.cos(np.pi*np.linalg.norm(y)))*t_func(t,delta_t)

def p_func(y,delta_p,t_func,t,delta_t):
    return d_bar*(1 + delta_d*np.cos(np.pi*np.linalg.norm(y)))*t_func(t,delta_t)

def L_bar(h,F,B,eta):
    return F + B + 10*eta*np.log10(h)

def get_L_min(Q,F,B,eta):
    return F + 10*eta*np.log10(min(Q['h'])) + B

def get_bigM(Q,F,B,eta):
    max_distance = np.sqrt((max(Q['x']) - min(Q['x']))**2 + (max(Q['y']) - min(Q['y']))**2 + (max(Q['h']))**2)
    L_max = F + 10*eta*np.log10(max_distance) - B
    return L_max

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def save_instance(file_name,content):
    with open('{}\\{}.txt'.format(basepath,file_name),'w') as f:
        f.write(content)
        f.close()

def generate_save_file(Y,W,D,p):
    txt = 'p,{}\n'.format(p)
    txt += 'User,Period,w,x,y,d\n'
    for g,periods in Y.items():
        for t,cors in periods.items():
            txt += '{},{},{},{},{},{}\n'.format(g,t,W[g],cors[0],cors[1],D[g][t])

    return txt

def read_instance_from_file(path):
        df = pd.read_csv(path)
        p = float(df.iloc[:0].columns[1])
        df = pd.read_csv(path,skiprows=[0])
        I = list(df['User'].unique())
        T = list(df['Period'].unique())
        df2 = df.set_index(['User','Period'])
        W = {i: df2['w'][i,0] for i in I}
        Y,D = dict(),dict()
        for i in I:
            Y[i] = {t: [df2['x'][i,t],df2['y'][i,t]] for t in T}
            D[i] = {t: df2['d'][i,t] for t in T}
        
        return {'Y':Y,'W':W,'D':D,'p':p}
    
def read_instances_from_directory(path):
    problems = {} 
 
    for filename in os.listdir(path): 
        if filename.endswith(".txt") and not filename in problems: 
            instance = read_instance_from_file('{}{}'.format(path,filename))
            problems[filename[:-4]] = instance
    
    return problems

def write_result(path,instance,X_ca,X_lda,X_grb):
    txt = 'X_ca_x,X_ca_y,X_ca_h,X_lda_x,X_lda_y,X_lda_h,X_grb_x,X_grb_y,X_grb_h\n'
    for t in X_ca.keys():
        if t == len(X_ca.keys()) - 1:
            txt += '{},{},{},{},{},{},{},{},{}'.format(X_ca[t][0],X_ca[t][1],X_ca[t][2],
                                                       X_lda[t][0],X_lda[t][1],X_lda[t][2],
                                                       X_grb[t][0],X_grb[t][1],X_grb[t][2])
        else:
            txt += '{},{},{},{},{},{},{},{},{}\n'.format(X_ca[t][0],X_ca[t][1],X_ca[t][2],
                                                        X_lda[t][0],X_lda[t][1],X_lda[t][2],
                                                        X_grb[t][0],X_grb[t][1],X_grb[t][2])
    with open('{}{}.txt'.format(path,instance),'w') as f:
        f.write(txt)
        f.close()
            