
import numpy as np
from scipy import integrate
from scipy.linalg import expm
import nlopt
import argparse
from math import exp
import json



# setup args
def setup_args():
    parser = argparse.ArgumentParser(description="opt by ll and ibd")
    parser.add_argument('--result-path', type=str,
                        default="../results/debug.json", help="path to save result")
    parser.add_argument('--par-path', type=str,
                        default="../par/par_0.json", help="path to load ibd label and true par bound.json")
    return parser.parse_args()
args = setup_args()
print(args)

# load true par and the label(ibd)
def load_par(par_path):
    with open(par_path) as f:
        data = json.load(f)
    return data
true_par = load_par(args.par_path) 

# use fix==0 来确定该参数是需要进行估计的，否则fix==1，并且使用fix_par作为固定值


# load data; must before the def of func
# def load_txt(data_path):
#     with open(data_path, "r", encoding='utf-8') as f:  #打开文本
#         data=[]
#         for ann in f.readlines():
#             ann = ann.strip('\n') # 删除回车
#             if len(ann) == 0: continue # 跳过最后一行的只有回车的行
#             data.append(float(ann)*(1e-8))
#     return data
# data = load_txt(true_par['data_path'])

def load_txt(data_path, reps):
    data=[]
    for rep in reps:
        data_path_temp = data_path 
        #print(data_path_temp)
        with open(data_path_temp, "r", encoding='utf-8') as f:  #打开文本
            for ann in f.readlines():
                ann = ann.strip('\n') # 删除回车
                if len(ann) == 0: continue # 跳过最后一行的只有回车的行
                data.append(float(ann)*(1e-2))
    return data
reps = true_par['rep']
data = load_txt(true_par['data_path'], reps)
print(len(data))




def update_step(data, true_par):

    fix = true_par['fix']
    fix_name = true_par['fix_name']
    fix_val = true_par['fix_val']

    # define the obj func: x: [N_hap, tau_a, m]-> loss
    def func(in_x, grad=None):
        j=0
        for i in range(3):
            if fix[i] == 1:
                globals()[fix_name[i]]=fix_val[i]
            elif fix[i] == 0:
                globals()[fix_name[i]]=in_x[j]
                j+=1
        # fixed par
        L = 100000000
        u = 0.01 # threshold (M)：1M = 1cM centi-Morgan
        r = 1 # recombination rate 1cM/Mb
        gamma = true_par['gamma']
        beta = true_par['beta']
        ibd_label = true_par['ibd_label']
        #eps = 1e-23  # 为了计算log（f）时候的稳定，也可以log(f+eps)
        sample_all = data

        def H(t):
            a2 = 2*N_hap*r*u + 1
            a1 = 4*N_hap*r*u + 1 + a2*2*r*u*t
            a3 = exp(-(2*r*u+1/N_hap)*t)
            result = a1*a3/(a2*a2)
            return result

        # ibd sharing
        def ibd_sharing(tau):
            ibdsharing = H(0) - 2*m*(1-m)*H(tau)
            return ibdsharing

        def I(t,l):
            b1 = (2*N_hap*r)**2 * (2*r*l+1/N_hap)**2 * t**2 + 2*(2*N_hap*r)**2 * (2*r*l+1/N_hap)*t + 2*(2*N_hap*r)**2
            b2 = (2*N_hap*r*l+1)**3
            b3 = l*exp(-(2*r*l+1/N_hap)*t)
            return b1*b3/b2

        def J(t):
            c1 = 2*N_hap*r+2*N_hap*r*(2*r*u+1/N_hap)*t
            c2 = (2*N_hap*r*u+1)**2
            c3 = exp(-(2*r*u+1/N_hap)*t)
            return c1*c3/c2
        # 条件长度分布        
        def conditional_length_distribution(tau,s):
            density = (I(0,s)-2*m*(1-m)*I(tau,s))/(s*(J(0)-2*m*(1-m)*J(tau)))
            return density
        
        def truncation_ibd_sharing(tau):
            "replace 2Nru+1 with 2Nru"
            trunc_ibdsharing=1.0/N_hap*(1.0/(r*u)-2*m*(1-m)*(1+r*u*tau)/(r*u)*exp(-2*r*u*tau))
            return trunc_ibdsharing
    
        def truncation_length_distribution(tau,s):
            "replace 2Nru+1 with 2Nru"
            a = 2*u*u*(1-2*m*(1-m)*(2*(r*s*tau)**2+2*r*s*tau+1)*exp(-2*r*s*tau))
            b = s**3*(1-2*m*(1-m)*(1+2*r*u*tau)*exp(-2*r*u*tau))
            trunc_density = a/b
            return trunc_density
        
        density_all = []
        for sample in sample_all:
            desity = truncation_length_distribution(tau_a,sample)
            density_all.append(desity)
        
        ibd = truncation_ibd_sharing(tau_a)

        loss = -beta*np.sum(np.log(density_all)) + gamma*(ibd - ibd_label)**2
        return loss

    def func_eval(in_x, grad=None):  # use this func to see the two parts of the loss function
        j=0
        for i in range(3):
            if fix[i] == 1:
                globals()[fix_name[i]]=fix_val[i]
            elif fix[i] == 0:
                globals()[fix_name[i]]=in_x[j]
                j+=1


        # fixed par
        L = 100000000
        u = 0.01 # threshold (M)：1M = 1cM centi-Morgan
        r = 1 # recombination rate 1cM/Mb

        gamma = true_par['gamma']
        beta = true_par['beta']
        ibd_label = true_par['ibd_label']
        #eps = 1e-23  # 为了计算log（f）时候的稳定，变成了log(f+eps)
        sample_all = data

        def H(t):
            a2 = 2*N_hap*r*u + 1
            a1 = 4*N_hap*r*u + 1 + a2*2*r*u*t
            a3 = exp(-(2*r*u+1/N_hap)*t)
            result = a1*a3/(a2*a2)
            return result

        # ibd sharing
        def ibd_sharing(tau):
            ibdsharing = H(0) - 2*m*(1-m)*H(tau)
            return ibdsharing

        def I(t,l):
            b1 = (2*N_hap*r)**2 * (2*r*l+1/N_hap)**2 * t**2 + 2*(2*N_hap*r)**2 * (2*r*l+1/N_hap)*t + 2*(2*N_hap*r)**2
            b2 = (2*N_hap*r*l+1)**3
            b3 = l*exp(-(2*r*l+1/N_hap)*t)
            return b1*b3/b2

        def J(t):
            c1 = 2*N_hap*r+2*N_hap*r*(2*r*u+1/N_hap)*t
            c2 = (2*N_hap*r*u+1)**2
            c3 = exp(-(2*r*u+1/N_hap)*t)
            return c1*c3/c2
        # 条件长度分布        
        def conditional_length_distribution(tau,s):
            density = (I(0,s)-2*m*(1-m)*I(tau,s))/(s*(J(0)-2*m*(1-m)*J(tau)))
            return density
        
        def truncation_ibd_sharing(tau):
            "replace 2Nru+1 with 2Nru"
            trunc_ibdsharing=1.0/N_hap*(1.0/(r*u)-2*m*(1-m)*(1+r*u*tau)/(r*u)*exp(-2*r*u*tau))
            return trunc_ibdsharing
    
        def truncation_length_distribution(tau,s):
            "replace 2Nru+1 with 2Nru"
            a = 2*u*u*(1-2*m*(1-m)*(2*(r*s*tau)**2+2*r*s*tau+1)*exp(-2*r*s*tau))
            b = s**3*(1-2*m*(1-m)*(1+2*r*u*tau)*exp(-2*r*u*tau))
            trunc_density = a/b
            return trunc_density
        
        density_all = []
        for sample in sample_all:
            desity = truncation_length_distribution(tau_a,sample)
            density_all.append(desity)
        
        ibd = truncation_ibd_sharing(tau_a)

        loss = -beta*np.sum(np.log(density_all)) + gamma*(ibd - ibd_label)**2
        loss1 = -np.sum(np.log(density_all))
        loss2  = (ibd - ibd_label)**2
        return [loss, loss1, loss2, ibd]


    # in_x: [N_hap, tau_a, m]
    N_hap_bound = true_par['N_hap_bound']
    tau_a_bound = true_par['tau_a_bound']
    m_bound = true_par['m_bound']
    opt = nlopt.opt(nlopt.LN_BOBYQA,3-sum(fix))


    opt.set_min_objective(func)
    #stopval = 0.01
    #opt.set_stopval(stopval)
    tol = 0.00001
    opt.set_xtol_rel(tol)
    #opt.get_xtol_rel()

    upper_b = []
    lower_b = []
    for i in range(3):
        if fix[i] == 0:
            upper_b.append(true_par[fix_name[i]+'_bound'][1])
            lower_b.append(true_par[fix_name[i]+'_bound'][0])


    opt.set_lower_bounds(lower_b)
    opt.set_upper_bounds(upper_b)

    init_par = true_par['init_par']
    init = []
    for i in range(3):
        if fix[i] == 0:
            init.append(init_par[i])
    xopt = opt.optimize(init)

    minf = opt.last_optimum_value()

    [loss, loss1, loss2, ibd_aproximated] = func_eval(xopt)

    j=0
    for i in range(3):
        if fix[i] == 0:
            globals()[fix_name[i]+'_opt'] = xopt[j]
            j+=1
        else:
            globals()[fix_name[i]+'_opt'] = fix_val[i]


    results = {
        "opt N_hap aproximated":N_hap_opt,
        "opt tau_a aproximated":tau_a_opt,
        "opt m aproximated":m_opt,
        "opt loss":minf,
        "loss1":loss1,
        "loss2":loss2,
        "ibd aproximated":ibd_aproximated
    }

    results = dict(**true_par, **results) # 参数字典和结果字典拼接
    # print(results)

    return results

save_dict ={}
#save_dict['args'] = args


results = update_step(data, true_par)
true_par_temp = true_par.copy()

save_dict['round0'] = results

for i in range(1):
    
    true_par_temp['fix'] = [1,0,1] # update t
    true_par_temp['gamma'] = 0
    true_par_temp['beta'] = 1
    true_par_temp['init_par'] = [true_par['N_hap_true'], \
                                 results['opt tau_a aproximated'], \
                                 results['opt m aproximated'] ]
    true_par_temp['fix_val'] = [ results['N_hap_true'], \
                                 results['opt tau_a aproximated'], \
                                 results['opt m aproximated'] ]
    results = update_step(data, true_par_temp)
    #print('round t'+str(i+1), results)
    save_dict['round t'+str(i+1)] = results.copy()
    
    
    true_par_temp['fix'] = [0,1,1] # update n,m
    true_par_temp['gamma'] = 1
    true_par_temp['beta'] = 0
    true_par_temp['init_par'] = [ results['opt N_hap aproximated'], \
                                 results['opt tau_a aproximated'], \
                                 results['opt m aproximated'] ]
    true_par_temp['fix_val'] = [ results['opt N_hap aproximated'], \
                                 results['opt tau_a aproximated'], \
                                 results['opt m aproximated'] ]
    results = update_step(data, true_par_temp)
    #print('round mn'+str(i+1), results)
    save_dict['round mn'+str(i+1)] = results.copy()

print(save_dict)





def save_results(result_path, results):
    # save results as a json file
    with open(result_path, 'w') as f:
        json.dump(results, f, sort_keys=False, indent=4)
    print("results saved as %s" % result_path)
save_results(args.result_path, save_dict)
