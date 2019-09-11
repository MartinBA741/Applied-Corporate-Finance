# Maximum likelihood
import numpy as np
import scipy.optimize as opt

r = train[mylist[1]]

def GJR_sigma(mu, omega, alpha, gamma, beta, r):
    sigma_2 = []
    sigma_2.append(omega / (1 - alpha - gamma/2 - beta))

    for i in range(len(r)):
        if r[i]>0:
            sigma_2.append(omega + alpha* r[i]**2 + beta* sigma_2[i])
        else:
            sigma_2.append(omega + (alpha+gamma)*r[i]**2 + beta*sigma_2[i])
    
    return np.array(sigma_2)


def GJR_loglik(vP, r):
    mu = vP[0]
    omega = vP[1]
    alpha = vP[2]
    gamma = vP[3]
    beta = vP[4]
    
    sigma_2 = GJR_sigma(mu, omega, alpha, gamma, beta, r)
    
    #LogL = - np.sum(-np.log(np.array(sigma_2[1:])) - (r)**2/np.array(sigma_2[1:]))  # minus first to minimize
    LogL = 0.5 * (np.sum(np.log(2*math.pi)+np.log(sigma_2[:-1]) + ((r-mu)**2)/sigma_2[:-1])) 
    
    return LogL


cons = ({'type': 'ineq', 'func': lambda x: np.array(x)})
bnds = ((None,None),(0.0001,None),(0.00001,None),(0.00001,None),(0.00001,None))
vP0 = (0.026, 0.044, 0.103, 0.086, 0.788)

#OWN_res = opt.minimize(GJR_loglik, vP0, args=r, bounds = bnds, options={'disp':True}) 

OWN_res = opt.minimize(GJR_loglik, vP0, args=(r), method='BFGS', jac=None, tol=None, callback=None, 
         options={'gtol': 1e-05, 'norm': 1000, 'eps': 1.4901161193847656e-08, 'maxiter': None, 
                  'disp': False, 'return_all': False})