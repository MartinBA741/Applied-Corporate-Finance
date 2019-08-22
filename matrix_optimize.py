import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import math
import datetime 
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy.stats import describe
from pylab import plt, mpl
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
import random

def port_ret(mu, weights):
    '''Compute portfolio return'''
    return np.sum(mu * weights) *252


def port_vol(omega, weights):
    '''Compute portfolio variance'''
    return np.sqrt(np.dot(weights.T, np.dot(omega * 252, weights)))


def sharpe_ratio(mu, omega, weights):
    '''The Sharpe Ratio (negative) to be minimized'''
    return -port_ret(mu, weights) / port_vol(omega, weights)


def min_vol(noa, omega):
    '''Solve for the portfolio weights that minimize the portfolio volatitility, given the var-cov matrix.'''
    
    # a. define vectors to store values
    b = np.zeros(noa+1)
    b[-1] = 1
    one = np.ones(noa)
    one_0 = np.ones(noa+1)
    one_0[-1] = 0
    z_m = np.ones(noa+1)     # weights vector

    Omega = omega.values

    A_m = 2*Omega
    A_m = np.column_stack((A_m, one))
    A_m = np.vstack([A_m, one_0])   

    # Solve for weights z securing minimum vol:
    z_m = np.linalg.pinv(A_m).dot(b)

    return z_m.round(3)[:noa]


def eff_vol(noa, omega, mu, target_return):
    '''Solve the portfolio weights that minimize volatility given a target return''' 

    Mu = np.zeros(noa+1)
    Mu[0:noa] = mu
    one_0 = np.ones(noa+1)
    one_0[-1] = 0
    one_00 = np.ones(noa+2)
    one_00[-2:] = 0

    b_e = np.zeros(noa+2)
    b_e[-2] = target_return
    b_e[-1] = 1

    z_e = np.ones(noa+2)

    A_e = 2*omega
    A_e = np.column_stack((A_e, Mu[0:noa]))
    A_e = np.vstack([A_e, Mu.T])
    A_e = np.column_stack((A_e, one_0))
    A_e = np.vstack([A_e, one_00])
    
    # Solve optimal weight
    z_e = np.linalg.pinv(A_e).dot(b_e)

    return z_e.round(3)[:noa]


def efficient_frontier(min_weights, eff_weights, omega, mu):
    '''Compute various linear combinations of min vol and eff portfolios. 
    - Also it computes all the frontier annualized colatilities and returns'''
    a = np.linspace(-2, 2, 200) 
    z = []                     # all different weight combinations
    frontier_vol = []

    for i,j in enumerate(a):
        z.append(j*min_weights + (1-j)*eff_weights)
        frontier_vol.append(port_vol(omega=omega, weights=z[i]))

    # The returns eff frontier
    frontier_ret = np.array(z).dot(np.array(mu))*252
    return z, np.array(frontier_vol), frontier_ret   


def tangent_portfolio(front_vol, front_ret):
    # Index position of minimum vol portfolio
    ind = np.argmin(np.flip(front_vol))                      # Find minimum variance portfolio - the # in list
    evols = np.flip(front_vol)[ind:]                         # Relevant portfolio vol from min vol and onwards from min vol and on...
    erets = np.flip(front_ret)[ind:]                         # Relevant portfolio ret

    tck = sci.splrep(evols, erets)                                # Cubic splines interpolation on these values

    def f(x):
        '''Effecient frontier function (splines approximation).'''
        return sci.splev(x, tck, der=0)

    def df(x):
        '''First derivative of Efficient frontier fct.'''
        return sci.splev(x, tck, der=1)

    rf=0.0
    def equations(p, rf=0.0):
        eq1 = rf - p[0]
        eq2 = rf + p[1] * p[2] - f(p[2])
        eq3 = p[1] - df(p[2])
        return eq1, eq2, eq3

    opt = sco.fsolve(equations, [0.0, 1.42, 17.74])

    print('Optimal parameter values = ', opt.round(4))

    print('Check that eq holds:', np.round(equations(opt), 6))

    return opt, evols, erets, f(opt[2])