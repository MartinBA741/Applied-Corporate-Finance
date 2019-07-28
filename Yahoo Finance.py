
#%%
get_ipython().run_line_magic('reset', '')


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import math
import datetime 
import scipy.optimize as sco
import scipy.interpolate as sci
from pylab import plt, mpl
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
from arch import arch_model
# ARCH does not import



#%%



#%%
plt.style.use('seaborn')

#%% [markdown]
# https://www.bankrate.com/investing/best-etfs/ 
# 
# Invesco QQQ Trust (QQQ) - Non-financial Nasdaq - Tech-heavy
# 
# iShares Core S&P 500 ETF (IVV) - Black Rock - very low cost
# 
# ProShares VIX Short-Term Futures ETF (VIXY) - ETF on VIX (volatility)
# 
# Vanguard High Dividend Yield (VYM) - FTSE high-dividend paying stocks
# 
# Vanguard Health Care Index Fund (VHT)
# 
# https://www.thebalance.com/best-etfs-4173857
# 
# iShares Russell 2000 ETF (IWM) - Small cap stock
# 
# SPDR Gold Trust (GLD)
# 
# Vanguard FTSE Developed Markets (VEA) - Develeoped contries except USA
# 
# 
# 
# 
# Vanguard FTSE Europe Index Fund ETF Shares (VGK)
# 
# iShares Short Treasury Bond ETF (SHV)
# 
# iShares 20+ Year Treasury Bond ETF (TLT)
# 
# iShares Core € Corp Bond UCITS ETF (IEAC)

#%%
# List shares of interest as Yahoo Fiance "Ticker"/"Symbol"
mylist = ['QQQ', 'IVV', 'VIXY', 'VYM', 'VHT', 'IWM', 'GLD', 'VEA', 'VGK', 'SHV', 'TLT']
        # ["danske.co", "MAERSK-B.CO", 'CHR.CO', "NZYM-B.CO", 'LUN.CO', 'GN.CO', 'CARL-B.CO', 'GEN.CO', 'ORSTED.CO']
noa = len(mylist)

# Create DataFrame to store data
df = pd.DataFrame()
df_change = pd.DataFrame()
rets = pd.DataFrame()


#%%
for x in mylist:
    df[x] = pdr.get_data_yahoo(x, start=datetime.datetime(2011, 1, 1))['Adj Close'] 
    df_change['pct_c_{}'.format(x)] = df[x].pct_change()   # 
    rets[x] = np.log(df[x] / df[x].shift(1))


#%%
rets.describe().round(4)


#%%
mpl.rcParams['font.family'] = 'serif'
get_ipython().run_line_magic('matplotlib', 'inline')
rets.plot(lw=1.0, figsize=(10,6))  #.cumsum()


#%%
# Random assigned weight
weights = np.random.random(noa)
weights /= np.sum(weights) 
weights


#%%
# Expected portfolio return (annualized)
EPR = np.sum(rets.mean() * weights) * 252

# Portfolio Variance annualized
PV = np.dot(weights.T, np.dot(rets.cov() * 252, weights))

# Portfolio Volatility (Standard deviation)
P_std = math.sqrt(PV)

display(EPR, PV, P_std)


#%%
# Monte Carlo simulation

def port_ret(weights):
    return np.sum(rets.mean() * weights) *252

def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    
p_rets = []
p_vols = []
    
for p in range (50000):
    weights = np.random.random(noa)
    weights /= np.sum(weights) 
    p_rets.append(port_ret(weights))
    p_vols.append(port_vol(weights))
    
p_rets = np.array(p_rets)
p_vols = np.array(p_vols)


#%%
plt.figure(figsize=(10,6))
plt.scatter(p_vols, p_rets, c=p_rets / p_vols, marker = '.', cmap='coolwarm')
plt.xlabel('Expected volatility')
plt.ylabel('Expected return')
plt.colorbar(label='Sharpe ratio')


#%%
# Maximisation of Sharpe Ratio 

# Function to minimize
def min_func_sharpe(weights):
    return -port_ret(weights) / port_vol(weights)

# Equality constraint
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for parameters
bnds = tuple((0,1) for x in range(noa))

# Equal weights 
eweights = np.array(noa * [1. / noa,])

-min_func_sharpe(eweights)


#%%
# Max return
#%%time
opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)
# print(opts)
print('opts = opt. port. weights = ', opts['x'].round(3))
print('port_ret = ', port_ret(opts['x']).round(3))
print('port_vol = ', port_vol(opts['x']).round(3))
print('Maximum Sharpe Ratio = ', port_ret(opts['x']) / port_vol(opts['x']))


#%%
# Minimize Vol
optv = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
print('weights = ', optv['x'].round(3))
print('vol = ', port_vol(optv['x']).round(3))
print('return = ', port_ret(optv['x']).round(3))
print('Sharpe ratio = ', (port_ret(optv['x']) / port_vol(optv['x'])).round(3))


#%%
# Effecient Frontier

# Binding Constraint for efficient frontier
cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds
bnds = tuple((0,1) for x in weights)

# Target Return
trets = np.linspace(0.0, 0.15, 50)
tvols = []
for tret in trets:
    res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)


#%%
plt.figure(figsize=(10,6))
plt.scatter(p_vols, p_rets, c=p_rets / p_vols, marker='.', alpha=0.8, cmap='coolwarm')
plt.plot(tvols, trets, 'b', lw=4.0)
plt.plot(port_vol(opts['x']), port_ret(opts['x']), 'g*', markersize=15.0)
plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize=15.0)
plt.xlabel('Expected volatility')
plt.ylabel('Expected return')
plt.colorbar(label='Sharpe ratio')


#%%
# Capital Market Line

# Index position of minimum vol portfolio
ind = np.argmin(tvols)
evols = tvols[ind:]     # Relevant portfolio vol
erets = trets[ind:]     # Relevant portfolio ret

tck = sci.splrep(evols, erets)   # Cubic splines interpolation on these values

def f(x):
    '''Effecient frontier function (splines approximation).'''
    return sci.splev(x, tck, der=0)

def df(x):
    '''First derivative of Efficient frontier fct.'''
    return sci.splev(x, tck, der=1)


#%%



#%%
# page 427
def equations(p, rf=0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

opt = sco.fsolve(equations, [0.01, 0.5, 0.15])
opt

np.round(equations(opt), 6)


#%%
plt.figure(figsize=(10,6))
plt.scatter(p_vols, p_rets, c=(p_rets-0.01) / p_vols,                      # Check rf = 0.01
            marker='.', alpha=0.8, cmap='coolwarm')
plt.plot(evols, erets, 'b', lw=4.0)

cx = np.linspace(0.0, 0.15)
plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5)

plt.plot(opt[2], f(opt[2]), 'y*', markersize=15)

plt.grid(True)

plt.axhline(0,color='k', ls='--', lw=2.0)
plt.axvline(0,color='k', ls='--', lw=2.0)

plt.xlabel('Expected volatility')
plt.ylabel('Expected return')
plt.colorbar(label='Sharpe ratio')


#%%


#%% [markdown]
# ## Moving on to GARCH modelleing

#%%



#%%
squared_rets = rets ** 2


#%%
plot_acf(squared_rets['QQQ'][-100:])
pyplot.show()


#%%
# define GARCH model
model = arch_model(train, mean='Zero', vol='GARCH', p=15, q=15)


#%%



