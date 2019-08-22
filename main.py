# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython


#%%
# %reset


#%%
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


#%%
from arch import arch_model
import matrix_optimize
from matrix_optimize import port_ret, port_vol, min_vol, eff_vol, sharpe_ratio, efficient_frontier, tangent_portfolio


#%%
plt.style.use('seaborn')


#%%
mydict = {'IGF': 'Global Infrastructure', 
          'IFGL':'International Developed Real Estate ETF',
          'IXN': 'iShares Global Tech ETF',
          'IXJ': 'iShares Global Healthcare ETF',
          'VDE': 'Vanguard Energy ETF',
          'KXI': 'iShares Global Consumer Staples ETF',
          'SPGP':'iShares Gold Producers UCITS ETF',
          'VIS': 'Vanguard Industrial ETF',
          'RXI': 'iShares Global Consumer Discretionary ETF'
         }
# List shares of interest as Yahoo Fiance "Ticker"/"Symbol"
mylist = list(mydict)[:]
noa = len(mylist)

# Load Data
rets = pd.read_excel(r'C:\Users\Ma-Bi\OneDrive\Kandidat\Topics in Financial Econometrics - Seminar\log_rets.xlsx', 
                   sheet_name='Sheet', index=True, index_col=[0])
rets.head()


#%%
TABLE = rets.describe()[1:3].round(3).transpose()
TABLE['skewness'] = np.array(rets.skew().round(3))
TABLE['kurtosis'] = np.array(rets.kurt().round(3))
TABLE['ticker'] = np.array(list(mydict)[:])
TABLE = TABLE[['ticker', 'mean', 'std', 'skewness', 'kurtosis']]
TABLE = TABLE.set_index(np.array(list(mydict.values())[:]))
display(TABLE)
# print(TABLE.to_latex(index=True))


#%%
fig, axs = plt.subplots(1,2,figsize=(15,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

mpl.rcParams['font.family'] = 'serif'
#%matplotlib inline

plt.subplot(1, 2, 1)
plt.plot(rets.cumsum(), lw=1.0)
plt.ylabel('Accumulated daily log-return')
plt.legend(list(mydict))

plt.subplot(2, 2, 2)
plt.plot(rets.iloc[:, 0:5], lw=1.0)
plt.ylabel('Daily log-return, %')
plt.legend(list(mydict)[0:5])


plt.subplot(2, 2, 4)
plt.plot(rets.iloc[:, 5:], lw=1.0)
plt.ylabel('Daily log-return, %')
plt.legend(list(mydict)[5:])

#plt.show()
plt.savefig('historic_development.png', dpi=300, bbox_inches='tight')


#%%
# Random assigned weight
weights = np.random.random(noa)
weights /= np.sum(weights) 
weights


#%%
print('Covariance Matrix \n')
print(np.round(rets.cov(), 2).to_latex())
# plt.matshow(rets.cov(), cmap='coolwarm')
# plt.show()


#%%
# Monte Carlo simulation

p_rets = []
p_vols = []
    
for p in range (50000):
    weights = np.random.random(noa)
    weights /= np.sum(weights) 
    p_rets.append(port_ret(mu=rets.mean(),weights=weights))
    p_vols.append(port_vol(omega=rets.cov(),weights=weights))
    
p_rets = np.array(p_rets)
p_vols = np.array(p_vols)

#%% [markdown]
# # MATRIX
# 
#%% [markdown]
# ## Historically based minimum vol portfolio

#%%
# a. Find minimum variance weights
hist_w_min = matrix_optimize.min_vol(noa=noa, omega=rets.cov())

# b. Compute annualized portfolio return
hist_ret_min = port_ret(mu=rets.mean(),weights=hist_w_min)

# c. Compute annualized portfolio volatility
hist_vol_min = port_vol(omega=rets.cov(),weights=hist_w_min)

# d. Print results:
print('Minimum vol weights', hist_w_min)
print('Annualized portfolio return = ', hist_ret_min.round(3))
print('Annualized portfolio vol = ', hist_vol_min.round(3))
print('Maximum Sharpe Ratio = ',(hist_ret_min/hist_vol_min).round(3),
      '=', -sharpe_ratio(mu=rets.mean(), omega=rets.cov(),weights=hist_w_min).round(3))

#%% [markdown]
# ## Historically based efficient portfolio - given target return

#%%
# a. Find efficient weights given target return
hist_w_eff = matrix_optimize.eff_vol(noa=noa, omega=rets.cov(), mu=rets.mean(), target_return=30/252)

# b. Compute annualized portfolio return
hist_ret_eff = port_ret(mu=rets.mean(),weights=hist_w_eff)

# c. Compute annualized portfolio volatility
hist_vol_eff = port_vol(omega=rets.cov(),weights=hist_w_eff)

# d. Print resuts:
print('Efficient weights', hist_w_eff)
print('Annualized portfolio return = ', hist_ret_eff.round(3))
print('Annualized portfolio vol = ', hist_vol_eff.round(3))
print('Maximum Sharpe Ratio = ',(hist_ret_eff/hist_vol_eff).round(3),
      '=', -sharpe_ratio(mu=rets.mean(), omega=rets.cov(),weights=hist_w_eff).round(3))

#%% [markdown]
# # Effecient frontier

#%%
hist_z, hist_front_vol, hist_front_ret = efficient_frontier(min_weights=hist_w_min,
                                                   eff_weights=hist_w_eff, omega=rets.cov(), mu=rets.mean())


#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))

plt.plot(hist_front_vol, hist_front_ret)

plt.plot(hist_vol_eff, hist_ret_eff, 'b*', markersize=15.0, label='Target return portfolio')
plt.plot(hist_vol_min, hist_ret_min, 'r*', markersize=15.0, label='min vol. portfolio')

# plot individual ETFs
for i,type in enumerate(mylist):
    x2 = np.sqrt((rets[type]).var()*252)
    y2 = rets[type].mean()*252
    plt.scatter(x2, y2, marker='o')  
    plt.text(x2+0.05, y2+0.05, type, fontsize=9)
    
plt.xlabel('Annualized Expected Volatility, $\%$')
plt.ylabel('Annualized Expected Return, $\%$')
plt.title('Annualized Portfolio Return and Volatility')

plt.legend()
plt.show()

#%% [markdown]
# ## Tangent Portfolio

#%%
opt, evols, erets, f_tangent = tangent_portfolio(front_vol=hist_front_vol, front_ret=hist_front_ret)


#%%
rf=0
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))

plt.plot(evols, erets, 'b', lw=3.0, label='Effecient frontier')

cx = np.linspace(0.0, 20.0)
plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5)

plt.plot(opt[2], f_tangent, 'g*', markersize=15, label='Tangent Portfolio')
plt.plot(hist_vol_min, hist_ret_min, 'r*', markersize=15.0, label='min vol. portfolio')

# plot individual ETFs
for i,type in enumerate(mylist):
    x2 = np.sqrt((rets[type]).var()*252)
    y2 = rets[type].mean()*252
    plt.scatter(x2, y2, marker='o')  
    plt.text(x2+0.05, y2+0.05, type, fontsize=9)


plt.grid(True)
axes = plt.gca()
axes.set_xlim([0,22])
axes.set_ylim([-1.5,28])
plt.xlabel('Annualized Expected Volatility, $\%$')
plt.ylabel('Annualized Expected Return, $\%$')
plt.title('Historical Mean-Variance optimization')
plt.legend()

plt.savefig('tangent_port.png', dpi=300, bbox_inches='tight')


#%%
# a. Find tangent portfolio weight
ind = np.where(np.array(hist_front_vol.round(1)==opt[2].round(1)))[0][0]
hist_w_tan = hist_z[ind]

# b. Compute annualized portfolio return
hist_ret_tan = port_ret(mu=rets.mean(),weights=hist_w_tan)

# c. Compute annualized portfolio volatility
hist_vol_tan = port_vol(omega=rets.cov(),weights=hist_w_tan)

# d. Print resuts:
print('Efficient weights', hist_w_tan.round(3))
print('Annualized portfolio return = ', hist_ret_tan.round(3))
print('Annualized portfolio vol = ', hist_vol_tan.round(3))
print('Maximum Sharpe Ratio = ',(hist_ret_tan/hist_vol_tan).round(3),
      '=', -sharpe_ratio(mu=rets.mean(), omega=rets.cov(),weights=hist_w_tan).round(3))

#%% [markdown]
# 
# # Conclusion

#%%
# Weights: minimum vol, max Sharpe
conc_1 = np.vstack([mylist, hist_w_min.round(3)])
conc_1 = np.vstack([conc_1, hist_w_eff.round(3)])
conc_1 = np.vstack([conc_1, hist_w_tan.round(3)])

# Portfolio return
P_R_1 = np.ones(4)
P_R_1[1:] = np.array([hist_ret_min , hist_ret_eff, hist_ret_tan]).round(3)
conc_1 = np.column_stack((conc_1, P_R_1))

# Portfolio volatility
P_V_1 = np.ones(4)
P_V_1[1:] = np.array([hist_vol_min , hist_vol_eff, hist_vol_tan]).round(3)
conc_1 = np.column_stack((conc_1, P_V_1))

# Portfolio Sharpe Ratio
P_SR_1 = np.ones(4)
P_SR_1[1:] = np.array([-sharpe_ratio(mu=rets.mean(), omega=rets.cov(),weights=hist_w_min),
                       -sharpe_ratio(mu=rets.mean(), omega=rets.cov(),weights=hist_w_eff), 
                       -sharpe_ratio(mu=rets.mean(), omega=rets.cov(),weights=hist_w_tan)]).round(3)
conc_1 = np.column_stack((conc_1, P_SR_1))

conc_1 = pd.DataFrame(conc_1, index=['Ticker', 'Minimum vol. portfolio', 
                                     'Target return portfolio', 'Tangent portfolio'])


conc_1.columns = conc_1.iloc[0]
conc_1 = conc_1.drop(conc_1.index[0])
conc_1.columns.values[[-3, -2, -1]] = ['Exp. return', 'Exp. volatility', 'Exp. SR']
conc_1

#%% [markdown]
# ## Moving on to GARCH modelleing

#%%
n_time = 100
n_test = 30
H = -n_time+n_test


parameters = np.zeros((noa,4))
forecasts = np.zeros((noa,n_test))
forecast_variance = np.zeros(noa)
std_resid = np.zeros((noa,len(rets['IGF'][:-n_time])))



for i,n in enumerate(mylist):
    # split into train/test = in-sample/out-of-sample
    train, test = rets[n][:-n_time], rets[n][-n_time : H] 

    # define GARCH model
    model = arch_model(train, mean='Constant', vol='GARCH', p=1, q=1)
    # model = arch_model(train, p=1, o=1, q=1) # GJR Garch - just change parameters to 5 instead of 4

    # fit model
    res = model.fit(disp='off')

    # Save parameter estimates
    parameters[i][:] = res.params[:]
    
    # Save standardized residuals
    std_resid[i][:] = res.resid / res.conditional_volatility
    
    # forecast variance
    forecasts[i][:] = res.forecast(horizon=n_test).variance.dropna().values.T[:,0]
    forecast_variance[i] = forecasts[i].mean() # Each asset's forecasted variance for period n_test period ahead

#%% [markdown]
# ## Forecast - not FHS

#%%
# All mu (expected return)
F_mu = parameters[:, 0]
display(F_mu)

# Forecasted variances
F_var = forecast_variance
display(F_var)

#parameters


#%%
Omega_forecast = rets.cov()
np.fill_diagonal(Omega_forecast.values, F_var)
Omega_forecast.round(2)

#%% [markdown]
# ## Simulation FHS
#%% [markdown]
# \begin{align}
#     r_{t+1} &= \mu + \sigma_{t+1} z_{t+1} \\
#     \sigma^2_{t+1} &= \omega + \alpha r_t^2 + \beta \sigma_t^2  \\
#     z &\sim \textit{i.i.d. } \mathcal{N}(0,1)
# \end{align}

#%%
import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl


#%%
mpl.rcParams['font.family'] = 'serif'
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
# Filtered Historical Simulation

#npr.seed(741)
N = n_test # 10

# Create dictionaries
SIM_RET = {}
SIM_SIGMA = {}
ACC_RET = {}
AVG_VOL = {}

# Loop over assets
for j,x in enumerate(mylist):  #range(noa):

    # Define parameters for the given asset
    param = parameters[j]
    mu = param[0]
    omega = param[1]
    alpha = param[2]
    beta = param[3]
    acc_ret = []                                                   # space for accumulated returns
    avg_vol = []
    
    # Loop for simulations
    for i in range(50000):
        z = random.sample(list(std_resid[j][:-n_test]), N)            # Bootstrap random draw from std.res.
        sim_ret = np.empty(N)                                          # Define empty vector for simulated return
        
        
        # Define vector for simulated sigma_(t+1) for GARCH(1,1)
        sim_sigma = np.empty(N)
        sim_sigma[0] = ((train[-1] - mu)/ std_resid[j][-n_test])       # sigma_t+1 is predetermined
        
        # Loop for horizon
        for n in range(N-1):
            sim_ret[n] = mu + sim_sigma[n] * z[n]
            sim_sigma[n+1] =  math.sqrt(omega + alpha * sim_ret[n]**2 + beta * sim_sigma[n]**2)
        
        acc_ret.append(np.sum(sim_ret))                               # all 50.000 accumulated return
        avg_vol.append(sum(sim_sigma)/len(sim_sigma))                 # all 50.000 average sigma 

    ACC_RET["asset_{0}".format(x)] = acc_ret
    AVG_VOL["asset_{0}".format(x)] = avg_vol


#%%
plt.figure(figsize=(10,6))
for i in mylist:
    plt.plot(AVG_VOL['asset_%s' %i], label='Asset %s vol' %i)
plt.legend()
plt.show()

#%% [markdown]
# Now We have obtained estimates for the 10-period expected return and variance for each asset.
# 
# Assuming constant correlation between the assets, we have obtained a new covariance matrix. And expected return, so we can construct a new effecient portfolio. 

#%%
# N-day accumulated return (average):
FHS_mu = pd.DataFrame.from_dict(ACC_RET).mean()/N
#display(FHS_mu)   # compare to rets.mean()

# Daily average volatilty:
FHS_var = pd.DataFrame.from_dict(AVG_VOL).mean()**2
#display(FHS_var)  # compare to rets.var()


#%%
FHS_omega = rets.cov()
np.fill_diagonal(FHS_omega.values, FHS_var)
FHS_omega.round(2)

#%% [markdown]
# ## FHS based minimum vol portfolio

#%%
# a. Find minimum variance weights
FHS_w_min = matrix_optimize.min_vol(noa=noa, omega=FHS_omega)

# b. Compute annualized portfolio return
FHS_ret_min = port_ret(mu=FHS_mu,weights=FHS_w_min)

# c. Compute annualized portfolio volatility
FHS_vol_min = port_vol(omega=FHS_omega,weights=FHS_w_min)

# d. Print results:
print('Minimum vol weights', FHS_w_min)
print('Annualized portfolio return = ', FHS_ret_min.round(3))
print('Annualized portfolio vol = ', FHS_vol_min.round(3))
print('Maximum Sharpe Ratio = ',(FHS_ret_min/FHS_vol_min).round(3),
      '=', -sharpe_ratio(mu=FHS_mu, omega=FHS_omega,weights=FHS_w_min).round(3))

#%% [markdown]
# ## FHS efficient portfolio - given target return

#%%
# a. Find efficient weights given target return
FHS_w_eff = matrix_optimize.eff_vol(noa=noa, omega=FHS_omega, mu=FHS_mu, target_return=30/252)

# b. Compute annualized portfolio return
FHS_ret_eff = port_ret(mu=FHS_mu, weights=FHS_w_eff)

# c. Compute annualized portfolio volatility
FHS_vol_eff = port_vol(omega=FHS_omega, weights=FHS_w_eff)

# d. Print resuts:
print('Efficient weights', FHS_w_eff)
print('Annualized portfolio return = ', FHS_ret_eff.round(3))
print('Annualized portfolio vol = ', FHS_vol_eff.round(3))
print('Maximum Sharpe Ratio = ',(FHS_ret_eff/FHS_vol_eff).round(3),
      '=', -sharpe_ratio(mu=FHS_mu, omega=FHS_omega,weights=FHS_w_eff).round(3))

#%% [markdown]
# # Effecient frontier

#%%
FHS_z, FHS_front_vol, FHS_front_ret = efficient_frontier(min_weights=FHS_w_min,
                                                   eff_weights=FHS_w_eff, omega=FHS_omega, mu=FHS_mu)


#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))

plt.plot(FHS_front_vol, FHS_front_ret)

plt.plot(FHS_vol_eff, FHS_ret_eff, 'b*', markersize=15.0, label='Target return portfolio')
plt.plot(FHS_vol_min, FHS_ret_min, 'r*', markersize=15.0, label='min vol. portfolio')

# plot individual ETFs
for i,type in enumerate(mylist):
    x2 = np.sqrt((rets[type]).var()*252)
    y2 = rets[type].mean()*252
    plt.scatter(x2, y2, marker='o')  
    plt.text(x2+0.05, y2+0.05, type, fontsize=9)
    
plt.xlabel('Annualized Expected Volatility, $\%$')
plt.ylabel('Annualized Expected Return, $\%$')
plt.title('Annualized Portfolio Return and Volatility')

plt.legend()
plt.show()

#%% [markdown]
# ## Tangent Portfolio

#%%
FHS_opt, FHS_evols, FHS_erets, FHS_f_tangent = tangent_portfolio(front_vol=FHS_front_vol, front_ret=FHS_front_ret)


#%%
rf=0
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))

plt.plot(FHS_evols, FHS_erets, 'b', lw=3.0, label='Effecient frontier')

cx = np.linspace(0.0, 40.0)
plt.plot(cx, FHS_opt[0] + FHS_opt[1] * cx, 'r', lw=1.5)

plt.plot(FHS_opt[2], FHS_f_tangent, 'g*', markersize=15, label='Tangent Portfolio')
plt.plot(FHS_vol_min, FHS_ret_min, 'r*', markersize=15.0, label='min vol. portfolio')

# plot individual ETFs
for i,type in enumerate(mylist):
    x2 = np.sqrt((rets[type]).var()*252)
    y2 = rets[type].mean()*252
    plt.scatter(x2, y2, marker='o')  
    plt.text(x2+0.05, y2+0.05, type, fontsize=9)


plt.grid(True)
axes = plt.gca()
axes.set_xlim([0,25])
axes.set_ylim([-1.5,40])
plt.xlabel('Annualized Expected Volatility, $\%$')
plt.ylabel('Annualized Expected Return, $\%$')
plt.title('Annualized Portfolio Return and Volatility')
plt.legend()

plt.savefig('tangent_port.png', dpi=300, bbox_inches='tight')


#%%
# a. Find tangent portfolio weight
ind = np.where(np.array(FHS_front_vol.round(1)==FHS_opt[2].round(1)))[0][0]
FHS_w_tan = FHS_z[ind]

# b. Compute annualized portfolio return
FHS_ret_tan = port_ret(mu=FHS_mu,weights=FHS_w_tan)

# c. Compute annualized portfolio volatility
FHS_vol_tan = port_vol(omega=FHS_omega,weights=FHS_w_tan)

# d. Print resuts:
print('Efficient weights', FHS_w_tan.round(3))
print('Annualized portfolio return = ', FHS_ret_tan.round(3))
print('Annualized portfolio vol = ', FHS_vol_tan.round(3))
print('Maximum Sharpe Ratio = ',(FHS_ret_tan/FHS_vol_tan).round(3),
      '=', -sharpe_ratio(mu=FHS_mu, omega=FHS_omega,weights=FHS_w_tan).round(3))

#%% [markdown]
# 
# # Conclusion

#%%
# Weights: minimum vol, max Sharpe
conc_2 = np.vstack([mylist, FHS_w_min.round(3)])
conc_2 = np.vstack([conc_2, FHS_w_eff.round(3)])
conc_2 = np.vstack([conc_2, FHS_w_tan.round(3)])

# Portfolio return
P_R_2 = np.ones(4)
P_R_2[1:] = np.array([FHS_ret_min , FHS_ret_eff, FHS_ret_tan]).round(3)
conc_2 = np.column_stack((conc_2, P_R_2))

# Portfolio volatility
P_V_2 = np.ones(4)
P_V_2[1:] = np.array([FHS_vol_min , FHS_vol_eff, FHS_vol_tan]).round(3)
conc_2 = np.column_stack((conc_2, P_V_2))

# Portfolio Sharpe Ratio
P_SR_2 = np.ones(4)
P_SR_2[1:] = np.array([-sharpe_ratio(mu=FHS_mu, omega=FHS_omega,weights=FHS_w_min),
                       -sharpe_ratio(mu=FHS_mu, omega=FHS_omega,weights=FHS_w_eff), 
                       -sharpe_ratio(mu=FHS_mu, omega=FHS_omega,weights=FHS_w_tan)]).round(3)
conc_2 = np.column_stack((conc_2, P_SR_2))

conc_2 = pd.DataFrame(conc_2, index=['FHS', 'Minimum vol. portfolio', 
                                     'Target return portfolio', 'Tangent portfolio'])


conc_2.columns = conc_2.iloc[0]
conc_2 = conc_2.drop(conc_2.index[0])
conc_2.columns.values[[-3, -2, -1]] = ['Exp. return', 'Exp. volatility', 'Exp. SR']
conc_2

#%% [markdown]
# # Backtest

#%%
train, test = rets[:-n_time], rets[-n_time : H]
test.cumsum().plot()


#%%
#Test period accumulated return
hist_perform_ret = hist_w_tan.dot(test.cumsum().iloc[-1])
print('Hist portfolio acc. return = ',hist_perform_ret*1/(n_time/252))

FHS_perform_ret = FHS_w_tan.dot(test.cumsum().iloc[-1])
print('FHS portfolio acc. return = ',FHS_perform_ret*1/(n_time/252))


#%%
hist_w_tan.dot(test.cov().dot(hist_w_tan))


#%%
FHS_w_tan.dot(test.cov().dot(FHS_w_tan))


#%%



#%%



