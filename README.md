# Applied-Corporate-Finance
This repo contains the code from my seminar on Topics in Financial Econometrics.

In the project I download financial data from yahoo finance, and fit a GJR-GARCH model. Based on the fitted model I apply a simulation method called Filtered Historical Simulation (FHS) to simulate the volatility. Based on the simulated volatility I conduct a portfolio selection based on Modern portfolio theory by Markowitch (1952).

This Repo contains the following files.

* main.ipynb: The main notebook that takes you through the project
* matrix_optimize: Mathemathical implementation of the Modern Portfolio Theory
* GJR_GARCH: The volatility model defined
* Backtest: backtest the model predictions