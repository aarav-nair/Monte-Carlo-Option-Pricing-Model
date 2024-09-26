import numpy as np
import matplotlib.pyplot as plt

# S0: initial stock price
# mu: expected return
# sigma: volatility
# T: time in years
# steps: number of time steps
# simulations: number of simulations

def monte_carlo_simulation(S0, mu, sigma, T, steps, simulations):
   dt = T / steps
   St = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(simulations,steps)).T)
   St = np.vstack([np.ones(simulations), St])
   prices = S0 * St.cumprod(axis=0)
   return prices


   # dt = T / steps
   # dW = np.random.normal(0, np.sqrt(dt), (steps, simulations))
   # price_factors = np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

   # t = np.arange(0, T, dt)
   # S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.cumsum(dW))
   # prices = np.zeros((steps + 1, simulations))
   # prices[0] = S0
   # prices[1:] = S0 * np.cumprod(price_factors, axis=0)
   # return prices

def monte_carlo_antithetic(S0, mu, sigma, T, steps, simulations):
   dt = T / steps
   dW_positive = np.random.normal(0, np.sqrt(dt), (steps, simulations))
   dW_negative = -dW_positive
   dW = np.concatenate((dW_positive, dW_negative), axis=1)
   price_factors = np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

   prices = np.zeros((steps + 1, simulations * 2))
   prices[0] = S0
   prices[1:] = S0 * np.cumprod(price_factors, axis=0)
   return prices

def monte_carlo_stratified(S0, mu, sigma, T, steps, simulations):
   dt = T / steps
   strata = (np.arange(1, simulations + 1) - 0.5) / simulations
   dW = np.sqrt(-2 * np.log(1 - strata)) * np.sin(2 * np.pi * np.random.rand(steps, simulations)) * np.sqrt(dt)
   price_factors = np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

   prices = np.zeros((steps + 1, simulations))
   prices[0] = S0
   prices[1:] = S0 * np.cumprod(price_factors, axis=0)
   return prices