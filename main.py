from matplotlib import pyplot as plt
import montecarlo as mc
import numpy as np

def get_parameters():   
   S0 = float(input("Enter initial stock price (S0): "))
   mu = float(input("Enter expected return (mu): "))
   sigma = float(input("Enter volatility (sigma): "))
   T = float(input("Enter time to maturity (T): "))
   steps = int(input("Enter number of time steps: "))
   simulations = int(input("Enter number of simulations: "))
   return S0, mu, sigma, T, steps, simulations

def get_plot(prices):
   plt.figure(figsize=(10,6))
   plt.plot(prices)
   plt.title('Monte Carlo Simulation of Stock Prices')
   plt.xlabel('Days')
   plt.ylabel('Price')
   plt.show()

def get_stats(prices, simulations):
   final_prices = prices[-1]
   mean_price = np.mean(final_prices)
   std_price = np.std(final_prices)
   conf_interval = (mean_price - 1.96 * std_price / np.sqrt(simulations), mean_price + 1.96 * std_price / np.sqrt(simulations))
   return mean_price, conf_interval

def main():
   print("Welcome to the Monte Carlo Simulation CLI!")
   print("\nChoose the Monte Carlo method:")
   print("1 - Standard Simulation")
   print("2 - Antithetic Variates")
   print("3 - Stratified Sampling")
   
   method_choice = int(input("Enter the number corresponding to your choice: "))
   if method_choice < 1 or method_choice > 3:
      print("Invalid choice. Please restart and choose a valid method.")
      return

   S0, mu, sigma, T, steps, simulations = get_parameters()

   if method_choice == 1:
      tt, st = mc.monte_carlo_simulation(S0, mu, sigma, T, steps, simulations)
      print("Vectorized Operations Simulation complete.")
   elif method_choice == 2:
      prices = mc.monte_carlo_antithetic(S0, mu, sigma, T, steps, simulations)
      print("Antithetic Variates Simulation complete.")
   elif method_choice == 3:
      prices = mc.monte_carlo_stratified(S0, mu, sigma, T, steps, simulations)
      print("Stratified Sampling Simulation complete.")

   mean_price, conf_interval = get_stats(st, simulations)
   print(f"Mean price: {mean_price}")
   print(f"95% Confidence interval: {conf_interval}")
   get_plot(st)

if __name__ == '__main__':
   main()