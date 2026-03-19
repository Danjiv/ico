import pandas as pd
import numpy as np
import xpress as xp
import preprocessing
from UCWLP_subproblem_model import UCWLP_subproblem_model


def main():

   filename = "cap102"

   supply_cost_df, capacity, fixed_cost, demand, n_customers, n_warehouses = preprocessing.read_in_input_data(filename)

   if sum(demand) > sum(capacity):
       print(f"Problem is infeasible for {filename}: total demand is greater than total capacity")
 
   else:

      print(f"Processing file {filename}...")
      print("Testing for a good initial value for lambdas...")

      # try a bunch of initial lambda vectors and select the maximum value to start with

      lambdas_to_try = range(10, 5000, 10)
      lambda_vals = []
      objective_function_lambda_vals = []

      for initial_lambda in lambdas_to_try:
         set_initial_lambdas = [initial_lambda for i in range(n_customers)]
   
         objective_function_val, x, y = UCWLP_subproblem_model(supply_cost_df, capacity, fixed_cost,
                                                               demand, n_customers, n_warehouses,
                                                               lambdas = set_initial_lambdas)
         lambda_vals.append(initial_lambda)
         objective_function_lambda_vals.append(objective_function_val)

      max_val = max(objective_function_lambda_vals)
      max_lambda = lambda_vals[objective_function_lambda_vals.index(max_val)]

      print(f"Maximum objective function value: {max_val}. Found when all lambdas are set to {max_lambda}")

      print(f"Starting with a lambda vector with all entries equal to {max_lambda}")

      set_initial_lambdas = [max_lambda for i in range(n_customers)]

      objective_function_val, x, y = UCWLP_subproblem_model(supply_cost_df, capacity, fixed_cost,
                                                               demand, n_customers, n_warehouses,
                                                               lambdas = set_initial_lambdas)
      capacity_of_warehouses_opened = [c for c, open in zip(capacity, y) if open == 1]

      if sum(capacity_of_warehouses_opened) > sum(demand):
         print("Can serve customer demand from the warehouses opened in the solution to the subproblem")
         print(f"Total demand: {sum(demand)}  Capacity: {sum(capacity_of_warehouses_opened)}")
      else:
         print("Cannot meet customer demand from the warehouses opened in the solution to the subproblem")
         print(f"Total demand: {sum(demand)}  Capacity: {sum(capacity_of_warehouses_opened)}")
         
   
      #x_df = pd.DataFrame(data = x, index = range(n_customers))
      #x_df.to_csv("checking_assignment.csv")   
      #print(y)
      #print(y==1)

   

if __name__ == "__main__":
    main()