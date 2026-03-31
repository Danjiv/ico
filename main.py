import pandas as pd
import numpy as np
import xpress as xp
import preprocessing
from CWLP_subproblem_model import CWLP_subproblem_model
from CWLP_model import CWLP_model
from CWLP_model_LP_relaxation import CWLP_model_LP_relaxation
from support_functions import check_if_solution_is_feasible, test_lambdas, solve_lagrangian_dual


def main():

   filename = "cap61"

   supply_cost_df, capacity, fixed_cost, demand, n_customers, n_warehouses = preprocessing.read_in_input_data(filename)
   # test if customer demand is greater than total capacity of all the warehouses
   if sum(demand) > sum(capacity):
       print(f"Problem is infeasible for {filename}: total demand is greater than total capacity")   
 
   else:

      print(f"Processing file {filename}...")

      
      # Get the LP relaxation of the MIP, and test to see if the solution is feasible for the MIP

      lp_objective_function_value, lp_x, lp_y = CWLP_model_LP_relaxation(supply_cost_df, capacity, fixed_cost,
                                                                          demand, n_customers, n_warehouses)

      if check_if_solution_is_feasible(lp_x, lp_y, demand, capacity):
         print(f"Feasible solution for {filename} from the LP relaxation")
         print(f"LP relaxation obj: {lp_objective_function_value}")

      else:

         print("Testing for a good initial value for lambdas...")
         max_lambdas, max_obj_val = test_lambdas(supply_cost_df, capacity, fixed_cost,
                                                 demand, n_customers, n_warehouses)


         # look to see if the lp relaxation objective function value is >= the best found UB for the lagrangian subproblem
         # as an indication of whether the integrality property holds

         if max_obj_val <= lp_objective_function_value:
            print("Unclear if the integrality property holds")

         else:
            print("The integrality property does not hold. Continuing to find a solution using the lagrangian...")
            dual_obj_val, dual_x, dual_y, feasible_obj_val, feasible_x, feasible_y = solve_lagrangian_dual(supply_cost_df, capacity, fixed_cost,
                                                                                                           demand, n_customers, n_warehouses,
                                                                                                           max_lambdas)
            
            print(f"LP relaxation obj: {lp_objective_function_value}  Dual obj: {dual_obj_val}  Feasible obj: {feasible_obj_val}")
            print("Warehouses opened in Lagrangian solution...")
            print(dual_y)
            print("Warehouses opened in feasible solution")
            print(feasible_y)
   

if __name__ == "__main__":
    main()