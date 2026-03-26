import pandas as pd
import numpy as np
import xpress as xp
import preprocessing
from UCWLP_subproblem_model import UCWLP_subproblem_model
from UCWLP_model import UCWLP_model
from UCWLP_model_LP_relaxation import UCWLP_model_LP_relaxation
from support_functions import is_capacity_met, check_if_solution_is_feasible, repair_solution, test_lambdas


def main():

   filename = "cap61"

   supply_cost_df, capacity, fixed_cost, demand, n_customers, n_warehouses = preprocessing.read_in_input_data(filename)
   # test if customer demand is greater than total capacity of all the warehouses
   if sum(demand) > sum(capacity):
       print(f"Problem is infeasible for {filename}: total demand is greater than total capacity")   
 
   else:

      print(f"Processing file {filename}...")

      print("Testing for a good initial value for lambdas...")
      max_lambdas = test_lambdas(supply_cost_df, capacity, fixed_cost,
                                demand, n_customers, n_warehouses)

      print(f"Starting with a lambda vector {max_lambdas}")

      # Get the LP relaxation of the MIP, and test to see if the solution is feasible for the MIP

      lp_objective_function_value, lp_x, lp_y = UCWLP_model_LP_relaxation(supply_cost_df, capacity, fixed_cost,
                                                                          demand, n_customers, n_warehouses)
      
      if check_if_solution_is_feasible(lp_x, lp_y, demand, capacity):
         print(f"Feasible solution for {filename} from the LP relaxation")
         # need a bit here to print off results
      else:
         # check if the integrality property holds!
         objective_function_val, x, y = UCWLP_subproblem_model(supply_cost_df, capacity, fixed_cost,
                                                               demand, n_customers, n_warehouses,
                                                               lambdas = max_lambdas)
         capacity_of_warehouses_opened = [c for c, open in zip(capacity, y) if open == 1]

         if sum(capacity_of_warehouses_opened) > sum(demand):
            print("Can serve customer demand from the warehouses opened in the solution to the subproblem")
            objective_function_val_feasible, x_feasible, y_feasible = UCWLP_model(supply_cost_df, capacity, fixed_cost,
                                                                               demand, n_customers, n_warehouses,
                                                                               capacity_met=True, open = y)
         else:
            print("Cannot meet customer demand from the warehouses opened in the solution to the subproblem")
            print("Repairing solution...")
            objective_function_val_feasible, x_feasible, y_feasible = UCWLP_model(supply_cost_df, capacity, fixed_cost,
                                                                               demand, n_customers, n_warehouses,
                                                                               capacity_met=False, open = y)
         print(f"LP relaxation obj val: {lp_objective_function_value}  Lagrangian obj val: {objective_function_val}    Feasible val: {objective_function_val_feasible}")
         print("Warehouses opened in the LP relaxation...")
         print(lp_y)
         print("Warehouses opened in Lagrangian solution...")
         print(y)
         print("Warehouses opened in feasible solution...")
         print(y_feasible)
   
         x_feasible_df = pd.DataFrame(data = x_feasible, index = range(n_customers))
         x_feasible_df.to_csv("checking_assignment.csv")   
         #print(y)
         #print(y==1)

   

if __name__ == "__main__":
    main()