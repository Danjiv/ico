import pandas as pd
import numpy as np
import xpress as xp
import preprocessing
from CWLP_subproblem_model import CWLP_subproblem_model
from CWLP_model import CWLP_model
from CWLP_model_LP_relaxation import CWLP_model_LP_relaxation
from support_functions import check_if_solution_is_feasible, test_lambdas, solve_lagrangian_dual
import time


def main():

   best_lambda = False

   filenames = ["cap61", "cap62", "cap71", "cap72", "cap81", "cap82", "cap101", "cap102"]
   range_lambdas_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
   alpha_halves = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


   results_dict = {"filename": [],
                   "range_lambdas": [],
                   "alpha_half": [],
                   "run_time": [],
                   "best_feasible_obj": [],
                   "best_lb": [],
                   "number_lagrangian_iterations": [],
                   "what_stopped_lagrangian": []
                   }
   
   for filename in filenames:
      for range_lambdas in range_lambdas_options:
         for alpha_half in alpha_halves:
   
            start_time = time.time()

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
                                                         demand, n_customers, n_warehouses, range_lambdas, best_lambda,
                                                         lp_objective_function_value)


                  # look to see if the lp relaxation objective function value is >= the best found UB for the lagrangian subproblem
                  # as an indication of whether the integrality property holds

                  if max_obj_val <= lp_objective_function_value:
                     print("Unclear if the integrality property holds")
                     results_dict["filename"].append(filename)
                     results_dict["range_lambdas"].append(range_lambdas)
                     results_dict["alpha_half"].append(alpha_half)
                     results_dict["run_time"].append(time.time() - start_time)
                     results_dict["best_feasible_obj"].append(0)
                     results_dict["best_lb"].append(0)
                     results_dict["number_lagrangian_iterations"].append(0)
                     results_dict["what_stopped_lagrangian"].append("unclear if the intergrality property holds")

                  else:
                     print("The integrality property does not hold. Continuing to find a solution using the lagrangian...")
                     dual_obj_val, dual_x, dual_y, feasible_obj_val, feasible_x, feasible_y, iterations, reason_for_stopping = solve_lagrangian_dual(supply_cost_df, capacity, fixed_cost,
                                                                                                                                                   demand, n_customers, n_warehouses,
                                                                                                                                                   max_lambdas, alpha_half)
                     
                     print(f"LP relaxation obj: {lp_objective_function_value}  Dual obj: {dual_obj_val}  Feasible obj: {feasible_obj_val}")
                     print("Warehouses opened in Lagrangian solution...")
                     print(dual_y)
                     print("Warehouses opened in feasible solution")
                     print(feasible_y)

                     results_dict["filename"].append(filename)
                     results_dict["range_lambdas"].append(range_lambdas)
                     results_dict["alpha_half"].append(alpha_half)
                     results_dict["run_time"].append(time.time() - start_time)
                     results_dict["best_feasible_obj"].append(feasible_obj_val)
                     results_dict["best_lb"].append(dual_obj_val)
                     results_dict["number_lagrangian_iterations"].append(iterations)
                     results_dict["what_stopped_lagrangian"].append(reason_for_stopping)


   results_df = pd.DataFrame(results_dict)
   if best_lambda is True:
      results_df.to_csv("results_best_lambda.csv")
   else:
      results_df.to_csv("results_first_lambda.csv")                     


if __name__ == "__main__":
    main()