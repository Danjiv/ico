import pandas as pd
import numpy as np
import xpress as xp
import preprocessing
from UCWLP_subproblem_model import UCWLP_subproblem_model
from UCWLP_model import UCWLP_model
from UCWLP_model_LP_relaxation import UCWLP_model_LP_relaxation
from typing import Tuple



def is_capacity_met(demand: int, capacity: int, y: np.array) -> bool:
    """
    Idea is to check whether the solution has opened enough warehouses to
    meet total customer demand
    """
    capacity_of_warehouses_opened = [c for c, open in zip(capacity, y) if open == 1]

    if sum(capacity_of_warehouses_opened) <= sum(demand):
        return False
    else:
        return True


def check_if_solution_is_feasible(x: np.array, y: np.array, demand: int, capacity: int) -> bool:
    """
    Idea here is to check if the provided solution is feasible for the MIP
    This means checking that sufficient warehouses have been opened to meet total customer demand
    That all customers have their full demand serviced
    And that the resulting solution provides binary y values (warehouses opened decision variables)
    """

    capacity_met = is_capacity_met(demand, capacity, y)

    if capacity_met is False:
        #print("Not enough warehouses opened to meet total customer demand. Will need to repair solution")
        return False
    
    else:
        rowsum_x = np.sum(x, axis=1)
        customers = len(rowsum_x)
        customers_demand_met = len([c for c in rowsum_x if c == 1])

        if customers == customers_demand_met:
            n_warehouses = len(y)
            n_warehouses_binary_vals = len([warehouse for warehouse in y if warehouse == 0 or warehouse == 1])

            if n_warehouses == n_warehouses_binary_vals:
                return True
            
        else:
            return False



def repair_solution(supply_cost_df: pd.DataFrame, capacity: list[int], fixed_cost: list[int],
                    demand: list[int], n_customers: int, n_warehouses: int, y: list[int]) -> Tuple[float, np.array, np.array]:
    
    """
    Idea is to take an infeasible solution, repair it, and return a feasible solution.
    """

    capacity_met = is_capacity_met(demand, capacity, y)

    objective_function_val_feasible, x_feasible, y_feasible = UCWLP_model(supply_cost_df, capacity, fixed_cost,
                                                                          demand, n_customers, n_warehouses,
                                                                          capacity_met=capacity_met, open = y)
    
    return (objective_function_val_feasible, x_feasible, y_feasible)



def test_lambdas(supply_cost_df: pd.DataFrame, capacity: list[int], fixed_cost: list[int],
                    demand: list[int], n_customers: int, n_warehouses: int) -> list[float]: 
      """
      Idea is to test lambda values to pick a 'good' vector to start with
      """
    
      lambdas_to_try = range(-5000, 5000, 10)
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

      return max_lambda