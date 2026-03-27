import pandas as pd
import numpy as np
import xpress as xp
import preprocessing
from UCWLP_subproblem_model import UCWLP_subproblem_model
from UCWLP_model import UCWLP_model
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
                    demand: list[int], n_customers: int, n_warehouses: int) -> Tuple[np.array, float]: 
    """
    Idea is to test lambda values to pick a 'good' vector to start with
    Return this vector and the maximum found objective function value for the Lagrangian subproblem
    """

    supply_cost_array = supply_cost_df.to_numpy()

    min_supply_cost = np.min(supply_cost_array, axis=1)

    max_supply_cost = np.max(supply_cost_array, axis=1)

    obj_vals = []
    delta = []

    for i in range(200):
        test_lambdas = min_supply_cost + (max_supply_cost - min_supply_cost)/(i+1)
        objective_function_val, x, y = UCWLP_subproblem_model(supply_cost_df, capacity, fixed_cost,
                                                               demand, n_customers, n_warehouses,
                                                               lambdas = test_lambdas)
        delta.append(i+1)
        obj_vals.append(objective_function_val)

    max_val = max(obj_vals)
    max_delta = delta[obj_vals.index(max_val)]

    return (min_supply_cost + (max_supply_cost-min_supply_cost)/max_delta, max_val)


def solve_lagrangian_dual(supply_cost_df: pd.DataFrame, capacity: list[int],
                          fixed_cost: list[int], demand: list[int],
                          n_customers: int, n_warehouses: int,
                          initial_lambdas = np.array)->Tuple[float, np.array, np.array, float, np.array, np.array]:
    
    """
    Iterate through...
    """
    subgradient_tolerance = 0.1
    dual_gap_tolerance = 0.0001
    iterations = 1
    working_lambdas = initial_lambdas
    alpha = 2
    subgradient_small = False
    dual_gap_small = False

    while subgradient_small is False and dual_gap_small is False:

        #if iterations == 5:
        #    break

        # halve the value of alpha every 10 iterations
        if iterations % 10 == 0:
            alpha = alpha/2

        #get the lagrangian subproblem solution for the given lambdas            
        objective_function_val, x, y = UCWLP_subproblem_model(supply_cost_df, capacity, fixed_cost,
                                                              demand, n_customers, n_warehouses,
                                                              lambdas = working_lambdas)
        
        # get a feasible solution using the suproblem solution

        # check to see if solution has opened enough warehouses to meet customer demand
        capacity_of_warehouses_opened = [c for c, open in zip(capacity, y) if open == 1]
        capacity_met = sum(capacity_of_warehouses_opened) > sum(demand)

        objective_function_val_feasible, x_feasible, y_feasible = repair_solution(supply_cost_df, capacity, fixed_cost,
                                                                                  demand, n_customers, n_warehouses,
                                                                                  y)
            
        # check to see if the subgradients are close enough to zero
        rowsum_x = np.sum(x, axis=1)
        s = 1 - rowsum_x
        z = np.sum((1-rowsum_x)**2)

        if z <= subgradient_tolerance:
            subgradient_small = True
            print("Optimal solution found - reached subgradient tolerance...")
            return(objective_function_val, x, y, objective_function_val_feasible, x_feasible, y_feasible)

        # check to see if the feasible solution is close enough to the solution for the subproblem
        dual_gap = (objective_function_val_feasible - objective_function_val) / objective_function_val_feasible

        if dual_gap <= dual_gap_tolerance:
            dual_gap_small = True
            print("Optimal solution found - reached dual gap tolerance")
            return(objective_function_val, x, y, objective_function_val_feasible, x_feasible, y_feasible)
        
        print(f"iteration: {iterations}  Z: {z}  Dual Gap: {dual_gap}")
        # if we haven't reached the above stopping criteria, update values for lambdas
        mu = alpha*((objective_function_val_feasible - objective_function_val)/z)
        # don't think we want the max of the below and 0 given that we've relaxed an equality constraint - hence lambda values can be negative
        working_lambdas = working_lambdas + mu*s
        iterations += 1



