import pandas as pd
import numpy as np
import xpress as xp
import preprocessing
from CWLP_subproblem_model import CWLP_subproblem_model
from CWLP_model import CWLP_model
from typing import Tuple
import time



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
                    demand: list[int], n_customers: int, n_warehouses: int, y: np.array) -> Tuple[float, np.array, np.array]:
    
    """
    Idea is to take an infeasible solution, repair it, and return a feasible solution.
    If a solution is infeasible, we haven't opened enough warehouses to meet total demand, then
    start openining additional warehouses starting from the first unopened, ordered by position in the raw data,
    not targeted to finding the best possible solution
    """
    
    while is_capacity_met(demand, capacity, y) is False:
        first_unopened = np.where(y==0)[0]
        y[first_unopened] = 1

    objective_function_val_feasible, x_feasible, y_feasible =  CWLP_model(supply_cost_df, capacity, fixed_cost,
                                                                          demand, n_customers, n_warehouses,
                                                                          capacity_met=True, open = y)
    
    return (objective_function_val_feasible, x_feasible, y_feasible)


def test_lambdas(supply_cost_df: pd.DataFrame, capacity: list[int], fixed_cost: list[int],
                    demand: list[int], n_customers: int, n_warehouses: int,
                      range_lambdas: int, best_lambda: bool,  LR_value: float) -> Tuple[np.array, float]: 
    """
    Idea is to test lambda values to pick a 'good' vector to start with
    Return this vector and the maximum found objective function value for the Lagrangian subproblem
    """

    supply_cost_array = supply_cost_df.to_numpy()

    min_supply_cost = np.min(supply_cost_array, axis=1)

    max_supply_cost = np.max(supply_cost_array, axis=1)

    obj_vals = []
    delta = []

    for i in range(range_lambdas):
        test_lambdas = min_supply_cost + (max_supply_cost - min_supply_cost)/(i+1)
        objective_function_val, x, y =  CWLP_subproblem_model(supply_cost_df, capacity, fixed_cost,
                                                               demand, n_customers, n_warehouses,
                                                               lambdas = test_lambdas)
        # the best_lambda bool here refers to whether or not we should try to find the 'best', meaning
        # largest value for the lagrangian within the parameters specified before continuing.
        # best_lambda being false means within the specified lambda range, we stop as soon as 
        # find any value that proves the integrality property does not hold.
        # have added in the + 100 here as there is no gurantee the solver returns precisely
        # the same value.
        if best_lambda is False and objective_function_val > LR_value + 100:
            return (test_lambdas, objective_function_val)
        
        delta.append(i+1)
        obj_vals.append(objective_function_val)

    max_val = max(obj_vals)
    max_delta = delta[obj_vals.index(max_val)]

    return (min_supply_cost + (max_supply_cost-min_supply_cost)/max_delta, max_val)


def solve_lagrangian_dual(supply_cost_df: pd.DataFrame, capacity: list[int],
                          fixed_cost: list[int], demand: list[int],
                          n_customers: int, n_warehouses: int,
                          initial_lambdas: np.array, alpha_half: int)->Tuple[float, np.array, np.array, float, np.array, np.array]:
    
    """
    Ideas here is to check the lagrangian for the initial lambda values, if optimal then return solution.
    Otherwise, use the lagrangian value as an initial lower bound, and get an initial upper bound
    by repairing the solution
    Then, iterate through updating the lambdas and bounds until a stopping criteria is hit.
    The stopping criteria are as follows:
    1. The solution for the lagrangian is optimal
    2. the dual gap, gap between best upper bound and lower bound is within a threshold, specified below
    3. the square of the L2 norm for the subgradients is within a threshold, specified below
    4. we have reached the maximum number of iterations, specified below
    5. we have reached maximum computign time, specified below
    """
    # To start, we want to check if the lagrangian is optimal at the given lambdas, and if not,
    # get a feasible upper bound then start the iteration

    #get the lagrangian subproblem solution for the given lambdas            
    objective_function_val, x, y =  CWLP_subproblem_model(supply_cost_df, capacity, fixed_cost,
                                                          demand, n_customers, n_warehouses,
                                                          lambdas = initial_lambdas)
    
    # Note that because the relaxed constraints here are equality constraints,
    # if we get a solution for the lagrangian subproblem that is feasible for the MIP,
    # it must be optimal for the MIP
    
    if check_if_solution_is_feasible(x, y, demand, capacity) is True:
        print("Optimal solution found before starting iteration...")
        return(objective_function_val, x, y, objective_function_val, x, y, 0, "dual was optimal for initial lambdas")

    # If the solution for the initial lambdas is not optimal, get a feasible solution and set initial
    # upper and lower bounds

    z_lb = objective_function_val
    z_lb_x = x
    z_lb_y = y

    # Note, if we open sufficent warehouses to meet total demand in the lagrangian above
    # the solution is not really 'repaired', per se, customers are just optimally assigned
    # with open warehouses fixed to those opened in the lagrangian solution.

    objective_function_val_feasible, x_feasible, y_feasible = repair_solution(supply_cost_df, capacity, fixed_cost,
                                                                              demand, n_customers, n_warehouses,
                                                                              y)
    
    z_ub = objective_function_val_feasible
    z_ub_x = x_feasible
    z_ub_y = y_feasible

    ##############################################################################################################

    # Start the iteration

    ##############################################################################################################

    subgradient_tolerance = 0.1
    dual_gap_tolerance = 0.0001
    iterations = 1
    working_lambdas = initial_lambdas
    alpha = 2
    subgradient_small = False
    dual_gap_small = False
    max_iterations = 200
    # max_computing_time is in seconds, so 300 equates to 5 minutes
    max_computing_time = 300

    start_time = time.time()

    while subgradient_small is False and dual_gap_small is False:

        if iterations > max_iterations:
            print("We've reached the maximum number of iterations for solving the dual...")
            print("The best found solutions are...")
            return(z_lb, z_lb_x, z_lb_y, z_ub, z_ub_x, z_ub_y, iterations, "max iterations")
        
        if time.time() - start_time > max_computing_time:
            print("We've reached the maximum time limit for computing the dual...")
            print("The best found solution is...")
            return(z_lb, z_lb_x, z_lb_y, z_ub, z_ub_x, z_ub_y, iterations, "max computing time")

        # halve the value of alpha every 'alpha_half' iterations
        if iterations % alpha_half == 0:
            alpha = alpha/2

        #get the lagrangian subproblem solution for the given lambdas            
        objective_function_val, x, y =  CWLP_subproblem_model(supply_cost_df, capacity, fixed_cost,
                                                              demand, n_customers, n_warehouses,
                                                              lambdas = working_lambdas)
        # update lower bound
        if objective_function_val > z_lb:
            z_lb = objective_function_val
            z_lb_x = x
            z_lb_y = y
        
        # get a feasible solution using the sub-problem solution

        objective_function_val_feasible, x_feasible, y_feasible = repair_solution(supply_cost_df, capacity, fixed_cost,
                                                                                  demand, n_customers, n_warehouses,
                                                                                  y)
        # update upper bound
        if objective_function_val_feasible < z_ub:
            z_ub = objective_function_val_feasible
            z_ub_x = x_feasible
            z_ub_y = y_feasible
            
        # check to see if the subgradients are close enough to zero and
        # check to see if the feasible solution is close enough to the solution for the subproblem
        rowsum_x = np.sum(x, axis=1)
        s = 1 - rowsum_x
        subgradient_square_sum = np.sum((1-rowsum_x)**2)

        dual_gap = (z_ub - z_lb) / z_ub

        if subgradient_square_sum <= subgradient_tolerance and dual_gap <= dual_gap_tolerance:
            subgradient_small = True
            dual_gap_small = True
            print("Optimal solution found - reached both the subgradient tolerance and dual gap tolerance")
            return(z_lb, z_lb_x, z_lb_y, z_ub, z_ub_x, z_ub_y, iterations, "reached subgradient tolerance and dual gap tolerance") 

        if subgradient_square_sum <= subgradient_tolerance:
            subgradient_small = True
            print("Optimal solution found - reached subgradient tolerance...")
            return(z_lb, z_lb_x, z_lb_y, z_ub, z_ub_x, z_ub_y, iterations, "reached subgradient tolerance")  

        if dual_gap <= dual_gap_tolerance:
            dual_gap_small = True
            print("Optimal solution found - reached dual gap tolerance")
            return(z_lb, z_lb_x, z_lb_y, z_ub, z_ub_x, z_ub_y, iterations, "reached dual gap tolerance")
        
        print(f"iteration: {iterations}  Subgradient square sum: {subgradient_square_sum }  Dual Gap: {dual_gap}  Alpha: {alpha}")
        # if we haven't reached the above stopping criteria, update values for lambdas
        mu = alpha*((z_ub - objective_function_val)/subgradient_square_sum )
        # don't think we want the max of the below and 0 given that we've relaxed an equality constraint - hence lambda values can be negative
        working_lambdas = working_lambdas + mu*s
        iterations += 1



