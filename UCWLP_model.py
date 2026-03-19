import xpress as xp
import numpy as np
import pandas as pd
from typing import Tuple


def UCWLP_model(supply_cost_df: pd.DataFrame, capacity: list[int], fixed_cost: list[int],
                demand: list[int], n_customers: int, n_warehouses: int, capacity_met: bool, open: list[int]) -> Tuple[float, np.array, np.array]:
    
    """
    Solve the MIP, fixing the warehouses specified in the 'open' list to be open
    and return the objective function value, the customer assignment array,
    and the warehouse opening array
    """

    supply_cost_array = supply_cost_df.to_numpy()
    customers = range(n_customers)
    warehouses = range(n_warehouses)

    # ===================================================================================

    # Build optimization model

    # ===================================================================================

    prob = xp.problem("UCWLP")

    xp.setOutputEnabled(False)
    # prob.controls.maxtime = -300

    # ===================================================================================

    # Declarations

    # ===================================================================================

    y = np.array([prob.addVariable(name = 'y_{0}'.format(w), vartype = xp.binary)
                  for w in warehouses], dtype = xp.npvar).reshape(n_warehouses)
    
    x = np.array([prob.addVariable(name = 'x_{0}_{1}'.format(c, w), vartype=xp.continuous, ub=1, lb=0)
                  for c in customers for w in warehouses], dtype=xp.npvar).reshape(n_customers, n_warehouses)
    
    # ===================================================================================

    # Objective function

    # ===================================================================================

    prob.setObjective(xp.Sum(supply_cost_array[i, j]*x[i, j] for i in customers for j in warehouses) +
                      xp.Sum(fixed_cost[j]*y[j] for j in warehouses),
                      sense = xp.minimize)
    
    # ===================================================================================

    # Constraints

    # ===================================================================================

    # ensure we don't supply customers more than warehouse capacity

    prob.addConstraint(xp.Sum(demand[i]*x[i, j] for i in customers) <= capacity[j]*y[j] for j in warehouses)

    # ensure demand for each customer is met

    prob.addConstraint(xp.Sum(x[i, j] for j in warehouses)==1 for i in customers)

    # if the solution to the subproblem opens sufficient warehouses to meet total customer demand
    # fix these warehouses to be open and minimize the assignment of customers to these warehouses
    if capacity_met is True:
        prob.addConstraint(y[j] == open[j] for j in warehouses)
    # if the solution to the subproblem does not open enough warehouses to meet total customer demand
    # fix the warehouses that have been opened in the solution for the subproblem to be open
    else:
        prob.addConstraint(y[j] == 1 for j in warehouses if open[j] == 1)


    prob.solve()
    #print(f'the objective function value is {prob.attributes.objval}')

    objective_function_val = prob.attributes.objval
    y = prob.getSolution(y)
    x = prob.getSolution(x)

    return (objective_function_val, x, y)
