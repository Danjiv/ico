import xpress as xp
import numpy as np
import pandas as pd
from typing import Tuple


def CWLP_model_LP_relaxation(supply_cost_df: pd.DataFrame, capacity: list[int], fixed_cost: list[int],
                              demand: list[int], n_customers: int, n_warehouses: int) -> Tuple[float, np.array, np.array]:
    
    """
    Solve the LP relaxation for the MIP,and return the objective function value,
    the customer assignment array, and the warehouse opening array
    """

    supply_cost_array = supply_cost_df.to_numpy()
    customers = range(n_customers)
    warehouses = range(n_warehouses)

    # ===================================================================================

    # Build optimization model

    # ===================================================================================

    prob = xp.problem("CWLP")

    xp.setOutputEnabled(False)
    # prob.controls.maxtime = -300

    # ===================================================================================

    # Declarations

    # ===================================================================================

    y = np.array([prob.addVariable(name = 'y_{0}'.format(w), vartype = xp.continuous)
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

    prob.solve()
    #print(f'the objective function value is {prob.attributes.objval}')

    objective_function_val = prob.attributes.objval
    y = prob.getSolution(y)
    x = prob.getSolution(x)

    return (objective_function_val, x, y)
