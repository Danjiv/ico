import pandas as pd
import numpy as np
import xpress as xp
import preprocessing
from UCWLP_model import UCWLP_model


def main():
   supply_cost_df, capacity, fixed_cost, demand, n_customers, n_warehouses = preprocessing.read_in_input_data("cap61")
   
   set_initial_lambdas = [3000 for i in range(n_customers)]

   objective_function_val, x, y = UCWLP_model(supply_cost_df, capacity, fixed_cost,
                                              demand, n_customers, n_warehouses,
                                              lambdas = set_initial_lambdas) 
   
   x_df = pd.DataFrame(data = x, index = range(n_customers))
   x_df.to_csv("checking_assignment.csv")   
   print(y)
   print(y==1)

   

if __name__ == "__main__":
    main()