import pandas as pd
from typing import Tuple

def read_in_input_data(filename: str) -> Tuple[pd.DataFrame, list[int], list[int], list[int], int, int]:

    filename_with_extension = "data_sets/" +  filename + ".txt"

    capacity = []
    fixed_cost = []
    demand = []
    supply_cost = []

    with open(filename_with_extension) as f:
        lines = f.readlines()
        #get number of warehouses and number of customers
        warehouses_customers = lines[0].strip().split()
        n_warehouses = int(warehouses_customers[0])
        n_customers = int(warehouses_customers[1])

        for i in range(len(lines)+1):
            if i == 0:
                continue
            elif i <= n_warehouses:
                # get warehouse capacity and fixed costs
                capacity_fixed_costs = lines[i].strip().split()
                capacity.append(int(capacity_fixed_costs[0]))
                fixed_cost.append(int(capacity_fixed_costs[1].replace(".", "")))
            elif i < len(lines):
                # get customer demand and cost of supplying 
                vals = lines[i].strip().split()
                if len(vals) == 1:
                    demand.append(int(vals[0]))
                    # cost_of_supply variable won't exist for the first customer's cost 
                    if 'cost_of_supply' in locals():
                        supply_cost.append(cost_of_supply)
                    cost_of_supply = []
                else:
                    for val in vals:
                        cost_of_supply.append(float(val))

            else:
                supply_cost.append(cost_of_supply)

            # return supply_cost as a dataframe
            colnames = [f"c{i+1}" for i in range(n_warehouses)]

            supply_cost_df = pd.DataFrame(supply_cost, columns = colnames)

    return (supply_cost_df, capacity, fixed_cost, demand, n_customers, n_warehouses)