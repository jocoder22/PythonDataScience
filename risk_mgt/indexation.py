import numpy as np
import pandas as pd

PriceReturn = [5.2, 4.1, -2.6]
IncomeReturn = [1.5, 1.5, 1.5]

data = pd.DataFrame({"PriceIncome":PriceReturn, "IncomeReturn":IncomeReturn})

data["TotalReturn"] = data.sum(axis = 1)

