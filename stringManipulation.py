from pandas import DataFrame
import re, string
from datetime import datetime

eassy = """This is the begining of time. But with all good and noble
           intention comes failure, only work hard as you and others
            can and expect the best. Be careful while going slowly
            on a journey of life."""

print(eassy.upper())
print(eassy.lower())

eassy_split = eassy.lower().split(" ")
eassy_split[:10]

"_".join(eassy[:10])
"_".join(eassy_split[:10])