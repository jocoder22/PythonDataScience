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

string.punctuation
all_join = "".join(c for c in eassy.lower() if c not in string.punctuation)
all_join
all_join.strip()
eassy_split.strip()

# Replace all double spaaces
Nospace = re.sub('\s+', ' ', all_join)
Nospace22 = Nospace.split()
Nospace22[:10]
