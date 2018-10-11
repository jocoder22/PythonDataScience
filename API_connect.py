import numpy as np
import pandas as pd
from requests import get


base_url = "api.census.gov/data/timeseries/idb/5year"
# ?get=NAME,POP,CBR,CDR,E0,AREA_KM2&FIPS=NO&time=2012

secret_key = "34a25d4f997aea1ccb52474dc8a7ba4fae696173"
parameters = {"key": secret_key,
              "get": ",".join(["NAME", "POP", "CBR", "CDR", "E0", "AREA_KM2"]),
              "time": "from 2013 to 2017",
              "FIPS": "*"}

response = get(base_url, params=parameters)