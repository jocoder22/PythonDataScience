import pandas as pd
from requests import get


base_url = "http://api.census.gov/data/timeseries/idb/5year"
# ?get=NAME,POP,CBR,CDR,E0,AREA_KM2&FIPS=NO&time=2012

secret_key = "34a25d4f997aea1ccb52474dc8a7ba4"
parameters = {"key": secret_key,
              "get": ",".join(["NAME", "POP", "CBR", "CDR", "E0", "AREA_KM2"]),
              "time": "from 2013 to 2017",
              "FIPS": "*"}

col = ["AREA_KM2", "ASFR15_19", "ASFR20_24", "ASFR25_29", "ASFR30_34", 
       "ASFR40_44", "ASFR40_44", "ASFR45_49", "CBR", "E0", "E0_F", "FIPS",
       "FMR0_4", "FMR0_4", "FMR1_4", "FPOP", "FPOP0_4"]

response = get(base_url, params=parameters)

response.status_code
response.url
response.content

resp_obj = response.json()
type(resp_obj)  # <class 'list'>

popdata = pd.DataFrame(resp_obj[1:], columns=resp_obj[0])
popdata.head()
popdata.tail()
