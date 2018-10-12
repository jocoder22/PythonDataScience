import pandas as pd
from requests import get


base_url = "http://api.census.gov/data/timeseries/idb/5year"


secret_key = "34a25d4f997aea1ccb52474dc8a7ba4"
parameters = {"key": secret_key,
              "get": ",".join(["NAME", "POP", "CBR", "CDR", "E0", "AREA_KM2"]),
              "time": "from 2013 to 2017",
              "FIPS": "*"}

response = get(base_url, params=parameters)

response.status_code
response.url
response.content

resp_obj = response.json()
type(resp_obj)  # <class 'list'>

popdata = pd.DataFrame(resp_obj[1:], columns=resp_obj[0])
popdata.head()
popdata.tail()


col = ["NAME", "POP", "AREA_KM2", "ASFR15_19", "ASFR20_24", "ASFR25_29",
       "ASFR40_44", "ASFR40_44", "ASFR45_49", "CBR", "E0", "E0_F", "FIPS",
       "FMR0_4", "FMR0_4", "FMR1_4", "FPOP", "FPOP0_4"]

fullcol = ['AREA_KM2', 'ASFR15_19',	'ASFR20_24', 'ASFR25_29', 'ASFR30_34',
           'ASFR35_39',	'ASFR40_44', 'ASFR45_49', 'CBR',	'CDR',	'E0',
           'E0_F',	'E0_M',	'FIPS',	'FMR0_4', 'FMR1_4',	'FPOP',	'FPOP0_4',
           'FPOP10_14',	'FPOP100_',	'FPOP15_19', 'FPOP20_24', 'FPOP25_29',
           'FPOP30_34',	'FPOP35_39', 'FPOP40_44', 'FPOP45_49', 'FPOP5_9',
           'FPOP50_54',	'FPOP55_59', 'FPOP60_64', 'FPOP65_69', 'FPOP70_74',
           'FPOP75_79',	'FPOP80_84', 'FPOP85_89', 'FPOP90_94', 'FPOP95_99',
           'GR', 'GRR',	'IMR', 'IMR_F', 'IMR_M', 'MMR0_4',	'MMR1_4', 'MPOP',
           'MPOP0_4', 'MPOP10_14', 'MPOP100_', 'MPOP15_19', 'MPOP20_24',
           'MPOP30_34',	'MPOP35_39', 'MPOP40_44', 'MPOP45_49', 'MPOP5_9',
           'MPOP50_54',	'MPOP55_59', 'MPOP60_64', 'MPOP65_69', 'MPOP70_74',
           'MPOP75_79',	'MPOP80_84', 'MPOP85_89', 'MPOP90_94', 'MPOP95_99',
           'MR0_4',	'MR1_4', 'NAME', 'NMR',	'POP', 'POP0_4', 'POP10_14',
           'POP100_', 'POP15_19', 'POP20_24', 'POP25_29',	'POP30_34',
           'POP40_44', 'POP45_49', 'POP5_9', 'POP50_54', 'POP55_59',
           'POP65_69', 'POP70_74',	'POP75_79',	'POP80_84',	'POP85_89',
           'POP95_99',	'RNI',	'SRB',	'TFR',	'time',	'YR', 'MPOP25_29',
           'POP60_64', 'POP90_94', 'POP90_94', 'POP35_39']

par2 = {'key': secret_key,
        'get': ','.join(col),
        'time': 'from 2016 to 2018',
        'FIPS': '*'}

resp2 = get(base_url, params=par2)
resp2.status.code
resp2_obj = resp2.json()
ppdata = pd.DataFrame(resp2_obj[1:], columns=resp2_obj[0])
ppdata.tail()


# Using url directly;
response2 = get("https://api.census.gov/data/timeseries/idb/5year?get=NAME,POP,CBR,CDR,E0,AREA_KM2&FIPS=%2A&time=2012")
response2 = response2.json()
ppdata2 = pd.DataFrame(response2[1:], columns=response2[0])
ppdata2.tail()
