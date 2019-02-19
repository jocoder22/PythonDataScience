#!/usr/bin/env python

def eda(dataframe):
    from collections import defaultdict
    import pandas as pd
    ddd = dict(dataframe.isnull().sum())
    dd = defaultdict(list)
    nn = dataframe.shape[0]
    for key, value in ddd.items():
        dd['name'].append(key)
        dd['missing'].append(value)
        if value == 0:
            dd['%_missing'].append(value)
        else:
            val = 100 * value / nn
            dd['%_missing'].append(f'{val:.2f}')

    mm = pd.DataFrame(dd)
    print("Printing .. percentage missing data", mm, sep='\n')

    dd2 = defaultdict(list)
    for item in list(dataframe.columns):
        dd2['name'].append(item)
        dd2['dtype'].append(dataframe[item].dtype)
        unique = len(dataframe[item].unique())
        dd2['number_unique'].append(unique)

    mm2 = pd.DataFrame(dd2)
    print('Printing ... number of unique items per feature', mm2 , sep='\n')

    sp = '\n#################################################\n#################################################\n'
    print(dataframe.head(), dataframe.info(), dataframe.columns,
            dataframe.shape, sep=sp)
