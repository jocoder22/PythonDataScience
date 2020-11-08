#!/usr/bin/env python
import os
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV
import bisect 


from sklearn.preprocessing import StandardScaler
from printdescribe import  print2, changepath

import statsmodels.api as sm
from statsmodels.formula.api import rlm

pth = r"D:\Wqu_FinEngr\Case_Studies_Risk_Mgt\GroupWork"

with changepath(pth):
    data = pd.read_excel("greece_quarterly_30Y_reduced_20201102.xlsx", sheet_name="Reduced")
    print2(data.iloc[:,:6].head())

clean_data = data.iloc[2:,:].set_index("Name")

print2(clean_data.iloc[:,:6].head())

# Do tree bases modelselection
X, y = clean_data.iloc[:,1:], clean_data.iloc[:,0]
clf = DecisionTreeRegressor()
clf = clf.fit(X, y)
print2(clf.feature_importances_ )
pf = pd.DataFrame(clf.feature_importances_, index = X.columns.tolist(), columns=["Features"])
pf_sorted = pf.sort_values(by="Features", ascending=False)
print2(pf_sorted.head(6).index)


# Do lasso based selection
clf = LassoCV().fit(X, y)
importance = np.abs(clf.coef_)
# print(importance)

feature_names = X.columns.tolist()
idx_third = importance.argsort()[-7]
threshold = importance[idx_third] + 0.01

idx_features = (-importance).argsort()[:6]
name_features = np.array(feature_names)[idx_features]
print('Selected features: {}'.format(name_features))


# Doing pca
pca = PCA()
pca.fit_transform(X)
pce = pca.explained_variance_ratio_
print2(pce.cumsum())

# fit 2 component pca
pca = PCA(n_components=2)
pca.fit_transform(X)

# get the variable loading for each components
loadings = pd.DataFrame(np.abs(pca.components_.T), columns=['PC1', 'PC2'], 
                        index = clean_data.iloc[:,1:].columns).sort_values(by="PC2", ascending=False)

mylist = []
for i in loadings.columns:
    highvalue = loadings[i].sort_values(ascending=False)
    print(highvalue.index[0])
    mylist.append(highvalue.index[0])


# standard the dataset
scaler = StandardScaler()
st_X = scaler.fit_transform(X)

# perform exploratory pca
pca = PCA()
pca.fit_transform(st_X)
pce = pca.explained_variance_ratio_
pcelist = pce.cumsum()

res = list(map(lambda i: i> 0.96, pcelist)).index(True)
answer = list(filter(lambda i: i > 0.96, pcelist))[0] 
res2 = list(pcelist).index(answer)
res3 = next(x for x, val in enumerate(pcelist) if val > 0.96) 
res4 = bisect.bisect_left(pcelist, 0.96)
print2(res, res2, res3, res4)



pca = PCA(n_components=res)
pca.fit_transform(st_X)
colname = ["PC"+ str(i) for i in range(1,res+1)]

loadings = pd.DataFrame(np.abs(pca.components_.T), columns=colname, 
                        index = X.columns)#.sort_values(by="PC3", ascending=False)


mylist = []
for i in loadings.columns:
    highvalue = loadings[i].sort_values(ascending=False)
    print(highvalue.index[0])
    mylist.append(highvalue.index[0])

print2(mylist)
print2(clean_data.shape)


# create linear regression
X2 = sm.add_constant(X)
est = sm.OLS(y.astype(float), X2.astype(float))
est2 = est.fit()
print(est2.summary())


# do linear regression on gdp and external debt, m3 outstanding
xx = clean_data.loc[:,["GR EXTERNAL DEBT CURN","GR M3 OUTSTANDING AMOUNTS CURN"]]
X2 = sm.add_constant(xx)
est = sm.OLS(y.astype(float), X2.astype(float))
est2 = est.fit()
print(est2.summary())

xxx = clean_data.iloc[:,1:]
X2 = sm.add_constant(xxx)
est = sm.OLS(y.astype(float), X2.astype(float))
est2 = est.fit()
print(est2.summary())


xmm = mylist + ["GR EXTERNAL DEBT CURN","GR M3 OUTSTANDING AMOUNTS CURN"]
xxmm = clean_data.loc[:,xmm]
# xxx = clean_data.iloc[:,1:]
X2 = sm.add_constant(xxmm)
est = sm.OLS(y.astype(float), X2.astype(float))
est2 = est.fit()
print(est2.summary())

kk2 = ['ATHEX COMPOSITE - PRICE INDEX', 'GR EXPORTS OF GOODS & SERVICES CONN',
        'GR PPI NADJ', 'GR GOVERNMENT BOND - 15 YEAR NADJ',
        'GR EXTERNAL DEBT CURN', 'GR M3 OUTSTANDING AMOUNTS CURN']

xxkk = clean_data.loc[:,kk2]
# xxx = clean_data.iloc[:,1:]
Xk = sm.add_constant(xxkk)
est = sm.OLS(y.astype(float), Xk.astype(float))
est2 = est.fit()
print(est2.summary())


rlm_model = sm.RLM(y.astype(float), Xk.astype(float),  M=sm.robust.norms.HuberT())
est2 = rlm_model.fit()
print(est2.summary())


clean_data["gdp_diff"] = clean_data.iloc[:,0].pct_change()
clean_data["dummy"] = clean_data["gdp_diff"].apply(lambda x: 1 if x > 0 else 0)
y_d = clean_data.iloc[:,-1]
xxkk = clean_data.loc[:,kk2]
Xk = sm.add_constant(xxkk)
est = sm.Logit(y_d.astype(float), Xk.astype(float))
est2 = est.fit()
print(est2.summary())


y_d = clean_data.iloc[:,-1].dropna()
xxkk = clean_data.iloc[:,15:].drop(columns=["gdp_diff","dummy",
                                            "GR DISCOUNT RATE / SHORT TERM EURO REPO RATE NADJ",
                                          "EURO OVERNIGHT DEPOSIT (ECB) - MIDDLE RATE",
                                          "EURO MAIN REFINANCING ECB - MIDDLE RATE"]).dropna()
# xxkk = clean_data.iloc[:,16:].drop(columns=["gdp_diff","dummy"]).dropna()
Xk = sm.add_constant(xxkk)
est = sm.Logit(y_d.astype(float), Xk.astype(float), method='bfgs')
est2 = est.fit()
print(est2.summary())

xxkk = clean_data.iloc[:,1:].drop(columns=["gdp_diff","dummy"])
print2(xxkk)