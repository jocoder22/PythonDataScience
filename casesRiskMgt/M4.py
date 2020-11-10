#!/usr/bin/env python
# coding: utf-8

# ## MScFE 660 Case Studies in Risk Management (C19-S3)
# ### October 25, 2020
# ### Collaborative Review Task M4
# <hr>

# #### Questions:
# 
# 1.Visually analyze the covariance between various factors and identify the variance explained in principle components of these <br>  factors. 
# Next, consider the ACF and PACF of the process and its square.
# 
# 2.Using PCA provide a 2-dimensional representation of the weight-space of a set of linear models representing the covariance <br> between our factors and the different benchmark portfolios. Comment on the distribution of the benchmark portfolios across the <br>  weight-space.
# 
# 3.Using linear regression test for the significance of these factors, as per the original work of Fama and French, under the <br>  equation:
# 
#     ExpectedReturns = rf + β1(rm−rf) + β2SMB + β3HML + β4RMW + β5CMA
#   
#  <hr>
# 

# The five-factor asset pricing model is an extension of the Fama and French three-factor asset pricing model that include <br> profitability and investment factors:
# 
#     ExpectedReturns = rf + β1(rm−rf) + β2SMB + β3HML + β4RMW + β5CMA
# 
# Where rf   = riskfree return
# 
#     rm = return on value-weight market portfolio
#   
#     SMB = return on portfolio of small stocks minus portfolio of big stocks
#   
#     HML = return on portfolio of high minus low B/M stocks
#     
#     RMW = return on portfolio of robust minus weak profitability stocks
#     
#     CMA = return on portfolio of low minus high investment firms
#   
#   
# <hr>
# 

# In[280]:


# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import linear_model

from printdescribe import print2, describe2


from functools import reduce
from operator import mul

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

import holoviews as hv
import hvplot
import hvplot.pandas


import warnings
warnings.filterwarnings('ignore')

pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web

np.random.seed(42)
hv.extension('bokeh')


# In[283]:


get_ipython().run_line_magic('opts', "Curve[width=900 height=400] NdOverlay [legend_position='right']")


# In[18]:


# Download datasets
portfolios100 = web.DataReader('100_Portfolios_10x10_Daily', 'famafrench')
factors5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_Daily', 'famafrench')


# In[19]:


portfolios100['DESCR']


# In[20]:


factors5['DESCR']


# In[22]:


# select the Average Value Weighted Returns -- Daily (1220 rows x 100 cols)
portfolios100 = portfolios100[0]
factors5 = factors5[0]


# In[36]:


# Checking for missing values porfolios dataset
print(portfolios100[portfolios100.iloc[:,0] >98.0].sum().sum(),
portfolios100.isnull().sum().sum())


# In[37]:


# Checking for missing values in factors dataset
print(factors5[portfolios100.iloc[:,0] >98.0].sum().sum(),
factors5.isnull().sum().sum())


# In[30]:


portfolios100.iloc[:, 90:100].head()


# In[33]:


portfolios100.shape


# In[34]:


factors5.head()


# In[35]:


factors5.shape


# # 1a.	Visually analyze the covariance between various factors and identify <br> the variance explained in principle components of  these factors. 
# 
# ## 1b. Next, consider the ACF and PACF of the process and its square.
# 

# In[284]:


pd.melt(factors5.add(1).cumprod().reset_index(), id_vars=["Date"]).hvplot.line(x='Date', y='value', by='variable')


# In[ ]:





# In[38]:


factors_cov = factors5.cov()
factors_cov


# In[ ]:





# In[46]:


plt.figure(figsize = [10, 6])
# Visualize the covariance matrix using a heatmap
sns.heatmap(round(factors_cov,3),annot=True, linewidths=.5, cmap="YlGnBu", cbar=False)
plt.yticks(rotation=0, fontsize="10", va="center")
plt.title('Heatmap of factors covariane');


# From the heatmap, there mostly positive low covariances. We notice zero or negative zero covariance of riskfree return with <br> other factors.

# In[52]:


# calculate the variance
pd.DataFrame(factors5.var().reset_index().values,columns=["Factors", "Variance"])


# In[53]:


# compute the correlation matrix
factors5.corr()


# ## Next, consider the ACF and PACF of the process.

# In[88]:


# plot the factors together
plt.figure(figsize = [10, 6])
plt.plot(factors5)
plt.grid(axis='y');


# In[113]:


# plot the factors
factor_labels = factors5.columns.tolist()
fig, ax = plt.subplots(nrows=3, ncols=2, figsize = (14,12))
for idx, ax in enumerate(ax.flatten(),start=0):
    ax.plot(factors5.iloc[:,idx], label=factor_labels[idx])
    ax.set_xlabel(factor_labels[idx])
    ax.grid()
    ax.legend()


# In[167]:


# plot the acf and pacf for the process
fig, ax = plt.subplots(nrows=6, ncols=2, figsize = (15,22))
p = ax.flatten().tolist()
for indx, colname in enumerate(factors5.columns):
    ba =  indx*2
    plot_acf(factors5.iloc[:,indx], title = f'Autocorrelation for {colname}', ax=p[ba])
    plot_pacf(factors5.iloc[:, indx], title = f'Partial Autocorrelation for {colname}', ax=p[ba+1])
    fig.subplots_adjust(hspace=.3)


# ## Next, consider the ACF and PACF of the process square.

# In[169]:


# plot the acf and pacf for the process square
factor5_sq = factors5**2
fig, ax = plt.subplots(nrows=6, ncols=2, figsize = (15,22))
p = ax.flatten().tolist()
for indx, colname in enumerate(factors5.columns):
    ba =  indx*2
    plot_acf(factor5_sq.iloc[:,indx], title = f'Autocorrelation for {colname} square', ax=p[ba])
    plot_pacf(factor5_sq.iloc[:, indx], title = f'Partial Autocorrelation for {colname} square', ax=p[ba+1])
    fig.subplots_adjust(hspace=.3)


# ## 2.Using PCA provide a 2-dimensional representation of the weight-space <br> of a set of linear models representing the covariance between our factors <br>and the different benchmark portfolios. Comment on the distribution of the <br> benchmark portfolios across the weight-space.

# In[184]:


def pca_function(dataframe, transformer=StandardScaler()):
    
    portfolios_standard_ = transformer.fit_transform(dataframe)
    portfolios_standard = pd.DataFrame(portfolios_standard_, columns=dataframe.columns, index=dataframe.index)

    n = 2
    
    equities_ = dataframe.columns
    n_tickers = len(equities_)

    pca = None
    cov_matrix = pd.DataFrame(data=np.ones(shape=(n_tickers, n_tickers)), columns=equities_)
    
    cov_matrix = portfolios_standard.cov()
    pca = PCA()
    pca.fit(cov_matrix) 
    
    return portfolios_standard, pca.explained_variance_ratio_  


def explain_com(pca_explained_variance_ratio):
    # cumulative variance explained
    var_threshold_list = [0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.99, 0.9999]

    for xx in var_threshold_list:
        var_explained = np.cumsum(pca_explained_variance_ratio)
        num_comp = np.where(np.logical_not(var_explained < xx))[0][0] + 1  # +1 due to zero based-arrays
        print(f'{num_comp} components explain {round(100* xx,2)}% of variance')
        
def pca_plot(dataframe, variance_e):

    bar_width = 0.9
    n_asset = int((1 / 10) * dataframe.shape[1])
    
    if n_asset > len(dataframe.columns):
        n_asset = len(dataframe.columns)
        
    elif n_asset <= 0:
        n_asset = len(dataframe.columns) 
             
    x_indx = np.arange(n_asset)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)
    
    # Eigenvalues are measured as percentage of explained variance.
    rects = ax.bar(x_indx, variance_e[:n_asset], bar_width, color='deepskyblue')
    ax.set_xticks(x_indx)
    ax.set_xticklabels(list(range(1,n_asset+1)), rotation=45)
    ax.set_title('Percent variance explained')
    ax.legend((rects[0],), ('Percent variance explained by principal components',))


# In[185]:


std2, pca22 = pca_function(factors5)


# In[186]:


explain_com(pca22)


# In[187]:


pca_plot(std2, pca22)


# In[ ]:





# In[196]:


std2, pca22 = pca_function(portfolios100)


# In[197]:


explain_com(pca22)


# In[199]:


pca_plot(std2, pca22)


# In[273]:


colname = portfolios100.columns
n = 2

# Using Pipeline
pipe3 = Pipeline([('scaler', StandardScaler()),
            ('pca', PCA(n_components=n))])

# Fit it to the dataset and extract the component vectors
pcomp_pro = pipe3.fit_transform(portfolios100.cov())
pcompfactors= pipe3.fit(portfolios100.cov())
# pcomp_pro = pipe3.fit_transform(portfolios100)
# pcompfactors= pipe3.fit(portfolios100)

plt.figure(figsize = [14, 10])
# plt.xticks(rotation=45)    
per_var = np.round(pipe3.steps[1][1].explained_variance_ratio_, 4)
labels = [f"Principal Component {i}" for i in range(1,len(per_var)+1)]
plt.bar(labels,per_var)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Pricipal Components")
plt.title("Scree Plot")
plt.show()


plt.figure(figsize = [14, 10])
pca_df = pd.DataFrame(pcomp_pro, index=colname, 
                      columns=labels)
# pca_df = pd.DataFrame(pcomp_pro, 
#                       columns=labels)
colname2 = pca_df.columns.tolist()
plt.scatter(pca_df[colname2[0]], pca_df[colname2[1]])

for sample in pca_df.index:
    plt.annotate(sample, (pca_df[colname2[0]].loc[sample], 
                          pca_df[colname2[1]].loc[sample]))
       
plt.ylabel(f"Principal Component 2: {per_var[1]*100}%")
plt.xlabel(f"Principal Component 1: {per_var[0]*100}%")
plt.title("PCA Plot")
plt.show()


# In[ ]:





# In[400]:


colname = portfolios100.columns
n = 2

# Using Pipeline
pipe3 = Pipeline([('scaler', StandardScaler()),
            ('pca', PCA(n_components=n))])

# Fit it to the dataset and extract the component vectors
# pcomp_pro = pipe3.fit_transform(portfolios100.cov())
# pcompfactors= pipe3.fit(portfolios100.cov())
pcomp_pro = pipe3.fit_transform(portfolios100)
pcompfactors= pipe3.fit(portfolios100)

# plt.figure(figsize = [14, 10])
# # plt.xticks(rotation=45)    
# per_var = np.round(pipe3.steps[1][1].explained_variance_ratio_, 4)
# labels = [f"Principal Component {i}" for i in range(1,len(per_var)+1)]
# plt.bar(labels,per_var)
# plt.ylabel("Percentage of Explained Variance")
# plt.xlabel("Pricipal Components")
# plt.title("Scree Plot")
# plt.show()


plt.figure(figsize = [14, 10])
# pca_df = pd.DataFrame(pcomp_pro, index=colname, 
#                       columns=labels)
pca_df = pd.DataFrame(pcomp_pro, 
                      columns=labels)
colname2 = pca_df.columns.tolist()
plt.scatter(pca_df[colname2[0]], pca_df[colname2[1]])

for sample in pca_df.index:
    plt.annotate(sample, (pca_df[colname2[0]].loc[sample], 
                          pca_df[colname2[1]].loc[sample]))
       
plt.ylabel(f"Principal Component 2")
plt.xlabel(f"Principal Component 1")
plt.title("PCA Plot")
plt.show()


# In[ ]:





# In[272]:



# Using Pipeline
lreg = Pipeline([('scaler', StandardScaler()),
            ('lnReg', linear_model.LinearRegression())])

# Fit it to the dataset and extract the component vectors
pcomp_pro = lreg.fit(pca_df, factors5)   
per_var = np.round(lreg.steps[1][1].coef_, 4)
labels = [f"Pc_reg {i}" for i in range(1,len(per_var)+1)]

colname = factors5.columns.tolist()
reg_df = pd.DataFrame(per_var, index=colname,
                      columns=["x", "y"])

colname2 = reg_df.columns.tolist()

plt.figure(figsize = [8, 6])
plt.scatter(reg_df[colname2[0]], reg_df[colname2[1]])

for sample in reg_df.index:
    plt.annotate(sample, (reg_df.x.loc[sample], 
                          reg_df.y.loc[sample]))
       
plt.ylabel(f"Principal Component 2")
plt.xlabel(f"Principal Component 1")
plt.title("Distribution of factors across the weight space")
plt.show()


# In[ ]:





# # 3.Using linear regression test for the significance of these factors, <br> as per the original work of Fama and French.

# In[311]:


import statsmodels.api as sm

factors6, _ = pca_function(factors5)
portfolio2, _ = pca_function(portfolios100)

# factors6 = sm.add_constant(factors6, prepend=False)

# Fit and summarize OLS model
model = sm.OLS(portfolio2.iloc[:,0], factors6)

result = model.fit()

print(result.summary())


# In[347]:


n_portfolios = len(portfolios100.columns)
rfree = np.array([0.] * n_portfolios)
beta1 = np.array([0.] * n_portfolios)
beta2 = np.array([0.] * n_portfolios)
beta3 = np.array([0.] * n_portfolios)
beta4 = np.array([0.] * n_portfolios)
beta5 = np.array([0.] * n_portfolios)


# In[353]:


for i in range(portfolios100.shape[1]):
    model = sm.OLS(portfolio2.iloc[:,i], factors6)
    r_ = model.fit()
    pval = round(r_.pvalues, 3)
    beta1[i], beta2[i], beta3[i] , beta4[i], beta5[i], rfree[i] = pval.T.values
 
    


# In[ ]:





# In[393]:


data = pd.DataFrame(data={'RF':rfree, 'Mkt-RF':beta1, 'SMB':beta2, 'HML':beta3, 'RMW':beta4, 'CMA':beta5},
                   index = portfolios100.columns) #.reset_index().rename(columns={"index":"Porfolios"})


# In[394]:


data.head()


# In[398]:



data.iloc[:,1:5].plot(figsize=(14,8))
plt.xticks(rotation=45)
plt.title("Significance Distribution");


# Clearly only Mkt-RF, SMB, HML and RWL are statistically significant.
