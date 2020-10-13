
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import iqr
from scipy import signal

import pandas_datareader.wb as wb

import holoviews as hv
import hvplot.pandas

hv.extension('bokeh')
np.random.seed(42)


def i(*args, **kwargs):
    i = np.linspace(-6,6,100).reshape(-1,1)
    i = i[i!=0]
    return i

def IS(i=i(), Z_1=0, df=True):
    return pd.DataFrame([i, -i+Z_1], index=['Real Interest','Real Output']).T if df else -i+Z_1


def LM(i=i(), Z_2=0, df=True):
    return pd.DataFrame([i, i-Z_2], index=['Real Interest','Real Output']).T if df else i-Z_2


def IP(i=i(), Z_3=0, df=True):
    return pd.DataFrame([i, i-Z_3], index=['Real Interest','Exchange Rate']).T if df else i+Z_3


def findIntersection(fun1, fun2, x0):
    return fsolve(lambda x: fun1(x) - fun2(x), x0


def IS_LM_IP(z_1=0, z_2=0, z_3=0):
    
    is_eq = IS(i(), Z_1=0)
    lm_eq = LM(i(), Z_2=0)
    ip_eq = IP(i(), Z_3=0)
    
    is_shock = IS(i(), Z_1=z_)
    lm_shock = LM(i(), Z_2=z_2)
    ip_shock = IP(i(), Z_3=z_3)
    
    result = findIntersection(lambda x: LM(i-x, Z_2=z_2, df=False), lambda x: IS(i-x, Z_1=z_1, df=False), 0.0)
    result = result + 1e-4 if result==0 else result
    
    is_im_plot = hv.Curve(lm_eq, vdims="Real Interest", kdims="Real Output").options(alpha=0.2, color='#1883F5') *\
            hv.Curve(is_eq, vdims="Real Interest", kdims="Real Output").options(alpha=0.2, color='orange') *\
            hv.Curve(lm_shock, vdims="Real Interest", kdims="Real Output", label='LM').options(color='#1883F5') *\
            hv.Curve(is_shock, vdims="Real Interest", kdims="Real Output", label='IS').options(alpha=1,color='orange') *\
            hv.VLine(LM(i=result[0], Z_2=z_2, df=False)).options(line_width=1, alpha=0.2, color='black') *\
            hv.HLine(result[0]).options(line_width=1, alpha=0.2, color='black') 
    
    ip_plot = hv.Curve(ip_eq, vdims="Real Interest", kdims='Exchange Rate').options(alpha=0.2, color='#33CC00') *\
            hv.Curve(ip_shock, vdims="Real Interest", kdims='Exchange Rate', label='IP').options(color='#33CC00') *\
            hv.VLine(LM(i=result[0], Z_3=z_3, df=False)).options(line_width=1, alpha=0.2, color='black') *\
            hv.HLine(result[0]).options(line_width=1, alpha=0.2, color='black') 
    
    return is_lm_ip + ip_plot
    
