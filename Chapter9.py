# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:10:38 2023

@author: XIE Ming
@github: https://github.com/BobMing
@linkedIn: https://www.linkedin.com/in/tseming
@email: xieming_xm@163.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

IRS_data=pd.read_excel(io=r'E:\OneDrive\附件\数据\第9章\利率互换交易规模.xls',sheet_name='Sheet1',header=0,index_col=0)
name=IRS_data.index #获取数据框关于参考利率类型的利率名称
volume=(np.array(IRS_data)).ravel() #将数据框涉及名义本金的数值转为一维数组
plt.figure(figsize=(9,7))
plt.pie(x=volume,labels=name,textprops={'fontsize':13})
plt.axis('equal')
plt.show()

currency=['美元与人民币','非美元外币与人民币']
volume1=[158.45,10.67]
tenor=['不超过1年','超过1年']
volume2=[141.29,27.83]
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.pie(x=volume1,labels=currency,textprops={'fontsize':13})
plt.axis('equal')
plt.title(u'不同交换币种',fontsize=14)
plt.subplot(1,2,2)
plt.pie(x=volume2,labels=tenor,textprops={'fontsize':13})
plt.axis('equal')
plt.title(u'不同期限',fontsize=14)
plt.tight_layout()
plt.show()

CRM_data=pd.read_excel(io=r'E:\OneDrive\附件\数据\第9章\未到期信用风险缓释工具合约面值（2020年末）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
type_CRM=CRM_data.index #获取数据框中关于合约创设机构类型
par_CRM=(np.array(CRM_data)).ravel() #将数据框涉及合约面值的数值转为一维数组
plt.figure(figsize=(9,7))
plt.pie(x=par_CRM,labels=type_CRM,textprops={'fontsize':13})
plt.axis('equal')
plt.tight_layout()
plt.show()

def IRS_cashflow(R_flt,R_fix,L,m,position):
    '''定义一个计算利率互换合约存续期内每期支付利息净额的函数
    R_flt: 代表利率互换的每期浮动利率，以数组格式输入；
    R_fix: 代表利率互换的固定利率。
    L: 代表利率互换的本金。
    m: 代表利率互换存续期内每年交换利息的频次。
    position: 代表头寸方向，输入position='long'代表多头（支付固定利息、收取浮动利息），输入其他则代表空头（支付浮动利息、收取固定利息）'''
    if position=='long':
        cashflow=(R_flt-R_fix)*L/m
    else:
        cashflow=(R_fix-R_flt)*L/m
    return cashflow
# Example 9-1
rate_float=np.array([0.031970,0.032000,0.029823,0.030771,0.044510,0.047093,0.043040,0.032750,0.029630,0.015660])
rate_fixed=0.037
par=1e8
M=2
Netpay_A=IRS_cashflow(R_flt=rate_float,R_fix=rate_fixed,L=par,m=M,position='long')
print(Netpay_A)
Netpay_B=IRS_cashflow(R_flt=rate_float,R_fix=rate_fixed,L=par,m=M,position='short')
print(Netpay_B)

def swap_rate(m,y,T):
    '''定义一个计算互换利率的函数
    m: 代表利率互换合约存续期内每年交换利息的频次。
    y: 代表合约初始日对应于每期利息交换期限、连续复利的零息利率（贴现利率），用数组格式输入。
    T: 代表利率互换的期限（年）'''
    n_list=np.arange(1,m*T+1) #创建一个1~mT的整数数组
    t=n_list/m #计算合约初始日距离每期利息交换日的期限数组
    q=np.exp(-y*t) #计算针对不同期限的贴现因子（数组格式）
    rate=m*(1-q[-1])/np.sum(q)
    return rate
# Example 9-2
freq=2
tenor=3
r_list=np.array([0.020579,0.021276,0.022080,0.022853,0.023527,0.024036])
R_July1=swap_rate(m=freq,y=r_list,T=tenor)
print('2020年7月1日利率互换合约的互换利率',round(R_July1,4))

def swap_value(R_fix,R_flt,t,y,m,L,position):
    '''定义一个计算合约存续期内利率互换合约价值的函数
    R_fix: 代表利率互换合约的固定（互换）利率。
    R_flt: 代表距离合约定价日最近的下一期利息交换的浮动利率。
    t: 代表合约定价日距离每期利息交换日的期限（年），用数组格式输入。
    y: 代表期限为t且连续复利的零息利率（贴现利率），用数组格式输入。
    m: 代表利率互换合约每年交换利息的频次。
    L: 代表利率互换合约的本金。
    position: 代表头寸方向，输入position='long'代表多头（支付固定利息、收取浮动利息），输入其他则代表空头（支付浮动利息、收取固定利息）'''
    from numpy import exp
    B_fix=(R_fix*sum(exp(-y*t))/m+exp(-y[-1]*t[-1]))*L
    B_flt=(R_flt/m+1)*L*exp(-y[0]*t[0])
    if position=='long':
        value=B_flt-B_fix
        
    else:
        value=B_fix-B_flt
    return value
# Example 9-3
# 第1步：通过表中已知零息利率，运用3阶样条曲线插值法计算1.5年期和2.5年期的零息利率
import scipy.interpolate as si
T=np.array([1/12,2/12,3/12,6/12,9/12,1.0,2.0,3.0])
R_July10=np.array([0.017219,0.017526,0.021012,0.021100,0.021764,0.022165,0.025040,0.026894])
R_July20=np.array([0.016730,0.018373,0.019934,0.020439,0.021621,0.022540,0.024251,0.025256])
func_July10=si.interp1d(x=T,y=R_July10,kind='cubic') #运用2020年7月10日的零息利率数据和3阶样条曲线插值法构建插值函数
func_July20=si.interp1d(x=T,y=R_July20,kind='cubic')
T_new=np.array([1/12,2/12,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0])
R_new_July10=func_July10(T_new)
print(R_new_July10)
R_new_July20=func_July20(T_new)
print(R_new_July20)
# 第2步：计算合约定价日距离每期利息交换日的期限
import datetime as dt
T1=dt.datetime(2020,7,10)
T2=dt.datetime(2020,7,20)
T3=dt.datetime(2021,1,1) #输入下一期利息交换日
tenor1=(T3-T1).days/365
tenor2=(T3-T2).days/365
T=3
M=2
T_list1=np.arange(T*M)/M #创建存放2020年7月10日距离每期利息交换日期限的初始数组
T_list1+=tenor1 #计算相关期限
T_list2=np.arange(T*M)/M #创建存放2020年7月20日距离每期利息交换日期限的初始数组
T_list2+=tenor2
# 第3步：用自定义函数swap_value计算2个不同交易日的合约价值
yield_July10=np.zeros_like(T_list1) #创建存放2020年7月10日对应每期利息交换期限的零息利率初始数组
yield_July10[0]=R_new_July10[3] #存放2020年7月10日6个月期零息利率
yield_July10[1:]=R_new_July10[5:] #存放2020年7月10日1年期、1.5年期、2年期、2.5年期、3年期零息利率
yield_July20=np.zeros_like(T_list2)
yield_July20[0]=R_new_July20[3]
yield_July20[1:]=R_new_July20[5:]
rate_fix=0.0241
rate_float=0.02178
par=1e8
value_July10_long=swap_value(R_fix=rate_fix,R_flt=rate_float,t=T_list1,y=yield_July10,m=M,L=par,position='long')
value_July10_short=swap_value(R_fix=rate_fix,R_flt=rate_float,t=T_list1,y=yield_July10,m=M,L=par,position='short')
print('2020年7月10日C银行（多头）的利率互换合约价值',round(value_July10_long,2))
print('2020年7月10日D银行（空头）的利率互换合约价值',round(value_July10_short,2))
value_July20_long=swap_value(R_fix=rate_fix,R_flt=rate_float,t=T_list2,y=yield_July20,m=M,L=par,position='long')
value_July20_short=swap_value(R_fix=rate_fix,R_flt=rate_float,t=T_list2,y=yield_July20,m=M,L=par,position='short')
print('2020年7月20日C银行（多头）的利率互换合约价值',round(value_July20_long,2))
print('2020年7月20日D银行（空头）的利率互换合约价值',round(value_July20_short,2))

def CCS_fixed_cashflow(La,Lb,Ra_fix,Rb_fix,m,T,trader,par):
    '''定义一个计算双固定利率货币互换在存续期间每期现金流的函数
    合约的交易双方分别用A交易方、B交易方表示
    La: 代表在合约初始日A交易方支付的一种货币本金（合约到期日A交易方收回的货币本金）。
    Lb: 代表在合约初始日B交易方支付的 另一种货币本金（合约到期日B交易方收回的货币本金）。
    Ra_fix: 代表基于本金La的固定利率。
    Rb_fix: 代表基于本金Lb的固定利率。
    m: 代表每年交换利息的频次。
    T: 代表货币互换合约的期限（年）。
    trader: 代表合约的交易方，输入trader='A'代表计算A交易方发生的期间现金流，输入其他则表示计算B交易方发生的期间现金流。
    par: 代表计算现金流所依据的本金，输入par='La'表示计算的现金流基于本金La，输入其他则表示计算的现金流基于本金Lb'''
    cashflow=np.zeros(m*T+1) #创建存放每期现金流的初始数组
    if par=='La':
        cashflow[0]=-La
        cashflow[1:-1]=Ra_fix*La/m
        cashflow[-1]=(Ra_fix/m+1)*La
        if trader=='A':
            return cashflow
        else:
            return -cashflow
    else:
        cashflow[0]=Lb
        cashflow[1:-1]=-Rb_fix*Lb/m
        cashflow[-1]=-(Rb_fix/m+1)*Lb
        if trader=='A':
            return cashflow
        else:
            return -cashflow
# Example 9-4
par_RMB=6.4e8
par_USD=1e8
rate_RMB=0.02
rate_USD=0.01
M=2
tenor=5
cashflow_Ebank_RMB=CCS_fixed_cashflow(La=par_RMB,Lb=par_USD,Ra_fix=rate_RMB,Rb_fix=rate_USD,m=M,T=tenor,trader='A',par='La')
cashflow_Ebank_USD=CCS_fixed_cashflow(La=par_RMB,Lb=par_USD,Ra_fix=rate_RMB,Rb_fix=rate_USD,m=M,T=tenor,trader='A',par='Lb')
print('E银行基于人民币本金的每期现金流（人民币）\n',cashflow_Ebank_RMB)
print('E银行基于美元本金的每期现金流（美元）\n',cashflow_Ebank_USD)
cashflow_Fbank_RMB=CCS_fixed_cashflow(La=par_RMB,Lb=par_USD,Ra_fix=rate_RMB,Rb_fix=rate_USD,m=M,T=tenor,trader='B',par='La')
cashflow_Fbank_USD=CCS_fixed_cashflow(La=par_RMB,Lb=par_USD,Ra_fix=rate_RMB,Rb_fix=rate_USD,m=M,T=tenor,trader='B',par='Lb')
print('F银行基于人民币本金的每期现金流（人民币）\n',cashflow_Fbank_RMB)
print('F银行基于美元本金的每期现金流（美元）\n',cashflow_Fbank_USD)

def CCS_fixflt_cashflow(La,Lb,Ra_fix,Rb_flt,m,T,trader,par):
    '''定义一个计算固定对浮动货币互换在存续期间每期现金流的函数
    合约的交易双方分别用A交易方、B交易方表示
    La: 代表在合约初始日A交易方支付的一种货币本金（合约到期日A交易方收回的货币本金）。
    Lb: 代表在合约初始日B交易方支付的 另一种货币本金（合约到期日B交易方收回的货币本金）。
    Ra_fix: 代表基于本金La的固定利率。
    Rb_flt: 代表基于本金Lb的浮动利率，并且以数组格式输入。
    m: 代表每年交换利息的频次。
    T: 代表货币互换合约的期限（年）。
    trader: 代表合约的交易方，输入trader='A'代表计算A交易方发生的期间现金流，输入其他则表示计算B交易方发生的期间现金流。
    par: 代表计算现金流所依据的本金，输入par='La'表示计算的现金流基于本金La，输入其他则表示计算的现金流基于本金Lb'''
    cashflow=np.zeros(m*T+1) #创建存放每期现金流的初始数组
    if par=='La':
        cashflow[0]=-La
        cashflow[1:-1]=Ra_fix*La/m
        cashflow[-1]=(Ra_fix/m+1)*La
        if trader=='A':
            return cashflow
        else:
            return -cashflow
    else:
        cashflow[0]=Lb
        cashflow[1:-1]=-Rb_flt[:-1]*Lb/m #A交易方第2至倒数第2期现金流
        cashflow[-1]=-(Rb_flt[-1]/m+1)*Lb #A交易方最后一期现金流
        if trader=='A':
            return cashflow
        else:
            return -cashflow

def CCS_float_cashflow(La,Lb,Ra_flt,Rb_flt,m,T,trader,par):
    '''定义一个计算双浮动货币互换在存续期间每期现金流的函数
    合约的交易双方分别用A交易方、B交易方表示
    La: 代表在合约初始日A交易方支付的一种货币本金（合约到期日A交易方收回的货币本金）。
    Lb: 代表在合约初始日B交易方支付的 另一种货币本金（合约到期日B交易方收回的货币本金）。
    Ra_fix: 代表基于本金La的浮动利率，并且以数组格式输入。
    Rb_flt: 代表基于本金Lb的浮动利率，并且以数组格式输入。
    m: 代表每年交换利息的频次。
    T: 代表货币互换合约的期限（年）。
    trader: 代表合约的交易方，输入trader='A'代表计算A交易方发生的期间现金流，输入其他则表示计算B交易方发生的期间现金流。
    par: 代表计算现金流所依据的本金，输入par='La'表示计算的现金流基于本金La，输入其他则表示计算的现金流基于本金Lb'''
    cashflow=np.zeros(m*T+1) #创建存放每期现金流的初始数组
    if par=='La':
        cashflow[0]=-La
        cashflow[1:-1]=Ra_flt[:-1]*La/m #A交易方第2至倒数第2期现金流
        cashflow[-1]=(Ra_flt[-1]/m+1)*La #A交易方最后一期现金流
        if trader=='A':
            return cashflow
        else:
            return -cashflow
    else:
        cashflow[0]=Lb
        cashflow[1:-1]=-Rb_flt[:-1]*Lb/m #A交易方第2至倒数第2期现金流
        cashflow[-1]=-(Rb_flt[-1]/m+1)*Lb #A交易方最后一期现金流
        if trader=='A':
            return cashflow
        else:
            return -cashflow

# Example 9-5
par_RMB1=6.9e8
par_USD=1e8
par_RMB2=1.8e8
par_HKD=2e8
M1=2
M2=1
T1=3
T2=4
rate_fix=0.03
Libor=np.array([0.012910,0.014224,0.016743,0.024744,0.028946,0.025166])
Shibor=np.array([0.031600,0.046329,0.035270,0.031220])
Hibor=np.array([0.013295,0.015057,0.026593,0.023743])
cashflow_Gbank_RMB1=CCS_fixflt_cashflow(La=par_RMB1,Lb=par_USD,Ra_fix=rate_fix,Rb_flt=Libor,m=M1,T=T1,trader='A',par='La')
cashflow_Gbank_USD=CCS_fixflt_cashflow(La=par_RMB1,Lb=par_USD,Ra_fix=rate_fix,Rb_flt=Libor,m=M1,T=T1,trader='A',par='Lb')
print('第1份货币互换合约在存续期内G银行的人民币现金流\n',cashflow_Gbank_RMB1)
print('第1份货币互换合约在存续期内G银行的美元现金流\n',cashflow_Gbank_USD)
cashflow_Hbank_RMB1=CCS_fixflt_cashflow(La=par_RMB1,Lb=par_USD,Ra_fix=rate_fix,Rb_flt=Libor,m=M1,T=T1,trader='B',par='La')
cashflow_Hbank_USD=CCS_fixflt_cashflow(La=par_RMB1,Lb=par_USD,Ra_fix=rate_fix,Rb_flt=Libor,m=M1,T=T1,trader='B',par='Lb')
print('第1份货币互换合约在存续期内H银行的人民币现金流\n',cashflow_Hbank_RMB1)
print('第1份货币互换合约在存续期内H银行的美元现金流\n',cashflow_Hbank_USD)

cashflow_Gbank_RMB2=CCS_float_cashflow(La=par_RMB2,Lb=par_HKD,Ra_flt=Shibor,Rb_flt=Hibor,m=M2,T=T2,trader='A',par='La')
cashflow_Gbank_HKD=CCS_float_cashflow(La=par_RMB2,Lb=par_HKD,Ra_flt=Shibor,Rb_flt=Hibor,m=M2,T=T2,trader='A',par='Lb')
print('第2份货币互换合约在存续期内G银行的人民币现金流\n',cashflow_Gbank_RMB2)
print('第2份货币互换合约在存续期内G银行的美元现金流\n',cashflow_Gbank_HKD)
cashflow_Hbank_RMB2=CCS_float_cashflow(La=par_RMB2,Lb=par_HKD,Ra_flt=Shibor,Rb_flt=Hibor,m=M2,T=T2,trader='B',par='La')
cashflow_Hbank_HKD=CCS_float_cashflow(La=par_RMB2,Lb=par_HKD,Ra_flt=Shibor,Rb_flt=Hibor,m=M2,T=T2,trader='B',par='Lb')
print('第2份货币互换合约在存续期内H银行的人民币现金流\n',cashflow_Hbank_RMB2)
print('第2份货币互换合约在存续期内H银行的美元现金流\n',cashflow_Hbank_HKD)

def CCS_value(types,La,Lb,Ra,Rb,ya,yb,E,m,t,trader):
    '''定义一个计算在合约存续期内货币互换合约价值的函数，交易双方是A交易方、B交易方，同时约定A交易方在合约初始日支付A货币本金，B交易方在合约初始日支付B货币本金
    types: 代表货币互换类型，输入types='双固定利率货币互换'表示计算双固定利率货币互换，输入types='双浮动利率货币互换'表示计算双浮动利率货币互换，输入其他则表示计算固定对浮动货币互换；并约定针对固定对浮动货币互换，固定利率针对A货币本金，浮动利率针对B货币本金。
    La: 代表A货币本金。
    Lb: 代表B货币本金。
    Ra: 代表针对A货币本金的利率。
    Rb: 代表针对B货币本金的利率。
    ya: 代表在合约定价日针对A货币本金并对应不同期限、连续复利的零息利率，用数组格式输入。
    yb: 代表在合约定价日针对B货币本金并对应不同期限、连续复利的零息利率，用数组格式输入。
    E: 代表合约定价日的即期汇率，标价方式是1单位B货币对应的A货币数量。
    m: 代表每年交换利息的频次。
    t: 代表合约定价日距离剩余每期利息交换日的期限长度，用数组格式输入。
    trader: 代表交易方，输入trader='A'表示A交易方，输入其他则表示B交易方'''
    from numpy import exp
    if types=='双固定利率货币互换':
        Bond_A=(Ra*sum(exp(-ya*t))/m+exp(-ya[-1]*t[-1]))*La
        Bond_B=(Rb*sum(exp(-yb*t))/m+exp(-yb[-1]*t[-1]))*Lb
    elif types=='双浮动利率货币互换':
        Bond_A=(Ra/m+1)*exp(-ya[0]*t[0])*La
        Bond_B=(Rb/m+1)*exp(-yb[0]*t[0])*Lb
    else:
        Bond_A=(Ra*sum(exp(-ya*t))/m+exp(-ya[-1]*t[-1]))*La
        Bond_B=(Rb/m+1)*exp(-yb[0]*t[0])*Lb
    if trader=='A':
        swap_value=Bond_A-Bond_B*E
    else:
        swap_value=Bond_B-Bond_A/E
    return swap_value
# Example 9-6
y_RMB_Apr1=np.array([0.016778,0.019062,0.019821])
M=1
tenor=3
rate_RMB=swap_rate(m=M,y=y_RMB_Apr1,T=tenor)
print('货币互换合约针对人民币本金的固定利率',round(rate_RMB,4))
FX_Apr1=7.0771
par_USD=1e8
par_RMB=par_USD*FX_Apr1
Libor_Apr1=0.010024
y_RMB_Jun18=np.array([0.021156,0.023294,0.023811])
y_USD_Jun18=np.array([0.0019,0.0019,0.0022])
FX_Jun18=7.0903
y_RMB_Jul20=np.array([0.022540,0.024251,0.025256])
y_USD_Jul20=np.array([0.0014,0.0016,0.0018])
FX_Jul20=6.9928
t0=dt.datetime(2020,4,1)
t1=dt.datetime(2020,6,18)
t2=dt.datetime(2020,7,20)
t1_list=np.arange(1,tenor+1)-(t1-t0).days/365
t2_list=np.arange(1,tenor+1)-(t2-t0).days/365
value_RMB_Jun18=CCS_value(types='固定对浮动汇率货币互换',La=par_RMB,Lb=par_USD,Ra=rate_RMB,Rb=Libor_Apr1,ya=y_RMB_Jun18,yb=y_USD_Jun18,E=FX_Jun18,m=M,t=t1_list,trader='A')
value_USD_Jun18=CCS_value(types='固定对浮动汇率货币互换',La=par_RMB,Lb=par_USD,Ra=rate_RMB,Rb=Libor_Apr1,ya=y_RMB_Jun18,yb=y_USD_Jun18,E=FX_Jun18,m=M,t=t1_list,trader='B')
value_RMB_Jul20=CCS_value(types='固定对浮动汇率货币互换',La=par_RMB,Lb=par_USD,Ra=rate_RMB,Rb=Libor_Apr1,ya=y_RMB_Jul20,yb=y_USD_Jul20,E=FX_Jul20,m=M,t=t2_list,trader='A')
value_USD_Jul20=CCS_value(types='固定对浮动汇率货币互换',La=par_RMB,Lb=par_USD,Ra=rate_RMB,Rb=Libor_Apr1,ya=y_RMB_Jul20,yb=y_USD_Jul20,E=FX_Jul20,m=M,t=t2_list,trader='B')
print('2020年6月18日J银行的货币互换合约价值（人民币）',round(value_RMB_Jun18,2))
print('2020年6月18日K银行的货币互换合约价值（美元）',round(value_USD_Jun18,2))
print('2020年7月20日J银行的货币互换合约价值（人民币）',round(value_RMB_Jul20,2))
print('2020年7月20日K银行的货币互换合约价值（美元）',round(value_USD_Jul20,2))

def CDS_cashflow(S,m,T1,T2,L,recovery,trader):
    '''定义一个计算信用违约互换期间现金流的函数
    S: 代表信用违约互换价差（信用保护费用）。
    m: 代表信用违约互换价差每年支付的频次，且不超过2次。
    T1: 代表合约期限（年）。
    T2: 代表合约初始日距离信用事件发生日的期限长度（年），信用事件未发生则输入T2='Na'。
    L: 代表合约的本金。
    recovery: 代表信用事件发生时的回收率，信用事件未发生则输入recovery='Na'。
    trader: 代表交易方，输入trader='buyer'表示买方，输入其他则表示卖方。
    event: 代表信用事件，输入event='N'表示合约存续期内信用事件未发生，输入其他则表示合约存续期内信用事件发生'''
    if T2=='Na':
        n=m*T1
        cashflow=S/m*L*np.ones(n)
    else:
        default_pay=(1-recovery)*L
        if m==1:
            n=int(T2)+1 #进一法，最后一次是信用事件发生时
            cashflow=S*L*np.ones(n)
            spread_end=(T2-int(T2))*S*L
        else:
            if T2-int(T2)<0.5: #信用事件发生在上半年
                n=int(T2)*m+1
                cashflow=S/m*L*np.ones(n)
                spread_end=(T2-int(T2))*S*L
            else:
                n=(int(T2)+1)*m
                cashflow=S/m*L*np.ones(n)
                spread_end=(T2-int(T2)-0.5)*S*L
        cashflow[-1]=spread_end-default_pay
    if trader=='buyer':
        CF=-cashflow
    else:
        CF=cashflow
    return CF
# Example 9-7
spread=0.012
M=1
tenor=3
par=1e8
cashflow_buyer1=CDS_cashflow(S=spread, m=M, T1=tenor, T2='Na', L=par, recovery='Na', trader='buyer')
cashflow_seller1=CDS_cashflow(S=spread, m=M, T1=tenor, T2='Na', L=par, recovery='Na', trader='seller')
print('未发生信用事件情形下合约期间买方的现金流',cashflow_buyer1)
print('未发生信用事件情形下合约期间卖方的现金流',cashflow_seller1)
T_default=28/12
rate=0.35
cashflow_buyer2=CDS_cashflow(S=spread, m=M, T1=tenor, T2=T_default, L=par, recovery=rate, trader='buyer')
cashflow_seller2=CDS_cashflow(S=spread, m=M, T1=tenor, T2=T_default, L=par, recovery=rate, trader='seller')
print('发生信用事件情形下合约期间买方的现金流',cashflow_buyer2)
print('发生信用事件情形下合约期间卖方的现金流',cashflow_seller2)

# Example 9-8
M_new=2
T_default_new=32/12
cashflow_buyer3=CDS_cashflow(S=spread,m=M_new,T1=tenor,T2=T_default_new,L=par,recovery=rate,trader='buyer')
cashflow_seller3=CDS_cashflow(S=spread, m=M_new, T1=tenor, T2=T_default_new, L=par, recovery=rate, trader='seller')
print('发生信用事件情况下合约期间买方的现金流（新）\n',cashflow_buyer3)
print('发生信用事件情况下合约期间卖方的现金流（新）\n',cashflow_seller3)

# Example 9-9 累积违约概率CDP、存活率SR、边际违约概率MDP
h=0.03 #连续复利的违约概率
T=5 #期限
CDP=np.ones(T) #创建存放累积违约概率的初始数组
for t in range(1,T+1):
    CDP[t-1]=1-np.exp(-h*t) #计算累积违约概率
print(CDP.round(4))
SR=1-CDP
print(SR.round(4))
MDP=np.ones_like(CDP)
MDP[0]=CDP[0]
for t in range(1,T):
    MDP[t]=SR[t-1]-SR[t]
print(MDP.round(4))

def CDS_spread(m,Lamda,T,R,y):
    '''定义一个计算信用违约互换价差（年化）的函数
    m: 代表信用违约互换价差（信用保护费用）每年支付的频次。
    Lamda: 代表连续复利的年化违约概率。
    T: 代表合约期限（年）。
    R: 代表信用事件发生时的回收率。
    y: 代表对应合约初始日距离每期信用保护费用支付日的期限且连续复利的零息利率，用数组格式输入'''
    from numpy import arange,exp
    t_list=arange(m*T+1)/m
    A=sum(exp(-Lamda*t_list[:-1]-y*t_list[1:]))
    B=sum(exp(-(Lamda+y)*t_list[1:]))
    spread=m*(1-R)*(A/B-1)
    return spread
# Example 9-10 计算信用违约互换价差
zero_rate=np.array([0.021276,0.022853,0.024036,0.025010,0.025976])
recovery=0.4
M=1
tenor=5
h=0.03 #连续复利的年化违约概率
spread=CDS_spread(m=M,Lamda=h,T=tenor,R=recovery,y=zero_rate)
print('计算得到信用违约互换价差',spread.round(4))

# Example 9-11 敏感性分析：连续复利的违约概率h、违约回收率R对信用违约互换价差s的影响
h_list=np.linspace(0.01, 0.06, 200)
spread_list1=np.zeros_like(h_list)
for i in range(len(h_list)):
    spread_list1[i]=CDS_spread(m=M,Lamda=h_list[i],T=tenor,R=recovery,y=zero_rate)
r_list=np.linspace(0.1, 0.6, 200)
spread_list2=np.zeros_like(r_list)
for i in range(len(h_list)):
    spread_list2[i]=CDS_spread(m=M, Lamda=h, T=tenor, R=r_list[i], y=zero_rate)
plt.figure(figsize=(11,6))
plt.subplot(1,2,1)
plt.plot(h_list,spread_list1,'r-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'违约概率',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'信用违约互换价差',fontsize=13)
plt.title(u'违约概率与信用违约互换价差的关系图',fontsize=14)
plt.grid()
plt.subplot(1,2,2,sharey=plt.subplot(1,2,1))
plt.plot(r_list,spread_list2,'b-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'违约回收率',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'违约回收率与信用违约互换价差的关系图',fontsize=14)
plt.grid()
plt.show()