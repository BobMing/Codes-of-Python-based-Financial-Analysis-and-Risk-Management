# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:25:39 2023

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

# Example 14-1 测度首只违约债券——超日债的违约概率
# Step 1: 计算股票收益率的年化波动率
price_Sun=pd.read_excel(r'E:\OneDrive\附件\数据\第14章\超日太阳股票收盘价（2012年8月至2014年2月）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
return_Sun=np.log(price_Sun/price_Sun.shift(1)) #计算股票每日收益率
sigma_Sun=np.sqrt(252)*np.std(return_Sun) #计算股票年化波动率
sigma_Sun=float(sigma_Sun) #转换为浮点型数据
print('超日太阳股票收益率的年化波动率',round(sigma_Sun,4))
# Step 2: 计算超日太阳在2014年2月19日的企业价值（变量V0）及企业价值的年化波动率（变量σV）
equity=21.85 #2014年2月19日股票总市值（亿元）
debt=63.90 #2013年9月末公司负债金额（亿元）
tenor=1 #债务期限为1年
rate=0.050001 #2014年2月19日1年期Shibor（无风险收益率）
def f(x): #通过定义一个函数计算企业价值和企业价值的年化波动率
    from numpy import exp,log,sqrt
    from scipy.stats import norm
    V,sigma_V=x #设定两个变量分别是当前企业价值和企业价值的年化波动率
    d1=(log(V/debt)+(rate+sigma_V**2/2)*tenor)/(sigma_V*sqrt(tenor))
    d2=d1-sigma_V*sqrt(tenor)
    eq1=V*norm.cdf(d1)-debt*exp(-rate*tenor)*norm.cdf(d2)-equity #运用式（14-2）并等于0
    eq2=sigma_Sun*equity-norm.cdf(d1)*sigma_V*V #运用式（14-3）并等于0
    return [eq1,eq2]
import scipy.optimize as sco
result=sco.fsolve(func=f,x0=[80,0.5]) #设定初始值分别是企业价值80亿元、企业价值的年化波动率50%
print('计算得到2014年2月19日超日太阳的企业价值（亿元）',round(result[0],4))
print('计算得到超日太阳企业价值的年化波动率',round(result[-1],4))
# Step 3: 计算在2014年2月19日超日太阳的违约概率
def PD_Merton(E,D,V,sigma,r,T):
    '''定义一个运用默顿模型计算企业违约概率的函数
    E: 代表当前的股票市值（亿元）。
    D: 代表债务本金（亿元）。
    V: 代表当前的企业价值（亿元）。
    sigma: 代表企业价值的年化波动率。
    r: 代表无风险收益率。
    T: 代表债务的期限'''
    from numpy import log,sqrt
    from scipy.stats import norm
    d1=(log(V/D)+(r+sigma**2/2)*T)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    PD=norm.cdf(-d2) #计算违约概率 1-N(d2)=N(-d2)
    return PD
PD_Sun=PD_Merton(equity,debt,result[0],result[-1],rate,tenor)
print('2014年2月19日超日太阳的违约概率',round(PD_Sun,6))
# Step 4: 计算2011年末超日太阳的违约概率
equity_new=70.75 #2011年12月30日股票总市值（亿元）
debt_new=28.02 #2011年9月末负债金额（亿元）
rate_new=0.052378 #2011年12月30日1年期Shibor
sigma_new=0.4654 #股票的年化波动率
def g(x): #重新定义一个函数用于计算企业价值和企业价值的年化波动率
    from numpy import exp,log,sqrt
    from scipy.stats import norm
    V,sigma_V=x
    d1=(log(V/debt_new)+(rate_new+sigma_V**2/2)*tenor)/(sigma_V*sqrt(tenor))
    d2=d1-sigma_V*sqrt(tenor)
    eq1=V*norm.cdf(d1)-debt_new*exp(-rate_new*tenor)*norm.cdf(d2)-equity_new
    eq2=sigma_new*equity_new-norm.cdf(d1)*sigma_V*V #运用式（14-3）并等于0
    return [eq1,eq2]
result_new=sco.fsolve(func=g,x0=[80,0.5])
print('2011年12月30日超日太阳的企业价值（亿元）',round(result_new[0],4))
print('超日太阳企业价值的年化波动率',round(result_new[-1],4))
PD_Sun_new=PD_Merton(equity_new,debt_new,result_new[0],result_new[-1],rate_new,tenor)
print('2011年12月30日超日太阳的违约概率',round(PD_Sun_new,6))
M=PD_Sun/PD_Sun_new
print('2014年2月19日违约概率与2011年末违约概率的倍数',round(M,2))

# 2010年至2020年期间A股市场的可转换债券数量和存续规模
data_CB=pd.read_excel(r'E:\OneDrive\附件\数据\第14章\可转换债券数量和存续金额（2010年至2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
data_CB.plot(kind='bar',subplots=True,layout=(1,2),figsize=(9,6),grid=True,fontsize=13) #可视化
plt.subplot(1,2,1)
plt.ylabel(u'数量或金额',fontsize=13)

# Example 14-2 可转换债券的定价
def value_CB(S,sigma,par,X,Lambda,r,R,Q2,T,N):
    '''定义一个运用N步二叉树模型计算可转换债券（可转债）价值的函数，同时假定可转债是一份零息债券
    S: 代表股票的厨师价格（当前价格）。
    sigma: 代表股票收益率的年化波动率。
    par: 代表可转债本金。
    X: 代表1份可转债转换为股票的股数（转股比例）。
    Lambda: 代表连续复利的年化违约概率。
    r: 代表连续复利的无风险收益率。
    R: 代表可转债违约时的回收率。
    Q2: 代表可转债的赎回价格。
    T: 代表可转债的期限（年）。
    N: 代表二叉树模型的步数'''
    #第1步：计算相关参数
    t=T/N
    u=np.exp(np.sqrt((sigma**2-Lambda)*t))
    d=1/u
    Pu=(np.exp(r*t)-d*np.exp(-Lambda*t))/(u-d)
    Pd=(u*np.exp(-Lambda*t)-np.exp(r*t))/(u-d)
    P_default=1-np.exp(-Lambda*t)
    D_value=par*R
    CB_matrix=np.zeros((N+1,N+1)) #构建N+1行、N+1列的零矩阵，用于后续存放每个节点的可转债价值
    #第2步：计算可转债到期时节点的股价与债券价值
    N_list=np.arange(N+1) #创建从0到N的自然数数列（数组格式）
    S_end=S*u**(N-N_list)*d**N_list #计算可转债到期时节点的股价（按照节点从上往下排序）
    Q1=par #可转债到期时的本金（不转股、不赎回）
    Q3=X*S_end #可转债到期时转换为股票的价值
    CB_matrix[:,-1]=np.maximum(np.minimum(Q1,Q2),Q3) #计算可转债到期时节点的债券价值（按照节点从上往下排序）
    #第3步：计算可转债非到期时节点的股价与债券价值
    i_list=list(range(N)) #创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()
    for i in i_list:
        j_list=np.arange(i+1) #创建从0到i的自然数数列（数组格式）
        Si=S*u**(i-j_list)*d**j_list #计算在iΔt时刻节点的股价（按照节点从上往下排序）
        Q1=np.exp(-r*t)*(Pu*CB_matrix[:i+1,i+1]+Pd*CB_matrix[1:i+2,i+1]+P_default*D_value) #计算在iΔt时刻节点不转股、不赎回时的债券价值
        Q3=X*Si
        CB_matrix[:i+1,i]=np.maximum(np.minimum(Q1,Q2),Q3) #计算在iΔt时刻节点的可转债价值
    return CB_matrix[0,0]
tenor=9/12
step1=3 #二叉树模型的步数为3步
S0=50
sigma_A=0.2
par_CB=100
share=2 #转股比例
Lambda_A=0.01
rate=0.05
R_A=0.4
Q2_A=110
V1_CB=value_CB(S0,sigma_A,par_CB,share,Lambda_A,rate,R_A,Q2_A,tenor,step1)
print('运用',step1,'步二叉树模型计算可转换债券初始价值',round(V1_CB,4))

step2=100
V2_CB=value_CB(S0,sigma_A,par_CB,share,Lambda_A,rate,R_A,Q2_A,tenor,step2)
print('运用',step2,'步二叉树模型计算可转换债券初始价值',round(V2_CB,4))
step3=300
V3_CB=value_CB(S0,sigma_A,par_CB,share,Lambda_A,rate,R_A,Q2_A,tenor,step3)
print('运用',step3,'步二叉树模型计算可转换债券初始价值',round(V3_CB,4))

def Black_model(F,K,sigma,r,T,typ):
    '''定义一个运用布莱克模型计算欧式期货期权价格的函数
    F: 代表标的期货合约的当前价格。
    K: 代表期货期权的行权价格。
    sigma: 代表期货收益率的年化波动率。
    r: 代表连续复利的无风险收益率。
    T: 代表期货期权的剩余期限（年）。
    typ: 代表期货期权类型，输入typ='call'表示看涨期货期权，输入其他则表示看跌期货期权'''
    from numpy import exp,log,sqrt
    from scipy.stats import norm
    d1=(log(F/K)+sigma**2/2*T)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    if typ=='call':
        price=exp(-r*T)*(F*norm.cdf(d1)-K*norm.cdf(d2))
    else:
        price=exp(-r*T)*(K*norm.cdf(-d2)-F*norm.cdf(-d1))
    return price
# Example 14-3 欧式期货期权的定价——布莱克模型
# Step 1: 导入期货合约价格数据，计算期货合约收益率及其年化波动率
price_AU2012=pd.read_excel(r'E:\OneDrive\附件\数据\第14章\黄金期货AU2012合约结算价（2019年11月18日至2020年9月11日）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
return_AU2012=np.log(price_AU2012/price_AU2012.shift(1)) #计算期货合约每日收益率
Sigma_AU2012=np.sqrt(252)*np.std(return_AU2012) #计算期货合约收益率的年化波动率
Sigma_AU2012=float(Sigma_AU2012) #转换为浮点型数据
print('黄金期货AU2012合约收益率的年化波动率',round(Sigma_AU2012,4))
# Step 2: 运用自定义函数Black_model，分别计算“黄金2012购380”和“黄金2012沽380”期权合约的价格
import datetime as dt
t0=dt.datetime(2020,9,11)
t1=dt.datetime(2020,11,24)
tenor=(t1-t0).days/365 #期货期权的剩余期限
strike=380 #期货期权的行权价格
shibor_Sep11=0.02697 #2020年9月11日的无风险收益率
price_Sep11=420.36 #2020年9月11日的期货结算价
price_call=Black_model(price_Sep11,strike,Sigma_AU2012,shibor_Sep11,tenor,'call')
price_put=Black_model(price_Sep11,strike,Sigma_AU2012,shibor_Sep11,tenor,'put')
print('2020年9月11日黄金2012购380期权合约（看涨期货期权）的价格',round(price_call,4))
print('2020年9月11日黄金2012沽380期权合约（看跌期货期权）的价格',round(price_put,4))

def FutOption_call_Amer(F,K,sigma,r,T,N):
    '''定义运用N步二叉树模型计算美式看涨期货期权价值的函数
    F: 代表标的期货合约的当前价格。
    K: 代表期货期权的行权价格。
    sigma: 代表标的期货收益率的年化波动率。
    r: 代表连续复利的无风险收益率。
    T: 代表期货期权的期限（年）。
    N: 代表二叉树模型的步数'''
    t=T/N
    u=np.exp(sigma*np.sqrt(t)) #标的期货价格上涨时的比例
    d=1/u
    p=(1-d)/(u-d)
    call_matrix=np.zeros((N+1,N+1)) #创建N+1行、N+1列的零矩阵，用于后续存放每个节点的期货期权价值
    N_list=np.arange(N+1) #创建从0到N的自然数数列（数组格式）
    F_end=F*u**(N-N_list)*d**N_list #计算期权到期时节点标的期货价格（按照节点从上往下排序）
    call_matrix[:,-1]=np.maximum(F_end-K,0)
    i_list=list(range(N)) #创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()
    for i in i_list:
        j_list=np.arange(i+1) #创建从0到i的自然数数列（数组格式）
        Fi=F*u**(i-j_list)*d**j_list #计算在iΔt时刻各节点上的标的期货价格（按照节点从上往下排序）
        call_strike=np.maximum(Fi-K,0) #计算提前行权时的期货期权收益
        call_nostrike=np.exp(-r*t)*(p*call_matrix[:i+1,i+1]+(1-p)*call_matrix[1:i+2,i+1]) #计算不提前行权时的期货期权价值
        call_matrix[:i+1,i]=np.maximum(call_strike,call_nostrike)
    return call_matrix[0,0]
def FutOption_put_Amer(F,K,sigma,r,T,N):
    '''定义运用N步二叉树模型计算美式看跌期货期权价值的函数
    F: 代表标的期货合约的当前价格。
    K: 代表期货期权的行权价格。
    sigma: 代表标的期货收益率的年化波动率。
    r: 代表连续复利的无风险收益率。
    T: 代表期货期权的期限（年）。
    N: 代表二叉树模型的步数'''
    t=T/N
    u=np.exp(sigma*np.sqrt(t)) #标的期货价格上涨时的比例
    d=1/u
    p=(1-d)/(u-d)
    put_matrix=np.zeros((N+1,N+1)) #创建N+1行、N+1列的零矩阵，用于后续存放每个节点的期货期权价值
    N_list=np.arange(N+1) #创建从0到N的自然数数列（数组格式）
    F_end=F*u**(N-N_list)*d**N_list #计算期权到期时节点标的期货价格（按照节点从上往下排序）
    put_matrix[:,-1]=np.maximum(K-F_end,0)
    i_list=list(range(N)) #创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()
    for i in i_list:
        j_list=np.arange(i+1) #创建从0到i的自然数数列（数组格式）
        Fi=F*u**(i-j_list)*d**j_list #计算在iΔt时刻各节点上的标的期货价格（按照节点从上往下排序）
        put_strike=np.maximum(K-Fi,0) #计算提前行权时的期货期权收益
        put_nostrike=np.exp(-r*t)*(p*put_matrix[:i+1,i+1]+(1-p)*put_matrix[1:i+2,i+1]) #计算不提前行权时的期货期权价值
        put_matrix[:i+1,i]=np.maximum(put_strike,put_nostrike)
    return put_matrix[0,0]
# Example 14-4 美式期货期权的定价——二叉树模型
price_M2103=pd.read_excel(r'E:\OneDrive\附件\数据\第14章\豆粕期货2103合约结算价（2020年3月16日至11月5日）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
return_M2103=np.log(price_M2103/price_M2103.shift(1)) #计算期货合约每日收益率
Sigma_M2103=np.sqrt(252)*np.std(return_M2103)
Sigma_M2103=float(Sigma_M2103)
print('豆粕期货M2103合约收益率的年化波动率',round(Sigma_M2103,4))
T_3M=3/12
strike=3000
shibor_Nov5=0.02996
price_Nov5=3221
step=100
value_Amercall=FutOption_call_Amer(price_Nov5,strike,Sigma_M2103,shibor_Nov5,T_3M,step)
value_Amerput=FutOption_put_Amer(price_Nov5,strike,Sigma_M2103,shibor_Nov5,T_3M,step)
print('2020年11月5日豆粕2103购3000期权合约（美式看涨）的价值',round(value_Amercall,4))
print('2020年11月5日豆粕2103沽3000期权合约（美式看跌）的价值',round(value_Amerput,4))

def caplet(L,R,F,Rk,sigma,t1,t2):
    '''定义一个计算利率上限单元价值的函数
    L: 代表利率上限单元的本金，即利率上限期权的本金。
    R: 代表连续复利的无风险收益率。
    F: 代表初始0时刻观察到的从ti时刻至ti+1时刻期间的远期利率。
    Rk: 代表上限利率（行权价格）。
    sigma: 代表远期利率的年化波动率。
    t1: 代表ti时刻，以年为单位。
    t2: 代表ti+1时刻，以年为单位'''
    from numpy import exp,log,sqrt
    from scipy.stats import norm
    d1=(log(F/Rk)+sigma**2*t1/2)/(sigma*sqrt(t1))
    d2=d1-sigma*sqrt(t1)
    tau=t2-t1
    value=L*tau*exp(-R*t2)*(F*norm.cdf(d1)-Rk*norm.cdf(d2)) #利率上限单元价值
    return value
# Example 14-6 利率上限期权的价值
# Step 1: 计算相关远期利率的波动率
shibor_list=pd.read_excel(r'E:\OneDrive\附件\数据\第14章\Shibor利率（2019年1月至2020年3月20日）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
shibor_list.columns
def Rf(R1,R2,T1,T2): #6.4.1节的自定义函数
    '''定义一个计算远期利率的函数
    R1: 表示对应期限为T1的零息利率。
    R2: 表示对应期限为T2的零息利率。
    T1: 表示对应于零息利率R1的期限长度（年）。
    T2: 表示对应于零息利率R2的期限长度（年）'''
    forward_rate=(R2*T2-R1*T1)/(T2-T1)
    return forward_rate
FR1_list=Rf(shibor_list['SHIBOR(3M)'],shibor_list['SHIBOR(6M)'],3/12,6/12) #3个月后的3个月期Shibor
FR2_list=Rf(shibor_list['SHIBOR(6M)'],shibor_list['SHIBOR(9M)'],6/12,9/12) #6个月后的3个月期Shibor
FR3_list=Rf(shibor_list['SHIBOR(9M)'],shibor_list['SHIBOR(12M)'],9/12,12/12) #9个月后的3个月期Shibor
return_FR1=np.log(FR1_list/FR1_list.shift(1)) #3个月后的远期3个月期Shibor的涨跌幅（日收益率）
return_FR2=np.log(FR2_list/FR2_list.shift(1)) #6个月后的远期3个月期Shibor的涨跌幅（日收益率）
return_FR3=np.log(FR3_list/FR3_list.shift(1)) #9个月后的远期3个月期Shibor的涨跌幅（日收益率）
sigma_FR1=np.sqrt(252)*return_FR1.std() #3个月后的远期3个月期Shibor的年化波动率
sigma_FR2=np.sqrt(252)*return_FR2.std() #6个月后的远期3个月期Shibor的年化波动率
sigma_FR3=np.sqrt(252)*return_FR3.std() #9个月后的远期3个月期Shibor的年化波动率
print('3个月后的远期3个月期Shibor的年化波动率',round(sigma_FR1,6))
print('6个月后的远期3个月期Shibor的年化波动率',round(sigma_FR2,6))
print('9个月后的远期3个月期Shibor的年化波动率',round(sigma_FR3,6))
# Step 2: 计算出每一个利率上限单元的价值
FR1_Mar20=FR1_list[-1] #2020年3月20日的3个月后远期3个月期Shibor
FR2_Mar20=FR2_list[-1] #2020年3月20日的6个月后远期3个月期Shibor
FR3_Mar20=FR3_list[-1] #2020年3月20日的9个月后远期3个月期Shibor
R_6M=0.017049 #2020年3月20日6个月期无风险收益率（连续复利）
R_9M=0.018499 #2020年3月20日9个月期无风险收益率（连续复利）
R_12M=0.018682 #2020年3月20日12个月期无风险收益率（连续复利）
par=1e8 #利率上限期权的本金 
cap_rate=0.022 #上限利率
caplet1=caplet(par,R_6M,FR1_Mar20,cap_rate,sigma_FR1,3/12,6/12) #计算利率重置日2020年6月20日、收益支付日2020年9月20日的利率上限单元价值
caplet2=caplet(par,R_9M,FR2_Mar20,cap_rate,sigma_FR2,6/12,9/12) #计算利率重置日2020年9月20日、收益支付日2020年12月20日的利率上限单元价值
caplet3=caplet(par,R_12M,FR3_Mar20,cap_rate,sigma_FR3,9/12,12/12) #计算利率重置日2020年12月20日、收益支付日2021年3月20日的利率上限单元价值
print('利率重置日2020年6月20日、收益支付日2020年9月20日的利率上限单元价值',round(caplet1,2))
print('利率重置日2020年9月20日、收益支付日2020年12月20日的利率上限单元价值',round(caplet2,2))
print('利率重置日2020年12月20日、收益支付日2021年3月20日的利率上限单元价值',round(caplet3,2))
# Step 3: 将3个利率上限单元的价值相加即可求出利率上限期权的价值
cap=caplet1+caplet2+caplet3
print('2020年3月20日利率上限期权的价值',round(cap,2))

def floorlet(L,R,F,Rk,sigma,t1,t2):
    '''定义一个计算利率下限单元价值的函数
    L: 代表利率下限单元的本金，即利率下限期权的本金。
    R: 代表连续复利的无风险收益率。
    F: 代表0时刻观察到的从ti时刻到ti+1时刻期间的远期利率。
    Rk: 代表下限利率（行权价格）。
    sigma: 代表远期利率的年化波动率。
    t1: 代表ti时刻，以年为单位。
    t2: 代表ti+1时刻，以年为单位'''
    from numpy import exp,log,sqrt
    from scipy.stats import norm
    d1=(log(F/Rk)+sigma**2*t1/2)/(sigma*sqrt(t1))
    d2=d1-sigma*sqrt(t1)
    tau=t2-t1
    value=L*tau*exp(-R*t2)*(Rk*norm.cdf(-d2)-F*norm.cdf(-d1))
    return value
# Example 14-7 利率下限期权的价值
floor_rate=0.025 #下限利率
floorlet1=floorlet(par,R_6M,FR1_Mar20,floor_rate,sigma_FR1,3/12,6/12)
floorlet2=floorlet(par,R_9M,FR2_Mar20,floor_rate,sigma_FR2,6/12,9/12)
floorlet3=floorlet(par,R_12M,FR3_Mar20,floor_rate,sigma_FR3,9/12,12/12)
print('利率重置日2020年6月20日、收益支付日2020年9月20日的利率下限单元价值',round(floorlet1,2))
print('利率重置日2020年9月20日、收益支付日2020年12月20日的利率下限单元价值',round(floorlet2,2))
print('利率重置日2020年12月20日、收益支付日2021年3月20日的利率下限单元价值',round(floorlet3,2))
floor=floorlet1+floorlet2+floorlet3
print('2020年3月20日利率下限期权的价值',round(floor,2))

# Example 14-8 利率双限期权
par_new=1e9
cap_rate_new=0.029
floor_rate_new=0.023
caplet1_new=caplet(par_new,R_6M,FR1_Mar20,cap_rate_new,sigma_FR1,3/12,6/12)
caplet2_new=caplet(par_new,R_9M,FR2_Mar20,cap_rate_new,sigma_FR2,6/12,9/12)
caplet3_new=caplet(par_new,R_12M,FR3_Mar20,cap_rate_new,sigma_FR3,9/12,1)
floorlet1_new=floorlet(par_new,R_6M,FR1_Mar20,floor_rate_new,sigma_FR1,3/12,6/12)
floorlet2_new=floorlet(par_new,R_9M,FR2_Mar20,floor_rate_new,sigma_FR2,6/12,9/12)
floorlet3_new=floorlet(par_new,R_12M,FR3_Mar20,floor_rate_new,sigma_FR3,9/12,12/12)
cap_new=caplet1_new+caplet2_new+caplet3_new
print('2020年3月20日利率双限期权中的利率上限期权价值',round(cap_new,2))
floor_new=floorlet1_new+floorlet2_new+floorlet3_new
print('2020年3月20日利率双限期权中的利率下限期权价值',round(floor_new,2))
collar_long=cap_new-floor_new
print('2020年3月20日利率双限期权多头头寸的价值',round(collar_long,2))

def swaption(L,Sf,Sk,m,sigma,t,n,R_list,direction):
    '''定义一个计算利率互换期权价值的函数
    L: 代表利率互换期权的本金。
    Sf: 代表远期互换利率。
    Sk: 代表利率互换合约的固定利率。
    m: 代表每年利率支付频次（复利频次）。
    sigma: 代表远期互换利率的年化波动率。
    t: 代表期权的期限（年）。
    n: 代表对应利率互换合约的期限（年）。
    R_list: 代表期权定价日距离利率互换每期利息支付日的期限Ti对应的无风险收益率（连续复利），以数组格式输入。
    direction: 代表期权多头是否在利率互换中支付固定利息，输入direction='pay'代表支付固定利息，输入其他则代表收取固定利息'''
    from numpy import arange,exp,log,sqrt
    from scipy.stats import norm
    d1=(log(Sf/Sk)+sigma**2*t/2)/(sigma*sqrt(t))
    d2=d1-sigma*sqrt(t)
    T_list=t+arange(1,m*n+1)/m #创建期权定价日距离利率互换每笔利息支付日的期限Ti的数组
    if direction=='pay':
        value=np.sum(exp(-R_list*T_list)*L/m*(Sf*norm.cdf(d1)-Sk*norm.cdf(d2)))
    else:
        value=np.sum(exp(-R_list*T_list)*L/m*(Sk*norm.cdf(-d2)-Sf*norm.cdf(-d1)))
    return value
def forward_swaprate(S_list,t,n,m):
    '''定义一个计算远期互换利率的函数
    S_list: 代表在利率互换期权初始日观察到不同期限的互换利率，以数组格式输入。
    t: 代表期权的期限（年）。
    n: 代表利率互换合约的期限（年）。
    m: 代表每年利率支付频次（复利频次）'''
    t_list=m*t+np.arange(1,m*n+1) #考虑复利频次的期限数组
    A=(1+S_list[0]/m)**(-m*t)-(1+S_list[-1]/m)**(-m*(t+n)) #式(14-35)的分子
    B=(1/m)*np.sum((1+S_list[1:]/m)**(-t_list)) #式(14-35)的分母
    return A/B
# Example 14-9 利率互换期权
swaprate_list=pd.read_excel(r'E:\OneDrive\附件\数据\第14章\Shibor互换利率数据（2019年1月至2020年9月1日）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
swaprate_list.columns #查看列名
swaprate_list.index #查看索引名
T_swaption=0.5 #利率互换期权期限
T_swap=0.5 #利率互换合约期限
M=4 #每年复利频次（按季复利）
forward_list=np.zeros(len(swaprate_list.index)) #创建存放远期互换利率的初始数组
for i in range(len(swaprate_list.index)):
    forward_list[i]=forward_swaprate(swaprate_list.iloc[i],T_swaption,T_swap,M)
forward_list=pd.DataFrame(data=forward_list,index=swaprate_list.index,columns=['远期互换利率']) #转换为数据框
forward_list.plot(figsize=(9,6),grid=True)
plt.ylabel(u'利率',fontsize=11)

return_forward=np.log(forward_list/forward_list.shift(1)) #计算2019年1月至2020年9月1日期间远期互换利率的每日百分比变化
sigma_forward=np.sqrt(252)*return_forward.std() #计算远期互换利率的年化波动率
sigma_forward=float(sigma_forward) #(Series)转换为浮点型数据
print('计算的到远期互换利率的年化波动率',round(sigma_forward,6))
forward_Sep1=float(forward_list.iloc[-1]) #2020年9月1日的远期互换利率
print('2020年9月1日的远期互换利率',round(forward_Sep1,6))
par=1e8
rate_fixed=0.029
R_norisk=np.array(swaprate_list.iloc[-1]) #以数组格式存放2020年9月1日无风险收益率(6M、9M、12M)
def Rc(Rm,m): #6.3.2节的自定义函数
    '''定义一个已知复利频次和对应的复利利率，计算连续复利利率的函数
    Rm: 代表复利频次为m的复利利率。
    m: 代表复利频次'''
    r=m*np.log(1+Rm/m) #计算等价的连续复利利率
    return r
Rc_norisk=Rc(R_norisk,M) #按季复利的无风险收益率转换为连续复利的无风险收益率
Rc_9M_12M=Rc_norisk[1:] #取9个月期和1年期的无风险收益率
value=swaption(par,forward_Sep1,rate_fixed,M,sigma_forward,T_swaption,T_swap,Rc_9M_12M,'receive') #计算2020年9月1日利率互换期权的价值
print('2020年9月1日利率互换期权的价值',round(value,2))