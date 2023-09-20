# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:15:22 2023

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

def delta_EurOpt(S,K,sigma,r,T,optype,positype):
    '''定义一个计算欧式期权Delta的函数
    S: 代表期权基础资产的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    optype: 代表期权的类型，输入optype='call'表示看涨期权，输入其他则表示看跌期权。
    positype: 代表期权头寸的方向，输入positype='long'表示期权多头，输入其他则表示期权空头'''
    from scipy.stats import norm
    from numpy import log,sqrt
    d1=(log(S/K)+(r+sigma**2/2)*T)/(sigma*sqrt(T)) #d1表达式
    if optype=='call':
        if positype=='long':
            delta=norm.cdf(d1)
        else:
            delta=-norm.cdf(d1)
    else:
        if positype=='long':
            delta=norm.cdf(d1)-1
        else:
            delta=1-norm.cdf(d1)
    return delta

# Example 12-1
S_ABC=3.27 #农业银行股价
K_ABC=3.6 #期权的行权价格
sigma_ABC=0.19 #农业银行股票年化波动率
shibor_6M=0.02377 #6个月期Shibor（无风险收益率）
T_ABC=0.5 #期权期限
delta_EurOpt1=delta_EurOpt(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,'call','long') #计算欧式看涨期权多头的delta
delta_EurOpt2=delta_EurOpt(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,'call','short') #计算欧式看涨期权空头的delta
delta_EurOpt3=delta_EurOpt(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,'put','long') #计算欧式看跌期权多头的delta
delta_EurOpt4=delta_EurOpt(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,'put','short') #计算欧式看跌期权空头的delta
print('农业银行A股欧式看涨期权多头的Delta',round(delta_EurOpt1,4))
print('农业银行A股欧式看涨期权空头的Delta',round(delta_EurOpt2,4))
print('农业银行A股欧式看跌期权多头的Delta',round(delta_EurOpt3,4))
print('农业银行A股欧式看跌期权空头的Delta',round(delta_EurOpt4,4))

# Example 12-2 利用Delta的近似计算
def option_BSM(S,K,sigma,r,T,opt):
    '''定义一个运用布莱克-斯科尔斯-默顿模型计算欧式期权价格的函数
    S: 代表期权基础资产的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    opt: 代表期权类型，输入opt='call'表示看涨期权，输入其他则表示看跌期权'''
    from numpy import log,exp,sqrt
    from scipy.stats import norm
    d1=(log(S/K)+(r+sigma**2/2)*T)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    if opt=='call':
        value=S*norm.cdf(d1)-K*exp(-r*T)*norm.cdf(d2)
    else:
        value=K*exp(r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return value
S_list1=np.linspace(2.5,4.5,200) #创建农业银行股价的等差数列
value_list=option_BSM(S_list1,K_ABC,sigma_ABC,shibor_6M,T_ABC,'call') #不同基础资产价格对应的期权价格（运用BSM模型）
value_one=option_BSM(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,'call') #农行股价等于3.27元/股（2020年7月16日收盘价）对应的期权价格
value_approx1=value_one+delta_EurOpt1*(S_list1-S_ABC) #用Delta计算不同农行股价对应的近似期权价格
plt.figure(figsize=(9,6))
plt.plot(S_list1,value_list,'b-',label=u'运用BSM模型计算得到的看涨期权价格',lw=2.5)
plt.plot(S_list1,value_approx1,'r-',label=u'运用Delta计算得到的看涨期权近似价格',lw=2.5)
plt.xlabel(u'股票价格',fontsize=13)
plt.ylabel(u'期权价格',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'运用BSM模型计算得到的期权价格与运用Delta计算得到的近似期权价格的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 12-3 基础资产价格与期权Delta的关系，期权类型作对比
S_list2=np.linspace(1.0,6.0,200)
Delta_EurCall=delta_EurOpt(S_list2,K_ABC,sigma_ABC,shibor_6M,T_ABC,'call','long')
Delta_EurPut=delta_EurOpt(S_list2,K_ABC,sigma_ABC,shibor_6M,T_ABC,'put','long')
plt.figure(figsize=(9,6))
plt.plot(S_list2,Delta_EurCall,'b-',label=u'欧式看涨期权多头',lw=2.5)
plt.plot(S_list2,Delta_EurPut,'r-',label=u'欧式看跌期权多头',lw=2.5)
plt.xlabel(u'股票价格',fontsize=13)
plt.ylabel(u'Delta',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'股票价格与欧式期权多头Delta',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 12-4 期权期限与期权Delta的关系，实虚值（行权价格）做对比
S1=4.0 #实值看涨期权对应的股价
S2=3.6 #平价看涨期权对应的股价
S3=3.0 #虚值看涨期权对应的股价
T_list=np.linspace(0.1,5.0,200)
Delta_list1=delta_EurOpt(S1,K_ABC,sigma_ABC,shibor_6M,T_list,'call','long')
Delta_list2=delta_EurOpt(S2,K_ABC,sigma_ABC,shibor_6M,T_list,'call','long')
Delta_list3=delta_EurOpt(S3,K_ABC,sigma_ABC,shibor_6M,T_list,'call','long')
plt.figure(figsize=(9,6))
plt.plot(T_list,Delta_list1,'b-',label=u'实值看涨期权多头',lw=2.5)
plt.plot(T_list,Delta_list2,'r-',label=u'平价看涨期权多头',lw=2.5)
plt.plot(T_list,Delta_list3,'g-',label=u'虚值看涨期权多头',lw=2.5)
plt.xlabel(u'期权期限（年）',fontsize=13)
plt.ylabel(u'Delta',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'期权期限与欧式看涨期权多头Delta的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 12-5 基于Delta的对冲
# Step 1: 采用静态对冲策略，在整个对冲期间用于对冲农业银行股票数量保持不变，计算2020年7月16日需买入农行A股股票数量及在2020年8月31日该策略的对冲效果
N_put=1e6 #持有看跌期权多头头寸
N_ABC=np.abs(delta_EurOpt3*N_put) #用于对冲的农业银行A股股票数量（变量delta_EurOpt3再例12-1中已设定）
N_ABC=int(N_ABC) #转换为整型
print('2020年7月16日买入基于期权Delta对冲的农业银行A股数量',N_ABC)
import datetime as dt
T0=dt.datetime(2020,7,16)
T1=dt.datetime(2020,8,31)
T2=dt.datetime(2021,1,16)
T_new=(T2-T1).days/365 #2020年8月31日至期权到期日的剩余期限（年）
S_Aug31=3.21
shibor_Aug31=0.02636
put_Jul16=option_BSM(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,'put') #期权初始日看跌期权价格
put_Aug31=option_BSM(S_Aug31,K_ABC,sigma_ABC,shibor_Aug31,T_new,'put') #2020年8月31日看跌期权价格
print('2020年7月16日农业银行A股欧式看跌期权价格',round(put_Jul16,4))
print('2020年8月31日农业银行A股欧式看跌期权价格',round(put_Aug31,4))
port_chagvalue=N_ABC*(S_Aug31-S_ABC)*N_put*(put_Aug31-put_Jul16) #静态对冲策略下2020年8月31日投资组合的累积盈亏
print('静态对冲策略下2020年8月31日投资组合的累积盈亏',round(port_chagvalue,2))
# Step 2: 计算在2020年8月31日看跌期权的Delta及保持该交易日期权Delta中性而需要针对基础资产新增交易情况
delta_Aug31=delta_EurOpt(S_Aug31,K_ABC,sigma_ABC,shibor_Aug31,T_new,'put','long')
print('2020年8月31日农业银行A股欧式看跌期权Delta',round(delta_Aug31,4))
N_ABC_new=np.abs(delta_Aug31*N_put) #2020年8月31日保持Delta中性而用于对冲的农业银行A股股票数量
N_ABC_new=int(N_ABC_new)
print('2020年8月31日保持Delta中性而用于对冲的农业银行A股股票数量',N_ABC_new)
N_ABC_change=N_ABC_new-N_ABC
print('2020年8月31日保持Delta中性而发生的股票数量变化',N_ABC_change)

def delta_AmerCall(S,K,sigma,r,T,N,positype):
    '''定义一个运用N步二叉树模型计算美式看涨期权Delta的函数
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数。
    positype: 代表期权头寸方向，输入positype='long'表示期权多头，输入其他则表示期权空头'''
    t=T/N #计算每一步步长期限（年）
    u=np.exp(sigma*np.sqrt(t)) #计算基础资产价格上涨时的比例
    d=1/u #计算基础资产价格下跌时的比例
    p=(np.exp(r*t)-d)/(u-d) #计算基础资产价格上涨的概率
    call_matrix=np.zeros((N+1,N+1)) #创建N+1行、N+1列的零矩阵，用于后续存放每个节点的期权价值。
    N_list=np.arange(0,N+1) #创建从0到N的自然数数列（数组格式）
    S_end=S*u**(N-N_list)*d**(N_list) #计算期权到期时节点的基础资产价格（按照节点从上往下排序）
    call_matrix[:,-1]=np.maximum(S_end-K,0) #计算期权到期时节点的看涨期权价值（按照节点从上往下顺序）
    i_list=list(range(0,N)) #创建从0到N-1的自然数数列（列表格式）
    i_list.reverse() #将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list=np.arange(i+1) #创建从0到i的自然数数列（数组格式）
        Si=S*u**(i-j_list)*d**(j_list) #计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        call_strike=np.maximum(Si-K,0) #计算提前行权时的期权收益
        call_nostrike=np.exp(-r*t)*(p*call_matrix[:i+1,i+1]+(1-p)*call_matrix[1:i+2,i+1]) #计算不提前行权时的期权价值
        call_matrix[:i+1,i]=np.maximum(call_strike,call_nostrike) #取提前行权时的期权收益与不提前行权时的期权价值中的最大值
    Delta=(call_matrix[0,1]-call_matrix[1,1])/(S*u-S*d) #计算期权Delta=(Π1,1-Π1,0)/(S0u-S0d)
    if positype=='long':
        result=Delta
    else:
        result=-Delta
    return result

def delta_AmerPut(S,K,sigma,r,T,N,positype):
    '''定义一个运用N步二叉树模型计算美式看跌期权Delta的函数
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数。
    positype: 代表期权头寸方向，输入positype='long'表示期权多头，输入其他则表示期权空头'''
    t=T/N #计算每一步步长期限（年）
    u=np.exp(sigma*np.sqrt(t)) #计算基础资产价格上涨时的比例
    d=1/u #计算基础资产价格下跌时的比例
    p=(np.exp(r*t)-d)/(u-d) #计算基础资产价格上涨的概率
    put_matrix=np.zeros((N+1,N+1)) #创建N+1行、N+1列的零矩阵，用于后续存放每个节点的期权价值。
    N_list=np.arange(0,N+1) #创建从0到N的自然数数列（数组格式）
    S_end=S*u**(N-N_list)*d**(N_list) #计算期权到期时节点的基础资产价格（按照节点从上往下排序）
    put_matrix[:,-1]=np.maximum(K-S_end,0) #计算期权到期时节点的看涨期权价值（按照节点从上往下顺序）
    i_list=list(range(0,N)) #创建从0到N-1的自然数数列（列表格式）
    i_list.reverse() #将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list=np.arange(i+1) #创建从0到i的自然数数列（数组格式）
        Si=S*u**(i-j_list)*d**(j_list) #计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        put_strike=np.maximum(K-Si,0) #计算提前行权时的期权收益
        put_nostrike=np.exp(-r*t)*(p*put_matrix[:i+1,i+1]+(1-p)*put_matrix[1:i+2,i+1]) #计算不提前行权时的期权价值
        put_matrix[:i+1,i]=np.maximum(put_strike,put_nostrike) #取提前行权时的期权收益与不提前行权时的期权价值中的最大值
    Delta=(put_matrix[0,1]-put_matrix[1,1])/(S*u-S*d) #计算期权Delta=(Π1,1-Π1,0)/(S0u-S0d)
    if positype=='long':
        result=Delta
    else:
        result=-Delta
    return result

# Example 12-6 美式期权的Delta
step=100
delta_AmerOpt1=delta_AmerCall(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step,'long') #计算美式看涨期权多头的Delta
delta_AmerOpt2=delta_AmerCall(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step,'short') #计算美式看涨期权空头的Delta
delta_AmerOpt3=delta_AmerPut(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step,'long') #计算美式看跌期权多头的Delta
delta_AmerOpt4=delta_AmerPut(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step,'short') #计算美式看跌期权空头的Delta
print('农业银行A股美式看涨期权多头的Delta',round(delta_AmerOpt1,4))
print('农业银行A股美式看涨期权空头的Delta',round(delta_AmerOpt2,4))
print('农业银行A股美式看跌期权多头的Delta',round(delta_AmerOpt3,4))
print('农业银行A股美式看跌期权空头的Delta',round(delta_AmerOpt4,4))

def gamma_EurOpt(S,K,sigma,r,T):
    '''定义一个计算欧式期权Gamma的函数
    S: 代表期权基础资产的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的剩余期限（年）'''
    from numpy import exp,log,pi,sqrt
    d1=(log(S/K)+(r+sigma**2/2)*T)/(sigma*sqrt(T))
    gamma=exp(-d1**2/2)/(S*sigma*sqrt(2*pi*T))
    return gamma
# Example 12-7 期权的Gamma
gamma_Eur=gamma_EurOpt(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC)
print('农业银行A股欧式期权的Gamma',round(gamma_Eur,4))

# Example 12-7 利用Delta和Gamma的近似计算
value_approx2=value_one+delta_EurOpt1*(S_list1-S_ABC)+0.5*gamma_Eur*(S_list1-S_ABC)**2
plt.figure(figsize=(9,6))
plt.plot(S_list1,value_list,'b-',label=u'运用BSM模型计算的看涨期权价格',lw=2.5)
plt.plot(S_list1,value_approx1,'r-',label=u'仅用Delta计算的看涨期权近似价格',lw=2.5)
plt.plot(S_list1,value_approx2,'m-',label=u'用Delta和Gamma计算的看涨期权近似价格',lw=2.5)
plt.plot(S_ABC,value_one,'o',label=u'股价等于3.27元/股对应的期权价格',lw=2.5)
plt.xlabel(u'股票价格',fontsize=13)
plt.ylabel(u'期权价格',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'运用BSM模型、仅用Delta以及用Delta和Gamma计算的期权价格',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 12-9 基础资产价格与期权Gamma的关系
gamma_list=gamma_EurOpt(S_list2,K_ABC,sigma_ABC,shibor_6M,T_ABC)
plt.figure(figsize=(9,6))
plt.plot(S_list2,gamma_list,'b-',lw=2.5)
plt.xlabel(u'股票价格',fontsize=13)
plt.ylabel('Gamma',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'股票价格与期权Gamma的关系图',fontsize=13)
plt.grid()
plt.show()

# Example 12-10 期权期限与期权Gamma的关系
gamma_list1=gamma_EurOpt(S1,K_ABC,sigma_ABC,shibor_6M,T_list) #实值看涨期权的Gamma
gamma_list2=gamma_EurOpt(S2,K_ABC,sigma_ABC,shibor_6M,T_list) #平价看涨期权的Gamma
gamma_list3=gamma_EurOpt(S3,K_ABC,sigma_ABC,shibor_6M,T_list) #虚值看涨期权的Gamma
plt.figure(figsize=(9,6))
plt.plot(T_list,gamma_list1,'b-',label=u'实值看涨期权',lw=2.5)
plt.plot(T_list,gamma_list2,'r-',label=u'平价看涨期权',lw=2.5)
plt.plot(T_list,gamma_list3,'g-',label=u'虚值看涨期权',lw=2.5)
plt.xlabel(u'期权期限',fontsize=13)
plt.ylabel('Gamma',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'期权期限与期权Gamma的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

def gamma_AmerCall(S,K,sigma,r,T,N):
    '''定义一个运用N步二叉树模型计算美式看涨期权Gamma的函数
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数'''
    t=T/N
    u=np.exp(sigma*np.sqrt(t))
    d=1/u
    p=(np.exp(r*t)-d)/(u-d)
    call_matrix=np.zeros((N+1,N+1)) #创建N+1行、N+1列矩阵且元素均为0，用于后续存放每个节点的期权价值
    N_list=np.arange(0,N+1)
    S_end=S*u**(N-N_list)*d**N_list #计算期权到期时节点的基础资产价格（按照节点从上往下排序）
    call_matrix[:,-1]=np.maximum(S_end-K,0) #计算期权到期时节点的看涨期权价值（按照节点从上往下排序）
    i_list=list(range(0,N)) #创建从0到N-1的自然数数列（列表格式）
    i_list.reverse() #将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list=np.arange(i+1)
        Si=S*u**(i-j_list)*d**j_list #计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        call_strike=np.maximum(Si-K,0)
        call_nostrike=np.exp(-r*t)*(p*call_matrix[:i+1,i+1]+(1-p)*call_matrix[1:i+2,i+1])
        call_matrix[:i+1,i]=np.maximum(call_strike,call_nostrike)
    Delta1=(call_matrix[0,2]-call_matrix[1,2])/(S*u**2-S) #计算一个Delta
    Delta2=(call_matrix[1,2]-call_matrix[2,2])/(S-S*d**2) #计算另一个Delta
    Gamma=2*(Delta1-Delta2)/(S*u**2-S*d**2) #计算美式看涨期权Gamma
    return Gamma

def gamma_AmerPut(S,K,sigma,r,T,N):
    '''定义一个运用N步二叉树模型计算美式看跌期权Gamma的函数
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数'''
    t=T/N
    u=np.exp(sigma*np.sqrt(t))
    d=1/u
    p=(np.exp(r*t)-d)/(u-d)
    put_matrix=np.zeros((N+1,N+1))
    N_list=np.arange(0,N+1)
    S_end=S*u**(N-N_list)*d**N_list
    put_matrix[:,-1]=np.maximum(K-S_end,0)
    i_list=list(range(0,N))
    i_list.reverse()
    for i in i_list:
        j_list=np.arange(i+1)
        Si=S*u**(i-j_list)*d**j_list
        put_strike=np.maximum(K-Si,0)
        put_nostrike=np.exp(-r*t)*(p*put_matrix[:i+1,i+1]+(1-p)*put_matrix[1:i+2,i+1])
        put_matrix[:i+1,i]=np.maximum(put_strike,put_nostrike)
    Delta1=(put_matrix[0,2]-put_matrix[1,2])/(S*u**2-S)
    Delta2=(put_matrix[1,2]-put_matrix[2,2])/(S-S*d**2)
    Gamma=2*(Delta1-Delta2)/(S*u**2-S*d**2)
    return Gamma
# Example 12-11 美式期权的Gamma
gamma_AmerOpt1=gamma_AmerCall(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step)
gamma_AmerOpt2=gamma_AmerPut(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step)
print('农业银行A股美式看涨期权的Gamma',round(gamma_AmerOpt1,4))
print('农业银行A股美式看跌期权的Gamma',round(gamma_AmerOpt2,4))

def theta_EurOpt(S,K,sigma,r,T,optype):
    '''定义一个计算欧式期权Theta的函数
    S: 代表基础资产的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的剩余期限（年）。
    optype: 代表期权的类型，输入optype='call'表示看涨期权，输入其他则表示看跌期权'''
    from numpy import exp,log,pi,sqrt
    from scipy.stats import norm
    d1=(log(S/K)+(r+sigma**2/2)*T)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    theta_call=-S*sigma*exp(-d1**2/2)/(2*sqrt(2*pi*T))-r*K*exp(-r*T)*norm.cdf(d2)
    theta_put=theta_call+r*K*np.exp(-r*T)
    if optype=='call':
        return theta_call
    else:
        return theta_put

# Example 12-12 欧式期权的Theta
day1=365
day2=252
theta_EurCall=theta_EurOpt(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,'call')
theta_EurPut=theta_EurOpt(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,'put')
print('农业银行A股欧式看涨期权Theta',round(theta_EurCall,6))
print('农业银行A股欧式看涨期权每日历天Theta',round(theta_EurCall/day1,6))
print('农业银行A股欧式看涨期权每交易日Theta',round(theta_EurCall/day2,6))
print('农业银行A股欧式看跌期权Theta',round(theta_EurPut,6))
print('农业银行A股欧式看跌期权每日历天Theta',round(theta_EurPut/day1,6))
print('农业银行A股欧式看跌期权每交易日Theta',round(theta_EurPut/day2,6))

# Example 12-13 基础资产价格与期权Theta的关系
theta_EurCall_list=theta_EurOpt(S_list2,K_ABC,sigma_ABC,shibor_6M,T_ABC,'call')
theta_EurPut_list=theta_EurOpt(S_list2,K_ABC,sigma_ABC,shibor_6M,T_ABC,'put')
plt.figure(figsize=(9,6))
plt.plot(S_list2,theta_EurCall_list,'b-',label=u'欧式看涨期权',lw=2.5)
plt.plot(S_list2,theta_EurPut_list,'r-',label=u'欧式看跌期权',lw=2.5)
plt.xlabel(u'股票价格',fontsize=13)
plt.ylabel('Theta',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'股票价格与期权Theta的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 12-14 期权期限与期权Theta的关系
theta_list1=theta_EurOpt(S1,K_ABC,sigma_ABC,shibor_6M,T_list,'call') #实值看涨期权的Theta
theta_list2=theta_EurOpt(S2,K_ABC,sigma_ABC,shibor_6M,T_list,'call') #平价看涨期权的Theta
theta_list3=theta_EurOpt(S3,K_ABC,sigma_ABC,shibor_6M,T_list,'call') #虚值看涨期权的Theta
plt.figure(figsize=(9,6))
plt.plot(T_list,theta_list1,'b-',label=u'实值看涨期权',lw=2.5)
plt.plot(T_list,theta_list2,'r-',label=u'平价看涨期权',lw=2.5)
plt.plot(T_list,theta_list3,'g-',label=u'虚值看涨期权',lw=2.5)
plt.xlabel(u'期权期限',fontsize=13)
plt.ylabel('Theta',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'期权期限与期权Theta的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

def theta_AmerCall(S,K,sigma,r,T,N):
    '''定义一个运用N步二叉树模型计算美式看涨期权Theta的函数
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数'''
    t=T/N
    u=np.exp(sigma*np.sqrt(t)) #计算基础资产价格上涨时的比例
    d=1/u #计算基础资产价格下跌时的比例
    p=(np.exp(r*t)-d)/(u-d) #计算基础资产价格上涨的概率
    call_matrix=np.zeros((N+1,N+1))
    N_list=np.arange(0,N+1) #创建从0到N的自然数数列（数组格式）
    S_end=S*u**(N-N_list)*d**N_list #计算期权到期时节点的基础资产价格（按照节点从上往下排序）
    call_matrix[:,-1]=np.maximum(S_end-K,0)
    i_list=list(range(0,N)) #创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()
    for i in i_list:
        j_list=np.arange(i+1)
        Si=S*u**(i-j_list)*d**j_list #计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        call_strike=np.maximum(Si-K,0)
        call_nostrike=np.exp(-r*t)*(p*call_matrix[:i+1,i+1]+(1-p)*call_matrix[1:i+2,i+1])
        call_matrix[:i+1,i]=np.maximum(call_strike,call_nostrike)
    Theta=(call_matrix[1,2]-call_matrix[0,0])/(2*t)
    return Theta
def theta_AmerPut(S,K,sigma,r,T,N):
    '''定义一个运用N步二叉树模型计算美式看跌期权Theta的函数
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数'''
    t=T/N
    u=np.exp(sigma*np.sqrt(t))
    d=1/u
    p=(np.exp(r*t)-d)/(u-d)
    put_matrix=np.zeros((N+1,N+1))
    N_list=np.arange(0,N+1)
    S_end=S*u**(N-N_list)*d**N_list
    put_matrix[:,-1]=np.maximum(K-S_end,0)
    i_list=list(range(0,N))
    i_list.reverse()
    for i in i_list:
        j_list=np.arange(i+1)
        Si=S*u**(i-j_list)*d**j_list
        put_strike=np.maximum(K-Si,0)
        put_nostrike=np.exp(-r*t)*(p*put_matrix[:i+1,i+1]+(1-p)*put_matrix[1:i+2,i+1])
        put_matrix[:i+1,i]=np.maximum(put_strike,put_nostrike)
    Theta=(put_matrix[1,2]-put_matrix[0,0])/(2*t)
    return Theta
# Example 12-15 美式期权的Theta
theta_AmerOpt1=theta_AmerCall(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step)
theta_AmerOpt2=theta_AmerPut(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step)
print('农业银行A股美式看涨期权的Theta',round(theta_AmerOpt1,4))
print('农业银行A股美式看跌期权的Theta',round(theta_AmerOpt2,4))

def vega_EurOpt(S,K,sigma,r,T):
    '''定义一个计算欧式期权Vega的函数
    S: 代表基础资产的价格。
    K: 代表期权的行权价格。
    sigama: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的剩余期限（年）'''
    from numpy import exp,log,pi,sqrt
    d1=(log(S/K)+(r+sigma**2/2)*T)/(sigma*sqrt(T))
    vega=S*sqrt(T)*exp(-d1**2/2)/sqrt(2*pi)
    return vega
# Example 12-16 欧式期权的Vega
vega_Eur=vega_EurOpt(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC)
print('农业银行A股欧式期权的Vega',round(vega_Eur,4))
sigma_chg=0.01
value_chg=vega_Eur*sigma_chg
print('波动率增加1%导致期权价格变动额',round(value_chg,4))

# Example 12-17 基础资产价格与期权Vega的变化
vega_list=vega_EurOpt(S_list2,K_ABC,sigma_ABC,shibor_6M,T_ABC)
plt.figure(figsize=(9,6))
plt.plot(S_list2,vega_list,'b-',lw=2.5)
plt.xlabel(u'股票价格',fontsize=13)
plt.ylabel('Vega',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'基础资产（股票）价格与期权Vega的关系图',fontsize=13)
plt.grid()
plt.show()

# Example 12-18 期权期限与期权Vega的关系
vega_list1=vega_EurOpt(S1,K_ABC,sigma_ABC,shibor_6M,T_list)
vega_list2=vega_EurOpt(S2,K_ABC,sigma_ABC,shibor_6M,T_list)
vega_list3=vega_EurOpt(S3,K_ABC,sigma_ABC,shibor_6M,T_list)
plt.figure(figsize=(9,6))
plt.plot(T_list,vega_list1,'b-',label=u'实值看涨期权',lw=2.5)
plt.plot(T_list,vega_list2,'r-',label=u'平价看涨期权',lw=2.5)
plt.plot(T_list,vega_list3,'g-',label=u'虚值看涨期权',lw=2.5)
plt.xlabel(u'期权期限',fontsize=13)
plt.ylabel('Vega',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'期权期限与期权Vega的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

def vega_AmerCall(S,K,sigma,r,T,N):
    '''定义一个运用N步二叉树模型计算美式看涨期权Vega的函数，
       并且假定基础资产收益率的波动率是增加0.0001
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数'''
    def American_call(S,K,sigma,r,T,N):
        t=T/N
        u=np.exp(sigma*np.sqrt(t))
        d=1/u
        p=(np.exp(r*t)-d)/(u-d)
        call_matrix=np.zeros((N+1,N+1))
        N_list=np.arange(0,N+1)
        S_end=S*u**(N-N_list)*d**N_list
        call_matrix[:,-1]=np.maximum(S_end-K,0)
        i_list=list(range(0,N))
        i_list.reverse()
        for i in i_list:
            j_list=np.arange(0,i+1)
            Si=S*u**(i-j_list)*d**j_list
            call_strike=np.maximum(Si-K,0)
            call_nostrike=np.exp(-r*t)*(p*call_matrix[:i+1,i+1]+(1-p)*call_matrix[1:i+2,i+1])
            call_matrix[:i+1,i]=np.maximum(call_strike,call_nostrike)
        return call_matrix[0,0]
    Value1=American_call(S,K,sigma,r,T,N) #原二叉树模型计算的期权价值
    Value2=American_call(S,K,sigma+0.0001,r,T,N) #新二叉树模型计算的期权价值
    return (Value2-Value1)/0.0001

def vega_AmerPut(S,K,sigma,r,T,N):
    '''定义一个运用N步二叉树模型计算美式看跌期权Vega的函数，
       依然假定基础资产收益率的波动率是增加0.0001
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数'''
    def American_put(S,K,sigma,r,T,N):
        t=T/N
        u=np.exp(sigma*np.sqrt(t))
        d=1/u
        p=(np.exp(r*t)-d)/(u-d)
        put_matrix=np.zeros((N+1,N+1))
        N_list=np.arange(N+1)
        S_end=S*u**(N-N_list)*d**(N_list)
        put_matrix[:,-1]=np.maximum(K-S_end,0)
        i_list=list(range(N))
        i_list.reverse()
        for i in i_list:
            j_list=np.arange(i+1)
            Si=S*u**(i-j_list)*d**j_list
            put_strike=np.maximum(K-Si,0)
            put_nostrike=np.exp(-r*t)*(p*put_matrix[:i+1,i+1]+(1-p)*put_matrix[1:i+2,i+1])
            put_matrix[:i+1,i]=np.maximum(put_strike,put_nostrike)
        return put_matrix[0,0]
    Value1=American_put(S,K,sigma,r,T,N)
    Value2=American_put(S,K,sigma+0.0001,r,T,N)
    return (Value2-Value1)/0.0001

# Example 12-19 美式期权的Vega
vega_AmerOpt1=vega_AmerCall(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step)
vega_AmerOpt2=vega_AmerPut(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step)
print('农业银行A股美式看涨期权的Vega',round(vega_AmerOpt1,4))
print('农业银行A股美式看跌期权的Vega',round(vega_AmerOpt2,4))

def rho_EurOpt(S,K,sigma,r,T,optype):
    '''定义一个计算欧式期权Rho的函数
    S: 代表基础资产的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的剩余期限（年）。
    optype: 代表期权的类型，输入optype='call'表示看涨期权，输入其他则表示看跌期权'''
    from numpy import exp,log,sqrt
    from scipy.stats import norm
    d2=(log(S/K)+(r-sigma**2/2)*T)/(sigma*sqrt(T))
    if optype=='call':
        rho=K*T*exp(-r*T)*norm.cdf(d2)
    else:
        rho=-K*T*exp(-r*T)*norm.cdf(-d2)
    return rho

# Example 12-20 欧式期权的Rho
rho_EurCall=rho_EurOpt(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,'call')
rho_EurPut=rho_EurOpt(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,'put')
print('农业银行A股欧式看涨期权的Rho',round(rho_EurCall,4))
print('农业银行A股欧式看跌期权的Rho',round(rho_EurPut,4))

r_chg=0.001
call_chg=rho_EurCall*r_chg
put_chg=rho_EurPut*r_chg
print('无风险收益率上涨10个基点导致欧式看涨期权价格变化',round(call_chg,4))
print('无风险收益率上涨10个基点导致欧式看跌期权价格变化',round(put_chg,4))

# Example 12-21 基础资产价格与期权Rho的关系
rho_EurCall_list=rho_EurOpt(S_list2,K_ABC,sigma_ABC,shibor_6M,T_ABC,'call')
rho_EurPut_list=rho_EurOpt(S_list2,K_ABC,sigma_ABC,shibor_6M,T_ABC,'put')
plt.figure(figsize=(9,6))
plt.plot(S_list2,rho_EurCall_list,'b-',label=u'欧式看涨期权',lw=2.5)
plt.plot(S_list2,rho_EurPut_list,'r-',label=u'欧式看跌期权',lw=2.5)
plt.xlabel(u'股票价格',fontsize=13)
plt.ylabel('Rho',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'股票价格与期权Rho的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 12-22 期权期限与期权Rho的关系
rho_list1=rho_EurOpt(S1,K_ABC,sigma_ABC,shibor_6M,T_list,'call')
rho_list2=rho_EurOpt(S2,K_ABC,sigma_ABC,shibor_6M,T_list,'call')
rho_list3=rho_EurOpt(S3,K_ABC,sigma_ABC,shibor_6M,T_list,'call')
plt.figure(figsize=(9,6))
plt.plot(T_list,rho_list1,'b-',label=u'实值看涨期权',lw=2.5)
plt.plot(T_list,rho_list2,'r-',label=u'平价看涨期权',lw=2.5)
plt.plot(T_list,rho_list3,'g-',label=u'虚值看涨期权',lw=2.5)
plt.xlabel(u'期权期限',fontsize=13)
plt.ylabel('Rho',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'期权期限与期权Rho的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

def rho_AmerCall(S,K,sigma,r,T,N):
    '''定义一个运用N步二叉树模型计算美式看涨期权Rho的函数，
       并且假定无风险收益率增加0.0001（1个基点）
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数'''
    def American_call(S,K,sigma,r,T,N):
        t=T/N
        u=np.exp(sigma*np.sqrt(t))
        d=1/u
        p=(np.exp(r*t)-d)/(u-d)
        call_matrix=np.zeros((N+1,N+1))
        N_list=np.arange(N+1)
        S_end=S*u**(N-N_list)*d**N_list
        call_matrix[:,-1]=np.maximum(S_end-K,0)
        i_list=list(range(N))
        i_list.reverse()
        for i in i_list:
            j_list=np.arange(i+1)
            Si=S*u**(i-j_list)*d**j_list
            call_strike=np.maximum(Si-K,0)
            call_nostrike=np.exp(-r*t)*(p*call_matrix[:i+1,i+1]+(1-p)*call_matrix[1:i+2,i+1])
            call_matrix[:i+1,i]=np.maximum(call_strike,call_nostrike)
        return call_matrix[0,0]
    Value1=American_call(S,K,sigma,r,T,N)
    Value2=American_call(S,K,sigma,r+0.0001,T,N)
    return (Value2-Value1)/0.0001

def rho_AmerPut(S,K,sigma,r,T,N):
    '''定义一个运用N步二叉树模型计算美式看跌期权Rho的函数，
       依然假定无风险收益率增加0.0001（1个基点）
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数'''
    def American_put(S,K,sigma,r,T,N):
        t=T/N
        u=np.exp(sigma*np.sqrt(t))
        d=1/u
        p=(np.exp(r*t)-d)/(u-d)
        put_matrix=np.zeros((N+1,N+1))
        N_list=np.arange(N+1)
        S_end=S*u**(N-N_list)*d**N_list
        put_matrix[:,-1]=np.maximum(K-S_end,0)
        i_list=list(range(0,N))
        i_list.reverse()
        for i in i_list:
            j_list=np.arange(i+1)
            Si=S*u**(i-j_list)*d**j_list
            put_strike=np.maximum(K-Si,0)
            put_nostrike=np.exp(-r*t)*(p*put_matrix[:i+1,i+1]+(1-p)*put_matrix[1:i+2,i+1])
            put_matrix[:i+1,i]=np.maximum(put_strike,put_nostrike)
        return put_matrix[0,0]
    Value1=American_put(S,K,sigma,r,T,N)
    Value2=American_put(S,K,sigma,r+0.0001,T,N)
    return (Value2-Value1)/0.0001

# Example 12-23 美式期权的Rho
rho_AmerOpt1=rho_AmerCall(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step)
rho_AmerOpt2=rho_AmerPut(S_ABC,K_ABC,sigma_ABC,shibor_6M,T_ABC,step)
print('农业银行A股美式看涨期权的Rho',round(rho_AmerOpt1,4))
print('农业银行A股美式看跌期权的Rho',round(rho_AmerOpt2,4))

def impvol_call_Newton(C,S,K,r,T):
    '''定义一个运用BSM模型计算欧式看涨期权的隐含波动率的函数，且使用牛顿迭代法
    C: 代表观察到的看涨期权市场价格。
    S: 代表基础资产的价格。
    K: 代表期权的行权价格。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的剩余期限（年）'''
    from numpy import log,exp,sqrt
    from scipy.stats import norm
    def call_BSM(S,K,sigma,r,T):
        d1=(log(S/K)+(r+sigma**2/2)*T)/(sigma*sqrt(T))
        d2=d1-sigma*sqrt(T)
        call=S*norm.cdf(d1)-K*exp(-r*T)*norm.cdf(d2)
        return call
    sigma0=0.2
    diff=C-call_BSM(S,K,sigma0,r,T)
    i=0.0001
    while abs(diff)>0.0001:
        diff=C-call_BSM(S,K,sigma0,r,T)
        #print('Call_dff',diff)
        if diff>0:
            sigma0+=i
        else:
            sigma0-=i
    return sigma0

def impvol_put_Newton(P,S,K,r,T):
    '''定义一个运用BSM模型计算欧式看跌期权的隐含波动率的函数，且使用牛顿迭代法
    C: 代表观察到的看跌期权市场价格。
    S: 代表基础资产的价格。
    K: 代表期权的行权价格。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的剩余期限（年）'''
    from numpy import log,exp,sqrt
    from scipy.stats import norm
    def put_BSM(S,K,sigma,r,T):
        d1=(log(S/K)+(r+sigma**2/2)*T)/(sigma*sqrt(T))
        d2=d1-sigma*sqrt(T)
        put=K*exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
        return put
    sigma0=0.2
    diff=P-put_BSM(S,K,sigma0,r,T)
    i=0.0001
    while abs(diff)>0.0001:
        diff=P-put_BSM(S,K,sigma0,r,T)
        #print('Put_diff',diff)
        if diff>0:
            sigma0+=i
        else:
            sigma0-=i
    return sigma0

# Example 12-24 牛顿迭代法计算期权隐含波动率
import datetime as dt
T0=dt.datetime(2020,9,1) #隐含波动率的计算日
T1=dt.datetime(2021,3,24) #期权到期日
tenor=(T1-T0).days/365 #计算期权的剩余期限（年）
price_call=0.2826
price_put=0.1975
price_50ETF=3.406 #上证50ETF基金净值
shibor_6M=0.02847
K_50ETF=3.3
sigma_call=impvol_call_Newton(price_call,price_50ETF,K_50ETF,shibor_6M,tenor)
print('50ETF购3月3300期权合约的隐含波动率（牛顿迭代法）',round(sigma_call,4))
sigma_put=impvol_put_Newton(price_put,price_50ETF,K_50ETF,shibor_6M,tenor)
print('50ETF沽3月3300期权合约的隐含波动率（牛顿迭代法）',round(sigma_put,4))

def impvol_call_Binary(C,S,K,r,T):
    '''定义一个运用BSM模型计算欧式看涨期权隐含波动率的函数，且使用二分查找法迭代
    C: 代表观察到的看跌期权市场价格。
    S: 代表基础资产的价格。
    K: 代表期权的行权价格。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的剩余期限（年）'''
    from numpy import log,exp,sqrt
    from scipy.stats import norm
    def call_BSM(S,K,sigma,r,T):
        d1=(log(S/K)+(r+sigma**2/2)*T)/(sigma*sqrt(T))
        d2=d1-sigma*sqrt(T)
        call=S*norm.cdf(d1)-K*exp(-r*T)*norm.cdf(d2)
        return call
    sigma_min=0.001
    sigma_max=1.000
    sigma_mid=(sigma_min+sigma_max)/2
    call_min=call_BSM(S,K,sigma_min,r,T)
    call_max=call_BSM(S,K,sigma_max,r,T)
    call_mid=call_BSM(S,K,sigma_mid,r,T)
    diff=C-call_mid
    if C<call_min or C>call_max:
        print('Error')
    while abs(diff)>1e-6: #当差异值的绝对值大于0.000001
        diff=C-call_BSM(S,K,sigma_mid,r,T)
        print(diff)
        sigma_mid=(sigma_min+sigma_max)/2
        call_mid=call_BSM(S,K,sigma_mid,r,T)
        if C>call_mid:
            sigma_min=sigma_mid
        else:
            sigma_max=sigma_mid
    return sigma_mid

def impvol_put_Binary(P,S,K,r,T):
    '''定义一个运用BSM模型计算欧式看跌期权隐含波动率的函数，且使用二分查找法迭代
    P: 代表观察到的看跌期权市场价格。
    S: 代表基础资产的价格。
    K: 代表期权的行权价格。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的剩余期限（年）'''
    from numpy import log,exp,sqrt
    from scipy.stats import norm
    def put_BSM(S,K,sigma,r,T):
        d1=(log(S/K)+(r+sigma**2/2)*T)/(sigma*sqrt(T))
        d2=d1-sigma*sqrt(T)
        put=K*exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
        return put
    sigma_min=0.001
    sigma_max=1.000
    sigma_mid=(sigma_min+sigma_max)/2
    put_min=put_BSM(S,K,sigma_min,r,T)
    put_max=put_BSM(S,K,sigma_max,r,T)
    put_mid=put_BSM(S,K,sigma_mid,r,T)
    diff=P-put_mid
    if P<put_min or P>put_max:
        print('Error')
    while abs(diff)>1e-6:
        diff=P-put_BSM(S,K,sigma_mid,r,T)
        print(diff)
        sigma_mid=(sigma_min+sigma_max)/2
        put_mid=put_BSM(S,K,sigma_mid,r,T)
        if P>put_mid:
            sigma_min=sigma_mid
        else:
            sigma_max=sigma_mid
    return sigma_mid

# Example 12-25 二分查找法计算期权隐含波动率
sigma_call=impvol_call_Binary(price_call,price_50ETF,K_50ETF,shibor_6M,tenor)
sigma_put=impvol_put_Binary(price_put,price_50ETF,K_50ETF,shibor_6M,tenor)
print('50ETF购3月3300期权合约的隐含波动率（二分查找法）',round(sigma_call,4))
print('50ETF沽3月3300期权合约的隐含波动率（二分查找法）',round(sigma_put,4))

# Example 12-26 波动率微笑
S_Dec31=3.635
R_Dec31=0.02838
T2=dt.datetime(2020,12,31)
T3=dt.datetime(2021,6,23)
tenor1=(T3-T2).days/365
Put_list=np.array([0.0202,0.0306,0.0458,0.0671,0.0951,0.1300,0.1738,0.2253,0.2845,0.3540,0.4236])
K_list1=np.array([3.0000,3.1000,3.2000,3.3000,3.4000,3.5000,3.6000,3.7000,3.8000,3.9000,4.0000])
n1=len(K_list1) #不同行权价格的看跌期权合约数量
sigma_list1=np.zeros_like(Put_list) #构建存放看跌期权隐含波动率的初始数组
for i in np.arange(n1):
    sigma_list1[i]=impvol_put_Newton(Put_list[i],S_Dec31,K_list1[i],R_Dec31,tenor1)
    #print(i,' ',sigma_list1[i])
plt.figure(figsize=(9,6))
plt.plot(K_list1,sigma_list1,'b-',lw=2.5)
plt.xlabel(u'期权的行权价格',fontsize=13)
plt.ylabel(u'隐含波动率',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'期权的行权价格与上证50ETF认沽期权隐含波动率',fontsize=13)
plt.grid()
plt.show()

# Example 12-27 波动率（向上）斜偏
S_Sep30=4.5848
R_Sep30=0.02691
T4=dt.datetime(2020,9,30)
T5=dt.datetime(2021,3,24)
tenor2=(T5-T4).days/365
Call_list=np.array([0.4660,0.4068,0.3529,0.3056,0.2657,0.2267,0.1977,0.1707,0.1477,0.1019]) #沪深300ETF认购期权结算价
K_list2=np.array([4.2000,4.3000,4.4000,4.5000,4.6000,4.7000,4.8000,4.9000,5.0000,5.25000]) #期权的行权价格
n2=len(K_list2)
sigma_list2=np.zeros_like(Call_list)
for i in np.arange(n2):
    sigma_list2[i]=impvol_call_Binary(Call_list[i],S_Sep30,K_list2[i],R_Sep30,tenor2)
plt.figure(figsize=(9,6))
plt.plot(K_list2,sigma_list2,'r-',lw=2.5)
plt.xlabel(u'期权的行权价格',fontsize=13)
plt.ylabel(u'隐含波动率',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'期权的行权价格与沪深300ETF认购期权隐含波动率',fontsize=13)
plt.grid()
plt.show()