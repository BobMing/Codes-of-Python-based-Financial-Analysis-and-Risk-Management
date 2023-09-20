# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:10:46 2023

@author: XIE Ming
@github: https://github.com/BobMing
@linkedIn: https://www.linkedin.com/in/tseming
@email: xieming_xm@163.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif']=['FangSong']
mpl.rcParams['axes.unicode_minus']=False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# 贷款市场报价利率
LPR=pd.read_excel('E:\OneDrive\附件\数据\第6章\贷款市场报价利率（LPR）数据.xlsx',sheet_name='Sheet1',header=0,index_col=0)
LPR.plot(figsize=(9,6),grid=True,fontsize=13)
plt.ylabel(u'利率',fontsize=11)
# 在2020年前5个月降幅十分明显，根本原因是为了有效对冲20年初突如其来的“新冠”疫情对经济的负面影响，运用货币政策强化逆周期调节，在引导货币市场利率中枢下移的同时，通过LPR传导进一步降低实体经济的融资成本

LPR=pd.read_excel('E:\OneDrive\附件\数据\第6章\贷款市场报价利率（LPR）数据.xlsx',sheet_name='Sheet2',header=0,index_col=0)
LPR.plot(figsize=(9,6),grid=True,fontsize=13)
plt.ylabel(u'LPR利率',fontsize=11)
# 1年期LPR在21年12月、22年1月连续降低5个基点，22年8月再降5个基点至3.65%

# 银行间同业拆借利率
IBL=pd.read_excel('E:\OneDrive\附件\数据\第6章\银行间同业拆借利率（2019年-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
IBL.plot(figsize=(9,6),grid=True,fontsize=13)
plt.ylabel(u'利率',fontsize=11)

# 银行间回购利率
FR=pd.read_excel('E:\OneDrive\附件\数据\第6章\银行间回购定盘利率（2019年-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
FR.plot(figsize=(9,6),grid=True,fontsize=13)
plt.ylabel(u'利率',fontsize=11)

# Shibor 上海银行间同业拆放利率
Shibor=pd.read_excel('E:\OneDrive\附件\数据\第6章\Shibor利率（2019年至2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
Shibor.plot(figsize=(9,6),grid=True,fontsize=13)
plt.ylabel(u'利率',fontsize=11)
# 典型V字走势。在2020年第1季度Shibor出现较大幅度的下降，是为了应对“新冠”疫情对经济的负面冲击，央行推出了较宽松的货币政策，使得市场利率出现了较明显的下降。

# 人民币汇率
exchange=pd.read_excel('E:\OneDrive\附件\数据\第6章\人民币汇率每日中间价（2005年7月21日至2020年末）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
exchange.plot(subplots=True,sharex=True,layout=(2,2),figsize=(9,6),grid=True,fontsize=13)
plt.subplot(2,2,1) # 第1张子图
plt.ylabel(u'汇率',fontsize=11,position=(0,0)) # 增加第1张子图的纵坐标标签

# 人民币汇率指数
index_RMB=pd.read_excel('E:\OneDrive\附件\数据\第6章\人民币汇率指数（2015年11月30日至2020年末）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
index_RMB.plot(figsize=(9,6),grid=True,fontsize=13)
plt.ylabel(u'汇率指数',fontsize=11)

# Example 6-1 利率的相对性
par=1e4 #本金为1W元
r=0.02 #2%的1年期利率
M=[1,2,4,12,52,365] #不同的复利频次
name=['每年复利1次','每半年复利1次','每季度复利1次','每月复利1次','每周复利1次','每天复利1次']
value=[] #建立存放1年后本息合计金额的初始数列
i=0 #设置一个标量，用于后续的for循环语句
for m in M:
    value.append(par*(1+r/m)**m)
    print(name[i],'本息合计金额',round(value[i],2))
    i+=1
# 结论推广：FV=A(1+R/m)^(mn)
def FV(A,n,R,m):
    '''定义一个用于计算不同复利频次本息和的函数
    A: 初始的投资本金
    n：投资期限（年）
    R：年利率
    m：每年复利频次，'Y'代表每年复利1次，'S'代表每半年复利1次，'Q'代表每季度复利1次，'M'代表每月复利1次，'W'代表每周复利1次，其他输入表示每天复利1次
    '''
    if m=='Y':
        value=A*pow(1+R,n)
    elif m=='S':
        value=A*pow(1+R/2,2*n)
    elif m=='Q':
        value=A*pow(1+R/4,4*n)
    elif m=='M':
        value=A*pow(1+R/12,12*n)
    elif m=='W':
        value=A*pow(1+R/52,52*n)
    else:
        value=A*pow(1+R/365,365*n)
    return value
N=1
FV_week=FV(A=par,n=N,R=r,m='W')
print('每周复利1次得到的本息和',round(FV_week,2))
# 复利频次与本息和的关系
par_new=100
M_list=np.arange(1,201) #生成从1到200的自然数数组
Value_list=par_new*pow(1+r/M_list,M_list)
plt.figure(figsize=(9,6))
plt.plot(M_list,Value_list,'r-',lw=2.5)
plt.xlabel(u'复利频次',fontsize=13)
plt.xlim(0,200)
plt.ylabel(u'本息和',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'复利频次与本息和之间的关系图',fontsize=13)
plt.show()

# Example 6-2 不同复利频次的利率之间存在等价关系
def R_m2(R_m1,m1,m2):
    '''
    定义一个已知复利频次m1的利率，计算等价的新复利频次m2利率的函数
    ----------
    R_m1 : 对应于复利频次m1的利率
    m1 : 对应于利率R1的复利频次
    m2 : 新的复利频次
    '''
    r=m2*(pow(1+R_m1/m1,m1/m2)-1)
    return r
R_semiannual=0.03 #按半年复利的利率
m_semiannual=2 #按半年复利的频次
m_month=12 #按月复利的频次
R_month=R_m2(R_m1=R_semiannual,m1=m_semiannual,m2=m_month)
print('计算等价的按月复利对应的利率',round(R_month,6))

# Example 6-3 连续复利利率与每年复利m次的利率之间的等价关系
def Rc(Rm,m):
    '''
    定义一个已知复利频次和对应利率，计算等价的连续复利利率的函数
    Rm: 复利频次m的利率
    m: 复利频次
    '''
    return m*np.log(1+Rm/m)
def Rm(Rc,m):
    '''
    定义一个已知复利频次和连续复利利率，计算对应复利频次的利率的函数
    Rc: 连续复利利率
    m: 复利频次
    '''
    return m*(np.exp(Rc/m)-1)
R1=0.04
M1=4
R_c=Rc(Rm=R1,m=M1)
print('等价的连续复利利率',round(R_c,6))
R2=0.05
M2=12
R_m=Rm(Rc=R2,m=M2)
print('等价的按月复利的利率',round(R_m,6))

# Example 6-5 T年期零息利率 zero-coupon interest rate
R3=0.03
T=3
value_3y=FV(A=par,n=T,R=R3,m='Y')
print('3年后到期时的本息和',round(value_3y,2))

# Example 6-6 远期利率的测算
par=100 #本金100元
zero_rate=np.array([0.02,0.022,0.025,0.028,0.03]) #包含零息利率的数组
T_list=np.array([1,2,3,4,5]) #包含期限的数组
import scipy.optimize as sco
def f(Rf):
    from numpy import exp
    R2,R3,R4,R5=Rf #设置不同的远期利率
    year2=par*exp(zero_rate[0]*T_list[0])*exp(R2*T_list[0])-par*exp(zero_rate[1]*T_list[1]) #计算第2年远期利率的等式
    year3=par*exp(zero_rate[1]*T_list[1])*exp(R3*T_list[0])-par*exp(zero_rate[2]*T_list[2]) #计算第3年远期利率的等式
    year4=par*exp(zero_rate[2]*T_list[2])*exp(R4*T_list[0])-par*exp(zero_rate[3]*T_list[3]) #计算第4年远期利率的等式
    year5=par*exp(zero_rate[3]*T_list[3])*exp(R5*T_list[0])-par*exp(zero_rate[-1]*T_list[-1]) #计算第5元远期利率的等式
    return np.array([year2,year3,year4,year5])
R0=[0.1,0.1,0.1,0.1] #包含猜测的初始远期利率的数组
forward_rates=sco.fsolve(func=f,x0=R0) #计算远期利率
print('第2年远期利率',round(forward_rates[0],6))
print('第3年远期利率',round(forward_rates[1],6))
print('第4年远期利率',round(forward_rates[2],6))
print('第5年远期利率',round(forward_rates[3],6))

def Rf(R1,R2,T1,T2):
    '''
    定义一个计算远期利率的函数
    ----------
    R1 : 表示对应期限为T1的零息利率（连续复利）.
    R2 : 表示对应期限为T2的零息利率（连续复利）.
    T1 : 表示对应于零息利率R1的期限（年）.
    T2 : 表示对应于零息利率R2的期限（年）.
    '''
    forward_rate=R2+(R2-R1)*T1/(T2-T1) #计算远期利率
    return forward_rate
Rf_result=Rf(R1=zero_rate[:4],R2=zero_rate[1:],T1=T_list[:4],T2=T_list[1:]) #计算远期利率
print('第2年远期利率',round(Rf_result[0],6))
print('第3年远期利率',round(Rf_result[1],6))
print('第4年远期利率',round(Rf_result[2],6))
print('第5年远期利率',round(Rf_result[-1],6))

def Cashflow_FRA(Rk,Rm,L,T1,T2,position,when):
    '''
    定义一个计算远期利率协议现金流的函数
    Rk: 表示远期利率协议中约定的固定利率。
    Rm: 表示在T1时点观察到的[T1,T2]的参考利率。
    L: 表示远期利率协议的本金。
    T1: 表示期限。
    T2: 表示期限，T2>T1。
    position: 表示远期利率协议多头或空头，输入position='long'表示多头，输入其他则表示空头。
    when: 表示现金流发生的具体时点，输入when='begin'表示在T1时点发生现金流，输入其他则表示在T2时点发生现金流。
    '''
    if position=='long': #针对远期利率协议多头
        if when=='begin': #当现金流发生在T1时点
            cashflow=(Rm-Rk)*(T2-T1)*L/(1+(T2-T1)*Rm)
        else: #当现金流发生在T2时点
            cashflow=(Rm-Rk)*(T2-T1)*L
    else: #针对远期利率协议空头
        if when=='begin': #当现金流发生在T1时点
            cashflow=(Rk-Rm)*(T2-T1)*L/(1+(T2-T1)*Rm)
        else:
            cashflow=(Rk-Rm)*(T2-T1)*L
    return cashflow
# Example 6-7 计算远期利率协议现金流的案例
par_FRA=1e8
R_fix=0.02 #固定利率
Shibor_3M=0.02756 #2020年12月31日的3个月期Shibor 
tenor1=1 #期限1年（T1）
tenor2=1.25 #期限1.25年（T2）
FRA_long_end=Cashflow_FRA(Rk=R_fix, Rm=Shibor_3M, L=par_FRA, T1=tenor1, T2=tenor2, position='long', when='end') #远期利率协议多头（I公司）在第1.25年年末的现金流
FRA_short_end=Cashflow_FRA(Rk=R_fix, Rm=Shibor_3M, L=par_FRA, T1=tenor1, T2=tenor2, position='short', when='end') #远期利率协议空头（J银行）在第1.25年年末的现金流
FRA_long_begin=Cashflow_FRA(Rk=R_fix, Rm=Shibor_3M, L=par_FRA, T1=tenor1, T2=tenor2, position='long', when='begin') #远期利率协议多头（I公司）在第1年年末的现金流
FRA_short_begin=Cashflow_FRA(Rk=R_fix, Rm=Shibor_3M, L=par_FRA, T1=tenor1, T2=tenor2, position='short', when='begin') #远期利率协议空头（J银行）在第1年年末的现金流
print('I企业现金流发生在2021年3月31日的金额',round(FRA_long_end,2))
print('J银行现金流发生在2021年3月31日的金额',round(FRA_short_end,2))
print('I企业现金流发生在2020年12月31日的金额',round(FRA_long_begin,2))
print('J银行现金流发生在2020年12月31日的金额',round(FRA_short_begin,2))

def Value_FRA(Rk,Rf,R,L,T1,T2,position):
    '''
    定义一个计算远期利率协议价值的函数
    Rk: 表示远期利率协议中约定的固定利率。
    Rf: 表示定价日观察到的未来[T1,T2]的远期参考利率。
    R: 表示期限为T2的无风险利率，且是连续复利。
    L: 表示远期利率协议的本金。
    T1: 表示期限。
    T2: 表示期限，T2>T1。
    position: 表示远期利率协议多头或空头，输入position='long'表示多头，输入其他则表示空头
    '''
    if position=='long': #对于远期利率协议的多头
        value=L*(Rf-Rk)*(T2-T1)*np.exp(-R*T2)
    else:
        value=L*(Rk-Rf)*(T2-T1)*np.exp(-R*T2)
    return value
# Example 6-8 远期利率协议定价的案例
# Step 1: 用174~184行的自定义函数Rf，计算在2020年12月31日当天处于2021年7月1日至9月30日期间的远期3个月期Shibor
Shibor_6M=0.02838 #6个月期Shibor
Shibor_9M=0.02939 #9个月期Shibor
Tenor1=0.5 #设置期限0.5年（T1）
Tenor2=0.75 #设置期限0.75年（T2）
FR_Shibor=Rf(R1=Shibor_6M,R2=Shibor_9M,T1=Tenor1,T2=Tenor2) #计算远期的3个月期Shibor
print('计算得到2020年12月31日远期的3个月期Shibor',round(FR_Shibor,6))
# Step 2: 用自定义函数Value_FRA计算远期利率协议的价值
Par_FRA=2e8 #远期利率协议的面值
R_fix=0.03 #远期利率协议中约定的固定利率
R_riskfree=0.024477 #9个月期的无风险利率
FRA_short=Value_FRA(Rk=R_fix, Rf=FR_Shibor, R=R_riskfree, L=Par_FRA, T1=Tenor1, T2=Tenor2, position='short') #计算远期利率协议空头（M公司）的协议价值
FRA_long=Value_FRA(Rk=R_fix, Rf=FR_Shibor, R=R_riskfree, L=Par_FRA, T1=Tenor1, T2=Tenor2, position='long') #计算远期利率协议多头（N银行）的协议价值
print('2020年12月31日M企业的远期利率协议价值',round(FRA_short,2))
print('2020年12月31日N银行的远期利率协议价值',round(FRA_long,2))

# Example 6-9 汇率报价/汇兑计算
def exchange(E,LC,FC,quote):
    '''定义一个通过汇率计算汇兑金额的函数
    E: 代表汇率报价。
    LC: 代表用于兑换的以本币计价的币种金额，输入LC='Na'表示未已知相关金额。
    FC: 代表用于兑换的以外币计价的币种金额，输入FC='Na'表示未已知相关金额。
    quote: 代表汇率标价方法，输入quote='direct'表示直接标价法，输入其他则表示间接标记法。
    '''
    if LC=='Na': #将外币兑换为本币
        if quote=='direct': #汇率标价方法是直接标记法
            value=FC*E #计算兑换得到本币的金额
        else:              #汇率标价方法是间接标记法
            value=FC/E #计算兑换得到本币的金额
    else: #将本币兑换为外币
        if quote=='direct':
            value=LC/E
        else:
            value=LC*E
    return value
USD_RMB=7.1277
GBP_EUR=1.1135
Amount_USD=6e6
Amount_EUR=8e6
Amount_RMB=exchange(E=USD_RMB,LC='Na',FC=Amount_USD,quote='direct')
Amount_GBP=exchange(E=GBP_EUR,LC='Na',FC=Amount_EUR,quote='indirect')
print('P企业将600万美元兑换成人民币的金额（单位：元）',round(Amount_RMB,2))
print('Q企业将800万欧元兑换成英镑的金额（单位：英镑）',round(Amount_GBP,2))

# Example 6-10 三角套利
def tri_arbitrage(E1,E2,E3,M,A,B,C):
    '''定义一个计算三角套利收益并显示套利路径的函数
    E1: 代表A货币兑B货币的汇率，以若干个单位A货币表示1个单位B货币。
    E2: 代表B货币兑C货币的汇率，以若干个单位B货币表示1个单位C货币。
    E3: 代表A货币兑C货币的汇率，以若干个单位A货币表示1个单位C货币。
    M: 代表A货币计价的初始本金。
    A: 代表A货币的名称，例如输入A='人民币'就代表A货币是人民币。
    B: 代表B货币的名称，例如输入B='美元'就代表B货币是美元。
    C: 代表C货币的名称，例如输入C='欧元'就代表C货币是欧元。
    '''
    E3_new=E1*E2
    if E3_new>E3: #当交叉汇率高于直接的汇率报价
        profit=M*(E3_new/E3-1)
        sequence=['三角套利的路径：',A,'→',C,'→',B,'→',A]
    elif E3_new<E3:
        profit=M*(E3/E3_new-1)
        sequence=['三角套利的路径：',A,'→',B,'→',C,'→',A]
    else:
        profit=0
        sequence=['三角套利的路径：不存在']
    return [profit,sequence] #输出包含套利收益和套利路径的列表
USD_RMB=7.0965 #美元兑人民币汇率
USD_RUB=68.4562 #美元兑卢布汇率
RMB_RUB=9.7150 #人民币兑卢布汇率
value_RMB=1e8
arbitrage=tri_arbitrage(E1=USD_RMB, E2=1/USD_RUB, E3=1/RMB_RUB, M=value_RMB, A='人民币', B='美元', C='卢布')
print('三角套利的收益',round(arbitrage[0],2))
print(arbitrage[1])

def FX_forward(spot,r_A,r_B,T):
    '''定义一个计算远期汇率的函数，并且两种货币分别是A、B货币
    spot: 代表即期汇率，标价方式是以若干个单位A货币表示1个单位B货币。
    r_A: 代表A货币的无风险利率，且每年复利1次。
    r_B: 代表B货币的无风险利率，且每年复利1次。
    T: 代表远期汇率的期限，且以年为单位。
    '''
    forward=spot*(1+r_A*T)/(1+r_B*T)
    return forward
# Example 6-11 远期汇率的测算
FX_spot=7.0965 #即期汇率
Tenor=np.array([1/12,3/12,6/12,1.0]) #4个不同期限的数据
Shibor=np.array([0.015820,0.015940,0.016680,0.019030]) #Shibor
Libor=np.array([0.001801,0.003129,0.004813,0.006340]) #Libor
FX_forward_list=np.zeros_like(Tenor) #与期限数组形状相同的初始远期汇率数组
for i in range(len(Tenor)):
    FX_forward_list[i]=FX_forward(spot=FX_spot, r_A=Shibor[i], r_B=Libor[i], T=Tenor[i]) #计算不同期限的远期汇率
print('1个月期的美元兑人民币远期汇率',round(FX_forward_list[0],4))
print('3个月期的美元兑人民币远期汇率',round(FX_forward_list[1],4))
print('6个月期的美元兑人民币远期汇率',round(FX_forward_list[2],4))
print('1年期的美元兑人民币远期汇率',round(FX_forward_list[-1],4))

# Example 6-12 抵补套利
def cov_arbitrage(S,F,M_A,M_B,r_A,r_B,T,A,B):
    '''定义一个计算抵补套利收益并显示套利路径的函数，且两种货币分别是A、B货币
    S: 代表即期汇率，以若干个单位A货币表示1个单位B货币。
    F: 代表外汇市场报价的远期汇率，标价方式与即期汇率一致。
    M_A: 代表借入A货币的本金，输入M_A='Na'表示未已知相关金额。
    M_B: 代表借入B货币的本金，输入M_B='Na'表示未已知相关金额。
    r_A: 代表A货币的无风险利率，且每年复利1次。
    r_B: 代表B货币的无风险利率，且每年复利1次。
    T: 代表远期汇率的期限，且以年为单位。
    A: 代表A货币的名称，例如输入A='人民币'表示A货币是人民币。
    B: 代表B货币的名称，例如输入B='美元'表示B货币是美元。
    '''
    #第1步：计算均衡远期汇率且当均衡远期汇率小于实际远期汇率时
    F_new=S*(1+r_A*T)/(1+r_B*T)
    if F_new<F: #均衡远期汇率小于实际远期汇率
        if M_B=='Na': #借入A货币的本金
            profit=M_A*(1+T*r_B)*F/S-M_A*(1+r_A*T) #计算初始借入A货币抵补套利的套利收益
            if profit>0: #套利收益大于0
                sequence=['套利路径如下',
                          '(1)初始时刻接入的货币名称：',A,
                          '(2)按照即期汇率兑换后并投资的货币名称：',B,
                          '(3)按照远期汇率在投资结束时兑换后的货币名称：',A,
                          '(4)偿还初始时刻的借入资金']
            else:
                sequence=['不存在套利机会']
        else: #借入B货币的本金
            profit=M_B*S*(1+r_A*T)/F-M_B*(1+r_B*T)
            if profit>0:
                sequence=['套利路径如下',
                          '(1)初始时刻接入的货币名称：',B,
                          '(2)按照即期汇率兑换后并投资的货币名称：',A,
                          '(3)按照远期汇率在投资结束时兑换后的货币名称：',B,
                          '(4)偿还初始时刻的借入资金']
            else:
                sequence=['不存在套利机会']
    #第2步：当均衡远期汇率大于实际远期汇率时
    elif F_new>F:
        if M_B=='Na': #借入A货币的本金
            profit=M_A*(1+T*r_B)*F/S-M_A*(1+r_A*T) #计算初始借入A货币抵补套利的套利收益
            if profit>0: #套利收益大于0
                sequence=['套利路径如下',
                          '(1)初始时刻接入的货币名称：',A,
                          '(2)按照即期汇率兑换后并投资的货币名称：',B,
                          '(3)按照远期汇率在投资结束时兑换后的货币名称：',A,
                          '(4)偿还初始时刻的借入资金']
            else:
                sequence=['不存在套利机会']
        else: #借入B货币的本金
            profit=M_B*S*(1+r_A*T)/F-M_B*(1+r_B*T)
            if profit>0:
                sequence=['套利路径如下',
                          '(1)初始时刻接入的货币名称：',B,
                          '(2)按照即期汇率兑换后并投资的货币名称：',A,
                          '(3)按照远期汇率在投资结束时兑换后的货币名称：',B,
                          '(4)偿还初始时刻的借入资金']
            else:
                sequence=['不存在套利机会']
    #第3步：当均衡远期汇率等于实际远期汇率时
    else:
        profit=0
        sequence=['不存在套利机会']
    return [profit,sequence]
value_RMB=1e8
value_USD=1.4e7
Shibor_3M=0.01594
Libor_3M=0.003129
tenor=3/12
FX_spot=7.0965
FX_forward=7.1094
arbitrage_RMB=cov_arbitrage(S=FX_spot,F=FX_forward,M_A=value_RMB,M_B='Na',r_A=Shibor_3M,r_B=Libor_3M,T=tenor,A='人民币',B='美元')
print('借入人民币1亿元开展抵补套利的收益（元）',round(arbitrage_RMB[0],2))
print(arbitrage_RMB[1]) #输出套利路径
arbitrage_USD=cov_arbitrage(S=FX_spot, F=FX_forward, M_A='Na', M_B=value_USD, r_A=Shibor_3M, r_B=Libor_3M, T=tenor, A='人民币', B='美元')
print('借入1400万美元开展抵补套利的收益（美元）',round(arbitrage_USD[0],2))
print(arbitrage_USD[1])

# Example 6-13 远期外汇合约
def Value_FX_Forward(F1,F2,S,par,R,t,pc,vc,position):
    '''定义一个计算远期外汇合约价值的函数，两种货币分别是A货币、B货币
    F1: 代表合约初始日约定的远期汇率R_fT，以若干个单位A货币表示1个单位B货币。
    F2: 代表合约定价日的远期汇率R_fτ，标价方式同F1。
    S: 代表合约定价日的即期汇率E_s，标价方式与F1相同。
    par: 代表合约本金，且计价货币与确定币种参数pc保持一致。
    R: 代表非合约本金计价货币的无风险利率（连续复利），比如本金(pc)是A货币，则该利率(R)为B货币的无风险利率。
    t: 代表合约的剩余期限τ，以年为单位。
    pc: 代表合约本金的币种，输入pc='A'表示选择A货币，输入其他则表示选择B货币。
    vc: 代表合约价值的币种，输入vc='A'表示选择A货币，输入其他则表示选择B货币。
    position: 代表合约的头寸方向，输入position='long'表示多头，输入其他则表示合约空头。
    '''
    from numpy import exp
    if pc=='A': #情形1：本金par=M单位A货币
        if position=='long':
            if vc=='A':
                value=S*(par/F2-par/F1)*exp(-R*t)
            else:
                value=(par/F2-par/F1)*exp(-R*t) #括号内A货币本金交割为B货币，按B货币无风险利率贴现
        else:
            if vc=='A':
                value=S*(par/F1-par/F2)*exp(-R*t)
            else:
                value=(par/F1-par/F2)*exp(-R*t)
    else: #情形2：本金par=N单位B货币
        if position=='long':
            if vc=='A':
                value=(par*F2-par*F1)*exp(-R*t)
            else:
                value=(par*F2-par*F1)*exp(-R*t)/S
        else:
            if vc=='A':
                value=(par*F1-par*F2)*exp(-R*t)
            else:
                value=(par*F1-par*F2)*exp(-R*t)/S
    return value
# 20年2月28日 合约订立日数据
par_RMB=1e8
FX_spot_Feb28=7.0066
Shibor_6M_Feb28=0.0256
Libor_6M_Feb28=0.013973
T1=6/12
FX_forward_Feb28=FX_forward(spot=FX_spot_Feb28,r_A=Shibor_6M_Feb28,r_B=Libor_6M_Feb28,T=T1) #计算20年2月28日的6个月期远期汇率
print('2020年2月28日的6个月期美元兑人民币远期汇率',FX_forward_Feb28)
# 20年5月28日 定价日数据
FX_spot_May28=7.1277
Shibor_3M_May28=0.0143
Libor_3M_May28=0.0035
rate_USD=0.003494
T2=3/12
FX_forward_May28=FX_forward(spot=FX_spot_May28,r_A=Shibor_3M_May28,r_B=Libor_3M_May28,T=T2) #计算20年5月28日的3个月期远期汇率
print('2020年5月28日的3个月期美元兑人民币远期汇率',FX_forward_May28)
value_short=Value_FX_Forward(F1=FX_forward_Feb28, F2=FX_forward_May28, S=FX_spot_May28, par=par_RMB, R=rate_USD, t=T2, pc='A', vc='A', position='short') #合约空头（V企业）的人民币价值（到期收人民币(本金)，支付美元）
value_long=Value_FX_Forward(F1=FX_forward_Feb28, F2=FX_forward_May28, S=FX_spot_May28, par=par_RMB, R=rate_USD, t=T2, pc='A', vc='B', position='long') #合约多头（W银行）的美元价值（到期收美元，支付人民币(本金)）
print('合约空头（V公司）在2020年5月28日的远期外汇合约价值（元）',round(value_short,2))
print('合约多头（W银行）在2020年5月28日的远期外汇合约价值（美元）',round(value_long,2))