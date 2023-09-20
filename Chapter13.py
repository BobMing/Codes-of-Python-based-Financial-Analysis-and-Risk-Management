# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:45:20 2023

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

# Example 13-1 抽象金融市场的保本票据合成策略运用
par_ppn=100 #保本票据的本金
par_bond=100 #无风险零息债券面值
price_bond=96 #无风险零息债券价格
price_call=0.4 #欧式看涨期权报价
K=5.0 #期权行权价格

N_bond=par_ppn/par_bond #购买的无风险零息债券数量
N_call=(par_ppn-N_bond*price_bond)/price_call #购买的欧式看涨期权数量
print('构建1份保本票据需要购买的无风险零息债券数量',N_bond)
print('构建1份保本票据需要购买的欧式看涨期权数量',N_call)

price_z_list=np.linspace(3,7,120) #创建期权到期时Z股票价格等差数列
profit_call=np.maximum(price_z_list-K,0) #欧式看涨期权到期时的收益
profit_ppn=N_bond*par_bond+N_call*profit_call-par_ppn #保本票据到期的收益金额
return_ppn=profit_ppn/par_ppn #保本票据到期的收益率
plt.figure(figsize=(9,6))
plt.plot(price_z_list,return_ppn,'r-',lw=2.5)
plt.xlabel(u'Z股票价格',fontsize=13)
plt.ylabel(u'保本票据收益率',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'Z股票价格与保本票据收益率的关系图',fontsize=13)
plt.grid()
plt.show()

# Example 13-2 现实金融市场的保本票据合成策略运用
par_PPN=1e8 #保本票据的面值
par_SC=100 #18四川01债券的面值
coupon=0.0373 #18四川01债券的票面利率
price_SC=102.2682 #2020年8月24日18四川01债券的价格
price_opt=0.2230 #2020年8月24日沪深300ETF认购期权的价格
price_300ETF=4.8207 #2020年8月24日沪深300ETF基金的净值
K_300ETF=5.0 #期权行权价格
price_HS300=4755.85 #2020年8月24日沪深300指数的收盘价
N1=10 #债券的交易单位（10张）
N2=10000 #每张期权合约单位（10000份沪深300ETF基金）

cashflow_SC=par_SC*(1+coupon) #18四川01债券到期日的本息
from math import ceil
N_SC=N1*ceil(par_PPN/(N1*cashflow_SC)) #计算债券数量（10张的整数倍）
print('购买18四川01债券数量（张）',N_SC)
N_opt=(par_PPN-price_SC*N_SC)/(price_opt*N2) #计算期权合约数量
N_opt=int(N_opt) #确保期权合约数量是整数
print('购买300ETF购3月5000期权合约数量（张）',N_opt)
cash=par_PPN-price_SC*N_SC-N_opt*price_opt*N2 #未购买债券和期权的剩余现金
print('保本票据本金未用于购买债券和期权的剩余现金',round(cash,2))
K_HS300=K_300ETF*price_HS300/price_300ETF #等于期权行权价格的沪深300指数点位
print('恰好等于期权行权价格的沪深300指数的点位',round(K_HS300,2))

HS300_chg=np.array([0.05,0.1,0.2,0.3]) #创建沪深300指数涨幅的数组
profit_opt=N_opt*N2*np.maximum(price_300ETF*(1+HS300_chg)-K_300ETF,0) #计算期权收益
profit_PPN=cashflow_SC*N_SC+cash+profit_opt-par_PPN #计算保本票据的收益金额
R_PPN=profit_PPN/par_PPN
print('到期日沪深300指数上涨5%时保本票据收益率',round(R_PPN[0],6))
print('到期日沪深300指数上涨10%时保本票据收益率',round(R_PPN[1],6))
print('到期日沪深300指数上涨20%时保本票据收益率',round(R_PPN[2],6))
print('到期日沪深300指数上涨30%时保本票据收益率',round(R_PPN[-1],6))

HS300_list=np.linspace(4000,7000,500) #保本票据到期时沪深300指数的等差数列
price_300ETF_list=HS300_list*(price_300ETF/price_HS300) #沪深300ETF基金净值数组
profit_opt_list=N_opt*N2*np.maximum(price_300ETF_list-K_300ETF,0) #期权收益金额数组
profit_PPN_list=cashflow_SC*N_SC+cash+profit_opt_list-par_PPN #保本票据的收益金额数组
R_PPN_list=profit_opt_list/par_PPN #保本票据收益率数组
plt.figure(figsize=(9,6))
plt.plot(HS300_list,R_PPN_list,'r-',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'保本票据收益率',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与保本票据收益率的关系图',fontsize=13)
plt.grid()
plt.show()

# Example 13-3 买入备兑看涨期权
C=0.2605 #策略构建日（2020年6月1日）看涨期权价格
K=4.0 #看涨期权行权价格
S0_ETF=4.0364 #策略构建日沪深300ETF基金净值
S0_index=3971.34 #策略构建日沪深300指数收盘价
St_index=np.linspace(3000,5000,500) #策略到期日沪深300指数等差数列
St_ETF=S0_ETF*St_index/S0_index #对应不同沪深300指数的沪深300ETF基金净值
N_ETF=1e4 #沪深300ETF基金空头头寸数量
N_call=1 #沪深300ETF认购期权多头头寸数量
N_underlying=1e4 #1张期权基础资产是10000份基金
profit_ETF_short=-N_ETF*(St_ETF-S0_ETF) #期权到期日沪深300ETF
profit_call_long=N_call*N_underlying*(np.maximum(St_ETF-K,0)-C) #期权到期日沪深300认购期权多头头寸的收益
profit_covcall_long=profit_ETF_short+profit_call_long #期权到期日买入备兑看涨期权的收益
plt.figure(figsize=(9,6))
plt.plot(St_index,profit_ETF_short,'b--',label=u'沪深300ETF基金空头',lw=2.5)
plt.plot(St_index,profit_call_long,'g--',label=u'沪深300ETF认购期权多头',lw=2.5)
plt.plot(St_index,profit_covcall_long,'r-',label=u'买入备兑看涨期权策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与买入备兑看涨期权收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-4 卖出备兑看涨期权
C=0.201 #策略构建日（2020年7月1日）看涨期权价格
K=4.3
S0_ETF=4.3429 #策略构建日沪深300ETF基金净值
S0_index=4247.78 #策略构建日沪深300指数收盘价
St_index=np.linspace(3500,5500,500)
St_ETF=St_index*S0_ETF/S0_index
profit_ETF_long=N_ETF*(St_ETF-S0_ETF) #期权到期日沪深300ETF基金多头头寸的收益
profit_call_short=-N_call*N_underlying*(np.maximum(St_ETF-K,0)-C) #期权到期日沪深300ETF认购期权空头头寸的收益
profit_covcall_short=profit_ETF_long+profit_call_short #期权到期日卖出备兑看涨期权的收益
plt.figure(figsize=(9,6))
plt.plot(St_index,profit_ETF_long,'b--',label=u'沪深300ETF基金多头',lw=2.5)
plt.plot(St_index,profit_call_short,'g--',label=u'沪深300ETF认购期权空头',lw=2.5)
plt.plot(St_index,profit_covcall_short,'r-',label=u'卖出备兑看涨期权策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与卖出备兑看涨期权收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-5 买入保护看跌期权
P=0.4416
K=4.9
S0_ETF=4.9168
S0_index=4771.31
St_index=np.linspace(4000,6000,500)
St_ETF=St_index*S0_ETF/S0_index
N_put=1 #沪深300ETF认沽期权多头头寸的数量
profit_ETF_long=N_ETF*(St_ETF-S0_ETF) #基资多头收益
profit_put_long=N_put*N_underlying*(np.maximum(K-St_ETF,0)-P) #看跌期权多头收益
profit_protput_long=profit_ETF_long+profit_put_long
plt.figure(figsize=(9,6))
plt.plot(St_index,profit_ETF_long,'b--',label=u'沪深300ETF基金多头',lw=2.5)
plt.plot(St_index,profit_put_long,'g--',label=u'沪深300ETF认沽期权多头',lw=2.5)
plt.plot(St_index,profit_protput_long,'r-',label=u'买入保护看跌期权策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与买入保护看跌期权收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-6 卖出保护看跌期权
P=0.4211
K=5.0
S0_ETF=4.9966
S0_index=4842.12
St_index=np.linspace(3800,5800,500)
St_ETF=St_index*S0_ETF/S0_index
profit_ETF_short=-N_ETF*(St_ETF-S0_ETF)
profit_put_short=-N_put*N_underlying*(np.maximum(K-St_ETF,0)-P)
profit_protput_short=profit_ETF_short+profit_put_short
plt.figure(figsize=(9,6))
plt.plot(St_index,profit_ETF_short,'b--',label=u'沪深300ETF基金空头',lw=2.5)
plt.plot(St_index,profit_put_short,'g--',label=u'沪深300ETF认沽期权空头',lw=2.5)
plt.plot(St_index,profit_protput_short,'r-',label=u'卖出保护看跌期权策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与卖出保护看跌期权收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-7 策略的期间收益
price=pd.read_excel(r'E:\OneDrive\附件\数据\第13章\沪深300ETF期权价格与沪深300ETF基金净值（2019年12月至2020年6月）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
price.index=pd.DatetimeIndex(price.index) #将数据框的行索引转换为daatetime格式
price.head()
price.tail()
P0_call=price['看涨期权'].iloc[0] #策略构建日看涨期权收盘价
P0_put=price['看跌期权'].iloc[0] #策略构建日看跌期权收盘价
P0_ETF=price['沪深300ETF'].iloc[0] #策略构建日沪深300ETF基金净值
profit_call=N_call*N_underlying*(price['看涨期权']-P0_call) #看涨期权的期间收益
profit_put=N_put*N_underlying*(price['看跌期权']-P0_put) #看跌期权的期间收益
profit_ETF=N_underlying*(price['沪深300ETF']-P0_ETF) #沪深300ETF基金的期间收益
profit_covcall_long=-profit_ETF+profit_call #买入备兑看涨期权策略的期间收益
profit_covcall_short=-profit_covcall_long #卖出备兑看涨期权策略的期间收益
profit_protput_long=profit_ETF+profit_put #买入保护看跌期权策略的期间收益
profit_protput_short=-profit_protput_long #卖出保护看跌期权策略的期间收益
plt.figure(figsize=(9,9))
plt.subplot(2,1,1)
plt.plot(profit_covcall_long,'g-',label=u'买入备兑看涨期权策略',lw=2.0)
plt.plot(profit_covcall_short,'c-',label=u'卖出备兑看涨期权策略',lw=2.0)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.subplot(2,1,2)
plt.plot(profit_protput_long,'m-',label=u'买入保护看跌期权策略',lw=2.0)
plt.plot(profit_protput_short,'y-',label=u'卖出保护看跌期权策略',lw=2.0)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel(u'日期',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-8 运用看涨期权构建牛市价差策略
K1=4500
K2=5000
C1=474.4
C2=293.0
S0=4762.76
St=np.linspace(3500,6500,500)
N1=1 #较低行权价格的期权多头头寸数量
N2=1 #较高行权价格的期权空头头寸数量
M=100 #合约乘数是每点100元(后面案例会用到)
profit_C1_long=N1*M*(np.maximum(St-K1,0)-C1)
profit_C2_short=N2*M*(C2-np.maximum(St-K2,0))
profit_bullspread=profit_C1_long+profit_C2_short
plt.figure(figsize=(9,6))
plt.plot(St,profit_C1_long,'b--',label=u'较低行权价格沪深300股指认购期权多头',lw=2.5)
plt.plot(St,profit_C2_short,'g--',label=u'较高行权价格沪深300股指认购期权空头',lw=2.5)
plt.plot(St,profit_bullspread,'r-',label=u'牛市价差策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与牛市价差策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-9 运用看跌期权构建牛市价差策略
K1=4400
K2=5200
P1=290.0
P2=873.8
S0=4812.76
St=np.linspace(3000,6000,500)
profit_P1_long=N1*M*(np.maximum(K1-St,0)-P1) #期权到期日较低行权价格看跌期权多头头寸收益
profit_P2_short=N2*M*(P2-np.maximum(K2-St,0)) #期权到期日较高行权价格看跌期权空头头寸收益
profit_bullspread=profit_P1_long+profit_P2_short
plt.figure(figsize=(9,6))
plt.plot(St,profit_P1_long,'b--',label=u'较低行权价格沪深300股指认沽期权多头',lw=2.5)
plt.plot(St,profit_P2_short,'g--',label=u'较高行权价格沪深300股指认沽期权空头',lw=2.5)
plt.plot(St,profit_bullspread,'r-',label=u'牛市价差策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与牛市价差策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-10 用看跌期权构建熊市价差策略
K1=4500
K2=5400
P1=237.0
P2=818.2
S0=4844.27
St=np.linspace(3400,6400,500)
profit_P1_short=N1*M*(P1-np.maximum(K1-St,0))
profit_P2_long=N2*M*(np.maximum(K2-St,0)-P2)
profit_bearspread=profit_P1_short+profit_P2_long
plt.figure(figsize=(9,6))
plt.plot(St,profit_P1_short,'b--',label=u'较低行权价格沪深300股指认沽期权空头',lw=2.5)
plt.plot(St,profit_P2_long,'g--',label=u'较高行权价格沪深300股指认沽期权多头',lw=2.5)
plt.plot(St,profit_bearspread,'r-',label=u'熊市价差策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与熊市价差策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-11 用看涨期权构建熊市价差策略
K1=4300
K2=5200
C1=486.0
C2=152.4
S0=4694.39
St=np.linspace(3300,6200,500)
profit_C1_short=N1*M*(C1-np.maximum(St-K1,0))
profit_C2_long=N2*M*(np.maximum(St-K2,0)-C2)
profit_bearspread=profit_C1_short+profit_C2_long
plt.figure(figsize=(9,6))
plt.plot(St,profit_C1_short,'b--',label=u'较低行权价格沪深300股指认购期权空头',lw=2.5)
plt.plot(St,profit_C2_long,'g--',label=u'较高行权价格沪深300股指认购期权多头',lw=2.5)
plt.plot(St,profit_bearspread,'r-',label=u'熊市价差策略',lw=2.5)
plt.xlabel(u'沪深300股指',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与熊市价差策略价值的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-12 盒式价差策略
K1=4000
K2=4600
C1=161.6
C2=33.2
P1=285.4
P2=776.0
S0=4044.38
St=np.linspace(3000,5000,500)
profit_C1_long=N1*M*(np.maximum(St-K1,0)-C1)
profit_P1_short=N1*M*(P1-np.maximum(K1-St,0))
profit_C2_short=N2*M*(C2-np.maximum(St-K2,0))
profit_P2_long=N2*M*(np.maximum(K2-St,0)-P2)
profit_boxspread=profit_C1_long+profit_P1_short+profit_C2_short+profit_P2_long
plt.figure(figsize=(9,6))
plt.plot(St,profit_C1_long,'b--',label=u'较低行权价格沪深300股指认购期权多头',lw=2.5)
plt.plot(St,profit_P1_short,'g--',label=u'较低行权价格沪深300股指认沽期权空头',lw=2.5)
plt.plot(St,profit_C2_short,'c--',label=u'较高行权价格沪深300股指认购期权空头',lw=2.5)
plt.plot(St,profit_P2_long,'m--',label=u'较高行权价格沪深300股指认沽期权多头',lw=2.5)
plt.plot(St,profit_boxspread,'r-',label=u'盒式价差策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与盒式价差策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

shibor=0.02115 #策略构建日6个月期Shibor
tenor=0.5 #期权剩余期限
PV_boxspread=profit_boxspread[0]*np.exp(-shibor*tenor) #策略构建日该策略的收益现值
print('策略构建日（2020年6月18日）盒式价差策略收益',round(PV_boxspread,2))

# Example 13-13 用看涨期权构建蝶式价差策略
K1=4400
K2=4800
K3=5200
C1=571.6
C2=388.6
C3=255.0
S0=4842.12
St=np.linspace(3400,6200,500)
N1=1
N2=2
N3=1
profit_C1_long=N1*M*(np.maximum(St-K1,0)-C1)
profit_C2_short=N2*M*(C2-np.maximum(St-K2,0))
profit_C3_long=N3*M*(np.maximum(St-K3,0)-C3)
profit_buttpread=profit_C1_long+profit_C2_short+profit_C3_long
plt.figure(figsize=(9,6))
plt.plot(St,profit_C1_long,'b--',label=u'较低行权价格沪深300股指认购期权多头',lw=2.5)
plt.plot(St,profit_C2_short,'g--',label=u'中间行权价格沪深300股指认购期权空头',lw=2.5)
plt.plot(St,profit_C3_long,'c--',label=u'较高行权价格沪深300股指认购期权多头',lw=2.5)
plt.plot(St,profit_buttpread,'r-',label=u'蝶式价差策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与蝶式价差策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-14 用看跌期权构建蝶式价差策略
K1=4200
K2=4600
K3=5000
P1=264.2
P2=476.2
P3=748.2
S0=4581.98
St=np.linspace(3200,6000,500)
profit_P1_long=N1*M*(np.maximum(K1-St,0)-P1)
profit_P2_short=N2*M*(P2-np.maximum(K2-St,0))
profit_P3_long=N3*M*(np.maximum(K3-St,0)-P3)
profit_buttpread=profit_P1_long+profit_P2_short+profit_P3_long
plt.figure(figsize=(9,6))
plt.plot(St,profit_P1_long,'b--',label=u'较低行权价格沪深300股指认沽期权多头',lw=2.5)
plt.plot(St,profit_P2_short,'g--',label=u'中间行权价格沪深300股指认沽期权空头',lw=2.5)
plt.plot(St,profit_P3_long,'c--',label=u'较高行权价格沪深300股指认沽期权多头',lw=2.5)
plt.plot(St,profit_buttpread,'r-',label=u'蝶式价差策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与蝶式价差策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-15 用看涨期权构建日历价差策略
K_same=4600
C1=224.0
C2=348.0
S0=4584.59
St=np.linspace(3000,6000,500)
N1=1
N2=1
profit_C1_short=N1*M*(C1-np.maximum(St-K_same,0))
def BTM_Nstep(S,K,sigma,r,T,N,types): #Chapter 11 Example 11-14 前的自定义函数
    '''定义一个运用N步二叉树模型计算欧式期权价值的函数
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数。
    types: 代表期权类型，输入types='call'表示欧式看涨期权，输入其他则表示欧式看跌期权'''
    from math import factorial
    from numpy import exp, maximum, sqrt
    t=T/N #每一步步长期限（年）
    u=exp(sigma*sqrt(t)) #基础资产价格上涨时的变化比例
    d=1/u #基础资产价格下跌时的变化比例
    p=(exp(r*t)-d)/(u-d) #基础资产价格上涨的概率
    N_list=range(0,N+1) #创建从0到N的自然数数列
    A=[] #空列表，用于存储求和公式中各元素
    for j in N_list:
        C_Nj=maximum(S*u**j*d**(N-j)-K,0) #期权到期日某节点的期权价值
        Num=factorial(N)/((factorial(j))*factorial(N-j)) #到达到期日该节点的实现路径数量
        A.append(Num*p**j*(1-p)**(N-j)*C_Nj) #在列表尾部每次增加一个新元素
    call=exp(-r*T)*sum(A)
    put=call+K*exp(-r*T)-S
    if types=='call':
        value=call
    else:
        value=put
    return value
tenor=0.5 #策略到期日较远到期日期权的剩余期限
sigma_index=0.22 #沪深300指数的年化波动率
shibor=0.02922 #6个月期Shibor（无风险收益率）
step=120
Ct=np.ones_like(St) #创建存放较远到期日看涨期权价值的数组
for i in range(len(Ct)):
    Ct[i]=BTM_Nstep(St[i],K_same,sigma_index,shibor,tenor,step,'call')
profit_C2_long=N2*M*(Ct-C2)
profit_calendarpread=profit_C1_short+profit_C2_long
plt.figure(figsize=(9,6))
plt.plot(St,profit_C1_short,'b--',label=u'较近到期日沪深300股指认购期权空头',lw=2.5)
plt.plot(St,profit_C2_long,'g--',label=u'较远到期日沪深300股指认购期权多头',lw=2.5)
plt.plot(St,profit_calendarpread,'r-',label=u'日历价差策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与日历价差策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-16 用看跌期权构建日历价差策略
K_same=5000
P1=597.6
P2=746.4
St=np.linspace(3500,6500,500)
profit_P1_short=N1*M*(P1-np.maximum(K_same-St,0))
Pt=np.ones_like(St)
for i in range(len(Ct)):
    Pt[i]=BTM_Nstep(St[i],K_same,sigma_index,shibor,tenor,step,'put')
profit_P2_long=N2*M*(Pt-P2)
profit_calendarpread=profit_P1_short+profit_P2_long
plt.figure(figsize=(9,6))
plt.plot(St,profit_P1_short,'b--',label=u'较近到期日沪深300股指认沽期权空头',lw=2.5)
plt.plot(St,profit_P2_long,'g--',label=u'较远到期日沪深300股指认沽期权多头',lw=2.5)
plt.plot(St,profit_calendarpread,'r-',label=u'日历价差策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与日历价差策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-17 底部跨式组合策略=同行权价看涨、跌期权多头
K=4700
C=336.4
P=326.4
S0=4698.13
St=np.linspace(3000,6400,500)
N_C=1
N_P=1
profit_C_long=N_C*M*(np.maximum(St-K,0)-C)
profit_P_long=N_P*M*(np.maximum(K-St,0)-P)
profit_bottomstraddle=profit_C_long+profit_P_long
plt.figure(figsize=(9,6))
plt.plot(St,profit_C_long,'b--',label=u'沪深300股指认购期权多头',lw=2.5)
plt.plot(St,profit_P_long,'g--',label=u'沪深300股指认沽期权多头',lw=2.5)
plt.plot(St,profit_bottomstraddle,'r-',label=u'底部跨式组合策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与底部跨式组合策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-18 顶部跨式组合策略=同行权价看涨、跌期权空头
plt.figure(figsize=(9,6))
plt.plot(St,-profit_C_long,'b--',label=u'沪深300股指认购期权空头',lw=2.5)
plt.plot(St,-profit_P_long,'g--',label=u'沪深300股指认沽期权空头',lw=2.5)
plt.plot(St,-profit_bottomstraddle,'r-',label=u'顶部跨式组合策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与顶部跨式组合策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-19 序列组合策略与带式组合策略
K=4600
C=290.0
P=387.0
S0=4635.71
St=np.linspace(3000,6200,500)
N1=1
N2=2
profit_C_strip=N1*M*(np.maximum(St-K,0)-C)
profit_P_strip=N2*M*(np.maximum(K-St,0)-P)
profit_strip=profit_C_strip+profit_P_strip

profit_C_strap=N2*M*(np.maximum(St-K,0)-C)
profit_P_strap=N1*M*(np.maximum(K-St,0)-P)
profit_strap=profit_C_strap+profit_P_strap

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.plot(St,profit_C_strip,'b--',label=u'沪深300股指认购期权（1张）',lw=2.0)
plt.plot(St,profit_P_strip,'g--',label=u'沪深300股指认沽期权（2张）',lw=2.0)
plt.plot(St,profit_strip,'r-',label=u'序列组合策略',lw=2.0)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与序列组合策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.subplot(1,2,2)
plt.plot(St,profit_C_strap,'b--',label=u'沪深300股指认购期权（2张）',lw=2.0)
plt.plot(St,profit_P_strap,'g--',label=u'沪深300股指认沽期权（1张）',lw=2.0)
plt.plot(St,profit_strap,'r-',label=u'带式组合策略',lw=2.0)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与带式组合策略收益的关系图',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 13-20 买入宽跨式组合策略
K1=4200
K2=4900 #看涨期权行权价格
P=264.2
C=245.0
S0=4581.98
St=np.linspace(3000,6000,500)
N_P=1
N_C=1
profit_P_long=N_P*M*(np.maximum(K1-St,0)-P)
profit_C_long=N_C*M*(np.maximum(St-K2,0)-C)
profit_strangle=profit_P_long+profit_C_long
plt.figure(figsize=(9,6))
plt.plot(St,profit_P_long,'b--',label=u'沪深300股指认购期权多头',lw=2.5)
plt.plot(St,profit_C_long,'g--',label=u'沪深300股指认沽期权多头',lw=2.5)
plt.plot(St,profit_strangle,'r-',label=u'买入宽跨式组合策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与买入宽跨式组合策略收益的关系图',fontsize=13)
plt.legend(loc=9,fontsize=13) #图例放置在中上方
plt.grid()
plt.show()

# Example 13-21 买入宽跨式组合策略新案例(收窄俩行权价间差距、抬高期权策略构建成本)
K1=4500
K2=4700
P=417.4
C=305.0
profit_P_long=N_P*M*(np.maximum(K1-St,0)-P)
profit_C_long=N_C*M*(np.maximum(St-K2,0)-C)
profit_strangle=profit_P_long+profit_C_long
plt.figure(figsize=(9,6))
plt.plot(St,profit_P_long,'b--',label=u'沪深300股指认购期权多头',lw=2.5)
plt.plot(St,profit_C_long,'g--',label=u'沪深300股指认沽期权多头',lw=2.5)
plt.plot(St,profit_strangle,'r-',label=u'新的买入宽跨式组合策略',lw=2.5)
plt.xlabel(u'沪深300指数',fontsize=13)
plt.ylabel(u'收益金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数与新的买入宽跨式组合策略收益的关系图',fontsize=13)
plt.legend(loc=9,fontsize=13) #图例放置在中上方
plt.grid()
plt.show()

# Example 13-22 卖出宽跨式组合策略——英国联合里昂食品公司巨亏事件
K1=1.9 #较低期权行权价格
K2=2.0 #较高期权行权价格（看涨期权行权价格）
C=0.0104
P=0.0116
V1=K1-P-C #卖出宽跨式组合策略盈亏平衡的汇率临界值1
V2=K2+P+C #卖出宽跨式组合策略盈亏平衡的汇率临界值2 > V1
print('卖出宽跨式组合策略盈亏平衡的汇率临界值V1: ',round(V1,4))
print('卖出宽跨式组合策略盈亏平衡的汇率临界值V2: ',round(V2,4))
St=np.linspace(1.6,2.3,100)
profit_C_short=N_C*M*(C-np.maximum(St-K2,0))
profit_P_short=N_P*M*(P-np.maximum(K1-St,0))
profit_strangle_short=profit_C_short+profit_P_short
plt.figure(figsize=(9,6))
plt.plot(St,profit_C_short,'b--',label=u'英镑看涨期权空头',lw=2.5)
plt.plot(St,profit_P_short,'g--',label=u'英镑看跌期权空头',lw=2.5)
plt.plot(St,profit_strangle_short,'r-',label=u'卖出宽跨式组合策略',lw=2.5)
plt.xlabel(u'英镑兑美元汇率',fontsize=13)
plt.ylabel(u'盈亏金额',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'汇率与卖出宽跨式组合策略盈亏的关系图',fontsize=13)
plt.annotate(u'盈亏平衡的汇率临界值1',xy=(V1,0.0),xytext=(1.58,-0.01),arrowprops=dict(shrink=0.01),fontsize=13)
plt.annotate(u'盈亏平衡的汇率临界值2',xy=(V2,0.0),xytext=(2.08,-0.01),arrowprops=dict(shrink=0.01),fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

USD_GBP=pd.read_excel(r'E:\OneDrive\附件\数据\第13章\英镑兑美元的汇率（1991年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
USD_GBP.index=pd.DatetimeIndex(USD_GBP.index) #将数据框的行索引转换为datetime格式
plt.figure(figsize=(9,6))
plt.plot(USD_GBP,'b-',lw=2.0)
plt.xlabel(u'日期',fontsize=13)
plt.ylabel(u'汇率',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'英镑兑美元每日汇率走势图（1991年）',fontsize=13)
plt.annotate(u'1991年3月8日跌破策略盈亏平衡的汇率临界值',xy=('1991-03-08',1.87),xytext=('1991-04-30',1.88),arrowprops=dict(shrink=0.01),fontsize=13)
plt.grid()
plt.show()