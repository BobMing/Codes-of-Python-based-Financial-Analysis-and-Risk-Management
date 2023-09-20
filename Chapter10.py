# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:21:06 2023

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

data_AU2008=pd.read_excel(io=r'E:\OneDrive\附件\数据\第10章\黄金期货AU2008合约.xlsx',sheet_name='Sheet1',header=0,index_col=0)
data_AU2008.plot(figsize=(10,9),subplots=True,layout=(2,2),grid=True,fontsize=13)
plt.subplot(2,2,1)
plt.ylabel(u'金额或数量',fontsize=11,loc='bottom')
plt.subplot(2,2,3)
plt.xticks(rotation=15)
plt.subplot(2,2,4)
plt.xticks(rotation=15)

data_IF2009=pd.read_excel('E:\OneDrive\附件\数据\第10章\沪深300指数期货IF2009合约.xlsx',sheet_name='Sheet1',header=0,index_col=0)
data_IF2009.plot(figsize=(10,9),subplots=True,layout=(2,2),grid=True,fontsize=13)
plt.subplot(2,2,1)
plt.ylabel(u'金额或数量',fontsize=11,loc='bottom')
plt.subplot(2,2,3)
plt.xticks(rotation=30)
plt.subplot(2,2,4)
plt.xticks(rotation=30)

data_T2009=pd.read_excel(r'E:\OneDrive\附件\数据\第10章\10年期国债期货T2009合约.xlsx',sheet_name='Sheet1',header=0,index_col=0)
data_T2009.plot(figsize=(10,9),subplots=True,layout=(2,2),grid=True,fontsize=13)
plt.subplot(2,2,1)
plt.ylabel(u'金额或数量',fontsize=11,loc='bottom')
plt.subplot(2,2,3)
plt.xticks(rotation=30)
plt.subplot(2,2,4)
plt.xticks(rotation=30)

def price_futures(S,r,y,u,c,T):
    '''定义一个计算期货价格的函数
    S: 代表当前的期货价格。
    r: 代表无风险利率（连续复利）。
    y: 代表现货的便利收益率（连续复利）。
    u: 代表现货的期间收益率（连续复利）。
    c: 代表在期货到期日（交割日）以现金形式支付的年化仓储费用。
    T: 代表期货的剩余期限（年）'''
    from numpy import exp
    c_pv=c*exp(-r*T) #计算年化仓储费用的贴现值
    F=(S+c_pv*T)*exp((r-y-u)*T)
    return F
# Example 10-1
spot=400.53
R_riskfree=0.02438
Y_conv=0.002
R_lease=0.005
C_storage=0.438
tenor=9/12
price_AU2104=price_futures(S=spot,r=R_riskfree,y=Y_conv,u=R_lease,c=C_storage,T=tenor)
print('2020年7月15日黄金期货AU2104合约的理论价格',round(price_AU2104,2))

# Example 10-2 敏感性分析
R_riskfree_list=np.linspace(0.02,0.03)
futures_list1=price_futures(S=spot,r=R_riskfree_list,y=Y_conv,u=R_lease,c=C_storage,T=tenor)
Y_conv_list=np.linspace(0.001,0.004)
futures_list2=price_futures(S=spot,r=R_riskfree,y=Y_conv_list,u=R_lease,c=C_storage,T=tenor)
R_lease_list=np.linspace(0.002,0.008)
futures_list3=price_futures(S=spot,r=R_riskfree,y=Y_conv,u=R_lease_list,c=C_storage,T=tenor)
C_storage_list=np.linspace(0.3,1.2)
futures_list4=price_futures(S=spot,r=R_riskfree,y=Y_conv,u=R_lease,c=C_storage_list,T=tenor)
plt.figure(figsize=(10,11))
plt.subplot(2,2,1)
plt.plot(R_riskfree_list,futures_list1,'r-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'无风险利率',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'期货价格',fontsize=13,rotation=90)
plt.grid()
plt.subplot(2,2,2,sharey=plt.subplot(2,2,1))
plt.plot(Y_conv_list,futures_list2,'b-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'便利收益率',fontsize=13)
plt.yticks(fontsize=13)
plt.grid()
plt.subplot(2,2,3,sharey=plt.subplot(2,2,1))
plt.plot(R_lease_list,futures_list3,'c-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'黄金租借利率（期间收益率）',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'期货价格',fontsize=13,rotation=90)
plt.grid()
plt.subplot(2,2,4,sharey=plt.subplot(2,2,1))
plt.plot(C_storage_list,futures_list4,'m-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'仓储费用',fontsize=13)
plt.yticks(fontsize=13)
plt.grid()
plt.show()

# Example 10-3 期货价格的收敛性
price_AU2004_AU9995=pd.read_excel(io=r'E:\OneDrive\附件\数据\第10章\黄金期货AU2004、AU2010合约以及现货价格.xlsx',sheet_name='Sheet1',header=0,index_col=0)
price_AU2004_AU9995.plot(figsize=(9,6),grid=True,fontsize=13,title=u'期货价格收敛性（以黄金期货AU2004合约为例）')
plt.ylabel(u'金额',fontsize=11)
price_AU2010_AU9995=pd.read_excel(io=r'E:\OneDrive\附件\数据\第10章\黄金期货AU2004、AU2010合约以及现货价格.xlsx',sheet_name='Sheet2',header=0,index_col=0)
price_AU2010_AU9995.plot(figsize=(9,6),grid=True,fontsize=13,title=u'期货价格收敛性（以黄金期货AU2010合约为例）')
plt.ylabel(u'金额',fontsize=11)

# Example 10-4 short hedge
fund=1.2e8 #购买基金时的基金市值
index=4000 #购买基金时沪深300指数的点位
N=100 #持有沪深300股指期货合约空头数量
M=300 #沪深300股指期货合约乘数
index_list=np.linspace(3500,4500,200) #创建沪深300指数不同点位的数组
profit_spot=(index_list-index)*fund/index #现货投资的收益
profit_future=-(index_list-index)*N*M #期货合约的收益
profit_portfolio=profit_spot+profit_future #整个投资组合的收益
plt.figure(figsize=(9,6))
plt.plot(index_list,profit_spot,label=u'沪深300指数ETF基金',lw=2.5)
plt.plot(index_list,profit_future,label=u'沪深300指数期货合约',lw=2.5)
plt.plot(index_list,profit_portfolio,label=u'套期保值的投资组合',lw=2.5)
plt.xlabel(u'沪深300指数点位',fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel(u'盈亏',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'空头套期保值的盈亏情况')
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 10-6 E基金公司20年6月19日~7月17日每个交易日期货合约收益、累积收益、保证金余额（不考虑保证金追加）
price_IF2007=pd.read_excel(r'E:\Onedrive\附件\数据\第10章\沪深300股指期货2007合约结算价（2020年6月19日至7月17日）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
margin0=1.8e7 #初始保证金
N_short=100 #合约空头数量
P0=4000 #成交价格
M=300 #合约乘数
profit_sum_IF2007=-N_short*M*(price_IF2007-P0) #计算期货合约累积盈亏
profit_sum_IF2007=profit_sum_IF2007.rename(columns=({'IF2007 合约结算价':'合约累积盈亏'})) #变更列名
profit_daily_IF2007=profit_sum_IF2007-profit_sum_IF2007.shift(1) #计算期货合约每日的盈亏
profit_daily_IF2007.iloc[0]=profit_sum_IF2007.iloc[0] #首个交易日当日盈亏等于当天累积盈亏
profit_daily_IF2007=profit_daily_IF2007.rename(columns={'合约累积盈亏':'合约当日盈亏'}) #变更列名
margin_daily_IF2007=profit_sum_IF2007+margin0 #计算每日期货合约保证金余额（不考虑追加保证金）
margin_daily_IF2007=margin_daily_IF2007.rename(colmuns={'合约累积盈亏':'保证金余额'})
data_IF2007=pd.concat([profit_daily_IF2007,profit_sum_IF2007,margin_daily_IF2007],axis=1) #将3个数据框按列拼接
data_IF2007

# Example 10-7 basis risk
data_price=pd.read_excel(r'E:\Onedrive\附件\数据\第10章\上证50股指2009期货合约结算价和上证50指数收盘价.xlsx',sheet_name='Sheet1',header=0,index_col=0)
data_price.index=pd.DatetimeIndex(data_price.index) #将数据框的行索引转换为datetime格式
data_price.index
data_price.columns
basis=data_price['IH2009期货合约结算价']-data_price['上证50指数收盘价']
basis.describe()
zero_basis=np.zeros_like(basis)
zero_basis=pd.DataFrame(zero_basis,index=basis.index) #创建基差等于0的时间序列
plt.figure(figsize=(9,6))
plt.plot(basis,'b-',label=u'基差',lw=3.0)
plt.plot(zero_basis,'r-',label=u'基差等于0',lw=3.0)
plt.xlabel(u'日期',fontsize=13)
plt.ylabel(u'基差',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'上证50股指期货IH2009合约的基差趋势图',fontsize=13)
plt.legend(fontsize=13,loc=9)
plt.grid()
plt.show()

# Example 10-8 最优套保比率
# Step 1: 导入基金净值、期货合约收盘价数据，计算相应日收益率
fund_future=pd.read_excel(r'E:\Onedrive\附件\数据\第10章\上证180指数基金净值和三只A股股指期货合约收盘价数据.xlsx',sheet_name='Sheet1',header=0,index_col=0)
fund_future.columns #查看数据框列名
R_fund=np.log(fund_future['上证180指数基金']/fund_future['上证180指数基金'].shift(1)) #计算基金日收益率
R_fund=R_fund.dropna() #删除缺失值
R_IH2009=np.log(fund_future['IH2009合约']/fund_future['IH2009合约'].shift(1))
R_IH2009=R_IH2009.dropna()
R_IF2009=np.log(fund_future['IF2009合约']/fund_future['IF2009合约'].shift(1))
R_IF2009=R_IF2009.dropna()
R_IC2009=np.log(fund_future['IC2009合约']/fund_future['IC2009合约'].shift(1))
R_IC2009=R_IC2009.dropna()
# Step 2: 建立以基金日收益率为被解释变量、以上证50股指期货IH2009合约日收益率为解释变量的线性回归模型
import statsmodels.api as sm
R_IH2009_addcons=sm.add_constant(R_IH2009) #生成增加常数项的时间序列
model_fund_IH2009=sm.OLS(R_fund, R_IH2009_addcons).fit() #构建上证50股指期货IH2009合约日收益率解释基金日收益率的线性回归模型
model_fund_IH2009.summary() #输出线性回归模型的结果
# Step 3: 建立以基金日收益率为被解释变量、以沪深300股指期货IF2009合约日收益率为解释变量的线性回归模型
R_IF2009_addcons=sm.add_constant(R_IF2009)
model_fund_IF2009=sm.OLS(R_fund, R_IF2009_addcons).fit()
model_fund_IF2009.summary()
# Step 4: 建立以基金日收益率为被解释变量、以中证500股指期货IC2009合约日收益率为解释变量的线性回归模型
R_IC2009_addcons=sm.add_constant(R_IC2009)
model_fund_IC2009=sm.OLS(R_fund, R_IC2009_addcons).fit()
model_fund_IC2009.summary()
# 综合以上3个线性回归模型的结果，以沪深300股指期货IF2009合约的日收益率作解释变量的线性回归模型判定系数R²最高，达0.928（IH2009的为0.922，IC2009的为0.795），因此选择沪深300股指期货IF2009合约作为套期保值的期货合约
# Step 5: 将最终拟合得到的最优套保比率通过可视化方式展示
model_fund_IF2009.params #输出线性回归的常数项和斜率（贝塔值）
cons=model_fund_IF2009.params[0]
beta=model_fund_IF2009.params[1]
plt.figure(figsize=(9,6))
plt.scatter(R_IF2009,R_fund,marker='o') #绘制散点图，先横后纵(坐标)
plt.plot(R_IF2009,cons+beta*R_IF2009,'r-',lw=2.5) #绘制拟合的直线
plt.xlabel(u'沪深300指数期货IF2009合约',fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel(u'上证180股指基金',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数期货IF2009合约与上证180指数基金的日收益率散点图',fontsize=13)
plt.grid()
plt.show()

def N_future(h,Q_A,Q_F):
    '''定义一个计算套期保值的最优合约数量的函数
    h: 代表最优套保比率。
    Q_A: 代表被套期保值资产的数量/金额。
    Q_F: 代表1份期货合约的规模/金额'''
    N=h*Q_A/Q_F
    return N
# Example 10-9
share_fund=5e7 #购买上证180指数基金的份数
price_fund=4.058 #2020年10月12日基金收盘净值
value_fund=share_fund*price_fund #基金的市值
price_IF2011=4775.6 #2020年11月12日期货合约结算价
M=300 #期货合约乘数
value_IF2011=price_IF2011*M #（1份）期货合约价值
h_IF2011=model_fund_IF2009.params[1] #IF2011合约的最优套保比率
N_IF2011=N_future(h=h_IF2011,Q_A=value_fund,Q_F=value_IF2011)
print('用于套期保值的沪深300股指期货IF2011合约数量（张）',round(N_IF2011,0))

# Example 10-10 套期保值整体投资组合的动态盈亏
N=117
fund_list=np.array([4.0580,4.0143,3.9089,4.0951]) #套期保值首日及其他3个交易日的基金净值
IF2011_list=np.array([4775.6,4758.0,4683.4,4942.4]) #套期保值首日及其他3个交易日的期货结算价
profit_list=share_fund*(fund_list[1:]-fund_list[0])-N*M*(IF2011_list[1:]-IF2011_list[0]) #计算3个交易日的套期保值整体投资组合的累积盈亏
print('2020年10月20日套期保值组合的累积盈亏',round(profit_list[0],2))
print('2020年10月30日套期保值组合的累积盈亏',round(profit_list[1],2))
print('2020年11月10日套期保值组合的累积盈亏',round(profit_list[-1],2))

# Example 10-11 滚动套期保值与移仓风险
def stack_roll(F_open,F_close,M,N,position):
    '''定义一个计算滚动套期保值期间期货合约盈亏的函数
    F_open: 代表期货合约开立时的期货价格，以数组格式输入。
    F_close: 代表期货合约平仓时的期货价格，以数组格式输入。
    M: 代表期货合约乘数。
    N: 代表持有期货合约的数量。
    position: 代表期货合约的头寸方向，输入position='long'表示多头头寸，输入其他则表示空头头寸'''
    if position=='long':
        profit_list=(F_close-F_open)*M*N #计算每次期货合约移仓的盈亏
    else:
        profit_list=(F_open-F_close)*M*N
    profit_sum=np.sum(profit_list)
    return profit_sum

# Example 10-12
price_open=np.array([3300.00,3790.60,3994.00,3389.00,3728.00,3918.20,3526.8])
price_close=np.array([3833.40,4081.29,3386.46,3740.14,3932.45,3624.55,4638.20])
M_future=300 #沪深300股指期货合约乘数
N_future=100 #持有沪深300股指期货合约的数量（空头）
profit_sum=stack_roll(F_open=price_open,F_close=price_close,M=M_future,N=N_future,position='short')
print('滚动套期保值期间期货移仓盈亏合计数',round(profit_sum,2))
profit_list=(price_open-price_close)*M_future*N_future #计算每次期货移仓的盈亏
profit_list=list(profit_list) #将数组格式转换为列表格式
profit_list.append(profit_sum) #在列表末尾新增期货合约移仓盈亏合计数
name=['IF1709合约','IF1803合约','IF1809合约','IF1903合约','IF1909合约','IF2003合约','IF2009合约','合计'] #创建期货合约名称的列表
plt.figure(figsize=(9,6))
plt.barh(y=name,width=profit_list,height=0.6,label=u'期货移仓的盈亏额') #绘制横向条形图
plt.xticks(fontsize=13)
plt.xlabel(u'盈亏额',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'滚动套期保值期间期货移仓的盈亏',fontsize=13)
plt.legend(loc=3,fontsize=13)
plt.grid()
plt.show()

import datetime as dt
def accrued_interest(par,c,m,t1,t2,t3,t4,rule):
    '''定义一个按照不同计息天数规则计算债券期间的应急利息的函数
    par: 代表债券的本金。
    c: 代表债券的票面利率。
    m: 代表每年票息的支付频次。
    t1: 代表非参考期间的起始日，以datetime模块的时间对象方式输入。
    t2: 代表非参考期间的到期日，输入方式同t1。
    t3: 代表参考期间的起始日，输入方式同t1。
    t4: 代表参考区间的到期日，输入方式同t1。
    rule: 选择计息天数规则，输入rule='actual/actual'表示“实际天数/实际天数” ，rule='actual/360'表示“实际天数/360”，输入其他则表示“实际天数/365”'''
    d1=(t2-t1).days #计算非参考期间的天数
    if rule=='actual/actual':
        d2=(t4-t3).days #计算参考期间的天数
        interest=(d1/d2)*par*c/m
    elif rule=='actual/360':
        interest=(d1/360)*par*c
    else:
        interest=(d1/365)*par*c
    return interest
# Example 10-13 国债计息天数规则
par_TB06=1e6
C_TB06=0.0268
m_TB06=2
t1_TB06=dt.datetime(2020,5,28)
t2_TB06=dt.datetime(2020,10,16)
t3_TB06=dt.datetime(2020,5,21)
t4_TB06=dt.datetime(2020,11,21)
R1_TB06=accrued_interest(par_TB06,c=C_TB06,m=m_TB06,t1=t1_TB06,t2=t2_TB06,t3=t3_TB06,t4=t4_TB06,rule='actual/actual')
R2_TB06=accrued_interest(par_TB06,c=C_TB06,m=m_TB06,t1=t1_TB06,t2=t2_TB06,t3=t3_TB06,t4=t4_TB06,rule='actual/360')
R3_TB06=accrued_interest(par_TB06,c=C_TB06,m=m_TB06,t1=t1_TB06,t2=t2_TB06,t3=t3_TB06,t4=t4_TB06,rule='actual/365')
print('按照“实际天数/实际天数”规则计算期间利息',round(R1_TB06,2))
print('按照“实际天数/360”规则计算期间利息',round(R2_TB06,2))
print('按照“实际天数/365”规则计算期间利息',round(R3_TB06,2))

# Example 10-14 国债的报价
t_begin=dt.datetime(2020,5,21)
t_mature=dt.datetime(2030,5,21)
t_pricing=dt.datetime(2020,8,6)
t_next1=dt.datetime(2020,11,21) #20附息国债06下一次付息日
N=((t_mature-t_pricing).days//365+1)*m_TB06 #剩余的票息支付次数(当中有闰年，必定不足10年)
tenor=(t_next1-t_pricing).days/365 #定价日距离下一次付息日的期限（年）
t_list=np.arange(N)/2+tenor #剩余每期票息支付日距离定价日的期限（年）
t_list #显示输出结果
bond_par=100
y_TB06=0.0301 #20附息国债06连续复利的到期收益率
def Bondprice_onediscount(C,M,m,y,t):
    '''定义一个基于单一贴现率计算债券价格的函数
    C: 代表债券的票面利率，若输入0则表示零息债券。
    M: 代表债券的本金（面值）。
    m: 代表债券票息每年支付的频次。
    y: 代表单一贴现率。
    t: 代表定价日至后续每一期票息支付日的期限长度，用数组格式输入；零息债券可直接输入数字。
    '''
    if C==0:
        price=np.exp(-y*t)*M #零息债券价格直接为本金连续复利贴现
    else:
        coupon=np.ones_like(t)*M*C/m #创建每一期票息金额的数组
        NPV_coupon=np.sum(coupon*np.exp(-y*t)) #计算每一期票息在定价日的现值之和
        NPV_par=M*np.exp(-y*t[-1]) #计算本金在定价日的现值
        price=NPV_coupon+NPV_par
    return price
dirty_price=Bondprice_onediscount(C=C_TB06, M=bond_par, m=m_TB06, y=y_TB06, t=t_list)
print('2020年8月6日20附息国债06的全价',round(dirty_price,4))
bond_interest=accrued_interest(par=bond_par,c=C_TB06,m=m_TB06,t1=t_begin,t2=t_pricing,t3=t_begin,t4=t_next1,rule='actual/actual')
print('2020年8月6日20附息国债06的应计利息金额',round(bond_interest,4))
clean_price=dirty_price-bond_interest
print('2020年8月6日20附息国债06的净价',round(clean_price,4))

def CF(x,n,c,m):
    '''定义一个计算可交割债券转换因子的函数
    在定义过程中，国债期货基础资产（合约标的）的票面利率直接取0.03
    x: 表示国债期货交割月至可交割债券下一付息月的月份数。
    n: 表示国债期货到期后可交割债券的剩余付息次数。
    c: 表示可交割债券的票面利率。
    m: 表示可交割债券每年的付息次数'''
    r=0.03
    A=1/pow(1+r/m,x*m/12) #式(10-17)方括号前面的因子式
    B=c/m+c/r+(1-c/r)/pow(1+r/m,n-1) #式(10-17)方括号里面的表达式
    D=c/m*(1-x*m/12) #式(10-17)方括号后面的因子式
    value=A*B-D
    return value
# Example 10-15 计算转换因子
t_settel1=dt.datetime(2020,12,16) #10年期国债期货T2012合约最后交割日
t_next2=dt.datetime(2021,5,21) #20附息国债06在期货交割日之后的下一个付息日
months=12+(t_next2.month-t_settel1.month) #交割月至下一个付息月的月份
N2=((t_mature-t_settel1).days//365)*m_TB06+1 #20附息国债06在期货交割后的剩余付息次数
CF_TB06=CF(x=months,n=N2,c=C_TB06,m=m_TB06)
print('10年期国债期货T2012合约可交割债券20附息国债06的转换因子',round(CF_TB06,4))

# Example 10-16
t_settle2=dt.datetime(2020,12,15) #国债期货第2个交割日
bond_interest2=accrued_interest(par=bond_par, c=C_TB06, m=m_TB06, t1=t_next1, t2=t_settle2, t3=t_next1, t4=t_next2, rule='actual/actual')
print('20附息国债06作为可交割债券的应计利息',round(bond_interest2,4))

def CTD_cost(price1, price2, CF, name):
    '''定义一个用于计算国债期货可交割债券的交割成本并找出最廉价交割债券的函数
    price1: 表示可交割债券的净价，用数组格式输入。
    price2: 表示国债期货的价格。
    CF: 表示可交割债券的转换因子，用数组格式输入。
    name: 表示可交割债券的名称，用数组格式输入'''
    cost=price1-price2*CF #计算可交割债券的交割成本
    cost=pd.DataFrame(data=cost,index=name,columns=['交割成本']) #转换为数据框
    CTD_bond=cost.idxmin() #找出最廉价交割债券
    CTD_bond=CTD_bond.rename(index={'交割成本':'最廉价交割债券'}) #更改索引名称
    return cost,CTD_bond #输出可交割债券的交割成本以及最廉价交割债券
# Example 10-17
price_3bond=np.array([94.9870,98.6951,96.1669]) #3只可交割债券的净价
price_T2012=97.225 #国债期货的结算价格
CF_3bond=np.array([0.9739,1.0101,0.9884]) #3只可交割债券的转换因子
name_3bond=np.array(['20附息国债06','19附息国债15','20抗疫国债04']) #3只可交割债券的名称 
result=CTD_cost(price1=price_3bond,price2=price_T2012,CF=CF_3bond,name=name_3bond) #计算结果
print(result[0]) #输出3只可交割债券的交割成本
print(result[-1]) #输出最廉价交割债券

def N_TBF(Pf, par, value, Df, Dp):
    '''定义一个计算基于久期套期保值的国债期货合约数量的函数
    Pf: 表示国债期货价格。
    par: 表示1手国债期货合约基础资产对应的国债面值。
    value: 表示被套期保值的投资组合当前市值。
    Df: 表示国债期货合约基础资产在套期保值到期日的麦考利久期。
    Dp: 表示被套期保值的投资组合在套期保值到期日的麦考利久期'''
    value_TBF=Pf*par/100 #计算1手国债期货合约的价格
    N=value*Dp/(value_TBF*Df) #计算国债期货合约数量
    return N
# Example 10-18
C_TB04=0.0286 #20抗疫国债04票面利率
y_TB04=0.0295 #20抗疫国债04到期收益率
m_TB04=2 #20抗疫国债04票面利率每年支付次数
par_TB04=100 #20抗疫国债04面值
t_T2009=dt.datetime(2020,9,11) #10年期国债期货T2009合约到期日（套期保值到期日）
t1_TB04=dt.datetime(2021,1,16) #20抗疫国债04下一次付息日
t2_TB04=dt.datetime(2030,7,16) #20抗疫国债04到期日
N_TB04=((t2_TB04-t_T2009).days//365+1)*m_TB04 #套期保值到期日之后20抗疫国债04剩余的票息支付次数（2029年9月11日到2030年7月16日间>0.5年，还有个付息日2030年1月16日，与期末合计2次）
tenor=(t1_TB04-t_T2009).days/365 #套期保值到期日距离20抗疫国债04下一次付息日的期限
t_list=np.arange(N_TB04)/m_TB04+tenor #套期保值到期日距离20抗疫国债04剩余现金流支付日的期限数组
def Mac_Duration(C,M,m,y,t):
    '''定义一个计算债券麦考利久期的函数
    C: 代表债券的票面利率。
    M: 代表债券的面值。
    m: 代表债券票息每年支付的频次。
    y: 代表债券的到期收益率（连续复利）。
    t: 代表定价日至后续每一期现金流支付日的期限长度，用数组格式输入；零息债券可直接输入数字'''
    if C==0:
        duration=t
    else:
        coupon=np.ones_like(t)*M/m*C
        NPV_coupon=np.sum(coupon*np.exp(-y*t))
        NPV_par=M*np.exp(-y*t[-1])
        Bond_value=NPV_coupon+NPV_par #计算定价日的债券价格
        cashflow=coupon #现金流数组初始设定为票息
        cashflow[-1]+=M #现金流数组最后元素调整加上本金
        weight=cashflow*np.exp(-y*t)/Bond_value #计算时间的权重
        duration=np.sum(weight*t)
    return duration
D_TB04=Mac_Duration(C=C_TB04, M=par_TB04, m=m_TB04, y=y_TB04, t=t_list) #计算套期保值到期日20抗疫国债04的麦考利久期
print('2020年9月11日（套期保值到期日）20抗疫国债04的麦考利久期',round(D_TB04,4))
par_T2009=1e6 #1手10年期国债期货T2009合约基础资产对应的国债面值
price_T2009=99.51 #10年期国债期货T2009合约在2020年8月17日的结算价
value_fund=1e9 #债券投资组合（债券基金）的市值
D_fund=8.68 #债券投资组合的麦考利久期
N_T2009=N_TBF(Pf=price_T2009, par=par_T2009, value=value_fund, Df=D_TB04, Dp=D_fund) #计算国债期货合约数量
print('用于对冲债券投资组合的10年期国债期货T2009合约数量',round(N_T2009,2))