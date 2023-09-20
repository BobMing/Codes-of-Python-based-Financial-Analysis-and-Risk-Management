# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:55:55 2023

@author: XIE Ming
@github: https://github.com/BobMing
@linkedIn: https://www.linkedin.com/in/tseming
@email: xieming_xm@163.com
"""

import numpy as np
import pandas as pd
#pd.set_option('display.unicode.ambiguous_as_wide',True) #处理数据的列名与其对应列数据无法对齐的情况
#pd.set_option('display.unicode.east_asian_width',True) #无法对齐主要是因为列名是中文
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import scipy.stats as st
x=0.95 #设定95%的置信水平
z=st.norm.ppf(q=1-x) #计算正态分布的分位数
x=np.linspace(-4,4,200) #创建从-4到4的等差数列（投资组合盈亏）
y=st.norm.pdf(x) #计算正态分布的概率密度函数值
x1=np.linspace(-4,z,100) #创建从-4到z的等差数列
y1=st.norm.pdf(x1)

plt.figure(figsize=(9,6))
plt.plot(x,y,'r-',lw=2.0)
plt.fill_between(x1,y1) #颜色填充
plt.xlabel(u'投资组合盈亏',fontsize=13)
plt.ylabel(u'概率密度',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,0.42)
plt.annotate('VaR',xy=(z-0.02,st.norm.pdf(z)+0.005),xytext=(-2.3,0.17),arrowprops=dict(shrink=0.01),fontsize=13) #绘制箭头
plt.title(u'假定投资组合盈亏服从正态分布的风险价值（VaR）',fontsize=13)
plt.grid()
plt.show()

def VaR_VCM(Value,Rp,Vp,X,N):
    '''定义一个运用方差-协方差法计算风险价值的函数
    Value: 代表投资组合的价值或市值。
    Rp: 代表投资组合日平均收益率。
    Vp: 代表投资组合收益率的日波动率。
    X: 代表置信水平。
    N: 代表持有期，用天数表示'''
    import scipy.stats as st
    from numpy import sqrt
    z=abs(st.norm.ppf(q=1-X)) #计算标准正态分布下1-X的分位数并取绝对值
    VaR_1day=Value*(z*Vp-Rp) #计算持有期为1天的风险价值
    VaR_Nday=sqrt(N)*VaR_1day #计算持有期为N天的风险价值
    return VaR_Nday
# Example 15-1 方差-协方差法的应用
# Step 1
price=pd.read_excel(r'E:\OneDrive\附件\数据\第15章\投资组合配置资产的每日价格（2018年至2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
price=price.dropna() #删除缺失值
price.index=pd.DatetimeIndex(price.index) #将数据框行索引转换为datetime格式
(price/price.iloc[0]).plot(figsize=(9,6),grid=True) #将首个交易日价格归一并且可视化
R=np.log(price/price.shift(1)) #计算对数收益率
R=R.dropna() #删除缺失值
R.describe() #显示表述性统计指标
R_mean=R.mean() #计算每个资产的日平均收益率
print('2018年至2020年期间日平均收益率\n',R_mean)
R_vol=R.std() #计算每个资产收益率的日波动率
print('2018年至2020年期间日波动率\n',R_vol)
R_cov=R.cov() #计算每个资产收益率之间的协方差矩阵
R_corr=R.corr() #计算每个资产收益率之间的相关系数矩阵
R_corr #输出相关系数矩阵
# Step 2 按照投资组合当前每个资产的权重计算投资组合的日平均收益率、日波动率
W=np.array([0.15,0.20,0.50,0.05,0.10]) #投资组合中各资产配置的权重
Rp_daily=np.sum(W*R_mean) #计算投资组合日平均收益率
print('2018年至2020年期间投资组合的日平均收益率',round(Rp_daily,6))
Vp_daily=np.sqrt(np.dot(W,np.dot(R_cov,W.T))) #计算投资组合日波动率，运用2.4.3节NumPy模块求矩阵之间内积的函数dot
print('2018年至2020年期间投资组合的日波动率',round(Vp_daily,6))
# Step 3 用自定义函数VaR_VCM计算方差-协方差法测算的风险价值
value_port=1e10 #投资组合的最新市值为100亿元
D1=1 #持有期为1天
D2=10 #持有期为10天
X1=0.95 #置信水平为95%
X2=0.99 #置信水平为99%
VaR95_1day_VCM=VaR_VCM(value_port,Rp_daily,Vp_daily,X1,D1)
VaR99_1day_VCM=VaR_VCM(value_port,Rp_daily,Vp_daily,X2,D1)
print('方差-协方差法计算持有期为1天、置信水平为95%的风险价值',round(VaR95_1day_VCM,2))
print('方差-协方差法计算持有期为1天、置信水平为99%的风险价值',round(VaR99_1day_VCM,2))
VaR95_10day_VCM=VaR_VCM(value_port,Rp_daily,Vp_daily,X1,D2)
VaR99_10day_VCM=VaR_VCM(value_port,Rp_daily,Vp_daily,X2,D2)
print('方差-协方差法计算持有期为10天、置信水平为95%的风险价值',round(VaR95_10day_VCM,2))
print('方差-协方差法计算持有期为10天、置信水平为99%的风险价值',round(VaR99_10day_VCM,2))

# Example 15-2 历史模拟法的运用
value_past=value_port*W #用投资组合最新市值和资产权重计算其中每个资产的最新市值
profit_past=np.dot(R,value_past) #2018年至2020年每个交易日投资组合模拟盈亏金额
profit_past=pd.DataFrame(data=profit_past,index=R.index,columns=['投资组合的模拟日收益']) #转换为数据框
profit_past.plot(figsize=(9,6),grid=True) #将投资组合的模拟日收益可视化

plt.figure(figsize=(9,6))
plt.hist(np.array(profit_past),bins=30,facecolor='y',edgecolor='k') #绘制投资组合的模拟日收益金额直方图并在输入时将数据框转换为数组
plt.xticks(fontsize=13)
plt.xlabel(u'投资组合的模拟日收益金额',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13)
plt.title(u'投资组合模拟日收益金额的直方图',fontsize=13)
plt.grid()
plt.show()
st.kstest(rvs=profit_past['投资组合的模拟日收益'],cdf='norm') #Kolmogorov-Smirnov检验
st.anderson(x=profit_past['投资组合的模拟日收益'],dist='norm') #Anderson-Darling检验
st.shapiro(profit_past['投资组合的模拟日收益']) #Shapiro-Wilk检验
st.normaltest(profit_past['投资组合的模拟日收益']) #一般的正态性检验

VaR95_1day_history=np.abs(profit_past.quantile(q=1-X1)) #结果是Series对象
VaR99_1day_history=np.abs(profit_past.quantile(q=1-X2))
VaR95_1day_history=float(VaR95_1day_history) #转换为浮点型数据
VaR99_1day_history=float(VaR99_1day_history)
print('历史模拟法计算持有期为1天、置信水平为95%的风险价值',round(VaR95_1day_history,2))
print('历史模拟法计算持有期为1天、置信水平为99%的风险价值',round(VaR99_1day_history,2))
VaR95_10day_history=np.sqrt(D2)*VaR95_1day_history
VaR99_10day_history=np.sqrt(D2)*VaR99_1day_history
print('历史模拟法计算持有期为10天、置信水平为95%的风险价值',round(VaR95_10day_history,2))
print('历史模拟法计算持有期为10天、置信水平为99%的风险价值',round(VaR99_10day_history,2))

# Example 15-3 蒙特卡罗模拟法的运用
# Step 1: 输入相关参数，运用式（15-10）模拟得到投资组合中每个资产在下个交易日的价格
import numpy.random as npr
I=100000 #模拟的次数
n=8 #学生t分布的自由度
epsilon=npr.standard_t(df=n,size=I) #从学生t分布进行抽样
P1=price.iloc[-1,0] #投资组合中第1个资产（贵州茅台）最新收盘价
P2=price.iloc[-1,1] #投资组合中第2个资产（交通银行）最新收盘价
P3=price.iloc[-1,2] #投资组合中第3个资产（嘉实增强信用基金）最新基金净值
P4=price.iloc[-1,3] #投资组合中第4个资产（华夏恒生ETF基金）最新基金净值
P5=price.iloc[-1,4] #投资组合中第5个资产（博士标普500ETF基金）最新基金净值
R_mean=R.mean()*252 #每个资产的年化平均收益率
R_vol=R.std()*np.sqrt(252) #每个资产收益率的年化波动率
dt=1/252 #设定步长为一个交易日
P1_new=P1*np.exp((R_mean[0]-R_vol[0]**2/2)*dt+R_vol[0]*epsilon*np.sqrt(dt))
P2_new=P2*np.exp((R_mean[1]-R_vol[1]**2/2)*dt+R_vol[1]*epsilon*np.sqrt(dt))
P3_new=P3*np.exp((R_mean[2]-R_vol[2]**2/2)*dt+R_vol[2]*epsilon*np.sqrt(dt))
P4_new=P4*np.exp((R_mean[3]-R_vol[3]**2/2)*dt+R_vol[3]*epsilon*np.sqrt(dt))
P5_new=P5*np.exp((R_mean[4]-R_vol[4]**2/2)*dt+R_vol[4]*epsilon*np.sqrt(dt))
# Step 2: 模拟单个资产和整个投资组合在下个交易日的收益并可视化
profit1=(P1_new/P1-1)*value_port*W[0]
profit2=(P2_new/P2-1)*value_port*W[1]
profit3=(P3_new/P3-1)*value_port*W[2]
profit4=(P4_new/P4-1)*value_port*W[3]
profit5=(P5_new/P5-1)*value_port*W[-1]
profit_port=profit1+profit2+profit3+profit4+profit5
plt.figure(figsize=(9,6))
plt.hist(profit_port,bins=50,facecolor='y',edgecolor='k') #投资组合模拟日收益金额的直方图
plt.xticks(fontsize=13)
plt.xlabel(u'投资组合模拟的日收益金额',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13)
plt.title(u'通过蒙特卡罗模拟（服从学生t分布）得到投资组合日收益金额的直方图',fontsize=13)
plt.grid()
plt.show()
# Step 3：假定资产收益率服从学生t分布情况下计算投资组合的风险价值
VaR95_1day_MCst=np.abs(np.percentile(a=profit_port,q=(1-X1)*100))
VaR99_1day_MCst=np.abs(np.percentile(a=profit_port,q=(1-X2)*100))
print('蒙特卡罗模拟法（服从学生t分布）计算持有期为1天、置信水平为95%的风险价值',round(VaR95_1day_MCst,2))
print('蒙特卡罗模拟法（服从学生t分布）计算持有期为1天、置信水平为99%的风险价值',round(VaR99_1day_MCst,2))
VaR95_10day_MCst=np.sqrt(D2)*VaR95_1day_MCst
VaR99_10day_MCst=np.sqrt(D2)*VaR99_1day_MCst
print('蒙特卡罗模拟法（服从学生t分布）计算持有期为10天、置信水平为95%的风险价值',round(VaR95_10day_MCst,2))
print('蒙特卡罗模拟法（服从学生t分布）计算持有期为10天、置信水平为99%的风险价值',round(VaR99_10day_MCst,2))
# Step 4: 假定资产收益率服从正态分布，用蒙特卡洛模拟法计算投资组合的风险价值，作为比较
P=np.array(price.iloc[-1]) #单个资产的最新收盘价或净值（数组格式）
epsilon_norm=npr.standard_normal(I) #从正态分布中抽取样本
P_new=np.zeros(shape=(I,len(R_mean))) #创建存放模拟下个交易日单一资产价格的初始数组
for i in range(len(R_mean)):
    P_new[:,i]=P[i]*np.exp((R_mean[i]-R_vol[i]**2/2)*dt+R_vol[i]*epsilon_norm*np.sqrt(dt)) #依次模拟投资组合每个资产下个交易日的收盘价
profit_port_norm=(np.dot(P_new/P-1,W))*value_port #投资组合下个交易日的收益
plt.figure(figsize=(9,6))
plt.hist(profit_port_norm,bins=30,facecolor='y',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'投资组合模拟的日收益金额',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13)
plt.title(u'通过蒙特卡罗模拟（服从正态分布）得到投资组合日收益金额的直方图',fontsize=13)
plt.grid()
plt.show()
VaR95_1day_MCnorm=np.abs(np.percentile(a=profit_port_norm,q=(1-X1)*100))
VaR99_1day_MCnorm=np.abs(np.percentile(a=profit_port_norm,q=(1-X2)*100))
print('蒙特卡罗模拟法（服从正态分布）计算持有期为1天、置信水平为95%的风险价值',round(VaR95_1day_MCnorm,2))
print('蒙特卡罗模拟法（服从正态分布）计算持有期为1天、置信水平为99%的风险价值',round(VaR99_1day_MCnorm,2))
VaR95_10day_MCnorm=np.sqrt(D2)*VaR95_1day_MCnorm
VaR99_10day_MCnorm=np.sqrt(D2)*VaR99_1day_MCnorm
print('蒙特卡罗模拟法（服从正态分布）计算持有期为10天、置信水平为95%的风险价值',round(VaR95_10day_MCnorm,2))
print('蒙特卡罗模拟法（服从正态分布）计算持有期为10天、置信水平为99%的风险价值',round(VaR99_10day_MCnorm,2))

# Example 15-4 回溯检验
# Step 1：据历史模拟法计算得出的18至20年期间投资组合日收益金额数据，依次生成每一年投资组合日收益金额的时间序列，且将每年的投资组合日收益与风险价值所对应的亏损可视化
profit_2018=profit_past.loc['2018-01-01':'2018-12-31']
profit_2019=profit_past.loc['2019-01-01':'2019-12-31']
profit_2020=profit_past.loc['2020-01-01':'2020-12-31']
VaR_2018_neg=-VaR95_1day_VCM*np.ones_like(profit_2018)
VaR_2019_neg=-VaR95_1day_VCM*np.ones_like(profit_2019)
VaR_2020_neg=-VaR95_1day_VCM*np.ones_like(profit_2020)
VaR_2018_neg=pd.DataFrame(data=VaR_2018_neg,index=profit_2018.index) #2018年风险价值对应亏损的时间序列
VaR_2019_neg=pd.DataFrame(data=VaR_2019_neg,index=profit_2019.index) #2019年风险价值对应亏损的时间序列
VaR_2020_neg=pd.DataFrame(data=VaR_2020_neg,index=profit_2020.index) #2020年风险价值对应亏损的时间序列
plt.figure(figsize=(9,12))
plt.subplot(3,1,1)
plt.plot(profit_2018,'b-',label=u'2018年投资组合日收益')
plt.plot(VaR_2018_neg,'r-',label=u'风险价值对应的亏损',lw=2.0)
plt.ylabel(u'收益')
plt.legend(fontsize=12)
plt.grid()
plt.subplot(3,1,2)
plt.plot(profit_2019,'b-',label=u'2019年投资组合日收益')
plt.plot(VaR_2019_neg,'r-',label=u'风险价值对应的亏损',lw=2.0)
plt.ylabel(u'收益')
plt.legend(fontsize=12)
plt.grid()
plt.subplot(3,1,3)
plt.plot(profit_2020,'b-',label='2020年投资组合日收益')
plt.plot(VaR_2020_neg,'r-',label=u'风险价值对应的亏损',lw=2.0)
plt.xlabel(u'日期')
plt.ylabel(u'收益')
plt.legend(fontsize=12)
plt.grid()
plt.show()
# Step 2：计算18至20年期间，每年的交易天数、每年内投资组合日亏损超出风险价值对应亏损的具体天数及占当年交易天数的比重
days_2018=len(profit_2018)
days_2019=len(profit_2019)
days_2020=len(profit_2020)
print('2018年的全部交易天数',days_2018)
print('2019年的全部交易天数',days_2019)
print('2020年的全部交易天数',days_2020)
dayexcept_2018=len(profit_2018[profit_2018['投资组合的模拟日收益']<-VaR95_1day_VCM])
dayexcept_2019=len(profit_2019[profit_2019['投资组合的模拟日收益']<-VaR95_1day_VCM])
dayexcept_2020=len(profit_2020[profit_2020['投资组合的模拟日收益']<-VaR95_1day_VCM])
print('2018年超过风险价值对应亏损的天数',dayexcept_2018)
print('2019年超过风险价值对应亏损的天数',dayexcept_2019)
print('2020年超过风险价值对应亏损的天数',dayexcept_2020)
ratio_2018=dayexcept_2018/days_2018
ratio_2019=dayexcept_2019/days_2019
ratio_2020=dayexcept_2020/days_2020
print('2018年超过风险价值对应亏损的天数占全年交易天数的比例',round(ratio_2018,4))
print('2019年超过风险价值对应亏损的天数占全年交易天数的比例',round(ratio_2019,4))
print('2020年超过风险价值对应亏损的天数占全年交易天数的比例',round(ratio_2020,4))

# Example 15-5 压力风险价值
# Step 1: 计算压力期间投资组合的日收益时间序列并可视化
price_stress=pd.read_excel(r'E:\OneDrive\附件\数据\第15章\投资组合配置资产压力期间的每日价格.xlsx',sheet_name='Sheet1',header=0,index_col=0)
price_stress=price_stress.dropna() #删除缺失值
price_stress.index=pd.DatetimeIndex(price_stress.index) #将数据框行索引转换为datetime格式
R_stress=np.log(price_stress/price_stress.shift(1)) #计算对数收益率
R_stress=R_stress.dropna()
profit_stress=np.dot(R_stress,value_past) #压力期间投资组合的日收益金额
profit_stress=pd.DataFrame(data=profit_stress,index=R_stress.index,columns=['投资组合的模拟日收益']) #转换为数据框
profit_stress.describe()
profit_zero=np.zeros_like(profit_stress) #创建压力期间收益为0的书组
profit_zero=pd.DataFrame(data=profit_zero,index=profit_stress.index)
plt.figure(figsize=(9,6))
plt.plot(profit_stress,'b-',label=u'压力期间投资组合的日收益')
plt.plot(profit_zero,'r-',label=u'收益等于0',lw=2.5)
plt.xlabel(u'日期',fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel(u'收益',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'压力期间投资组合的收益表现情况',fontsize=13)
plt.legend(fontsize=12)
plt.grid()
plt.show()
# Step 2: 根据压力期间投资组合的日收益时间序列，计算压力风险价值
SVaR95_1day=np.abs(np.percentile(a=profit_stress,q=(1-X1)*100))
SVaR99_1day=np.abs(np.percentile(a=profit_stress,q=(1-X2)*100))
print('持有期为1天、置信水平为95%的压力风险价值',round(SVaR95_1day,2))
print('持有期为1天、置信水平为99%的压力风险价值',round(SVaR99_1day,2))
SVaR95_10day=np.sqrt(D2)*SVaR95_1day
SVaR99_10day=np.sqrt(D2)*SVaR99_1day
print('持有期为10天、置信水平为95%的压力风险价值',round(SVaR95_10day,2))
print('持有期为10天、置信水平为99%的压力风险价值',round(SVaR99_10day,2))

def CVaR(T,X,L,R,Lambda,rou):
    '''定义一个计算投资组合信用风险价值的函数
    T: 代表信用风险价值的持有期，单位是年。
    X: 代表信用风险价值的置信水平。
    L: 代表投资组合的总金额。
    R: 代表投资组合中每个主体的违约回收率且每个主体均相同。
    Lambda: 代表投资组合中每个主体连续复利的年化违约概率且每个主体均相同。
    rou: 代表投资组合中任意两个主体之间的违约相关系数且均相同'''
    from scipy.stats import norm
    from numpy import exp
    C=1-exp(-Lambda*T) #计算每个主体的累积违约概率
    V=norm.cdf((norm.ppf(C)+rou**0.5*norm.ppf(X))/(1-rou)**0.5) #计算阈值V(T,X)
    VaR=L*(1-R)*V #计算信用风险价值
    return VaR
# Example 15-6 测度信用风险价值
# Step 1: 输入相关参数并运用自定义函数CVaR，计算持有期为1年、置信水平为99.9%的信贷资产组合信用风险价值
tenor=1 #信用风险价值的持有期（年）
prob=0.999 #信用风险价值的置信水平
par=2e11 #投资组合的总金额
recovery=0.5 #每个借款主体的违约回收率
PD=0.015 #每个借款主体的违约概率
corr=0.2 #任意两个借款主体之间的违约相关系数
credit_VaR=CVaR(tenor,prob,par,recovery,PD,corr)
print('持有期为1年、置信水平为99.9%的信贷资产组合信用风险价值（亿元）',round(credit_VaR/1e8,4))
print('信用风险价值占整个信贷资产组合总金额的比重',round(credit_VaR/par,6))
# Step 2: 置信水平取[80%,99.9%]区间的等差数列时，计算相应的信用风险价值，且将置信水平与信用风险价值的关系可视化
prob_list=np.linspace(0.8,0.999,200)
CVaR_list1=CVaR(tenor,prob_list,par,recovery,PD,corr)
plt.figure(figsize=(9,6))
plt.plot(prob_list,CVaR_list1,'r-',lw=2.5)
plt.xlabel(u'置信水平',fontsize=13)
plt.ylabel(u'信用风险价值',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'置信水平与信用风险价值的关系图',fontsize=13)
plt.grid()
plt.show()
# Step 3: 违约概率取[0.5%,5%]区间的等差数列时，计算相应的信用风险价值，且将违约概率与信用风险价值的关系可视化
PD_list=np.linspace(0.005,0.05,200)
CVaR_list2=CVaR(tenor,prob,par,recovery,PD_list,corr)
plt.figure(figsize=(9,6))
plt.plot(PD_list,CVaR_list2,'m-',lw=2.5)
plt.xlabel(u'违约概率',fontsize=13)
plt.ylabel(u'信用风险价值',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'违约概率与信用风险价值的关系图',fontsize=13)
plt.grid()
plt.show()
# Step 4: 违约相关系数取[0.1,0.6]区间的等差数列时，计算相应的信用风险价值，且将违约相关系数与信用风险价值的关系可视化
corr_list=np.linspace(0.1,0.6,200)
CVaR_list3=CVaR(tenor,prob,par,recovery,PD,corr_list)
plt.figure(figsize=(9,6))
plt.plot(corr_list,CVaR_list3,'b-',lw=2.5)
plt.xlabel(u'违约相关系数',fontsize=13)
plt.ylabel(u'信用风险价值',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'违约相关系数与信用风险价值的关系图',fontsize=13)
plt.grid()
plt.show()