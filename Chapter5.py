# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:42:24 2023

@author: XIE Ming
"""

import scipy
scipy.__version__

import numpy as np
import pandas as pd
pd.set_option('display.unicode.ambiguous_as_wide',True) #处理数据的列名与其对应列数据无法对齐的情况//实际作用不明，可以不写
pd.set_option('display.unicode.east_asian_width',True) #无法对齐主要是因为列名是中文//设置输出右对齐
import matplotlib.pyplot as plt

from pylab import mpl #从pylab导入子模块mpl
mpl.rcParams['font.sans-serif']=['FangSong'] #以仿宋字体显示中文
mpl.rcParams['axes.unicode_minus']=False #在图像中正常显示负号
from pandas.plotting import register_matplotlib_converters #导入注册日期时间转换器的函数
register_matplotlib_converters() #注册日期时间转换器

# Example 5-1 求积分，以标准正态分布的定积分为例
import scipy.integrate as sci

def f(x):
    equation=np.exp(-0.5*x**2)/pow(2*np.pi,0.5) #标准正态分布的概率密度函数公式
    return equation

# Table 5-2 integrate子模块中的积分函数与运用
sci.quad(func=f,a=-2.0,b=2.0) #自适应求积分
sci.fixed_quad(func=f,a=-2.0,b=2.0) #固定高斯求积分
sci.quadrature(func=f,a=-2.0,b=2.0) #自适应高斯求积分
sci.romberg(function=f,a=-2.0,b=2.0) #自适应龙贝格求积分，输出结果仅有积分值、不包含最大误差

# Example 5-2 插值法，以最常用的一维数据的插值运算为例
from scipy import interpolate
rates=np.array([0.011032,0.012465,0.013460,0.013328,0.016431,0.016716,0.021576]) #创建已有利率的数组
t=np.array([0.25,0.5,0.75,1.0,2.0,3.0,5.0]) #创建已有期限的数组
t_new=np.insert(t,6,4.0) #在t数组的3和5之间插入一个4
types=['nearest','zero','slinear','quadratic','cubic'] #最临近、阶梯（0阶条样曲线）、线性（1阶条样曲线）、2阶条样曲线、3阶条样曲线
plt.figure(figsize=(9,6))
for i in types:
    f=interpolate.interp1d(x=t,y=rates,kind=i)
    rates_new=f(t_new) #计算插值后的利率数组
    print(i,'4年期国债到期收益率',rates_new[-2])
    plt.plot(t_new,rates_new,'o')
    plt.plot(t_new,rates_new,'-',label=i)
    plt.xticks(fontsize=13)
    plt.xlabel(u'期限',fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel(u'收益率',fontsize=13)
    plt.legend(loc=0,fontsize=13)
    plt.grid()
plt.title(u'运用插值法之后的国债到期收益率',fontsize=13)
plt.show()

# Example 5-3 求解方程组
from scipy import linalg
R_stock=np.array([[-0.035099,-0.013892,0.005848,0.021242],[0.017230,0.024334,-0.002907,0.002133],[-0.003450,-0.033758,0.005831,-0.029803],[-0.024551,0.014622,0.005797,-0.002743]]) #创建股票日涨跌幅数组
R_portfolio=np.array([0.00191555,0.00757775,-0.01773255,-0.00040620]) #投资组合日收益率数组
name=np.array(['中国卫星','中国软件','中国银行','上汽集团']) #股票名称数组
weight=linalg.solve(a=R_stock,b=R_portfolio) #计算股票权重
for i in range(len(name)):
    print(name[i],round(weight[i],4))
    
import scipy.optimize as sco
def g(w):
    w1,w2,w3,w4 = w
    eq1=-0.035099*w1-0.013892*w2+0.005848*w3+0.021242*w4
    eq2=0.017230*w1+0.024334*w2-0.002907*w3+0.002133*w4
    eq3=-0.003450*w1-0.033758*w2+0.005831*w3-0.029803*w4
    eq4=-0.024551*w1+0.014622*w2+0.005797*w3-0.002743*w4
    return [eq1,eq2,eq3,eq4]
w0=[0.1,0.1,0.1,0.1] #初始猜测各股票权重
result=sco.fsolve(func=g,x0=w0) #计算投资组合中每只股票权重
for i in range(len(name)):
    print(name[i],round(weight[i],4))
    
# Example 5-4 最优化方法：一个案例
R=np.array([0.054703,0.053580,0.216717,0.049761,0.086041]) #股票平均年化收益率
P=np.array([4.99,7.49,19.16,19.20,7.18]) #股票收盘价
PE=np.array([5.6961,16.9758,20.2258,13.3713,43.6949]) #股票市盈率
def f(w): #定义求最优值的函数
    w=np.array(w)
    return -np.sum(R*w) #需要加负号，使其要求的最大值其相反数在minimize()中最小
cons=({'type':'eq','fun':lambda w: np.sum(w)-1},{'type':'ineq','fun':lambda w: 15-np.sum(w*PE)}) #以字典格式输入约束条件
bnds=((0,1),(0,1),(0,1),(0,1),(0,1)) #以元组格式输入边界值
W0=[0.25,0.25,0.25,0.25,0.25] #针对股票权重的初始猜测值
result=sco.minimize(fun=f,x0=W0,method='SLSQP',bounds=bnds,constraints=cons) #计算最优解
print('应该配置的每只股票的权重',result['x'].round(4)) #直接输出每只股票的权重
print('投资组合符合条件的最高期望收益率',-f(result['x'].round(4))) #计算投资组合的最高期望收益率
fund=1e7 #投资的总资金
shares=fund*result['x']/P #计算每只股票的购买数量
print('工商银行的股数（向下取整数）',int(shares[0]))
print('中国国航的股数（向下取整数）',int(shares[1]))
print('长江电力的股数（向下取整数）',int(shares[2]))
print('上海医药的股数（向下取整数）',int(shares[3]))
print('永辉超市的股数（向下取整数）',int(shares[-1]))

# Example 5-5 最优化方法：变更的案例
cons_new=({'type':'eq','fun':lambda w: np.sum(w)-1},{'type':'ineq','fun':lambda w: 20-np.sum(w*PE)}) #设置新的约束条件
result_new=sco.minimize(fun=f,x0=W0,method='SLSQP',bounds=bnds,constraints=cons_new) #计算新的最优解
result_new['x'].round(4) #直接输出每只股票的权重
print('投资组合符合新条件的最高期望收益率',-f(result_new['x'].round(4))) #计算投资组合新的最高期望收益率
shares_new=fund*result_new['x']/P #计算每只股票新的购买数量
print('工商银行的股数（向下取整数）',int(shares_new[0]))
print('中国国航的股数（向下取整数）',int(shares_new[1]))
print('长江电力的股数（向下取整数）',int(shares_new[2]))
print('上海医药的股数（向下取整数）',int(shares_new[3]))
print('永辉超市的股数（向下取整数）',int(shares_new[-1]))

# Example 5-6 描述性统计
import scipy.stats as st
index=pd.read_excel('E:/OneDrive/附件/数据/第5章/A股中小板指数和创业板指数日涨跌幅（2018年至2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
index.describe()
# Table 5-8 stats子模块中的统计函数及演示
st.describe(index) #查看描述性统计信息，与pandas的describe函数有部分相似之处；结果依次是：样本量、最值、均值、方差、偏度、峰度
st.kurtosis(index) #计算峰度
st.moment(index,moment=3) #计算3阶矩
st.mode(index) #计算众数
st.skew(index) #计算偏度

# 概率分析
# Table 5-10 除了以下4个分布函数，还有残/生存函数sf(survivor function)=1-cdf 残存函数的逆函数isf 对样本拟合（最大似然估计法得出最适合的概率密度函数系数）fit
# Example 5-7 rvs 生成服从指定分布的随机数
I=100000 #设定随机抽样次数为10W
r_mean=0.04 #利率的均值
r_std=0.01 #利率的标准差
rand_norm=st.norm.rvs(loc=r_mean,scale=r_std,size=I) #从均值为4%、标准差为1%的正态分布中抽取样本
plt.figure(figsize=(9,6))
plt.hist(rand_norm,bins=30,facecolor='y',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频次',fontsize=13)
plt.title(u'正态分布的抽样',fontsize=13)
plt.grid()
plt.show()

# Example 5-8 cdf 累积分布函数 cumulative distribution function
r1=0.03 #利率变量等于3%
prob=st.norm.cdf(x=r1,loc=r_mean,scale=r_std) #利率变量小于3%的概率
print('利率变量小于3%的概率',round(prob,6))

# Example 5-9 pdf 概率密度函数 probability density function
r2=0.05 #利率变量等于5%
value_pdf=st.norm.pdf(x=r2,loc=r_mean,scale=r_std) #利率变量等于5%时对应的概率密度函数值
print('利率变量等于5%时对应的概率密度函数值',round(value_pdf,6))

# Example 5-10 ppf 分位点函数 percent point function
prob=0.9
value_ppf=st.norm.ppf(q=prob,loc=r_mean,scale=r_std) #概率等于90%时对应的利率变量临界值
print('概率等于90%时对应的利率变量临界值',round(value_ppf,6))

# Example 5-11 正态性统计检验
# Step 1: (Kolmogorov-Smirnov)KS检验，原假设：样本值服从的分布等于给定的分布；备择假设：样本值服从的分布不等于给定的分布
st.kstest(rvs=index.iloc[:,0],cdf='norm',args=(0,0.017)) #对中小板指数日涨跌幅进行正态性检验
st.kstest(rvs=index.iloc[:,-1],cdf='norm',args=(0,0.017)) #对创业板指数日涨跌幅进行正态性检验
# 输出的结果中，第1个是统计量，第2个是P值。P值均小于5%，KS检验结果表明5%的显著性水平上拒绝中小板指数、创业板指数的日涨跌幅服从正态分布的（原）假设
# 第3个（在单样本检验中）是KS统计量对应的rvs值，即经验分布函数和假设的累积分布函数之间的距离是在该观测点测量的；
# 第4个（在单样本检验中）若KS统计量是经验分布函数和假设的累积分布函数之间的最大正差(D+)，则此值为+1；若统计量是最大负差(D-)，则此值为-1

# Step 2: Anderson-Darling检验，原假设与KS检验的原假设保持一致
st.anderson(x=index.iloc[:,0],dist='norm') #对中小板指数日涨跌幅进行正态性检验
st.anderson(x=index.iloc[:,-1],dist='norm') #对创业板指数日涨跌幅进行正态性检验
# 结果中，第1个是统计量，第2个是临界值的统计量，第3个是对应临界值的显著性水平（15%、10%、5%、2.5%、1%）。从Anderson-Darling检验的结果中可以得出结论：在1%的显著性水平上拒绝中小板指数、创业板指数的日涨跌幅服从正态分布的（原）假设
# 如果输出的统计量值statistic < critical_values，则表示在相应significance_level下，不拒绝原假设，即认为样本数据来自给定的分布
# 第4个是fit_result:https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.FitResult.html#scipy.stats._result_classes.FitResult

# Step 3: Shapiro-Wilk检验，仅用于正态性检验，且若样本量超过5000，其结果可能不准确
st.shapiro(index.iloc[:,0])
st.shapiro(index.iloc[:,-1])

# Step 4: normaltest函数，仅用于正态性检验，且支持多变量的样本值
st.normaltest(index,axis=0) #同步检验中小板指数和创业板指数
# 结果中，第1个是统计量，第2个是P值

import statsmodels #专注统计分析与建模的模块，最早起源于SciPy子模块stats中的models工具包
statsmodels.__version__
"""
常用的线性回归模型类型和对应的函数
OLS(endog,exog) 普通最小二乘法回归 ordinary least square regression
GLS 广义最小二乘法回归 generalized least square regression
WLS 加权最小二乘法回归 weighted least square regression
GLASAR 带有自相关误差模型的广义最小二乘法回归 GLS with autoregressive error model
GLM 广义线性模型 generalized linear model
RLM 使用M个估计量的鲁棒线性模型 robust linear model using M estimators
mixed 混合效应模型 mixed effects model
gam 广义加性模型 generalized additive model
"""
# Example 5-12 考察A股走势对H股走势的影响，以18-20年交行日收益率为例
# Step 1: 导入外部数据并可视化，计算股票的日收益率（采用对数收益率形式）
P_BoComm=pd.read_excel('E:\OneDrive\附件\数据\第5章\交通银行A股和H股每日收盘价数据（2018年至2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
P_BoComm.plot(figsize=(9,6),grid=True,fontsize=13) #股价格式化
R_BoComm=np.log(P_BoComm/P_BoComm.shift(1)) #计算交通银行股票的对数收益率
R_BoComm=R_BoComm.dropna() #删除缺失值
R_BoComm.describe() #查看描述性统计
# Step 2: 导入statsmodels子模块api，以交行H股日收益率作因变量，A股日收益率作自变量，构建普通最小二乘法回归模型
import statsmodels.api as sma
Y=R_BoComm.iloc[:,-1] #设定因变量样本值（交行H股日收益率）
X=R_BoComm.iloc[:,0] #设定自变量样本值（交行A股日收益率）
X_addcons=sma.add_constant(X) #对自变量的样本值增加一列常数项
model=sma.OLS(endog=Y,exog=X_addcons) #构建普通最小二乘法回归模型
result=model.fit() #生成一个线性回归的结果对象
result.summary() #输出完成的线性回归结果信息
result.params #输出截距项和系数
# Step 3: 结合散点图对线性回归模型进行可视化
plt.figure(figsize=(9,6))
plt.scatter(X,Y,c='b',marker='o') #散点图
plt.plot(X,result.params[0]+result.params[1]*X,'r-',lw=2.5) #拟合一条直线
plt.xticks(fontsize=13)
plt.xlabel(u'交通银行A股日收益率',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'交通银行H股日收益率',fontsize=13)
plt.title(u'交通银行A股与H股日收益率的散点图和线性拟合',fontsize=13)
plt.grid()
plt.show()

# Example 5-13
import arch
arch.__version__
# Step 1: 构建ARCH(1)模型并输出相关模型结果
from arch import arch_model
MS_Index=index.iloc[:,0] #从例5-6创建的数据框中提取中小板指数的数据
MS_Index.index=pd.DatetimeIndex(MS_Index.index) #将行索引转化为Datetime格式
model_arch=arch_model(y=MS_Index,mean='Constant',lags=0,vol='ARCH',p=1,o=0,q=0,dist='normal') #构建ARCH(1)模型
result_arch=model_arch.fit() #运用fit函数针对结果对象进行拟合得到拟合结果的对象
result_arch.summary() #输出拟合结果（最终的模型参数和统计量）
# Step 2: 构建GARCH(1,1)模型并输出相关模型结果
model_garch=arch_model(y=MS_Index,mean='Constant',lags=0,vol='GARCH',p=1,o=0,q=1,dist='normal')
result_garch=model_garch.fit()
result_garch.summary()
# Step 3: 通过params函数输出模型相关参数并对参数进行运算
result_garch.params
vol=np.sqrt(result_garch.params[1]/(1-result_garch.params[2]-result_garch.params[3]))
print('利用GARCH(1,1)模型得到的长期波动率（每日）',round(vol,4))
result_arch.plot()
result_garch.plot()

import datetime as dt
# Example 5-14 
T1=dt.datetime(2020,5,28)
T1
# Example 5-15
T2=dt.datetime(2020,4,18,17,28,58,678) #2020年4月18日17点28分58秒678微秒
T2
# Example 5-16
now=dt.datetime.now()
now
today=dt.datetime.today()
today
# Table 5-16 访问时间对象的属性
T2.isocalendar() #以ISO标准化日期格式显示
T2.ctime() #以字符串格式输出
now.ctime() #输出内容依次是星期几、月份、日期数、时\分\秒、年份
# Table 5-17 时间对象的比较
T1.__eq__(T2)
T1==T2
T1.__ge__(T2)
T1>=T2
T1.__gt__(T2)
T1>T2
T2.__le__(today)
T2<=today
T2.__lt__(today)
T2<today
T2.__ne__(today)
T2!=today
# Table 5-18 时间间隔的计算
T_delta1=T1-T2 #计算时间间隔
T_delta1.days #查看时间间隔的天数
T_delta2=today-T2
T_delta2.seconds
T_delta2.microseconds
