# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:49:22 2023

@author: XIE Ming
@github: https://github.com/BobMing
@linkedIn: https://www.linkedin.com/in/tseming
@email: xieming_xm@163.com
"""

# 指数数据可视化
import numpy as np
import pandas as pd
pd.set_option('display.unicode.ambiguous_as_wide',True) #处理数据的列名与其对应列数据无法对齐的情况
pd.set_option('display.unicode.east_asian_width',True) #无法对齐主要是因为列名是中文
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['STFangSong']
plt.rcParams['axes.unicode_minus']=False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

index_data=pd.read_excel('E:\OneDrive\附件\数据\第8章/四只A股市场股指的日收盘价数据（2018-2020）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
index_data.plot(subplots=True,layout=(2,2),figsize=(10,10),fontsize=13,grid=True)
plt.subplot(2,2,1) #第一张子图
plt.ylabel(u'指数点位',fontsize=11,position=(0,0)) #增加第1张子图的纵坐标标签

def value_ZGM(D,r):
    '''定义一个运用零增长模型计算股票内在价值的函数
    D: 代表企业已支付的最近一期每股股息金额。
    r: 代表与企业的风险相匹配的贴现利率（每年复利1次）'''
    value=D/r
    return value
# Example 8-1 招行股票内在价值
Div=1.2 #招商银行A股的固定股息
rate=0.1118 #贴现利率
value=value_ZGM(D=Div,r=rate)
print('运用零增长模型计算招商银行A股股票内在价值',round(value,4))

def value_CGM(D,g,r):
    '''定义一个运用不变增长模型计算股票内在价值的函数
    D: 代表企业已支付的最近一期每股股息金额。
    g: 代表企业的股息增长率，并且数值要小于贴现利率。
    r: 代表与企业的风险相匹配的贴现利率（每年复利1次）'''
    if r>g:
        value=D*(1+g)/(r-g)
    else:
        value='输入的贴现利率小于或等于股息增长率而导致结果不存在'
    return value
# Example 8-2
growth=0.1
value_new=value_CGM(D=Div,g=growth,r=rate)
print('运用不变增长模型计算招商银行A股股票的内在价值',round(value_new,4))

def value_2SGM(D,g1,g2,T,r):
    '''定义一个运用二阶段增长模型计算股票内在价值的函数
    D: 代表企业已支付的最近一期每股股息金额。
    g1: 代表企业在第1个阶段的股息增长率。
    g2: 代表企业在第2个阶段的股息增长率，且数值要小于贴现利率。
    T: 代表企业第1个阶段的期限，单位是年。
    r: 代表与企业的风险相匹配的贴现利率（每年复利1次）'''
    if r>g2:
        T_list=np.arange(1,T+1)
        V1=D*np.sum(pow(1+g1,T_list)/pow(1+r,T_list))
        V2=D*pow(1+g1,T)*(1+g2)/(pow(1+r,T)*(r-g2))
        value=V1+V2
    else:
        value='输入的贴现利率不大于第2个阶段的股息增长率而导致结果不存在'
    return value
# Example 8-3
g_stage1=0.11
g_stage2=0.08
T_stage1=10
value_2stages=value_2SGM(D=Div, g1=g_stage1, g2=g_stage2, T=T_stage1, r=rate)
print('运用二阶段增长模型计算招商银行A股股票内在价值',round(value_2stages,4))

# 敏感性分析不同阶段股息增长率与股票内在价值之间的关系
g1_list=np.linspace(0.06,0.11,100)
g2_list=np.linspace(0.03,0.08,100)
value_list1=np.zeros_like(g1_list) #创建存放对应第1个阶段股息增长率变化的股票内在价值初始数组
for i in range(len(g1_list)):
    value_list1[i]=value_2SGM(D=Div, g1=g1_list[i], g2=g_stage2, T=T_stage1, r=rate)
value_list2=np.zeros_like(g2_list)
for i in range(len(g2_list)):
    value_list2[i]=value_2SGM(D=Div, g1=g_stage1, g2=g2_list[i], T=T_stage1, r=rate)
plt.figure(figsize=(11,6))
plt.subplot(1,2,1)
plt.plot(g1_list,value_list1,'r-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'第1个阶段股息增长率',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'股票内在价值',fontsize=13)
plt.title(u'第1个阶段股息增长率与股票内在价值的关系图',fontsize=14)
plt.grid()
plt.subplot(1,2,2,sharey=plt.subplot(1,2,1))
plt.plot(g2_list,value_list2,'b-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'第2个阶段股息增长率',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'第2个阶段股息增长率与股票内在价值的关系图',fontsize=14)
plt.grid()
plt.show()

def value_3SGM(D,ga,gb,Ta,Tb,r):
    '''定义一个运用三阶段增长模型计算股票内在价值的函数
    D: 代表企业已支付的最近一期每股股息金额。
    ga: 代表企业在第1阶段的股息增长率。
    gb: 代表企业在第3阶段的股息增长率。
    Ta: 代表企业第1个阶段的期限（年）。
    Tb: 代表企业第1、2个阶段的期限之和（年）。
    r: 代表与企业的风险相匹配的贴现利率（每年复利1次）'''
    if r>gb:
        # Step 1: 计算第1个阶段股息贴现之和
        Ta_list=np.arange(1,Ta+1)
        D_stage1=D*pow(1+ga,Ta_list)
        V1=np.sum(D_stage1/pow(1+r,Ta_list))
        # Step 2: 计算第2个阶段股息贴现之和
        Tb_list=np.arange(Ta+1,Tb+1)
        D_t=D_stage1[-1] #第1个阶段最后一期股息
        D_stage2=[] #创建存放地2个阶段每期股息的空列表
        for i in range(len(Tb_list)):
            gt=ga-(ga-gb)*(Tb_list[i]-Ta)/(Tb-Ta) #以此计算第2个阶段每期股息增长率
            D_t*=(1+gt) #以此计算第2个阶段的每期股息金额
            D_stage2.append(D_t)
        D_stage2=np.array(D_stage2) #将列表转换为数组格式
        V2=np.sum(D_stage2/pow(1+r,Tb_list))
        # Step 3: 计算第3个阶段股息贴现之和
        D_Tb=D_stage2[-1] #第2个阶段最后一期股息
        V3=D_Tb*(1+gb)/(1+r)**Tb/(r-gb)
        # Step 4: 计算股票的内在价值
        value=V1+V2+V3
    else:
        value='输入的贴现利率不大于第3个阶段的股息增长率而导致结果不存在'
    return value
# Example 8-4
g_stage1=0.11
g_stage3=0.075
T_stage1=6
T_stage2=4
value_3stages=value_3SGM(D=Div, ga=g_stage1, gb=g_stage3, Ta=T_stage1, Tb=T_stage1+T_stage2, r=rate)
print('运用三阶段增长模型计算招商银行A股股票的内在价值',round(value_3stages,4))
# 敏感性分析，考察最近一期已支付的股息金额、贴现利率、第1个阶段股息增长率、第3个阶段股息增长率 对股票内在价值的影响，且可视化
Div_list=np.linspace(0.8,1.6,100)
rate_list=np.linspace(0.08,0.12,100)
ga_list=np.linspace(0.07,0.11,100)
gb_list=np.linspace(0.04,0.08,100)
value_list1=np.zeros_like(Div_list) #创建不同股息金额的股票内在价值初始数组
for i in range(len(Div_list)):
    value_list1[i]=value_3SGM(D=Div_list[i], ga=g_stage1, gb=g_stage3, Ta=T_stage1, Tb=T_stage1+T_stage2, r=rate)
value_list2=np.zeros_like(rate_list) #创建不同贴现利率的股票内在价值初始数组
for i in range(len(rate_list)):
    value_list2[i]=value_3SGM(D=Div, ga=g_stage1, gb=g_stage3, Ta=T_stage1, Tb=T_stage1+T_stage2, r=rate_list[i])
value_list3=np.zeros_like(ga_list) #创建对应第1个阶段不同股息增长率的股票内在价值初始数组
for i in range(len(ga_list)):
    value_list3[i]=value_3SGM(D=Div, ga=ga_list[i], gb=g_stage3, Ta=T_stage1, Tb=T_stage1+T_stage2, r=rate)
value_list4=np.zeros_like(gb_list) #创建对应第3个阶段不同股息增长率的股票内在价值初始数组
for i in range(len(gb_list)):
    value_list4[i]=value_3SGM(D=Div, ga=g_stage1, gb=gb_list[i], Ta=T_stage1, Tb=T_stage1+T_stage2, r=rate)
plt.figure(figsize=(10,11))
plt.subplot(2,2,1)
plt.plot(Div_list,value_list1,'r-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'最近一期已支付的股息金额',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'股票内在价值',fontsize=13,rotation=90)
plt.grid()
plt.subplot(2,2,2)
plt.plot(rate_list,value_list2,'b-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'贴现利率',fontsize=13)
plt.yticks(fontsize=13)
plt.grid()
plt.subplot(2,2,3)
plt.plot(ga_list,value_list3,'r-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'第1个阶段股息增长率',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'股票内在价值',fontsize=13,rotation=90)
plt.grid()
plt.subplot(2,2,4)
plt.plot(gb_list,value_list4,'b-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'第3个阶段股息增长率',fontsize=13)
plt.yticks(fontsize=13)
plt.grid()
plt.show()

# Example 8-5 模拟服从几何布朗运动的股价
# Step 1: 导入数据并计算股价年化收益率、年化波动率
S=pd.read_excel('E:\OneDrive\附件\数据\第8章/招商银行A股日收盘价数据（2018-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
R=np.log(S/S.shift(1)) #计算招商银行A股日收益率（连续复利的收益率）
mu=R.mean()*252 #股票的平均年化收益率
mu=float(mu) #转换为浮点型数据类型
print('招商银行A股平均年化收益率',round(mu,6))
sigma=R.std()*np.sqrt(252) #股票收益的年化波动率
sigma=float(sigma)
print('招商银行A股年化波动率',round(sigma,6))
# Step 2: 输入模拟参数
import numpy.random as npr
date=pd.date_range(start='2021-01-04',end='2023-12-31',freq='B') #创建2021年至2023年的工作日数列
N=len(date) #计算date的元素个数
I=500 #设定摸你的路径数量（随机抽样次数）
dt=1.0/252 #单位时间长度（1交易日，换算成交易年）
S_GBM=np.zeros((N,I)) #创建存放模拟服从几何布朗运动的未来股价初始数组
S_GBM[0]=43.17 #模拟的起点设为2021年1月4日的收盘价
# Step 3: 创建摸你的未来股价时间序列
for t in range(1,N):
    epsilon=npr.standard_normal(I) #基于标准正态分布的随机抽样
    S_GBM[t]=S_GBM[t-1]*np.exp((mu-0.5*sigma**2)*dt+sigma*epsilon*np.sqrt(dt))
S_GBM=pd.DataFrame(S_GBM,index=date) #将存放服从GBM的模拟未来股价数组转换为带时间索引的数据框
S_GBM.head() #显示数据框的开头5行
S_GBM.tail() #显示数据框的末尾5行
S_GBM.describe() #显示数据框的描述性统计指标
# Step 4: 将模拟股价的结果可视化
plt.figure(figsize=(9,6))
plt.plot(S_GBM)
plt.xlabel(u'日期',fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel(u'招商银行股价',fontsize=13)
plt.yticks()
plt.title(u'2021-2023年服从几何布朗运动的股价模拟路径',fontsize=13)
plt.grid()
plt.show()
# 为更清晰地展示模拟路径，下面仅可视化模拟的前20条路径
plt.figure(figsize=(9,6))
plt.plot(S_GBM.iloc[:,0:20])
plt.xlabel(u'日期',fontsize=13)
plt.ylabel(u'招商银行股价',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'2021-2023年服从几何布朗运动的股价的前20条模拟路径',fontsize=13)
plt.grid()
plt.show()

x=npr.rand(5) #从均匀分布中随机抽取5个随机数
weight=x/np.sum(x) #创建权重数组
weight
round(sum(weight),2) #验证权重随机数之和是否等于1

# Example 8-6 计算一个投资组合的预期收益率和波动率
data_stocks=pd.read_excel('E:\OneDrive\附件\数据\第8章/5只A股股票的收盘价（2018年至2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
(data_stocks/data_stocks.iloc[0]).plot(figsize=(9,6),grid=True) #将股价按首个交易日进行归一化处理且可视化
R=np.log(data_stocks/data_stocks.shift(1)) #计算股票的对数收益率
R.describe()
R.hist(bins=40,figsize=(9,11)) #将股票收益率用直方图展示
R_mean=R.mean()*252 #计算股票的年化平均收益率
print(R_mean)
R_vol=R.std()*np.sqrt(252) #计算股票收益率的年化波动率
print(R_vol)
R_cov=R.cov()*252 #计算股票的协方差矩阵且进行年化处理
print(R_cov)
R_corr=R.corr() #计算股票的相关系数矩阵
print(R_corr)
n=5 #投资组合中的个股数量
w=np.ones(n)/n #投资组合中每只股票相同权重的权重数组
R_port=np.sum(w*R_mean)
print('投资组合年化的预期收益率',round(R_port,4))
vol_port=np.sqrt(np.dot(w,np.dot(R_cov,w.T)))
print('投资组合年化的波动率',round(vol_port,4))

# Example 8-7 可行集
I=2000 #需要创建权重数组的数量
Rp_list=np.ones(I)
Vp_list=np.ones(I)
for i in range(I):
    x=np.random.rand(n) #从均匀分布中随机抽取0~1的5个随机数
    weights=x/sum(x)
    Rp_list[i]=np.sum(weights*R_mean)
    Vp_list[i]=np.sqrt(np.dot(weights,np.dot(R_cov,weights.T)))
plt.figure(figsize=(9,6))
plt.scatter(Vp_list,Rp_list)
plt.xlabel(u'波动率',fontsize=13)
plt.ylabel(u'预期收益率',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'投资组合预期收益率与波动率的关系图',fontsize=13)
plt.grid()
plt.show()

# Example 8-8 有效前沿：给定投资组合年化的预期收益率=15%
import scipy.optimize as sco
def f(w): #定义一个求最优解的函数
    w=np.array(w) #将l列表数组化
    Rp_opt=np.sum(w*R_mean)
    Vp_opt=np.sqrt(np.dot(w,np.dot(R_cov,w.T)))
    return np.array([Rp_opt,Vp_opt])
def Vmin_f(w): #定义一个计算最小波动率所对应权重的函数
    return f(w)[1] #输出结果是投资组合的波动率
cons=({'type':'eq','fun':lambda x: np.sum(x)-1},{'type':'eq','fun':lambda x: f(x)[0]-0.15}) #权重的约束条件（以字典格式输入）
bnds=((0,1),)*5 #权重的边界条件（以元组格式输入）
w0=np.ones(5)*0.2 #创建权重相等的数组作为迭代运算的初始值
result=sco.minimize(fun=Vmin_f,x0=w0,method='SLSQP',bounds=bnds,constraints=cons)
print('投资组合预期收益率15%对应投资组合的波动率',round(result['fun'],4))
print('投资组合预期收益率15%对应长江电力的权重',round(result['x'][0],4))
print('投资组合预期收益率15%对应平安银行的权重',round(result['x'][1],4))
print('投资组合预期收益率15%对应上海机场的权重',round(result['x'][2],4))
print('投资组合预期收益率15%对应中信证券的权重',round(result['x'][3],4))
print('投资组合预期收益率15%对应顺丰控股的权重',round(result['x'][4],4))

# Example 8-9 有效前沿：波动率的全局最小值、对应的预期收益率及股票权重
cons_vmin=({'type':'eq','fun':lambda x:np.sum(x)-1}) #设置波动率是全局最小值的约束条件
result_vmin=sco.minimize(fun=Vmin_f,x0=w0,method='SLSQP',bounds=bnds,constraints=cons_vmin)
Vp_vmin=result_vmin['fun']
print('在可行集上属于全局最小值的波动率',round(Vp_vmin,4))
Rp_vmin=np.sum(R_mean*result_vmin['x']) #计算相应的投资组合预期收益率
print('全局最小值的波动率对应投资组合的预期收益率',round(Rp_vmin,4))
print('全局最小值的波动率对应长江电力的权重',round(result_vmin['x'][0],4))
print('全局最小值的波动率对应平安银行的权重',round(result_vmin['x'][1],4))
print('全局最小值的波动率对应上海机场的权重',round(result_vmin['x'][2],4))
print('全局最小值的波动率对应中信证券的权重',round(result_vmin['x'][3],4))
print('全局最小值的波动率对应顺丰控股的权重',round(result_vmin['x'][4],4))

# Example 8-10 有效前沿：以对应波动率全局最小值的预期收益率作区间下限、以30%作区间上限的目标预期收益率等差数列，计算对应的波动率数组，从而构建有效前沿
Rp_target=np.linspace(Rp_vmin,0.3,200)
Vp_target=[]
for r in Rp_target:
    cons_new=({'type':'eq','fun':lambda x:np.sum(x)-1},{'type':'eq','fun':lambda x:f(x)[0]-r})
    result_new=sco.minimize(fun=Vmin_f, x0=w0,method='SLSQP',bounds=bnds,constraints=cons_new)
    Vp_target.append(result_new['fun']) #存放每一次计算得到的波动率
plt.figure(figsize=(9,6))
plt.scatter(Vp_list,Rp_list)
plt.plot(Vp_target,Rp_target,'r-',lw=2.5,label=u'有效前沿')
plt.plot(Vp_vmin,Rp_vmin,'g*',markersize=13,label=u'全局最小波动率')
plt.xlabel(u'波动率',fontsize=13)
plt.ylabel(u'预期收益率',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(0.15,0.3)
plt.ylim(0.06,0.2)
plt.title(u'投资组合的有效前沿',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 8-11 资本市场线 Capital Market Line, CML
Rf=0.0385 #1年期LPR利率（无风险利率）
def F(w):
    w=np.array(w)
    Rp_opt=np.sum(w*R_mean) #计算投资组合的预期收益率
    Vp_opt=np.sqrt(np.dot(w,np.dot(R_cov,w.T))) #计算投资组合的波动率
    Slope=(Rp_opt-Rf)/Vp_opt #计算资本市场线的斜率
    return np.array([Rp_opt,Vp_opt,Slope])
def Slope_F(w): #定义使负的资本市场线斜率最小化（即使其本身最大化）的函数
    return -F(w)[-1] #输出结果是负的资本市场线斜率
cons_Slope=cons_vmin
result_Slope=sco.minimize(fun=Slope_F, x0=w0,method='SLSQP',bounds=bnds,constraints=cons_Slope)
Slope=-result_Slope['fun'] #还原资本市场线斜率（最大值）
print('资本市场线的斜率',round(Slope,4))
Wm=result_Slope['x'] #市场组合的每只股票配置权重
print('市场组合配置的长江电力的权重',round(Wm[0],4))
print('市场组合配置的平安银行的权重',round(Wm[1],4))
print('市场组合配置的上海机场的权重',round(Wm[2],4))
print('市场组合配置的中信证券的权重',round(Wm[3],4))
print('市场组合配置的顺丰控股的权重',round(Wm[4],4))
Rm=np.sum(R_mean*Wm)
Vm=(Rm-Rf)/Slope
print('市场组合的预期收益率',round(Rm,4))
print('市场组合的波动率',round(Vm,4))

Rp_CML=np.linspace(Rf,0.25,200) #资本市场线的投资组合预期收益率数组（线上纵坐标）
Vp_CML=(Rp_CML-Rf)/Slope #资本市场线的投资组合波动率数组（根据式(8-55)算出的横坐标）
plt.figure(figsize=(9,6))
plt.scatter(Vp_list,Rp_list)
plt.plot(Vp_target,Rp_target,'r-',lw=2.5,label=u'有效前沿')
plt.plot(Vp_vmin,Rp_vmin,'g*',markersize=14,label=u'全局最小波动率')
plt.plot(Vp_CML,Rp_CML,'b--',lw=2.5,label=u'资本市场线')
plt.plot(Vm,Rm,'y*',markersize=14,label=u'市场组合')
plt.xlabel(u'波动率',fontsize=13)
plt.ylabel(u'预期收益率',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(0.0,0.3)
plt.ylim(0.03,0.22)
plt.title(u'投资组合理论的可视化',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 8-12 上证180指数成分股模拟投资组合(逐次增加成分股股票数量)，演示其系统风险与非系统风险
price_stocks=pd.read_excel('E:\OneDrive\附件\数据\第8章/上证180指数成分股日收盘价（2018-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
price_stocks.columns
price_stocks.index
# 计算每只股票的日收益率数据
return_stocks=np.log(price_stocks/price_stocks.shift(1)) #建立股票的日收益率时间序列
n=len(return_stocks.columns)
vol_port=np.zeros(n) #创建存放投资组合波动率的初始数组
for i in range(1,n+1):
    w=np.ones(i)/i #逐次计算股票的等权重数组
    cov=252*return_stocks.iloc[:,:i].cov() #逐次计算不同股票之间的年化协方差
    vol_port[i-1]=np.sqrt(np.dot(w,np.dot(cov,w.T))) #逐次计算投资组合的年化波动率
# 可视化股票数量与投资组合波动率之间的关系
N_list=np.arange(n)+1
plt.figure(figsize=(9,6))
plt.plot(N_list,vol_port,'r-',lw=2.0)
plt.xlabel(u'投资组合中的股票数量',fontsize=13)
plt.ylabel(u'投资组合波动率',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'投资组合中的股票数量与投资组合波动率之间的关系图',fontsize=13)
plt.grid()
plt.show()

def Ri_CAPM(beta,Rm,Rf):
    '''定义一个运用资本资产定价模型计算股票预期收益率的函数
    beta: 代表股票的贝塔值。
    Rm: 代表市场收益率。
    Rf: 代表无风险利率'''
    Ri=Rf+beta*(Rm-Rf)
    return Ri
# Example 8-13 以招商银行A股作分析对象，演示计算贝塔值，并得到该股票预期收益率
P_bank_index=pd.read_excel('E:\OneDrive\附件\数据\第8章\招商银行A股与沪深300指数日收盘价数据（2017-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
R_bank_index=np.log(P_bank_index/P_bank_index.shift(1))
R_bank_index=R_bank_index.dropna()
R_bank_index.describe()
import statsmodels.api as sm
R_bank=R_bank_index['招商银行'] #取招商银行A股的日收益率序列（因变量）
R_index=R_bank_index['沪深300指数'] #取沪深300指数的日收益率序列（自变量）
R_index_addcons=sm.add_constant(R_index) #对自变量的样本值增加一列常数项
model=sm.OLS(endog=R_bank,exog=R_index_addcons) #构建普通最小二乘法的线性回归模型
result=model.fit() #拟合线性回归模型
result.summary()
result.params
LPR_1Y=0.0385 #1年期LPR利率（无风险利率）
R_market=252*R_index.mean() #计算沪深300指数的年化收益率
R_stock=Ri_CAPM(beta=result.params[-1],Rm=R_market,Rf=LPR_1Y) #计算招商银行A股的预期收益率（年化）
print('招商银行A股的年化预期收益率',round(R_stock,6))

# Example 8-14 Security Market Line, SML
beta_list=np.linspace(0,2.0,100) #设定[0,2.0)的贝塔值数组
R_stock_list=Ri_CAPM(beta=beta_list,Rm=R_market,Rf=LPR_1Y) #计算招商银行A股预期收益率
plt.figure(figsize=(9,6))
plt.plot(beta_list,R_stock_list,'r',label=u'证券市场线',lw=2.0)
plt.plot(result.params[-1],R_stock,'o',lw=2.5)
plt.axis('tight')
plt.xticks(fontsize=13)
plt.xlabel(u'贝塔值',fontsize=13)
plt.xlim(0,2.0)
plt.yticks(fontsize=13)
plt.ylabel(u'股票预期收益率',fontsize=13)
plt.ylim(0,0.2)
plt.title(u'资本资产定价模型（以招商银行A股为例）',fontsize=13)
plt.annotate(u'贝塔值等于0.958对应的收益率',fontsize=14,xy=(0.96,0.1115),xytext=(1.0,0.06),arrowprops=dict(facecolor='b',shrink=0.05))
plt.legend(fontsize=13)
plt.grid()
plt.show()

# Example 8-15 A股市场4只开放式股票型基金
def SR(Rp,Rf,Vp):
    '''定义一个计算夏普比率的函数
    Rp: 代表投资组合的年化收益率。
    Rf: 代表无风险利率。
    Vp: 代表投资组合的年化波动率'''
    sharpe_ratio=(Rp-Rf)/Vp
    return sharpe_ratio

# Example 8-16 计算4只基金的夏普比率
fund=pd.read_excel('E:\OneDrive\附件\数据\第8章\国内4只开放式股票型基金净值数据（2018-2020）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
fund.plot(figsize=(9,6),grid=True) #基金净值可视化
R_fund=np.log(fund/fund.shift(1)) #创建基金日收益率的时间序列
R_fund=R_fund.dropna() #删除缺失值
R_mean=R_fund.mean()*252 #计算全部3年的平均年化收益率
Sigma=R_fund.std()*np.sqrt(252) #计算全部3年的年化波动率
R_f=0.015 #1年期银行存款基准利率作为无风险利率
SR_3years=SR(Rp=R_mean,Rf=R_f,Vp=Sigma)
print('2018年至2020年3年平均的夏普比率\n',round(SR_3years,4))
R_fund2018=R_fund.loc['2018-01-01':'2018-12-31'] #获取2018年的日收益率
R_fund2019=R_fund.loc['2019-01-01':'2019-12-31'] #获取2019年的日收益率
R_fund2020=R_fund.loc['2020-01-01':'2020-12-31'] #获取2020年的日收益率
R_mean_2018=R_fund2018.mean()*252 #计算2018年的年化收益率
R_mean_2019=R_fund2019.mean()*252 #计算2019年的年化收益率
R_mean_2020=R_fund2020.mean()*252 #计算2020年的年化收益率
Sigma_2018=R_fund2018.std()*np.sqrt(252) #计算2018年的年化波动率
Sigma_2019=R_fund2019.std()*np.sqrt(252) #计算2019年的年化波动率
Sigma_2020=R_fund2020.std()*np.sqrt(252) #计算2020年的年化波动率
SR_2018=SR(Rp=R_mean_2018,Rf=R_f,Vp=Sigma_2018) #计算2018年的夏普比率
SR_2019=SR(Rp=R_mean_2019,Rf=R_f,Vp=Sigma_2019) #计算2019年的夏普比率
SR_2020=SR(Rp=R_mean_2020,Rf=R_f,Vp=Sigma_2020) #计算2020年的夏普比率
print('2018年的夏普比率\n',round(SR_2018,4))
print('2019年的夏普比率\n',round(SR_2019,4))
print('2020年的夏普比率\n',round(SR_2020,4))

def SOR(Rp,Rf,Vd):
    '''定义一个计算索提诺比率的函数
    Rp: 表示投资组合的年化收益率。
    Rf: 表示无风险利率。
    Vd: 表示投资组合收益率的年化下行标准差'''
    sortino_ratio=(Rp-Rf)/Vd
    return sortino_ratio
# Example 8-17 计算4只基金的索提诺比率
V_down=np.zeros_like(R_mean) #创建存放基金收益率下行标准差的初始数组
for i in range(len(V_down)):
    R_neg=R_fund.iloc[:,i][R_fund.iloc[:,i]<0] #生成基金收益率为负的时间序列
    N_down=len(R_neg) #计算亏损的交易日天数
    V_down[i]=np.sqrt(252)*np.sqrt(np.sum(R_neg**2)/N_down) #计算年化下行标准差
    print(R_fund.columns[i],'年化下行标准差',round(V_down[i],4))
SOR_3years=SOR(Rp=R_mean,Rf=R_f,Vd=V_down)
print('2018年至2020年3年平均的索提诺比率\n',round(SOR_3years,4))

def TR(Rp,Rf,beta):
    '''定义一个计算特雷诺比率的函数
    Rp: 表示投资组合的年化收益率。
    Rf: 表示无风险利率。
    beta: 表示投资组合的贝塔值'''
    treynor_ratio=(Rp-Rf)/beta
    return treynor_ratio
# Example 8-18 计算4只基金的特雷诺比率
HS300=pd.read_excel('E:\OneDrive\附件\数据\第8章\沪深300指数日收盘价（2018-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
R_HS300=np.log(HS300/HS300.shift(1))
R_HS300=R_HS300.dropna()
X_addcons=sm.add_constant(R_HS300) #沪深300指数日收益率序列（自变量）增加一列常数项
betas=np.zeros_like(R_mean) #创建放置基金贝塔值的初始数组
cons=np.zeros_like(R_mean) #创建放置线性回归方程常数项的初始数组
for i in range(len(R_mean)):
    Y=R_fund.iloc[:,i] #设定因变量的样本值
    model=sm.OLS(endog=Y,exog=X_addcons)
    result=model.fit()
    cons[i]=result.params[0]
    betas[i]=result.params[1]
    print(R_fund.columns[i],'贝塔值',round(betas[i],4))
X_list=np.linspace(np.min(R_HS300),np.max(R_HS300),200)
plt.figure(figsize=(11,10))
for i in range(len(R_mean)):
    plt.subplot(2,2,i+1)
    plt.scatter(R_HS300,R_fund.iloc[:,i])
    plt.plot(X_list,cons[i]+betas[i]*X_list,'r-',label=u'线性回归拟合',lw=2.0)
    plt.xlabel(u'沪深300指数',fontsize=13)
    plt.ylabel(R_fund.columns[i],fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.grid()
plt.show()
TR_3years=TR(Rp=R_mean,Rf=R_f,beta=betas)
print('2018年至2020年3年平均的特雷诺比率\n',round(TR_3years,4))

def CR(Rp,MDD):
    '''定义一个计算卡玛比率的函数
    Rp: 表示投资组合的年化收益率。
    MDD: 表示投资组合的最大回撤率'''
    calmar_ratio=Rp/MDD
    return calmar_ratio
def MDD(data):
    '''定义一个计算投资组合（以基金为例）最大回撤率的函数
    data: 代表某只基金的净值数据，以序列或者数据框格式输入'''
    N=len(data) #计算期间的交易日天数
    DD=np.zeros((N-1,N-1)) #创建元素为0的N-1行、N-1列数组，用于存放回撤率数据
    for i in range(N-1):
        Pi=data.iloc[i] #第i个交易日的基金净值
        for j in range(i+1,N):
            Pj=data.iloc[j] #第j个交易日的基金净值
            DD[i,j-1]=(Pi-Pj)/Pi
    Max_DD=np.max(DD)
    return Max_DD
# Example 8-19 计算4只基金的卡玛比率
fund_zhonghai=fund['中海量化策略基金']
fund_nanfang=fund['南方新蓝筹基金']
fund_jiaoyin=fund['交银精选基金']
fund_tianhong=fund['天弘惠利基金']
MDD_zhonghai=MDD(data=fund_zhonghai)
MDD_nanfang=MDD(data=fund_nanfang)
MDD_jiaoyin=MDD(data=fund_jiaoyin)
MDD_tianhong=MDD(data=fund_tianhong)
print('2018年至2020年中海量化策略基金的最大回撤率',round(MDD_zhonghai,4))
print('2018年至2020年南方新蓝筹基金的最大回撤率',round(MDD_nanfang,4))
print('2018年至2020年交银精选基金的最大回撤率',round(MDD_jiaoyin,4))
print('2018年至2020年天弘惠利基金的最大回撤率',round(MDD_tianhong,4))
CR_zhonghai=CR(Rp=R_mean['中海量化策略基金'],MDD=MDD_zhonghai)
CR_nanfang=CR(Rp=R_mean['南方新蓝筹基金'],MDD=MDD_nanfang)
CR_jiaoyin=CR(Rp=R_mean['交银精选基金'],MDD=MDD_jiaoyin)
CR_tianhong=CR(Rp=R_mean['天弘惠利基金'],MDD=MDD_tianhong)
print('2018年至2020年中海量化策略基金的卡玛比率',round(CR_zhonghai,4))
print('2018年至2020年南方新蓝筹基金的卡玛比率',round(CR_nanfang,4))
print('2018年至2020年交银精选基金的卡玛比率',round(CR_jiaoyin,4))
print('2018年至2020年天弘惠利基金的卡玛比率',round(CR_tianhong,4))

def IR(Rp,Rb,TE):
    '''定义一个计算信息比率的函数
    Rp: 表示投资组合的年化收益率。
    Rb: 表示基准组合的年化收益率。
    TE: 表示跟踪误差'''
    information_ratio=(Rp-Rb)/TE
    return information_ratio
# Example 8-20 计算4只基金的信息比率
TE_fund=np.zeros_like(R_mean) #创建存放基金跟踪误差的初始数组
for i in range(len(R_mean)):
    TD=np.array(R_fund.iloc[:,i])-np.array(R_HS300.iloc[:,0])
    TE_fund[i]=TD.std()*np.sqrt(252)
    print(R_fund.columns[i],'跟踪误差',round(TE_fund[i],4))
R_mean_HS300=R_HS300.mean()*252 #计算沪深300指数的年化收益率
R_mean_HS300=float(R_mean_HS300) #将1元素序列转换为浮点型
IR_3years=IR(Rp=R_mean,Rb=R_mean_HS300,TE=TE_fund)
print('2018年至2020年3年平均的信息比率\n',round(IR_3years,4))