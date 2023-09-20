# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:03:02 2023

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

from matplotlib import font_manager
ttf_lists = font_manager.fontManager.ttflist
for font in ttf_lists:
    print(font)
print(len(ttf_lists))
    
from matplotlib.font_manager import _rebuild
_rebuild() #reload一下
plt.rcParams['font.family']=['FangSong','FangSong_GB2312','STSong','Arial Unicode MS']
plt.rcParams['font.sans-serif']=['FangSong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# Figure 7-1
bond_GDP=pd.read_excel('E:\OneDrive\附件\数据\第7章\债券存量规模与GDP（2010-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
bond_GDP.plot(kind='bar',figsize=(9,6),fontsize=13,grid=True)
plt.ylabel(u'金额',fontsize=11)

# Figure 7-2
bond=pd.read_excel('E:\OneDrive\附件\数据\第7章/2020年末存量债券的市场分布情况.xlsx',sheet_name='Sheet1',header=0,index_col=0)
plt.figure(figsize=(12,6))
plt.pie(x=bond['债券余额（亿元）'],labels=bond.index)
plt.axis('equal') #使饼图是个圆形
plt.legend(loc=2, fontsize=13) #图例在左上方
plt.title(u'2020年末存量债券的市场分布图',fontsize=13)
plt.show()

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
# Example 7-1 20贴现国债27
C_TB2027=0 #票面利率
par=100
T_TB2027=0.5 #期限
m_TB2027=0 #每年支付票息的频次
y_TB2027=0.01954 #贴现利率
value_TB2027=Bondprice_onediscount(C=C_TB2027, M=par, m=m_TB2027, y=y_TB2027, t=T_TB2027)
print('2020年6月8日20贴现国债27的价格',round(value_TB2027,4))

# Example 7-2 20附息国债06
C_TB2006=0.0268 #票面利率
m_TB2006=2 #每年支付票息的频次
y_TB2006=0.02634 #贴现利率
T_TB2006=10 #期限
Tlist_TB2006=np.arange(1,m_TB2006*T_TB2006+1)/m_TB2006 #定价日至每期票息支付日的期限数组
value_TB2006=Bondprice_onediscount(C=C_TB2006, M=par, m=m_TB2006, y=y_TB2006, t=Tlist_TB2006)
print('2020年5月21日20附息国债06的价格',round(value_TB2006,4))

# Example 7-3 带票息债券的到期收益率
def YTM(P,C,M,m,t):
    '''定义一个计算债券到期收益率的函数
    P: 代表观察到的债券市场价格。
    C: 代表债券的票面利率。
    M: 代表债券的本金。
    m: 代表债券票息每年支付的频次。
    t: 代表定价日至后续每一期票息支付日的期限长度，用数组格式输入；零息债券可直接输入数字。
    '''
    import scipy.optimize as so
    def f(y):
        coupon=np.ones_like(t)*M*C/m #创建每一期票息金额的数组
        NPV_coupon=np.sum(coupon*np.exp(-y*t)) #计算每一期票息在定价日的现值之和
        NPV_par=M*np.exp(-y*t[-1]) #计算本金在定价日的现值
        value=NPV_coupon+NPV_par #定价日的债券现金流现值之和
        return value-P #债券现金流现值之和减去债券市场价格
    if C==0: #针对零息债券
        y=(np.log(M/P))/t
    else: #针对带票息债券
        y=so.fsolve(func=f,x0=0.1) #第2个参数是任意输入的初始值
    return y
P_TB0911=104.802 #09附息国债11的市场价格
C_TB0911=0.0369 #09附息国债11的票面利率
m_TB0911=2 #09附息国债11票息支付的频次
T_TB0911=4 #09附息国债11的剩余期限
Tlist_TB0911=np.arange(1,m_TB0911*T_TB0911+1)/m_TB0911 #定价日至每期票息支付日的期限数组
Bond_yield=YTM(P=P_TB0911,C=C_TB0911,M=par,m=m_TB0911,t=Tlist_TB0911) #此时得到的结果为float单元素数组
Bond_yield=float(Bond_yield) #转换为单一的浮点型
print('2020年6月11日09附息国债11的到期收益率',round(Bond_yield,6))
price=Bondprice_onediscount(C=C_TB0911, M=par, m=m_TB0911, y=Bond_yield, t=Tlist_TB0911) #用33~48行的自定义函数Bondprice_onediscount计算债券价格 验证该YTM计算结果
print('09附息国债11的债券价格（用于验证）',round(price,4))

def Bondprice_diffdiscount(C,M,m,y,t) :
    '''定义一个基于不同期限贴现率计算债券价格的函数
    C: 代表债券的票面利率，如果输入0则表示零息债券。
    M: 代表债券的本金。
    m: 代表债券票息每年支付的频次。
    y: 代表不同期限的贴现利率，用数组格式输入；零息债券可直接输入数字（y_T，期限为T年的连续复利贴现利率）。
    t: 代表定价日至后续每一期票息支付日的期限长度，用数组格式输入；零息债券可直接输入数字（T，债券/剩余期限）'''
    if C==0: #针对零息债券
        price=np.exp(-y*t)*M
    else: #针对带票息债券
        coupon=np.ones_like(t)*M*C/m #创建每一期票息金额的数组
        NPV_coupon=np.sum(coupon*np.exp(-y*t)) #计算每一期票息在定价日的现值之和
        NPV_par=M*np.exp(-y[-1]*t[-1]) #计算本金在定价日的现值
        price=NPV_coupon+NPV_par #计算在定价日的债券价格
    return price

# Example 7-4 票息剥离法（bootstrap method）计算零息利率
P=np.array([99.5508,99.0276,100.8104,102.1440,102.2541]) #不同期限债券价格
T=np.array([0.25,0.5,1.0,1.5,2.0]) #债券的期限结构
C=np.array([0,0,0.0258,0.0357,0.0336]) #第4只和第5只债券的付息频次
m=2 #第4只和第5只债券的付息频次

def f(R): #联立方程组
    from numpy import exp
    R1,R2,R3,R4,R5=R #不同期限的零息利率
    B1=P[0]*exp(R1*T[0])-par
    B2=P[1]*exp(R2*T[1])-par
    B3=P[2]*exp(R3*T[2])-par*(1+C[2])
    B4=par*(C[3]/m*exp(-R2*T[1])+C[3]/m*exp(-R3*T[2])+(C[3]/m+1)*exp(-R4*T[3]))-P[3]
    B5=par*(C[-1]/m*exp(-R2*T[1])+C[-1]/m*exp(-R3*T[2])+C[-1]/m*exp(-R4*T[3])+(C[-1]/m+1)*exp(-R5*T[-1]))-P[-1]
    return np.array([B1,B2,B3,B4,B5])
import scipy.optimize as so
r0=[0.1]*5
rates=so.fsolve(func=f,x0=r0)
print('0.25年期的零息利率（连续复利）',round(rates[0],6))
print('0.5年期的零息利率（连续复利）',round(rates[1],6))
print('1年期的零息利率（连续复利）',round(rates[2],6))
print('1.5年期的零息利率（连续复利）',round(rates[3],6))
print('2年期的零息利率（连续复利）',round(rates[-1],6))
plt.figure(figsize=(9,6))
plt.plot(T,rates,'b-') #蓝色折现
plt.plot(T,rates,'ro') #红色圆点
plt.xlabel(u'期限（年）',fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel(u'利率',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'运用票息剥离法得到的零息曲线',fontsize=13)
plt.grid()
plt.show()

# Example 7-5 插值处理
import scipy.interpolate as si
func=si.interp1d(x=T,y=rates,kind="cubic") #运用已有数据构建插值函数且运用3阶样条曲线插值法
T_new=np.linspace(0.25,2.0,8) #创建包含0.75、1.25、1.75年期限的数组
rates_new=func(T_new) #计算基于插值法的零息利率
for i in range(len(T_new)):
    print(T_new[i],'年期限的零息利率（连续复利）',round(rates_new[i],6))
plt.figure(figsize=(9,6))
plt.plot(T_new,rates_new,'o')
plt.plot(T_new,rates_new,'-')
plt.xlabel(u'期限（年）',fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel(u'利率',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'基于3阶样条曲线插值方法得到的零息曲线',fontsize=13)
plt.grid()
plt.show()

# Example 7-6 用（不同期限的）零息利率对债券定价
C_new=0.036 #债券的票面利率
m_new=4 #债券票息支付频次
price_new=Bondprice_diffdiscount(C=C_new, M=par, m=m_new, y=rates_new, t=T_new)
print('基于不同期限的贴现利率计算债券价格',round(price_new,4))

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
# Example 7-7 09附息国债11 麦考利久期的计算过程
C_TB0911=0.0369
m_TB0911=2
y_TB0911=0.024 #round(Bond_yield,4)
T_TB0911=4
Tlist_TB0911=np.arange(1,m_TB0911*T_TB0911+1)/m_TB0911
D1_TB0911=Mac_Duration(C=C_TB0911, M=par, m=m_TB0911, y=y_TB0911, t=Tlist_TB0911)
print('2020年6月12日09附息国债11的麦考利久期',round(D1_TB0911,4))

# Example 7-8 债券价格变化金额≈到期收益率变化前的债券价格×麦考利久期×到期收益率变化金额
price_before=Bondprice_onediscount(C=C_TB0911, M=par, m=m_TB0911, y=y_TB0911, t=Tlist_TB0911) #计算到期收益率变化前的债券价格
print('2020年6月12日到期收益率变化前的09附息国债11价格',round(price_before,4))
y_change=0.0005 #债券到期收益率变化金额
price_change1=-D1_TB0911*price_before*y_change #用麦考利久期计算债券价格变化金额
print('用麦考利久期计算09附息国债11的价格变化金额',round(price_change1,4))
price_new1=price_before+price_change1 #用麦考利久期近似计算到期收益率变化后的债券价格
print('用麦考利久期近似计算到期收益率变化后的09附息国债11价格',round(price_new1,4))
# 运用债券定价公式计算精确的债券价格
price_new2=Bondprice_onediscount(C=C_TB0911, M=par, m=m_TB0911, y=y_TB0911+y_change, t=Tlist_TB0911) #精确计算到期收益率变化后的债券价格
print('精确计算2020年6月12日到期收益率变化后的09附息国债11价格',round(price_new2,4))

# Example 7-9 票面利率、到期收益率如何影响债券的麦考利久期
# 情形1: 09附息国债11的票面利率在[2%,6%]区间等差取值，其它参数保持不变
# 情形2: 09附息国债11的到期收益率在[1%,5%]区间等差取值，其它参数保持不变
C_list=np.linspace(0.02,0.06,200)
y_list=np.linspace(0.01,0.05,200)
D_list1=np.ones_like(C_list)
D_list2=np.ones_like(y_list)
for i in range(len(C_list)):
    D_list1[i]=Mac_Duration(C=C_list[i], M=par, m=m_TB0911, y=y_TB0911, t=Tlist_TB0911)
for i in range(len(y_list)):
    D_list2[i]=Mac_Duration(C=C_TB0911, M=par, m=m_TB0911, y=y_list[i], t=Tlist_TB0911)
plt.figure(figsize=(11,6))
plt.subplot(1,2,1)
plt.plot(C_list,D_list1,'r-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'票面利率',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'麦考利久期',fontsize=13)
plt.title(u'票面利率与麦考利久期的关系图',fontsize=13)
plt.grid()
plt.subplot(1,2,2,sharey=plt.subplot(1,2,1))
plt.plot(y_list,D_list2,'b-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'到期收益率',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'到期收益率与麦考利久期的关系图',fontsize=13)
plt.grid()
plt.show()

def Mod_Duration(C,M,m1,m2,y,t):
    '''定义一个计算债券修正久期的函数
    C: 代表债券的票面利率。
    M: 代表债券的面值。
    m1: 代表债券票息每年支付的频次。
    m2: 代表债券到期收益率每年复利频次，通常m2等于m1。
    y: 代表每年复利m2次的到期收益率。
    t: 代表定价日至后续每一次现金流支付日的期限长度，用数组格式输入；零息债券可直接输入数字'''
    if C==0: #针对零息债券
        Macaulay_duration=t
    else:
        r=m2*np.log(1+y/m2) #计算等价的连续复利到期收益率
        coupon=np.ones_like(t)*M*C/m1 #创建每一期票息金额的数组
        NPV_coupon=np.sum(coupon*np.exp(-r*t))
        NPV_par=M*np.exp(-r*t[-1])
        price=NPV_coupon+NPV_par
        cashflow=coupon
        cashflow[-1]+=M
        weight=cashflow*np.exp(-r*t)/price
        Macaulay_duration=np.sum(t*weight)
    Macaulay_duration/=(1+y/m2) #D*=D/(1+y_m/m)
    return Macaulay_duration
# Example 7-10 修正久期
def Rm(Rc,m):
    '''定义一个已知复利频次和连续复利利率，计算等价的复利利率的函数
    Rc: 代表连续复利利率。
    m: 代表复利频次'''
    r=m*(np.exp(Rc/m)-1)
    return r
y1_TB0911=Rm(Rc=y_TB0911,m=m_TB0911)
print('计算09附息国债11每年复利2次的到期收益率',round(y1_TB0911,6))
D2_TB0911=Mod_Duration(C=C_TB0911,M=par,m1=m_TB0911,m2=m_TB0911,y=y1_TB0911,t=Tlist_TB0911)
print('2020年6月12日09附息国债11的修正久期',round(D2_TB0911,4))
price_change2=-D2_TB0911*price_before*y_change #用修正久期计算债券价格变化
print('用修正久期计算09附息国债11价格变化',round(price_change2,4))
price_new3=price_before+price_change2
print('用修正久期近似计算到期收益率变化后的09附息国债11价格',round(price_new3,4))
def Rc(Rm,m):
    '''定义一个已知复利频次和对应利率，计算等价的连续复利利率的函数
    Rm: 代表复利频次为m的利率。
    m: 代表复利频次'''
    r=m*np.log(1+Rm/m)
    return r
yc_TB0911=Rc(Rm=y1_TB0911+y_change,m=m_TB0911)
print('计算09附息国债11新的连续复利到期收益率',round(yc_TB0911,6))
price_new4=Bondprice_onediscount(C=C_TB0911, M=par, m=m_TB0911, y=yc_TB0911, t=Tlist_TB0911)
print('精确计算每年复利2次的到期收益率变化后的09附息国债11价格',round(price_new4,4))

# dollar duration=Bondprice * modified duration =Bondprice * Macaulay duration / (1+Rm/m)
def Dollar_Duration(C,M,m1,m2,y,t):
    '''定义一个计算债券美元久期的函数
    C: 代表债券的票面利率。
    M: 代表债券的面值。
    m1: 代表债券票息每年支付的频次。
    m2: 代表债券到期收益率每年复利频次，通常m2等于m1（或m1为m2的整数n倍，即每复利了n期的票息合并支付一次）。
    y: 代表每年复利m2次的债券到期收益率。
    t: 代表定价日至后续每一期现金流支付日的期限长度，用数组格式输入；零息债券可直接输入数字'''
    r=m2*np.log(1+y/m2) #计算等价的连续复利到期收益率，用于麦考利久期的计算
    if C==0:
        price=M*np.exp(-r*t)
        Macaulay_D=t
    else:
        coupon=np.ones_like(t)*M*C/m1
        NPV_coupon=np.sum(coupon*np.exp(-r*t))
        NPV_par=M*np.exp(-r*t[-1])
        price=NPV_coupon+NPV_par #计算定价日的债券价格B
        cashflow=coupon
        cashflow[-1]+=M
        weight=cashflow*np.exp(-r*t)/price #计算时间的权重c_i*e^(-y_c*t_i)/B
        Macaulay_D=np.sum(t*weight) #计算带息债券的麦考利久期Σw_i*t_i
    Modified_D=Macaulay_D/(1+y/m2) #每年复利m2次的到期收益率，用于修正久期的计算
    Dollar_D=price*Modified_D
    return Dollar_D
# Example 7-11
D3_TB0911=Dollar_Duration(C=C_TB0911, M=par, m1=m_TB0911, m2=m_TB0911, y=y1_TB0911, t=Tlist_TB0911)
print('2020年6月12日09附息国债11的美元久期',round(D3_TB0911,2))

# Example 7-12 如果收益率变化较大（比如100个基点），利用久期得到的近似债券价格与实际债券价格之间的差异
y_newchange=0.01
y_new=y_TB0911+y_newchange #上升100个基点后的债券到期收益率
price_new5=Bondprice_onediscount(C=C_TB0911, M=par, m=m_TB0911, y=y_new, t=Tlist_TB0911) #精确计算到期收益率变化后的债券价格
print('精确计算到期收益率上升100个基点后的09附息国债11价格',round(price_new5,4))

def Convexity(C,M,m,y,t):
    '''定义一个计算债券凸性的函数
    C: 代表债券的票面利率。
    M: 代表债券的面值。
    m: 代表债券票息每年支付的频次。
    y: 代表债券的到期收益率（连续复利）。
    t: 代表定价日至后续每一期现金流支付日的期限长度，用数组格式输入；零息债券可直接输入数字'''
    if C==0:
        convexity=pow(t,2)
    else:
        coupon=np.ones_like(t)*M*C/m
        NPV_coupon=np.sum(coupon*np.exp(-y*t))
        NPV_par=M*np.exp(-y*t[-1])
        price=NPV_coupon+NPV_par
        cashflow=coupon
        cashflow[-1]+=M
        weight=cashflow*np.exp(-y*t)/price
        convexity=np.sum(pow(t,2)*weight)
    return convexity
# Example 7-13 09附息国债11的凸性
Convexity_TB0911=Convexity(C=C_TB0911, M=par, m=m_TB0911, y=y_TB0911, t=Tlist_TB0911)
print('2020年6月12日09附息国债11的凸性',round(Convexity_TB0911,4))

# Example 7-14 凸性的核心作用——更精确地衡量债券到期收益率变化（Δy）对债券价格（B）的影响
def Bondprice_change(B,D,C,y_chg):
    '''定义一个运用麦考利久期和凸性计算债券价格变化金额的函数
    B: 代表到期收益率变化之前的债券价格。
    D: 代表债券的麦考利久期。
    C: 代表债券的凸性。
    y_chg: 代表债券到期收益率的变化金额'''
    price_change1=-D*B*y_chg
    price_change2=0.5*C*B*pow(y_chg,2)
    price_change=price_change1+price_change2
    return price_change
price_change3=Bondprice_change(B=price_before, D=D1_TB0911, C=Convexity_TB0911, y_chg=y_newchange)
print('考虑麦考利久期和凸性之后的09附息国债11价格变化',round(price_change3,4))
price_new6=price_before+price_change3
print('考虑麦考利久期和凸性之后的09附息国债11最新价格',round(price_new6,4))

# Example 7-15 凸性对债券价格修正效应的可视化
y_change_list=np.linspace(-0.015,0.015,200) #创建到期收益率变化金额的等差数列
y_new_list=y_TB0911+y_change_list #变化后的到期收益率
price_change_list1=-D1_TB0911*price_before*y_change_list #仅用麦考利久期计算债券价格变化金额
price_new_list1=price_change_list1+price_before #仅用麦考利久期计算债券的新价格
price_change_list2=Bondprice_change(B=price_before,D=D1_TB0911,C=Convexity_TB0911,y_chg=y_change_list)
price_new_list2=price_change_list2+price_before #用麦考利久期和凸性计算债券的新价格
price_new_list3=np.ones_like(y_new_list) #创建存放债券定价模型计算的债券新价格初始数组
for i in range(len(y_new_list)):
    price_new_list3[i]=Bondprice_onediscount(C=C_TB0911, M=par, m=m_TB0911, y=y_new_list[i], t=Tlist_TB0911) #债券定价模型计算债券新价格
price_diff_list1=price_new_list1-price_new_list3 #仅用麦考利久期计算得到的债券新价格与债券定价模型得到的债券新价格之间的差异
price_diff_list2=price_new_list2-price_new_list3 #用麦考利久期和凸性计算得到的债券新价格与债券定价模型得到的债券新价格之间的差异
plt.figure(figsize=(9,6))
plt.plot(y_change_list,price_diff_list1,'b-',label=u'仅考虑麦考利久期',lw=2.5)
plt.plot(y_change_list,price_diff_list2,'m-',label=u'考虑麦考利久期和凸性',lw=2.5)
plt.xlabel(u'债券到期收益率的变化',fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel(u'与债券定价模型之间的债券价格差异',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'凸性对债券价格的修正效应',fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()

def default_prob(y1,y2,R,T):
    '''定义一个通过债券到期收益率计算连续复利违约概率的函数
    y1: 代表无风险零息利率（连续复利）。
    y2: 代表存在信用风险的债券到期收益率（连续复利）。
    R: 代表债券的违约回收率。
    T: 代表债券的期限（年）'''
    A=(np.exp(-y2*T)-R*np.exp(-y1*T))/(1-R)
    prob=-np.log(A)/T-y1
    return prob
# Example 7-16
T_yz=3
T_jj=5
y_yz=0.073611
y_jj=0.042471
R_yz=0.381
R_jj=0.699
rate_3y=0.02922
rate_5y=0.029811
default_yz=default_prob(y1=rate_3y, y2=y_yz, R=R_yz, T=T_yz)
default_jj=default_prob(y1=rate_5y, y2=y_jj, R=R_jj, T=T_jj)
print('16宜章养老债连续复利的违约概率',round(default_yz,4))
print('14冀建投连续复利的违约概率',round(default_jj,4))

# 考察债券到期收益率、债券违约回收率对违约概率的影响
y_jj_list=np.linspace(0.03,0.06,100)
default_jj_list1=default_prob(y1=rate_5y, y2=y_jj_list, R=R_jj, T=T_jj) #计算不同的到期收益率对应的违约概率
R_jj_list=np.linspace(0.4,0.8,100)
default_jj_list2=default_prob(y1=rate_5y, y2=y_jj, R=R_jj_list, T=T_jj) #计算不同的违约回收率对应的违约概率
plt.figure(figsize=(11,6))
plt.subplot(1,2,1)
plt.plot(y_jj_list,default_jj_list1,'r-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'债券到期收益率',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'违约概率',fontsize=13,rotation=90)
plt.title(u'债券到期收益率与违约概率的关系图',fontsize=14)
plt.grid()
plt.subplot(1,2,2)
plt.plot(R_jj_list,default_jj_list2,'b-',lw=2.5)
plt.xticks(fontsize=13)
plt.xlabel(u'债券违约回收率',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'债券违约回收率与违约概率的关系图',fontsize=14)
plt.grid()
plt.show()