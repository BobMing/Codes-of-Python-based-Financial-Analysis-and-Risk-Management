# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:15:50 2023

@author: XIE Ming
"""
import matplotlib
matplotlib.__version__

import matplotlib.pyplot as plt

from pylab import mpl #从pylab导入子模块mpl
mpl.rcParams['font.sans-serif']=['FangSong'] #以仿宋字体显示中文
mpl.rcParams['axes.unicode_minus']=False #在图像中正常显示负号

plt.rcParams['font.family']=['FangSong','FangSong_GB2312','STSong','Arial Unicode MS']
plt.rcParams['font.sans-serif']=['FangSong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters #导入注册日期时间转换器的函数
register_matplotlib_converters() #注册日期时间转换器

# Example 4-1 单一曲线图：可视化住房按揭贷款等额本息还款的每月还款金额
import numpy_financial as npf
r=0.05 #贷款年利率
n=30 #贷款的期限（年）
principle=8e6 #贷款的本金
pay_month=npf.pmt(rate=r/12,nper=n*12,pv=principle,fv=0,when='end') #计算每月支付的本息之和
print('每月偿还的金额',round(pay_month,2))
T_list=np.arange(n*12)+1 #生成一个包含每次还款期限的数组
prin_month=npf.ppmt(rate=r/12,per=T_list,nper=n*12,pv=principle,fv=0,when='end') #计算每月偿还的本金金额
inte_month=npf.ipmt(rate=r/12,per=T_list,nper=n*12,pv=principle,fv=0,when='end') #计算每月偿还的利息金额
pay_month_list=pay_month*np.ones_like(prin_month) #创建每月偿还金额的数组
plt.figure(figsize=(9,6),frameon=False) #宽9高6（英寸），显示边框
plt.plot(T_list,-pay_month_list,'r-',label=u'每月偿还金额',lw=2.5) #横坐标填每次还款期限数组，纵坐标填每月偿还总额，红色实线，曲线宽2.5磅
plt.plot(T_list,-prin_month,'m--',label=u'每月偿还本金金额',lw=2.5) #纵坐标填每月偿还本金，品红短画线，曲线宽2.5磅
plt.plot(T_list,-inte_month,'b--',label=u'每月偿还利息金额',lw=2.5) #纵坐标填每月偿还利息，蓝色短画线，曲线宽2.5磅
plt.xticks(fontsize=14) #x轴刻度字体大小14磅
plt.xlim(0,360)
plt.xlabel(u'逐次偿还的期限（月）',fontsize=14) #x轴坐标该内容、标签字体大小14磅
plt.yticks(fontsize=13)
plt.ylabel(u'金额',fontsize=13)
plt.title(u'等额本息还款规则下每月偿还的金额以及本金与利息',fontsize=14)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.show()

r_list=np.linspace(0.03,0.07,100) #模拟不同的贷款利率[3%,7%]
pay_month_list=npf.pmt(rate=r_list/12,nper=n*12,pv=principle,fv=0,when='end') #计算不同贷款利率条件下的每月偿还本息之和
plt.figure(figsize=(9,6))
plt.plot(r_list,-pay_month_list,'r-',label=u'每月偿还金额',lw=2.5)
plt.plot(r,-pay_month,'o',label=u'贷款利率5%对应的每月偿还金额',lw=2.5)
plt.xticks(fontsize=14)
plt.xlabel(u'贷款利率',fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(u'金额',fontsize=14)
plt.annotate(u'贷款利率等于5%',fontsize=14,xy=(0.05,43000),xytext=(0.045,48000),arrowprops=dict(facecolor='m',shrink=0.05)) #添加带箭头的注释
plt.title(u'不同贷款利率与每月偿还金额之间的关系',fontsize=14)
plt.legend(loc=0,fontsize=14)
plt.grid()
plt.show()

# Example 4-2 多图绘制
SZ_Index=pd.read_excel('E:/OneDrive/附件/数据/第4章/深证成指每日价格数据（2018-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
SZ_Index.index #显示行索引的格式
SZ_Index.index=pd.DatetimeIndex(SZ_Index.index) #将数据框的行索引转换为Datetime格式
SZ_Index.index
plt.figure(figsize=(11,9))
plt.subplot(2,2,1) #第1张子图
plt.plot(SZ_Index['开盘价'],'r-',label=u'深证成指开盘价',lw=2.0)
plt.xticks(fontsize=13,rotation=30)
plt.xlabel(u'日期',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'价格',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.subplot(2,2,2) #第2张子图
plt.plot(SZ_Index['最高价'],'b-',label=u'深证成指最高价',lw=2.0)
plt.xticks(fontsize=13,rotation=30)
plt.xlabel(u'日期',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'价格',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.subplot(2,2,3) #第3张子图
plt.plot(SZ_Index['最低价'],'c-',label=u'深证成指最低价',lw=2.0)
plt.xticks(fontsize=13,rotation=30)
plt.xlabel(u'日期',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'价格',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.subplot(2,2,4) #第4张子图
plt.plot(SZ_Index['收盘价'],'k-',label=u'深证成指收盘价',lw=2.0)
plt.xticks(fontsize=13,rotation=30)
plt.xlabel(u'日期',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'价格',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.show()

# Example 4-3 单一样本直方图
import numpy.random as npr
I=10000 #随机抽样次数1W
x_norm=npr.normal(loc=0.8,scale=1.6,size=I) #从均值等于0.8、标准差等于1.6的正态分布中随机抽样
x_logn=npr.lognormal(mean=0.5,sigma=1.0,size=I) #从均值等于0.5、标准差等于1.0的对数正态分布中随机抽样
x_chi=npr.chisquare(df=5,size=I) #从自由度等于5的卡方分布中随机抽样
x_beta=npr.beta(a=2,b=6,size=I) #从α=2、β=6的贝塔分布中随机抽样
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.hist(x_norm,label=u'正态分布的抽样',bins=20,facecolor='y',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.subplot(2,2,2)
plt.hist(x_logn,label=u'对数正态分布的抽样',bins=20,facecolor='r',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.subplot(2,2,3)
plt.hist(x_chi,label=u'卡方分布的抽样',bins=20,facecolor='b',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.subplot(2,2,4)
plt.hist(x_beta,label=u'贝塔分布的抽样',bins=20,facecolor='c',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.show()

# Example 4-4 多个样本的直方图：堆叠（stacked）展示
SH_SZ_Index=pd.read_excel('E:/OneDrive/附件/数据/第4章/上证综指和深证成指的日涨跌幅数据（2019-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
SH_SZ_Index=np.array(SH_SZ_Index) #将数据框格式转为数组格式
plt.figure(figsize=(9,6))
plt.hist(SH_SZ_Index,label=[u'上证综指日涨跌幅',u'深证成指日涨跌幅'],stacked=True,edgecolor='k',bins=30) #两组样本值堆叠展示
plt.xticks(fontsize=13)
plt.xlabel(u'日涨跌幅',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13)
plt.title(u'上证综指和深证成指日涨跌幅堆叠的直方图',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.show()

# Example 4-5 多个样本的直方图：并排展示
plt.figure(figsize=(9,6))
plt.hist(SH_SZ_Index,label=[u'上证综指日涨跌幅',u'深证成指日涨跌幅'],stacked=False,edgecolor='k',bins=30) #两组样本值堆叠展示
plt.xticks(fontsize=13)
plt.xlabel(u'日涨跌幅',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13)
plt.title(u'上证综指和深证成指日涨跌幅并排的直方图',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.show()

# Example 4-6 垂直条形图，柱形图（column chart）
R_array=np.array([[-0.035099,0.017230,-0.003450,-0.024551,0.039368],[-0.013892,0.024334,-0.033758,0.014622,0.000128],[0.005848,-0.002907,0.005831,0.005797,-0.005764],[0.021242,0.002133,-0.029803,-0.002743,-0.014301]]) #创建4股5日涨跌幅数组
date=['2020-05-25','2020-05-26','2020-05-27','2020-05-28','2020-05-29'] #创建交易日列表
name=['中国卫星','中国软件','中国银行','上汽集团'] #创建股票名称列表
R_dataframe=pd.DataFrame(data=R_array.T,index=date,columns=name) #创建数据框
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.bar(x=R_dataframe.columns,height=R_dataframe.iloc[0],width=0.5,label=u'2020年5月25日涨跌幅',facecolor='y')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'日涨跌幅',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.subplot(2,2,2,sharex=plt.subplot(2,2,1),sharey=plt.subplot(2,2,1)) #与第1个字图的x轴和y轴相同
plt.bar(x=R_dataframe.columns,height=R_dataframe.iloc[1],width=0.5,label=u'2020年5月26日涨跌幅',facecolor='c')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'日涨跌幅',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.subplot(2,2,3,sharex=plt.subplot(2,2,1),sharey=plt.subplot(2,2,1))
plt.bar(x=R_dataframe.columns,height=R_dataframe.iloc[3],width=0.5,label=u'2020年5月28日涨跌幅',facecolor='b')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'日涨跌幅',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.subplot(2,2,4,sharex=plt.subplot(2,2,1),sharey=plt.subplot(2,2,1))
plt.bar(x=R_dataframe.columns,height=R_dataframe.iloc[4],width=0.5,label=u'2020年5月29日涨跌幅',facecolor='g')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'日涨跌幅',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.show()

# Example 4-7 水平条形图
plt.figure(figsize=(9,6))
plt.barh(y=R_dataframe.columns,width=R_dataframe.iloc[1],height=0.5,label=u'2020年5月26日涨跌幅')
plt.barh(y=R_dataframe.columns,width=R_dataframe.iloc[2],height=0.5,label=u'2020年5月27日涨跌幅')
plt.xticks(fontsize=13)
plt.xlabel(u'涨跌幅',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'水平条形图可视化股票的涨跌幅',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid()
plt.show()

# Example 4-8 双（y）轴图
M2=pd.read_excel('E:/OneDrive/附件/数据/第4章/我国广义货币供应量M2的数据（2019-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
fig, ax1=plt.subplots(figsize=(9,6)) #运用左轴纵坐标绘制图形
plt.bar(x=M2.index,height=M2.iloc[:,0],color='y',label=u'M2每月余额')
plt.xticks(fontsize=13,rotation=90)
plt.xlabel(u'日期',fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,250)
plt.ylabel(u'金额（万亿元）',fontsize=13)
plt.legend(loc=2,fontsize=13) #图例位置设置在左上方
ax2=ax1.twinx() #运用右侧纵坐标绘制图形
plt.plot(M2.iloc[:,-1],label=u'M2每月同比增长率',lw=2.5)
plt.yticks(fontsize=13)
plt.ylim(0,0.13)
plt.ylabel(u'增长率',fontsize=13)
plt.title(u'广义货币供应量M2每月余额和每月同比增长率',fontsize=13)
plt.legend(loc=1,fontsize=13) #图例位置设置在右上方
plt.grid()
plt.show()

# Example 4-9 散点图（scatter plot）
ICBC_CCB=pd.read_excel('E:/OneDrive/附件/数据/第4章/工商银行与建设银行A股周涨跌幅数据（2016-2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
ICBC_CCB.describe() #查看数据框的描述性统计
ICBC_CCB.corr() #工商银行与建设银行周涨跌幅的相关数据
plt.figure(figsize=(9,6))
plt.scatter(x=ICBC_CCB['工商银行'],y=ICBC_CCB['建设银行'],c='r',marker='o')
plt.xticks(fontsize=13)
plt.xlabel(u'工商银行周涨跌幅',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'建设银行周涨跌幅',fontsize=13)
plt.title(u'工商银行与建设银行周涨跌幅的散点图',fontsize=13)
plt.grid()
plt.show()

# Example 4-10 饼图（pie chart）
currency=['美元','欧元','人民币','日元','英镑'] #创建存放币种名称的列表
perc=[0.4173, 0.3093, 0.1092, 0.0833, 0.0809] #创建存放不同货币权重的列表
plt.figure(figsize=(9,7))
plt.pie(x=perc,labels=currency,textprops={'fontsize':13},autopct='%1.1f%%')
plt.axis('equal') #使饼图是一个圆形，equal即x轴、y轴的比例相等
plt.title(u'特别提款权中不同币种的比重',fontsize=13) #参数textprops用于控制饼图中标签的字体大小
plt.legend(loc=2,fontsize=13) #图例在左上方
plt.show()

# Example 4-11 雷达图（radar chart）戴布拉图、网络图、蜘蛛图、星图
company=['中国人寿','中国人保','中国太保','中国平安','新华保险'] #创建存放公司名称的列表
indicator=['营业收入增长率','净利润增长率','净资产收益率','偿付能力充足率'] #创建存放指标名称的列表
ranking=np.array([5,4,3,2]) #创建存放中国太保各项指标排名的数组
N_company=len(company)
N_indicator=len(indicator)
ranking_new=np.concatenate([ranking,[ranking[0]]]) #在中国太保各项指标排名的数组末尾增加一个该数组的首位数字，以实现绘图的闭合
angles=np.linspace(0,2*np.pi,N_indicator,endpoint=False) #将圆形按照指标数量进行均匀切分
angles_new=np.concatenate([angles,[angles[0]]]) #在已创建的angles数组的末尾增加一个该数组的首位数字，以实现绘图的闭合
plt.figure(figsize=(8,8))
plt.polar(angles_new,ranking_new,'--') #绘制雷达图
plt.thetagrids(angles*180/np.pi,indicator,fontsize=13) #绘制圆形的指标名称
plt.ylim(0,5)
plt.yticks(range(N_company+1),fontsize=13) #刻度按照公司数量设置
plt.fill(angles_new,ranking_new,facecolor='r',alpha=0.3) #对图中相关部分用颜色填充
plt.title(u'中国太保各项指标在5家A股上市保险公司中的排名',fontsize=13)
plt.show()

# Example 4-12 K线图，蜡烛图（candlestick cahrt）
import mplfinance as mpf #导入mplfinance，自Matplotlib 2.0起独立出来形成新的第三方模块
mpf.__version__
SH_Index=pd.read_excel('E:/OneDrive/附件/数据/第4章/2020年第3季度上证综指的日交易数据.xlsx',sheet_name='Sheet1',header=0,index_col=0)
SH_Index.index=pd.DatetimeIndex(SH_Index.index) #数据框的行索引转换为Datetime格式
SH_Index.columns #显示数据框的列名
SH_Index=SH_Index.rename(columns={'开盘价':'Open','最高价':'High','最低价':'Low','收盘价':'Close','成交额(万亿元)':'Volume'}) #将数据框的列名调整为英文
mpf.plot(data=SH_Index,type='candle',mav=5,volume=True,figratio=(9,7),style='classic',ylabel='price',ylabel_lower='volume(trillion)') #绘制经典风格的K线图
"""
data: 数据框格式，且①行索引须是Datetime格式；②列名须依次用 Open,High,Low,Close,Volume 等英文字母表示
type: ohlc 条形图（默认） candle 蜡烛图 line 折线图 renko 砖型图 pnf OX图或点数图（point and figure chart）
mav: （若干条）均线，e.g. mav=5 表示生成5日均线（以日K线为例） mav=(5,10) 表示分别生成5日均线和10日均线
volume: 默认=True，绘制交易量
figratio: 相当于figsize
style: K线图的图案风格，有9种可供选择；还可使用make_marketcolors和make_mpf_style自定义阳线、阴线等图案颜色
ylabel: y轴的坐标标签
ylabel_lower: 对应绘制交易量图形的y轴坐标标签
"""
color=mpf.make_marketcolors(up='r',down='g') #设置阳线用红色表示、阴线用绿色表示
style_color=mpf.make_mpf_style(marketcolors=color) #运用make_mpf_style函数设置图案风格
mpf.plot(data=SH_Index,type='candle',mav=(5,10),volume=True,figratio=(9,6),style=style_color,ylabel='price',ylabel_lower='volume(trillion)') #绘制自定义K线图

