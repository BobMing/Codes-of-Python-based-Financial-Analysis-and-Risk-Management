# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:56:07 2023

@author: XIE Ming
"""

import pandas as pd
pd.__version__
pd.set_option('display.unicode.ambiguous_as_wide',True) #处理数据的列名与其对应列数据无法对齐的情况
pd.set_option('display.unicode.east_asian_width',True) #无法对齐主要是因为列名是中文
pd.options.display.max_columns=12 #展示12列数据，若超过则以省略号显示
pd.options.display.max_rows #=10，默认60
pd.options.display.width=200 #界面宽度扩展值20000，默认80

# Example 3-1 series
name=['中国卫星','中国软件','中国银行','上汽集团']
list_May25=[-0.035099,-0.013892,0.005848,0.021242]
series_May25=pd.Series(data=list_May25,index=name)
print(series_May25)

import numpy as np
return_array1=np.array([[-0.035099, 0.017230, -0.003450, -0.024551, 0.039368],
                       [-0.013892, 0.024334, -0.033758, 0.014622, 0.000128],
                       [0.005848, -0.002907, 0.005831, 0.005797, -0.005764],
                       [0.021242, 0.002133, -0.029803, -0.002743, -0.014301]]) #输入日涨跌幅数据
series_May27=pd.Series(data=return_array1[:,2],index=name) #通过数组创建序列
print(series_May27)

date=['2020-05-25','2020-05-26','2020-05-27','2020-05-28','2020-05-29']
series_BOC=pd.Series(data=return_array1[2,:],index=date) #创建中国银行的序列（运用日期作索引）
print(series_BOC)

# Example 3-2 dataframe
return_dataframe=pd.DataFrame(data=return_array1.T,index=date,columns=name)
print(return_dataframe)

# Example 3-3 导出存放
return_dataframe.to_excel('.\四只股票涨跌幅数据.xlsx') #默认.\位置是 C:\Users\Administrator\
return_dataframe.to_csv('.\四只股票涨跌幅数据.csv')
return_dataframe.to_csv('.\四只股票涨跌幅数据.txt')

# Example 3-4 从外部文件导入数据
SH_Index=pd.read_excel('E:/OneDrive/附件/数据/第3章/上证综指每个交易日价格数据（2020年）.xlsx',sheet_name="Sheet1",header=0,index_col=0)
SH_Index.head() #显示开头5行
SH_Index.tail() #显示末尾5行

# Example 3-5 从金融数据终端与Python的API导入数据，以yfinance模块为例
"""
yf.download(tickers = "SPY AAPL",  # list of tickers
            period = "1y",         # time period, either set period parameter or use start and end(not included the end timeline)
            interval = "1d",       # trading interval
            prepost = False,       # download pre/post market hours data?
            repair = True)         # repair obvious price errors e.g. 100x?
"""
import yfinance as yf
yf.__version__
SH_Index_yFinance=yf.download('000001.SS',start='2020-01-01',end='2021-01-01',interval='1d')[['Open','High','Low','Close']]
SH_Index_yFinance

# Example 3-6 创建2021年至2022年每个工作日的时间数列
time1=pd.date_range(start='2021-01-01',end='2022-12-31',freq='B')
time1

# Example 3-7 创建2021年1月4日交易日上午9点30分开始以秒为频次且包含7200个时间元素的时间数列
time2=pd.date_range(start='2021-01-04 09:30:00',periods=7200,freq='S')
time2

from pylab import mpl #从pylab导入子模块mpl
mpl.rcParams['font.sans-serif']=['FangSong'] #以仿宋字体显示中文
mpl.rcParams['axes.unicode_minus']=False #在图像中正常显示负号“-”

# Example 3-8 对例3-4中创建的数据框进行可视化
SH_Index.plot(kind='line',subplots=True,sharex=True,sharey=True,layout=(2,2),figsize=(11,9),title=u'2020年上证综指每个交易日价格走势图',grid=True,fontsize=13)

# Example 3-9
SH_Index.index #查看数据框的行索引名
SH_Index.columns #查看数据框的列名

# Example 3-10
SH_Index.shape #查看数据框的行数(不包括列名)和列数（不包括行索引）

# Example 3-11
SH_Index.describe() #查看数据框的基本统计指标

SH_Index.loc['2020-02-18'] #行索引，查看2020年2月18日的数据
SH_Index.iloc[7] #行号，查看第8行的数据

# Example 3-12 一般性截取
SH_Index[:5] #截取数据框前5行（不含列名）的数据
SH_Index[7:12] #截取数据框第8行至第12行的数据
SH_Index.iloc[16:19,1:3] #截取第17行至第19行及第2、3列（不含行索引）的数据
SH_Index.loc['2020-05-18':'2020-05-22'] #截取2020年5月18日至22日的数据

# Example 3-13 条件性截取
SH_Index[SH_Index['收盘价']>=3450] #截取收盘价超过3450点的数据

# Example 3-14
SH_Index[(SH_Index['最高价']>=3440)&(SH_Index['最低价']<=3380)] #截取最高价超过3440点且最低价低于3380点的数据

# Example 3-15 按行索引的大小排序
SH_Index.sort_index(ascending=True) #按照交易日由远到近（升序）排序
SH_Index.sort_index(ascending=False) #按照交易日由近到远（降序）排序

# Example 3-16 按列名对应的数值大小排序
SH_Index.sort_values(by='开盘价',ascending=True) #按照开盘价由小到大（升序）排序
SH_Index.sort_values(by='收盘价',ascending=False) #按照收盘价由大到小（降序）排序

# Example 3-17 修改行索引与列名
SH_Index_new=SH_Index.rename(index={'2020-01-02':'2020年1月2日'}) #修改行索引
SH_Index_new=SH_Index_new.rename(columns={'收盘价':'收盘点位'}) #修改列名
SH_Index_new.head() #显示列名修改后的前5行

# Example 3-18 缺失值的查找
Index_global=pd.read_excel('E:/OneDrive/附件/数据/第3章/全球主要股指2020年4月日收盘价数据.xlsx',sheet_name='Sheet1',header=0,index_col=0)
Index_global.isnull().any() #用isnull函数查找每一列是否存在缺失值
Index_global.isna().any() #用isna函数查找每一列是否存在缺失值
Index_global[Index_global.isnull().values==True] #用isnull函数查找缺失值所在行
Index_global[Index_global.isna().values==True] #用isna函数查找缺失值所在行

# Example 3-19 缺失值的处理
Index_dropna=Index_global.dropna() #直接删除法
Index_dropna
Index_fillzero=Index_global.fillna(value=0) #零值补齐法
Index_fillzero
Index_ffill=Index_global.fillna(method='ffill') #前值补齐法
Index_ffill
Index_bfill=Index_global.fillna(method='bfill') #后值补齐法
Index_bfill

# Example 3-20 创建新的数据框1:2019上证综指日交易价
SH_Index_2019=pd.read_excel('E:/OneDrive/附件/数据/第3章/上证综指每个交易日价格数据（2019年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
SH_Index_2019.head()
SH_Index_2019.tail()
# Example 3-21 创建新的数据框2:2020上证综指日交易价
SH_Index_volume=pd.read_excel('E:/OneDrive/附件/数据/第3章/上证综指每个交易日成交额与总市值数据（2020年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)
SH_Index_volume.head()
SH_Index_volume.tail()

# Example 3-22 concat函数 按行合并
SH_Index_new1=pd.concat([SH_Index_2019,SH_Index],axis=0) #按行合并
SH_Index_new1.head()
SH_Index_new1.tail()

# Example 3-23 concat函数 按列合并
SH_Index_new2=pd.concat([SH_Index,SH_Index_volume],axis=1) #按列合并
SH_Index_new2.head()
SH_Index_new2.tail()

# Example 3-24 merge函数
SH_Index_new3=pd.merge(left=SH_Index,right=SH_Index_volume,left_index=True,right_index=True) #按列合并,2个数据框的行索引完全相同，所以采用了同时按照2个数据框的行索引进行合并
SH_Index_new3.head()
SH_Index_new3.tail()

# Example 3-25 join函数
SH_Index_new4=SH_Index.join(SH_Index_volume,on='日期') #按列合并，on表示按照某（几）个索引进行合并，默认情况是据2个数据框的行索引进行合并
SH_Index_new4.head()
SH_Index_new4.tail()

# Table 3-7 静态统计函数
SH_Index_new1.min() #查看（各列）最小值
SH_Index_new1.idxmin() #查看最小值的行索引值
SH_Index_new1.max()
SH_Index_new1.idxmax()
SH_Index_new1.median() #查看中位数
SH_Index_new1.quantile(q=0.05) #查看5%的分位数
SH_Index_new1.quantile(q=0.5) #查看50%的分位数（中位数）
SH_Index_new1.mean()
SH_Index_new1.var()
SH_Index_new1.std()
SH_Index_new1.skew() #计算偏度 skewness,>0表示分布具有正/右偏态，<0则表示数据分布具有负/左偏态
SH_Index_new1.kurt() #计算峰度 kurtosis，>0表示数据分布相比正态分布更加陡峭，<0则表示数据分布相比正态分布更加扁平
SH_Index_shift1=SH_Index_new1.shift(1) #每行均向下移动一行
SH_Index_shift1.head()
SH_Index_shift1.tail()
SH_Index_diff=SH_Index_new1.diff() #计算一阶差分(各行当前值减上一行对应列值)
SH_Index_diff.head()
SH_Index_perc=SH_Index_new1.pct_change() #计算百分比（环比）变化
SH_Index_perc.head()
SH_Index_perc=SH_Index_perc.dropna()
SH_Index_perc.sum() #对百分比变化的数据框求和
SH_Index_cumsum=SH_Index_perc.cumsum() #对百分比变化的数据框累积求和 cumulative sum
SH_Index_cumsum.head() #查看前5行
SH_Index_chag=SH_Index_perc+1
SH_Index_cumchag=SH_Index_chag.cumprod() #对新数据框累积求积 cumulative product
SH_Index_cumchag.head()
SH_Index_perc.cov() #计算协方差 covariance

# Example 3-26 移动平均（移动窗口的均值）
SH_Index_MA10=SH_Index_new1['收盘价'].rolling(window=10).mean() #创建10日均值收盘价的序列，默认axis=0按行实现函数
SH_Index_MA10=SH_Index_MA10.to_frame() #将序列变成数据框 Series不支持更改列名
SH_Index_MA10=SH_Index_MA10.rename(columns={'收盘价':'10日平均收盘价（MA10）'}) #修改数据框列名
SH_Index_close=SH_Index_new1['收盘价'].to_frame() #创建一个每日收盘价的数据框
SH_Index_new5=pd.concat([SH_Index_close,SH_Index_MA10],axis=1) #合并成一个包括每日收盘价、10日均值收盘价的数据框
SH_Index_new5.plot(figsize=(9,6),title=u'2019-2020年上证综指走势',grid=True,fontsize=13)

# Example 3-27 移动波动率（移动窗口的标准差）
SH_Index_rollstd=SH_Index_new1['收盘价'].rolling(window=30).std() #创建30日移动波动率的序列
SH_Index_rollstd=SH_Index_rollstd.to_frame() #将序列变成数据框
SH_Index_rollstd=SH_Index_rollstd.rename(columns={'收盘价':'30日收盘价的移动波动率'}) #修改数据框列名
SH_Index_rollstd.plot(figsize=(9,6),title=u'2019-2020年上证综指移动波动率的走势',grid=True,fontsize=12)

# Example 3-28 移动相关系数（移动窗口的相关性矩阵）
SH_Index_rollcorr=SH_Index_new1.rolling(window=60).corr() #计算移动相关系数
SH_Index_rollcorr=SH_Index_rollcorr.dropna() #删除缺失值
SH_Index_rollcorr.head()
SH_Index_rollcorr.tail()
