# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:05:34 2023

@author: XIE Ming

Chapter 2
"""

# 例2-1
return_May25=[-0.035099, -0.013892, 0.005848, 0.021242] #2020年5月25日4支股票的日涨跌幅
weight_list=[0.15, 0.20, 0.25, 0.40]                    #4只股票的配置权重
n=len(weight_list)                                      #股票数量
return_weight=[]    #创建存放每只股票收益率与配置权重数乘积的空列表
for i in range(n):
    return_weight.append(return_May25[i]*weight_list[i])    #将计算结果存放在列表末尾
return_port_May25=sum(return_weight)    #计算2020年5月25日投资组合的收益率
print("2020年5月25日投资组合的收益率",round(return_port_May25, 6))

import numpy as np
np.__version__  #1.23.5

# 例2-2
weight_array1=np.array([0.15, 0.2, 0.25, 0.4])
type(weight_array1) # numpy.ndarray
weight_array1.shape # (4,)  →一维（4行）数组

# 例2-3
return_array1=np.array([[-0.035099, 0.017230, -0.003450, -0.024551, 0.039368],
                       [-0.013892, 0.024334, -0.033758, 0.014622, 0.000128],
                       [0.005848, -0.002907, 0.005831, 0.005797, -0.005764],
                       [0.021242, 0.002133, -0.029803, -0.002743, -0.014301]]) #输入日涨跌幅数据
return_array1
return_array1.shape     # (4, 5) 一个4行5列的二维数组

# 例2-4
weight_array2=np.array(weight_list) #将列表转换为一维数组
weight_array2

# 例2-5
return_list=[-0.035099, 0.017230, -0.003450, -0.024551, 0.039368,
             -0.013892, 0.024334, -0.033758, 0.014622, 0.000128,
             0.005848, -0.002907, 0.005831, 0.005797, -0.005764,
             0.021242, 0.002133, -0.029803, -0.002743, -0.014301]   #以列表格式输入日涨跌幅数据
return_array2=np.array(return_list)         #转换为一维数组
return_array2=return_array2.reshape(4,5)    #转换为4行5列的二维数组
return_array2   #查看输出结果

# 例2-6
a=np.arange(10) #创建0~9的整数数列
a
b=np.arange(1,18,3) #创建1~18且步长为3的整数数列
b

# 例2-7
c=np.linspace(0,100,51) #创建等差数列的数组
c
len(c)  #51

# 例2-8
zero_array1=np.zeros(8) #创建一个8个元素的一维零数组
zero_array1

# 例2-9
zero_array2=np.zeros((5,7)) #创建一个5×7的二维数组
zero_array2

# 例2-10
zero_weight=np.zeros_like(weight_array1) #创建与weight_array1相同形状的零数组
zero_weight

zero_return=np.zeros_like(return_array1) #创建与return_array1相同形状的零数组
zero_return

# 例2-11
one_weight=np.ones_like(weight_array1) #创建与weight_array1相同形状且元素均为1的数组
one_weight

one_return=np.ones_like(return_array1) #创建与return_array1相同形状且元素为1的数组
one_return

# 例2-12
d=np.eye(6) #创建单位矩阵
d

# 例2-13
return_array1[1,3] #索引第2行、第4列的元素

# 例2-14
np.where(return_array1>0.014) #涨幅超过1.4%的元素的索引值，输出的两个数组分别存放结果们的行、列索引值

# 例2-15
return_array1[1:3,1:4] #提取第2、3行中第2至4列的数据

# 例2-16
return_array1[1]    #提取第2行整行数据

return_array1[:,2]  #提取第3列整列数据

# 例2-17
np.sort(return_array1,axis=0)   #按列对元素升序排序，axis=0代表列内排序，axis=1代表行内排序

np.sort(return_array1,axis=1)   #按行对元素升序排序

np.sort(return_array1)  #默认按行（axis=1）升序排序

# 例2-18 运用append函数合并数组
return_BOC=np.array([0.005848,-0.002907,0.005831,0.005797,-0.005764]) #中国银行的日涨跌幅数据
return_SAIC=np.array([0.021242,0.002133,-0.029803,-0.002743,-0.014301]) #上汽集团的日涨跌幅数据
return_2stock=np.append([return_BOC],[return_SAIC],axis=0) #按列合并（↓列取各数组最大值，依次向下，行数为各数组行数之和）
return_2stock

return_2stock_new1=np.append([return_BOC],[return_SAIC],axis=1) #按行合并（→行取各数组最大值，依次向右，列数为各数组列数之和）
return_2stock_new1

return_2stock_new2=np.append([return_BOC],[return_SAIC]) #默认按行（axis=1）合并
return_2stock_new2

# 例2-19 运用concatenate函数合并数组
return_CAST=np.array([-0.035099,0.017230,-0.003450,-0.024551,0.039368]) #中国卫星的日涨跌幅数据
return_CSS=np.array([-0.013892,0.024334,-0.033758,0.014622,0.000128]) #中国软件的日涨跌幅数据
return_4stock=np.concatenate(([return_CAST], [return_CSS], [return_BOC], [return_SAIC]), axis=0) #按列合并
return_4stock

return_4stock_new1=np.concatenate(([return_CAST], [return_CSS], [return_BOC], [return_SAIC]), axis=1) #按行合并
return_4stock_new1

return_4stock_new2=np.concatenate(([return_CAST], [return_CSS], [return_BOC], [return_SAIC])) #默认按行（axis=0）合并
return_4stock_new2

# 例2-20 数组求和
return_array1.sum(axis=0) #按列求和，每列之和，元素数为列数5

return_array1.sum(axis=1) #按行求和，每行之和，元素数为行数4

return_array1.sum() #默认为全部元素之和

# 例2-21 数组求乘积
return_array1.prod(axis=0) #按列求乘积，每列之积，元素数为列数5

return_array1.prod(axis=1) #按行求乘积，每行之积，元素数为行数4

return_array1.prod() #默认为全部元素之积

# 例2-22 数组求最值
return_array1.max(axis=0) #按列求最大值，结果为5元素
return_array1.max(axis=1) #按行求最大值，结果为4元素
return_array1.max() #默认求全部元素的最大值

return_array1.min(axis=0) #按列求最小值，结果为5元素
return_array1.min(axis=1) #按行求最小值，结果为4元素
return_array1.min() #默认求全部元素的最小值

# 例2-23 数组求均值
return_array1.mean(axis=0) #按列求均值
return_array1.mean(axis=1) #按行求均值
return_array1.mean() #默认求全部元素的均值

# 例2-24 数组求方差和标准差
return_array1.var(axis=0) #按列求方差
return_array1.var(axis=1) #按行求方差
return_array1.var() #默认求全部元素的方差

return_array1.std(axis=0) #按列求标准差
return_array1.std(axis=1) #按行求标准差
return_array1.std() #默认求全部元素的标准差

# 例2-25 数组内幂运算
np.sqrt(return_array1) #对每个元素计算开（平）方
np.square(return_array1) #对每个元素计算平方
np.exp(return_array1) #对每个元素计算以e为底的指数次方

# 例2-26 数组内对数运算
np.log(return_array1) #对每个元素计算自然对数ln(x)
np.log2(return_array1) #对每个元素计算底数为2的对数
np.log10(return_array1) #对每个元素计算底数为10的对数
np.log1p(return_array1) #对(1+每个元素)计算自然对数ln(1+x)即ln(x)图像往左平移1个单位

# 例2-27
new_array1=return_array1+one_return #两个二维数组相加
new_array1

new_array2=return_array1-one_return #两个二维数组相减
new_array2

new_array3=new_array1*new_array2 #两个新的二维数组相乘
new_array3

new_array4=new_array1/new_array2 #两个新的二维数组相除
new_array4

new_array5=new_array1**new_array2 #两个新的二维数组进行幂运算
new_array5

new_array6=pow(new_array1,new_array2) #两个新的二维数组之间的幂运算采用pow函数
new_array6==new_array5

new_array7=new_array6+np.array([1,0,1,0,1]) #二维数组与一维数组相加，一维数组每个元素与对应列的每行元素依次运算
new_array7

# 例2-28
new_array8=return_array1+1 #数组的每个元素均加上1
new_array8

new_array9=return_array1-1 #数组的每个元素均减去1
new_array9

new_array10=return_array1*2 #数组的每个元素均乘2
new_array10

new_array11=return_array1/2 #数组的每个元素均除以2
new_array11

new_array12=return_array1**2 #数组的每个元素均进行平方
new_array12

new_array13=pow(return_array1,2) #数组的每个元素均进行平方采用pow函数
new_array13

# 例2-29
return_max=np.maximum(return_array1,zero_return) #创建以两个数组对应元素的最大值作为元素的新数组
return_max

return_min=np.minimum(return_array1,zero_return) #创建以两个数组对应元素的最小值作为元素的新数组
return_min

# 例2-30
corr_return=np.corrcoef(return_array1) #计算相关系数(correlation coefficient)
corr_return

np.diag(corr_return) #查看矩阵的对角线diagonal line
np.triu(corr_return) #查看矩阵的上三角upper triangular
np.tril(corr_return) #查看矩阵的下三角lower triangular
np.trace(corr_return) #查看矩阵的迹（一个矩阵对角线上各元素的总和）
np.transpose(return_array1) #矩阵转置
return_array1.T #利用数组的属性获得矩阵的转置

# 例2-31 矩阵的内积inner product
return_daily=np.dot(weight_array1,return_array1) #计算投资组合的日收益率 W=[wi]*R=[rij],i=number of stocks(row),j=range of days(column)
return_daily #元素为5（天）

import numpy.linalg as la #导入NumPy的子模块linalg并缩写为la
la.det(corr_return) #计算矩阵的行列式 determinant

la.inv(corr_return) #计算矩阵的逆矩阵 inverse matrix
I=np.dot(la.inv(corr_return),corr_return) #原矩阵与逆矩阵的内积，结果是个单位矩阵
I.round(4) #每个元素保留至小数点后4位

la.eig(corr_return) #矩阵的特征值(eigenvalue)分解，输出结果中：第1个数组代表特征值，第2个数组代表特征向量；注意：仅方阵才可分解特征值

la.svd(corr_return) #矩阵的奇异值分解(singular value decomposition)，输出结果中：第1、3个数组是酉矩阵(unitary matrix)，第2个数组代表奇异值

import numpy.random as npr
# 例2-32 正态分布抽取随机数（10万次）
I=100000 #随机抽样次数
mean1=1.5 #均值
std1=2.5 #标准差
x_norm=npr.normal(loc=mean1,scale=std1,size=I) #从正态分布中随机抽样
print('从正态分布中随机抽样的均值',x_norm.mean())
print('从正态分布中抽样的标准差',x_norm.std())

# 例2-33 3个从标准正态分布中抽取随机数的函数对比：randn、standard_normal、normal(0,1,)
x_snorm1=npr.randn(I)
x_snorm2=npr.standard_normal(size=I)
mean2=0
std2=1
x_snorm3=npr.normal(loc=mean2,scale=std2,size=I)
print('运用 randn 函数从标准正态分布中抽样的均值',x_snorm1.mean())
print('运用 randn 函数从标准正态分布中抽样的标准差',x_snorm1.std())
print('运用 standard_normal 函数从标准正态分布中抽样的均值',x_snorm2.mean())
print('运用 standard_normal 函数从标准正态分布中抽样的标准差',x_snorm2.std())
print('运用 normal 函数从标准正态分布中抽样的均值',x_snorm3.mean())
print('运用 normal 函数从标准正态分布中抽样的标准差',x_snorm3.std())

# 例2-34 基于对数正态分布的随机抽样
mean3=0.4
std3=1.2
x_logn=npr.lognormal(mean=mean3,sigma=std3,size=I)
print('从对数正态分布中抽样的均值',x_logn.mean())
print('从对数正态分布中抽样的标准差',x_logn.std())

# 例2-35 基于卡方分布的随机抽样
freedom1=6
freedom2=98
x_chi1=npr.chisquare(df=freedom1,size=I)
x_chi2=npr.chisquare(df=freedom2,size=I)
print('从自由度是6的卡方分布中抽样的均值',x_chi1.mean())
print('从自由度是6的卡方分布中抽样的标准差',x_chi1.std())
print('从自由度是98的卡方分布中抽样的均值',x_chi2.mean())
print('从自由度是98的卡方分布中抽样的标准差',x_chi2.std())

# 例2-36 基于学生t分布的随机抽样
freedom3=3
freedom4=130
x_t1=npr.standard_t(df=freedom3,size=I)
x_t2=npr.standard_t(df=freedom4,size=I)
print('从自由度是3的学生t分布中抽样的均值',x_t1.mean())
print('从自由度是3的学生t分布中抽样的标准差',x_t1.std())
print('从自由度是130的学生t分布中抽样的均值',x_t2.mean())
print('从自由度是130的学生t分布中抽样的标准差',x_t2.std())

# 例2-37 基于F分布的随机抽样
freedom5=4
freedom6=10
x_f=npr.f(dfnum=freedom5,dfden=freedom6,size=I)
print('从F分布中抽样的均值',x_f.mean())
print('从F分布中抽样的标准差',x_f.std())

# 例2-38 基于贝塔分布的随机抽样
a1=3
b1=7
x_beta=npr.beta(a=a1,b=b1,size=I)
print('从贝塔分布中抽样的均值',x_beta.mean())
print('从贝塔分布中抽样的标准差',x_beta.std())

# 例2-39 基于伽马分布的随机抽样
a2=2
b2=8
x_gamma=npr.gamma(shape=a2,scale=b2,size=I)
print('从伽马分布中抽样的均值',x_gamma.mean())
print('从伽马分布中抽样的标准差',x_gamma.std())

import numpy_financial as npf # 导入numpy_financial模块
npf.__version__

# 例2-40 现金流终值 future value
V0=2e7 #初始投资金额
V1=3e6 #每年固定金额投资
T=5    #投资期限（年）
r=0.08 #年化投资回报率
FV1=npf.fv(rate=r,nper=T,pmt=-V1,pv=-V0,when='end') #计算项目终值且期间追加投资发生在每年年末
print('计算得到项目终值（期间追加投资发生在每年年末',round(FV1,2))
FV2=npf.fv(rate=r,nper=T,pmt=-V1,pv=-V0,when='begin') #计算项目终值且期间追加投资发生在每年年初
print('计算得到项目终值（期间追加投资发生在每年年初',round(FV2,2))
FV_diff=FV2-FV1
print('期间追加投资发生时点不同而导致项目终值的差异',round(FV_diff,2))

# 例2-41 现金流现值 present value
V1=2e6
Vt=2.5e7
T=6
R=0.06
PV1=npf.pv(rate=R,nper=T,pmt=V1,fv=Vt,when=0) #计算项目现值并且期间现金流发生在每年年末
print('计算得到项目现值（期间现金流发生在每年年末）',round(PV1,2))
PV2=npf.pv(rate=R,nper=T,pmt=V1,fv=Vt,when=1) #计算项目现值并且期间现金流发生在每年年初
print('计算得到项目现值（期间现金流发生在每年年初）',round(PV2,2))
PV_diff=PV2-PV1
print('期间现金流发生时点不同而导致项目现值的差异',round(PV_diff,2))

# 例2-42 净现值
R1=0.09
cashflow=np.array([-2.8e7,7e6,8e6,9e6,1e7]) #项目净现金流（数组格式）
NPV1=npf.npv(rate=R1,values=cashflow)
print('计算得到项目净现值',NPV1)
R2=0.06
NPV2=npf.npv(rate=R2,values=cashflow)
print('计算得到项目新的净现值',NPV2)

# 例2-43 内含报酬率
IRR=npf.irr(values=cashflow)
print('计算得到项目的内含报酬率',round(IRR,6))

# 例2-44 住房按揭贷款的等额本息还款
prin_loan=5e6 #住房按揭贷款本金
tenor_loan=5*12 #贷款期限（月）
rate_loan=0.06/12 #贷款月利率
payment=npf.pmt(rate=rate_loan,nper=tenor_loan,pv=prin_loan,fv=0,when='end')
print('计算得到住房按揭贷款每月还款总金额',round(payment,2))
tenor_list=np.arange(tenor_loan)+1 #创建包含每次还款期限的数组
payment_interest=npf.ipmt(rate=rate_loan,per=tenor_list,nper=tenor_loan,pv=prin_loan,fv=0,when='end')
print('计算住房按揭贷款每月偿还的利息金额',payment_interest)
payment_principle=npf.ppmt(rate=rate_loan,per=tenor_list,nper=tenor_loan,pv=prin_loan,fv=0,when='end')
print('计算住房按揭贷款每月偿还的本金金额',payment_principle)
print((payment_interest+payment_principle).round(2)) #验证是否与每月还款总金额保持一致并且小数点后保留2位

