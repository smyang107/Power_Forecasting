import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets,ensemble
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import BayesianRidge
import xgboost as xgb

# plt.style.use('seaborn-darkgrid')
# sns.set(style = 'darkgrid')
# import warnings
# warnings.filterwarnings("ignore")
# #############数据预处理
# data = pd.read_csv('./data/附件1-区域15分钟负荷数据.csv')
# data['数据时间'].str.split('-',expand=True)
# #data['数据时间']='20'+data['数据时间'].str.split('-',expand=True)[2]+'-'+data['数据时间'].str.split('/',expand=True)[0].str.zfill(2)+'-'+data['数据时间'].str.split('/',expand=True)[1].str.zfill(2)
# #data['数据时间']=data['数据时间'].str.split('-',expand=True)[2]+'-'+data['数据时间'].str.split('-',expand=True)[0].str.zfill(2)+'-'+data['数据时间'].str.split('-',expand=True)[1].str.zfill(2)
# data['year']=data['数据时间'].str.split('-',expand=True)[0]
# data['month']=data['数据时间'].str.split('-',expand=True)[1]
# data['day']=data['数据时间'].str.split('-',expand=True)[2].str.split(' ',expand=True)[0]
# data['hour']=data['数据时间'].str.split('-',expand=True)[2].str.split(' ',expand=True)[1].str.split(':',expand=True)[0]
# data['minute']=data['数据时间'].str.split('-',expand=True)[2].str.split(' ',expand=True)[1].str.split(':',expand=True)[1]
# print(data['month'])
# print(data['day'])
# print(data['year'])
# print(data['minute'])
# print(type(data))#<class 'pandas.core.frame.DataFrame'>
# print(type(data['minute']))#<class 'pandas.core.series.Series'>
# data.to_csv('data/data1_X.csv')

########################### 均值 ###########################
#（1）MA5：过去5个时间点的均值
# （2）MA10：过去10个时间点的均值
# （3）MA20：过去20个时间点的均值
Y = pd.read_csv('./data/data1_Y.csv')
Y=Y.values
print(Y)
# print(type(Y[0]))#<class 'pandas.core.frame.DataFrame'>
MA5=np.array([])
MA10=np.array([])
MA20=np.array([])
for i in range(5,len(Y)):
    MA5=np.append(MA5,np.sum(Y[i-5:i])/5)
    # MA5=np.c_(MA5,np.sum(Y[i-5:i])/5)

for i in range(10,len(Y)):
    # MA10.append(np.sum(Y[i-10:i])/10)
    MA10 = np.append(MA10, np.sum(Y[i - 10:i]) / 10)

for i in range(20,len(Y)):
    MA20 = np.append(MA20, np.sum(Y[i - 20:i]) / 20)

##预处理归一化
# MA5=np.array(MA5).T
# MA10=np.array(MA10).T
# MA20=np.array(MA20).T
MA5 = MA5.reshape((-1,1))
MA10 = MA10.reshape((-1,1))
MA20 = MA20.reshape((-1,1))
print(MA5.shape) #(128150,)
print(MA10.shape) #(128145,)
print(MA20.shape) #(128135,)
# from sklearn.preprocessing import StandardScaler,MinMaxScaler
# X_sc = StandardScaler()
# MA5=X_sc.fit_transform(MA5.reshape(-1,1))
# MA10=X_sc.fit_transform(MA10.reshape(-1,1))
# MA20=X_sc.fit_transform(MA20.reshape(-1,1))

#（4）P_change5：过去5个时间点的负荷量的变化趋势：做最小二乘拟合，把序列拟合成一条直线，斜率
# （5）P_change10：过去10个时间点的负荷量的变化趋势：做最小二乘拟合，把序列拟合成一条直线，斜率
# （6）P_change20：过去20个时间点的负荷量的变化趋势：做最小二乘拟合，把序列拟合成一条直线，斜率

def fit(datax, datay):
    m=len(datay)
    x_bar=np.mean(datax)
    sum_yx=0
    sum_x2=0
    sum_delta=0
    for i in range(m):
        x=datax[i]
        y=datay
        sum_yx+=y*(x-x_bar)
        sum_x2+=x**2#根据公式计算W
    w=sum_yx/(sum_x2-m*(x_bar**2))
    for i in range(m):
        x=datax[i]
        y=datay[i]
        sum_delta+=(y-w*x)
    # b=sum_delta /m
    return w

def fit2(X,Y):
    n = len(X)
    xy, x, y, x2, = 0, 0, 0, 0

    for i in range(n):
        xy += X[i] * Y[i]  # xy的和
        x += X[i]  # x的和
        y += Y[i]  # y的和
        x2 += X[i] ** 2  # x的平方的和

    a1 = (n * xy - x * y) / (n * x2 - x ** 2)  # 系数K
    a0 = (x2 * y - x * xy) / (n * x2 - x ** 2)  # 常数b
    return a1
# P_change5=[]
# P_change10=[]
# P_change20=[]
# for i in range(5,len(Y)):
#     P_change5.append(fit(range(0,5),Y[i-5:i]))
#
# for i in range(10,len(Y)):
#     P_change10.append(fit(range(0,10),Y[i-10:i]))
#
# for i in range(20,len(Y)):
#     P_change20.append(fit(range(0,20),Y[i-20:i]))

########################### 变化趋势 ###########################
P_change5=np.array([])
P_change10=np.array([])
P_change20=np.array([])
for i in range(5,len(Y)):
    # P_change5=np.append(P_change5,fit(range(0,5),Y[i-5:i]-200000*np.ones(5)))
    P_change5 = np.append(P_change5, fit2(range(0, 5), Y[i-5:i]))
    # MA5=np.c_(MA5,np.sum(Y[i-5:i])/5)

for i in range(10,len(Y)):
    # MA10.append(np.sum(Y[i-10:i])/10)
    P_change10 = np.append(P_change10,fit2(range(0,10),Y[i-10:i]))

for i in range(20,len(Y)):
    P_change20 = np.append(P_change20,fit2(range(0,20),Y[i-20:i]))

print(P_change5.shape) #(1338,)
print(P_change5)

P_change5=P_change5.reshape(-1,1)
P_change10=P_change10.reshape(-1,1)
P_change20=P_change20.reshape(-1,1)

# X_sc = StandardScaler()
# P_change5=X_sc.fit_transform(P_change5.reshape(-1,1))
# P_change10=X_sc.fit_transform(P_change10.reshape(-1,1))
# P_change20=X_sc.fit_transform(P_change20.reshape(-1,1))

print(P_change5.shape) #(1338,)

############################ 综合变量 ############################
MA5=np.concatenate((np.zeros(shape=(5,1)),MA5),axis=0)
MA10=np.concatenate((np.zeros(shape=(10,1)),MA10),axis=0)
MA20=np.concatenate((np.zeros(shape=(20,1)),MA20),axis=0)
P_change5=np.concatenate((np.zeros(shape=(5,1)),P_change5),axis=0)
P_change10=np.concatenate((np.zeros(shape=(10,1)),P_change10),axis=0)
P_change20=np.concatenate((np.zeros(shape=(20,1)),P_change20),axis=0)
all_feature=np.concatenate((MA5,MA10,MA20,P_change5,P_change10,P_change20),axis=1)

# all_feature.tofile("Allfeature.csv")
all_feature=pd.DataFrame(all_feature)
all_feature.to_csv("data\\Allfeature.csv")
