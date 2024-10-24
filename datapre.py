##参考：https://zhuanlan.zhihu.com/p/474637174
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib  inline
plt.style.use('seaborn-darkgrid')
sns.set(style = 'darkgrid')
import warnings
warnings.filterwarnings("ignore")
# import lightgbm as lgb
# from sklearn.preprocessing import scale
# import lightgbm as lgb
# import xgboost as xgb
# from catboost import CatBoostRegressor
# import time
# from tqdm import tqdm
# from sklearn.preprocessing import LabelEncoder
y = pd.read_csv('./data/附件1-区域15分钟负荷数据.csv')
indu = pd.read_csv('./data/附件2-行业日负荷数据.csv')
tianqi = pd.read_csv('./data/附件3-气象数据.csv')
############ 时间特征处理
tianqi['年']=tianqi['日期'].str.split('年',expand=True)[0]
tianqi['月']=tianqi['日期'].str.split('年',expand=True)[1].str.split('月',expand=True)[0]
tianqi['日']=tianqi['日期'].str.split('年',expand=True)[1].str.split('月',expand=True)[1].str.split('日',expand=True)[0]

########温度特征处理
tianqi['最高温度'] = tianqi['最高温度'].map(lambda d: d.replace('℃','')).astype(int)
tianqi['最低温度'] = tianqi['最低温度'].map(lambda d: d.replace('℃','')).astype(int)

print(type(tianqi))
#########天气状况特征处理

series = tianqi.join(tianqi['天气状况'].str.split('/',expand=True))
tianqi['天气1'] = series[0]
tianqi['天气2'] = series[1]
tianqi.info()

#####风向特征处理

tianqi['白天风力风向'].unique()

tianqi['夜晚风力风向'].unique()

dic = {'无持续风向<3级':0,
        '北风1-2级':1,
        '北风3～4级':2,
        '北风4～5级':3,
        '北风3':4,
        '西南风1-2级':5,
       '西南风3-4级':6,
       '东南风1-2级':7,
       '东南风3-4级':8,
        '东南风3～4级':9,
       '东南风4-5级':10,
       '无持续风向1-2级':11,
        '微风<3级':12,
       '无持续风向微风':13,
       '南风1-2级':14,
       '南风3-4级':15,
       '南风3～4级':16,
       '南风4～5级':17,
       '东北偏东风2':18,
       '东北风1-2级':19,
       '东北风3-4级':20,
       '东北风3～4级':21,
       '北风3-4级':22,
       '东风3-4级':23,
       '东风3～4级':24,
       '东风8-9级':25,
       '东风1-2级':26,
       '北风4-5级':27}
tianqi['白天风力风向'] = tianqi['白天风力风向'].map(dic)
tianqi['夜晚风力风向'] = tianqi['夜晚风力风向'].map(dic)

####天气进行有序编码
tianqi['天气1'].unique()

dic1 = {'晴':1,
        '多云':2,
        '阴':3,
        '小雨':4,
        '小雨-中雨':5,
        '中雨':6,
        '中雨-大雨':7,
        '大雨':8,
        '北风':9,
        '阵雨':10,
        '雾':11,
        '雷阵雨':12,
        '暴雨':13,
        '局部多云':14,
        '小到中雨':15,
        '中到大雨':16,
        '大到暴雨':17,
        '晴间多云':18}
tianqi['天气1'] = tianqi['天气1'].map(dic1)
tianqi['天气2'] = tianqi['天气2'].map(dic1)
del tianqi['天气状况']
print(tianqi)
tianqi.to_csv('./data/tianqi1.csv')



###### 附录1的数据与天气的数据对齐
data1 = pd.read_csv('data/data1_X.csv')
#最高温度,最低温度,白天风力风向,夜晚风力风向,天气1,天气2
# 建立一个映射
dict1=dict(zip(tianqi['日期'], tianqi['最高温度']))
dict2=dict(zip(tianqi['日期'], tianqi['最低温度']))
dict3=dict(zip(tianqi['日期'], tianqi['白天风力风向']))
dict4=dict(zip(tianqi['日期'], tianqi['夜晚风力风向']))
dict5=dict(zip(tianqi['日期'], tianqi['天气1']))
dict6=dict(zip(tianqi['日期'], tianqi['天气2']))
data1['最高温度']=pd.Series()
data1['最低温度']=pd.Series()
data1['白天风力风向']=pd.Series()
data1['夜晚风力风向']=pd.Series()
data1['天气1']=pd.Series()
data1['天气2']=pd.Series()

res1=[]
res2=[]
res3=[]
res4=[]
res5=[]
res6=[]
for i in range(data1.values.shape[0]):
    it=''+str(data1['year'][i])+'年'+str(data1['month'][i])+'月'+str(data1['day'][i])+'日'
    # data1['最高温度'].append(dict1[it])
    # data1['最低温度'].append(dict2[it])
    # data1['白天风力风向'].append(dict3[it])
    # data1['夜晚风力风向'].append(dict4[it])
    # data1['天气1'].append(dict5[it])
    # data1['天气2'].append(dict6[it])
    res1.append(dict1[it])
    res2.append(dict2[it])
    res3.append(dict3[it])
    res4.append(dict4[it])
    res5.append(dict5[it])
    res6.append(dict6[it])

data1['最高温度']=pd.Series(res1)
data1['最低温度']=pd.Series(res2)
data1['白天风力风向']=pd.Series(res3)
data1['夜晚风力风向']=pd.Series(res4)
data1['天气1']=pd.Series(res5)
data1['天气2']=pd.Series(res6)
data1.to_csv('data/predict1_X.csv')