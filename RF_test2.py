#-*- coding : utf-8-*-
# coding:unicode_escape
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets,ensemble
from sklearn.model_selection import cross_val_score
import shap
import pickle

plt.style.use('seaborn-darkgrid')
sns.set(style = 'darkgrid')
import warnings
warnings.filterwarnings("ignore")
############# 数据预处理 #############
# data = pd.read_csv('./data/附件1-区域15分钟负荷数据.csv')
# data['数据时间'].str.split('-',expand=True)
# #data['数据时间']='20'+data['数据时间'].str.split('-',expand=True)[2]+'-'+data['数据时间'].str.split('/',expand=True)[0].str.zfill(2)+'-'+data['数据时间'].str.split('/',expand=True)[1].str.zfill(2)
# #data['数据时间']=data['数据时间'].str.split('-',expand=True)[2]+'-'+data['数据时间'].str.split('-',expand=True)[0].str.zfill(2)+'-'+data['数据时间'].str.split('-',expand=True)[1].str.zfill(2)
# data['year']=data['数据时间'].str.split('-',expand=True)[0]
# data['month']=data['数据时间'].str.split('-',expand=True)[1]
# data['day']=data['数据时间'].str.split('-',expand=True)[2].str.split(' ',expand=True)[0]
# data['hour']=data['数据时间'].str.split('-',expand=True)[2].str.split(' ',expand=True)[1].str.split(':',expand=True)[0]
# data['minute']=data['数据时间'].str.split('-',expand=True)[2].str.split(' ',expand=True)[1].str.split(':',expand=True)[1]

# data['year']=data['year']-2000*np.ones((len(data['year'],1)))
#
AllX = pd.read_csv('data/AllData1_X.csv') ### 加上气候信息、时间信息和新增的特征
X=AllX.iloc[:119336, :].values
print(X.shape)#(1343, 5)numpy.ndarray
AllY = pd.read_csv('data/data1_Y.csv')
Y=AllY.iloc[:119336, :].values
print(Y.shape)#(1343, 1)


### 读取新曾特征
Features = pd.read_csv('data/Allfeature.csv')
features=Features.values[:119336,2:]
# features=Features.values[:,2:]
X=np.concatenate((X,features),axis=1)

#### 打乱数据
data=np.append(X, Y, axis=1)# data：(1343, 6)
## 缺失值处理,无穷数据处理
##Input contains NaN, infinity or a value too large for dtype('float32').
# data=pd.read_csv('./data/predict1_XY.csv')
# data.replace(-np.inf, np.nan)
# data.replace(np.inf, np.nan)
# data.dropna(inplace=True)
# data = data.reset_index(drop=True)
##np.isinf(X).any()=False
#np.isnan(data).any()
# print(data4.isnull().any())


data=np.random.permutation(data)#默认对第一维打乱(这里打乱了)
X=data[:,:-1] #(128155, 6)
Y=data[:,-1]
# X=pd.DataFrame(data['year'],daxta['month'],data['day'],data['minute'])

#### 删除前面的20数据
# X_train=X[20:int(X.shape[0]*0.9),:]
# # X_test=X[int(X.shape[0]*0.9):int(X.shape[0]*0.9),:]
# X_test=X[int(X.shape[0]*0.9):,:]
# Y_train=Y[20:int(Y.shape[0]*0.9)]
# # Y_test=Y[int(Y.shape[0]*0.9):int(Y.shape[0]*0.9)]
# Y_test=Y[int(Y.shape[0]*0.9):]

#### 保留前面的20数据
X_train=X[:int(X.shape[0]*0.9),:]
# X_test=X[int(X.shape[0]*0.8):int(X.shape[0]*0.9),:]
X_test=X[int(X.shape[0]*0.9):,:]
Y_train=Y[:int(Y.shape[0]*0.9)]
# Y_test=Y[int(Y.shape[0]*0.8):int(Y.shape[0]*0.9)]
Y_test=Y[int(Y.shape[0]*0.9):]


## 标准化
# from sklearn.preprocessing import StandardScaler
# X_sc = StandardScaler()
# # X_train = X_sc.fit_transform(X_train)###不应该对时间标准化
# # X_test = X_sc.fit_transform(X_test)
# Y_train = X_sc.fit_transform(Y_train.reshape(-1, 1))##应该对因变量标准化
# Y_test = X_sc.fit_transform(Y_test.reshape(-1, 1))

# ##############模型训练
# n_estimators=100
# rf=RandomForestRegressor(n_estimators=1000)
# model = rf.fit(X_train, Y_train)

regressor = RandomForestRegressor(n_estimators=200,random_state=0)
#ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
regressor.fit(X_train, Y_train)
cross_val_score(regressor, X_train, Y_train, cv=10,scoring = "neg_mean_squared_error")
#cv： 交叉验证折数或可迭代的次数

##### 评估
# 查看所有可以用的评估指标
import sklearn#必须先导入sklearn，否则会报错
# print(sorted(sklearn.metrics.SCORERS.keys()))
Y1 = regressor.predict(X_train)
# val_yhat = regressor.predict(Y_train)
Y2 = regressor.predict(X_test)

##计算三个指标：MAE(Mean Absolute Error) 平均绝对误差 、MSE(Mean Square Error) 平均平方差、MAPE (Mean Absolute Percentage Error, 也叫mean absolute percentage deviation (MAPD)
def R2(y_test,y_pred):
    SStot=np.sum((y_test-np.mean(y_test))**2)
    SSres=np.sum((y_test-y_pred)**2)
    r2=1-SSres/SStot
    return r2


########### train集合  ###########
print("#########  RF  ###########")
print("#########  Train  ###########")

#MSE
MSE = np.linalg.norm(Y1-Y_train, ord=2)**2/len(Y_train)
print("MSE",MSE)#79.26742582898571

#RMSE--相当于y-y_hat的二阶范数/根号n
RMSE = np.linalg.norm(Y1-Y_train, ord=2)/len(Y_train)**0.5
print("RMSE",RMSE) #8.903225585650725


### MAE--相当于y-y_hat的一阶范数/n
MAE = np.linalg.norm(Y1-Y_train, ord=1)/len(Y_train)
print("MAE",MAE) #6.396413824234643

##
MAPE = np.mean(np.abs((Y1-Y_train) / Y_train)) * 100

print("MAPE",MAPE) #8.601955442711105

r2=R2(Y_train,Y1)
print("R2",r2)

#
# ######### val集合
# print("#########  val  ###########")
#
# #MSE
# MSE = np.linalg.norm(val_yhat-val_Y, ord=2)**2/len(val_Y)
# print("MSE",MSE)#79.26742582898571
#
# #RMSE--相当于y-y_hat的二阶范数/根号n
# RMSE = np.linalg.norm(val_yhat-val_Y, ord=2)/len(val_Y)**0.5
# print("RMSE",RMSE) #8.903225585650725
#
# ### MAE--相当于y-y_hat的一阶范数/n
# MAE = np.linalg.norm(val_yhat-val_Y, ord=1)/len(val_Y)
# print("MAE",MAE) #6.396413824234643
#
# ##
# MAPE = np.mean(np.abs((val_yhat-val_Y,) / val_Y)) * 100
#
# print("MAPE",MAPE) #8.601955442711105
#
# ##R2
# r2=R2(val_yhat,val_Y)
# print("R2",r2)

########### test集合  ###########
print("#########  Test  ###########")

#MSE
MSE = np.linalg.norm(Y2-Y_test, ord=2)**2/len(Y_test)
print("MSE",MSE)#79.26742582898571

#RMSE--相当于y-y_hat的二阶范数/根号n
RMSE = np.linalg.norm(Y2-Y_test, ord=2)/len(Y_test)**0.5
print("RMSE",RMSE) #8.903225585650725

### MAE--相当于y-y_hat的一阶范数/n
MAE = np.linalg.norm(Y2-Y_test, ord=1)/len(Y_test)
print("MAE",MAE) #6.396413824234643

##
MAPE = np.mean(np.abs((Y2-Y_test) / Y_test)) * 100

print("MAPE",MAPE) #8.601955442711105

r2=R2(Y_test,Y2)
print("R2",r2)



# ##可解释机器学习:LIMe和SHAP
# ###关于模型解释性，除了线性模型和决策树这种天生就有很好解释性的模型意外，sklean中有很多模型都有importance这一接口，可以查看特征的重要性。
#
# #在SHAP中进行模型解释需要先创建一个explainer
#
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(X_train)  # 传入特征矩阵X，计算SHAP值 (90, 9)

#Local可解释性提供了预测的细节
#每一行代表一个特征，横坐标为SHAP值
# shap.summary_plot(shap_values, train_X)

#Feature Importance:
feature_name=pd.read_csv('./data/featureName.csv').values #,encoding='unicode_escape'
feature_name=feature_name.flatten() #numpy.ndarray
print(feature_name)
# shap.summary_plot(shap_values, X_train,feature_names=feature_name, plot_type="bar",show=False,save=True,path='./res/RF.png')
shap.summary_plot(shap_values, X_train,feature_names=feature_name, plot_type="bar",show=False)
plt.savefig('./res/RF.png')

# 保存模型
# pickle.dump(regressor, open("./res/RF_tianqi.dat", "wb"))
s = pickle.dumps(regressor)
with open("./res/RF_tianqi.model",'wb+') as f:#注意此处mode是'wb+'，表示二进制写入
    f.write(s)
#IndexError: index 1 is out of bounds for axis 0 with size 1

######################## 结果########################

####给出该地区电网未来 3 个月日负荷的最大值和最小值预测结果，以及相应达到负荷最大值和最小值的时间，并分析其预测精度。
#2021-5-31  23:45:00--->2021-8-31

print('############  3 month prediction ############')
pre_X=AllX.iloc[119336:, :].values
pre_Y=AllY.iloc[119336:, :].values
features=Features.values[119336:,2:]
# pre_X=AllX.iloc[-200:, :].values
# pre_Y=AllY.iloc[-200:, :].values
# features=Features.values[-200:,2:]
pre_X=np.concatenate((pre_X,features),axis=1)
Y3 = regressor.predict(pre_X)
Y3 = np.reshape(Y3,(-1,1))

MSE = np.linalg.norm(Y3-pre_Y, ord=2)**2/len(pre_Y)
print("MSE",MSE)#79.26742582898571

#RMSE--相当于y-y_hat的二阶范数/根号n
RMSE = np.linalg.norm(Y3-pre_Y, ord=2)/len(pre_Y)**0.5
print("RMSE",RMSE) #8.903225585650725

### MAE--相当于y-y_hat的一阶范数/n
MAE = np.linalg.norm(Y3-pre_Y, ord=1)/len(pre_Y)
print("MAE",MAE) #6.396413824234643

##
MAPE = np.mean(np.abs((Y3-pre_Y) / pre_Y)) * 100

print("MAPE",MAPE) #8.601955442711105

r2=R2(pre_Y,Y3)
print("R2",r2)
Y3=pd.DataFrame(Y3)
Y3.to_csv('res/RF_tianqi_3_month.csv')


#### 给出该地区电网未来 10 天间隔 15 分钟的负荷预测结果，并分析其预测精度；
# 2021-8-21 23 45 23:45:00--->2021-8-31  23 45

print('############ 10 day prediction ############')
# 训练
f = open('./res/RF_tianqi.model','rb') #注意此处model是rb
s = f.read()
model = pickle.loads(s)
X_train=AllX.iloc[119336:127196, :].values
Y_train=AllY.iloc[119336:127196, :].values
features=Features.values[119336:127196:,2:]
# X_train=AllX.iloc[-200:-100, :].values
# Y_train=AllY.iloc[-200:-100, :].values
# features=Features.values[-200:-100:,2:]
X_train=np.concatenate((X_train,features),axis=1)
model.fit(X_train,Y_train)

# 预测
pre_X=AllX.iloc[127196:, :].values
pre_Y=AllY.iloc[127196:, :].values
features=Features.values[127196:,2:]
# pre_X=AllX.iloc[-100:, :].values
# pre_Y=AllY.iloc[-100:, :].values
# features=Features.values[-100:,2:]
pre_X=np.concatenate((pre_X,features),axis=1)
Y3 = model.predict(pre_X)
Y3 = np.reshape(Y3,(-1,1))

MSE = np.linalg.norm(Y3-pre_Y, ord=2)**2/len(pre_Y)
print("MSE",MSE)#79.26742582898571

#RMSE--相当于y-y_hat的二阶范数/根号n
RMSE = np.linalg.norm(Y3-pre_Y, ord=2)/len(pre_Y)**0.5
print("RMSE",RMSE) #8.903225585650725

### MAE--相当于y-y_hat的一阶范数/n
MAE = np.linalg.norm(Y3-pre_Y, ord=1)/len(pre_Y)
print("MAE",MAE) #6.396413824234643

##
MAPE = np.mean(np.abs((Y3-pre_Y) / pre_Y)) * 100

print("MAPE",MAPE) #8.601955442711105

r2=R2(pre_Y,Y3)
print("R2",r2)

Y3=pd.DataFrame(Y3)
Y3.to_csv('res/RF_tianqi_10_day.csv')
