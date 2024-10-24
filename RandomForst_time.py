from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets,ensemble
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from numpy import array

plt.style.use('seaborn-darkgrid')
sns.set(style = 'darkgrid')
import warnings
warnings.filterwarnings("ignore")

########### 参数
n_timestamp  = 10    # 时间戳
n_epochs     = 50   # 训练轮数
#############数据预处理(时间窗口)


data = pd.read_csv('data/data1_Y.csv')  #读入电荷数据

# data
data=data.values
training_set = data[:int(data.shape[0]*0.9)]
test_set = data[int(data.shape[0]*0.9):]
data_fng = data

######## 画图
# plt.plot(training_set, c='red')
# # X轴暂不设置标签
# plt.title('training data of FNG value', fontsize=16)
# plt.xlabel('Time', fontsize=16)
# # 设置标签
# plt.ylabel("value", fontsize=16)
#
# plt.tick_params(axis='both', which='major', labelsize=16)
#
# plt.show()

# data_fng= data['fng_value']
# plt.plot(data_fng[0:380], color='b', label='训练数据')
# plt.plot(data_fng[380:], color='r', label='测试数据')
# plt.axvline(380, 0, 10000)  # 分隔线
# plt.legend()
# plt.show()


#将数据归一化，范围是0到1
sc  = MinMaxScaler(feature_range=(0, 1))#归一化0-1
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled  = sc.transform(test_set)


# 取前 n_timestamp 天的数据为 X；n_timestamp+1天数据为 Y。
def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]#sequencex是size=10的window窗口，sequencey是第11th的数组，例如seq_x=[5:15], seq_x=[15]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

#  (370, 10, 1)--->(370, 1)
X_train, Y_train = data_split(training_set_scaled, n_timestamp)#以窗口大小进行数据分割(数据已经经过归一化处理)  training 是百分之70的数据
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)#shape[0]读取数据行数。shape[1]读取数据列数 例子 reshape（3，4）重整为3行4列，若是-1则代表为未知
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])

X_test, Y_test = data_split(testing_set_scaled, n_timestamp) #testing 是百分之30的数据
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])#reshape的作用是什么？


# ##############模型训练
# n_estimators=100
# rf=RandomForestRegressor(n_estimators=1000)
# model = rf.fit(X_train, Y_train)

regressor = RandomForestRegressor(n_estimators=200,random_state=0)
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
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(train_X)  # 传入特征矩阵X，计算SHAP值
#
# #Local可解释性提供了预测的细节
# #每一行代表一个特征，横坐标为SHAP值
# # shap.summary_plot(shap_values, train_X)
#
# #Feature Importance:
# shap.summary_plot(shap_values, train_X,feature_names=feature_name, plot_type="bar")

