import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional,GRU
# from keras.layers import Dense, LSTM, Bidirectional
import csv
# import keras
from tensorflow import keras
# from keras.models import load_model

from tensorflow.keras.models import load_model

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 确保结果尽可能重现,在每次执行代码时，使每次切分后的训练集、验证集输入结果相同，便于验证学习参数的有效性和查找问题。
from numpy.random import seed
seed(1)
# tf.random.set_seed(1)

# 设置相关参数
n_timestamp  = 20    # 时间戳（修改了时间戳：20---》50）
n_epochs     = 20  # 训练轮数
#============= =======================
# 选择模型：
#   1: 单层 LSTM
#  2: 多层 LSTM
# 3: 双向 LSTM
# ====================================
model_type = 4#改动了


AllX = pd.read_csv('附件1-区域15分钟负荷数据.csv')

data=AllX.values[:127196, :] #(128156, 2)4
#data=AllX.values[:1000, :] #(128156, 2)
training_set = data[:int(data.shape[0]*0.9), 1:2]
test_set = data[int(data.shape[0]*0.9):, 1:2]
data_fng = data[:, 1:2]

# plt.plot(training_set, c='red')
# # X轴暂不设置标签
# plt.title('training data of FNG value', fontsize=16)
# # plt.xlabel('Time', fontsize=16)
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
training_set_scaled = sc.fit_transform(training_set)#(380, 1)
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


X_train, y_train = data_split(training_set_scaled, n_timestamp)#以窗口大小进行数据分割(数据已经经过归一化处理)  training 是百分之70的数据
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)#shape[0]读取数据行数。shape[1]读取数据列数 例子 reshape（3，4）重整为3行4列，若是-1则代表为未知

X_test, y_test = data_split(testing_set_scaled, n_timestamp) #testing 是百分之30的数据
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)#reshape的作用是什么？
#X_train:(107382, 20, 1),X_test.shape:(11914, 20, 1)
#打乱数据
data_set_scaled = sc.fit_transform(data_fng)
X, y = data_split(data_set_scaled, n_timestamp)
y=np.reshape(y,(-1,1,1))
data_After=np.concatenate((X,y),axis=1)
data_After=np.random.permutation(data_After)
X_train=data_After[:int(data_After.shape[0]*0.9),:-1,:]
y_train=data_After[:int(data_After.shape[0]*0.9),-1,:]
y_train=np.reshape(y_train,(-1,1))
X_test=data_After[int(data_After.shape[0]*0.9):,:-1,:]
y_test=data_After[int(data_After.shape[0]*0.9):,-1,:]
y_test=np.reshape(y_test,(-1,1))
# 建构 LSTM模型
if model_type == 1:
    # 单层 LSTM
    model = Sequential()
    model.add(LSTM(units=50, activation='tanh', #神经单元50个，激活函数为relu函数，输入X_train的列数量，也就是说！把X_train 70%的数据拿去训练
                   input_shape=(X_train.shape[1], 1)))
    model.add(Dense(units=1))

if model_type == 2:#改动了
    # 多层 LSTM
    model = Sequential()
    model.add(LSTM(units=50, activation='tanh', return_sequences=True,
                   input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dense(10))
    model.add(Dense(1))

if model_type == 3:
    # 双向 LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='tanh'),
                            input_shape=(X_train.shape[1], 1)))
    model.add(Bidirectional(LSTM(50, activation='tanh'),
                            input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))

##LSTM和CRU两者通过门控机制来保留之前序列的有用的信息，保证了在long-term传播的时候也不会丢失。与此同时，GRU相对于LSTM少了一个门函数，
# 因此在参数的数量上要少于LSTM，所以整体上GRU的训练速度要快于LSTM的。
# 参考：https://zhuanlan.zhihu.com/p/417297309
if model_type == 4:
    model = Sequential()
    model.add(GRU(units=50, activation='tanh', return_sequences=True,input_shape=(X_train.shape[1], 1)))
    model.add(GRU(units=50, activation='tanh'))
    model.add(Dense(10))
    model.add(Dense(1))

model.summary()  # 输出模型结构


#动态学习率
import tensorflow.keras.backend as K
#import keras.backend as K
#from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import LearningRateScheduler
def scheduler(epoch):
    # 每隔10个epoch，学习率减小为原来的1/10
    if epoch % 10 == 0 and epoch != 0:#epoch为10的整数而且不为零
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)



# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
model.compile(optimizer=keras.optimizers.Adam(0.001),#改动了0.0001
              loss='mean_squared_error')  # 损失函数用均方误差


# history = model.fit(X_train, y_train,
#                     batch_size=10, #改动了
#                     epochs=n_epochs,
#                     callbacks=[reduce_lr],
#                     validation_data=(X_test, y_test),
#                     validation_freq=1)                  #测试的epoch间隔数
history = model.fit(X_train, y_train,
                    batch_size=10, #改动了
                    epochs=n_epochs,
                    callbacks=[reduce_lr],
                    validation_data=(X_test, y_test))
# model.summary()


import csv
from matplotlib import pyplot as plt

filename = 'btc_sentiment.csv'
# 打开文件并将结果文件对象存储在f中
with open(filename) as f:
    # 创建一个与该文件相关联的reader对象
    reader = csv.reader(f)
    # 只调用一次next()方法，得到文件的第一行，将第一行数据中的每一个元素存储在列表中
    header_row = next(reader)

    # 从文件中获取第二列的值（该列表示最高气温）
    highs = []
    # 遍历文件中余下的各行
    # reader对象从其当前所在的位置继续读取CSV文件，每次都自动返回当前所处位置的下一行
    for row in reader:
        # 转换为数字，便于后面让matplotlib能够读取它们
        high = float(row[1])
        highs.append(high)

# 根据数据绘制图形
fig = plt.figure(dpi=128, figsize=(10, 6))
# 将数据集传给绘图对象，并将数据点绘制为红色（表示最高气温）
plt.plot(highs, c='red')

# 设置图形的格式
# 这是字体大小和标签
plt.title("FNG value", fontsize=24)
# X轴暂不设置标签
plt.xlabel('Time', fontsize=16)
# 设置标签
plt.ylabel("dollar", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)

#plt.show()



#visualization
plt.plot(history.history['loss'] , label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('D:\数学建模\\2022泰迪杯\code\LSTM\LSTM\LSTM\pythonProject11\\res\lstm.png')

##计算三个指标：MAE(Mean Absolute Error) 平均绝对误差 、MSE(Mean Square Error) 平均平方差、MAPE (Mean Absolute Percentage Error, 也叫mean absolute percentage deviation (MAPD)
def R2(y_test,y_pred):
    SStot=np.sum((y_test-np.mean(y_test))**2)
    SSres=np.sum((y_test-y_pred)**2)
    r2=1-SSres/SStot
    return r2

predicted_fng = model.predict(X_train)                        # 测试集输入模型进行预测，经过reshape的30%的数据进行预测
predicted_fng = sc.inverse_transform(predicted_fng)  # 对预测数据还原---从（0，1）反归一化到原始范围
real_fng      = sc.inverse_transform(y_train)# 对真实数据还原---从（0，1）反归一化到原始范围
Y_train=real_fng
Y1=predicted_fng

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

#predition
predicted_fng = model.predict(X_test)                        # 测试集输入模型进行预测，经过reshape的30%的数据进行预测
predicted_fng = sc.inverse_transform(predicted_fng)  # 对预测数据还原---从（0，1）反归一化到原始范围
real_fng      = sc.inverse_transform(y_test)# 对真实数据还原---从（0，1）反归一化到原始范围
Y_test=real_fng
Y2=predicted_fng
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

######################## 结果########################
#保存模型
s = model.save('D:\数学建模\\2022泰迪杯\code\LSTM\LSTM\LSTM\pythonProject11\\res\lstm.h5')
#保存预训练结果
cp_net = tf.train.Checkpoint(backbone_layer=model.layers[:-2], dense_layer=model.layers[-2])
cp_net.save('D:\数学建模\\2022泰迪杯\code\LSTM\LSTM\LSTM\pythonProject11\\res\lstm_Checkpoint.h5')
### 加载https://blog.csdn.net/qq_30899353/article/details/119941462
# to_restore_net = tf.train.Checkpoint(backbone_layer=model.layers[:-1])
# to_restore_net.restore(filepath)

# print('############  3 month prediction ############')
# pre_X=AllX.iloc[119336:, 1:2].values
# sc  = MinMaxScaler(feature_range=(0, 1))#归一化0-1
# pre_X_scaled = sc.fit_transform(pre_X)#(380, 1)
# pre_X,pre_Y=data_split(pre_X_scaled, n_timestamp)#
# Y3 = model.predict(pre_X) #Y3:(8800, 1)
# pre_Y=sc.inverse_transform(pre_Y)
# Y3=sc.inverse_transform(Y3)
# Y3 = np.reshape(Y3,(-1,1))
#
#
# MSE = np.linalg.norm(Y3-pre_Y, ord=2)**2/len(pre_Y)
# print("MSE",MSE)#79.26742582898571
#
# #RMSE--相当于y-y_hat的二阶范数/根号n
# RMSE = np.linalg.norm(Y3-pre_Y, ord=2)/len(pre_Y)**0.5
# print("RMSE",RMSE) #8.903225585650725
#
# ### MAE--相当于y-y_hat的一阶范数/n
# MAE = np.linalg.norm(Y3-pre_Y, ord=1)/len(pre_Y)
# print("MAE",MAE) #6.396413824234643
#
# ##
# MAPE = np.mean(np.abs((Y3-pre_Y) / pre_Y)) * 100
#
# print("MAPE",MAPE) #8.601955442711105
#
# r2=R2(pre_Y,Y3)
# print("R2",r2)
# Y3=sc.inverse_transform(Y3)
# Y3=pd.DataFrame(Y3)
# Y3.to_csv('D:\数学建模\\2022泰迪杯\code\LSTM\LSTM\LSTM\pythonProject11\\res\\threemonthresult1.csv')


print('############ 10 day prediction ############')
# 训练
# f = open('D:\数学建模\\2022泰迪杯\code\LSTM\LSTM\LSTM\pythonProject11\\res\lstm1.h5',encoding='gb18030',errors='ignore') #注意此处model是rb
# s = f.read()
# model = load_model('D:\数学建模\\2022泰迪杯\code\LSTM\LSTM\LSTM\pythonProject11\\res\lstm1.h5')
# X_train=AllX.iloc[119336:127196, 1:2].values
# sc  = MinMaxScaler(feature_range=(0, 1))#归一化0-1
# X_train_scaled = sc.fit_transform(X_train)#(380, 1)
# X_train,Y_train=data_split(X_train_scaled, n_timestamp)#
# model.fit(X_train,Y_train,
#                     batch_size=10, #改动了
#                     epochs=n_epochs,#改动了(预训练模型)
#                     callbacks=[reduce_lr])

pre_X=AllX.iloc[127196:, 1:2].values
sc  = MinMaxScaler(feature_range=(0, 1))#归一化0-1
pre_X_scaled = sc.fit_transform(pre_X)#(380, 1)
pre_X,pre_Y=data_split(pre_X_scaled, n_timestamp)#
Y3 = model.predict(pre_X)
pre_Y=sc.inverse_transform(pre_Y)
Y3=sc.inverse_transform(Y3)
Y3 = np.reshape(Y3,(-1,1))

# 预测

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
Y3=sc.inverse_transform(Y3)
Y3=pd.DataFrame(Y3)
Y3.to_csv('D:\数学建模\\2022泰迪杯\code\LSTM\LSTM\LSTM\pythonProject11\\res\\tendaysresult_lstm.csv')