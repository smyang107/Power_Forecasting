import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
import csv
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



# 确保结果尽可能重现,在每次执行代码时，使每次切分后的训练集、验证集输入结果相同，便于验证学习参数的有效性和查找问题。
from numpy.random import seed
seed(1)
# tf.random.set_seed(1)

# 设置相关参数
n_timestamp  = 10    # 时间戳
n_epochs     = 50   # 训练轮数
#============= =======================
# 选择模型：
#   1: 单层 LSTM
#  2: 多层 LSTM
# 3: 双向 LSTM
# ====================================
model_type = 1#改动了

# data = pd.read_csv('./btc_historic.csv')  # 读取股票文件
#
# data
#
# training_set = data.iloc[0:402, 4:5].values
# test_set = data.iloc[402:574, 4:5].values
# #training 数据集
#
# plt.plot(training_set, c='red')
# # X轴暂不设置标签
# plt.title('training data of close', fontsize=16)
# plt.xlabel('Time', fontsize=16)
# # 设置标签
# plt.ylabel("dollar", fontsize=16)
#
# plt.tick_params(axis='both', which='major', labelsize=16)
#
# plt.show()
#
#
#
#
# #将数据归一化，范围是0到1
# sc  = MinMaxScaler(feature_range=(0, 1))#归一化0-1
# training_set_scaled = sc.fit_transform(training_set)
# testing_set_scaled  = sc.transform(test_set)
#
#
# # 取前 n_timestamp 天的数据为 X；n_timestamp+1天数据为 Y。
# def data_split(sequence, n_timestamp):
#     X = []
#     y = []
#     for i in range(len(sequence)):
#         end_ix = i + n_timestamp
#
#         if end_ix > len(sequence) - 1:
#             break
#
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]#sequencex是size=10的window窗口，sequencey是第11th的数组，例如seq_x=[5:15], seq_x=[15]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)
#
#
# X_train, y_train = data_split(training_set_scaled, n_timestamp)#以窗口大小进行数据分割(数据已经经过归一化处理)  training 是百分之70的数据
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)#shape[0]读取数据行数。shape[1]读取数据列数 例子 reshape（3，4）重整为3行4列，若是-1则代表为未知
#
# X_test, y_test = data_split(testing_set_scaled, n_timestamp) #testing 是百分之30的数据
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)#reshape的作用是什么？
#
#
#
#
# # 建构 LSTM模型
# if model_type == 1:
#     # 单层 LSTM
#     model = Sequential()
#     model.add(LSTM(units=50, activation='tanh', #神经单元50个，激活函数为relu函数，输入X_train的列数量，也就是说！把X_train 70%的数据拿去训练
#                    input_shape=(X_train.shape[1], 1)))
#     model.add(Dense(units=1))
#
# if model_type == 2:#改动了
#     # 多层 LSTM
#     model = Sequential()
#     model.add(LSTM(units=50, activation='tanh', return_sequences=True,
#                    input_shape=(X_train.shape[1], 1)))
#     model.add(LSTM(units=50, activation='tanh'))
#     model.add(Dense(1))
#
# if model_type == 3:
#     # 双向 LSTM
#     model = Sequential()
#     model.add(Bidirectional(LSTM(50, activation='tanh'),
#                             input_shape=(X_train.shape[1], 1)))
#     model.add(Dense(1))
#
# model.summary()  # 输出模型结构
#
#
#
# #动态学习率
# import keras.backend as K
# from keras.callbacks import LearningRateScheduler
# def scheduler(epoch):
#     # 每隔10个epoch，学习率减小为原来的1/10
#     if epoch % 10 == 0 and epoch != 0:#epoch为10的整数而且不为零
#         lr = K.get_value(model.optimizer.lr)
#         K.set_value(model.optimizer.lr, lr * 0.1)
#         print("lr changed to {}".format(lr * 0.1))
#     return K.get_value(model.optimizer.lr)
# reduce_lr = LearningRateScheduler(scheduler)
#

#
#
#
#
# # 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
# model.compile(optimizer=tf.keras.optimizers.Adam(0.001),#改动了0.0001
#               loss='mean_squared_error')  # 损失函数用均方误差
#
#
# history = model.fit(X_train, y_train,
#                     batch_size=10, #改动了
#                     epochs=n_epochs,
#                     callbacks=[reduce_lr],
#                     validation_data=(X_test, y_test),
#                     validation_freq=1)                  #测试的epoch间隔数
#
# model.summary()
#
#
#
# import csv
# from matplotlib import pyplot as plt
#
# filename = 'btc_historic.csv'
# # 打开文件并将结果文件对象存储在f中
# with open(filename) as f:
#     # 创建一个与该文件相关联的reader对象
#     reader = csv.reader(f)
#     # 只调用一次next()方法，得到文件的第一行，将第一行数据中的每一个元素存储在列表中
#     header_row = next(reader)
#
#     # 从文件中获取第二列的值（该列表示最高气温）
#     highs = []
#     # 遍历文件中余下的各行
#     # reader对象从其当前所在的位置继续读取CSV文件，每次都自动返回当前所处位置的下一行
#     for row in reader:
#         # 转换为数字，便于后面让matplotlib能够读取它们
#         high = float(row[5])
#         highs.append(high)
#
# # 根据数据绘制图形
# fig = plt.figure(dpi=128, figsize=(10, 6))
# # 将数据集传给绘图对象，并将数据点绘制为红色（表示最高气温）
# plt.plot(highs, c='red')
#
# # 设置图形的格式
# # 这是字体大小和标签
# plt.title("Closing Price", fontsize=24)
# # X轴暂不设置标签
# plt.xlabel('Time', fontsize=16)
# # 设置标签
# plt.ylabel("dollar", fontsize=16)
#
# plt.tick_params(axis='both', which='major', labelsize=16)
#
# plt.show()
#
#
#
# #visualization
# plt.plot(history.history['loss']    , label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()
#
#
# #predition
# predicted_stock_price = model.predict(X_test)                        # 测试集输入模型进行预测，经过reshape的30%的数据进行预测
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)  # 对预测数据还原---从（0，1）反归一化到原始范围
# real_stock_price      = sc.inverse_transform(y_test)# 对真实数据还原---从（0，1）反归一化到原始范围
#
# # 画出真实数据和预测数据的对比曲线
# plt.plot(real_stock_price, color='red', label='Stock Price')
# plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
# plt.title('Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()
#
# #comment
# """
# MSE  ：均方误差    ----->  预测值减真实值求平方后求均值
# RMSE ：均方根误差  ----->  对均方误差开方
# MAE  ：平均绝对误差----->  预测值减真实值求绝对值后求均值
# R2   ：决定系数，可以简单理解为反映模型拟合优度的重要的统计量
#
# """
# MSE   = metrics.mean_squared_error(predicted_stock_price, real_stock_price)
# RMSE  = metrics.mean_squared_error(predicted_stock_price, real_stock_price)**0.5
# MAE   = metrics.mean_absolute_error(predicted_stock_price, real_stock_price)
# R2    = metrics.r2_score(predicted_stock_price, real_stock_price)
#
# print('均方误差: %.5f' % MSE)
# print('均方根误差: %.5f' % RMSE)
# print('平均绝对误差: %.5f' % MAE)
# print('R2: %.5f' % R2)



data = pd.read_csv('./btc_sentiment.csv')  # 读取股票文件

# data

training_set = data.iloc[0:380, 1:2].values
test_set = data.iloc[380:, 1:2].values
data_fng = data.iloc[:, 1:2].values

plt.plot(training_set, c='red')
# X轴暂不设置标签
plt.title('training data of FNG value', fontsize=16)
plt.xlabel('Time', fontsize=16)
# 设置标签
plt.ylabel("value", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)

plt.show()

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
X_train, y_train = data_split(training_set_scaled, n_timestamp)#以窗口大小进行数据分割(数据已经经过归一化处理)  training 是百分之70的数据
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)#shape[0]读取数据行数。shape[1]读取数据列数 例子 reshape（3，4）重整为3行4列，若是-1则代表为未知

X_test, y_test = data_split(testing_set_scaled, n_timestamp) #testing 是百分之30的数据
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)#reshape的作用是什么？


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
    model.add(Dense(1))

if model_type == 3:
    # 双向 LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='tanh'),
                            input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))

model.summary()  # 输出模型结构



#动态学习率
import keras.backend as K
from keras.callbacks import LearningRateScheduler
def scheduler(epoch):
    # 每隔10个epoch，学习率减小为原来的1/10
    if epoch % 10 == 0 and epoch != 0:#epoch为10的整数而且不为零
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)



# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),#改动了0.0001
              loss='mean_squared_error')  # 损失函数用均方误差


history = model.fit(X_train, y_train,
                    batch_size=10, #改动了
                    epochs=n_epochs,
                    callbacks=[reduce_lr],
                    validation_data=(X_test, y_test),
                    validation_freq=1)                  #测试的epoch间隔数

model.summary()



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

plt.show()



#visualization
plt.plot(history.history['loss']    , label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


#predition
predicted_fng = model.predict(X_test)                        # 测试集输入模型进行预测，经过reshape的30%的数据进行预测
predicted_fng = sc.inverse_transform(predicted_fng)  # 对预测数据还原---从（0，1）反归一化到原始范围
real_fng      = sc.inverse_transform(y_test)# 对真实数据还原---从（0，1）反归一化到原始范围

# 画出真实数据和预测数据的对比曲线
plt.plot(real_fng, color='red', label='FNG')
plt.plot(predicted_fng, color='blue', label='Predicted fng')
plt.title('FNG Prediction')
plt.xlabel('Time')
plt.ylabel('FNG value')
plt.legend()
plt.show()

#comment
"""
MSE  ：均方误差    ----->  预测值减真实值求平方后求均值
RMSE ：均方根误差  ----->  对均方误差开方
MAE  ：平均绝对误差----->  预测值减真实值求绝对值后求均值
R2   ：决定系数，可以简单理解为反映模型拟合优度的重要的统计量

"""
MSE   = metrics.mean_squared_error(predicted_fng, real_fng)
RMSE  = metrics.mean_squared_error(predicted_fng, real_fng)**0.5
MAE   = metrics.mean_absolute_error(predicted_fng, real_fng)
R2    = metrics.r2_score(predicted_fng, real_fng)

print('均方误差: %.5f' % MSE)
print('均方根误差: %.5f' % RMSE)
print('平均绝对误差: %.5f' % MAE)
print('R2: %.5f' % R2)

