#参考：https://blog.csdn.net/Jason160918/article/details/80039804
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Bidirectional
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
import pandas as pd
import lightgbm as lgb
import numpy as np
import lightgbm as lgb_model
import pickle
import joblib

###### 处理数据
SEED=0#设立随机种子以便结果复现
np.random.seed(SEED)
AllX = pd.read_csv('data/AllData1_X.csv') ### 加上气候信息、时间信息和新增的特征
X=AllX.iloc[:119336, :].values
# X=AllX.iloc[:100, :].values
print(X.shape)#(1343, 5)numpy.ndarray
AllY = pd.read_csv('data/data1_Y.csv')
Y=AllY.iloc[:119336, :].values
# Y=AllY.iloc[:100, :].values
print(Y.shape)#(1343, 1)


### 读取新曾特征
Features = pd.read_csv('data/Allfeature.csv')
features=Features.values[:119336,2:]
# features=Features.values[:100,2:]
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

############### 定义函数
def get_models():
#"""Generate a library of base learners."""
    rfr = RandomForestRegressor(n_estimators=200,random_state=0)
    xgboost = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150)
    svr = SVR(kernel='linear')#LinearSVC
    br=BayesianRidge()
    # lstm = Sequential()
    # lstm.add(LSTM(units=50, activation='tanh', #神经单元50个，激活函数为relu函数，输入X_train的列数量，也就是说！把X_train 70%的数据拿去训练
    #                input_shape=(X_train.shape[1], 1)))
    # lstm.add(Dense(units=1))
    models = {
              'random forest': rfr,
              'XGBRegressor': xgboost,
              'SVR':svr,
              'BayesianRidge':br
             }
    return models

def train_predict(model_list):#预测
    """将每个模型的预测值保留在DataFrame中，行是每个样本预测值，列是模型"""
    P = np.zeros((Y_test.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(model_list.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(X_train, Y_train)
        P.iloc[:, i] = m.predict(X_test)
        cols.append(name)
        print("done")
    P.columns = cols
    print("ALL model Done.\n")
    return P #预测结果（所有基于预测模型1）

def R2(y_test,y_pred):
    SStot=np.sum((y_test-np.mean(y_test))**2)
    SSres=np.sum((y_test-y_pred)**2)
    r2=1-SSres/SStot
    return r2


if __name__ == "__main__":   # 加上这一行即可
    ###查看基预测器的预测结果
    base_learners = get_models()
    P = train_predict(base_learners)
    # score_models(P, Y_test)

    # ### 查看各个模型结果之间的相关性，相关性低的话集成的效果越好
    # from mlens.visualization import corrmat
    # import matplotlib.pyplot as plt
    # corrmat(P.corr(), inflate=False)
    # plt.show()

    #### 定义元学习器GBDT
    meta_learner = GradientBoostingRegressor(
        learning_rate=0.01,
        n_estimators=250,
        criterion='mse'
    )

    meta_learner.fit(P, Y_test)#用基学习器的预测值作为元学习器的输入，并拟合元学习器，元学习器一定要拟合，不然无法集成。

    # GradientBoostingClassifier(criterion='friedman_mse', init=None,
    #               learning_rate=0.05, loss='exponential', max_depth=5,
    #               max_features=3, max_leaf_nodes=None,
    #               min_impurity_decrease=0.0, min_impurity_split=None,
    #               min_samples_leaf=1, min_samples_split=2,
    #               min_weight_fraction_leaf=0.0, n_estimators=1000,
    #               presort='auto', random_state=123, subsample=0.8, verbose=0,
    #               warm_start=False)

    from mlens.ensemble import SuperLearner
    #9折集成
    sl = SuperLearner(
        folds=9,
        random_state=SEED,
        verbose=2,
        backend="multiprocessing"
    )

    # sl.add(list(base_learners.values()), proba=True) # 加入基学习器
    # sl.add_meta(meta_learner, proba=True)# 加入元学习器
    sl.add(list(base_learners.values()))  # 加入基学习器
    sl.add_meta(meta_learner)  # 加入元学习器
    # 训练集成模型
    sl.fit(X_train, Y_train)
    # 预测
    Y1 = sl.predict(X_train)
    Y2 = sl.predict(X_test)
    # print("\n超级学习器的AUC值: %.3f" % roc_auc_score(Y_test, p_sl[:, 1]))
    # print(p_sl)
    ########### train集合  ###########
    print("#########  boosting  ###########")
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

    ########### test集合  ###########
    print("#########  Test  ###########")

    # MSE
    MSE = np.linalg.norm(Y2 - Y_test, ord=2) ** 2 / len(Y_test)
    print("MSE", MSE)  # 79.26742582898571

    # RMSE--相当于y-y_hat的二阶范数/根号n
    RMSE = np.linalg.norm(Y2 - Y_test, ord=2) / len(Y_test) ** 0.5
    print("RMSE", RMSE)  # 8.903225585650725

    ### MAE--相当于y-y_hat的一阶范数/n
    MAE = np.linalg.norm(Y2 - Y_test, ord=1) / len(Y_test)
    print("MAE", MAE)  # 6.396413824234643

    ##
    MAPE = np.mean(np.abs((Y2 - Y_test) / Y_test)) * 100

    print("MAPE", MAPE)  # 8.601955442711105

    r2 = R2(Y_test, Y2)
    print("R2", r2)

    # s = pickle.dumps(sl)
    # with open("./res/boost_tianqi.model", 'wb+') as f:  # 注意此处mode是'wb+'，表示二进制写入
    #     f.write(s)
    # with open("/res/boost_tianqi.pkl", 'wb') as fo:
    #     joblib.dump(data, fo)
    joblib.dump(sl,"./res/boost_tianqi.pkl")
                # IndexError: index 1 is out of bounds for axis 0 with size 1
    #Traceback (most recent call last):File "test_boosting.py", line 216, in <module>s = pickle.dumps(sl)MemoryError

    ######################## 结果########################

    ####给出该地区电网未来 3 个月日负荷的最大值和最小值预测结果，以及相应达到负荷最大值和最小值的时间，并分析其预测精度。
    # 2021-5-31  23:45:00--->2021-8-31

    print('############  3 month prediction ############')
    pre_X = AllX.iloc[119336:, :].values
    pre_Y = AllY.iloc[119336:, :].values
    features = Features.values[119336:, 2:]
    # pre_X=AllX.iloc[-200:, :].values
    # pre_Y=AllY.iloc[-200:, :].values
    # features=Features.values[-200:,2:]
    pre_X = np.concatenate((pre_X, features), axis=1)
    Y3 = sl.predict(pre_X)

    MSE = np.linalg.norm(Y3 - pre_Y, ord=2) ** 2 / len(pre_Y)
    print("MSE", MSE)  # 79.26742582898571

    # RMSE--相当于y-y_hat的二阶范数/根号n
    RMSE = np.linalg.norm(Y3 - pre_Y, ord=2) / len(pre_Y) ** 0.5
    print("RMSE", RMSE)  # 8.903225585650725

    ### MAE--相当于y-y_hat的一阶范数/n
    MAE = np.linalg.norm(Y3 - pre_Y, ord=1) / len(pre_Y)
    print("MAE", MAE)  # 6.396413824234643

    ##
    MAPE = np.mean(np.abs((Y3 - pre_Y) / pre_Y)) * 100

    print("MAPE", MAPE)  # 8.601955442711105

    r2 = R2(pre_Y, Y3)
    print("R2", r2)
    Y3 = pd.DataFrame(Y3)
    Y3.to_csv('res/boost_tianqi_3_month.csv')

    #### 给出该地区电网未来 10 天间隔 15 分钟的负荷预测结果，并分析其预测精度；
    # 2021-8-21 23 45 23:45:00--->2021-8-31  23 45

    print('############ 10 day prediction ############')
    # 训练
    # f = open('./res/boost_tianqi.model', 'rb')  # 注意此处model是rb
    # s = f.read()
    # model = pickle.loads(s)
    # with open("./res/boost_tianqi.pkl", 'rb') as fo:
    #     joblib.load(fo)
    model = joblib.load("./res/boost_tianqi.pkl")
    X_train = AllX.iloc[119336:127196, :].values
    Y_train = AllY.iloc[119336:127196, :].values
    features = Features.values[119336:127196:, 2:]
    # X_train=AllX.iloc[-200:-100, :].values
    # Y_train=AllY.iloc[-200:-100, :].values
    # features=Features.values[-200:-100:,2:]
    X_train = np.concatenate((X_train, features), axis=1)
    model.fit(X_train, Y_train)

    # 预测
    pre_X = AllX.iloc[127196:, :].values
    pre_Y = AllY.iloc[127196:, :].values
    features = Features.values[127196:, 2:]
    # pre_X=AllX.iloc[-100:, :].values
    # pre_Y=AllY.iloc[-100:, :].values
    # features=Features.values[-100:,2:]
    pre_X = np.concatenate((pre_X, features), axis=1)
    Y3 = model.predict(pre_X)

    MSE = np.linalg.norm(Y3 - pre_Y, ord=2) ** 2 / len(pre_Y)
    print("MSE", MSE)  # 79.26742582898571

    # RMSE--相当于y-y_hat的二阶范数/根号n
    RMSE = np.linalg.norm(Y3 - pre_Y, ord=2) / len(pre_Y) ** 0.5
    print("RMSE", RMSE)  # 8.903225585650725

    ### MAE--相当于y-y_hat的一阶范数/n
    MAE = np.linalg.norm(Y3 - pre_Y, ord=1) / len(pre_Y)
    print("MAE", MAE)  # 6.396413824234643

    ##
    MAPE = np.mean(np.abs((Y3 - pre_Y) / pre_Y)) * 100

    print("MAPE", MAPE)  # 8.601955442711105

    r2 = R2(pre_Y, Y3)
    print("R2", r2)

    Y3 = pd.DataFrame(Y3)
    Y3.to_csv('res/boost_tianqi_10_day.csv')
