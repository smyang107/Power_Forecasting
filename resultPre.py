import pandas as pd
import numpy as np

def R2(y_test,y_pred):
    SStot=np.sum((y_test-np.mean(y_test))**2)
    SSres=np.sum((y_test-y_pred)**2)
    r2=1-SSres/SStot
    return r2
print('############ 10 day prediction ############')
Y3 = pd.read_csv('res/RF_tianqi_10_day.csv').values[:,1:2]
AllY = pd.read_csv('data/data1_Y.csv')
pre_Y=AllY.iloc[127196:, :].values

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



print('############  3 month prediction ############')
Y3 = pd.read_csv('res/RF_tianqi_3_month.csv').values[:,1:2]
AllY = pd.read_csv('data/data1_Y.csv')
pre_Y=AllY.iloc[119336:, :].values

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