import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
features = pd.read_csv('./data/Allfeature.csv')
x=features.values[:,2:]#(1343, 7)
# x = pd.DataFrame(np.random.rand(100, 8))

###### 方差膨胀系数
# 当VIF<10,说明不存在多重共线性；当10<=VIF<100,存在较强的多重共线性，当VIF>=100,存在严重多重共线性
# vif = [variance_inflation_factor(x.values, x.columns.get_loc(i)) for i in x.columns]
vif = [variance_inflation_factor(x, i) for i in range(x.shape[1])]

print(vif)
#[139.43119154083797, 184.79747027635932, 47.735705518094875, 2.454199397296282, 11.503324331886358, 9.03396898423558]


###特征值（Eigenvalue）
#该方法实际上就是对自变量做主成分分析，如果多个维度的特征值等于0，则可能有比较严重的共线性。
# eigenvalue, featurevector = np.linalg.eig(x)
# print(eigenvalue)
# print(featurevector)

### 相关系数
##如果相关系数R>0.8时就可能存在较强相关性
x = pd.DataFrame(x)
print(x.corr())