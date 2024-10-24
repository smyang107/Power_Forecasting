#参考：https://blog.csdn.net/Jason160918/article/details/80039804
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import lightgbm as lgb_model

###### 处理数据
SEED=123#设立随机种子以便结果复现
np.random.seed(SEED)
X = pd.read_csv('./data/data1_X.csv')
X=X.values
print(X.shape)#(1343, 5)numpy.ndarray
Y = pd.read_csv('./data/data1_Y.csv')
Y=Y.values
print(Y.shape)#(1343, 1)

data=np.append(X, Y, axis=1)# data：(1343, 6)
data=np.random.permutation(data)#默认对第一维打乱
X=data[:,:4]
Y=data[:,5]

# X=pd.DataFrame(data['year'],data['month'],data['day'],data['minute'])

X_train=X[:int(X.shape[0]*0.9),:].astype('int')
# X_test=X[int(X.shape[0]*0.8):int(X.shape[0]*0.9),:]
X_test=X[int(X.shape[0]*0.9):,:].astype('int')
Y_train=Y[:int(Y.shape[0]*0.9)].astype('int')
# Y_test=Y[int(Y.shape[0]*0.8):int(Y.shape[0]*0.9)]
Y_test=Y[int(Y.shape[0]*0.9):].astype('int')

############### 定义函数
def get_models():
#"""Generate a library of base learners."""
    nb = GaussianNB()#朴素贝叶斯
    svc = SVC(C=1,random_state=SEED,kernel="linear" ,probability=True)#kernel选用线性最好，因为kerenl太复杂容易过拟合，支持向量机
    knn = KNeighborsClassifier(n_neighbors=3)#K近邻聚类
    lr = LogisticRegression(C=100, random_state=SEED)#逻辑回归
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)#多层感知器
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)#GDBT
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)#随机森林
    etree=ExtraTreesClassifier(random_state=SEED)#etree
    adaboost=AdaBoostClassifier(random_state=SEED)#adaboost
    dtree=DecisionTreeClassifier(random_state=SEED)#决策树
    lgb=lgb_model.sklearn.LGBMClassifier(is_unbalance=False,learning_rate=0.04,n_estimators=110,max_bin=400,scale_pos_weight=0.8)#lightGBM，需要安装lightGBM，pip3 install lightGBM

    models = {
              'svm': svc,
              'knn': knn,
              'naive bayes': nb,
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr,
              'etree': etree,
              'adaboost': adaboost,
              'dtree': dtree,
              'lgb': lgb,
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

def score_models(P, y):#打印AUC值
#"""Score model in prediction DF"""
    print("Scoring AUC的值models.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")


###查看其模型预测AUC分数
from sklearn.metrics import roc_auc_score
base_learners = get_models()
P = train_predict(base_learners)
score_models(P, Y_test)

### 查看各个模型结果之间的相关性，相关性低的话集成的效果越好
from mlens.visualization import corrmat
import matplotlib.pyplot as plt
corrmat(P.corr(), inflate=False)
plt.show()


#### 定义元学习器GBDT
meta_learner = GradientBoostingClassifier(

    n_estimators=1000,

    loss="exponential",

    max_features=3,

    max_depth=5,

    subsample=0.8,

    learning_rate=0.05,

    random_state=SEED
)

meta_learner.fit(P, Y_test)#用基学习器的预测值作为元学习器的输入，并拟合元学习器，元学习器一定要拟合，不然无法集成。

GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.05, loss='exponential', max_depth=5,
              max_features=3, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=1000,
              presort='auto', random_state=123, subsample=0.8, verbose=0,
              warm_start=False)


from mlens.ensemble import SuperLearner
#5折集成
sl = SuperLearner(
    folds=5,
    random_state=SEED,
    verbose=2,
    backend="multiprocessing"
)

sl.add(list(base_learners.values()), proba=True) # 加入基学习器
sl.add_meta(meta_learner, proba=True)# 加入元学习器
# 训练集成模型
sl.fit(X_train, Y_train[:1000])
# 预测
p_sl = sl.predict_proba(X_test)
print("\n超级学习器的AUC值: %.3f" % roc_auc_score(Y_test, p_sl[:, 1]))

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#画roc曲线
def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):

    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    cm = [plt.cm.rainbow(i)
      for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]
    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])
    fpr, tpr, _ = roc_curve(ytest, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right",frameon=False)
    plt.show()
