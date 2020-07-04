# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:03:06 2019

@author: 10365
"""
#SvmLearning
#SVM Method


from sklearn.externals import joblib  #保存训练结果
import xgboost as xgb
import pandas as pds
import numpy as np
import matplotlib.pyplot as mplt
from sklearn.model_selection import GridSearchCV

params = {
 #通用参数 
'booster': 'gbtree',         #选择基分类器
'silent': 1,      #设置成1则没有运行信息输出，最好是设置为0
'nthread': 4,          #线程数
 #Tree Booster 参数
'objective': 'reg:squarederror',     #定义最小化损失函数类型
#'objective': 'binary:logistic',
'n_estimators':82,          #基分类器个数
'alpha': 0.001,               #正则化参数
'gamma': 0,             #后剪枝时，用于控制是否后剪枝的参数         
'max_depth': 10,           #每颗树的最大深度
'lambda': 1,            #控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
'subsample': 0.8,       #样本随机采样，较低的值使得算法更加保守，防止过拟合
'colsample_bytree': 0.7, #列采样，对每棵树的生成用的特征进行列采样.一般设置为： 0.5-1
'min_child_weight': 3,   #这个参数默认是 1，是每个叶子里面 h的和至少是多少    
'eta': 0.3,     #学习率
'seed': 1000,     #随机数的种子
'eval_metric':'rmse',
'scale_pos_weight':10
}

plst=params.items()
num_round=3000    #训练1000轮
#读取数据
filedir='./TrainningDataSet/'
fileNum=25
trainData=pds.read_csv(filedir+str(1)+'_dataSet.csv')
for i in range(2,fileNum+1):
    data=pds.read_csv(filedir+str(i)+'_dataSet.csv')
    trainData=trainData.append(data)
trainAry=trainData.values
x_train=trainAry[:,0:-2]    
y_train=trainAry[:,-1]
dtrain = xgb.DMatrix(x_train,y_train)
evals_res={}
watchlist = [(dtrain,'eval')]
cv_res= xgb.cv(params,dtrain,num_boost_round=num_round,early_stopping_rounds=30,nfold=5,metrics={'mae'},show_stdv=False)
model = xgb.train(plst, dtrain, num_boost_round=cv_res.shape[0],evals=watchlist, evals_result=evals_res)
#打印曲线
roundNumList=range(0,len(evals_res['eval']['rmse']))
mplt.plot(roundNumList,evals_res['eval']['rmse'])
mplt.show()
LossCurve=pds.DataFrame([roundNumList,evals_res['eval']['rmse']])
LossCurve = pds.DataFrame(LossCurve.values.T)
LossCurve.to_csv("LossCurve.csv",header=None,index=None)
#保存模型
model.save_model('xgb.model')




#调参过程----------------------------------------------------------------------------

#确定学习速率和tree_based 参数调优的估计器数目
#param_test2 = {
# 'n_estimators':range(20,100,1),
# 'learning_rate':[x*0.1 for x in range(0,5,1)]
#}
#xgb_model = xgb.XGBRegressor(booster='gbtree',learning_rate =0.1, n_estimators=10, max_depth=3,\
#min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'reg:squarederror', nthread=4,\
#scale_pos_weight=1, seed=0)
#gsearch2 = GridSearchCV(estimator =xgb_model, param_grid = param_test2,  verbose=1)
#gsearch2.fit(x_train,y_train)
#print(gsearch2.error_score)
#print('优化结果：')
#print(gsearch2.best_params_, gsearch2.best_score_)


#max_depth 和 min_weight 参数调优
#param_test1 = {
# 'max_depth':range(1,10,1),
# 'min_child_weight':range(1,6,1)
#}
#xgb_model = xgb.XGBRegressor(booster='gbtree',learning_rate =0.4, n_estimators=24, max_depth=3,\
#min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'reg:squarederror', nthread=4,\
#scale_pos_weight=1, seed=0)
#gsearch1 = GridSearchCV(estimator =xgb_model, param_grid = param_test1,  verbose=1)
#gsearch1.fit(x_train,y_train)
#print('最优的max_depth 和 min_weight：')
#print(gsearch1.best_params_, gsearch1.best_score_)

#gamma参数调优
#param_test3 = {
# 'gamma':[i/100.0 for i in range(0,20)]
#}
#xgb_model = xgb.XGBRegressor(booster='gbtree',learning_rate =0.4, n_estimators=24, max_depth=3,\
#min_child_weight=3, gamma=0.1, subsample=0.8, colsample_bytree=0.8,objective= 'reg:squarederror', nthread=4,\
#scale_pos_weight=1, seed=0)
#gsearch3 = GridSearchCV(estimator =xgb_model, param_grid = param_test3,  verbose=1)
#gsearch3.fit(x_train,y_train)
#print('最优的gamma：')
#print(gsearch3.best_params_, gsearch3.best_score_)

#调整subsample 和 colsample_bytree 参数
#param_test4 = {
# 'subsample':[i/10.0 for i in range(1,10)],
# 'colsample_bytree':[i/10.0 for i in range(1,10)]
#}
#xgb_model = xgb.XGBRegressor(booster='gbtree',learning_rate =0.4, n_estimators=24, max_depth=3,\
#min_child_weight=3, gamma=0.02, subsample=0.8, colsample_bytree=0.7,objective= 'reg:squarederror', nthread=4,\
#scale_pos_weight=1, seed=0)
#gsearch4 = GridSearchCV(estimator =xgb_model, param_grid = param_test4,  verbose=1)
#gsearch4.fit(x_train,y_train)
#print('最优的subsample 和 colsample_bytree 参数')
#print(gsearch4.best_params_, gsearch4.best_score_)

#正则化参数调优
#param_test5 = {
# 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
# 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
#}
#xgb_model = xgb.XGBRegressor(booster='gbtree',learning_rate =0.4, n_estimators=24, max_depth=3,\
#min_child_weight=3, gamma=0.02, subsample=0.8, colsample_bytree=0.7,objective= 'reg:squarederror', nthread=4,\
#scale_pos_weight=1, seed=0)
#gsearch5 = GridSearchCV(estimator =xgb_model, param_grid = param_test5,  verbose=1)
#gsearch5.fit(x_train,y_train)
#print('最优的subsample 和 colsample_bytree 参数')
#print(gsearch5.best_params_, gsearch5.best_score_)




#测试数据
filedir='./TestingdataSet/'
fileNum=25
testData=pds.read_csv(filedir+str(1)+'_dataSet.csv')
for i in range(2,fileNum+1):
    data=pds.read_csv(filedir+str(i)+'_dataSet.csv')
    testData=testData.append(data)
testAry=testData.values
x_test=testAry[:,0:-2]    
y_test=testAry[:,-1]
dtest=xgb.DMatrix(x_test)
bst=xgb.Booster(model_file='xgb.model')
#,ntree_limit=bst.best_iteration
preds = bst.predict(dtest)
index=range(0,len(preds))
error=preds-y_test
mplt.scatter(index,preds)
mplt.scatter(index,y_test)
mplt.show()
mplt.scatter(index,error)
mplt.show()
print('训练轮数：{0}'.format(num_round,'%d'))
print('标准差:{0}'.format(np.std(error),'%f'))
xgb.plot_importance(bst)