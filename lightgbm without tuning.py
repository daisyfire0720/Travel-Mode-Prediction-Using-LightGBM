#Set working directory
import os
os.chdir('C:\\Users\Daisy\\Desktop\\TRB Travel Mode\\NHTS data\\CSV data')
#import training and testing dataset
import pandas as pd
#import training data and testing data
train_nhts = pd.read_csv('tripfortrain_FinalSelection.csv')
test_nhts = pd.read_csv('tripfortest_FinalSelection.csv')
#split variables and response
train_x = train_nhts.drop(['TRPTRANS'], axis = 1, inplace = False)
train_y = train_nhts[['TRPTRANS']]
test_x = test_nhts.drop(['TRPTRANS'], axis = 1, inplace = False)
test_y = test_nhts[['TRPTRANS']]
#change training data type
train_x['EDUC'] = train_x['EDUC'].astype('category')
train_x['HHFAMINC'] = train_x['HHFAMINC'].astype('category')
train_x['TRANS'] = train_x['TRANS'].astype('category')
train_x['WORKER'] = train_x['WORKER'].astype('category')
train_x['HTHTNRNT'] = train_x['HTHTNRNT'].astype('category')
train_x['HTPPOPDN'] = train_x['HTPPOPDN'].astype('category')
train_x['WHYTRP1S'] = train_x['WHYTRP1S'].astype('category')
train_x['LIF_CYC'] = train_x['LIF_CYC'].astype('category')
train_x['TRIPPURP'] = train_x['TRIPPURP'].astype('category')
train_x['URBAN'] = train_x['URBAN'].astype('category')
train_x['URBANSIZE'] = train_x['URBANSIZE'].astype('category')
train_x['RAIL'] = train_x['RAIL'].astype('category')
train_x['R_SEX'] = train_x['R_SEX'].astype('category')
#change testing data type
test_x['EDUC'] = test_x['EDUC'].astype('category')
test_x['HHFAMINC'] = test_x['HHFAMINC'].astype('category')
test_x['TRANS'] = test_x['TRANS'].astype('category')
test_x['WORKER'] = test_x['WORKER'].astype('category')
test_x['HTHTNRNT'] = test_x['HTHTNRNT'].astype('category')
test_x['HTPPOPDN'] = test_x['HTPPOPDN'].astype('category')
test_x['WHYTRP1S'] = test_x['WHYTRP1S'].astype('category')
test_x['LIF_CYC'] = test_x['LIF_CYC'].astype('category')
test_x['TRIPPURP'] = test_x['TRIPPURP'].astype('category')
test_x['URBAN'] = test_x['URBAN'].astype('category')
test_x['URBANSIZE'] = test_x['URBANSIZE'].astype('category')
test_x['RAIL'] = test_x['RAIL'].astype('category')
test_x['R_SEX'] = test_x['R_SEX'].astype('category')
#import sklearn package
import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
#import lightgbm package
import lightgbm as lgb
import numpy as np
#build lightgbm dataset
lgb_train = lgb.Dataset(train_x,train_y,silent = True)
lgb_eval = lgb.Dataset(test_x,test_y,reference = lgb_train)
#build the final model
default_params = {
    'boosting_type':'gbdt',
    'objective':'multiclass',
    'metric':'multi_logloss',
    'num_class':6,
    'silent':True,
    'learning_rate':0.1,
    'num_leaves':31,
    'min_child_samples':20,
    'min_child_weight':0.001,
    'bagging_fraction':1,
    'feature_fraction':1,
    'reg_alpha':0,
    'reg_lambda':0
}
#record model build time 
import time
start_time = time.time()
lgb_model_optimal = lgb.train(default_params,
                              lgb_train,
                              num_boost_round = 100,
                              early_stopping_rounds = 50,
                              valid_sets = lgb_eval
                              )
print("--- %s seconds ---" % round(time.time() - start_time, 2))

#confusion matrix and overall accuracy for test dataset
from sklearn.metrics import accuracy_score
preds_test = lgb_model_optimal.predict(test_x,num_iteration = lgb_model_optimal.best_iteration)
for pred in preds_test:
    result = prediction = int(np.argmax(pred))
pred_y_test = [list(x).index(max(x)) for x in preds_test]
print(accuracy_score(test_y,pred_y_test))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y,pred_y_test))

#confusion matrix and overall accuracy for train dataset
from sklearn.metrics import accuracy_score
preds_train = lgb_model_optimal.predict(train_x,num_iteration = lgb_model_optimal.best_iteration)
for pred in preds_train:
    result = prediction = int(np.argmax(pred))
pred_y_train = [list(x).index(max(x)) for x in preds_train]
print(accuracy_score(train_y,pred_y_train))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(train_y,pred_y_train))