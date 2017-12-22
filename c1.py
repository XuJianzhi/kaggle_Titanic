
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import sklearn as sk
import numpy as np
import xgboost as xgb

way='/home/m/Titanic/'
train=pd.read_csv(way+'train.csv')
test=pd.read_csv(way+'test.csv')

train_x=train.set_index('PassengerId').drop(['Survived','Name','Ticket','Cabin'],axis=1)
train_y=train.set_index('PassengerId')['Survived']
test_x=test.set_index('PassengerId').drop(['Name','Ticket','Cabin'],axis=1)

#put age to some part to use in order to decrease overfitting
n_age=15
train_x.Age[train_x.Age.isnull()==False] = pd.qcut(train_x.Age[train_x.Age.isnull()==False],n_age,labels=False)

#fare : 0->nan, and to some parts
train_x.Fare[train_x.Fare==0.0] = np.nan
#train_x.Fare[train_x.Fare.isnull()==False] = (train_x.Fare[train_x.Fare.isnull()==False]/5).apply(int)
n_fare=3
train_x.Fare[train_x.Fare.isnull()==False] = pd.qcut(train_x.Fare[train_x.Fare.isnull()==False],n_fare,labels=False)

#encoder
lbl_1=sk.preprocessing.LabelEncoder()
train_x['Sex']=lbl_1.fit_transform(train_x['Sex'])
test_x['Sex']=lbl_1.transform(test_x['Sex'])

lbl_3=sk.preprocessing.LabelEncoder()
train_x['Embarked']=lbl_3.fit_transform(train_x['Embarked'])
test_x['Embarked']=lbl_3.transform(test_x['Embarked'])



########################
dtrain=xgb.DMatrix(train_x,train_y)
dtest=xgb.DMatrix(test_x)

watchlist=[(dtrain,'train'),(dtrain,'test')]
num_class=train_y.max()+1  
'''
params = {
            'objective': 'multi:softmax',
            'eta': 0.1,
            'max_depth': 7,
            'eval_metric': 'merror',
            'seed': 0,
            'missing': -999,
            'num_class':num_class,
            'silent' : 1
            }
'''
params = {
            'objective': 'multi:softmax',
            #'objective': 'binary:logistic',
            'eta': 0.01,
            'eval_metric': 'merror',
            #'eval_metric': 'mlogloss',
            'seed': 0,
            'missing': -999,
            'num_class':num_class,
            'silent' : 1,
            'gamma' : 1,
            'subsample' : 0.5,
            'alpha' : 0.5
            }
num_rounds=500

clf=xgb.train(params,dtrain,num_rounds,watchlist)
#test_y=pd.Series(clf.predict(dtest),index=test_x.index)
test_y=pd.DataFrame(clf.predict(dtest),index=test_x.index,columns=['Survived'])
test_y['Survived']=test_y['Survived'].apply(int)
test_y=test_y.reset_index()

way_out='/home/m/Titanic/result/11.20/'
test_y.to_csv(way_out+'result.csv',index=False)



#self test
dtest_x_self=xgb.DMatrix(train_x)
y_self=pd.Series(clf.predict(dtest_x_self)).apply(int)
y_real=train['Survived']

choice=y_self==y_real
right_num=len(choice[choice])	#the result is all right
print(float(right_num)/len(train))










