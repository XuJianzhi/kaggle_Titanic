
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import sklearn as sk
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

way='/home/m/Titanic/'
train=pd.read_csv(way+'train.csv')
#test=pd.read_csv(way+'test.csv')

train_x=train.set_index('PassengerId').drop(['Survived','Name','Ticket','Cabin'],axis=1)
train_y=train.set_index('PassengerId')['Survived']
#test_x=test.set_index('PassengerId').drop(['Name','Ticket','Cabin'],axis=1)

#put age to some parts to use in order to decrease overfitting
#train_x.Age[train_x.Age.isnull()==False] = (train_x.Age[train_x.Age.isnull()==False]/5).apply(int)
n_age=30
train_x.Age[train_x.Age.isnull()==False] = pd.qcut(train_x.Age[train_x.Age.isnull()==False],n_age,labels=False)

#fare : 0->nan, and to some parts
train_x.Fare[train_x.Fare==0.0] = np.nan
#train_x.Fare[train_x.Fare.isnull()==False] = (train_x.Fare[train_x.Fare.isnull()==False]/5).apply(int)
n_fare=20
train_x.Fare[train_x.Fare.isnull()==False] = pd.qcut(train_x.Fare[train_x.Fare.isnull()==False],n_fare,labels=False)


#encoder
lbl_1=sk.preprocessing.LabelEncoder()
train_x['Sex']=lbl_1.fit_transform(train_x['Sex'])
#test_x['Sex']=lbl_1.transform(test_x['Sex'])

#lbl_2=sk.preprocessing.LabelEncoder()
#train_x['Age']=lbl_2.fit_transform(train_x['Age'])
#test_x['Age']=lbl_2.transform(test_x['Age'])

lbl_3=sk.preprocessing.LabelEncoder()
train_x['Embarked']=lbl_3.fit_transform(train_x['Embarked'])
#test_x['Embarked']=lbl_3.transform(test_x['Embarked'])

#lbl_4=sk.preprocessing.LabelEncoder()
#train_x['Fare']=lbl_4.fit_transform(train_x['Fare'])
#test_x['Fare']=lbl_4.transform(test_x['Fare'])



#split
train_x, test_x, train_y, y_real = sk.model_selection.train_test_split(train_x,train_y,test_size=0.2)


########################
dtrain=xgb.DMatrix(train_x,train_y)
dtest=xgb.DMatrix(test_x)	###

watchlist=[(dtrain,'train'),(dtrain,'test')]
num_class=train_y.max()+1  
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
            'gamma' : 2,
            'subsample' : 0.5,
            'alpha' : 0.5
            }
num_rounds=1000

#clf=xgb.train(params,dtrain,num_rounds,watchlist)
clf=xgb.train(params,dtrain,num_rounds,watchlist)

test_y=pd.Series(clf.predict(dtest),index=test_x.index)
test_y=pd.DataFrame(clf.predict(dtest),index=test_x.index,columns=['Survived'])
test_y['Survived']=test_y['Survived'].apply(int)
test_y=test_y.reset_index()

#way_out='/home/m/Titanic/result/11.20/'
#test_y.to_csv(way_out+'result.csv')


'''
#self all test
dtest_x_self=xgb.DMatrix(train_x)
y_self=pd.Series(clf.predict(dtest_x_self)).apply(int)
y_self=train['Survived']

choice=y_self[y_self==y_self]
right_num=len(choice[choice])	#the result is all right
'''

#self rate test
dtest_x_self=xgb.DMatrix(test_x)
y_pred=pd.Series(clf.predict(dtest_x_self),index=test_x.index).apply(int)


choice = y_pred==y_real
right_num=len(choice[choice])	# 0.770949720670391
print(float(right_num)/len(y_pred))












