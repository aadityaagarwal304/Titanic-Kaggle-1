# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

#import the file
train= pd.read_csv("C:/Users/aaditya_agarwal23/Desktop/Coding Analytics/Problem Solving/Kaggle/Titanic/train.csv")
test= pd.read_csv("C:/Users/aaditya_agarwal23/Desktop/Coding Analytics/Problem Solving/Kaggle/Titanic/test.csv")


train['Source']='train'
test['Source']='test'

#concate the data
data= pd.concat([train,test],ignore_index='True')

#seperating title from name
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
    
    
data['Title'] = data['Name'].map(lambda x: get_title(x))

#Dropping column
data.drop('Name',axis=1,inplace=True)
data.drop('Ticket',axis=1,inplace=True)
data.drop('Cabin',axis=1,inplace=True)
data.drop('PassengerId',axis=1,inplace=True)
#Filling missing value
data.Embarked.fillna("S",inplace=True)

#One hot encoding for Embarked
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data['Embarked']=le.fit_transform(data['Embarked'])

data=pd.get_dummies(data,columns=['Embarked'],)

data=data.rename(columns={'Embarked_0':'Embarked_C', 'Embarked_1':'Embarked_Q', 'Embarked_2':'Embarked_S'})

#Missing value of age by grouping acc to title
data["Age"] = data.groupby("Title").transform(lambda x: x.fillna(x.mean()))

#Missing value of age by grouping acc to title
data["Fare"] = data.groupby("Pclass").transform(lambda x: x.fillna(x.mean()))

#One hot encoding for Sex
data['Sex']=le.fit_transform(data['Sex'])

data=pd.get_dummies(data,columns=['Sex'],)

data=data.rename(columns={'Sex_0':'Sex_Female', 'Sex_1':'Sex_Male'})

#Dropping column
data.drop('Title',axis=1,inplace=True)

#Converting data back to test and train
train = data.loc[data['Source']=="train"]
test = data.loc[data['Source']=="test"]

train.drop('Source',axis=1,inplace=True)
test.drop('Source',axis=1,inplace=True)

train_y= train.Survived

train_X=train.drop('Survived',axis=1,inplace=True)

test_X=test.drop('Survived',axis=1,inplace=True)

#Using Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#train, train_y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train, train_y)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

test_y= clf.predict(test)

