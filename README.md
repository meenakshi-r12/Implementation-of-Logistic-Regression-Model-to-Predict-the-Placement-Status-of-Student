# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
```
import pandas as pd
import numpy as np
df=pd.read_csv('Placement_Data.csv')
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis = 1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['gender']=le.fit_transform(df1['gender'])
df1['ssc_b']=le.fit_transform(df1['ssc_b'])
df1['hsc_b']=le.fit_transform(df1['hsc_b'])
df1['hsc_s']=le.fit_transform(df1['hsc_s'])
df1['degree_t']=le.fit_transform(df1['degree_t'])
df1['workex']=le.fit_transform(df1['workex'])
df1['specialisation']=le.fit_transform(df1['specialisation'])
df1['status']=le.fit_transform(df1['status'])
df1

x=df1.iloc[:,:-1]
x

y=df1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
