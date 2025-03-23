# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values. 

## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
```
Developed by: Meenakshi.R
RegisterNumber:212224220062
```
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
```

## Output:

![425357860-88cb02f0-d98e-4ace-8b99-486f160e728a](https://github.com/user-attachments/assets/bd9ae24a-ecbc-4f28-b3a7-f611a1bdade1)

![425358053-0384b301-5196-4b05-b644-ef33c5f9fad5](https://github.com/user-attachments/assets/4eba950e-bc3d-492c-8c0f-7dbc7a397e1a)

![425358218-6c6df437-ed23-4814-8790-9f420fd4a302](https://github.com/user-attachments/assets/706f3258-e03d-4bb8-a89d-1b031bb3cf92)

![425358503-a064ea05-6432-4c09-91ea-a1bd04f6c9cf](https://github.com/user-attachments/assets/453ee8bc-9259-46ba-85e1-277e64e5b1cd)

![425358693-bc2da33c-421b-45e4-a5ed-6f8688e4dae9](https://github.com/user-attachments/assets/c01b9701-be5a-4673-92dc-26ad42ecd17c)

![425358906-088d3ebf-cf47-4079-a81e-07335f3c229d](https://github.com/user-attachments/assets/dbd4a2ea-e24b-4936-b21e-4d21853ae551)

![425359009-b8ee1f56-a9bf-4e8c-acd7-e062f3536e1e](https://github.com/user-attachments/assets/5d8824e7-8e02-48cb-bd6c-83bb4c5b5d7b)

![425359197-2eeae7e6-91c3-454c-8c33-ab9f7b627889](https://github.com/user-attachments/assets/114c1a92-4379-47b4-b0db-b3b6bb458496)

![425359291-8c31fcdb-2c0e-4040-8a5b-ff5d1fafefc7](https://github.com/user-attachments/assets/c2e5aaea-9ebc-43bb-90f0-4aff4b445e0c)

![425359291-8c31fcdb-2c0e-4040-8a5b-ff5d1fafefc7](https://github.com/user-attachments/assets/bbb79462-dd2d-47f4-b985-e570a4e62011)

![425359356-528e8f28-a13f-4161-a161-724f40ac8951](https://github.com/user-attachments/assets/1d01c34d-567d-4bd5-a44c-c5f8948077d7)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
