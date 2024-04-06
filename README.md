# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries. 2.Upload and read the dataset. 3.Check for any null values using the isnull() function. 4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy. 5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.



## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NITHISH S
RegisterNumber:  212223220070
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
Data.head():

![318630456-311a67fb-1265-447c-938d-0c0272211d09](https://github.com/Nithish23013509/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149038138/fcd4e107-c4a0-4106-848d-cfb8d3dd746b)

Data.info():

![318630465-0d68fe8f-ca3a-497c-b849-cc795adc9da7](https://github.com/Nithish23013509/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149038138/b7307bd2-dee7-4272-aed8-aa764bb3f9ce)

isnull() and sum():

![318630486-a6d85076-d39f-4a37-9cc7-4faa482709b6](https://github.com/Nithish23013509/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149038138/e3dc2a21-6eb2-4901-a62d-2a2a71140f59)

Data Value Counts():

![318630500-4434df1a-11f9-4442-ade6-b6b8c287d53c](https://github.com/Nithish23013509/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149038138/82ae02a0-6c2b-46a3-ba2c-2dd05f77c67c)

Data.head() for salary:

![318630538-f6ec96b4-455c-4421-b7df-2b763100718e](https://github.com/Nithish23013509/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149038138/6cbefc79-0694-4d6f-9bf0-05422047963b)

x.head():

![318630552-af19bc4f-4230-42f7-9607-63bb491ecb2f](https://github.com/Nithish23013509/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149038138/83641623-1804-4256-b3b0-c4a0fd047432)

Accuracy Value:

![318630596-27177cea-238a-425c-9bf4-2a71622ebe82](https://github.com/Nithish23013509/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149038138/0c349b2a-795f-4bbe-8290-dcad0dfae4bc)

Data prediction:

![318630618-8d2cd76b-d23e-4894-b001-1061c09409fc](https://github.com/Nithish23013509/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149038138/6298ed90-cef1-4ced-b379-6d7797f88cac)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
