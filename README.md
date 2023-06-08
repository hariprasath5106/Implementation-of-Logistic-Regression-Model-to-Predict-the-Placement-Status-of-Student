# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and mathplotlib.pyplot
2. Trace the best fit line and calculate the cost function
3. Calculate the gradient descent and plot the graph for it
4. Predict the profit for two population sizes. 

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: Hari Prasath S
RegisterNumber: 212222240034

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print(data.head())
data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])
print(data1)
x = data1.iloc[:,:-1]
print(x)
y = data1["status"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))

*/
```


## Output:

##  Placement Data:
![image](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/6994af0d-c9a4-4c5a-b10d-e7785b84a4f4)

## Salary Data:
![image](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/4dd4fede-b0e0-447d-97b5-98904148e8ed)

## Checking the null() function:
![image](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/7f59638b-48d7-4239-996f-374b7e57255b)

## Data Duplicate:
![image](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/fda5143b-b0c6-4e44-8ee4-7ad98c1191f1)

## print Data:
![image](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/228a1adc-68ae-4ca3-a7bc-0f9c62f29d71)

## Data-Status:
![image](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/4792f579-d525-4b95-b1de-c1a56c307e79)

## y_prediction array:
![image](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/f29dec1b-606b-41c4-97b9-e98e11b6a57b)

## Accuracy value:
![image](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/27881173-f288-4321-99fb-fa2a259a719b)

## Confusion array:
![ml409](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/d9b945e0-4545-4590-af40-a9e4aaa9f535)

## Classification Report:
![ml410](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/e775eca8-1347-4928-b0c5-d44132957c21)

## Prediction of LR:
![ml411](https://github.com/gokulvijayaramanuja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119577543/6d62f505-0a6c-42cb-9ab2-800e20f4858b)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
