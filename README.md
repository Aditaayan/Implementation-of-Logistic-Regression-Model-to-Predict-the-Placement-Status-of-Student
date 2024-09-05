# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Gather data related to student features.

Step 3: Encode categorical variables using label encoding.

Step 4: Split the dataset into training and testing sets using train_test_split from sklearn.

Step 5: Instantiate the logistic regression model. Fit the model using the training data.

Step 6: Predict placement status on the test data

Step 7: Evaluate accuracy, classification report. Print the predicted value.

Step 8: End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:ADITAAYAN
RegisterNumber:  21222304006
*/

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") #libraryfor large linear classificiation
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1) 

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

![26cb376f-e4c1-4571-b8fd-9dee21d5810b](https://github.com/user-attachments/assets/28a820e3-c0fe-484e-8427-b6b32d77d0dd)

![79531ea2-9b71-497d-bb4b-2de665018421](https://github.com/user-attachments/assets/f37ed80f-8738-4550-a8a4-1b9011a11cc9)

![03bcd294-59ba-455b-9027-b915ef6d5931](https://github.com/user-attachments/assets/49ed3842-0599-4327-9c56-9984c3be0726)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
