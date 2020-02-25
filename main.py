import xlrd
import numpy as np
import pandas as pd

data=pd.read_excel('Diabetes.xls')

data.shape

data.head()

#x = data.iloc[:,:-1].values
#y= data.iloc[:,2].values


x = data.drop('Outcome', axis=1)
y = data['Outcome']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.10, random_state = 1)

print('the tarining set is')
print(x_train.shape)
print('====================')
print("the test set is")
print(x_test.shape)
print('====================')
y_train.value_counts()

from sklearn.linear_model import LinearRegression
LR = LinearRegression()

LR.fit(x_train, y_train)

y_pred = LR.predict(x_train)

print(x_test)
print(y_pred)

test_case = pd.read_excel('DiabetesTest2.xls')

print(test_case)

y_pred = LR.predict(test_case)

print(y_pred)
