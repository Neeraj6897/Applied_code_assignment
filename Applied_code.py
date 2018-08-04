#Importing the two basic libraries

import numpy as np 
import pandas as pd 

#Loading the training data
train_data = pd.read_csv('train.csv')

#Taking a look at our training data
print("This is how our training data looks like")
print(" ")
print(train_data.head())

#Handling all the missing values in the training data to make our dataset ready to use

total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#Droping the columns with missing data
train_data = train_data.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
#train_data.isnull().sum().max() #just to check that there's no missing data missing...

#Now we can see that there are a number of columns which are not necessary for our predictions here. So to reduce the #computation we are going to use only the following columns to predict

predictors = ['OverallQual','TotalBsmtSF','2ndFlrSF','GarageArea','YearBuilt','GrLivArea']

#Extracting only important columns from the training data
X = train_data[predictors]

#Using y as an output predictor
y = train_data.SalePrice


#Creating a Linear Regression model

#Importing the required libraries from scikit learn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Splitting the data into train and validation sets to train and test the model

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

#Setting up the Linear Regression Model

X = train_data[predictors]
training_model = LinearRegression()
training_model.fit(train_X,train_y)
predicted_prices = training_model.predict(val_X)

#Now comparing the predicted values with original values of Sale Prices will give us the idea about how efficient our #algorithm is be here:

final_values = pd.DataFrame({'Original Value': val_y, 'Predicted Prices': predicted_prices.round()})

print("Visualizing and comparing the final predicted values to original prices")
print(final_values.head(4)

print("We can see that our model has a good list of predicted prices") 
print(" ")
print("Calculating accuracy of our predicted prices")
print(" ")
from sklearn.metrics import r2_score
accuracy = r2_score(val_y,predicted_prices.round())*100
print("Accuracy is: ", accuracy)

"""This accuracy is achieved on the basic linear regression model. It can be further improved by using advanced regression models like XgBoost Model."""
















