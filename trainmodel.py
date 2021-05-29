import pandas
import joblib
ds = pandas.read_csv("Salary_Data.csv")
print("Salary_Data.csv file loaded")

#Defining dependent and independent variable --> y and x respectively.
x = ds["YearsExperience"].values.reshape(30,1)
y = ds["Salary"]

#Model training 
#Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
print("Model Established")
joblib.dump(model, 'Salary_Data_TM.pkl')
print("Trained model saved")
