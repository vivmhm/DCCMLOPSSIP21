#Checking the predicted Salary
import joblib
model=joblib.load("Salary_Data_TM.pkl")
yop=int(input("Enter years of experience:"))
predict=model.predict([[yop]])
print("Salary as per year of experience: ",predict)
