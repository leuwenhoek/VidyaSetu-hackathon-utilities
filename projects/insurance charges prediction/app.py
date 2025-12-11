import os
import joblib
import pandas as pd


model = joblib.load(os.path.join("projects","insurance charges prediction","model.pkl"))

age = int(input("Enter your age : "))
bmi = float(input("Enter your BMI : "))
children = int(input("Enter no. of children you have : "))
is_male = str(input("Enter your gender (m/f): "))
is_smoker = str(input("Do you smoke (y/n): "))

valid_regions = ['ne', 'nw', 'se', 'sw']

while True:
    region = input("Enter your region (ne/nw/se/sw): ").lower()
    if region in valid_regions:
        break
    else:
        print("Invalid input! Please enter again.")

is_male = 1 if is_male.lower() == 'm' else 0

if region.lower()[0] == 'n':
    if region.lower()[1] == 'w':
        se = 0
        sw = 0
        nw = 1
    else:
        se = 0
        sw=0
        nw=0
else:
    if region.lower()[1] == 'w':
        se = 0
        sw = 1
        nw = 0
    else:
        se = 1
        sw=0
        nw=0

is_smoker = 0 if is_smoker.lower() == 'n' else 1

data = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "is_smoker": is_smoker,
    "is_male": is_male,
    "region_southeast": se,
    "region_southwest": sw,
    "region_northwest": nw
}

data = pd.DataFrame([data])


prediction = model.predict(data)
print(prediction)