import numpy as np
import pandas as pd
from sklearn import tree

train = pd.read_csv('train.csv')  # DataFrame object
test = pd.read_csv('test.csv')
'''
print("Shape of training set:", train.shape)
print(train.describe())
'''
'''
# Passengers that survived vs passengers that passed away
print(train["Survived"].value_counts())

# As proportions
print(train["Survived"].value_counts(normalize=True))

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True))

# Normalized female survival
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True))
'''

pd.options.mode.chained_assignment = None  # default='warn'
'''
train["Child"] = float('NaN')

train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0
print(train["Child"])

# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize=True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize=True))
'''

'''
# Create a copy of test: test_one
test_one = test

# Initialize a Survived column to 0
test_one["Survived"] = 0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one["Survived"][test_one["Sex"] == 'female'] = 1
print(test_one["Survived"])
'''

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == 'male'] = 0
train["Sex"][train["Sex"] == 'female'] = 1
# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
# Print the Sex and Embarked columns
'''print(train["Sex"])
print(train["Embarked"])'''

# Fill NaN rows using median of ages
train["Age"] = train["Age"].fillna(train["Age"].median())
# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))
