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
test["Sex"][test["Sex"] == 'male'] = 0
test["Sex"][test["Sex"] == 'female'] = 1
# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")
# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
# Print the Sex and Embarked columns
'''print(train["Sex"])
print(train["Embarked"])'''

# Fill NaN rows using median of ages
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
# Look at the importance and score of the included features
'''print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))'''

# Impute the missing value with the median
test.Fare[152] = test["Fare"].median()
# Extract the features from the test set
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
'''
# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
# Check that your data frame has 418 entries
print(my_solution.shape)
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv('my_solution_one.csv', index_label=["PassengerId"])
'''

# Create a new array with the added features: features_two
features_two = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values

# Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=1)
my_tree_two = my_tree_two.fit(features_two, train["Survived"])
# Print the score of the new decison tree
print("Score after overfitting control: {}".format(my_tree_two.score(features_two, train["Survived"])))

test_features_two = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values
my_prediction_two = my_tree_two.predict(test_features_two)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction_two, PassengerId, columns=["Survived"])
my_solution.to_csv('my_solution_two.csv', index_label=["PassengerId"])
MY_SOLUTUION_TWO_SCORE = 0.76076
print("Kaggle score using test set is: {}".format(MY_SOLUTUION_TWO_SCORE))
