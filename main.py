import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')  # DataFrame object
# print("Shape of training set:", train.shape)
# print(train.describe())

# Passengers that survived vs passengers that passed away
print(train["Survived"].value_counts())

