import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

movies = pd.read_csv("file:///D:/DATA SCIENCE/ExcelR/Assignments/Association Rules/my_movies.csv")

my_movies = movies.iloc[:,5:]
my_movies.head(10)

my_movies.isnull().sum()

my_movies.describe()

frequent_movies = apriori(my_movies, min_support = 0.05, max_len = 3, use_colnames= True)

frequent_movies.sort_values("support", ascending = False, inplace = True)

rules = association_rules(frequent_movies, metric = "lift", min_threshold = 1)
rules.head(10)

rules.sort_values('lift', ascending = False).head(10)


